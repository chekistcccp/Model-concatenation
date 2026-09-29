from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .common import read_jsonl, stage_key
from .models import build_stitch_modules, target_grid


def _load_memmap(path: Path):
    return np.load(path, mmap_mode="r")


def modality_datasets(cfg: dict) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for ds in cfg["data"]["datasets"]:
        out.setdefault(cfg["data"]["modality_map"][ds], []).append(ds)
    return out


def target_modalities(cfg: dict) -> List[str]:
    return sorted(modality_datasets(cfg).keys())


def domain_group(modality: str) -> str:
    return "radiology" if modality in {"brain_mri", "liver_ct", "chest_xray"} else "non_radiology"


def _balanced_indices(n: int, take: int, seed: int) -> np.ndarray:
    if take >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=take, replace=False))


def load_source_arrays(
    cfg: dict,
    target_modality: str,
    target_name: str,
    source_stage: int,
    n_per_modality: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    cache_root = Path(cfg["paths"]["cache_dir"]) / "features"
    groups = modality_datasets(cfg)
    xs, ys = [], []
    target_key = f"target_{target_name}.npy"
    for mod, datasets in groups.items():
        if mod == target_modality:
            continue
        per_ds = max(1, math.ceil(n_per_modality / len(datasets)))
        mod_x, mod_y = [], []
        for di, ds in enumerate(datasets):
            root = cache_root / ds / "train"
            xmap = _load_memmap(root / f"{stage_key(source_stage)}.npy")
            ymap = _load_memmap(root / target_key)
            idx = _balanced_indices(len(xmap), min(per_ds, len(xmap)), seed + 101 * di + len(mod))
            mod_x.append(np.asarray(xmap[idx], dtype=np.float16))
            mod_y.append(np.asarray(ymap[idx], dtype=np.float16))
        mx = np.concatenate(mod_x, axis=0)[:n_per_modality]
        my = np.concatenate(mod_y, axis=0)[:n_per_modality]
        xs.append(mx); ys.append(my)
    if not xs:
        raise RuntimeError(f"No source modalities remain after excluding {target_modality}")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def train_adapter(
    cfg: dict,
    target,
    target_name: str,
    target_modality: str,
    source_stage: int,
    target_block: int,
    adapter_type: str,
    n_per_modality: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
):
    x_np, y_np = load_source_arrays(
        cfg, target_modality, target_name, source_stage, n_per_modality, seed
    )
    x = torch.from_numpy(np.array(x_np, copy=True))
    y = torch.from_numpy(np.array(y_np, copy=True))
    ds = TensorDataset(x, y)
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen,
                    num_workers=0, pin_memory=True, drop_last=False)

    adapter, tail = build_stitch_modules(
        cfg, target, target_name, source_stage, target_block, adapter_type
    )
    adapter = adapter.to(device)
    tail = tail.to(device).eval()
    for p in tail.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(cfg["runtime"].get("amp", True)) and device.type == "cuda"
    )
    amp = bool(cfg["runtime"].get("amp", True)) and device.type == "cuda"
    history = []

    for _ in range(epochs):
        adapter.train()
        total = 0.0
        count = 0
        for fmap, target_feat in dl:
            fmap = fmap.to(device, non_blocking=True)
            target_feat = target_feat.to(device, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prefix, patches = adapter(fmap)
                pred = tail(prefix, patches).float()
                patch_loss = 1.0 - F.cosine_similarity(pred, target_feat, dim=-1).mean()
                global_loss = 1.0 - F.cosine_similarity(
                    pred.mean(dim=1), target_feat.mean(dim=1), dim=-1
                ).mean()
                loss = patch_loss + 0.1 * global_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach()) * len(fmap)
            count += len(fmap)
        history.append(total / max(1, count))
    return adapter.eval(), tail, history


@torch.inference_mode()
def discrepancy(adapter, tail, fmap: torch.Tensor, target_feat: torch.Tensor, device: torch.device):
    prefix, patches = adapter(fmap.to(device, non_blocking=True))
    pred = tail(prefix, patches).float()
    target_feat = target_feat.to(device, non_blocking=True).float()
    return 1.0 - F.cosine_similarity(pred, target_feat, dim=-1)


def _score_from_map(d: torch.Tensor, frac: float, mode: str) -> torch.Tensor:
    k = max(1, int(round(d.shape[1] * frac)))
    if mode == "raw_topk":
        z = d
    else:
        med = d.median(dim=1, keepdim=True).values
        if mode == "contrast_topk":
            z = (d - med).clamp_min(0)
        elif mode == "robust_topk":
            mad = (d - med).abs().median(dim=1, keepdim=True).values.clamp_min(1e-6)
            z = ((d - med) / mad).clamp_min(0)
        else:
            raise ValueError(mode)
    return z.topk(k, dim=1).values.mean(dim=1)


def _map_for_localization(d: torch.Tensor, grid: int) -> torch.Tensor:
    med = d.median(dim=1, keepdim=True).values
    return (d - med).clamp_min(0).reshape(-1, grid, grid)


def _safe_auc(labels: Sequence[int], scores: Sequence[float]) -> Tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def _load_mask(path: str | None, grid: int) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with Image.open(p) as im:
        im = im.convert("L")
        im = TF.resize(im, [grid, grid], interpolation=InterpolationMode.NEAREST)
        arr = np.asarray(im, dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


@torch.inference_mode()
def evaluate_dataset(
    cfg: dict,
    target,
    target_name: str,
    adapter,
    tail,
    dataset: str,
    source_stage: int,
    device: torch.device,
    topk_fraction: float,
    score_modes: Sequence[str] = ("contrast_topk",),
    batch_size: int = 256,
) -> List[dict]:
    root = Path(cfg["paths"]["cache_dir"]) / "features" / dataset / "test"
    records = read_jsonl(root / "records.jsonl")
    fmap = _load_memmap(root / f"{stage_key(source_stage)}.npy")
    target_feat = _load_memmap(root / f"target_{target_name}.npy")
    labels = [int(r["label"]) for r in records]
    scores: Dict[str, List[float]] = {m: [] for m in score_modes}
    pixel_scores, pixel_labels = [], []
    has_masks = any(r.get("mask") for r in records)
    grid = target_grid(target, int(cfg["models"]["targets"][target_name]["input_size"]))

    for start in range(0, len(records), batch_size):
        stop = min(len(records), start + batch_size)
        sl = slice(start, stop)
        xf = torch.from_numpy(np.asarray(fmap[sl], dtype=np.float16))
        yt = torch.from_numpy(np.asarray(target_feat[sl], dtype=np.float16))
        d = discrepancy(adapter, tail, xf, yt, device)
        for mode in score_modes:
            scores[mode].extend(_score_from_map(d, topk_fraction, mode).cpu().tolist())
        loc = _map_for_localization(d, grid).cpu().numpy()
        for local_i, rec in enumerate(records[start:stop]):
            mask = _load_mask(rec.get("mask"), grid)
            if mask is None and has_masks and int(rec["label"]) == 0:
                mask = np.zeros((grid, grid), dtype=np.uint8)
            if mask is not None:
                pixel_scores.append(loc[local_i].reshape(-1))
                pixel_labels.append(mask.reshape(-1))

    out = []
    for mode in score_modes:
        auroc, aupr = _safe_auc(labels, scores[mode])
        row = {
            "dataset": dataset,
            "target_backbone": target_name,
            "score_mode": mode,
            "image_auroc": auroc,
            "image_aupr": aupr,
            "n_test": len(records),
        }
        if pixel_labels:
            pl, ps = np.concatenate(pixel_labels), np.concatenate(pixel_scores)
            pauroc, paupr = _safe_auc(pl, ps)
            row.update({"pixel_auroc": pauroc, "pixel_aupr": paupr, "n_mask_pixels": int(len(pl))})
        out.append(row)
    return out


@torch.inference_mode()
def compute_aoss(
    cfg: dict,
    target,
    target_name: str,
    adapter,
    tail,
    target_modality: str,
    source_stage: int,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    groups = modality_datasets(cfg)
    dn_all, sin_all, sout_all = [], [], []
    for mod, datasets in groups.items():
        if mod == target_modality:
            continue
        for ds in datasets:
            root = Path(cfg["paths"]["cache_dir"]) / "perturb" / ds
            if not (root / ".done").exists():
                continue
            n_x = _load_memmap(root / f"normal_{stage_key(source_stage)}.npy")
            n_y = _load_memmap(root / f"normal_target_{target_name}.npy")
            p_x = _load_memmap(root / f"pert_{stage_key(source_stage)}.npy")
            p_y = _load_memmap(root / f"pert_target_{target_name}.npy")
            masks = _load_memmap(root / f"mask_{target_name}.npy")
            for start in range(0, len(n_x), batch_size):
                sl = slice(start, min(len(n_x), start + batch_size))
                dn = discrepancy(
                    adapter, tail,
                    torch.from_numpy(np.asarray(n_x[sl], dtype=np.float16)),
                    torch.from_numpy(np.asarray(n_y[sl], dtype=np.float16)),
                    device,
                )
                dp = discrepancy(
                    adapter, tail,
                    torch.from_numpy(np.asarray(p_x[sl], dtype=np.float16)),
                    torch.from_numpy(np.asarray(p_y[sl], dtype=np.float16)),
                    device,
                )
                m = torch.from_numpy(np.asarray(masks[sl], dtype=np.uint8)).to(device).reshape(dp.shape).bool()
                dn_all.append(dn.mean().item())
                if m.any():
                    sin_all.append(dp[m].mean().item())
                if (~m).any():
                    sout_all.append(dp[~m].mean().item())
    dn = float(np.mean(dn_all)) if dn_all else float("nan")
    s_in = float(np.mean(sin_all)) if sin_all else float("nan")
    s_out = float(np.mean(sout_all)) if sout_all else float("nan")
    local = s_in - s_out
    score = local / (dn + 1e-8) if np.isfinite(dn) else float("nan")
    return {
        "normal_discrepancy": dn,
        "perturb_in": s_in,
        "perturb_out": s_out,
        "local_sensitivity": local,
        "aoss": score,
    }


def evaluate_cached_baselines(
    cfg: dict, dataset: str, target_name: str, topk_fraction: float = 0.05, batch_size: int = 512
) -> List[dict]:
    root = Path(cfg["paths"]["cache_dir"]) / "features" / dataset / "test"
    records = read_jsonl(root / "records.jsonl")
    labels = np.asarray([int(r["label"]) for r in records], dtype=np.int64)
    target = _load_memmap(root / f"target_{target_name}.npy")
    scores = []

    for start in range(0, len(records), batch_size):
        sl = slice(start, min(len(records), start + batch_size))
        z = np.asarray(target[sl], dtype=np.float32)
        zn = z / np.clip(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8, None)
        center = zn.mean(axis=1, keepdims=True)
        center /= np.clip(np.linalg.norm(center, axis=-1, keepdims=True), 1e-8, None)
        d = 1.0 - (zn * center).sum(-1)
        k = max(1, int(round(d.shape[1] * topk_fraction)))
        scores.extend(np.partition(d, -k, axis=1)[:, -k:].mean(axis=1).tolist())

    auroc, aupr = _safe_auc(labels, scores)
    return [{
        "dataset": dataset,
        "target_backbone": target_name,
        "baseline": "target_patch_dispersion",
        "image_auroc": auroc,
        "image_aupr": aupr,
        "n_test": len(labels),
    }]
