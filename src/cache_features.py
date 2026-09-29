from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from .common import ensure_dir, load_yaml, read_jsonl, seed_everything, write_jsonl
from .models import extract_source_features, extract_target_features, load_source, load_target, target_grid


SOURCE_MEAN = (0.485, 0.456, 0.406)
SOURCE_STD = (0.229, 0.224, 0.225)


def load_rgb_raw(path: str, size: int) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = TF.resize(im, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True)
        return TF.pil_to_tensor(im).float().div_(255.0)


class ImagePathDataset(Dataset):
    def __init__(self, rows: List[dict], decode_size: int):
        self.rows = rows
        self.decode_size = decode_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return load_rgb_raw(self.rows[idx]["image"], self.decode_size), idx


def _normalize_batch(x: torch.Tensor, size: int, mean, std) -> torch.Tensor:
    if x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def _source_input(cfg: dict, raw: torch.Tensor) -> torch.Tensor:
    return _normalize_batch(raw, int(cfg["project"]["image_size"]), SOURCE_MEAN, SOURCE_STD)


def _target_input(cfg: dict, target_name: str, raw: torch.Tensor) -> torch.Tensor:
    spec = cfg["models"]["targets"][target_name]
    return _normalize_batch(
        raw,
        int(spec.get("input_size", cfg["project"]["image_size"])),
        spec["normalization_mean"],
        spec["normalization_std"],
    )


def _decode_size(cfg: dict) -> int:
    sizes = [int(cfg["project"]["image_size"])]
    sizes += [int(s.get("input_size", cfg["project"]["image_size"])) for s in cfg["models"]["targets"].values()]
    return max(sizes)


def _deterministic_subset(rows: List[dict], limit: int, seed: int) -> List[dict]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(len(rows), size=limit, replace=False))
    return [rows[int(i)] for i in ids]


def _make_perturb(raw: torch.Tensor, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    _, h, w = raw.shape
    kind = seed % 3
    rh = int(torch.randint(max(12, h // 10), max(13, h // 3), (1,), generator=g))
    rw = int(torch.randint(max(12, w // 10), max(13, w // 3), (1,), generator=g))
    y0 = int(torch.randint(0, max(1, h - rh), (1,), generator=g))
    x0 = int(torch.randint(0, max(1, w - rw), (1,), generator=g))
    y1, x1 = y0 + rh, x0 + rw
    out = raw.clone()
    mask = torch.zeros((1, h, w), dtype=torch.float32)
    mask[:, y0:y1, x0:x1] = 1.0
    if kind == 0:
        delta = float(torch.empty(1).uniform_(0.12, 0.30, generator=g))
        sign = -1.0 if (seed // 3) % 2 else 1.0
        out[:, y0:y1, x0:x1] = (out[:, y0:y1, x0:x1] + sign * delta).clamp(0, 1)
    elif kind == 1:
        patch = out[:, y0:y1, x0:x1]
        k = 7 if min(rh, rw) >= 7 else 3
        out[:, y0:y1, x0:x1] = TF.gaussian_blur(patch, kernel_size=[k, k], sigma=[2.0, 2.0])
    else:
        sy = int(torch.randint(0, max(1, h - rh), (1,), generator=g))
        sx = int(torch.randint(0, max(1, w - rw), (1,), generator=g))
        out[:, y0:y1, x0:x1] = raw[:, sy:sy + rh, sx:sx + rw]
    return out, mask


def _open_arrays(out_dir: Path, n: int, shapes: Dict[str, tuple], dtype=np.float16):
    return {
        key: np.lib.format.open_memmap(out_dir / f"{key}.npy", mode="w+", dtype=dtype, shape=(n, *shape))
        for key, shape in shapes.items()
    }


def _load_models(cfg: dict, device: torch.device):
    source = load_source(cfg, device)
    targets = {name: load_target(cfg, name, device) for name in cfg["models"]["targets"]}
    return source, targets


def cache_split(cfg: dict, dataset: str, split: str, device: torch.device) -> None:
    manifest = read_jsonl(Path(cfg["paths"]["cache_dir"]) / "manifest.jsonl")
    rows = [r for r in manifest if r["dataset"] == dataset and r["split"] == split]
    if split == "train":
        rows = [r for r in rows if r["label"] == 0]
        rows = _deterministic_subset(
            rows, int(cfg["data"]["cache_train_limit_per_dataset"]), int(cfg["project"]["seed"])
        )
    if not rows:
        print(f"[cache] skip {dataset}/{split}: no rows")
        return

    out_dir = ensure_dir(Path(cfg["paths"]["cache_dir"]) / "features" / dataset / split)
    done = out_dir / ".done"
    target_cache_ok = all(
        (out_dir / f"target_{name}.npy").exists()
        for name in cfg["models"]["targets"]
    )
    if done.exists() and target_cache_ok:
        print(f"[cache] reuse {dataset}/{split}")
        return
    if done.exists() and not target_cache_ok:
        print(f"[cache] stale cache detected for {dataset}/{split}; rebuilding")
        done.unlink(missing_ok=True)

    source, targets = _load_models(cfg, device)
    ds = ImagePathDataset(rows, _decode_size(cfg))
    dl = DataLoader(
        ds,
        batch_size=int(cfg["cache"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["cache"]["num_workers"]),
        pin_memory=True,
        prefetch_factor=int(cfg["cache"]["prefetch_factor"]),
        persistent_workers=int(cfg["cache"]["num_workers"]) > 0,
    )

    arrays = None
    amp = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        for raw, idxs in tqdm(dl, desc=f"cache {dataset}/{split}"):
            raw = raw.to(device, non_blocking=True)
            feats = extract_source_features(source, _source_input(cfg, raw))
            for name, target in targets.items():
                feats[f"target_{name}"] = extract_target_features(target, _target_input(cfg, name, raw))
            if arrays is None:
                arrays = _open_arrays(out_dir, len(rows), {k: tuple(v.shape[1:]) for k, v in feats.items()})
            ids = idxs.numpy()
            for key, value in feats.items():
                arrays[key][ids] = value.detach().float().cpu().numpy().astype(np.float16)

    if arrays is not None:
        for arr in arrays.values():
            arr.flush()
    write_jsonl(out_dir / "records.jsonl", rows)
    done.write_text("medical+general-v2\n", encoding="utf-8")
    del source, targets
    torch.cuda.empty_cache()


def cache_perturb(cfg: dict, dataset: str, device: torch.device) -> None:
    manifest = read_jsonl(Path(cfg["paths"]["cache_dir"]) / "manifest.jsonl")
    rows = [r for r in manifest if r["dataset"] == dataset and r["split"] == "train" and r["label"] == 0]
    rows = _deterministic_subset(rows, int(cfg["data"]["perturb_samples_per_dataset"]), int(cfg["project"]["seed"]) + 17)
    if not rows:
        return
    out_dir = ensure_dir(Path(cfg["paths"]["cache_dir"]) / "perturb" / dataset)
    done = out_dir / ".done"
    perturb_cache_ok = all(
        (out_dir / f"normal_target_{name}.npy").exists()
        and (out_dir / f"pert_target_{name}.npy").exists()
        and (out_dir / f"mask_{name}.npy").exists()
        for name in cfg["models"]["targets"]
    )
    if done.exists() and perturb_cache_ok:
        print(f"[cache] reuse perturb {dataset}")
        return
    if done.exists() and not perturb_cache_ok:
        print(f"[cache] stale perturb cache detected for {dataset}; rebuilding")
        done.unlink(missing_ok=True)

    source, targets = _load_models(cfg, device)
    normal_store = pert_store = None
    mask_stores = {}
    for name, target in targets.items():
        g = target_grid(target, int(cfg["models"]["targets"][name]["input_size"]))
        mask_stores[name] = np.lib.format.open_memmap(
            out_dir / f"mask_{name}.npy", mode="w+", dtype=np.uint8, shape=(len(rows), g, g)
        )

    bs = int(cfg["cache"]["batch_size"])
    decode_size = _decode_size(cfg)
    amp = device.type == "cuda"

    for start in tqdm(range(0, len(rows), bs), desc=f"perturb {dataset}"):
        sub = rows[start:start + bs]
        normals, perts, masks = [], [], []
        for j, r in enumerate(sub):
            raw = load_rgb_raw(r["image"], decode_size)
            p, m = _make_perturb(raw, seed=int(cfg["project"]["seed"]) + start + j)
            normals.append(raw); perts.append(p); masks.append(m)
        n_raw = torch.stack(normals).to(device, non_blocking=True)
        p_raw = torch.stack(perts).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            nf = extract_source_features(source, _source_input(cfg, n_raw))
            pf = extract_source_features(source, _source_input(cfg, p_raw))
            for name, target in targets.items():
                nf[f"target_{name}"] = extract_target_features(target, _target_input(cfg, name, n_raw))
                pf[f"target_{name}"] = extract_target_features(target, _target_input(cfg, name, p_raw))
        if normal_store is None:
            normal_store = _open_arrays(out_dir, len(rows), {f"normal_{k}": tuple(v.shape[1:]) for k, v in nf.items()})
            pert_store = _open_arrays(out_dir, len(rows), {f"pert_{k}": tuple(v.shape[1:]) for k, v in pf.items()})
        sl = slice(start, start + len(sub))
        for k, v in nf.items():
            normal_store[f"normal_{k}"][sl] = v.detach().float().cpu().numpy().astype(np.float16)
        for k, v in pf.items():
            pert_store[f"pert_{k}"][sl] = v.detach().float().cpu().numpy().astype(np.float16)
        stacked_masks = torch.stack(masks).to(device)
        for name, target in targets.items():
            g = target_grid(target, int(cfg["models"]["targets"][name]["input_size"]))
            mg = F.interpolate(stacked_masks, size=(g, g), mode="nearest").squeeze(1).cpu().numpy().astype(np.uint8)
            mask_stores[name][sl] = mg

    if normal_store is not None:
        for arr in normal_store.values():
            arr.flush()
    if pert_store is not None:
        for arr in pert_store.values():
            arr.flush()
    for arr in mask_stores.values():
        arr.flush()
    write_jsonl(out_dir / "records.jsonl", rows)
    done.write_text("medical+general-v2\n", encoding="utf-8")
    del source, targets
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg["project"]["seed"]), bool(cfg["runtime"].get("deterministic", False)))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    for split in ("train", "test"):
        cache_split(cfg, args.dataset, split, device)
    cache_perturb(cfg, args.dataset, device)


if __name__ == "__main__":
    main()
