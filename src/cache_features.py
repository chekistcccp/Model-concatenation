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
from .models import extract_backbone_features, load_backbones


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_rgb_tensor(path: str, size: int) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = TF.resize(im, [size, size], interpolation=InterpolationMode.BICUBIC, antialias=True)
        x = TF.pil_to_tensor(im).float().div_(255.0)
    x = TF.normalize(x, MEAN, STD)
    return x


class ImagePathDataset(Dataset):
    def __init__(self, rows: List[dict], size: int):
        self.rows = rows
        self.size = size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return load_rgb_tensor(self.rows[idx]["image"], self.size), idx


def _deterministic_subset(rows: List[dict], limit: int, seed: int) -> List[dict]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(len(rows), size=limit, replace=False))
    return [rows[int(i)] for i in ids]


def _make_perturb(x: torch.Tensor, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # x is normalized CHW. Perturb in normalized space for speed and determinism.
    g = torch.Generator(device="cpu").manual_seed(seed)
    c, h, w = x.shape
    kind = seed % 3
    rh = int(torch.randint(max(12, h // 10), max(13, h // 3), (1,), generator=g))
    rw = int(torch.randint(max(12, w // 10), max(13, w // 3), (1,), generator=g))
    y0 = int(torch.randint(0, max(1, h - rh), (1,), generator=g))
    x0 = int(torch.randint(0, max(1, w - rw), (1,), generator=g))
    y1, x1 = y0 + rh, x0 + rw
    out = x.clone()
    mask = torch.zeros((1, h, w), dtype=torch.float32)
    mask[:, y0:y1, x0:x1] = 1.0
    if kind == 0:
        delta = float(torch.empty(1).uniform_(0.8, 1.8, generator=g))
        out[:, y0:y1, x0:x1] = out[:, y0:y1, x0:x1] + delta
    elif kind == 1:
        patch = out[:, y0:y1, x0:x1].unsqueeze(0)
        k = 7 if min(rh, rw) >= 7 else 3
        patch = TF.gaussian_blur(patch.squeeze(0), kernel_size=[k, k], sigma=[2.0, 2.0])
        out[:, y0:y1, x0:x1] = patch
    else:
        sy = int(torch.randint(0, max(1, h - rh), (1,), generator=g))
        sx = int(torch.randint(0, max(1, w - rw), (1,), generator=g))
        out[:, y0:y1, x0:x1] = x[:, sy:sy + rh, sx:sx + rw]
    mask14 = F.interpolate(mask.unsqueeze(0), size=(14, 14), mode="nearest").squeeze(0)
    return out, mask14


def _open_arrays(out_dir: Path, n: int, shapes: Dict[str, tuple], dtype=np.float16):
    arrays = {}
    for key, shape in shapes.items():
        arrays[key] = np.lib.format.open_memmap(
            out_dir / f"{key}.npy", mode="w+", dtype=dtype, shape=(n, *shape)
        )
    return arrays


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
    if done.exists():
        print(f"[cache] reuse {dataset}/{split}")
        return

    conv, vit = load_backbones(cfg, device)
    ds = ImagePathDataset(rows, int(cfg["project"]["image_size"]))
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
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for images, idxs in tqdm(dl, desc=f"cache {dataset}/{split}"):
            images = images.to(device, non_blocking=True)
            feats = extract_backbone_features(conv, vit, images)
            if arrays is None:
                shapes = {k: tuple(v.shape[1:]) for k, v in feats.items()}
                arrays = _open_arrays(out_dir, len(rows), shapes)
            ids = idxs.numpy()
            for key, value in feats.items():
                arrays[key][ids] = value.detach().float().cpu().numpy().astype(np.float16)

    if arrays is not None:
        for arr in arrays.values():
            arr.flush()
    write_jsonl(out_dir / "records.jsonl", rows)
    done.write_text("ok\n", encoding="utf-8")
    del conv, vit
    torch.cuda.empty_cache()


def cache_perturb(cfg: dict, dataset: str, device: torch.device) -> None:
    manifest = read_jsonl(Path(cfg["paths"]["cache_dir"]) / "manifest.jsonl")
    rows = [r for r in manifest if r["dataset"] == dataset and r["split"] == "train" and r["label"] == 0]
    rows = _deterministic_subset(rows, int(cfg["data"]["perturb_samples_per_dataset"]), int(cfg["project"]["seed"]) + 17)
    if not rows:
        return
    out_dir = ensure_dir(Path(cfg["paths"]["cache_dir"]) / "perturb" / dataset)
    done = out_dir / ".done"
    if done.exists():
        print(f"[cache] reuse perturb {dataset}")
        return

    conv, vit = load_backbones(cfg, device)
    normal_store = pert_store = None
    mask_store = np.lib.format.open_memmap(out_dir / "mask14.npy", mode="w+", dtype=np.uint8, shape=(len(rows), 14, 14))
    bs = int(cfg["cache"]["batch_size"])
    size = int(cfg["project"]["image_size"])

    for start in tqdm(range(0, len(rows), bs), desc=f"perturb {dataset}"):
        sub = rows[start:start + bs]
        normals, perts, masks = [], [], []
        for j, r in enumerate(sub):
            x = load_rgb_tensor(r["image"], size)
            p, m = _make_perturb(x, seed=int(cfg["project"]["seed"]) + start + j)
            normals.append(x); perts.append(p); masks.append(m)
        n_batch = torch.stack(normals).to(device, non_blocking=True)
        p_batch = torch.stack(perts).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            nf = extract_backbone_features(conv, vit, n_batch)
            pf = extract_backbone_features(conv, vit, p_batch)
        if normal_store is None:
            normal_store = _open_arrays(out_dir, len(rows), {f"normal_{k}": tuple(v.shape[1:]) for k, v in nf.items()})
            pert_store = _open_arrays(out_dir, len(rows), {f"pert_{k}": tuple(v.shape[1:]) for k, v in pf.items()})
        sl = slice(start, start + len(sub))
        for k, v in nf.items():
            normal_store[f"normal_{k}"][sl] = v.detach().float().cpu().numpy().astype(np.float16)
        for k, v in pf.items():
            pert_store[f"pert_{k}"][sl] = v.detach().float().cpu().numpy().astype(np.float16)
        mask_store[sl] = torch.stack(masks).squeeze(1).numpy().astype(np.uint8)

    if normal_store is not None:
        for arr in normal_store.values():
            arr.flush()
    if pert_store is not None:
        for arr in pert_store.values():
            arr.flush()
    mask_store.flush()
    write_jsonl(out_dir / "records.jsonl", rows)
    done.write_text("ok\n", encoding="utf-8")
    del conv, vit
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
