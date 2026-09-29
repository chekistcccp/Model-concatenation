from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope import snapshot_download
from safetensors.torch import load_file as load_safetensors
from torchvision.models import resnet50
from transformers import AutoModel

from .common import ensure_dir, write_json


_MIN_WEIGHT_BYTES = 1 * 1024 * 1024


_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")


def _is_archive(path: Path) -> bool:
    return path.name.lower().endswith(_ARCHIVE_SUFFIXES)


def _archive_stem(path: Path) -> str:
    name = path.name
    for suffix in sorted(_ARCHIVE_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return path.stem


def _extract_archive(arc: Path, target: Path) -> None:
    ensure_dir(target)
    try:
        shutil.unpack_archive(str(arc), str(target))
    except (shutil.ReadError, ValueError):
        if arc.suffix.lower() == ".zip":
            with zipfile.ZipFile(arc) as zf:
                zf.extractall(target)
        else:
            with tarfile.open(arc) as tf:
                tf.extractall(target)


def prepare_manual_model_archives(cfg: dict) -> dict:
    """Extract raw Google-Drive model archives placed directly in model/.

    The user does not need to rename or unpack the RadImageNet bundle. We
    recursively locate the official ResNet50 PyTorch checkpoint and materialize
    it at the canonical path expected by the rest of the pipeline.
    """
    model_root = ensure_dir(Path(cfg["paths"]["model_dir"]))
    extract_root = ensure_dir(model_root / "_manual_extracted")

    archives = sorted(
        p for p in model_root.iterdir()
        if p.is_file() and _is_archive(p)
    )
    queue = list(archives)
    seen = set()
    extracted = []
    while queue:
        arc = queue.pop(0)
        key = str(arc.resolve())
        if key in seen:
            continue
        seen.add(key)

        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        target = ensure_dir(extract_root / f"{digest}_{_archive_stem(arc)}")
        marker = target / ".extract_complete"
        signature = f"{arc.stat().st_size}:{int(arc.stat().st_mtime)}"
        if not marker.exists() or marker.read_text(encoding="utf-8").strip() != signature:
            print(f"[model] extracting manual archive {arc} -> {target}")
            for child in list(target.iterdir()):
                if child.name == ".extract_complete":
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            _extract_archive(arc, target)
            marker.write_text(signature + "\n", encoding="utf-8")
        extracted.append(str(target))

        nested = sorted(
            p for p in target.rglob("*")
            if p.is_file() and _is_archive(p)
        )
        queue.extend(nested)

    medical = cfg["models"]["sources"]["medical"]
    canonical = Path(medical["checkpoint"])
    if not _is_weight_file(canonical):
        search_roots = [extract_root, model_root]
        exact = []
        fuzzy = []
        for root in search_roots:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if not _is_weight_file(p):
                    continue
                low = p.name.lower()
                if low == "resnet50_torch.pt":
                    exact.append(p)
                elif "resnet50" in low and p.suffix.lower() in {".pt", ".pth"}:
                    fuzzy.append(p)

        candidates = exact or fuzzy
        # Never select the canonical destination itself as a source.
        candidates = [p for p in candidates if p.resolve() != canonical.resolve()]
        if candidates:
            src = sorted(candidates, key=lambda p: (0 if p.name.lower() == "resnet50_torch.pt" else 1, len(p.parts), str(p)))[0]
            ensure_dir(canonical.parent)
            print(f"[model] detected RadImageNet ResNet50: {src}")
            print(f"[model] materializing canonical checkpoint -> {canonical}")
            canonical.unlink(missing_ok=True)
            try:
                os.link(src, canonical)
            except OSError:
                shutil.copy2(src, canonical)

    return {
        "archives": [str(p) for p in archives],
        "extracted": extracted,
        "radimagenet_checkpoint": str(canonical),
        "radimagenet_ready": _is_weight_file(canonical),
    }



def _is_weight_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= _MIN_WEIGHT_BYTES


def _find_weight_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out = []
    for pattern in ("*.safetensors", "*.bin", "*.pth", "*.pt"):
        out.extend(p for p in root.rglob(pattern) if _is_weight_file(p))
    return sorted(set(out))


def _auto_download_enabled(cfg: dict) -> bool:
    return bool(cfg["models"].get("auto_download_missing", True))


def model_status(cfg: dict, prepare_manual: bool = True) -> dict:
    manual_archive_status = prepare_manual_model_archives(cfg) if prepare_manual else {}
    out = {"manual_archives": manual_archive_status, "sources": {}, "targets": {}}

    for name, spec in cfg["models"]["sources"].items():
        if spec.get("manual", False):
            ckpt = Path(spec["checkpoint"])
            out["sources"][name] = {
                "manual": True,
                "download": "Google Drive / user supplied",
                "checkpoint": str(ckpt),
                "ready": _is_weight_file(ckpt),
                "size_bytes": ckpt.stat().st_size if ckpt.is_file() else 0,
                "note": spec.get("note", ""),
            }
        else:
            root = Path(spec["local"])
            weights = _find_weight_files(root)
            out["sources"][name] = {
                "manual": False,
                "repo_id": spec["repo_id"],
                "local_dir": str(root),
                "config_exists": (root / "config.json").is_file(),
                "ready": bool(weights),
                "weight_files": [str(p) for p in weights],
                "note": spec.get("note", ""),
            }

    for name, spec in cfg["models"]["targets"].items():
        root = Path(spec["local"])
        weights = _find_weight_files(root)
        out["targets"][name] = {
            "manual": bool(spec.get("manual", False)),
            "repo_id": spec.get("repo_id"),
            "local_dir": str(root),
            "ready": (root / "config.json").is_file() and bool(weights),
            "weight_files": [str(p) for p in weights],
            "note": spec.get("note", ""),
        }
    return out


def _download_snapshot(repo_id: str, dst: Path) -> None:
    ensure_dir(dst)
    print(f"[model] ModelScope download {repo_id} -> {dst}")
    try:
        snapshot_download(repo_id=repo_id, local_dir=str(dst), max_workers=8)
    except TypeError:
        snapshot_download(repo_id, local_dir=str(dst))


def ensure_models(cfg: dict, allow_download: bool | None = None) -> dict:
    if allow_download is None:
        allow_download = _auto_download_enabled(cfg)

    prepare_manual_model_archives(cfg)
    status = model_status(cfg, prepare_manual=False)

    # Manual Google Drive item: never auto-download.
    manual_missing = [
        f"source:{name}"
        for name, item in status["sources"].items()
        if item["manual"] and not item["ready"]
    ]
    if manual_missing:
        write_json(Path(cfg["paths"]["cache_dir"]) / "model_audit.json", status)
        raise FileNotFoundError(
            "Manual Google Drive model is missing: "
            + ", ".join(manual_missing)
            + ". Put the original RadImageNet PyTorch archive downloaded from "
              "Google Drive directly under model/. Do not extract or rename it. "
              "The pipeline will unpack and locate ResNet50 automatically."
        )

    # Auto items: ModelScope.
    for name, spec in cfg["models"]["sources"].items():
        if spec.get("manual", False):
            continue
        if not status["sources"][name]["ready"]:
            if not allow_download:
                raise FileNotFoundError(
                    f"Auto model source:{name} is missing at {spec['local']}"
                )
            _download_snapshot(spec["repo_id"], Path(spec["local"]))

    for name, spec in cfg["models"]["targets"].items():
        if not status["targets"][name]["ready"]:
            if not allow_download:
                raise FileNotFoundError(
                    f"Auto model target:{name} is missing at {spec['local']}"
                )
            _download_snapshot(spec["repo_id"], Path(spec["local"]))

    status = model_status(cfg, prepare_manual=False)
    write_json(Path(cfg["paths"]["cache_dir"]) / "model_audit.json", status)

    missing = []
    for group in ("sources", "targets"):
        for name, item in status[group].items():
            if not item["ready"]:
                missing.append(f"{group[:-1]}:{name}")
    if missing:
        raise RuntimeError(
            "Model preparation is incomplete after ModelScope download/check: "
            + ", ".join(missing)
        )
    return status


def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _strip_module_prefix(state: dict) -> dict:
    if state and all(k.startswith("module.") for k in state):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


class TorchvisionResNetSource(nn.Module):
    """Feature wrapper for official RadImageNet PyTorch ResNet50 weights."""

    def __init__(self, checkpoint: str):
        super().__init__()
        base = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:9])

        state = _unwrap_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)
        )
        if not isinstance(state, dict):
            raise TypeError("Unsupported RadImageNet checkpoint object")
        state = _strip_module_prefix(state)

        if any(k.startswith("backbone.") for k in state):
            self.load_state_dict(state, strict=True)
        elif any(k.startswith("0.") for k in state):
            self.backbone.load_state_dict(state, strict=True)
        else:
            raise RuntimeError(
                "RadImageNet checkpoint does not match the official PyTorch "
                "Backbone format. Expected keys beginning with 'backbone.' "
                "or Sequential indices such as '0.'."
            )

    def forward_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        for idx in range(5):
            x = self.backbone[idx](x)
        x = self.backbone[5](x)
        s1 = x
        x = self.backbone[6](x)
        s2 = x
        x = self.backbone[7](x)
        s3 = x
        return {"s1": s1, "s2": s2, "s3": s3}


class TimmResNetSource(nn.Module):
    """Feature wrapper for ModelScope-downloaded timm ResNet50."""

    def __init__(self, architecture: str, model_dir: str):
        super().__init__()
        root = Path(model_dir)
        weights = _find_weight_files(root)
        safe = [p for p in weights if p.suffix == ".safetensors"]
        if not safe:
            raise FileNotFoundError(
                f"No safetensors weight found for timm ResNet at {root}"
            )
        self.base = timm.create_model(architecture, pretrained=False, num_classes=1000)
        state = load_safetensors(str(safe[0]), device="cpu")
        missing, unexpected = self.base.load_state_dict(state, strict=False)
        bad_missing = [k for k in missing if not k.startswith("fc.")]
        bad_unexpected = [k for k in unexpected if not k.startswith("fc.")]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                f"ModelScope timm ResNet checkpoint mismatch: "
                f"missing={bad_missing[:8]}, unexpected={bad_unexpected[:8]}"
            )

    def forward_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b = self.base
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.act1(x) if hasattr(b, "act1") else b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        s1 = x
        x = b.layer3(x)
        s2 = x
        x = b.layer4(x)
        s3 = x
        return {"s1": s1, "s2": s2, "s3": s3}


def load_source(cfg: dict, source_name: str, device: torch.device, dtype=torch.float16):
    spec = cfg["models"]["sources"][source_name]
    if spec["kind"] == "radimagenet_resnet50":
        model = TorchvisionResNetSource(spec["checkpoint"])
    elif spec["kind"] == "timm_resnet50":
        model = TimmResNetSource(spec["architecture"], spec["local"])
    else:
        raise ValueError(f"Unknown source kind: {spec['kind']}")
    model = model.to(device).eval().to(dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_target(cfg: dict, target_name: str, device: torch.device, dtype=torch.float16):
    spec = cfg["models"]["targets"][target_name]
    root = Path(spec["local"])
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"Missing target config: {root / 'config.json'}")
    model = AutoModel.from_pretrained(
        str(root),
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.inference_mode()
def extract_source_features(source, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
    return source.forward_stages(pixel_values.to(next(source.parameters()).dtype))


def target_num_prefix(target) -> int:
    return 1 + int(getattr(target.config, "num_register_tokens", 0))


def target_grid(target, input_size: int) -> int:
    patch = getattr(target.config, "patch_size", 14)
    if isinstance(patch, (tuple, list)):
        patch = patch[0]
    patch = int(patch)
    if input_size % patch != 0:
        raise ValueError(
            f"input_size={input_size} must be divisible by patch_size={patch}"
        )
    return input_size // patch


@torch.inference_mode()
def extract_target_features(target, pixel_values: torch.Tensor) -> torch.Tensor:
    out = target(
        pixel_values.to(next(target.parameters()).dtype),
        return_dict=True,
    )
    return out.last_hidden_state[:, target_num_prefix(target):, :]


def _target_layers(target):
    encoder = getattr(target, "encoder", None)
    if encoder is None:
        raise AttributeError("Unable to locate DINOv2 encoder")
    layers = getattr(encoder, "layer", None)
    if layers is None:
        layers = getattr(encoder, "layers", None)
    if layers is None:
        raise AttributeError("Unable to locate DINOv2 transformer layers")
    return layers


def _target_norm(target):
    for name in ("layernorm", "norm"):
        x = getattr(target, name, None)
        if x is not None:
            return x
    raise AttributeError("Unable to locate DINOv2 final norm")


def _prefix_and_patch_position(
    target, input_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid = target_grid(target, input_size)
    hidden = int(target.config.hidden_size)
    device = next(target.parameters()).device
    dtype = next(target.parameters()).dtype

    dummy = torch.zeros(
        1, 1 + grid * grid, hidden, device=device, dtype=dtype
    )
    pos = target.embeddings.interpolate_pos_encoding(
        dummy, input_size, input_size
    )

    prefix = target.embeddings.cls_token.detach().to(pos.dtype) + pos[:, :1]
    patch_pos = pos[:, 1:]
    return prefix.float(), patch_pos.float()


class StitchAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        grid: int,
        prefix_init: torch.Tensor,
        patch_position: torch.Tensor,
        adapter_type: str = "mlp",
    ):
        super().__init__()
        self.grid = int(grid)
        self.channel_proj = (
            nn.Conv2d(in_channels, hidden_dim, 1)
            if in_channels != hidden_dim
            else nn.Identity()
        )
        self.norm = nn.LayerNorm(hidden_dim)
        if adapter_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        elif adapter_type == "linear":
            self.mlp = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError(adapter_type)

        self.prefix_tokens = nn.Parameter(prefix_init.detach().clone().float())
        self.register_buffer(
            "patch_position",
            patch_position.detach().clone().float(),
            persistent=False,
        )

    def forward(self, fmap: torch.Tensor):
        x = fmap.float()
        h, w = x.shape[-2:]
        if (h, w) != (self.grid, self.grid):
            if h > self.grid and w > self.grid:
                x = F.adaptive_avg_pool2d(x, (self.grid, self.grid))
            else:
                x = F.interpolate(
                    x,
                    size=(self.grid, self.grid),
                    mode="bilinear",
                    align_corners=False,
                )
        x = self.channel_proj(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        patches = x + self.mlp(self.norm(x))
        patches = patches + self.patch_position
        prefix = self.prefix_tokens.expand(patches.shape[0], -1, -1)
        return prefix, patches


class TargetTail(nn.Module):
    def __init__(self, target, cut_block: int, compile_tail: bool = False):
        super().__init__()
        self.target = target
        self.cut_block = int(cut_block)
        self.layers = _target_layers(target)
        self.norm = _target_norm(target)
        self.num_prefix = target_num_prefix(target)
        if not 0 <= self.cut_block < len(self.layers):
            raise ValueError(f"cut_block {cut_block} out of range")
        if compile_tail and hasattr(torch, "compile"):
            self.forward = torch.compile(
                self.forward, mode="reduce-overhead"
            )

    def forward(self, prefix: torch.Tensor, patches: torch.Tensor):
        h = torch.cat([prefix, patches], dim=1)
        h = h.to(next(self.target.parameters()).dtype)
        for layer in self.layers[self.cut_block:]:
            h = layer(h)
            if isinstance(h, (tuple, list)):
                h = h[0]
        h = self.norm(h)
        return h[:, self.num_prefix:, :]


def build_stitch_modules(
    cfg: dict,
    target,
    source_name: str,
    target_name: str,
    source_stage: int,
    target_block: int,
    adapter_type: str,
):
    source_spec = cfg["models"]["sources"][source_name]
    target_spec = cfg["models"]["targets"][target_name]
    in_channels = int(
        source_spec["stage_channels"][int(source_stage) - 1]
    )
    hidden_dim = int(target.config.hidden_size)
    input_size = int(target_spec["input_size"])
    prefix, patch_pos = _prefix_and_patch_position(target, input_size)

    adapter = StitchAdapter(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        grid=target_grid(target, input_size),
        prefix_init=prefix,
        patch_position=patch_pos,
        adapter_type=adapter_type,
    )
    tail = TargetTail(
        target,
        target_block,
        bool(cfg["project"].get("compile_tail", False)),
    )
    return adapter, tail
