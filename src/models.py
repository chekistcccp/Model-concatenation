from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from torchvision.models import resnet50

from .common import write_json


_MIN_WEIGHT_BYTES = 1 * 1024 * 1024


def _is_real_weight_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= _MIN_WEIGHT_BYTES


def model_status(cfg: dict) -> dict:
    out: dict = {"sources": {}, "targets": {}}

    for name, spec in cfg["models"]["sources"].items():
        ckpt = Path(spec["checkpoint"])
        out["sources"][name] = {
            "kind": spec["kind"],
            "checkpoint": str(ckpt),
            "exists": ckpt.is_file(),
            "size_bytes": ckpt.stat().st_size if ckpt.is_file() else 0,
            "ready": _is_real_weight_file(ckpt),
            "note": spec.get("note", ""),
        }

    for name, spec in cfg["models"]["targets"].items():
        config_path = Path(spec["config"])
        ckpt = Path(spec["checkpoint"])
        ready = (
            config_path.is_file()
            and _is_real_weight_file(ckpt)
        )
        out["targets"][name] = {
            "kind": spec["kind"],
            "architecture": spec["architecture"],
            "config": str(config_path),
            "checkpoint": str(ckpt),
            "config_exists": config_path.is_file(),
            "weight_exists": ckpt.is_file(),
            "weight_size_bytes": ckpt.stat().st_size if ckpt.is_file() else 0,
            "ready": ready,
            "note": spec.get("note", ""),
        }
    return out


def ensure_models(cfg: dict) -> dict:
    status = model_status(cfg)
    missing = []
    for group in ("sources", "targets"):
        for name, item in status[group].items():
            if not item["ready"]:
                missing.append(f"{group[:-1]}:{name}")

    report_path = Path(cfg["paths"]["cache_dir"]) / "model_audit.json"
    write_json(report_path, status)
    if missing:
        raise FileNotFoundError(
            "Manual model preparation is incomplete: "
            + ", ".join(missing)
            + ". Read PREPARE_EXPERIMENT_CN.md and place the exact checkpoints "
              "under model/ before running experiments."
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


class ResNet50Source(nn.Module):
    """ResNet50 feature source with three cached stitch stages.

    s1 = layer2 output: 28x28x512
    s2 = layer3 output: 14x14x1024
    s3 = layer4 output: 7x7x2048
    """

    def __init__(self, checkpoint: str, kind: str):
        super().__init__()
        base = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:9])
        ckpt = Path(checkpoint)

        state = _unwrap_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported ResNet checkpoint object in {ckpt}")
        state = _strip_module_prefix(state)

        if kind == "radimagenet_resnet50":
            # Official RadImageNet PyTorch notebook loads the checkpoint into a
            # wrapper whose state-dict keys begin with 'backbone.'.
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
        elif kind == "imagenet_resnet50":
            # TorchVision ImageNet state dict uses named ResNet keys.
            named_base = resnet50(weights=None)
            named_base.load_state_dict(state, strict=True)
            self.backbone = nn.Sequential(*list(named_base.children())[:9])
        else:
            raise ValueError(f"Unknown source kind: {kind}")

    def forward_stages(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # conv1/bn/relu/maxpool/layer1
        for idx in range(5):
            x = self.backbone[idx](x)
        x = self.backbone[5](x)
        s1 = x
        x = self.backbone[6](x)
        s2 = x
        x = self.backbone[7](x)
        s3 = x
        return {"s1": s1, "s2": s2, "s3": s3}


def load_source(cfg: dict, source_name: str, device: torch.device, dtype=torch.float16):
    spec = cfg["models"]["sources"][source_name]
    model = ResNet50Source(spec["checkpoint"], spec["kind"]).to(device).eval()
    model = model.to(dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_timm_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_target(cfg: dict, target_name: str, device: torch.device, dtype=torch.float16):
    spec = cfg["models"]["targets"][target_name]
    config_path = Path(spec["config"])
    ckpt_path = Path(spec["checkpoint"])
    model_cfg = _load_timm_config(config_path)

    arch = str(spec["architecture"])
    file_arch = str(model_cfg.get("architecture", arch))
    if file_arch != arch:
        raise RuntimeError(
            f"Target {target_name} architecture mismatch: config.json says "
            f"{file_arch}, experiment expects {arch}"
        )

    model = timm.create_model(arch, pretrained=False, num_classes=0)
    state = load_safetensors(str(ckpt_path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # timm snapshots with num_classes=0 should match except for a possible
    # classifier head omitted by design. Fail on anything structurally important.
    bad_missing = [k for k in missing if not k.startswith("head.")]
    bad_unexpected = [k for k in unexpected if not k.startswith("head.")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Target {target_name} checkpoint mismatch. "
            f"missing={bad_missing[:8]}, unexpected={bad_unexpected[:8]}"
        )

    model = model.to(device).eval().to(dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.inference_mode()
def extract_source_features(source, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
    return source.forward_stages(pixel_values.to(next(source.parameters()).dtype))


def target_num_prefix(target) -> int:
    return int(getattr(target, "num_prefix_tokens", 1))


def target_grid(target, input_size: int) -> int:
    patch = getattr(target.patch_embed, "patch_size", (16, 16))
    patch = int(patch[0] if isinstance(patch, (tuple, list)) else patch)
    if input_size % patch != 0:
        raise ValueError(f"input_size={input_size} must be divisible by patch_size={patch}")
    return input_size // patch


@torch.inference_mode()
def extract_target_features(target, pixel_values: torch.Tensor) -> torch.Tensor:
    h = target.forward_features(pixel_values.to(next(target.parameters()).dtype))
    if isinstance(h, dict):
        # Defensive support for timm variants returning feature dictionaries.
        for key in ("x_norm_patchtokens", "x_prenorm", "x"):
            if key in h:
                h = h[key]
                break
    if h.ndim != 3:
        raise RuntimeError(f"Expected ViT token tensor [B,N,C], got shape={tuple(h.shape)}")
    return h[:, target_num_prefix(target):, :]


def _prefix_and_patch_position(target, input_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    npre = target_num_prefix(target)
    pos = target.pos_embed.detach().float()
    grid = target_grid(target, input_size)
    expected_patches = grid * grid

    if pos.shape[1] - npre != expected_patches:
        raise RuntimeError(
            f"Position embedding has {pos.shape[1] - npre} patch tokens, "
            f"but input grid requires {expected_patches}."
        )

    prefix_parts = []
    if getattr(target, "cls_token", None) is not None:
        prefix_parts.append(target.cls_token.detach().float())
    if getattr(target, "reg_token", None) is not None:
        prefix_parts.append(target.reg_token.detach().float())
    if not prefix_parts:
        raise RuntimeError("Target ViT has no prefix token to initialize stitching.")

    prefix = torch.cat(prefix_parts, dim=1)
    if prefix.shape[1] != npre:
        raise RuntimeError(
            f"Prefix-token count mismatch: constructed={prefix.shape[1]}, model={npre}"
        )
    prefix = prefix + pos[:, :npre]
    patch_pos = pos[:, npre:]
    return prefix, patch_pos


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
        self.blocks = target.blocks
        self.norm = target.norm
        self.num_prefix = target_num_prefix(target)
        if not 0 <= self.cut_block < len(self.blocks):
            raise ValueError(f"cut_block {cut_block} out of range")
        if compile_tail and hasattr(torch, "compile"):
            self.forward = torch.compile(self.forward, mode="reduce-overhead")

    def forward(self, prefix: torch.Tensor, patches: torch.Tensor):
        h = torch.cat([prefix, patches], dim=1)
        h = h.to(next(self.target.parameters()).dtype)
        for block in self.blocks[self.cut_block:]:
            h = block(h)
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
    stage_channels = source_spec["stage_channels"]
    in_channels = int(stage_channels[int(source_stage) - 1])
    hidden_dim = int(target_spec["hidden_dim"])
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
