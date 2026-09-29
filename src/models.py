from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope import snapshot_download
from transformers import AutoModel

from .common import ensure_dir


def ensure_models(cfg: dict) -> Dict[str, Path]:
    items = {"source": (cfg["models"]["source"]["id"], Path(cfg["models"]["source"]["local"]))}
    for name, spec in cfg["models"]["targets"].items():
        items[f"target:{name}"] = (spec["id"], Path(spec["local"]))
    out = {}
    for key, (repo_id, dst) in items.items():
        if not (dst / "config.json").exists():
            print(f"[model] ModelScope download {repo_id} -> {dst}")
            ensure_dir(dst)
            try:
                snapshot_download(repo_id=repo_id, local_dir=str(dst), max_workers=8)
            except TypeError:
                snapshot_download(repo_id, local_dir=str(dst))
        else:
            print(f"[model] reuse {dst}")
        out[key] = dst
    return out


def _load_local(path: str, device: torch.device, dtype=torch.float16):
    model = AutoModel.from_pretrained(path, local_files_only=True, torch_dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_source(cfg: dict, device: torch.device, dtype=torch.float16):
    return _load_local(cfg["models"]["source"]["local"], device, dtype)


def load_target(cfg: dict, target_name: str, device: torch.device, dtype=torch.float16):
    return _load_local(cfg["models"]["targets"][target_name]["local"], device, dtype)


def _conv_stages(conv):
    for x in (
        getattr(getattr(conv, "model", None), "stages", None),
        getattr(getattr(conv, "encoder", None), "stages", None),
        getattr(conv, "stages", None),
    ):
        if x is not None:
            return x
    raise AttributeError("Unable to locate DINOv3 ConvNeXt stages")


def _target_family(target) -> str:
    mt = str(getattr(target.config, "model_type", "")).lower()
    if mt == "dinov2":
        return "dinov2"
    if mt in {"dinov3_vit", "dinov3"}:
        return "dinov3"
    raise ValueError(f"Unsupported target model_type={mt}")


def _target_layers(target):
    return target.encoder.layer if _target_family(target) == "dinov2" else target.model.layer


def _target_norm(target):
    return target.layernorm if _target_family(target) == "dinov2" else target.norm


def target_num_prefix(target) -> int:
    return 1 if _target_family(target) == "dinov2" else 1 + int(getattr(target.config, "num_register_tokens", 0))


def target_grid(target, input_size: int) -> int:
    patch = getattr(target.config, "patch_size", 16)
    if isinstance(patch, (tuple, list)):
        patch = patch[0]
    if input_size % int(patch) != 0:
        raise ValueError(f"input_size={input_size} must be divisible by patch_size={patch}")
    return input_size // int(patch)


@torch.inference_mode()
def extract_source_features(source, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
    h = pixel_values.to(next(source.parameters()).dtype)
    feats = {}
    for idx, stage in enumerate(_conv_stages(source)):
        h = stage(h)
        if idx in (1, 2, 3):
            feats[f"s{idx}"] = h
    return feats


@torch.inference_mode()
def extract_target_features(target, pixel_values: torch.Tensor) -> torch.Tensor:
    out = target(pixel_values.to(next(target.parameters()).dtype), return_dict=True)
    return out.last_hidden_state[:, target_num_prefix(target):, :]


def _prefix_and_patch_position(target, input_size: int) -> Tuple[torch.Tensor, torch.Tensor | None]:
    if _target_family(target) == "dinov3":
        prefix = torch.cat([target.embeddings.cls_token.detach(), target.embeddings.register_tokens.detach()], dim=1)
        return prefix.float(), None
    grid = target_grid(target, input_size)
    hidden = int(target.config.hidden_size)
    device = next(target.parameters()).device
    dtype = next(target.parameters()).dtype
    dummy = torch.zeros(1, 1 + grid * grid, hidden, device=device, dtype=dtype)
    pos = target.embeddings.interpolate_pos_encoding(dummy, input_size, input_size)
    prefix = target.embeddings.cls_token.detach().to(pos.dtype) + pos[:, :1]
    return prefix.float(), pos[:, 1:].float()


class StitchAdapter(nn.Module):
    CHANNELS = {1: 192, 2: 384, 3: 768}

    def __init__(self, source_stage: int, hidden_dim: int, grid: int,
                 prefix_init: torch.Tensor, patch_position: torch.Tensor | None,
                 adapter_type: str = "mlp"):
        super().__init__()
        self.grid = int(grid)
        in_ch = self.CHANNELS[int(source_stage)]
        self.channel_proj = nn.Conv2d(in_ch, hidden_dim, 1) if in_ch != hidden_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)
        if adapter_type == "mlp":
            self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))
        elif adapter_type == "linear":
            self.mlp = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError(adapter_type)
        self.prefix_tokens = nn.Parameter(prefix_init.detach().clone().float())
        self.register_buffer("patch_position", None if patch_position is None else patch_position.detach().clone().float(), persistent=False)

    def forward(self, fmap: torch.Tensor):
        x = fmap.float()
        h, w = x.shape[-2:]
        if (h, w) != (self.grid, self.grid):
            x = F.adaptive_avg_pool2d(x, (self.grid, self.grid)) if h > self.grid else F.interpolate(
                x, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        x = self.channel_proj(x).flatten(2).transpose(1, 2).contiguous()
        patches = x + self.mlp(self.norm(x))
        if self.patch_position is not None:
            patches = patches + self.patch_position
        return self.prefix_tokens.expand(patches.shape[0], -1, -1), patches


class TargetTail(nn.Module):
    def __init__(self, target, cut_block: int, input_size: int, compile_tail: bool = False):
        super().__init__()
        self.target = target
        self.family = _target_family(target)
        self.cut_block = int(cut_block)
        self.num_prefix = target_num_prefix(target)
        self.layers = _target_layers(target)
        self.norm = _target_norm(target)
        if not 0 <= self.cut_block < len(self.layers):
            raise ValueError(f"cut_block {cut_block} out of range")
        if self.family == "dinov3":
            dummy = torch.zeros(1, 3, input_size, input_size, device=next(target.parameters()).device,
                                dtype=next(target.parameters()).dtype)
            with torch.inference_mode():
                cos, sin = target.rope_embeddings(dummy)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.register_buffer("rope_cos", None, persistent=False)
            self.register_buffer("rope_sin", None, persistent=False)
        if compile_tail and hasattr(torch, "compile"):
            self.forward = torch.compile(self.forward, mode="reduce-overhead")

    def forward(self, prefix: torch.Tensor, patches: torch.Tensor):
        h = torch.cat([prefix, patches], 1).to(next(self.target.parameters()).dtype)
        if self.family == "dinov3":
            rope = (self.rope_cos, self.rope_sin)
            for layer in self.layers[self.cut_block:]:
                h = layer(h, position_embeddings=rope)
                if isinstance(h, (tuple, list)):
                    h = h[0]
        else:
            for layer in self.layers[self.cut_block:]:
                h = layer(h)
                if isinstance(h, (tuple, list)):
                    h = h[0]
        return self.norm(h)[:, self.num_prefix:, :]


def build_stitch_modules(cfg: dict, target, target_name: str, source_stage: int,
                         target_block: int, adapter_type: str):
    spec = cfg["models"]["targets"][target_name]
    input_size = int(spec.get("input_size", cfg["project"]["image_size"]))
    prefix, patch_pos = _prefix_and_patch_position(target, input_size)
    adapter = StitchAdapter(source_stage, int(target.config.hidden_size), target_grid(target, input_size),
                            prefix, patch_pos, adapter_type)
    tail = TargetTail(target, target_block, input_size, bool(cfg["project"].get("compile_tail", False)))
    return adapter, tail
