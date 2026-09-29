from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from modelscope import snapshot_download
from transformers import AutoModel

from .common import ensure_dir


MODEL_SUBDIRS = {
    "convnext": "dinov3_convnext_tiny",
    "vit": "dinov3_vits16",
}


def ensure_models(cfg: dict) -> Dict[str, Path]:
    model_root = ensure_dir(cfg["paths"]["model_dir"])
    pairs = {
        "convnext": (cfg["models"]["convnext_id"], model_root / MODEL_SUBDIRS["convnext"]),
        "vit": (cfg["models"]["vit_id"], model_root / MODEL_SUBDIRS["vit"]),
    }
    out: Dict[str, Path] = {}
    for key, (repo_id, dst) in pairs.items():
        config_file = dst / "config.json"
        if not config_file.exists():
            print(f"[model] downloading {repo_id} -> {dst}")
            ensure_dir(dst)
            try:
                snapshot_download(repo_id=repo_id, local_dir=str(dst), max_workers=8)
            except TypeError:
                snapshot_download(repo_id, local_dir=str(dst))
        else:
            print(f"[model] reuse {dst}")
        out[key] = dst
    return out


def load_backbones(cfg: dict, device: torch.device, dtype: torch.dtype = torch.float16):
    conv_path = Path(cfg["models"]["convnext_local"])
    vit_path = Path(cfg["models"]["vit_local"])
    if not conv_path.exists() or not vit_path.exists():
        raise FileNotFoundError("Model folders are missing. Run model download stage first.")
    conv = AutoModel.from_pretrained(
        str(conv_path), local_files_only=True, torch_dtype=dtype
    ).to(device).eval()
    vit = AutoModel.from_pretrained(
        str(vit_path), local_files_only=True, torch_dtype=dtype
    ).to(device).eval()
    for m in (conv, vit):
        for p in m.parameters():
            p.requires_grad_(False)
    return conv, vit


def _conv_stages(conv):
    candidates = [
        getattr(getattr(conv, "model", None), "stages", None),
        getattr(getattr(conv, "encoder", None), "stages", None),
        getattr(conv, "stages", None),
    ]
    for x in candidates:
        if x is not None:
            return x
    raise AttributeError("Unable to locate DINOv3 ConvNeXt stages in installed Transformers version")


def _vit_layers(vit):
    candidates = [
        getattr(getattr(vit, "model", None), "layer", None),
        getattr(getattr(vit, "encoder", None), "layer", None),
        getattr(getattr(vit, "encoder", None), "layers", None),
    ]
    for x in candidates:
        if x is not None:
            return x
    raise AttributeError("Unable to locate DINOv3 ViT transformer layers in installed Transformers version")


@torch.inference_mode()
def extract_backbone_features(conv, vit, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
    h = pixel_values.to(next(conv.parameters()).dtype)
    conv_feats = {}
    for idx, stage in enumerate(_conv_stages(conv)):
        h = stage(h)
        if idx in (1, 2, 3):
            conv_feats[f"s{idx}"] = h

    vout = vit(pixel_values.to(next(vit.parameters()).dtype), return_dict=True)
    num_prefix = 1 + int(getattr(vit.config, "num_register_tokens", 0))
    vit_patch = vout.last_hidden_state[:, num_prefix:, :]
    return {**conv_feats, "vit": vit_patch}


class StitchAdapter(nn.Module):
    CHANNELS = {1: 192, 2: 384, 3: 768}

    def __init__(
        self,
        source_stage: int,
        hidden_dim: int,
        prefix_init: torch.Tensor,
        adapter_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.source_stage = int(source_stage)
        self.hidden_dim = int(hidden_dim)
        in_ch = self.CHANNELS[self.source_stage]
        self.channel_proj = nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=True) if in_ch != hidden_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)
        self.adapter_type = adapter_type
        if adapter_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        elif adapter_type == "linear":
            self.mlp = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        self.prefix_tokens = nn.Parameter(prefix_init.detach().clone().float())

    def forward(self, fmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = fmap.float()
        if self.source_stage == 1:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        elif self.source_stage == 3:
            x = F.interpolate(x, size=(14, 14), mode="bilinear", align_corners=False)
        x = self.channel_proj(x)
        if x.shape[-2:] != (14, 14):
            x = F.interpolate(x, size=(14, 14), mode="bilinear", align_corners=False)
        x = x.flatten(2).transpose(1, 2).contiguous()
        base = x
        x = self.mlp(self.norm(x))
        patches = base + x
        prefix = self.prefix_tokens.expand(patches.shape[0], -1, -1)
        return prefix, patches


class ViTTail(nn.Module):
    def __init__(self, vit, cut_block: int, image_size: int = 224, compile_tail: bool = False):
        super().__init__()
        self.vit = vit
        self.cut_block = int(cut_block)
        self.num_prefix = 1 + int(getattr(vit.config, "num_register_tokens", 0))
        self.layers = _vit_layers(vit)
        if not 0 <= self.cut_block < len(self.layers):
            raise ValueError(f"cut_block {cut_block} is outside ViT range")
        dummy = torch.zeros(
            1, 3, image_size, image_size,
            device=next(vit.parameters()).device,
            dtype=next(vit.parameters()).dtype,
        )
        with torch.inference_mode():
            cos, sin = vit.rope_embeddings(dummy)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        if compile_tail and hasattr(torch, "compile"):
            self.forward = torch.compile(self.forward, mode="reduce-overhead")  # type: ignore[method-assign]

    def forward(self, prefix: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        dtype = next(self.vit.parameters()).dtype
        h = torch.cat([prefix, patches], dim=1).to(dtype)
        rope = (self.rope_cos, self.rope_sin)
        for layer in self.layers[self.cut_block:]:
            h = layer(h, position_embeddings=rope)
            if isinstance(h, (tuple, list)):
                h = h[0]
        h = self.vit.norm(h)
        return h[:, self.num_prefix:, :]


def build_stitch_modules(cfg: dict, vit, source_stage: int, target_block: int, adapter_type: str):
    prefix_init = torch.cat(
        [vit.embeddings.cls_token.detach(), vit.embeddings.register_tokens.detach()], dim=1
    )
    adapter = StitchAdapter(
        source_stage=source_stage,
        hidden_dim=int(cfg["models"]["hidden_dim"]),
        prefix_init=prefix_init,
        adapter_type=adapter_type,
    )
    tail = ViTTail(
        vit,
        target_block,
        image_size=int(cfg["project"]["image_size"]),
        compile_tail=bool(cfg["project"].get("compile_tail", False)),
    )
    return adapter, tail
