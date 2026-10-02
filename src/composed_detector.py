"""Materialize inherited parameter modules as an image-to-anomaly model.

This is the original method, with its original reference branch and score.
The stitched path owns only the selected CNN prefix and ViT suffix. The suffix
shares parameter objects with the reference encoder; it is not an ensemble of
CNN/ViT anomaly scores. Bundles contain all weights and need no model downloads.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache_features import _normalize
from .models import (_target_layers, _target_norm, build_stitch_modules,
                     target_num_prefix)
from .train_eval import _score_from_map


def source_prefix(source, stage):
    if stage not in (1, 2, 3):
        raise ValueError("Source stage must be 1, 2 or 3")
    if hasattr(source, "backbone"):
        modules = list(source.backbone.children())[:6 + stage - 1]
    else:
        b = source.base
        activation = b.act1 if hasattr(b, "act1") else b.relu
        modules = [b.conv1, b.bn1, activation, b.maxpool, b.layer1, b.layer2,
                   b.layer3, b.layer4][:6 + stage - 1]
    prefix = nn.Sequential(*modules).eval()
    for p in prefix.parameters():
        p.requires_grad_(False)
    return prefix


class ParameterTail(nn.Module):
    """Actual inherited suffix only; no hidden registration of the full ViT."""
    def __init__(self, target, cut):
        super().__init__()
        layers = _target_layers(target)
        if not 0 <= cut < len(layers):
            raise ValueError("Target cut outside Transformer layers")
        self.layers = nn.ModuleList(list(layers[cut:]))
        self.norm = _target_norm(target)
        self.num_prefix = target_num_prefix(target)

    def forward(self, prefix, patches):
        h = torch.cat([prefix, patches], 1).to(next(self.norm.parameters()).dtype)
        for layer in self.layers:
            h = layer(h)
            if isinstance(h, (tuple, list)):
                h = h[0]
        return self.norm(h)[:, self.num_prefix:]


class DirectFeatureTail(nn.Module):
    """No Transformer blocks: same adapter, fixed inherited output norm.

    A trained control, not an inference-time removal from a trained stitch.
    Prefix tokens are unused here; report effective trainable parameter counts.
    """
    def __init__(self, target):
        super().__init__()
        self.norm = _target_norm(target)

    def forward(self, prefix, patches):
        return self.norm(patches.to(next(self.norm.parameters()).dtype))


class ComposedAnomalyDetector(nn.Module):
    def __init__(self, prefix, adapter, target, source_spec, target_spec, cut):
        super().__init__()
        self.prefix = prefix
        self.adapter = adapter
        # Persist positions: a bundle must not depend on external initialization.
        adapter._non_persistent_buffers_set.discard("patch_position")
        self.reference = target
        self.tail = ParameterTail(target, cut)
        self.source_spec = copy.deepcopy(source_spec)
        self.target_spec = copy.deepcopy(target_spec)
        for module in [self.prefix, self.reference]:
            for p in module.parameters():
                p.requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not torch.isfinite(rgb).all():
            raise ValueError("Expected finite NCHW RGB in [0,1]")
        if rgb.min() < 0 or rgb.max() > 1:
            raise ValueError("Expected RGB in [0,1]")
        x = _normalize(rgb, self.source_spec).to(next(self.prefix.parameters()).dtype)
        y = _normalize(rgb, self.target_spec).to(next(self.reference.parameters()).dtype)
        fmap = self.prefix(x)
        prefix, patches = self.adapter(fmap)
        stitched = self.tail(prefix, patches).float()
        reference = self.reference(y, return_dict=True).last_hidden_state[:, target_num_prefix(self.reference):].float()
        d = 1 - F.cosine_similarity(stitched, reference, dim=-1)
        score = _score_from_map(d, .05, "contrast_topk")
        grid = self.adapter.grid
        return dict(score=score, discrepancy=d.reshape(-1, grid, grid),
                    anomaly_map=(d - d.median(1, keepdim=True).values).clamp_min(0).reshape(-1, grid, grid))

    def parameter_counts(self):
        # nn.Module.parameters deduplicates shared suffix/reference parameters.
        return dict(total_unique=sum(p.numel() for p in self.parameters()),
                    cnn_prefix=sum(p.numel() for p in self.prefix.parameters()),
                    adapter=sum(p.numel() for p in self.adapter.parameters()),
                    inherited_suffix=sum(p.numel() for p in self.tail.parameters()),
                    full_reference=sum(p.numel() for p in self.reference.parameters()))


def tensor_digest(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        x = tensor.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode()); h.update(x.numpy().tobytes())
    return h.hexdigest()


def save_bundle(path, detector, cfg, job, held, provenance):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    states = {k: {n: v.detach().cpu().clone() for n, v in getattr(detector, k).state_dict().items()}
              for k in ["prefix", "adapter", "reference"]}
    payload = dict(format="medstitch_parameter_composition_v1", cfg=copy.deepcopy(cfg),
                   job=job, held_out=held, reference_config=detector.reference.config.to_dict(),
                   attention_implementation=getattr(detector.reference.config, "_attn_implementation", None),
                   states=states, state_sha256={k: tensor_digest(v) for k, v in states.items()},
                   provenance=provenance, score_mode="contrast_topk", topk_fraction=.05)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload["state_sha256"]


def load_bundle(path, device):
    """Construct topology locally and load strictly; never read original weights."""
    import timm
    from torchvision.models import resnet50
    from transformers import AutoConfig, AutoModel

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload["format"] != "medstitch_parameter_composition_v1"
            or payload["score_mode"] != "contrast_topk" or payload["topk_fraction"] != .05):
        raise ValueError("Unsupported composition bundle/protocol")
    for key, state in payload["states"].items():
        if tensor_digest(state) != payload["state_sha256"][key]:
            raise ValueError(f"Corrupt parameter module: {key}")
    cfg, job = payload["cfg"], payload["job"]
    spec = cfg["models"]["sources"][job["source_backbone"]]
    wrapper = nn.Module()
    if spec["kind"] == "radimagenet_resnet50":
        wrapper.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:9])
    elif spec["kind"] == "timm_resnet50":
        wrapper.base = timm.create_model(spec["architecture"], pretrained=False, num_classes=1000)
    else:
        raise ValueError("Unsupported source topology")
    conf = dict(payload["reference_config"])
    model_type = conf.pop("model_type")
    implementation = payload.get("attention_implementation")
    kwargs = dict(attn_implementation=implementation) if implementation else {}
    target = AutoModel.from_config(AutoConfig.for_model(model_type, **conf), **kwargs)
    dtype = next(v.dtype for v in payload["states"]["reference"].values() if v.is_floating_point())
    target = target.to(dtype=dtype)
    target.load_state_dict(payload["states"]["reference"], strict=True)
    prefix = source_prefix(wrapper, job["source_stage"]).to(dtype=dtype)
    prefix.load_state_dict(payload["states"]["prefix"], strict=True)
    adapter, _ = build_stitch_modules(cfg, target, job["source_backbone"], job["target_backbone"],
                                     job["source_stage"], job["target_block"], "mlp")
    detector = ComposedAnomalyDetector(prefix, adapter, target, spec,
                cfg["models"]["targets"][job["target_backbone"]], job["target_block"])
    adapter.load_state_dict(payload["states"]["adapter"], strict=True)
    return detector.to(device).eval(), payload
