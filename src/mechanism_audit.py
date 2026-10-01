"""Source-normal mechanism audit of the original NFFA/TargetTail; no training/selection."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .cache_features import _normalize, load_rgb_raw
from .common import configure_torch, seed_everything
from .followup_io import common_jobs, read_json, read_records
from .models import TargetTail, _target_layers, build_stitch_modules, load_target, target_num_prefix
from .prediction_io import checkpoint_identity, sha256, validate_final
from .run_followup import inputs_unchanged, write_json


SAMPLES = 16
CUTS = (3, 6, 9)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_datasets(mapping, held_out):
    require(held_out in set(mapping.values()), "Unknown held-out modality")
    return [d for d, m in mapping.items() if m != held_out]


def sample_indices(n):
    require(n >= SAMPLES, "Need at least 16 cached normal training images")
    return np.linspace(0, n - 1, SAMPLES, dtype=int)


def loss_components(pred, reference):
    pred, reference = pred.float(), reference.float()
    patch = 1 - F.cosine_similarity(pred, reference, dim=-1).mean()
    glob = 1 - F.cosine_similarity(pred.mean(1), reference.mean(1), dim=-1).mean()
    return patch, glob, patch + .1 * glob


@torch.no_grad()
def tail_identity(target, pixels, cuts=CUTS):
    """Capture inputs to actual blocks, never infer hidden-state tuple indexing."""
    captured, handles = {}, []
    for cut in cuts:
        def capture(module, args, cut=cut):
            captured[cut] = args[0].detach().clone()
        handles.append(_target_layers(target)[cut].register_forward_pre_hook(capture))
    try:
        full = target(pixels.to(next(target.parameters()).dtype), return_dict=True).last_hidden_state
    finally:
        for h in handles:
            h.remove()
    prefix = target_num_prefix(target)
    expected = full[:, prefix:]
    rows = []
    for cut in cuts:
        h = captured[cut]
        actual = TargetTail(target, cut).eval()(h[:, :prefix], h[:, prefix:])
        require(actual.shape == expected.shape and torch.isfinite(actual).all(), "Invalid tail output")
        delta = (actual.float() - expected.float()).abs()
        # Fixed numerical gate for replaying the same layers/inputs, not an AUROC threshold.
        ok = torch.allclose(actual.float(), expected.float(), atol=1e-3, rtol=1e-3)
        rows.append(dict(cut_index=cut, first_human_block=cut + 1, max_abs_delta=float(delta.max()),
                         mean_abs_delta=float(delta.mean()), within_tolerance=bool(ok)))
    return expected, rows


def gradient_probe(adapter, tail, fmap, reference):
    require(not any(p.requires_grad for p in tail.parameters()), "Backbone is not frozen")
    adapter.zero_grad(set_to_none=True)
    prefix, patches = adapter(fmap)
    loss = loss_components(tail(prefix, patches), reference)[2]
    require(torch.isfinite(loss), "Nonfinite probe loss")
    loss.backward()
    grads = {n: p.grad for n, p in adapter.named_parameters() if p.requires_grad}
    require(grads and all(g is not None and torch.isfinite(g).all() for g in grads.values()), "Missing/nonfinite adapter gradients")
    norm = sum(float(g.float().square().sum()) for g in grads.values()) ** .5
    require(norm > 0 and all(p.grad is None for p in tail.parameters()), "Gradient flow/freeze failure")
    adapter.zero_grad(set_to_none=True)
    return dict(loss=float(loss.detach()), adapter_gradient_norm=norm, backbone_gradients_absent=True,
                optimizer_steps=0)


def preflight(cfg, config):
    root = Path(cfg["paths"]["results_dir"])
    selected = read_json(root / "selected_configs.json")
    jobs = [j for j in common_jobs(cfg, selected) if j["reuse_original"]]
    paths = [Path(config), root / "selected_configs.json"]
    paths += [Path(__file__), Path(__file__).with_name("models.py"), Path(__file__).with_name("train_eval.py"),
              Path(__file__).with_name("cache_features.py")]
    records, arrays, samples = {}, [], {}
    for ds in cfg["data"]["datasets"]:
        directory = Path(cfg["paths"]["cache_dir"]) / "features" / ds / "train"
        require((directory / ".done").is_file(), f"Missing train cache: {directory}")
        rec = read_records(directory / "records.jsonl")
        require(all(r["dataset"] == ds and r["split"] == "train" and r["label"] == 0
                    and r["modality"] == cfg["data"]["modality_map"][ds] for r in rec), "Need normal train records")
        require(len({r["image"] for r in rec}) == len(rec), "Duplicate train paths")
        ids = sample_indices(len(rec))
        records[ds], samples[ds] = rec, ids.tolist()
        paths += [directory / "records.jsonl", directory / ".done"]
        # Only two images per dataset are decoded for target interface/cache checks.
        paths += [Path(rec[int(i)]["image"]) for i in ids[:2]]
        for key in ["source_medical_s2", "source_general_s2", "target_medical", "target_general"]:
            p = directory / f"{key}.npy"
            require(len(np.load(p, mmap_mode="r")) == len(rec), f"Cache length differs: {p}")
            arrays.append(p)
    for job in jobs:
        p = root / "final" / f"{job['job']}.json"
        final = read_json(p)
        validate_final(cfg, selected, final, job["source_backbone"], job["target_backbone"], 2,
                       job["target_block"], job["seed"], "mlp", .05, ["contrast_topk"])
        paths.append(p)
        paths += [root / "final/checkpoints" / job["job"] / f"{m}.pt" for m in set(cfg["data"]["modality_map"].values())]
    for spec in cfg["models"]["targets"].values():
        directory = Path(spec["local"])
        paths.append(directory / "config.json")
        weights = list(directory.glob("*.safetensors")) + list(directory.glob("pytorch_model*.bin"))
        require(bool(weights), f"Missing local target weights: {directory}")
        arrays += weights
    require(all(p.is_file() for p in paths), "Missing raw normal images, checkpoints or model configs; no download/rebuild fallback")
    return jobs, records, samples, paths, arrays


def cached(cfg, dataset, name, ids, device):
    p = Path(cfg["paths"]["cache_dir"]) / "features" / dataset / "train" / f"{name}.npy"
    return torch.from_numpy(np.array(np.load(p, mmap_mode="r")[ids], copy=True)).to(device)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(cfg, jobs, records, samples, out, device):
    mapping = cfg["data"]["modality_map"]
    interface_rows, alignment, gradients, aoss_rows = [], [], [], []
    for target_name in ["medical", "general"]:
        target = load_target(cfg, target_name, device)
        for ds in cfg["data"]["datasets"]:
            ids = samples[ds][:2]
            raw = torch.stack([load_rgb_raw(records[ds][i]["image"], 224) for i in ids]).to(device)
            pixels = _normalize(raw, cfg["models"]["targets"][target_name])
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                fresh, checks = tail_identity(target, pixels)
            old = cached(cfg, ds, f"target_{target_name}", ids, device)
            require(fresh.shape == old.shape and torch.isfinite(old).all(), "Target cache geometry/nonfinite")
            cache_delta = float((fresh.float() - old.float()).abs().max())
            cache_cos = float(loss_components(fresh, old)[0])
            for r in checks:
                interface_rows.append(dict(target=target_name, dataset=ds, n_images=2, **r,
                                           cache_max_abs_delta=cache_delta, cache_patch_cosine_loss=cache_cos))
            write_csv(out / "interfaces.csv", interface_rows)
            require(all(r["within_tolerance"] for r in checks), "TargetTail does not reproduce full encoder; stop before adapter audit")
        for job in [j for j in jobs if j["target_backbone"] == target_name]:
            final = read_json(Path(cfg["paths"]["results_dir"]) / "final" / f"{job['job']}.json")
            for held in sorted(set(mapping.values())):
                seed_everything(job["seed"], bool(cfg["runtime"].get("deterministic", False)))
                ckpt = Path(cfg["paths"]["results_dir"]) / "final/checkpoints" / job["job"] / f"{held}.pt"
                saved = torch.load(ckpt, map_location="cpu", weights_only=False)
                checkpoint_identity(saved, job["source_backbone"], target_name, 2, job["target_block"], "mlp", held)
                adapter, tail = build_stitch_modules(cfg, target, job["source_backbone"], target_name, 2, job["target_block"], "mlp")
                adapter.load_state_dict(saved["adapter"], strict=True)
                adapter, tail = adapter.to(device).eval(), tail.to(device).eval()
                for di, ds in enumerate(source_datasets(mapping, held)):
                    x = cached(cfg, ds, f"source_{job['source_backbone']}_s2", samples[ds], device)
                    y = cached(cfg, ds, f"target_{target_name}", samples[ds], device)
                    with torch.no_grad():
                        prefix, patches = adapter(x)
                        pred = tail(prefix, patches)
                        # Cyclic shift pairs each prediction with another image's reference.
                        for control, p in [("matched", pred), ("shifted_image", pred.roll(1, 0))]:
                            patch, glob, total = loss_components(p, y)
                            require(torch.isfinite(total), "Nonfinite alignment")
                            alignment.append(dict(job=job["job"], held_out=held, source_dataset=ds,
                                source_modality=mapping[ds], control=control, n_images=len(x),
                                patch_loss=float(patch), global_loss=float(glob), nffa_loss=float(total)))
                    if di == 0:
                        probe = gradient_probe(adapter, tail, x[:2], y[:2])
                        gradients.append(dict(job=job["job"], held_out=held, source_dataset=ds, **probe))
                a = next(r for r in final["aoss_rows"] if r["target_modality"] == held)
                expected = (a["perturb_in"] - a["perturb_out"]) / (a["normal_discrepancy"] + 1e-8)
                require(np.isfinite(expected) and abs(expected - a["aoss"]) <= 1e-6, "Stored AOSS formula mismatch")
                aoss_rows.append(dict(job=job["job"], held_out=held, **a, formula_abs_delta=abs(expected - a["aoss"])))
                del adapter, tail, saved
            write_csv(out / "alignment.csv", alignment)
            write_csv(out / "gradients.csv", gradients)
            write_csv(out / "aoss_components.csv", aoss_rows)
            print(f"[mechanism] Complete: {job['job']}", flush=True)
        del target
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    jobs, records, samples, paths, arrays = preflight(cfg, args.config)
    if args.check_only:
        print("Ready: 12 locked jobs / 60 checkpoints; normal train inputs only; no files written")
        return
    require(torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count(), "Need an available CUDA GPU")
    out = Path(cfg["paths"]["results_dir"]) / "followup/mechanism_audit"
    require(not (out / "run_manifest.json").exists(), "Existing audit: archive it before an explicit rerun; no overwrite")
    hashes = {str(p): sha256(p) for p in set(paths)}
    stats = {str(p): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in set(arrays)}
    manifest = dict(status="running", no_training=True, no_reselection=True, target_test_read=False,
        sample_indices=samples, original_sha256=hashes, array_weight_stats=stats,
        versions={p: importlib.metadata.version(p) for p in ["torch", "transformers", "timm", "numpy"]}, python=platform.python_version(),
        limits="Normal train subset may overlap adapter training. Shift control is within dataset. Cache deltas are descriptive, not an automatic cache-rebuild trigger. Large arrays/weights stat-only; legacy seed provenance unchanged.")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_manifest.json", manifest)
    configure_torch(bool(cfg["project"].get("allow_tf32", True)))
    try:
        run(cfg, jobs, records, samples, out, torch.device(f"cuda:{args.gpu}"))
        require(inputs_unchanged(hashes, stats), "Original inputs changed")
        manifest.update(status="complete", original_inputs_unchanged=True,
                        output_sha256={p.name: sha256(p) for p in out.glob("*.csv")})
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc), original_inputs_unchanged=inputs_unchanged(hashes, stats))
        raise
    finally:
        write_json(out / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
