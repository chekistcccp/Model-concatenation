"""Source-only paired perturbation audit of locked final checkpoints.

Measures normal spatial contrast and perturbation-induced response separately.
Never trains, selects a stitch, or reads target test/valid features.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import math
import platform
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import configure_torch, seed_everything
from .followup_io import check_cache, common_jobs, read_json, read_records
from .models import build_stitch_modules, load_target
from .prediction_io import checkpoint_identity, sha256, validate_final
from .run_followup import inputs_unchanged, write_json
from .train_eval import (_balanced_indices, _score_from_map, compute_aoss,
                         discrepancy, modality_datasets, target_modalities)


TOLERANCE = 1e-6
BATCH_SIZE = 256  # Original compute_aoss default; not a tunable audit parameter.


def require(ok, message):
    if not ok:
        raise ValueError(message)


def paired_regions(normal, perturbed, mask):
    """Per-image statistics; missing mask regions stay undefined, not zero."""
    n = np.asarray(normal, dtype=np.float64)
    p = np.asarray(perturbed, dtype=np.float64)
    m = np.asarray(mask)
    require(n.shape == p.shape == m.shape and n.ndim == 2, "Paired map geometry differs")
    require(np.isfinite(n).all() and np.isfinite(p).all(), "Nonfinite paired maps")
    require(np.isin(m, [0, 1]).all(), "Mask must be binary")
    m = m.astype(bool)
    rows = []
    for a, b, inside in zip(n, p, m):
        outside = ~inside
        ni, no = int(inside.sum()), int(outside.sum())
        normal_in = float(a[inside].mean()) if ni else None
        normal_out = float(a[outside].mean()) if no else None
        pert_in = float(b[inside].mean()) if ni else None
        pert_out = float(b[outside].mean()) if no else None
        delta_in = float((b - a)[inside].mean()) if ni else None
        delta_out = float((b - a)[outside].mean()) if no else None
        valid = bool(ni and no)
        baseline = normal_in - normal_out if valid else None
        raw = pert_in - pert_out if valid else None
        net = delta_in - delta_out if valid else None
        require(not valid or abs(raw - baseline - net) < 1e-10, "Paired decomposition failed")
        rows.append(dict(n_mask_in=ni, n_mask_out=no, valid_regions=valid,
            normal_in=normal_in, normal_out=normal_out, perturb_in=pert_in, perturb_out=pert_out,
            normal_spatial_contrast=baseline, raw_local_sensitivity=raw,
            delta_in=delta_in, delta_out=delta_out, net_local_response=net,
            normal_mean=float(a.mean()), perturb_mean=float(b.mean()), mean_delta=float((b - a).mean())))
    return rows


def reconstructed_training_images(cfg, held_out, seed, records):
    """Reconstruct current original sampling code; legacy history is not proven."""
    images = {}
    budget = cfg["final"]["n_per_source_modality"]
    for modality, datasets in modality_datasets(cfg).items():
        if modality == held_out:
            continue
        take = max(1, math.ceil(budget / len(datasets)))
        remaining = budget
        for di, ds in enumerate(datasets):
            indices = _balanced_indices(len(records[ds]), min(take, len(records[ds])), seed + 101 * di + len(modality))
            used = indices[:max(0, remaining)]
            images[ds] = {records[ds][int(i)]["image"] for i in used}
            remaining -= len(used)
    return images


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def versions():
    return {p: importlib.metadata.version(p) for p in ["torch", "transformers", "timm", "numpy"]}


def preflight(cfg, config):
    root = Path(cfg["paths"]["results_dir"])
    selected_path = root / "selected_configs.json"
    selected = read_json(selected_path)
    jobs = [j for j in common_jobs(cfg, selected) if j["reuse_original"]]
    prerequisite = root / "followup/mechanism_audit"
    prior = read_json(prerequisite / "run_manifest.json")
    require(prior["status"] == "complete" and prior["original_inputs_unchanged"]
            and prior["no_training"] and prior["no_reselection"] and not prior["target_test_read"], "Need complete original mechanism audit")
    require(prior["versions"] == versions() and prior["python"] == platform.python_version(),
            "Environment differs from mechanism audit; restore its environment, do not silently mix runs")
    expected = {"interfaces.csv", "alignment.csv", "gradients.csv", "aoss_components.csv"}
    require(set(prior["output_sha256"]) == expected, "Incomplete mechanism artifacts")
    paths = [Path(config), selected_path, prerequisite / "run_manifest.json"]
    for name, digest in prior["output_sha256"].items():
        p = prerequisite / name
        require(sha256(p) == digest, f"Changed mechanism artifact: {p}")
        paths.append(p)
    interface = csv_rows(prerequisite / "interfaces.csv")
    require(len(interface) == 36 and all(r["within_tolerance"] == "True" for r in interface), "Interface prerequisite failed")
    require(sha256(selected_path) == prior["original_sha256"][str(selected_path)]
            and sha256(config) == prior["original_sha256"][str(config)], "Original selection/config changed")
    for job in jobs:
        p = root / "final" / f"{job['job']}.json"
        final = read_json(p)
        validate_final(cfg, selected, final, job["source_backbone"], job["target_backbone"], 2,
                       job["target_block"], job["seed"], "mlp", .05, ["contrast_topk"])
        paths.append(p)
        for held in target_modalities(cfg):
            ckpt = root / "final/checkpoints" / job["job"] / f"{held}.pt"
            require(sha256(ckpt) == prior["original_sha256"][str(ckpt)], "Checkpoint differs from mechanism audit")
            paths.append(ckpt)
        require(sha256(p) == prior["original_sha256"][str(p)], "Original final differs")
    source_keys = [f"source_{s}_s2" for s in ["medical", "general"]]
    target_keys = [f"target_{t}" for t in ["medical", "general"]]
    train, _ = check_cache(cfg, "train", source_keys + target_keys)
    perturb, counts = check_cache(cfg, "perturb", [f"{a}_{k}" for a in ["normal", "pert"] for k in source_keys + target_keys]
                                 + [f"mask_{t}" for t in ["medical", "general"]])
    require(all(n == 256 for n in counts.values()), "Original perturb cache must contain 256 records per dataset")
    records = {ds: read_records(Path(cfg["paths"]["cache_dir"]) / "features" / ds / "train/records.jsonl") for ds in cfg["data"]["datasets"]}
    for p in train:
        if p.suffix != ".npy":
            require(sha256(p) == prior["original_sha256"][str(p)], "Normal train records/marker changed")
    all_inputs = train + perturb
    arrays = [p for p in all_inputs if p.suffix == ".npy"]
    paths += [p for p in all_inputs if p.suffix != ".npy"]
    for spec in cfg["models"]["targets"].values():
        directory = Path(spec["local"])
        p = directory / "config.json"
        require(sha256(p) == prior["original_sha256"][str(p)], "Target model config changed")
        paths.append(p)
        weights = list(directory.glob("*.safetensors")) + list(directory.glob("pytorch_model*.bin"))
        require(bool(weights), f"Missing target weights: {directory}")
        arrays += weights
    for p in arrays:
        if str(p) in prior["array_weight_stats"]:
            require(dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) == prior["array_weight_stats"][str(p)],
                    f"Prior array/weight stat differs: {p}")
    paths += [Path(__file__).parent / f"{name}.py" for name in ["paired_response_audit", "train_eval", "models", "cache_features", "followup_io", "prediction_io", "worker", "pipeline", "common"]]
    return jobs, records, sorted(set(paths), key=str), sorted(set(arrays), key=str)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


@torch.inference_mode()
def export_fold(cfg, job, held, adapter, tail, device, membership, output):
    rows = []
    for modality, datasets in modality_datasets(cfg).items():
        if modality == held:
            continue
        for ds in datasets:
            directory = Path(cfg["paths"]["cache_dir"]) / "perturb" / ds
            records = read_records(directory / "records.jsonl")
            source, target = job["source_backbone"], job["target_backbone"]
            names = [f"normal_source_{source}_s2", f"normal_target_{target}", f"pert_source_{source}_s2", f"pert_target_{target}"]
            arrays = [np.load(directory / f"{n}.npy", mmap_mode="r") for n in names]
            masks = np.load(directory / f"mask_{target}.npy", mmap_mode="r")
            for start in range(0, len(records), BATCH_SIZE):
                stop = min(len(records), start + BATCH_SIZE)
                tensors = [torch.from_numpy(np.array(a[start:stop], copy=True)) for a in arrays]
                dn = discrepancy(adapter, tail, tensors[0], tensors[1], device)
                dp = discrepancy(adapter, tail, tensors[2], tensors[3], device)
                mask = np.array(masks[start:stop], copy=True).reshape(tuple(dn.shape))
                stats = paired_regions(dn.cpu().numpy(), dp.cpu().numpy(), mask)
                ns = _score_from_map(dn, .05, "contrast_topk").cpu().tolist()
                ps = _score_from_map(dp, .05, "contrast_topk").cpu().tolist()
                for offset, stats_row in enumerate(stats):
                    i = start + offset
                    rec = records[i]
                    rows.append(dict(job=job["job"], seed=job["seed"], source_backbone=source, target_backbone=target,
                        held_out_modality=held, source_modality=modality, dataset=ds, record_index=i, image_path=rec["image"],
                        perturbation_kind_inferred=["intensity", "blur", "patch_copy"][(cfg["project"]["seed"] + i) % 3],
                        kind_evidence="inferred_from_current_generator_and_record_index",
                        training_membership_inferred=rec["image"] in membership[ds],
                        membership_evidence="reconstructed_current_original_sampling; legacy checkpoints lack historical sample manifest",
                        **stats_row, normal_contrast_topk=ns[offset], perturb_contrast_topk=ps[offset], contrast_delta=ps[offset] - ns[offset]))
    write_csv(output, rows)
    return len(rows)


def run(cfg, jobs, records, out, device):
    checks, membership_counts, n_rows = [], [], 0
    for target_name in ["medical", "general"]:
        target = load_target(cfg, target_name, device)
        for job in [j for j in jobs if j["target_backbone"] == target_name]:
            final = read_json(Path(cfg["paths"]["results_dir"]) / "final" / f"{job['job']}.json")
            for held in target_modalities(cfg):
                seed_everything(job["seed"], bool(cfg["runtime"].get("deterministic", False)))
                path = Path(cfg["paths"]["results_dir"]) / "final/checkpoints" / job["job"] / f"{held}.pt"
                saved = torch.load(path, map_location="cpu", weights_only=False)
                checkpoint_identity(saved, job["source_backbone"], target_name, 2, job["target_block"], "mlp", held)
                adapter, tail = build_stitch_modules(cfg, target, job["source_backbone"], target_name, 2, job["target_block"], "mlp")
                adapter.load_state_dict(saved["adapter"], strict=True)
                adapter, tail = adapter.to(device).eval(), tail.to(device).eval()
                membership = reconstructed_training_images(cfg, held, job["seed"], records)
                for ds, images in membership.items():
                    membership_counts.append(dict(job=job["job"], held_out_modality=held, dataset=ds,
                        train_cache_size=len(records[ds]), reconstructed_training_count=len(images),
                        remaining_normal_count=len(records[ds]) - len(images)))
                output = out / "jobs" / job["job"] / f"source_{held}.csv"
                n_rows += export_fold(cfg, job, held, adapter, tail, device, membership, output)
                # Replay the original authoritative aggregation, independently of per-image diagnostics.
                replay = compute_aoss(cfg, target, job["source_backbone"], target_name, adapter, tail, held, 2, device)
                original = next(r for r in final["aoss_rows"] if r["target_modality"] == held)
                for component in ["normal_discrepancy", "perturb_in", "perturb_out", "local_sensitivity", "aoss"]:
                    delta = replay[component] - original[component]
                    checks.append(dict(job=job["job"], held_out_modality=held, component=component,
                        original=original[component], replayed=replay[component], delta=delta,
                        within_tolerance=math.isfinite(delta) and abs(delta) <= TOLERANCE))
                write_csv(out / "aoss_replay.csv", checks)
                write_csv(out / "training_membership.csv", membership_counts)
                require(all(r["within_tolerance"] for r in checks[-5:]), f"Original AOSS replay differs: {job['job']} / {held}; keep failed evidence")
                del adapter, tail, saved
            print(f"[paired-response] Complete: {job['job']}", flush=True)
        del target
        torch.cuda.empty_cache()
    return n_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    jobs, records, paths, arrays = preflight(cfg, args.config)
    if args.check_only:
        print("Ready: 12 original jobs / 60 folds; all source perturb pairs; no target test/valid loaded, no files written")
        return
    require(torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count(), "Need an available CUDA GPU")
    out = Path(cfg["paths"]["results_dir"]) / "followup/paired_response_audit"
    require(not out.exists() or not any(out.iterdir()), "Existing paired-response output; no overwrite/resume")
    hashes = {str(p): sha256(p) for p in paths}
    stats = {str(p): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in arrays}
    manifest = dict(status="running", protocol="original_locked_final", no_training=True, no_reselection=True,
        target_test_read=False, target_valid_read=False, original_score="contrast_topk", topk_fraction=.05,
        batch_size=BATCH_SIZE, replay_tolerance=TOLERANCE, jobs=jobs, original_sha256=hashes,
        array_weight_stats=stats, versions=versions(), python=platform.python_version(),
        limits="Paired response is diagnostic, not a new selection metric. Per-image weighting differs from original AOSS. Perturbation types and training membership inferred; legacy historical sample provenance unavailable. Large arrays/weights stat-only.")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_manifest.json", manifest)
    configure_torch(bool(cfg["project"].get("allow_tf32", True)))
    try:
        manifest["n_source_rows"] = run(cfg, jobs, records, out, torch.device(f"cuda:{args.gpu}"))
        require(inputs_unchanged(hashes, stats), "Original inputs changed")
        manifest.update(status="complete", original_inputs_unchanged=True,
            output_sha256={p.relative_to(out).as_posix(): sha256(p) for p in out.rglob("*.csv")})
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc), original_inputs_unchanged=inputs_unchanged(hashes, stats))
        raise
    finally:
        write_json(out / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
