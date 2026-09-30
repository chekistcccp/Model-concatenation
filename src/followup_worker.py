"""Replay locked final checkpoints for spatial/source diagnostics only."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import configure_torch, seed_everything, write_json
from .followup_io import FOCUS, protocol_guard, read_json, read_records
from .models import build_stitch_modules, load_target, target_grid
from .prediction_io import checkpoint_identity, compare_metrics, sha256, validate_final
from .train_eval import (_load_mask, _safe_auc, _score_from_map, compute_aoss,
                         discrepancy, modality_datasets, target_modalities)


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


@torch.inference_mode()
def spatial_maps(cfg, target, adapter, tail, source, target_name, stage, dataset, cases, device, out, prediction_reference=None):
    directory = Path(cfg["paths"]["cache_dir"]) / "features" / dataset / "test"
    records = read_records(directory / "records.jsonl")
    x = np.load(directory / f"source_{source}_s{stage}.npy", mmap_mode="r")
    y = np.load(directory / f"target_{target_name}.npy", mmap_mode="r")
    grid = target_grid(target, cfg["models"]["targets"][target_name]["input_size"])
    wanted = {c["record_index"] for c in cases if c["dataset"] == dataset}
    maps, components, indices = [], [], []
    all_scores = []
    batch_size = max(cfg["final"]["batch_size"], 128)
    for start in range(0, len(records), batch_size):
        stop = min(len(records), start + batch_size)
        d = discrepancy(adapter, tail, torch.from_numpy(np.array(x[start:stop], copy=True)),
                        torch.from_numpy(np.array(y[start:stop], copy=True)), device)
        if not torch.isfinite(d).all():
            raise ValueError("Non-finite discrepancy map")
        score = _score_from_map(d, cfg["final"]["topk_fraction"], "contrast_topk")
        all_scores.extend(score.cpu().numpy().tolist())
        for offset, i in enumerate(range(start, stop)):
            if i not in wanted:
                continue
            maps.append(d[offset].cpu().numpy())
            indices.append(i)
            components.append(dict(record_index=i, image_path=records[i]["image"], label=records[i]["label"],
                score=float(score[offset]), discrepancy_mean=float(d[offset].mean()),
                discrepancy_median=float(d[offset].median()), discrepancy_max=float(d[offset].max())))
    if set(indices) != wanted:
        raise ValueError("Diagnostic case indices are not in the test cache")
    raw = torch.from_numpy(np.stack(maps))
    medians = raw.median(1, keepdim=True).values
    contrast = (raw - medians).clamp_min(0)
    topk = torch.zeros_like(contrast, dtype=torch.uint8)
    k = max(1, round(raw.shape[1] * cfg["final"]["topk_fraction"]))
    topk.scatter_(1, contrast.topk(k, dim=1).indices, 1)
    masks, has_masks = [], []
    for i in indices:
        mask = _load_mask(records[i].get("mask"), grid)
        has_masks.append(mask is not None)
        masks.append(mask if mask is not None else np.zeros((grid, grid), dtype=np.uint8))
    atomic_npz(out / f"{dataset}_maps.npz", record_indices=np.array(indices), labels=np.array([records[i]["label"] for i in indices]),
               discrepancy=raw.numpy().reshape(-1, grid, grid), contrast=contrast.numpy().reshape(-1, grid, grid),
               topk_mask=topk.numpy().reshape(-1, grid, grid), scores=np.array([r["score"] for r in components]),
               patch_median=medians.numpy().ravel(), ground_truth_mask=np.stack(masks), has_ground_truth_mask=np.array(has_masks))
    with (out / f"{dataset}_components.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(components[0]))
        writer.writeheader()
        writer.writerows(components)
    auroc, aupr = _safe_auc([r["label"] for r in records], all_scores)
    result = dict(dataset=dataset, score_mode="contrast_topk", n_test=len(records), image_auroc=auroc, image_aupr=aupr)
    if prediction_reference is not None:
        with Path(prediction_reference).open(newline="", encoding="utf-8") as stream:
            original_scores = list(csv.DictReader(stream))
        if len(original_scores) != len(records) or any(r["image_path"] != records[i]["image"] or int(r["label"]) != records[i]["label"] for i, r in enumerate(original_scores)):
            raise ValueError("Original prediction order/labels differ from spatial replay")
        delta = np.abs(np.array(all_scores) - np.array([float(r["score"]) for r in original_scores]))
        if not np.isfinite(delta).all() or delta.max() > 1e-6:
            raise RuntimeError("Spatial replay scores differ from original predictions by more than 1e-6")
        result["max_score_abs_delta"] = float(delta.max())
    return result


@torch.inference_mode()
def source_components(cfg, adapter, tail, source, target_name, stage, held_out, device, path):
    rows = []
    groups = modality_datasets(cfg)
    for modality, datasets in groups.items():
        if modality == held_out:
            continue
        for dataset in datasets:
            directory = Path(cfg["paths"]["cache_dir"]) / "perturb" / dataset
            records = read_records(directory / "records.jsonl")
            nx = np.load(directory / f"normal_source_{source}_s{stage}.npy", mmap_mode="r")
            ny = np.load(directory / f"normal_target_{target_name}.npy", mmap_mode="r")
            px = np.load(directory / f"pert_source_{source}_s{stage}.npy", mmap_mode="r")
            py = np.load(directory / f"pert_target_{target_name}.npy", mmap_mode="r")
            mask_array = np.load(directory / f"mask_{target_name}.npy", mmap_mode="r")
            for start in range(0, len(records), 256):
                stop = min(len(records), start + 256)
                dn = discrepancy(adapter, tail, torch.from_numpy(np.array(nx[start:stop], copy=True)),
                                 torch.from_numpy(np.array(ny[start:stop], copy=True)), device)
                dp = discrepancy(adapter, tail, torch.from_numpy(np.array(px[start:stop], copy=True)),
                                 torch.from_numpy(np.array(py[start:stop], copy=True)), device)
                if not torch.isfinite(dn).all() or not torch.isfinite(dp).all():
                    raise ValueError("Non-finite source perturbation discrepancy")
                masks = torch.from_numpy(np.array(mask_array[start:stop], copy=True)).to(device).reshape(dp.shape).bool()
                normal_score = _score_from_map(dn, .05, "contrast_topk")
                pert_score = _score_from_map(dp, .05, "contrast_topk")
                # Per-image CSV statistics are diagnostics. Transfer a whole batch
                # once; the authoritative original AOSS is replayed separately below.
                dn, dp, masks = dn.cpu(), dp.cpu(), masks.cpu()
                normal_score, pert_score = normal_score.cpu(), pert_score.cpu()
                for offset, i in enumerate(range(start, stop)):
                    mask = masks[offset]
                    n_in, n_out = int(mask.sum()), int((~mask).sum())
                    inside = float(dp[offset][mask].mean()) if n_in else None
                    outside = float(dp[offset][~mask].mean()) if n_out else None
                    local = inside - outside if inside is not None and outside is not None else None
                    # Legacy cache records do not store operation IDs. This annotation
                    # is inferred from the unchanged cache generator, not provenance.
                    kind = ["intensity", "blur", "patch_copy"][(cfg["project"]["seed"] + i) % 3]
                    rows.append(dict(held_out_modality=held_out, source_modality=modality, dataset=dataset,
                        record_index=i, image_path=records[i]["image"], perturbation_kind_inferred=kind,
                        kind_evidence="inferred_from_current_generator_and_record_index", n_mask_in=n_in, n_mask_out=n_out,
                        normal_mean=float(dn[offset].mean()), normal_median=float(dn[offset].median()),
                        normal_contrast_topk=float(normal_score[offset]), pert_mean=float(dp[offset].mean()),
                        pert_median=float(dp[offset].median()), pert_contrast_topk=float(pert_score[offset]),
                        perturb_in=inside, perturb_out=outside, local_sensitivity=local,
                        mean_delta=float(dp[offset].mean() - dn[offset].mean()),
                        contrast_delta=float(pert_score[offset] - normal_score[offset])))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if any(r["source_modality"] == held_out for r in rows):
        raise RuntimeError("Held-out modality entered source diagnostics")


def run_job(cfg, reference_path, case_plan, out, device):
    protocol_guard(cfg)
    reference = read_json(reference_path)
    selected = read_json(Path(cfg["paths"]["results_dir"]) / "selected_configs.json")
    source, target_name = reference["source_backbone"], reference["target_backbone"]
    stage, block, seed = reference["source_stage"], reference["target_block"], reference["seed"]
    validate_final(cfg, selected, reference, source, target_name, stage, block, seed,
                   cfg["final"]["adapter"], cfg["final"]["topk_fraction"], ["contrast_topk"])
    job = f"{reference['config']}_seed{seed}"
    expected_reference = Path(cfg["paths"]["results_dir"]) / "final" / f"{job}.json"
    expected_out = Path(cfg["paths"]["results_dir"]) / "followup/diagnostics/jobs" / job
    expected_cases = expected_out.parent.parent / "case_plan.json"
    if Path(reference_path).resolve() != expected_reference.resolve() or out.resolve() != expected_out.resolve() or Path(case_plan).resolve() != expected_cases.resolve():
        raise ValueError("Diagnostics must read original final and write only their isolated follow-up directory")
    out.mkdir(parents=True, exist_ok=True)
    cases = read_json(case_plan)["cases"]
    configure_torch(bool(cfg["project"].get("allow_tf32", True)))
    seed_everything(seed, bool(cfg["runtime"].get("deterministic", False)))
    target = load_target(cfg, target_name, device)
    groups = modality_datasets(cfg)
    metrics, aoss_checks, provenance = [], [], []
    for modality in target_modalities(cfg):
        checkpoint = Path(cfg["paths"]["results_dir"]) / "final/checkpoints" / job / f"{modality}.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_identity(saved, source, target_name, stage, block, cfg["final"]["adapter"], modality)
        adapter, tail = build_stitch_modules(cfg, target, source, target_name, stage, block, cfg["final"]["adapter"])
        adapter.load_state_dict(saved["adapter"], strict=True)
        adapter, tail = adapter.to(device).eval(), tail.to(device).eval()
        source_components(cfg, adapter, tail, source, target_name, stage, modality, device, out / f"source_{modality}.csv")
        # Authoritative AOSS is the unchanged function, with its original defaults.
        replay = compute_aoss(cfg, target, source, target_name, adapter, tail, modality, stage, device)
        original = next(r for r in reference["aoss_rows"] if r["target_modality"] == modality)
        for key in ["normal_discrepancy", "perturb_in", "perturb_out", "local_sensitivity", "aoss"]:
            delta = replay[key] - original[key]
            aoss_checks.append(dict(target_modality=modality, component=key, original=original[key],
                                    replayed=replay[key], delta=delta, within_tolerance=np.isfinite(delta).item() and abs(delta) <= 1e-6))
        for dataset in groups[modality]:
            if dataset in FOCUS:
                prediction = Path(cfg["paths"]["results_dir"]) / "predictions" / job / f"{dataset}.csv"
                metrics.append(spatial_maps(cfg, target, adapter, tail, source, target_name, stage, dataset, cases, device, out, prediction_reference=prediction))
        provenance.append(dict(target_modality=modality, checkpoint_sha256=sha256(checkpoint)))
        del adapter, tail
        torch.cuda.empty_cache()
    comparisons = compare_metrics(metrics, [r for r in reference["rows"] if r["dataset"] in FOCUS], 1e-6)
    payload = dict(job=job, config=reference["config"], source_backbone=source, target_backbone=target_name, seed=seed,
        posthoc_only=True, no_training=True, no_reselection=True, score_mode="contrast_topk", topk_fraction=.05,
        rows=metrics, comparison=comparisons, aoss_comparison=aoss_checks, checkpoint_provenance=provenance,
        replay_matches_original=all(c["within_tolerance"] for c in comparisons),
        aoss_replay_matches_original=all(c["within_tolerance"] for c in aoss_checks),
        original_final_sha256=sha256(reference_path), selected_configs_sha256=sha256(Path(cfg["paths"]["results_dir"]) / "selected_configs.json"),
        case_plan_sha256=sha256(case_plan),
        versions={p: importlib.metadata.version(p) for p in ["torch", "numpy", "transformers", "timm", "scikit-learn"]},
        limits="Image sampling is posthoc; operation IDs inferred, not stored in old caches; legacy checkpoints omit seed")
    write_json(out / "evaluation.json", payload)
    if not payload["replay_matches_original"] or not payload["aoss_replay_matches_original"]:
        raise RuntimeError("Diagnostic replay differs from original; preserve output for investigation, do not relax tolerance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-final", required=True)
    parser.add_argument("--case-plan", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("Run diagnostics on the original CUDA server")
    run_job(cfg, args.reference_final, args.case_plan, args.output_dir, torch.device(f"cuda:{args.gpu}"))


if __name__ == "__main__":
    main()
