"""Plans and guards for supplemental experiments; never ranks stitches."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .prediction_io import build_plan, sha256


FOCUS = {"Brain": ("medical", "medical"), "RESC": ("general", "medical"),
         "OCT2017": ("general", "medical")}
CASE_SEED = 20260930
POINTS = [(2, 3), (2, 9)]
DATASETS = ["Brain", "liver", "RESC", "OCT2017", "RSNA", "camelyon16"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_records(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def name(source, target, stage, block):
    return f"{source}_to_{target}_s{stage}_b{block}"


def protocol_guard(cfg):
    """Refuse silent changes to the existing main experiment."""
    expected = dict(seeds=[11, 22, 33], adapter="mlp", epochs=8,
                    n_per_source_modality=1000, topk_fraction=.05,
                    batch_size=64, lr=.001, weight_decay=.0001, top_configs_per_pair=1)
    if any(cfg["final"].get(k) != v for k, v in expected.items()):
        raise ValueError("Follow-up must retain the original final seeds/budget/score parameters")
    if cfg["data"]["datasets"] != DATASETS or set(map(tuple, cfg["models"]["pairs"])) != {
            (s, t) for s in ["medical", "general"] for t in ["medical", "general"]}:
        raise ValueError("Follow-up requires the original six datasets and four pairs")
    if cfg["data"]["modality_map"] != dict(Brain="brain_mri", liver="liver_ct", RESC="oct",
                                          OCT2017="oct", RSNA="chest_xray", camelyon16="pathology"):
        raise ValueError("The held-out modality grouping must not change")
    if cfg["screen"]["source_stages"] != [1, 2, 3] or cfg["screen"]["target_blocks"] != [3, 6, 9]:
        raise ValueError("Candidate stitch grid must not change")
    if cfg["project"]["seed"] != 2026 or cfg["data"]["perturb_samples_per_dataset"] != 256:
        raise ValueError("Keep the original cache sampling/generator specification")
    if cfg["models"]["sources"]["medical"]["kind"] != "radimagenet_resnet50" or cfg["models"]["sources"]["general"]["architecture"] != "resnet50.a1_in1k":
        raise ValueError("Original CNN backbone identities must not change")
    if cfg["models"]["targets"]["medical"]["repo_id"] != "microsoft/rad-dino" or cfg["models"]["targets"]["general"]["repo_id"] != "facebook/dinov2-base":
        raise ValueError("Original Transformer backbone identities must not change")
    for spec in cfg["models"]["targets"].values():
        if spec["input_size"] != 224 or spec["patch_size"] != 14:
            raise ValueError("Follow-up retains 224 input and 256 patches")
    if any(s["input_size"] != 224 for s in cfg["models"]["sources"].values()):
        raise ValueError("Original CNN input geometry must not change")


def common_jobs(cfg, selected):
    protocol_guard(cfg)
    items = selected.get("selected", [])
    pairs = {tuple(x) for x in cfg["models"]["pairs"]}
    if selected.get("selection_metric") != "source_only_AOSS" or len(items) != 4 or {
            (x["source_backbone"], x["target_backbone"]) for x in items} != pairs:
        raise ValueError("Need the original locked source_only_AOSS selection for four pairs")
    points = sorted({(x["source_stage"], x["target_block"]) for x in items})
    if points != POINTS:
        raise ValueError("Common points must be exactly the union of locked points s2b3/s2b9")
    by_pair = {(x["source_backbone"], x["target_backbone"]): x for x in items}
    for item in items:
        if item["config"] != name(item["source_backbone"], item["target_backbone"], item["source_stage"], item["target_block"]):
            raise ValueError("Locked selection identity mismatch")
    root = Path(cfg["paths"]["results_dir"])
    jobs = []
    for stage, block in points:
        for source, target in cfg["models"]["pairs"]:
            config = name(source, target, stage, block)
            for seed in cfg["final"]["seeds"]:
                job = f"{config}_seed{seed}"
                reuse = by_pair[(source, target)]["config"] == config
                jobs.append(dict(job=job, config=config, source_backbone=source, target_backbone=target,
                                 source_stage=stage, target_block=block, seed=seed, reuse_original=reuse,
                                 reference=str(root / "final" / f"{job}.json") if reuse else None))
    return jobs


def validate_common_result(cfg, job, result, test_counts):
    identity = {k: job[k] for k in ["config", "source_backbone", "target_backbone", "source_stage", "target_block", "seed"]}
    identity["adapter"] = cfg["final"]["adapter"]
    if any(result.get(k) != v for k, v in identity.items()):
        raise ValueError(f"Common-point job identity differs: {job['job']}")
    rows = result.get("rows", [])
    if len(rows) != 6 or {r["dataset"] for r in rows} != set(DATASETS):
        raise ValueError("Common-point datasets incomplete/duplicated")
    for row in rows:
        expected = {k: v for k, v in identity.items() if k != "config"}
        expected.update(epochs=cfg["final"]["epochs"], n_per_source_modality=cfg["final"]["n_per_source_modality"], score_mode="contrast_topk")
        if any(row.get(k) != v for k, v in expected.items()) or row["target_modality"] != cfg["data"]["modality_map"][row["dataset"]]:
            raise ValueError("Common-point row differs from the final budget/modality/score")
        if row["n_test"] != test_counts[row["dataset"]] or not all(math.isfinite(row[k]) for k in ["image_auroc", "image_aupr", "aoss", "final_train_loss"]):
            raise ValueError("Common-point result has invalid counts/metrics")
    aoss = result.get("aoss_rows", [])
    if len(aoss) != 5 or {r["target_modality"] for r in aoss} != set(cfg["data"]["modality_map"].values()):
        raise ValueError("Common-point held-out folds incomplete")


def case_indices(labels, scores, seed=CASE_SEED, n_random=10, n_hard=5):
    """Post-hoc image inspection only; never selects a stitch or threshold."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if set(labels) != {0, 1} or scores.ndim != 2 or len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError("Cases require aligned finite scores and binary labels")
    contributions = np.empty_like(scores, dtype=float)
    for j in range(scores.shape[1]):
        for label in [0, 1]:
            opposite = np.sort(scores[labels != label, j])
            own = scores[labels == label, j]
            f = (np.searchsorted(opposite, own, "left") + np.searchsorted(opposite, own, "right")) / (2 * len(opposite))
            contributions[labels == label, j] = f if label == 1 else 1 - f
    difficulty = contributions.mean(1)
    rng = np.random.default_rng(seed)
    reasons = {}
    for label in [0, 1]:
        ids = np.flatnonzero(labels == label)
        for reason, take in [("random", rng.choice(ids, min(n_random, len(ids)), replace=False)),
                             ("hard", ids[np.lexsort((ids, difficulty[ids]))[:n_hard]])]:
            for i in take:
                reasons.setdefault(int(i), []).append(reason)
    return [dict(record_index=i, label=int(labels[i]), reason="+".join(reasons[i]),
                 mean_pair_order_contribution=float(difficulty[i])) for i in sorted(reasons)]


def diagnostic_cases(cfg, locked):
    pred = Path(cfg["paths"]["results_dir"]) / "predictions"
    export = read_json(pred / "export_manifest.json")
    if export.get("status") != "complete" or not export.get("original_inputs_unchanged"):
        raise ValueError("Need complete, validated original per-image predictions")
    cases = []
    for ds, (source, target) in FOCUS.items():
        jobs = sorted([j for j in locked if j["source_backbone"] == source and j["target_backbone"] == target], key=lambda j: j["seed"])
        ordered, values = None, []
        for job in jobs:
            path = pred / job["job"] / f"{ds}.csv"
            receipt = read_json(path.parent / "evaluation.json")
            proof = next(r for r in receipt["provenance"] if r["dataset"] == ds)
            if not receipt["replay_matches_original"] or sha256(path) != proof["prediction_sha256"]:
                raise ValueError(f"Prediction receipt/hash invalid: {path}")
            if any(receipt.get(k) != job[k] for k in ["config", "seed", "source_backbone", "target_backbone"]):
                raise ValueError("Prediction receipt job identity differs")
            if receipt["original_final_sha256"] != sha256(job["reference"]) or receipt["selected_configs_sha256"] != sha256(Path(cfg["paths"]["results_dir"]) / "selected_configs.json"):
                raise ValueError("Predictions do not refer to the current locked final/selection")
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            records = read_records(Path(cfg["paths"]["cache_dir"]) / "features" / ds / "test/records.jsonl")
            if len(rows) != len(records) or any(int(r["record_index"]) != i or r["image_path"] != records[i]["image"] or int(r["label"]) != records[i]["label"] for i, r in enumerate(rows)):
                raise ValueError("Prediction/cache ordering or labels differ")
            if any(r["dataset"] != ds or r["config"] != job["config"] or int(r["seed"]) != job["seed"] or
                   r["score_mode"] != "contrast_topk" or float(r["topk_fraction"]) != .05 for r in rows):
                raise ValueError("Diagnostic prediction rows differ from the locked protocol")
            ids = [(r["image_path"], int(r["label"])) for r in rows]
            if ordered is not None and ordered != ids:
                raise ValueError("Diagnostic seed score order differs")
            ordered = ids
            values.append([float(r["score"]) for r in rows])
        chosen = case_indices([r[1] for r in ordered], np.array(values).T)
        for c in chosen:
            record = records[c["record_index"]]
            image, mask = Path(record["image"]), record.get("mask")
            if not image.is_file() or (mask and not Path(mask).is_file()):
                raise FileNotFoundError(f"Need original image/mask for spatial inspection: {image}")
            cases.append(dict(c, dataset=ds, focus_pair=f"{source}_to_{target}", image_path=str(image), mask_path=mask))
    return cases


def check_cache(cfg, split, keys):
    paths = []
    counts = {}
    for ds in DATASETS:
        directory = Path(cfg["paths"]["cache_dir"]) / ("perturb" if split == "perturb" else "features") / ds
        if split != "perturb":
            directory /= split
        marker, records_path = directory / ".done", directory / "records.jsonl"
        if not marker.is_file():
            raise FileNotFoundError(f"Missing completed cache: {directory}; no cache regeneration")
        records = read_records(records_path)
        if not records or any(r["dataset"] != ds or r["modality"] != cfg["data"]["modality_map"][ds] or
                              r["split"] != ("train" if split == "perturb" else split) or
                              r["label"] not in ({0} if split in {"train", "perturb"} else {0, 1}) for r in records):
            raise ValueError(f"Invalid source/test cache records: {directory}")
        if len({r["image"] for r in records}) != len(records):
            raise ValueError(f"Duplicate cache image paths: {directory}")
        if split == "test" and {r["label"] for r in records} != {0, 1}:
            raise ValueError(f"Need both normal and abnormal test images: {directory}")
        counts[ds] = len(records)
        paths += [records_path, marker]
        for key in keys:
            path = directory / f"{key}.npy"
            if len(np.load(path, mmap_mode="r")) != len(records):
                raise ValueError(f"Array/records length differs: {path}")
            paths.append(path)
        if split == "test":
            for r in records:
                if r.get("mask") and not Path(r["mask"]).is_file():
                    raise FileNotFoundError(f"Test mask missing: {r['mask']}")
                if r.get("mask"):
                    paths.append(Path(r["mask"]))
    return paths, counts


def preflight(cfg, stage):
    protocol_guard(cfg)
    locked = build_plan(cfg)
    if len(locked) != 12:
        raise ValueError("Need all 12 locked final jobs")
    source_keys = [f"source_{s}_s2" for s in ["medical", "general"]]
    target_keys = [f"target_{t}" for t in ["medical", "general"]]
    paths, counts = check_cache(cfg, "test", source_keys + target_keys)
    if stage == "diagnostics":
        # Check the locked-point union here too; no new points enter diagnostics.
        common_jobs(cfg, read_json(Path(cfg["paths"]["results_dir"]) / "selected_configs.json"))
        keys = [f"{p}_{key}" for p in ["normal", "pert"] for key in source_keys + target_keys]
        keys += [f"mask_{t}" for t in ["medical", "general"]]
        extra, _ = check_cache(cfg, "perturb", keys)
        paths += extra
        cases = diagnostic_cases(cfg, locked)
        return locked, cases, paths, counts
    if stage != "common_stitch":
        raise ValueError("Unknown follow-up stage")
    extra, _ = check_cache(cfg, "train", source_keys + target_keys)
    paths += extra
    keys = [f"{p}_{key}" for p in ["normal", "pert"] for key in source_keys + target_keys]
    keys += [f"mask_{t}" for t in ["medical", "general"]]
    extra, _ = check_cache(cfg, "perturb", keys)
    paths += extra
    return common_jobs(cfg, read_json(Path(cfg["paths"]["results_dir"]) / "selected_configs.json")), [], paths, counts
