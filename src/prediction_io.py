"""Prediction serialization and locked-final replay checks (no model imports)."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(ckpt, source, target, stage, block, adapter, modality):
    expected = dict(source_name=source, target_name=target, source_stage=stage,
                    target_block=block, adapter_type=adapter, target_modality=modality)
    mismatches = {k: (ckpt.get(k), v) for k, v in expected.items() if ckpt.get(k) != v}
    if mismatches:
        raise ValueError(f"Checkpoint identity mismatch: {mismatches}")
    if "adapter" not in ckpt:
        raise ValueError("Checkpoint has no adapter state")


def validate_final(cfg, selected, final, source, target, stage, block, seed, adapter, fraction, modes):
    """Read-only checks; never ranks AOSS or target metrics."""
    name = f"{source}_to_{target}_s{stage}_b{block}"
    items = [x for x in selected["selected"] if x["source_backbone"] == source and x["target_backbone"] == target]
    if selected.get("selection_metric") != "source_only_AOSS" or len(items) != 1 or items[0]["config"] != name:
        raise ValueError("Replay must use the existing locked source_only_AOSS selection")
    if seed not in cfg["final"]["seeds"] or adapter != cfg["final"]["adapter"]:
        raise ValueError("Replay seed/adapter differs from final protocol")
    if list(modes) != ["contrast_topk"] or fraction != cfg["final"]["topk_fraction"]:
        raise ValueError("Replay must retain final contrast_topk and topk_fraction")
    expected = dict(config=name, source_backbone=source, target_backbone=target,
                    source_stage=stage, target_block=block, seed=seed, adapter=adapter)
    if any(final.get(k) != v for k, v in expected.items()):
        raise ValueError("Original final job identity differs from locked replay")
    rows = final.get("rows", [])
    if len(rows) != len(cfg["data"]["datasets"]) or {r["dataset"] for r in rows} != set(cfg["data"]["datasets"]):
        raise ValueError("Original final datasets incomplete/duplicated")
    for row in rows:
        expected_row = {k: v for k, v in expected.items() if k != "config"}
        expected_row.update(score_mode="contrast_topk", epochs=cfg["final"]["epochs"],
                            n_per_source_modality=cfg["final"]["n_per_source_modality"])
        if any(row.get(k) != v for k, v in expected_row.items()):
            raise ValueError("Original final row differs from configured final protocol")
        if row["target_modality"] != cfg["data"]["modality_map"][row["dataset"]]:
            raise ValueError("Original final modality mismatch")


def compare_metrics(actual, expected, tolerance):
    """Report replay differences; tolerance never changes scores or selection."""
    rows = []
    ref = {(r["dataset"], r["score_mode"]): r for r in expected}
    if len(ref) != len(expected) or set(ref) != {(r["dataset"], r["score_mode"]) for r in actual} or len(actual) != len(ref):
        raise ValueError("Replayed metric cells differ from original final")
    for row in actual:
        old = ref[(row["dataset"], row["score_mode"])]
        if row["n_test"] != old["n_test"]:
            raise ValueError("Replayed test count differs from original final")
        for metric in ("image_auroc", "image_aupr"):
            delta = row[metric] - old[metric]
            rows.append(dict(dataset=row["dataset"], score_mode=row["score_mode"], metric=metric,
                             original=old[metric], replayed=row[metric], delta=delta,
                             within_tolerance=math.isfinite(delta) and abs(delta) <= tolerance))
    return rows


def write_predictions(path, records, scores, metadata, stats):
    """One CSV per dataset; export full precision scores in cache record order."""
    if len(records) != len(scores) or len(stats) != len(records):
        raise ValueError("Prediction/record length mismatch")
    if not np.isfinite(scores).all() or any(r["label"] not in (0, 1) for r in records):
        raise ValueError("Invalid labels or non-finite prediction scores")
    if len({r["image"] for r in records}) != len(records):
        raise ValueError("Duplicated test image paths")
    fields = ["record_index", "image_path", "label", "score", *metadata,
              "discrepancy_mean", "discrepancy_median", "discrepancy_max"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (record, score, stat) in enumerate(zip(records, scores, stats)):
            writer.writerow(dict(record_index=index, image_path=record["image"], label=int(record["label"]),
                                 score=float(score), **metadata, **stat))
    tmp.replace(path)


def build_plan(cfg):
    root = Path(cfg["paths"]["results_dir"])
    selected_path = root / "selected_configs.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    expected_pairs = {tuple(x) for x in cfg["models"]["pairs"]}
    items = selected.get("selected", [])
    if len(items) != len(expected_pairs) or {(x["source_backbone"], x["target_backbone"]) for x in items} != expected_pairs:
        raise ValueError("Need exactly one locked configuration per pair")
    jobs = []
    for item in items:
        for seed in cfg["final"]["seeds"]:
            job = f"{item['config']}_seed{seed}"
            reference = root / "final" / f"{job}.json"
            final = json.loads(reference.read_text(encoding="utf-8"))
            validate_final(cfg, selected, final, item["source_backbone"], item["target_backbone"],
                           item["source_stage"], item["target_block"], seed, cfg["final"]["adapter"],
                           cfg["final"]["topk_fraction"], ["contrast_topk"])
            checkpoints = root / "final" / "checkpoints" / job
            for modality in set(cfg["data"]["modality_map"].values()):
                if not (checkpoints / f"{modality}.pt").is_file():
                    raise FileNotFoundError(f"Missing saved final checkpoint: {checkpoints / (modality + '.pt')}; no training fallback")
            jobs.append(dict(item, seed=seed, job=job, reference=str(reference), checkpoint_root=str(checkpoints)))
    for ds in cfg["data"]["datasets"]:
        d = Path(cfg["paths"]["cache_dir"]) / "features" / ds / "test"
        if not (d / ".done").is_file():
            raise FileNotFoundError(f"Test cache not complete: {d}")
        records = [json.loads(line) for line in (d / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records or any(r["dataset"] != ds or r["split"] != "test" or r["modality"] != cfg["data"]["modality_map"][ds] or r["label"] not in (0, 1) for r in records):
            raise ValueError(f"Invalid test cache records: {ds}")
        if len({r["image"] for r in records}) != len(records):
            raise ValueError(f"Duplicate test image paths: {ds}")
        for job in jobs:
            original = json.loads(Path(job["reference"]).read_text(encoding="utf-8"))
            row = next(r for r in original["rows"] if r["dataset"] == ds)
            if row["n_test"] != len(records):
                raise ValueError(f"Test cache count differs from original final: {ds}")
        for item in items:
            for path in [d / f"source_{item['source_backbone']}_s{item['source_stage']}.npy", d / f"target_{item['target_backbone']}.npy"]:
                array = np.load(path, mmap_mode="r")
                if len(array) != len(records):
                    raise ValueError(f"Cache rows do not align: {path}")
        for target in cfg["models"]["targets"].values():
            model_dir = Path(target["local"])
            if not (model_dir / "config.json").is_file() or (not any(model_dir.glob("*.safetensors")) and not (model_dir / "pytorch_model.bin").is_file()):
                raise FileNotFoundError(f"Local target weights/config missing: {model_dir}; no download fallback")
    return jobs
