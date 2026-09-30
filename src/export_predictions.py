"""Replay the four locked final pairs and three seeds; no training or selection.

python -m src.export_predictions --config configs/experiment.yaml --check-only
GPUS=0,1,2,3 python -m src.export_predictions --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from .prediction_io import build_plan, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    jobs = build_plan(cfg)
    print(f"[predictions] Locked final replay: {len(jobs)} jobs, contrast_topk, topk={cfg['final']['topk_fraction']}")
    if args.check_only:
        print("[predictions] Required final JSONs, checkpoints and local test cache present; no files written")
        return
    # Import runtime only after the read-only preflight; never call pipeline.main.
    from .common import gpu_ids, write_json
    from .pipeline import _run_parallel

    results = Path(cfg["paths"]["results_dir"])
    selected = results / "selected_configs.json"
    original_paths = [Path(args.config), selected] + [Path(j["reference"]) for j in jobs]
    original_hashes = {str(p): sha256(p) for p in original_paths}
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = "unavailable"
    commands = []
    for job in jobs:
        directory = results / "predictions" / job["job"]
        # Rerun evaluation even if files exist; never assume stale outputs are valid.
        commands.append([sys.executable, "-m", "src.worker", "--config", args.config,
                         "--mode", "eval_saved", "--output", str(directory / "evaluation.json"),
                         "--prediction-dir", str(directory), "--reference-final", job["reference"],
                         "--checkpoint-root", job["checkpoint_root"], "--source-name", job["source_backbone"],
                         "--target-name", job["target_backbone"], "--source-stage", str(job["source_stage"]),
                         "--target-block", str(job["target_block"]), "--adapter", cfg["final"]["adapter"],
                         "--seed", str(job["seed"]), "--batch-size", str(cfg["final"]["batch_size"]),
                         "--topk-fraction", str(cfg["final"]["topk_fraction"]), "--score-modes", "contrast_topk"])
    receipt = results / "predictions" / "export_manifest.json"
    payload = dict(status="running", git_head=git_head, jobs=jobs, original_sha256=original_hashes,
                   posthoc_only=True, selection_policy="reuse_locked_selection_without_ranking",
                   provenance_limits="Legacy checkpoints omit seed and model/cache hashes; newly recorded hashes describe replay inputs")
    write_json(receipt, payload)
    try:
        _run_parallel(commands, gpu_ids(int(cfg["project"].get("max_gpus", 4))), "predictions")
    except BaseException as exc:
        payload.update(status="failed", error=str(exc))
        raise
    else:
        payload.update(status="complete")
    finally:
        unchanged = original_hashes == {str(p): sha256(p) for p in original_paths}
        payload["original_inputs_unchanged"] = unchanged
        if not unchanged:
            payload.update(status="failed", error="Original config/selection/final JSON changed during replay")
        write_json(receipt, payload)
    if not unchanged:
        raise RuntimeError(payload["error"])
    print(f"[predictions] Complete; original inputs unchanged; return {results / 'predictions'}")


if __name__ == "__main__":
    main()
