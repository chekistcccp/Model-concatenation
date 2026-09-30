"""Isolated follow-up entry points; never calls pipeline selection or preparation.

python -m src.run_followup --stage diagnostics --check-only
GPUS=0,1,2,3 bash run.sh diagnostics
GPUS=0,1,2,3 bash run.sh common_stitch
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

from .followup_io import (CASE_SEED, FOCUS, POINTS, preflight, read_json,
                          validate_common_result)
from .prediction_io import sha256


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def protected_inputs(cfg, config_path, cache_paths, stage, cases):
    root = Path(cfg["paths"]["results_dir"])
    paths = [Path(config_path), root / "selected_configs.json"]
    for group in ["screen", "final", "stitchmap", "ablation"]:
        paths += list((root / group).glob("*.json"))
    paths += list((root / "final/checkpoints").rglob("*.pt"))
    paths += [p for p in cache_paths if p.suffix != ".npy"]
    paths += [Path(__file__).parent / f"{m}.py" for m in ["worker", "train_eval", "models", "cache_features", "pipeline", "data", "common"]]
    cache = Path(cfg["paths"]["cache_dir"])
    paths += [cache / n for n in ["manifest.jsonl", "data_audit.json", "model_audit.json"] if (cache / n).is_file()]
    paths += [root / "baselines.json"] if (root / "baselines.json").is_file() else []
    if stage == "diagnostics":
        paths += [p for p in (root / "predictions").rglob("*") if p.is_file()]
        paths += [Path(c[k]) for c in cases for k in ["image_path", "mask_path"] if c.get(k)]
    paths = sorted(set(paths), key=str)
    hashes = {str(p): sha256(p) for p in paths}
    # Feature arrays can be hundreds of GB: check stat stability, not full SHA256.
    arrays = [p for p in cache_paths if p.suffix == ".npy"]
    for spec in cfg["models"]["targets"].values():
        model = Path(spec["local"])
        arrays += list(model.glob("*.safetensors")) + list(model.glob("pytorch_model*.bin"))
        paths_config = model / "config.json"
        hashes[str(paths_config)] = sha256(paths_config)
    stats = {str(p): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sorted(set(arrays), key=str)}
    return hashes, stats


def inputs_unchanged(hashes, stats):
    return all(Path(p).is_file() and sha256(p) == h for p, h in hashes.items()) and all(
        Path(p).is_file() and dict(size=Path(p).stat().st_size, mtime_ns=Path(p).stat().st_mtime_ns) == s
        for p, s in stats.items())


def run_parallel(commands, gpus, logs):
    """Wait for all children, including terminating every active child on failure."""
    logs.mkdir(parents=True, exist_ok=True)
    pending, active = list(commands), {}
    try:
        while pending or active:
            for gpu, (process, stream, job) in list(active.items()):
                status = process.poll()
                if status is not None:
                    stream.close()
                    del active[gpu]
                    if status != 0:
                        raise RuntimeError(f"Follow-up worker failed: {job}; see {logs / (job + '.log')}")
                    print(f"[followup] Complete: {job}", flush=True)
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                job, command = pending.pop(0)
                stream = (logs / f"{job}.log").open("w", encoding="utf-8")
                try:
                    process = subprocess.Popen(command + ["--gpu", str(gpu)], stdout=stream, stderr=subprocess.STDOUT)
                except BaseException:
                    stream.close()
                    raise
                active[gpu] = (process, stream, job)
                print(f"[followup] GPU{gpu}: {job}", flush=True)
            if active:
                time.sleep(.5)
    finally:
        for process, stream, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, stream, _ in active.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()


def common_command(cfg, config_path, job, output):
    f = cfg["final"]
    return [sys.executable, "-m", "src.worker", "--config", str(config_path), "--mode", "stitch",
            "--output", str(output), "--source-name", job["source_backbone"], "--target-name", job["target_backbone"],
            "--source-stage", str(job["source_stage"]), "--target-block", str(job["target_block"]),
            "--adapter", f["adapter"], "--seed", str(job["seed"]), "--n-per-modality", str(f["n_per_source_modality"]),
            "--epochs", str(f["epochs"]), "--batch-size", str(f["batch_size"]), "--lr", str(f["lr"]),
            "--weight-decay", str(f["weight_decay"]), "--topk-fraction", str(f["topk_fraction"]),
            "--score-modes", "contrast_topk", "--save-checkpoints"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--stage", choices=["diagnostics", "common_stitch"], required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    jobs, cases, cache_paths, test_counts = preflight(cfg, args.stage)
    if args.stage == "diagnostics":
        print(f"[followup] Diagnostics: {len(jobs)} locked jobs, {len(cases)} images (random/hard overlap deduplicated); no training")
    else:
        print(f"[followup] Common stitch: {len(jobs)} jobs at {POINTS}; 12 original results reused, 12 supplemental training jobs")
    if args.check_only:
        print("[followup] Required inputs checked; no files written, no training, no selection")
        return
    out = Path(cfg["paths"]["results_dir"]) / "followup" / args.stage
    hashes, stats = protected_inputs(cfg, args.config, cache_paths, args.stage, cases)
    versions = {p: importlib.metadata.version(p) for p in ["torch", "numpy", "transformers", "timm", "scikit-learn", "Pillow"]}
    plan_hash = hashlib.sha256(json.dumps(dict(stage=args.stage, jobs=jobs, cases=cases, input_sha256=hashes,
                                             versions=versions, python=sys.version), sort_keys=True).encode()).hexdigest()
    manifest_path = out / "run_manifest.json"
    if args.stage == "common_stitch" and any(out.glob("*_seed*.json")):
        previous = read_json(manifest_path) if manifest_path.is_file() else {}
        if previous.get("plan_sha256") != plan_hash:
            raise ValueError("Existing common-point outputs belong to different/unrecorded inputs; use a separate archived run")
    commands = []
    if args.stage == "common_stitch":
        for job in jobs:
            output = out / f"{job['job']}.json"
            if job["reuse_original"]:
                validate_common_result(cfg, job, read_json(job["reference"]), test_counts)
                if output.is_file() and sha256(output) != sha256(job["reference"]):
                    raise ValueError("A reused result differs from its original final JSON")
            elif output.is_file():
                validate_common_result(cfg, job, read_json(output), test_counts)
                ckpts = out / "checkpoints" / job["job"]
                if any(not (ckpts / f"{m}.pt").is_file() for m in set(cfg["data"]["modality_map"].values())):
                    raise FileNotFoundError(f"Completed supplemental job lacks checkpoints: {job['job']}")
            else:
                commands.append((job["job"], common_command(cfg, args.config, job, output)))
    else:
        for job in jobs:
            commands.append((job["job"], [sys.executable, "-m", "src.followup_worker", "--config", args.config,
                "--reference-final", job["reference"], "--case-plan", str(out / "case_plan.json"),
                "--output-dir", str(out / "jobs" / job["job"])]))
    if commands:
        import torch
        from .common import gpu_ids
        gpus = gpu_ids(int(cfg["project"].get("max_gpus", 4)))
        if not torch.cuda.is_available() or not gpus or len(set(gpus)) != len(gpus) or any(g < 0 or g >= torch.cuda.device_count() for g in gpus):
            raise RuntimeError("Need valid CUDA GPU IDs; use GPUS=0 or GPUS=0,1,2,3 on the server")
    else:
        gpus = []
    out.mkdir(parents=True, exist_ok=True)
    snapshots = out / "inputs"
    snapshots.mkdir(exist_ok=True)
    for path, destination in [(Path(args.config), snapshots / "experiment.yaml"),
                              (Path(cfg["paths"]["results_dir"]) / "selected_configs.json", snapshots / "selected_configs.json")]:
        shutil.copyfile(path, destination)
    for path in (Path(cfg["paths"]["results_dir"]) / "final").glob("*.json"):
        target_path = snapshots / "final" / path.name
        target_path.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target_path)
    for file in ["manifest.jsonl", "data_audit.json", "model_audit.json"]:
        path = Path(cfg["paths"]["cache_dir"]) / file
        if path.is_file():
            shutil.copyfile(path, snapshots / file)
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = "unavailable"
    receipt = dict(status="running", stage=args.stage, git_head=git_head, posthoc_supplement=True,
                   no_reselection=True, selection_policy="reuse_locked_selection; common controls are its fixed point union",
                   main_score="contrast_topk", topk_fraction=.05, plan_sha256=plan_hash, jobs=jobs,
                   original_sha256=hashes, array_weight_stats=stats,
                   provenance_limits="Large feature arrays/target weights have stat checks only; legacy checkpoints omit training seed provenance")
    receipt["versions"] = versions
    receipt["python"] = sys.version
    if args.stage == "diagnostics":
        receipt.update(case_seed=CASE_SEED, focus_pairs={d: f"{s}_to_{t}" for d, (s, t) in FOCUS.items()},
                       n_cases=len(cases), case_sampling="per dataset/label: 10 random plus 5 hard; overlap deduplicated; posthoc only")
    else:
        receipt.update(points=POINTS, n_original_reused=12, n_supplemental_jobs=12, n_new_jobs_this_run=len(commands),
                       training_budget=cfg["final"], both_points_must_be_reported=True)
    write_json(manifest_path, receipt)
    try:
        if args.stage == "diagnostics":
            write_json(out / "case_plan.json", dict(posthoc_only=True, sample_seed=CASE_SEED, cases=cases))
            export_case_images(cfg, out, cases)
        else:
            for job in jobs:
                if job["reuse_original"]:
                    shutil.copyfile(job["reference"], out / f"{job['job']}.json")
        run_parallel(commands, gpus, out / "logs")
        if args.stage == "diagnostics":
            for job in jobs:
                evaluation = read_json(out / "jobs" / job["job"] / "evaluation.json")
                if not evaluation["replay_matches_original"] or not evaluation["aoss_replay_matches_original"]:
                    raise ValueError(f"Diagnostic replay mismatch: {job['job']}")
                if evaluation["case_plan_sha256"] != sha256(out / "case_plan.json") or evaluation["original_final_sha256"] != sha256(job["reference"]) or evaluation["job"] != job["job"]:
                    raise ValueError(f"Diagnostic receipt refers to different inputs: {job['job']}")
        else:
            for job in jobs:
                validate_common_result(cfg, job, read_json(out / f"{job['job']}.json"), test_counts)
        receipt.update(status="complete")
    except BaseException as error:
        receipt.update(status="failed", error=str(error))
        raise
    finally:
        unchanged = inputs_unchanged(hashes, stats)
        receipt["original_inputs_unchanged"] = unchanged
        if not unchanged:
            receipt.update(status="failed", error="Protected original files or feature/weight stats changed")
        receipt["output_sha256"] = {str(p.relative_to(out)): sha256(p) for p in out.rglob("*") if p.is_file()
            and p != manifest_path and "checkpoints" not in p.relative_to(out).parts and p.suffix != ".tmp"}
        write_json(manifest_path, receipt)
    if not unchanged:
        raise RuntimeError(receipt["error"])
    print(f"[followup] Complete: {out}; original files unchanged; copy this directory back (omit common_stitch/checkpoints)", flush=True)


def export_case_images(cfg, out, cases):
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    rows = []
    for c in cases:
        directory = out / "cases" / c["dataset"]
        directory.mkdir(parents=True, exist_ok=True)
        basename = f"record_{c['record_index']:06}"
        image_file, mask_file = directory / f"{basename}.png", directory / f"{basename}_mask.png"
        with Image.open(c["image_path"]) as image:
            original_size = image.size
            TF.resize(image.convert("RGB"), [224, 224], interpolation=InterpolationMode.BICUBIC, antialias=True).save(image_file)
        exported_mask = None
        if c.get("mask_path"):
            with Image.open(c["mask_path"]) as mask:
                TF.resize(mask.convert("L"), [224, 224], interpolation=InterpolationMode.NEAREST).save(mask_file)
            exported_mask = str(mask_file.relative_to(out))
        rows.append(dict(c, image_file=str(image_file.relative_to(out)), mask_file=exported_mask,
                         original_width=original_size[0], original_height=original_size[1], exported_size=224))
    with (out / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
