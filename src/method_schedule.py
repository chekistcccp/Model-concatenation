"""Schedule independent pair/seed jobs; enforce global source-training barriers."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from .followup_io import read_json, read_records
from .prediction_io import sha256
from .run_followup import inputs_unchanged, run_parallel, write_json


PHASES = dict(compose="parameter_composition", train="source_training",
              evaluate="target_evaluation_after_all_training")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_gpus(value, count):
    if value == "auto":
        gpus = list(range(count))
    else:
        parts = value.split(",")
        require(all(p.strip().isdigit() for p in parts), "GPUS must be auto or comma-separated nonnegative indices")
        gpus = [int(p) for p in parts]
    require(gpus and len(gpus) == len(set(gpus)), "Need a nonempty list of distinct GPUs")
    require(all(i < count for i in gpus), "GPU index outside visible CUDA devices; check CUDA_VISIBLE_DEVICES")
    return gpus


def worker_path(out, phase, job):
    return out / "workers" / phase / f"{job}.json"


def command(config, phase, job):
    return [sys.executable, "-m", "src.method_controls", "--config", str(config),
            "--worker-phase", phase, "--job", job["job"]]


def collect(out, phase, jobs, gpus):
    payloads = []
    for job in jobs:
        r = read_json(worker_path(out, phase, job["job"]))
        require(r.get("status") == "complete" and r.get("phase") == phase
                and r.get("job") == job["job"] and r.get("gpu") in gpus,
                "Incomplete or mismatched GPU worker receipt")
        require(isinstance(r.get("device_name"), str) and r["device_name"]
                and isinstance(r.get("device_total_memory"), int) and r["device_total_memory"] > 0
                and len(r.get("compute_capability", [])) == 2, "Missing worker GPU device evidence")
        payloads.append(r)
    return payloads


def run_phases(cfg, jobs, config, out, gpus, manifest, hashes, stats):
    from .method_controls import validate_training_receipt, write_csv

    summaries = {}
    for phase in PHASES:
        if phase == "evaluate":
            require(inputs_unchanged(hashes, stats), "Original inputs changed before target evaluation")
            validate_training_receipt(cfg, jobs, out)  # Hash all 180 controls before any evaluation starts.
            manifest["training_completed_sha256"] = sha256(out / "training_completed.json")
        manifest["phase"] = PHASES[phase]
        write_json(out / "run_manifest.json", manifest)
        print(f"[method] Phase={phase}; {len(jobs)} independent jobs on visible GPUs {gpus}", flush=True)
        commands = [(j["job"], command(config, phase, j)) for j in jobs]
        run_parallel(commands, gpus, out / "logs" / phase)
        receipts = collect(out, phase, jobs, gpus)
        if phase == "compose":
            rows = [r for p in receipts for r in p["data"]["rows"]]
            require(len(rows) == 288, "Incomplete distributed composition matrix")
            write_csv(out / "composition_checks.csv", rows)
            summaries["compose"] = rows
        elif phase == "train":
            training = [r for p in receipts for r in p["data"]["rows"]]
            samples = [r for p in receipts for r in p["data"]["samples"]]
            require(len(training) == 180 and len(samples) == 60, "Incomplete distributed source-training matrix")
            write_json(out / "training_samples.json", dict(rows=samples))
            # Coordinator alone creates this gate after ALL workers succeed.
            write_json(out / "training_completed.json", dict(no_target_data_used=True, no_reselection=True, rows=training))
        else:
            data = {key: [r for p in receipts for r in p["data"][key]] for key in ["rows", "aoss_rows", "replays"]}
            require(len(data["rows"]) == 288 and len(data["aoss_rows"]) == 240 and len(data["replays"]) == 144,
                    "Incomplete distributed evaluation matrix")
            write_json(out / "metrics.json", data)
            summaries["evaluate"] = data["rows"]
    return summaries["compose"], summaries["evaluate"]


def run_worker(cfg, config, phase, job_name, gpu):
    import torch
    from .common import configure_torch
    from .method_controls import compose_checks, evaluate_phase, train_phase
    from .paired_response_audit import versions

    out = Path(cfg["paths"]["results_dir"]) / "followup/method_controls"
    manifest = read_json(out / "run_manifest.json")
    require(manifest["status"] == "running" and manifest["phase"] == PHASES[phase]
            and gpu in manifest["gpus"], "Worker not authorized by current coordinator phase/GPU plan")
    require(manifest["versions"] == versions() and cfg == read_json(out / "references/config.json"),
            "Worker environment/config differs from coordinator")
    selected = Path(cfg["paths"]["results_dir"]) / "selected_configs.json"
    for p in [Path(config), selected, Path(__file__), Path(__file__).with_name("method_controls.py")]:
        require(sha256(p) == manifest["original_sha256"][str(p)], f"Worker protocol/code input changed: {p}")
    matches = [j for j in manifest["jobs"] if j["job"] == job_name]
    require(len(matches) == 1, "Worker job outside locked pair/seed matrix")
    job = matches[0]
    path = worker_path(out, phase, job_name)
    require(not path.exists(), "Existing worker receipt; no overwrite/resume")
    torch.cuda.set_device(gpu)
    device = torch.device(f"cuda:{gpu}")
    configure_torch(cfg["project"].get("allow_tf32", True))
    props = torch.cuda.get_device_properties(gpu)
    receipt = dict(status="running", phase=phase, job=job_name, gpu=gpu,
                   device_name=props.name, device_total_memory=props.total_memory,
                   cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   compute_capability=list(torch.cuda.get_device_capability(gpu)))
    write_json(path, receipt)
    try:
        if phase == "compose":
            records = {ds: read_records(Path(cfg["paths"]["cache_dir"]) / "features" / ds / "train/records.jsonl")
                       for ds in cfg["data"]["datasets"]}
            data = dict(rows=compose_checks(cfg, [job], records, out, device, write_summary=False))
        elif phase == "train":
            data = train_phase(cfg, [job], out, device, write_summary=False)
        else:
            require(sha256(out / "training_completed.json") == manifest["training_completed_sha256"],
                    "Global training receipt changed; evaluation blocked")
            data = evaluate_phase(cfg, manifest["jobs"], out, device, evaluation_jobs=[job], write_summary=False)
        receipt.update(status="complete", data=data)
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(path, receipt)
        raise
    write_json(path, receipt)
