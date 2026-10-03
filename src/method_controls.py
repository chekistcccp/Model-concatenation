"""Method experiment: parameter composition and matched architecture controls.

All new adapters are fixed before any new target evaluation. Original selection,
checkpoints, caches, scores and results are read-only. No synthetic anomalies are
used for training. AOSS is reported, never used to choose a new model here.
"""
from __future__ import annotations

import argparse
import csv
import math
import platform
import signal
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .cache_features import _normalize, load_rgb_raw
from .common import configure_torch, seed_everything
from .composed_detector import (ComposedAnomalyDetector, DirectFeatureTail,
                                load_bundle, save_bundle, source_prefix, tensor_digest)
from .followup_io import check_cache, read_json, read_records
from .models import build_stitch_modules, extract_source_features, extract_target_features, load_source, load_target
from .paired_response_audit import preflight as source_preflight, versions
from .prediction_io import build_plan, checkpoint_identity, compare_metrics, sha256
from .run_followup import inputs_unchanged, write_json
from .train_eval import (_balanced_indices, compute_aoss, discrepancy, evaluate_dataset,
                         load_source_arrays, modality_datasets, target_modalities)


ARMS = ["original", "matched_tail", "no_tail", "untrained_adapter"]
NEW_ARMS = ARMS[1:]
TOLERANCE = 1e-6


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_csv(path, rows):
    require(bool(rows), f"No rows: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def training_records(cfg, job, held):
    """Exact ordered sampling used by load_source_arrays, including OCT truncation."""
    rows = []
    budget = cfg["final"]["n_per_source_modality"]
    for mod, datasets in modality_datasets(cfg).items():
        if mod == held:
            continue
        take, remaining = math.ceil(budget / len(datasets)), budget
        for di, ds in enumerate(datasets):
            path = Path(cfg["paths"]["cache_dir"]) / "features" / ds / "train/records.jsonl"
            records = read_records(path)
            ids = _balanced_indices(len(records), min(take, len(records)), job["seed"] + 101 * di + len(mod))
            for index in ids[:remaining]:
                r = records[int(index)]
                require(r["label"] == 0 and r["split"] == "train" and r["modality"] != held,
                        "Training contains held-out modality or non-normal/non-train image")
                rows.append(dict(dataset=ds, source_modality=mod, record_index=int(index), image=r["image"]))
            remaining = max(0, remaining - len(ids))
    return rows


def preflight(cfg, config):
    _, records, protected, arrays = source_preflight(cfg, config)
    root = Path(cfg["paths"]["results_dir"])
    p = root / "followup/paired_response_audit/run_manifest.json"
    prior = read_json(p)
    require(prior["status"] == "complete" and prior["original_inputs_unchanged"]
            and prior["no_training"] and prior["no_reselection"], "Need completed paired-response evidence")
    require(prior["versions"] == versions() and prior["python"] == platform.python_version(),
            "Restore the successful paired-response environment before this experiment")
    protected.append(p)
    for name, digest in prior["output_sha256"].items():
        path = p.parent / name
        require(sha256(path) == digest, f"Paired-response artifact changed: {path}")
        protected.append(path)
    jobs = build_plan(cfg)
    require(len(jobs) == 12, "Need all four locked pairs and three original seeds")
    keys = [f"source_{s}_s2" for s in ["medical", "general"]] + [f"target_{s}" for s in ["medical", "general"]]
    test, _ = check_cache(cfg, "test", keys)
    protected += [p for p in test if p.suffix != ".npy"]
    arrays += [p for p in test if p.suffix == ".npy"]
    for spec in cfg["models"]["sources"].values():
        files = [Path(spec["checkpoint"])] if spec["manual"] else list(Path(spec["local"]).glob("*.safetensors"))
        require(bool(files) and all(p.is_file() for p in files), "Need existing source weights; no download/rebuild")
        protected += files  # Hash actual inherited CNN weights too.
    for mod, datasets in modality_datasets(cfg).items():
        for ds in datasets:
            require(len(records[ds]) >= math.ceil(1000 / len(datasets)), "Insufficient original normal training cache")
            for r in records[ds][:2]:
                require(Path(r["image"]).is_file(), f"Need source image for composed-model check: {r['image']}")
                protected.append(Path(r["image"]))
    protected += [Path(__file__), Path(__file__).with_name("composed_detector.py"),
                  Path(__file__).with_name("method_schedule.py")]
    hashes = {str(p): sha256(p) for p in sorted(set(protected), key=str)}
    stats = {str(p): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
             for p in sorted(set(arrays), key=str)}
    return jobs, records, hashes, stats


def initialize(cfg, target, job, held, arm, device):
    # Reset for each arm/fold: identical normal samples, initial adapter and shuffle.
    # This differs explicitly from the historical worker's continuously advanced RNG.
    seed_everything(job["seed"], bool(cfg["runtime"].get("deterministic", False)))
    adapter, tail = build_stitch_modules(cfg, target, job["source_backbone"], job["target_backbone"],
                                      job["source_stage"], job["target_block"], "mlp")
    if arm == "no_tail":
        tail = DirectFeatureTail(target)
    adapter, tail = adapter.to(device), tail.to(device).eval()
    for p in tail.parameters():
        p.requires_grad_(False)
    return adapter, tail


def fit(cfg, adapter, tail, x_np, y_np, seed, device):
    budget = cfg["final"]
    ds = TensorDataset(torch.from_numpy(np.array(x_np, copy=True)), torch.from_numpy(np.array(y_np, copy=True)))
    dl = DataLoader(ds, batch_size=budget["batch_size"], shuffle=True,
                    generator=torch.Generator().manual_seed(seed), num_workers=0, pin_memory=True)
    opt = torch.optim.AdamW(adapter.parameters(), lr=budget["lr"], weight_decay=budget["weight_decay"])
    amp = cfg["runtime"]["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, steps, effective = [], 0, set()
    for _ in range(budget["epochs"]):
        adapter.train()
        total = count = 0
        for fmap, tgt in dl:
            fmap, tgt = fmap.to(device), tgt.to(device).float()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prefix, patches = adapter(fmap)
                pred = tail(prefix, patches).float()
                loss = (1 - F.cosine_similarity(pred, tgt, dim=-1).mean()
                        + .1 * (1 - F.cosine_similarity(pred.mean(1), tgt.mean(1), dim=-1).mean()))
            require(torch.isfinite(loss).item(), "Nonfinite normal alignment loss")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            for name, p in adapter.named_parameters():
                if p.grad is not None:
                    require(torch.isfinite(p.grad).all().item(), "Nonfinite adapter gradient")
                    effective.add(name)
            require(all(p.grad is None for p in tail.parameters()), "Inherited parameters received gradients")
            scaler.step(opt); scaler.update()
            steps += 1; total += loss.item() * len(fmap); count += len(fmap)
        history.append(total / count)
    require(steps == budget["epochs"] * math.ceil(len(ds) / budget["batch_size"]), "Training budget differs")
    return dict(history=history, optimizer_steps=steps,
                effective_trainable_parameters=sum(p.numel() for n, p in adapter.named_parameters() if n in effective))


def new_checkpoint(out, arm, job, held):
    return out / "checkpoints" / arm / job["job"] / f"{held}.pt"


def compose_checks(cfg, jobs, records, out, device, write_summary=True):
    rows = []
    for job in jobs:
        print(f"[composition] {job['job']}", flush=True)
        source = load_source(cfg, job["source_backbone"], device)
        target = load_target(cfg, job["target_backbone"], device)
        for held in target_modalities(cfg):
            ckpt_path = Path(job["checkpoint_root"]) / f"{held}.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            checkpoint_identity(ckpt, job["source_backbone"], job["target_backbone"], job["source_stage"], job["target_block"], "mlp", held)
            adapter, original_tail = build_stitch_modules(cfg, target, job["source_backbone"], job["target_backbone"],
                                                          job["source_stage"], job["target_block"], "mlp")
            adapter.load_state_dict(ckpt["adapter"], strict=True)
            adapter = adapter.to(device).eval()
            model = ComposedAnomalyDetector(source_prefix(source, job["source_stage"]), adapter, target,
                      cfg["models"]["sources"][job["source_backbone"]], cfg["models"]["targets"][job["target_backbone"]], job["target_block"])
            counts = model.parameter_counts()
            require(model.tail.layers[0] is original_tail.layers[job["target_block"]]
                    and model.tail.norm is original_tail.norm, "Inherited suffix is not shared")
            first_rgb = None
            for ds in cfg["data"]["datasets"]:
                if cfg["data"]["modality_map"][ds] == held:
                    continue
                rgb = torch.stack([load_rgb_raw(r["image"], 224) for r in records[ds][:2]]).to(device)
                with torch.inference_mode():
                    xf = extract_source_features(source, _normalize(rgb, model.source_spec))[f"s{job['source_stage']}"]
                    yf = extract_target_features(target, _normalize(rgb, model.target_spec))
                    expected = discrepancy(adapter, original_tail, xf, yf, device)
                    actual = model(rgb)
                    cache = Path(cfg["paths"]["cache_dir"]) / "features" / ds / "train"
                    old_x = torch.from_numpy(np.array(np.load(cache / f"source_{job['source_backbone']}_s{job['source_stage']}.npy", mmap_mode="r")[:2], copy=True)).to(device)
                    old_y = torch.from_numpy(np.array(np.load(cache / f"target_{job['target_backbone']}.npy", mmap_mode="r")[:2], copy=True)).to(device)
                    cached = discrepancy(adapter, original_tail, old_x, old_y, device)
                delta = (expected - actual["discrepancy"].flatten(1)).abs().max().item()
                require(delta <= TOLERANCE, f"Composed path differs from original live inference: {delta}")
                rows.append(dict(job=job["job"], held_out=held, source_dataset=ds, n_images=2,
                                 map_max_abs_delta=delta,
                                 fresh_source_cache_max_abs_delta=(xf.float()-old_x.float()).abs().max().item(),
                                 fresh_target_cache_max_abs_delta=(yf.float()-old_y.float()).abs().max().item(),
                                 fresh_cached_map_max_abs_delta=(expected-cached).abs().max().item(), **counts))
                if first_rgb is None:
                    first_rgb = rgb
            # Four self-contained examples; weights stay on the server, not in transfer archives.
            if job["seed"] == 11 and held == "brain_mri":
                path = out / "bundles" / f"{job['job']}_{held}.pt"
                save_bundle(path, model, cfg, job, held, dict(checkpoint_sha256=sha256(ckpt_path)))
                restored, _ = load_bundle(path, device)
                error = (restored(first_rgb)["discrepancy"] - model(first_rgb)["discrepancy"]).abs().max().item()
                require(error <= TOLERANCE, "Self-contained parameter bundle round trip differs")
                rows[-1]["bundle_roundtrip_delta"] = error
                del restored
            del model, adapter, original_tail
        del target, source
        torch.cuda.empty_cache()
    # Stable CSV schema even for cells without a bundle round-trip check.
    for r in rows:
        r.setdefault("bundle_roundtrip_delta", "")
    if write_summary:
        write_csv(out / "composition_checks.csv", rows)
    return rows


def train_phase(cfg, jobs, out, device, write_summary=True):
    training, sample_manifests = [], []
    for job in jobs:
        target = load_target(cfg, job["target_backbone"], device)
        for held in target_modalities(cfg):
            sampled = training_records(cfg, job, held)
            x, y = load_source_arrays(cfg, held, job["source_backbone"], job["target_backbone"],
                                      job["source_stage"], 1000, job["seed"])
            require(len(sampled) == len(x) == len(y) == 4000, "Source sample budget/order incomplete")
            sample_manifests.append(dict(job=job["job"], held_out=held, samples=sampled))
            initial = None
            for arm in NEW_ARMS:
                print(f"[train] {arm} {job['job']} exclude={held}", flush=True)
                adapter, tail = initialize(cfg, target, job, held, arm, device)
                digest = tensor_digest(adapter.state_dict())
                require(initial is None or digest == initial, "Controls have unequal adapter initialization")
                initial = digest
                proof = dict(history=[], optimizer_steps=0, effective_trainable_parameters=0)
                if arm != "untrained_adapter":
                    proof = fit(cfg, adapter, tail, x, y, job["seed"], device)
                path = new_checkpoint(out, arm, job, held)
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = dict(adapter=adapter.cpu().state_dict(), arm=arm, job=job["job"], held_out=held,
                               initial_adapter_sha256=digest, budget=cfg["final"], samples=sampled, **proof)
                torch.save(payload, path)
                training.append(dict(arm=arm, job=job["job"], held_out=held,
                                     initial_adapter_sha256=digest, checkpoint_sha256=sha256(path),
                                     n_normal_images=len(sampled), **proof))
                del adapter, tail
            del x, y
        del target
        torch.cuda.empty_cache()
    require(len(training) == 15 * len(jobs), "Incomplete source training controls")
    if write_summary:
        require(len(training) == 180, "Need all source controls before target evaluation")
        write_json(out / "training_samples.json", dict(rows=sample_manifests))
        write_json(out / "training_completed.json", dict(no_target_data_used=True, no_reselection=True, rows=training))
        return training
    return dict(rows=training, samples=sample_manifests)


def validate_training_receipt(cfg, jobs, out, checkpoint_jobs=None):
    require((out / "training_completed.json").is_file(), "Finish all source training before evaluating target")
    receipt = read_json(out / "training_completed.json")
    training = receipt.get("rows", [])
    expected = {(arm, j["job"], held) for arm in NEW_ARMS for j in jobs for held in target_modalities(cfg)}
    require(receipt.get("no_target_data_used") and receipt.get("no_reselection")
            and len(training) == len(expected) == 180
            and {(r["arm"], r["job"], r["held_out"]) for r in training} == expected,
            "Incomplete all-source-training receipt")
    by_job = {j["job"]: j for j in jobs}
    for r in training:
        if checkpoint_jobs is not None and r["job"] not in checkpoint_jobs:
            continue
        require(sha256(new_checkpoint(out, r["arm"], by_job[r["job"]], r["held_out"])) == r["checkpoint_sha256"],
                "Control weights changed after source training; target evaluation refused")


def evaluate_phase(cfg, jobs, out, device, evaluation_jobs=None, write_summary=True):
    chosen = jobs if evaluation_jobs is None else evaluation_jobs
    require(all(j in jobs for j in chosen), "Evaluation job outside the locked matrix")
    validate_training_receipt(cfg, jobs, out, None if evaluation_jobs is None else {j["job"] for j in chosen})
    metrics, aoss_rows, replays = [], [], []
    for job in chosen:
        target = load_target(cfg, job["target_backbone"], device)
        original_rows = []
        for held in target_modalities(cfg):
            for arm in ARMS:
                adapter, tail = initialize(cfg, target, job, held, arm, device)
                path = Path(job["checkpoint_root"]) / f"{held}.pt" if arm == "original" else new_checkpoint(out, arm, job, held)
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                if arm == "original":
                    checkpoint_identity(ckpt, job["source_backbone"], job["target_backbone"], job["source_stage"], job["target_block"], "mlp", held)
                else:
                    require((ckpt["arm"], ckpt["job"], ckpt["held_out"]) == (arm, job["job"], held), "Control checkpoint identity differs")
                adapter.load_state_dict(ckpt["adapter"], strict=True)
                adapter.eval()
                aoss = compute_aoss(cfg, target, job["source_backbone"], job["target_backbone"], adapter, tail,
                                    held, job["source_stage"], device)
                require(all(math.isfinite(v) for v in aoss.values()), "Nonfinite source-only AOSS report")
                aoss_rows.append(dict(arm=arm, job=job["job"], held_out=held, **aoss))
                for ds in modality_datasets(cfg)[held]:
                    meta = dict(arm=arm, dataset=ds, job=job["job"], pair=f"{job['source_backbone']}_to_{job['target_backbone']}",
                                seed=job["seed"], held_out=held, source_stage=job["source_stage"], target_block=job["target_block"],
                                score_mode="contrast_topk", topk_fraction=.05)
                    rows = evaluate_dataset(cfg, target, job["source_backbone"], job["target_backbone"], adapter, tail, ds,
                                            job["source_stage"], device, .05, ["contrast_topk"], 128,
                                            prediction_path=out / "predictions" / arm / job["job"] / f"{ds}.csv",
                                            prediction_metadata=meta)
                    for r in rows:
                        if arm == "original":
                            original_rows.append(r.copy())
                        metrics.append(dict(arm=arm, job=job["job"], seed=job["seed"], held_out=held, **r))
                del adapter, tail
        replay = compare_metrics(original_rows, read_json(job["reference"])["rows"], TOLERANCE)
        require(all(r["within_tolerance"] for r in replay), "Original AUROC/AP replay differs; stop before interpretation")
        replays.extend(dict(job=job["job"], **r) for r in replay)
        del target
        torch.cuda.empty_cache()
    require(len(metrics) == 24 * len(chosen) and len(aoss_rows) == 20 * len(chosen)
            and len(replays) == 12 * len(chosen), "Incomplete method comparison matrix")
    payload = dict(rows=metrics, aoss_rows=aoss_rows, replays=replays)
    if write_summary:
        require(len(chosen) == 12, "Aggregate evaluation requires all locked jobs")
        write_json(out / "metrics.json", payload)
        return metrics
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--gpus", help="Visible CUDA indices, e.g. 0,1,2,3; auto uses all visible GPUs")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--worker-phase", choices=["compose", "train", "evaluate"])
    ap.add_argument("--job")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    from .method_schedule import parse_gpus, run_phases, run_worker
    if args.worker_phase:
        require(not args.check_only and args.gpus is None and args.job, "Invalid worker arguments")
        run_worker(cfg, args.config, args.worker_phase, args.job, args.gpu)
        return
    require(args.job is None, "--job is only available to a phase worker")
    gpus = parse_gpus(args.gpus if args.gpus is not None else str(args.gpu), torch.cuda.device_count())
    jobs, records, hashes, stats = preflight(cfg, args.config)
    print(f"[preflight] GPUs={gpus}; 12 locked jobs; 60 original + 120 trained + 60 untrained controls; no selection changes", flush=True)
    if args.check_only:
        return
    require(torch.cuda.is_available(), "This full experiment requires an available CUDA GPU")
    out = Path(cfg["paths"]["results_dir"]) / "followup/method_controls"
    require(not out.exists() or not any(out.iterdir()), "Existing method experiment: refuse overwrite/resume")
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(status="running", jobs=jobs, arms=ARMS, original_sha256=hashes, array_weight_stats=stats,
                    versions=versions(), python=platform.python_version(),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    no_reselection=True, score_unchanged=True, original_budget_unchanged=True,
                    training_uses_target_data=False, training_uses_synthetic_anomalies=False,
                    global_selection_fold_exclusive=False, weights_excluded_from_transfer=True,
                    scheduling="one_job_per_gpu_v1", gpus=gpus)
    write_json(out / "run_manifest.json", manifest)
    try:
        for job in jobs:
            write_json(out / "references" / f"{job['job']}.json", read_json(job["reference"]))
        write_json(out / "references/config.json", cfg)
        write_json(out / "references/selected_configs.json", read_json(Path(cfg["paths"]["results_dir"]) / "selected_configs.json"))
        def stop(signum, frame):
            raise SystemExit(128 + signum)
        previous = signal.signal(signal.SIGTERM, stop)
        try:
            composition, metrics = run_phases(cfg, jobs, args.config, out, gpus, manifest, hashes, stats)
        finally:
            signal.signal(signal.SIGTERM, previous)
        require(inputs_unchanged(hashes, stats), "Original inputs changed during method experiment")
        weights = sorted(out.rglob("*.pt"))
        require(len(weights) == 184, "Need 180 control checkpoints and four composition bundles")
        manifest.update(status="complete", phase="complete", original_inputs_unchanged=True,
                        n_composition_checks=len(composition), n_metric_rows=len(metrics),
                        output_sha256={str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob("*"))
                                       if p.is_file() and p.name != "run_manifest.json" and p.suffix != ".pt"},
                        weights_sha256={str(p.relative_to(out)): sha256(p) for p in weights})
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                        original_inputs_unchanged=inputs_unchanged(hashes, stats))
        write_json(out / "run_manifest.json", manifest)
        raise
    write_json(out / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
