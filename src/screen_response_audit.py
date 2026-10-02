"""Replay the original 36 screening points on source perturb pairs. No training/selection/test."""
from __future__ import annotations

import argparse
import math
import platform
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from .common import configure_torch, seed_everything
from .followup_io import check_cache, read_json, read_records
from .models import build_stitch_modules, load_target
from .paired_response_audit import BATCH_SIZE, TOLERANCE, paired_regions, versions, write_csv
from .prediction_io import checkpoint_identity, sha256
from .run_followup import inputs_unchanged, write_json
from .screen_response_io import COMPONENTS, MAPPING, MODALITIES, require, screen_plan
from .train_eval import _score_from_map, compute_aoss, discrepancy


def preflight(cfg, config):
    root = Path(cfg["paths"]["results_dir"])
    selected_path = root / "selected_configs.json"
    jobs, paths = screen_plan(cfg, read_json(selected_path))
    prior_dir = root / "followup/paired_response_audit"
    prior_path = prior_dir / "run_manifest.json"
    prior = read_json(prior_path)
    require(prior["status"] == "complete" and prior["original_inputs_unchanged"]
            and prior["no_training"] and prior["no_reselection"]
            and not prior["target_test_read"] and not prior["target_valid_read"], "Need complete final paired-response prerequisite")
    require(prior["versions"] == versions() and prior["python"] == platform.python_version(),
            "Restore the paired-response environment; no silent version mixing")
    require(len(prior["output_sha256"]) == 62 and prior["n_source_rows"] == 73728,
            "Incomplete paired-response prerequisite")
    paths += [Path(config), selected_path, prior_path]
    for name, digest in prior["output_sha256"].items():
        p = (prior_dir / name).resolve()
        require(p.is_relative_to(prior_dir.resolve()) and sha256(p) == digest, "Changed paired-response prerequisite")
        paths.append(p)
    replay = np.genfromtxt(prior_dir / "aoss_replay.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    require(len(replay) == 300 and all(str(x) == "True" for x in replay["within_tolerance"])
            and np.isfinite(replay["delta"]).all() and np.max(np.abs(replay["delta"])) <= TOLERANCE,
            "Original final AOSS prerequisite failed")
    require(sha256(config) == prior["original_sha256"][str(Path(config))]
            and sha256(selected_path) == prior["original_sha256"][str(selected_path)], "Original config/selection changed")
    source = [f"source_{s}_s{stage}" for s in ["medical", "general"] for stage in [1, 2, 3]]
    target = [f"target_{t}" for t in ["medical", "general"]]
    cache, counts = check_cache(cfg, "perturb", [f"{a}_{k}" for a in ["normal", "pert"] for k in source + target]
                                 + [f"mask_{t}" for t in ["medical", "general"]])
    require(all(n == 256 for n in counts.values()), "Original perturb cache must have 256 rows per dataset")
    arrays = [p for p in cache if p.suffix == ".npy"]
    paths += [p for p in cache if p.suffix != ".npy"]
    for p in cache:
        if str(p) in prior["original_sha256"]:
            require(sha256(p) == prior["original_sha256"][str(p)], "Original perturb records/marker differs")
    for spec in cfg["models"]["targets"].values():
        d = Path(spec["local"])
        p = d / "config.json"
        require(sha256(p) == prior["original_sha256"][str(p)], "Target model config changed")
        paths.append(p)
        weights = list(d.glob("*.safetensors")) + list(d.glob("pytorch_model*.bin"))
        require(bool(weights), f"Missing frozen target weights: {d}")
        arrays += weights
    for p in arrays:
        if str(p) in prior["array_weight_stats"]:
            require(dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) == prior["array_weight_stats"][str(p)],
                    f"Original perturb array/target weight stat changed: {p}")
    code = ["screen_response_audit", "screen_response_io", "paired_response_audit", "models",
            "train_eval", "cache_features", "pipeline", "worker", "followup_io", "prediction_io", "common"]
    paths += [Path(__file__).parent / f"{n}.py" for n in code]
    for n in code:
        p = Path(__file__).parent / f"{n}.py"
        if str(p) in prior["original_sha256"]:
            require(sha256(p) == prior["original_sha256"][str(p)], f"Original research code changed: {p}")
    return jobs, sorted(set(paths), key=str), sorted(set(arrays), key=str)


@torch.inference_mode()
def export_fold(cfg, job, held, adapter, tail, device, output):
    rows = []
    for ds, modality in MAPPING.items():
        if modality == held:
            continue
        d = Path(cfg["paths"]["cache_dir"]) / "perturb" / ds
        records = read_records(d / "records.jsonl")
        s, t, stage = job["source_backbone"], job["target_backbone"], job["source_stage"]
        keys = [f"normal_source_{s}_s{stage}", f"normal_target_{t}",
                f"pert_source_{s}_s{stage}", f"pert_target_{t}"]
        arrays = [np.load(d / f"{k}.npy", mmap_mode="r") for k in keys]
        masks = np.load(d / f"mask_{t}.npy", mmap_mode="r")
        for start in range(0, len(records), BATCH_SIZE):
            stop = min(len(records), start + BATCH_SIZE)
            tensors = [torch.from_numpy(np.array(a[start:stop], copy=True)) for a in arrays]
            dn = discrepancy(adapter, tail, tensors[0], tensors[1], device)
            dp = discrepancy(adapter, tail, tensors[2], tensors[3], device)
            mask = np.array(masks[start:stop], copy=True).reshape(tuple(dn.shape))
            stats = paired_regions(dn.cpu().numpy(), dp.cpu().numpy(), mask)
            ns = _score_from_map(dn, .05, "contrast_topk").cpu().tolist()
            ps = _score_from_map(dp, .05, "contrast_topk").cpu().tolist()
            for offset, stat in enumerate(stats):
                i = start + offset
                rows.append(dict(**job, held_out_modality=held, source_modality=modality, dataset=ds,
                    record_index=i, image_path=records[i]["image"],
                    perturbation_kind_inferred=["intensity", "blur", "patch_copy"][(cfg["project"]["seed"] + i) % 3],
                    kind_evidence="inferred_from_current_generator_and_record_index",
                    **stat, normal_contrast_topk=ns[offset], perturb_contrast_topk=ps[offset],
                    contrast_delta=ps[offset] - ns[offset]))
    write_csv(output, rows)
    return len(rows)


def run(cfg, jobs, out, device):
    checks, count = [], 0
    base = Path(cfg["paths"]["results_dir"]) / "screen"
    for t in ["medical", "general"]:
        target = load_target(cfg, t, device)
        for job in [j for j in jobs if j["target_backbone"] == t]:
            ref = read_json(base / f"{job['job']}.json")
            for held in MODALITIES:
                seed_everything(11, bool(cfg["runtime"].get("deterministic", False)))
                checkpoint = base / "checkpoints" / job["job"] / f"{held}.pt"
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                checkpoint_identity(saved, job["source_backbone"], t, job["source_stage"], job["target_block"], "mlp", held)
                adapter, tail = build_stitch_modules(cfg, target, job["source_backbone"], t,
                                                     job["source_stage"], job["target_block"], "mlp")
                adapter.load_state_dict(saved["adapter"], strict=True)
                adapter, tail = adapter.to(device).eval(), tail.to(device).eval()
                count += export_fold(cfg, job, held, adapter, tail, device,
                                     out / "jobs" / job["job"] / f"source_{held}.csv")
                actual = compute_aoss(cfg, target, job["source_backbone"], t, adapter, tail,
                                      held, job["source_stage"], device)
                original = next(r for r in ref["aoss_rows"] if r["target_modality"] == held)
                for c in COMPONENTS:
                    delta = actual[c] - original[c]
                    checks.append(dict(job=job["job"], held_out_modality=held, component=c,
                        original=original[c], replayed=actual[c], delta=delta,
                        within_tolerance=math.isfinite(delta) and abs(delta) <= TOLERANCE))
                write_csv(out / "aoss_replay.csv", checks)
                require(all(r["within_tolerance"] for r in checks[-5:]),
                        f"Original SCREEN AOSS differs: {job['job']}/{held}; preserve evidence, no tolerance relaxation")
                del saved, adapter, tail
            print(f"[screen-response] Complete: {job['job']}", flush=True)
        del target
        torch.cuda.empty_cache()
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    jobs, paths, arrays = preflight(cfg, args.config)
    if args.check_only:
        print("Ready: 36 original screen jobs / 180 checkpoints; source only; no files written")
        return
    require(torch.cuda.is_available() and 0 <= args.gpu < torch.cuda.device_count(), "Need available CUDA GPU")
    out = Path(cfg["paths"]["results_dir"]) / "followup/screen_response_audit"
    require(not out.exists() or not any(out.iterdir()), "Existing screen-response output; no overwrite/resume")
    hashes = {str(p): sha256(p) for p in paths}
    stats = {str(p): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in arrays}
    m = dict(status="running", protocol="original_screen_diagnostic", jobs=jobs, no_training=True,
        no_reselection=True, target_test_read=False, target_valid_read=False,
        original_score="contrast_topk", topk_fraction=.05, batch_size=BATCH_SIZE,
        replay_tolerance=TOLERANCE, versions=versions(), python=platform.python_version(),
        original_sha256=hashes, array_weight_stats=stats,
        limits="Screen budget only; one seed. Legacy checkpoints lack historical sample/budget manifests. "
               "Net response is diagnostic, not a new AOSS. Large arrays/weights stat-only. "
               "Cross-fold global original selection boundary is retained and disclosed.")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_manifest.json", m)
    configure_torch(bool(cfg["project"].get("allow_tf32", True)))
    try:
        references = out / "references/screen"
        references.mkdir(parents=True)
        for j in jobs:
            shutil.copyfile(Path(cfg["paths"]["results_dir"]) / "screen" / f"{j['job']}.json", references / f"{j['job']}.json")
        shutil.copyfile(Path(cfg["paths"]["results_dir"]) / "selected_configs.json", out / "references/selected_configs.json")
        shutil.copyfile(args.config, out / "references/experiment.yaml")
        m["n_source_rows"] = run(cfg, jobs, out, torch.device(f"cuda:{args.gpu}"))
        require(m["n_source_rows"] == 221184 and inputs_unchanged(hashes, stats), "Incomplete grid or original input changed")
        m.update(status="complete", original_inputs_unchanged=True,
                 output_sha256={p.relative_to(out).as_posix(): sha256(p) for p in out.rglob("*")
                                if p.is_file() and p.name != "run_manifest.json"})
    except BaseException as exc:
        m.update(status="failed", error=str(exc), original_inputs_unchanged=inputs_unchanged(hashes, stats))
        raise
    finally:
        write_json(out / "run_manifest.json", m)


if __name__ == "__main__":
    main()
