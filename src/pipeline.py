from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
from scipy.stats import spearmanr

from .common import config_name, ensure_dir, gpu_ids, load_yaml, read_json, write_json
from .data import prepare_manifest
from .models import ensure_models


def _run_parallel(base_cmds: List[List[str]], gpus: Sequence[int], tag: str) -> None:
    if not base_cmds:
        print(f"[{tag}] nothing to run")
        return
    if not gpus:
        raise RuntimeError("No CUDA GPU detected. This experiment is designed for NVIDIA GPUs (e.g. RTX 3090).")
    pending = list(base_cmds)
    active: list[tuple[subprocess.Popen, int, list[str]]] = []
    failures = []
    while pending or active:
        busy = {gpu for _, gpu, _ in active}
        free = [g for g in gpus if g not in busy]
        while pending and free:
            gpu = free.pop(0)
            cmd = pending.pop(0) + ["--gpu", str(gpu)]
            print(f"[{tag}] GPU{gpu}: {' '.join(cmd)}")
            p = subprocess.Popen(cmd)
            active.append((p, gpu, cmd))
        time.sleep(0.5)
        still = []
        for p, gpu, cmd in active:
            rc = p.poll()
            if rc is None:
                still.append((p, gpu, cmd))
            elif rc != 0:
                failures.append((rc, cmd))
        active = still
        if failures:
            for p, _, _ in active:
                p.terminate()
            raise RuntimeError(f"{tag} worker failed: {failures[0]}")


def _python_module(module: str) -> List[str]:
    return [sys.executable, "-m", module]


def run_cache(cfg: dict, config_path: str, gpus: Sequence[int]) -> None:
    cmds = []
    for ds in cfg["data"]["datasets"]:
        done_train = Path(cfg["paths"]["cache_dir"]) / "features" / ds / "train" / ".done"
        done_test = Path(cfg["paths"]["cache_dir"]) / "features" / ds / "test" / ".done"
        done_pert = Path(cfg["paths"]["cache_dir"]) / "perturb" / ds / ".done"
        if done_train.exists() and done_test.exists() and done_pert.exists():
            continue
        cmds.append(_python_module("src.cache_features") + ["--config", config_path, "--dataset", ds])
    _run_parallel(cmds, gpus, "cache")


def run_screen(cfg: dict, config_path: str, gpus: Sequence[int]) -> None:
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "screen")
    sc = cfg["screen"]
    cmds = []
    for stage in sc["source_stages"]:
        for block in sc["target_blocks"]:
            out = out_dir / f"{config_name(stage, block)}.json"
            if out.exists():
                continue
            cmds.append(_python_module("src.worker") + [
                "--config", config_path, "--mode", "stitch", "--output", str(out),
                "--source-stage", str(stage), "--target-block", str(block),
                "--adapter", str(sc["adapter"]), "--seed", str(sc["seed"]),
                "--n-per-modality", str(sc["n_per_source_modality"]),
                "--epochs", str(sc["epochs"]), "--batch-size", str(sc["batch_size"]),
                "--lr", str(sc["lr"]), "--weight-decay", str(sc["weight_decay"]),
                "--topk-fraction", str(sc["topk_fraction"]),
                "--score-modes", "contrast_topk",
                "--save-checkpoints", "--skip-target-eval",
            ])
    _run_parallel(cmds, gpus, "screen")


def run_baselines(cfg: dict, config_path: str) -> None:
    baseline = Path(cfg["paths"]["results_dir"]) / "baselines.json"
    if baseline.exists():
        return
    cmd = _python_module("src.worker") + [
        "--config", config_path, "--mode", "baseline", "--output", str(baseline),
        "--topk-fraction", str(cfg["screen"]["topk_fraction"]),
    ]
    print("[baseline]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def summarize_screen(cfg: dict) -> List[dict]:
    screen_dir = Path(cfg["paths"]["results_dir"]) / "screen"
    stitchmap_dir = Path(cfg["paths"]["results_dir"]) / "stitchmap"
    rows = []
    for p in sorted(screen_dir.glob("s*_b*.json")):
        obj = read_json(p)
        aoss = [x["aoss"] for x in obj.get("aoss_rows", []) if np.isfinite(x.get("aoss", np.nan))]
        nd = [x["normal_discrepancy"] for x in obj.get("aoss_rows", []) if np.isfinite(x.get("normal_discrepancy", np.nan))]
        stitchmap_path = stitchmap_dir / p.name
        auroc = float("nan")
        if stitchmap_path.exists():
            auroc = read_json(stitchmap_path).get("mean_image_auroc", float("nan"))
        rows.append({
            "config": obj["config"],
            "source_stage": obj["source_stage"],
            "target_block": obj["target_block"],
            "mean_aoss": float(np.mean(aoss)) if aoss else float("nan"),
            "mean_normal_discrepancy": float(np.mean(nd)) if nd else float("nan"),
            "posthoc_mean_image_auroc": auroc,
        })
    # Source-only AOSS is the selection criterion. Target test metrics are NEVER used here.
    rows.sort(key=lambda r: np.nan_to_num(r["mean_aoss"], nan=-1e9), reverse=True)
    out_csv = Path(cfg["paths"]["results_dir"]) / "screen_summary.csv"
    ensure_dir(out_csv.parent)
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def run_final(cfg: dict, config_path: str, gpus: Sequence[int]) -> None:
    ranked = summarize_screen(cfg)
    if not ranked:
        raise RuntimeError("No screening results found")
    topn = int(cfg["final"]["top_configs"])
    selected = ranked[:topn]
    write_json(Path(cfg["paths"]["results_dir"]) / "selected_configs.json", selected)
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "final")
    fc = cfg["final"]
    cmds = []
    for item in selected:
        for seed in fc["seeds"]:
            out = out_dir / f"{item['config']}_seed{seed}.json"
            if out.exists():
                continue
            cmds.append(_python_module("src.worker") + [
                "--config", config_path, "--mode", "stitch", "--output", str(out),
                "--source-stage", str(item["source_stage"]), "--target-block", str(item["target_block"]),
                "--adapter", str(fc["adapter"]), "--seed", str(seed),
                "--n-per-modality", str(fc["n_per_source_modality"]),
                "--epochs", str(fc["epochs"]), "--batch-size", str(fc["batch_size"]),
                "--lr", str(fc["lr"]), "--weight-decay", str(fc["weight_decay"]),
                "--topk-fraction", str(fc["topk_fraction"]),
                "--score-modes", "contrast_topk",
                "--save-checkpoints",
            ])
    _run_parallel(cmds, gpus, "final")


def run_stitchmap(cfg: dict, config_path: str, gpus: Sequence[int]) -> None:
    # Post-hoc evaluation occurs only after AOSS-based configuration selection is locked.
    screen_dir = Path(cfg["paths"]["results_dir"]) / "screen"
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "stitchmap")
    sc = cfg["screen"]
    cmds = []
    for stage in sc["source_stages"]:
        for block in sc["target_blocks"]:
            name = config_name(stage, block)
            out = out_dir / f"{name}.json"
            if out.exists():
                continue
            ckpt_root = screen_dir / "checkpoints" / name
            cmds.append(_python_module("src.worker") + [
                "--config", config_path, "--mode", "eval_saved", "--output", str(out),
                "--checkpoint-root", str(ckpt_root),
                "--source-stage", str(stage), "--target-block", str(block),
                "--adapter", str(sc["adapter"]), "--seed", str(sc["seed"]),
                "--batch-size", str(sc["batch_size"]),
                "--topk-fraction", str(sc["topk_fraction"]),
                "--score-modes", "contrast_topk",
            ])
    _run_parallel(cmds, gpus, "stitchmap")
    summarize_screen(cfg)


def run_ablation(cfg: dict, config_path: str, gpus: Sequence[int]) -> None:
    ac = cfg["ablation"]
    if not ac.get("enabled", True):
        return
    selected_path = Path(cfg["paths"]["results_dir"]) / "selected_configs.json"
    selected = read_json(selected_path)
    best = selected[0]
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "ablation")
    jobs = []
    seen = set()
    for n in ac["calibration_sizes"]:
        key = (int(n), "mlp")
        seen.add(key)
        jobs.append(key)
    main_n = int(cfg["final"]["n_per_source_modality"])
    for adapter in ac["adapter_types"]:
        key = (main_n, str(adapter))
        if key not in seen:
            jobs.append(key)
            seen.add(key)

    cmds = []
    for n, adapter in jobs:
        out = out_dir / f"{best['config']}_n{n}_{adapter}.json"
        if out.exists():
            continue
        cmds.append(_python_module("src.worker") + [
            "--config", config_path, "--mode", "stitch", "--output", str(out),
            "--source-stage", str(best["source_stage"]), "--target-block", str(best["target_block"]),
            "--adapter", adapter, "--seed", str(ac["seed"]),
            "--n-per-modality", str(n), "--epochs", str(ac["epochs"]),
            "--batch-size", str(ac["batch_size"]), "--lr", str(ac["lr"]),
            "--weight-decay", str(ac["weight_decay"]),
            "--topk-fraction", str(cfg["final"]["topk_fraction"]),
            "--score-modes", *[str(x) for x in ac["score_modes"]],
        ])
    _run_parallel(cmds, gpus, "ablation")


def make_report(cfg: dict) -> None:
    results_root = Path(cfg["paths"]["results_dir"])
    screen = summarize_screen(cfg)
    final_objs = [read_json(p) for p in sorted((results_root / "final").glob("*.json"))]
    final_rows = [r for o in final_objs for r in o.get("rows", [])]
    if final_rows:
        fields = sorted({k for r in final_rows for k in r.keys()})
        with open(results_root / "final_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(final_rows)

    abl_objs = [read_json(p) for p in sorted((results_root / "ablation").glob("*.json"))]
    abl_rows = [r for o in abl_objs for r in o.get("rows", [])]
    if abl_rows:
        fields = sorted({k for r in abl_rows for k in r.keys()})
        with open(results_root / "ablation_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(abl_rows)

    rho = pval = float("nan")
    valid = [(r["mean_aoss"], r["posthoc_mean_image_auroc"]) for r in screen if np.isfinite(r["mean_aoss"]) and np.isfinite(r["posthoc_mean_image_auroc"])]
    if len(valid) >= 3:
        rho, pval = spearmanr([x[0] for x in valid], [x[1] for x in valid])

    selected = read_json(results_root / "selected_configs.json") if (results_root / "selected_configs.json").exists() else []
    lines = [
        "# MedStitch-ZS automated experiment report",
        "",
        f"- Screened stitch configurations: {len(screen)}",
        f"- Selected configs: {', '.join(x['config'] for x in selected) if selected else 'N/A'}",
        f"- AOSS vs. zero-shot AUROC Spearman rho: {rho:.4f} (p={pval:.4g})" if np.isfinite(rho) else "- AOSS correlation: insufficient completed results",
        "",
        "## Screening ranking",
        "",
        "| AOSS rank | Config | Mean AOSS | Post-hoc mean image AUROC | Mean normal discrepancy |",
        "|---:|---|---:|---:|---:|",
    ]
    for i, r in enumerate(screen, 1):
        lines.append(f"| {i} | {r['config']} | {r['mean_aoss']:.4f} | {r['posthoc_mean_image_auroc']:.4f} | {r['mean_normal_discrepancy']:.4f} |")
    lines += [
        "",
        "The final_results.csv file contains per-dataset/per-seed results for the selected configurations.",
        "The ablation_results.csv file contains calibration-size, adapter and score-mode ablations.",
    ]
    (results_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all", choices=["all", "prepare", "cache", "screen", "final", "stitchmap", "ablation", "report"])
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    for key in ("data_dir", "model_dir", "cache_dir", "results_dir"):
        ensure_dir(cfg["paths"][key])

    gpus = gpu_ids(int(cfg["project"].get("max_gpus", 4)))
    print(f"[runtime] GPUs: {gpus}")

    # Validate/discover data before downloading model weights so a bad data layout fails fast.
    if args.stage in ("all", "prepare"):
        prepare_manifest(cfg)
        if args.stage == "prepare":
            return
    elif args.stage != "report" and not (Path(cfg["paths"]["cache_dir"]) / "manifest.jsonl").exists():
        prepare_manifest(cfg)

    if args.stage != "report":
        ensure_models(cfg)

    if args.stage in ("all", "cache"):
        run_cache(cfg, args.config, gpus)
        if args.stage == "cache":
            return
    if args.stage in ("all", "screen"):
        run_screen(cfg, args.config, gpus)
        summarize_screen(cfg)
        if args.stage == "screen":
            return
    if args.stage in ("all", "final"):
        run_final(cfg, args.config, gpus)
        if args.stage == "final":
            return
    if args.stage == "all":
        run_baselines(cfg, args.config)
    if args.stage in ("all", "stitchmap"):
        run_stitchmap(cfg, args.config, gpus)
        if args.stage == "stitchmap":
            return
    if args.stage in ("all", "ablation"):
        run_ablation(cfg, args.config, gpus)
        if args.stage == "ablation":
            return
    if args.stage in ("all", "report"):
        make_report(cfg)


if __name__ == "__main__":
    main()
