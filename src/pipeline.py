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

from .common import config_name, ensure_dir, gpu_ids, load_yaml, pair_name, read_json, write_json
from .data import data_status, prepare_manifest
from .models import ensure_models, model_status


def run_preflight(cfg: dict) -> dict:
    data_error = None
    try:
        prepare_manifest(cfg)
    except Exception as exc:
        data_error = str(exc)

    mstatus = model_status(cfg)
    models_ready = all(
        item["ready"]
        for group in ("sources", "targets")
        for item in mstatus[group].values()
    )
    report = {
        "data": data_status(cfg),
        "models": mstatus,
        "models_ready": models_ready,
    }
    report["data"]["error"] = data_error
    report["ready_for_full_run"] = bool(report["data"]["ready"] and models_ready)
    out = Path(cfg["paths"]["cache_dir"]) / "preflight_report.json"
    write_json(out, report)

    print("\n[prepare] 实验准备检查")
    print(f"  data ready:   {report['data']['ready']}")
    print(f"  models ready: {models_ready}")
    if report["data"]["missing"]:
        print("  missing datasets: " + ", ".join(report["data"]["missing"]))
    for group in ("sources", "targets"):
        for name, item in mstatus[group].items():
            if item["ready"]:
                state = "READY"
            elif item.get("manual", False):
                state = "MISSING (manual Google Drive)"
            else:
                state = "MISSING (will auto-download from ModelScope)"
            print(f"  {group[:-1]} {name}: {state}")
    manual_archives = mstatus.get("manual_archives", {})
    if manual_archives.get("archives"):
        print("  model raw archive(s):")
        for x in manual_archives["archives"]:
            print(f"    - {x}")
    print(f"  report: {out}")
    print("  guide:  PREPARE_EXPERIMENT_CN.md")
    return report


def _run_parallel(cmds: List[List[str]], gpus: Sequence[int], tag: str) -> None:
    if not cmds:
        print(f"[{tag}] nothing to run")
        return
    if not gpus:
        raise RuntimeError("No CUDA GPU detected. Set GPUS=... on an NVIDIA machine.")
    pending = list(cmds)
    active = []
    while pending or active:
        busy = {gpu for _, gpu, _ in active}
        free = [g for g in gpus if g not in busy]
        while pending and free:
            gpu = free.pop(0)
            cmd = pending.pop(0) + ["--gpu", str(gpu)]
            print(f"[{tag}] GPU{gpu}: {' '.join(cmd)}")
            active.append((subprocess.Popen(cmd), gpu, cmd))
        time.sleep(0.5)
        still = []
        for p, gpu, cmd in active:
            rc = p.poll()
            if rc is None:
                still.append((p, gpu, cmd))
            elif rc != 0:
                for q, _, _ in still:
                    q.terminate()
                raise RuntimeError(f"{tag} worker failed ({rc}): {' '.join(cmd)}")
        active = still


def _py(module: str) -> List[str]:
    return [sys.executable, "-m", module]


def _cache_complete(cfg: dict, ds: str) -> bool:
    root = Path(cfg["paths"]["cache_dir"])
    for split in ("train", "test"):
        d = root / "features" / ds / split
        if not (d / ".done").exists():
            return False
        for source_name in cfg["models"]["sources"]:
            for stage in (1, 2, 3):
                if not (d / f"source_{source_name}_s{stage}.npy").exists():
                    return False
        for target_name in cfg["models"]["targets"]:
            if not (d / f"target_{target_name}.npy").exists():
                return False
    p = root / "perturb" / ds
    if not (p / ".done").exists():
        return False
    for source_name in cfg["models"]["sources"]:
        for stage in (1, 2, 3):
            if not (p / f"normal_source_{source_name}_s{stage}.npy").exists():
                return False
            if not (p / f"pert_source_{source_name}_s{stage}.npy").exists():
                return False
    for target_name in cfg["models"]["targets"]:
        for name in (
            f"normal_target_{target_name}.npy",
            f"pert_target_{target_name}.npy",
            f"mask_{target_name}.npy",
        ):
            if not (p / name).exists():
                return False
    return True


def run_cache(cfg, config_path, gpus):
    cmds = []
    for ds in cfg["data"]["datasets"]:
        if not _cache_complete(cfg, ds):
            cmds.append(_py("src.cache_features") + ["--config", config_path, "--dataset", ds])
    _run_parallel(cmds, gpus, "cache")


def run_screen(cfg, config_path, gpus):
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "screen")
    sc = cfg["screen"]
    cmds = []
    for source_name, target_name in cfg["models"]["pairs"]:
        for stage in sc["source_stages"]:
            for block in sc["target_blocks"]:
                name = config_name(source_name, target_name, stage, block)
                out = out_dir / f"{name}.json"
                if out.exists():
                    continue
                cmds.append(_py("src.worker") + [
                    "--config", config_path, "--mode", "stitch", "--output", str(out),
                    "--source-name", source_name, "--target-name", target_name,
                    "--source-stage", str(stage), "--target-block", str(block),
                    "--adapter", str(sc["adapter"]), "--seed", str(sc["seed"]),
                    "--n-per-modality", str(sc["n_per_source_modality"]),
                    "--epochs", str(sc["epochs"]), "--batch-size", str(sc["batch_size"]),
                    "--lr", str(sc["lr"]), "--weight-decay", str(sc["weight_decay"]),
                    "--topk-fraction", str(sc["topk_fraction"]),
                    "--score-modes", "contrast_topk", "--save-checkpoints", "--skip-target-eval",
                ])
    _run_parallel(cmds, gpus, "screen")


def summarize_screen(cfg):
    screen_dir = Path(cfg["paths"]["results_dir"]) / "screen"
    stitch_dir = Path(cfg["paths"]["results_dir"]) / "stitchmap"
    rows = []
    for p in sorted(screen_dir.glob("*.json")):
        obj = read_json(p)
        if "source_backbone" not in obj:
            continue
        vals = [x["aoss"] for x in obj.get("aoss_rows", []) if np.isfinite(x.get("aoss", np.nan))]
        post = stitch_dir / p.name
        rows.append({
            "config": obj["config"],
            "source_backbone": obj["source_backbone"],
            "target_backbone": obj["target_backbone"],
            "pair": pair_name(obj["source_backbone"], obj["target_backbone"]),
            "source_stage": obj["source_stage"],
            "target_block": obj["target_block"],
            "mean_aoss": float(np.mean(vals)) if vals else float("nan"),
            "posthoc_mean_image_auroc": read_json(post).get("mean_image_auroc", float("nan")) if post.exists() else float("nan"),
        })

    for source_name, target_name in cfg["models"]["pairs"]:
        pn = pair_name(source_name, target_name)
        subset = [r for r in rows if r["pair"] == pn]
        subset.sort(key=lambda r: np.nan_to_num(r["mean_aoss"], nan=-1e9), reverse=True)
        for rank, row in enumerate(subset, 1):
            row["aoss_rank_within_pair"] = rank

    rows.sort(key=lambda r: (r["pair"], r.get("aoss_rank_within_pair", 999)))
    if rows:
        with open(Path(cfg["paths"]["results_dir"]) / "screen_summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return rows


def select_configs(cfg):
    rows = summarize_screen(cfg)
    topn = int(cfg["final"]["top_configs_per_pair"])
    selected = []
    for source_name, target_name in cfg["models"]["pairs"]:
        pn = pair_name(source_name, target_name)
        subset = [r for r in rows if r["pair"] == pn]
        subset.sort(key=lambda r: np.nan_to_num(r["mean_aoss"], nan=-1e9), reverse=True)
        selected.extend(subset[:topn])
    payload = {
        "selection_metric": "source_only_AOSS",
        "primary_pair": pair_name(cfg["models"]["primary_source"], cfg["models"]["primary_target"]),
        "selected": selected,
    }
    write_json(Path(cfg["paths"]["results_dir"]) / "selected_configs.json", payload)
    return payload


def run_final(cfg, config_path, gpus):
    selected = select_configs(cfg)["selected"]
    fc = cfg["final"]
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "final")
    cmds = []
    for item in selected:
        for seed in fc["seeds"]:
            out = out_dir / f"{item['config']}_seed{seed}.json"
            if out.exists():
                continue
            cmds.append(_py("src.worker") + [
                "--config", config_path, "--mode", "stitch", "--output", str(out),
                "--source-name", item["source_backbone"], "--target-name", item["target_backbone"],
                "--source-stage", str(item["source_stage"]), "--target-block", str(item["target_block"]),
                "--adapter", str(fc["adapter"]), "--seed", str(seed),
                "--n-per-modality", str(fc["n_per_source_modality"]),
                "--epochs", str(fc["epochs"]), "--batch-size", str(fc["batch_size"]),
                "--lr", str(fc["lr"]), "--weight-decay", str(fc["weight_decay"]),
                "--topk-fraction", str(fc["topk_fraction"]), "--score-modes", "contrast_topk",
                "--save-checkpoints",
            ])
    _run_parallel(cmds, gpus, "final")


def run_stitchmap(cfg, config_path, gpus):
    sc = cfg["screen"]
    screen_dir = Path(cfg["paths"]["results_dir"]) / "screen"
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "stitchmap")
    cmds = []
    for source_name, target_name in cfg["models"]["pairs"]:
        for stage in sc["source_stages"]:
            for block in sc["target_blocks"]:
                name = config_name(source_name, target_name, stage, block)
                out = out_dir / f"{name}.json"
                if out.exists():
                    continue
                cmds.append(_py("src.worker") + [
                    "--config", config_path, "--mode", "eval_saved", "--output", str(out),
                    "--checkpoint-root", str(screen_dir / "checkpoints" / name),
                    "--source-name", source_name, "--target-name", target_name,
                    "--source-stage", str(stage), "--target-block", str(block),
                    "--adapter", str(sc["adapter"]), "--seed", str(sc["seed"]),
                    "--batch-size", str(sc["batch_size"]),
                    "--topk-fraction", str(sc["topk_fraction"]), "--score-modes", "contrast_topk",
                ])
    _run_parallel(cmds, gpus, "stitchmap")
    summarize_screen(cfg)


def run_baselines(cfg, config_path):
    out = Path(cfg["paths"]["results_dir"]) / "baselines.json"
    if out.exists():
        return
    subprocess.run(_py("src.worker") + [
        "--config", config_path, "--mode", "baseline", "--output", str(out),
        "--topk-fraction", str(cfg["screen"]["topk_fraction"]),
    ], check=True)


def run_ablation(cfg, config_path, gpus):
    if not cfg["ablation"].get("enabled", True):
        return
    ac = cfg["ablation"]
    payload = read_json(Path(cfg["paths"]["results_dir"]) / "selected_configs.json")
    candidates = [
        x for x in payload["selected"]
        if x["source_backbone"] == ac["source_name"] and x["target_backbone"] == ac["target_name"]
    ]
    best = candidates[0]
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "ablation")
    jobs, seen = [], set()
    for n in ac["calibration_sizes"]:
        jobs.append((int(n), "mlp")); seen.add((int(n), "mlp"))
    for adapter in ac["adapter_types"]:
        key = (int(cfg["final"]["n_per_source_modality"]), str(adapter))
        if key not in seen:
            jobs.append(key); seen.add(key)

    cmds = []
    for n, adapter in jobs:
        out = out_dir / f"{best['config']}_n{n}_{adapter}.json"
        if out.exists():
            continue
        cmds.append(_py("src.worker") + [
            "--config", config_path, "--mode", "stitch", "--output", str(out),
            "--source-name", ac["source_name"], "--target-name", ac["target_name"],
            "--source-stage", str(best["source_stage"]), "--target-block", str(best["target_block"]),
            "--adapter", adapter, "--seed", str(ac["seed"]), "--n-per-modality", str(n),
            "--epochs", str(ac["epochs"]), "--batch-size", str(ac["batch_size"]),
            "--lr", str(ac["lr"]), "--weight-decay", str(ac["weight_decay"]),
            "--topk-fraction", str(cfg["final"]["topk_fraction"]),
            "--score-modes", *[str(x) for x in ac["score_modes"]],
        ])
    _run_parallel(cmds, gpus, "ablation")


def _write_rows(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def make_report(cfg):
    root = Path(cfg["paths"]["results_dir"])
    screen = summarize_screen(cfg)
    final_rows = [r for p in sorted((root / "final").glob("*.json")) for r in read_json(p).get("rows", [])]
    abl_rows = [r for p in sorted((root / "ablation").glob("*.json")) for r in read_json(p).get("rows", [])]
    _write_rows(root / "final_results.csv", final_rows)
    _write_rows(root / "ablation_results.csv", abl_rows)

    lines = [
        "# MedStitch 2×2 自动实验报告", "",
        "主研究：CNN source 预训练域（medical/general）× Transformer target 预训练域（medical/general）。", "",
        "| Source | Target | Domain | Mean image AUROC | N |",
        "|---|---|---|---:|---:|",
    ]
    for source_name, target_name in cfg["models"]["pairs"]:
        for group in ("radiology", "non_radiology"):
            vals = [
                r["image_auroc"] for r in final_rows
                if r.get("source_backbone") == source_name
                and r.get("target_backbone") == target_name
                and r.get("domain_group") == group
                and r.get("score_mode") == "contrast_topk"
                and np.isfinite(r.get("image_auroc", np.nan))
            ]
            lines.append(f"| {source_name} | {target_name} | {group} | "
                         + (f"{np.mean(vals):.4f}" if vals else "N/A") + f" | {len(vals)} |")

    lines += ["", "## AOSS 与真实 AUROC 的 post-hoc 相关性", ""]
    for source_name, target_name in cfg["models"]["pairs"]:
        pn = pair_name(source_name, target_name)
        subset = [r for r in screen if r["pair"] == pn]
        valid = [(r["mean_aoss"], r["posthoc_mean_image_auroc"]) for r in subset
                 if np.isfinite(r["mean_aoss"]) and np.isfinite(r["posthoc_mean_image_auroc"])]
        if len(valid) >= 3:
            rho, p = spearmanr([x[0] for x in valid], [x[1] for x in valid])
            lines.append(f"- {pn}: Spearman rho={rho:.4f}, p={p:.4g}")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all",
                    choices=["all", "prepare", "models", "cache", "screen", "final", "stitchmap", "ablation", "report"])
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    for key in ("data_dir", "model_dir", "cache_dir", "results_dir"):
        ensure_dir(cfg["paths"][key])
    gpus = gpu_ids(int(cfg["project"].get("max_gpus", 4)))

    if args.stage == "prepare":
        run_preflight(cfg); return
    if args.stage == "models":
        ensure_models(cfg); run_preflight(cfg); return

    if args.stage != "report":
        prepare_manifest(cfg)
        ensure_models(cfg)

    if args.stage in ("all", "cache"):
        run_cache(cfg, args.config, gpus)
        if args.stage == "cache": return
    if args.stage in ("all", "screen"):
        run_screen(cfg, args.config, gpus); summarize_screen(cfg)
        if args.stage == "screen": return
    if args.stage in ("all", "final"):
        run_final(cfg, args.config, gpus)
        if args.stage == "final": return
    if args.stage == "all":
        run_baselines(cfg, args.config)
    if args.stage in ("all", "stitchmap"):
        run_stitchmap(cfg, args.config, gpus)
        if args.stage == "stitchmap": return
    if args.stage in ("all", "ablation"):
        run_ablation(cfg, args.config, gpus)
        if args.stage == "ablation": return
    if args.stage in ("all", "report"):
        make_report(cfg)


if __name__ == "__main__":
    main()
