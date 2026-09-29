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
        raise RuntimeError("No CUDA GPU detected. This experiment is designed for NVIDIA GPUs.")
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


def _cache_complete(cfg: dict, ds: str) -> bool:
    root = Path(cfg["paths"]["cache_dir"])
    target_names = list(cfg["models"]["targets"])
    for split in ("train", "test"):
        d = root / "features" / ds / split
        if not (d / ".done").exists():
            return False
        if not all((d / f"target_{name}.npy").exists() for name in target_names):
            return False
    p = root / "perturb" / ds
    if not (p / ".done").exists():
        return False
    for name in target_names:
        if not (p / f"normal_target_{name}.npy").exists():
            return False
        if not (p / f"pert_target_{name}.npy").exists():
            return False
        if not (p / f"mask_{name}.npy").exists():
            return False
    return True


def run_cache(cfg: dict, config_path: str, gpus: Sequence[int]) -> None:
    cmds = []
    for ds in cfg["data"]["datasets"]:
        if _cache_complete(cfg, ds):
            continue
        cmds.append(_python_module("src.cache_features") + [
            "--config", config_path, "--dataset", ds
        ])
    _run_parallel(cmds, gpus, "cache")


def run_screen(cfg: dict, config_path: str, gpus: Sequence[int]) -> None:
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "screen")
    sc = cfg["screen"]
    cmds = []
    for target_name in sc["target_names"]:
        for stage in sc["source_stages"]:
            for block in sc["target_blocks"]:
                name = config_name(stage, block, target_name)
                out = out_dir / f"{name}.json"
                if out.exists():
                    continue
                cmds.append(_python_module("src.worker") + [
                    "--config", config_path,
                    "--mode", "stitch",
                    "--output", str(out),
                    "--target-name", target_name,
                    "--source-stage", str(stage),
                    "--target-block", str(block),
                    "--adapter", str(sc["adapter"]),
                    "--seed", str(sc["seed"]),
                    "--n-per-modality", str(sc["n_per_source_modality"]),
                    "--epochs", str(sc["epochs"]),
                    "--batch-size", str(sc["batch_size"]),
                    "--lr", str(sc["lr"]),
                    "--weight-decay", str(sc["weight_decay"]),
                    "--topk-fraction", str(sc["topk_fraction"]),
                    "--score-modes", "contrast_topk",
                    "--save-checkpoints",
                    "--skip-target-eval",
                ])
    _run_parallel(cmds, gpus, "screen")


def run_baselines(cfg: dict, config_path: str) -> None:
    baseline = Path(cfg["paths"]["results_dir"]) / "baselines.json"
    if baseline.exists():
        try:
            old = read_json(baseline)
            expected = set(cfg["models"]["targets"])
            found = {r.get("target_backbone") for r in old.get("rows", [])}
            if expected.issubset(found):
                return
            print("[baseline] stale pre-medical baseline cache detected; rebuilding")
            baseline.unlink()
        except Exception:
            baseline.unlink(missing_ok=True)
    cmd = _python_module("src.worker") + [
        "--config", config_path,
        "--mode", "baseline",
        "--output", str(baseline),
        "--topk-fraction", str(cfg["screen"]["topk_fraction"]),
    ]
    print("[baseline]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def summarize_screen(cfg: dict) -> List[dict]:
    screen_dir = Path(cfg["paths"]["results_dir"]) / "screen"
    stitchmap_dir = Path(cfg["paths"]["results_dir"]) / "stitchmap"
    rows = []
    for p in sorted(screen_dir.glob("*.json")):
        obj = read_json(p)
        # Ignore pre-medical-version screening artifacts such as s2_b6.json.
        if "target_backbone" not in obj:
            continue
        target_name = obj["target_backbone"]
        if target_name not in cfg["models"]["targets"]:
            continue
        aoss = [x["aoss"] for x in obj.get("aoss_rows", []) if np.isfinite(x.get("aoss", np.nan))]
        nd = [x["normal_discrepancy"] for x in obj.get("aoss_rows", []) if np.isfinite(x.get("normal_discrepancy", np.nan))]
        radiology_aoss = [x["aoss"] for x in obj.get("aoss_rows", []) if x.get("domain_group") == "radiology" and np.isfinite(x.get("aoss", np.nan))]
        nonrad_aoss = [x["aoss"] for x in obj.get("aoss_rows", []) if x.get("domain_group") == "non_radiology" and np.isfinite(x.get("aoss", np.nan))]
        stitchmap_path = stitchmap_dir / p.name
        auroc = float("nan")
        if stitchmap_path.exists():
            auroc = read_json(stitchmap_path).get("mean_image_auroc", float("nan"))
        rows.append({
            "config": obj["config"],
            "target_backbone": target_name,
            "source_stage": obj["source_stage"],
            "target_block": obj["target_block"],
            "mean_aoss": float(np.mean(aoss)) if aoss else float("nan"),
            "radiology_aoss": float(np.mean(radiology_aoss)) if radiology_aoss else float("nan"),
            "non_radiology_aoss": float(np.mean(nonrad_aoss)) if nonrad_aoss else float("nan"),
            "mean_normal_discrepancy": float(np.mean(nd)) if nd else float("nan"),
            "posthoc_mean_image_auroc": auroc,
        })

    for target_name in cfg["screen"]["target_names"]:
        subset = [r for r in rows if r["target_backbone"] == target_name]
        subset.sort(key=lambda r: np.nan_to_num(r["mean_aoss"], nan=-1e9), reverse=True)
        for rank, row in enumerate(subset, 1):
            row["aoss_rank_within_target"] = rank

    rows.sort(key=lambda r: (r["target_backbone"], r.get("aoss_rank_within_target", 999)))
    out_csv = Path(cfg["paths"]["results_dir"]) / "screen_summary.csv"
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def select_configs(cfg: dict) -> dict:
    rows = summarize_screen(cfg)
    topn = int(cfg["final"]["top_configs_per_target"])
    selected = []
    for target_name in cfg["screen"]["target_names"]:
        subset = [r for r in rows if r["target_backbone"] == target_name]
        subset.sort(key=lambda r: np.nan_to_num(r["mean_aoss"], nan=-1e9), reverse=True)
        selected.extend(subset[:topn])
    payload = {
        "primary_target": cfg["models"]["primary_target"],
        "selection_metric": "source_only_AOSS",
        "selected": selected,
    }
    write_json(Path(cfg["paths"]["results_dir"]) / "selected_configs.json", payload)
    return payload


def run_final(cfg: dict, config_path: str, gpus: Sequence[int]) -> None:
    payload = select_configs(cfg)
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "final")
    fc = cfg["final"]
    cmds = []
    for item in payload["selected"]:
        for seed in fc["seeds"]:
            out = out_dir / f"{item['config']}_seed{seed}.json"
            if out.exists():
                continue
            cmds.append(_python_module("src.worker") + [
                "--config", config_path,
                "--mode", "stitch",
                "--output", str(out),
                "--target-name", str(item["target_backbone"]),
                "--source-stage", str(item["source_stage"]),
                "--target-block", str(item["target_block"]),
                "--adapter", str(fc["adapter"]),
                "--seed", str(seed),
                "--n-per-modality", str(fc["n_per_source_modality"]),
                "--epochs", str(fc["epochs"]),
                "--batch-size", str(fc["batch_size"]),
                "--lr", str(fc["lr"]),
                "--weight-decay", str(fc["weight_decay"]),
                "--topk-fraction", str(fc["topk_fraction"]),
                "--score-modes", "contrast_topk",
                "--save-checkpoints",
            ])
    _run_parallel(cmds, gpus, "final")


def run_stitchmap(cfg: dict, config_path: str, gpus: Sequence[int]) -> None:
    screen_dir = Path(cfg["paths"]["results_dir"]) / "screen"
    out_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "stitchmap")
    sc = cfg["screen"]
    cmds = []
    for target_name in sc["target_names"]:
        for stage in sc["source_stages"]:
            for block in sc["target_blocks"]:
                name = config_name(stage, block, target_name)
                out = out_dir / f"{name}.json"
                if out.exists():
                    continue
                ckpt_root = screen_dir / "checkpoints" / name
                cmds.append(_python_module("src.worker") + [
                    "--config", config_path,
                    "--mode", "eval_saved",
                    "--output", str(out),
                    "--checkpoint-root", str(ckpt_root),
                    "--target-name", target_name,
                    "--source-stage", str(stage),
                    "--target-block", str(block),
                    "--adapter", str(sc["adapter"]),
                    "--seed", str(sc["seed"]),
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
    payload = read_json(Path(cfg["paths"]["results_dir"]) / "selected_configs.json")
    target_name = str(ac.get("target_name", cfg["models"]["primary_target"]))
    candidates = [x for x in payload["selected"] if x["target_backbone"] == target_name]
    if not candidates:
        raise RuntimeError(f"No selected configuration for target={target_name}")
    best = candidates[0]
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
            "--config", config_path,
            "--mode", "stitch",
            "--output", str(out),
            "--target-name", target_name,
            "--source-stage", str(best["source_stage"]),
            "--target-block", str(best["target_block"]),
            "--adapter", adapter,
            "--seed", str(ac["seed"]),
            "--n-per-modality", str(n),
            "--epochs", str(ac["epochs"]),
            "--batch-size", str(ac["batch_size"]),
            "--lr", str(ac["lr"]),
            "--weight-decay", str(ac["weight_decay"]),
            "--topk-fraction", str(cfg["final"]["topk_fraction"]),
            "--score-modes", *[str(x) for x in ac["score_modes"]],
        ])
    _run_parallel(cmds, gpus, "ablation")


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_report(cfg: dict) -> None:
    results_root = Path(cfg["paths"]["results_dir"])
    screen = summarize_screen(cfg)
    final_objs = [read_json(p) for p in sorted((results_root / "final").glob("*.json"))]
    final_rows = [r for o in final_objs for r in o.get("rows", [])]
    _write_rows(results_root / "final_results.csv", final_rows)
    abl_objs = [read_json(p) for p in sorted((results_root / "ablation").glob("*.json"))]
    abl_rows = [r for o in abl_objs for r in o.get("rows", [])]
    _write_rows(results_root / "ablation_results.csv", abl_rows)
    payload = read_json(results_root / "selected_configs.json") if (results_root / "selected_configs.json").exists() else {"selected": []}

    lines = [
        "# MedStitch-ZS automated experiment report",
        "",
        f"- Primary target backbone: {cfg['models']['primary_target']}",
        f"- Screened stitch configurations: {len(screen)}",
        "- Selection metric: source-only AOSS (target test data never used for model selection)",
        "",
        "## Medical-vs-general pretraining comparison",
        "",
        "| Target | Domain group | Mean image AUROC | N rows |",
        "|---|---|---:|---:|",
    ]

    for target_name in cfg["screen"]["target_names"]:
        for group in ("radiology", "non_radiology"):
            vals = [
                r["image_auroc"] for r in final_rows
                if r.get("target_backbone") == target_name
                and r.get("domain_group") == group
                and r.get("score_mode") == "contrast_topk"
                and np.isfinite(r.get("image_auroc", np.nan))
            ]
            lines.append(
                f"| {target_name} | {group} | "
                + (f"{np.mean(vals):.4f}" if vals else "N/A")
                + f" | {len(vals)} |"
            )

    lines += [
        "",
        "## AOSS screening and post-hoc stitchability",
        "",
        "| Target | Rank | Config | Mean AOSS | Post-hoc mean image AUROC |",
        "|---|---:|---|---:|---:|",
    ]

    for target_name in cfg["screen"]["target_names"]:
        subset = [r for r in screen if r["target_backbone"] == target_name]
        for row in subset:
            auroc = row["posthoc_mean_image_auroc"]
            auroc_text = f"{auroc:.4f}" if np.isfinite(auroc) else "N/A"
            lines.append(
                f"| {target_name} | {row['aoss_rank_within_target']} | "
                f"{row['config']} | {row['mean_aoss']:.4f} | {auroc_text} |"
            )
        valid = [
            (row["mean_aoss"], row["posthoc_mean_image_auroc"])
            for row in subset
            if np.isfinite(row["mean_aoss"]) and np.isfinite(row["posthoc_mean_image_auroc"])
        ]
        if len(valid) >= 3:
            rho, pval = spearmanr([x[0] for x in valid], [x[1] for x in valid])
            lines.append(
                f"\n- {target_name}: AOSS vs target-test AUROC "
                f"Spearman rho={rho:.4f}, p={pval:.4g} (post-hoc only)."
            )

    selected_text = ", ".join(x["config"] for x in payload.get("selected", [])) or "N/A"
    lines += [
        "",
        "## Selected configurations",
        "",
        selected_text,
        "",
        "See final_results.csv for per-dataset/per-seed results and "
        "ablation_results.csv for primary-medical-target ablations.",
    ]
    (results_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:18]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--stage", default="all",
        choices=["all", "prepare", "cache", "screen", "final", "stitchmap", "ablation", "report"],
    )
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    for key in ("data_dir", "model_dir", "cache_dir", "results_dir"):
        ensure_dir(cfg["paths"][key])

    gpus = gpu_ids(int(cfg["project"].get("max_gpus", 4)))
    print(f"[runtime] GPUs: {gpus}")

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
