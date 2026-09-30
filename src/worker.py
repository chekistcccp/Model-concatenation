from __future__ import annotations

import argparse
import importlib.metadata
import platform
from pathlib import Path

import numpy as np
import torch

from .common import config_name, configure_torch, ensure_dir, load_yaml, seed_everything, write_json, read_json
from .prediction_io import checkpoint_identity, compare_metrics, sha256, validate_final
from .models import build_stitch_modules, load_target
from .train_eval import (
    compute_aoss,
    domain_group,
    evaluate_cached_baselines,
    evaluate_dataset,
    modality_datasets,
    target_modalities,
    train_adapter,
)


def run_stitch_job(cfg, args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    configure_torch(bool(cfg["project"].get("allow_tf32", True)))
    seed_everything(args.seed, bool(cfg["runtime"].get("deterministic", False)))
    target = load_target(cfg, args.target_name, device)
    groups = modality_datasets(cfg)
    rows, aoss_rows = [], []

    for target_mod in target_modalities(cfg):
        adapter, tail, history = train_adapter(
            cfg, target, args.source_name, args.target_name, target_mod,
            args.source_stage, args.target_block, args.adapter,
            args.n_per_modality, args.epochs, args.batch_size,
            args.lr, args.weight_decay, args.seed, device,
        )
        aoss = compute_aoss(
            cfg, target, args.source_name, args.target_name,
            adapter, tail, target_mod, args.source_stage, device
        )
        if args.save_checkpoints:
            ckpt_dir = Path(args.output).parent / "checkpoints" / Path(args.output).stem
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "adapter": adapter.state_dict(),
                "source_name": args.source_name,
                "target_name": args.target_name,
                "source_stage": args.source_stage,
                "target_block": args.target_block,
                "adapter_type": args.adapter,
                "target_modality": target_mod,
            }, ckpt_dir / f"{target_mod}.pt")

        aoss_rows.append({
            "target_modality": target_mod,
            "domain_group": domain_group(target_mod),
            **aoss,
            "train_loss": history[-1],
        })

        if not args.skip_target_eval:
            for ds in groups[target_mod]:
                metrics = evaluate_dataset(
                    cfg, target, args.source_name, args.target_name,
                    adapter, tail, ds, args.source_stage, device,
                    args.topk_fraction, args.score_modes, max(args.batch_size, 128)
                )
                for row in metrics:
                    row.update({
                        "target_modality": target_mod,
                        "domain_group": domain_group(target_mod),
                        "source_stage": args.source_stage,
                        "target_block": args.target_block,
                        "adapter": args.adapter,
                        "seed": args.seed,
                        "n_per_source_modality": args.n_per_modality,
                        "epochs": args.epochs,
                        "final_train_loss": history[-1],
                        "aoss": aoss["aoss"],
                    })
                    rows.append(row)
        del adapter, tail
        torch.cuda.empty_cache()

    finite = [
        r["image_auroc"] for r in rows
        if np.isfinite(r["image_auroc"]) and r["score_mode"] == args.score_modes[0]
    ]
    return {
        "config": config_name(args.source_name, args.target_name, args.source_stage, args.target_block),
        "source_backbone": args.source_name,
        "target_backbone": args.target_name,
        "source_stage": args.source_stage,
        "target_block": args.target_block,
        "adapter": args.adapter,
        "seed": args.seed,
        "mean_image_auroc": float(np.mean(finite)) if finite else float("nan"),
        "rows": rows,
        "aoss_rows": aoss_rows,
    }


def run_saved_eval(cfg, args):
    exporting = bool(args.prediction_dir)
    reference = None
    selected_path = Path(cfg["paths"]["results_dir"]) / "selected_configs.json"
    if exporting:
        if not args.reference_final:
            raise ValueError("Prediction export requires the original final JSON")
        reference = read_json(args.reference_final)
        validate_final(cfg, read_json(selected_path), reference, args.source_name, args.target_name,
                       args.source_stage, args.target_block, args.seed, args.adapter, args.topk_fraction, args.score_modes)
        expected_job = f"{reference['config']}_seed{args.seed}"
        expected_root = Path(cfg["paths"]["results_dir"]) / "final" / "checkpoints" / expected_job
        if Path(args.checkpoint_root).resolve() != expected_root.resolve():
            raise ValueError("Prediction export must load the original final checkpoint directory")
        if Path(args.reference_final).resolve() != (expected_root.parent.parent / f"{expected_job}.json").resolve():
            raise ValueError("Prediction reference must be the original final job JSON")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if exporting:
        configure_torch(bool(cfg["project"].get("allow_tf32", True)))
        seed_everything(args.seed, bool(cfg["runtime"].get("deterministic", False)))
    target = load_target(cfg, args.target_name, device)
    groups = modality_datasets(cfg)
    rows, provenance = [], []
    for target_mod in target_modalities(cfg):
        ckpt_path = Path(args.checkpoint_root) / f"{target_mod}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        checkpoint_identity(ckpt, args.source_name, args.target_name, args.source_stage, args.target_block, args.adapter, target_mod)
        adapter, tail = build_stitch_modules(
            cfg, target, args.source_name, args.target_name,
            args.source_stage, args.target_block, args.adapter
        )
        adapter.load_state_dict(ckpt["adapter"], strict=True)
        adapter, tail = adapter.to(device).eval(), tail.to(device).eval()
        for ds in groups[target_mod]:
            export_kwargs = {}
            if exporting:
                metadata = dict(dataset=ds, modality=target_mod, domain_group=domain_group(target_mod),
                                pair=f"{args.source_name}_to_{args.target_name}", source_backbone=args.source_name,
                                target_backbone=args.target_name, seed=args.seed, config=reference["config"],
                                source_stage=args.source_stage, target_block=args.target_block, adapter=args.adapter,
                                score_mode="contrast_topk", topk_fraction=args.topk_fraction)
                export_kwargs = dict(prediction_path=Path(args.prediction_dir) / f"{ds}.csv", prediction_metadata=metadata, evaluate_pixels=False)
            metrics = evaluate_dataset(
                cfg, target, args.source_name, args.target_name,
                adapter, tail, ds, args.source_stage, device,
                args.topk_fraction, args.score_modes, max(args.batch_size, 128), **export_kwargs
            )
            for row in metrics:
                row.update({
                    "target_modality": target_mod,
                    "domain_group": domain_group(target_mod),
                    "source_stage": args.source_stage,
                    "target_block": args.target_block,
                    "seed": args.seed,
                })
                rows.append(row)
            if exporting:
                records_path = Path(cfg["paths"]["cache_dir"]) / "features" / ds / "test" / "records.jsonl"
                provenance.append(dict(dataset=ds, checkpoint=str(ckpt_path), checkpoint_sha256=sha256(ckpt_path),
                                       records_sha256=sha256(records_path), prediction_sha256=sha256(Path(args.prediction_dir) / f"{ds}.csv")))
        del adapter, tail
        torch.cuda.empty_cache()
    finite = [r["image_auroc"] for r in rows if np.isfinite(r["image_auroc"])]
    payload = {
        "config": config_name(args.source_name, args.target_name, args.source_stage, args.target_block),
        "source_backbone": args.source_name,
        "target_backbone": args.target_name,
        "mean_image_auroc": float(np.mean(finite)) if finite else float("nan"),
        "rows": rows,
    }
    if exporting:
        comparisons = compare_metrics(rows, reference["rows"], args.replay_tolerance)
        versions = {p: importlib.metadata.version(p) for p in ("torch", "numpy", "transformers", "timm", "scikit-learn")}
        payload.update(seed=args.seed, posthoc_only=True, no_reselection=True,
                       checkpoint_seed_evidence="legacy checkpoints omit seed; seed checked by original final JSON and checkpoint directory",
                       comparison=comparisons, replay_matches_original=all(r["within_tolerance"] for r in comparisons),
                       replay_tolerance=args.replay_tolerance, provenance=provenance, versions=versions,
                       python=platform.python_version(), config_sha256=sha256(args.config),
                       selected_configs_sha256=sha256(selected_path), original_final_sha256=sha256(args.reference_final))
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["stitch", "baseline", "eval_saved"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source-name", default=None)
    ap.add_argument("--target-name", default=None)
    ap.add_argument("--source-stage", type=int, default=2)
    ap.add_argument("--target-block", type=int, default=6)
    ap.add_argument("--adapter", default="mlp")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-per-modality", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--topk-fraction", type=float, default=0.05)
    ap.add_argument("--score-modes", nargs="+", default=["contrast_topk"])
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--skip-target-eval", action="store_true")
    ap.add_argument("--checkpoint-root")
    ap.add_argument("--prediction-dir")
    ap.add_argument("--reference-final")
    ap.add_argument("--replay-tolerance", type=float, default=1e-6)
    args = ap.parse_args()
    if args.prediction_dir and args.mode != "eval_saved":
        ap.error("--prediction-dir is allowed only for eval_saved")
    if args.replay_tolerance < 0 or not np.isfinite(args.replay_tolerance):
        ap.error("--replay-tolerance must be finite and nonnegative")

    cfg = load_yaml(args.config)
    args.source_name = args.source_name or cfg["models"]["primary_source"]
    args.target_name = args.target_name or cfg["models"]["primary_target"]
    ensure_dir(Path(args.output).parent)

    if args.mode == "baseline":
        rows = []
        for source_name, target_name in cfg["models"]["pairs"]:
            for ds in cfg["data"]["datasets"]:
                rows.extend(evaluate_cached_baselines(
                    cfg, ds, source_name, target_name, args.topk_fraction
                ))
        write_json(args.output, {"rows": rows})
    elif args.mode == "eval_saved":
        if args.prediction_dir:
            expected_output = Path(cfg["paths"]["results_dir"]) / "predictions" / f"{config_name(args.source_name, args.target_name, args.source_stage, args.target_block)}_seed{args.seed}" / "evaluation.json"
            if Path(args.output).resolve() != expected_output.resolve() or Path(args.prediction_dir).resolve() != expected_output.parent.resolve():
                raise ValueError("Prediction exports must write only their results/predictions/job directory")
        result = run_saved_eval(cfg, args)
        write_json(args.output, result)
        if args.prediction_dir and not result["replay_matches_original"]:
            raise RuntimeError(f"Replay differs from original final; inspect {args.output}; original results preserved")
    else:
        write_json(args.output, run_stitch_job(cfg, args))


if __name__ == "__main__":
    main()
