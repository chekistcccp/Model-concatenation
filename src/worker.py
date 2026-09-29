from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from .common import config_name, configure_torch, ensure_dir, load_yaml, seed_everything, write_json
from .train_eval import (
    compute_aoss,
    evaluate_cached_baselines,
    evaluate_dataset,
    modality_datasets,
    target_modalities,
    train_adapter,
)


def load_vit(cfg: dict, device: torch.device):
    vit = AutoModel.from_pretrained(
        cfg["models"]["vit_local"], local_files_only=True, torch_dtype=torch.float16
    ).to(device).eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    return vit


def run_stitch_job(cfg: dict, args) -> dict:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    configure_torch(bool(cfg["project"].get("allow_tf32", True)))
    seed_everything(args.seed, bool(cfg["runtime"].get("deterministic", False)))
    vit = load_vit(cfg, device)
    groups = modality_datasets(cfg)
    rows = []
    aoss_rows = []

    for target_mod in target_modalities(cfg):
        adapter, tail, history = train_adapter(
            cfg=cfg, vit=vit, target_modality=target_mod,
            source_stage=args.source_stage, target_block=args.target_block,
            adapter_type=args.adapter, n_per_modality=args.n_per_modality,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, weight_decay=args.weight_decay,
            seed=args.seed, device=device,
        )
        aoss = compute_aoss(cfg, adapter, tail, target_mod, args.source_stage, device)
        if args.save_checkpoints:
            ckpt_dir = Path(args.output).parent / "checkpoints" / Path(args.output).stem
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"adapter": adapter.state_dict(), "source_stage": args.source_stage, "target_block": args.target_block, "adapter_type": args.adapter, "target_modality": target_mod}, ckpt_dir / f"{target_mod}.pt")
        aoss_rows.append({"target_modality": target_mod, **aoss, "train_loss": history[-1]})
        if not args.skip_target_eval:
            for ds in groups[target_mod]:
                metrics = evaluate_dataset(
                    cfg, adapter, tail, ds, args.source_stage, device,
                    topk_fraction=args.topk_fraction,
                    score_modes=args.score_modes,
                    batch_size=max(args.batch_size, 128),
                )
                for row in metrics:
                    row.update({
                        "target_modality": target_mod,
                        "source_stage": args.source_stage,
                        "target_block": args.target_block,
                        "adapter": args.adapter,
                        "seed": args.seed,
                        "n_per_source_modality": args.n_per_modality,
                        "epochs": args.epochs,
                        "final_train_loss": history[-1],
                        "aoss": aoss["aoss"],
                        "normal_discrepancy": aoss["normal_discrepancy"],
                        "local_sensitivity": aoss["local_sensitivity"],
                    })
                    rows.append(row)
        del adapter, tail
        torch.cuda.empty_cache()

    finite = [r["image_auroc"] for r in rows if np.isfinite(r["image_auroc"]) and r["score_mode"] == args.score_modes[0]]
    return {
        "config": config_name(args.source_stage, args.target_block),
        "source_stage": args.source_stage,
        "target_block": args.target_block,
        "adapter": args.adapter,
        "seed": args.seed,
        "mean_image_auroc": float(np.mean(finite)) if finite else float("nan"),
        "rows": rows,
        "aoss_rows": aoss_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["stitch", "baseline", "eval_saved"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source-stage", type=int, default=2)
    ap.add_argument("--target-block", type=int, default=6)
    ap.add_argument("--adapter", default="mlp")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-per-modality", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--topk-fraction", type=float, default=0.05)
    ap.add_argument("--score-modes", nargs="+", default=["contrast_topk"])
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--skip-target-eval", action="store_true")
    ap.add_argument("--checkpoint-root", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    ensure_dir(Path(args.output).parent)

    if args.mode == "baseline":
        rows = []
        for ds in cfg["data"]["datasets"]:
            rows.extend(evaluate_cached_baselines(cfg, ds, args.topk_fraction))
        write_json(args.output, {"rows": rows})
    elif args.mode == "eval_saved":
        if not args.checkpoint_root:
            raise ValueError("--checkpoint-root is required for eval_saved")
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        configure_torch(bool(cfg["project"].get("allow_tf32", True)))
        vit = load_vit(cfg, device)
        groups = modality_datasets(cfg)
        rows = []
        for target_mod in target_modalities(cfg):
            ckpt_path = Path(args.checkpoint_root) / f"{target_mod}.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            adapter, tail = __import__("src.models", fromlist=["build_stitch_modules"]).build_stitch_modules(
                cfg, vit, args.source_stage, args.target_block, args.adapter
            )
            adapter.load_state_dict(ckpt["adapter"], strict=True)
            adapter = adapter.to(device).eval(); tail = tail.to(device).eval()
            for ds in groups[target_mod]:
                metrics = evaluate_dataset(
                    cfg, adapter, tail, ds, args.source_stage, device,
                    topk_fraction=args.topk_fraction, score_modes=args.score_modes,
                    batch_size=max(args.batch_size, 128),
                )
                for row in metrics:
                    row.update({"target_modality": target_mod, "source_stage": args.source_stage, "target_block": args.target_block, "adapter": args.adapter, "seed": args.seed})
                    rows.append(row)
            del adapter, tail
            torch.cuda.empty_cache()
        finite = [r["image_auroc"] for r in rows if np.isfinite(r["image_auroc"]) and r["score_mode"] == args.score_modes[0]]
        write_json(args.output, {"config": config_name(args.source_stage, args.target_block), "mean_image_auroc": float(np.mean(finite)) if finite else float("nan"), "rows": rows})
    else:
        result = run_stitch_job(cfg, args)
        write_json(args.output, result)


if __name__ == "__main__":
    main()
