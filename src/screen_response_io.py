"""Locked screening identities for a read-only response audit; no model imports."""
from __future__ import annotations

import math
from pathlib import Path

from .followup_io import common_jobs, name, read_json


MODALITIES = ["brain_mri", "chest_xray", "liver_ct", "oct", "pathology"]
MAPPING = dict(Brain="brain_mri", liver="liver_ct", RESC="oct", OCT2017="oct",
               RSNA="chest_xray", camelyon16="pathology")
COMPONENTS = ["normal_discrepancy", "perturb_in", "perturb_out", "local_sensitivity", "aoss"]
SCREEN_BUDGET = dict(seed=11, n_per_source_modality=500, epochs=4, batch_size=64,
                     lr=.002, weight_decay=.0001, adapter="mlp", topk_fraction=.05)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def grid_jobs(cfg, selected):
    common_jobs(cfg, selected)  # Existing main-protocol guard, including locked final points.
    require(all(cfg["screen"].get(k) == v for k, v in SCREEN_BUDGET.items()),
            "Keep the original screening budget/seed; never use final checkpoints for the grid")
    return [dict(job=name(s, t, stage, block), config=name(s, t, stage, block),
                 source_backbone=s, target_backbone=t, source_stage=stage,
                 target_block=block, seed=11)
            for s, t in cfg["models"]["pairs"] for stage in [1, 2, 3] for block in [3, 6, 9]]


def validate_screen(result, job):
    identity = {k: job[k] for k in ["config", "source_backbone", "target_backbone",
                                   "source_stage", "target_block", "seed"]}
    identity["adapter"] = "mlp"
    require(all(result.get(k) == v for k, v in identity.items()), "Screen identity differs")
    require(result.get("rows") == [], "Screen contains target evaluation; original source-only screen required")
    rows = result.get("aoss_rows", [])
    require(len(rows) == 5 and {r["target_modality"] for r in rows} == set(MODALITIES),
            "Incomplete or duplicate source-only screening folds")
    for r in rows:
        require(all(math.isfinite(r[c]) for c in COMPONENTS + ["train_loss"]), "Nonfinite screen component")
        require(abs(r["local_sensitivity"] - (r["perturb_in"] - r["perturb_out"])) < 1e-10
                and abs(r["aoss"] - r["local_sensitivity"] / (r["normal_discrepancy"] + 1e-8)) < 1e-10,
                "Original screening AOSS arithmetic differs")


def screen_plan(cfg, selected):
    jobs = grid_jobs(cfg, selected)
    directory = Path(cfg["paths"]["results_dir"]) / "screen"
    require({p.stem for p in directory.glob("*.json")} == {j["job"] for j in jobs},
            "Need exactly the original 36 screening JSONs")
    protected = []
    scores = {}
    for job in jobs:
        path = directory / f"{job['job']}.json"
        result = read_json(path)
        validate_screen(result, job)
        scores[job["job"]] = sum(r["aoss"] for r in result["aoss_rows"]) / 5
        protected.append(path)
        for held in MODALITIES:
            ckpt = directory / "checkpoints" / job["job"] / f"{held}.pt"
            require(ckpt.is_file(), f"Missing original screening checkpoint: {ckpt}; no retraining")
            protected.append(ckpt)
    for item in selected["selected"]:
        pair = [j for j in jobs if (j["source_backbone"], j["target_backbone"]) ==
                (item["source_backbone"], item["target_backbone"])]
        require(abs(scores[item["config"]] - max(scores[j["job"]] for j in pair)) < 1e-12,
                "Existing locked selection differs from original five-fold mean AOSS; do not reselect")
    return jobs, protected
