"""CPU post-hoc screening response/selection diagnostics. Never writes a selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .analyze_paired_response import MEASURES, sha
from .screen_response_io import COMPONENTS, MAPPING, MODALITIES, grid_jobs, require, validate_screen


SIGN_MEASURES = ["net_positive_fraction", "net_negative_fraction", "negative_contrast_delta_fraction"]


def summarize_grid(frame, extra=()):
    f = frame.copy()
    f["net_positive_fraction"] = (f.net_local_response > 0).astype(float)
    f["net_negative_fraction"] = (f.net_local_response < 0).astype(float)
    f["negative_contrast_delta_fraction"] = (f.contrast_delta < 0).astype(float)
    measures = MEASURES + SIGN_MEASURES
    keys = ["job", "pair", "source_stage", "target_block", "held_out_modality", *extra]
    ds = f.groupby(keys + ["source_modality", "dataset"], as_index=False)[measures].mean()
    mod = ds.groupby(keys + ["source_modality"], as_index=False)[measures].mean()
    fold = mod.groupby(keys, as_index=False)[measures].mean()
    point = fold.groupby(["job", "pair", "source_stage", "target_block", *extra], as_index=False)[measures].mean()
    return ds, mod, fold, point


def validate_frame(f, job, held):
    for key, value in job.items():
        require(key in f and (f[key] == value).all(), f"Response identity differs: {key}")
    require((f.held_out_modality == held).all() and (f.source_modality != held).all(), "Held-out contamination")
    require(set(f.dataset) == {d for d, m in MAPPING.items() if m != held}
            and (f.source_modality == f.dataset.map(MAPPING)).all(), "Incomplete source modality matrix")
    require((f.groupby("dataset").size() == 256).all()
            and not f.duplicated(["dataset", "record_index"]).any(), "Missing or duplicate source rows")
    for _, ds in f.groupby("dataset"):
        require(set(ds.record_index) == set(range(256)) and ds.image_path.nunique() == 256, "Source records differ")
        require((ds.perturbation_kind_inferred == np.array(["intensity", "blur", "patch_copy"])[
            (2026 + ds.record_index.to_numpy(dtype=int)) % 3]).all(), "Inferred perturbation type differs")
    require(((f.n_mask_in + f.n_mask_out) == 256).all()
            and (f.n_mask_in >= 0).all() and (f.n_mask_out >= 0).all(), "Mask geometry differs")
    valid = (f.n_mask_in > 0) & (f.n_mask_out > 0)
    require((valid == f.valid_regions).all(), "Region validity differs")
    numeric = MEASURES + ["normal_mean", "perturb_mean", "normal_contrast_topk", "perturb_contrast_topk"]
    require(np.isfinite(f.loc[valid, numeric]).all().all(), "Nonfinite response")
    require(f.loc[~valid, ["normal_spatial_contrast", "raw_local_sensitivity", "net_local_response"]].isna().all().all(),
            "Undefined region filled with zero")
    for value, a, b in [("normal_spatial_contrast", "normal_in", "normal_out"),
                        ("raw_local_sensitivity", "perturb_in", "perturb_out"),
                        ("delta_in", "perturb_in", "normal_in"), ("delta_out", "perturb_out", "normal_out"),
                        ("net_local_response", "delta_in", "delta_out"),
                        ("mean_delta", "perturb_mean", "normal_mean"),
                        ("contrast_delta", "perturb_contrast_topk", "normal_contrast_topk")]:
        require(np.allclose(f.loc[valid, value], f.loc[valid, a] - f.loc[valid, b], atol=1e-8, rtol=0), "Paired arithmetic differs")
    require(np.allclose(f.loc[valid, "raw_local_sensitivity"], f.loc[valid, "normal_spatial_contrast"]
                        + f.loc[valid, "net_local_response"], atol=1e-8, rtol=0), "Response decomposition differs")


def selector_diagnostics(table, selected):
    """Expected uniform-random regret and fixed original selection; no new selector."""
    rows = []
    locked = {x["source_backbone"][0].upper() + x["target_backbone"][0].upper(): x["config"]
              for x in selected["selected"]}
    for (pair, held), f in table.groupby(["pair", "held_out_modality"]):
        require(len(f) == 9 and f.job.nunique() == 9, "Incomplete grid for selector diagnostics")
        f = f.sort_values("job")
        performance = f.image_auroc
        original = f[f.job == locked[pair]].iloc[0]
        maximum = performance.max()
        ranks = performance.rank(ascending=False, method="average")
        rows.append(dict(pair=pair, held_out_modality=held, n_points=9,
            locked_config=locked[pair], locked_auroc=original.image_auroc,
            locked_rank=float(ranks.loc[original.name]), oracle_upper_bound=maximum,
            locked_regret=maximum - original.image_auroc,
            random_expected_auroc=performance.mean(), random_expected_regret=maximum - performance.mean(),
            locked_minus_random=original.image_auroc - performance.mean(),
            boundary="original_global_five_fold_AOSS; not_fold_exclusive_selection"))
    return pd.DataFrame(rows)


def correlate(table):
    rows = []
    for (pair, held), f in table.groupby(["pair", "held_out_modality"]):
        for c in ["aoss", "normal_discrepancy", "local_sensitivity", "train_loss",
                  "normal_spatial_contrast", "net_local_response", "contrast_delta"]:
            rows.append(dict(pair=pair, held_out_modality=held, diagnostic=c, n_points=len(f),
                             spearman_rho=f[c].rank().corr(f.image_auroc.rank())
                             if f[c].nunique() > 1 and f.image_auroc.nunique() > 1 else np.nan))
    return pd.DataFrame(rows)


def load_test_metadata(root, jobs, cfg, digests):
    rows = []
    for job in jobs:
        path = root / "results/stitchmap" / f"{job['job']}.json"
        require(path.is_file(), "Need existing original stitchmap JSONs for post-hoc comparison; no test rerun")
        digests[str(path)] = sha(path)
        ref = json.loads(path.read_text(encoding="utf-8"))
        require(all(ref.get(k) == job[k] for k in ["config", "source_backbone", "target_backbone"]), "Stitchmap identity differs")
        require(len(ref["rows"]) == 6 and {r["dataset"] for r in ref["rows"]} == set(MAPPING), "Stitchmap incomplete/duplicate")
        for r in ref["rows"]:
            require(all(r.get(k) == job[k] for k in ["source_backbone", "target_backbone", "source_stage", "target_block", "seed"])
                    and r["score_mode"] == "contrast_topk" and r["target_modality"] == MAPPING[r["dataset"]]
                    and np.isfinite(r["image_auroc"]) and np.isfinite(r["image_aupr"]), "Stitchmap metric identity differs")
            require(r["n_test"] > 0, "Invalid stitchmap sample count")
            rows.append(dict(job=job["job"], held_out_modality=r["target_modality"], dataset=r["dataset"],
                             image_auroc=r["image_auroc"], image_aupr=r["image_aupr"]))
    # OCT datasets first averaged inside their single held-out modality.
    return pd.DataFrame(rows).groupby(["job", "held_out_modality"], as_index=False)[["image_auroc", "image_aupr"]].mean()


def analyze(root, inp):
    mp = inp / "run_manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    require(m["status"] == "complete" and m["protocol"] == "original_screen_diagnostic"
            and m["original_inputs_unchanged"] and m["no_training"] and m["no_reselection"]
            and not m["target_test_read"] and not m["target_valid_read"], "Incomplete or changed protocol")
    require(m["original_score"] == "contrast_topk" and m["topk_fraction"] == .05
            and m["batch_size"] == 256 and m["replay_tolerance"] == 1e-6, "Original scoring/replay settings changed")
    config = inp / "references/experiment.yaml"
    selected_path = inp / "references/selected_configs.json"
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    jobs = grid_jobs(cfg, selected)
    require(m["jobs"] == jobs, "Audit must contain all 36 original points in the fixed plan")
    expected = {"aoss_replay.csv", "references/experiment.yaml", "references/selected_configs.json"}
    expected.update(f"references/screen/{j['job']}.json" for j in jobs)
    expected.update(f"jobs/{j['job']}/source_{h}.csv" for j in jobs for h in MODALITIES)
    require(set(m["output_sha256"]) == expected, "Missing or unexpected audit outputs")
    digests = {str(mp): sha(mp)}
    for name, digest in m["output_sha256"].items():
        p = (inp / name).resolve()
        require(p.is_relative_to(inp.resolve()) and p.is_file() and sha(p) == digest, f"Changed output: {name}")
        digests[str(p)] = digest
    require(sha(config) == m["original_sha256"]["configs/experiment.yaml"]
            and sha(selected_path) == m["original_sha256"]["results/selected_configs.json"], "Copied config/selection differs")
    frames, components = [], []
    for job in jobs:
        p = inp / "references/screen" / f"{job['job']}.json"
        require(sha(p) == m["original_sha256"][f"results/screen/{job['job']}.json"], "Copied screen reference differs")
        ref = json.loads(p.read_text(encoding="utf-8"))
        validate_screen(ref, job)
        components += [dict(job=job["job"], held_out_modality=r["target_modality"],
                            **{k: r[k] for k in COMPONENTS + ["train_loss"]}) for r in ref["aoss_rows"]]
        for h in MODALITIES:
            f = pd.read_csv(inp / "jobs" / job["job"] / f"source_{h}.csv")
            validate_frame(f, job, h)
            frames.append(f.assign(pair=job["source_backbone"][0].upper() + job["target_backbone"][0].upper()))
    f = pd.concat(frames, ignore_index=True)
    require(len(f) == m["n_source_rows"] == 221184, "Incomplete source response rows")
    replay = pd.read_csv(inp / "aoss_replay.csv")
    require(len(replay) == 900 and replay.within_tolerance.eq(True).all() and np.isfinite(replay.delta).all()
            and (replay.delta.abs() <= 1e-6).all(), "Original screening AOSS replay failed")
    expected_checks = {(j["job"], h, c) for j in jobs for h in MODALITIES for c in COMPONENTS}
    require(not replay.duplicated(["job", "held_out_modality", "component"]).any()
            and set(replay[["job", "held_out_modality", "component"]].itertuples(index=False, name=None)) == expected_checks, "Replay matrix differs")
    component = pd.DataFrame(components)
    lookup = component.set_index(["job", "held_out_modality"])
    for r in replay.itertuples():
        require(abs(r.original - lookup.loc[(r.job, r.held_out_modality), r.component]) < 1e-12
                and abs(r.delta - (r.replayed - r.original)) < 1e-12, "Replay reference arithmetic differs")
    out = root / "results/analysis/screen_response_audit"
    out.mkdir(parents=True, exist_ok=True)
    valid = f[f.valid_regions].copy()
    for label, extra in [("all", ()), ("by_kind", ("perturbation_kind_inferred",))]:
        for name, table in zip(["dataset", "modality", "fold", "point"], summarize_grid(valid, extra)):
            table.to_csv(out / f"{label}_{name}.csv", index=False)
    f.groupby(["job", "held_out_modality", "dataset", "perturbation_kind_inferred"], as_index=False).agg(
        n=("record_index", "size"), valid_regions=("valid_regions", "sum")).to_csv(out / "counts.csv", index=False)
    ds, mod, fold, point = summarize_grid(valid)
    metrics = load_test_metadata(root, jobs, cfg, digests)
    table = fold.merge(component, on=["job", "held_out_modality"], validate="one_to_one").merge(
        metrics, on=["job", "held_out_modality"], validate="one_to_one")
    require(len(table) == 180, "Incomplete source/real-anomaly join")
    table["target_domain_group"] = np.where(table.held_out_modality.isin(
        ["brain_mri", "liver_ct", "chest_xray"]), "radiology", "non_radiology")
    table.to_csv(out / "posthoc_join.csv", index=False)
    correlations = correlate(table)
    correlations.to_csv(out / "posthoc_correlations.csv", index=False)
    selector_diagnostics(table, selected).to_csv(out / "original_selection_vs_random.csv", index=False)
    macro_measures = COMPONENTS + ["train_loss", "normal_spatial_contrast", "net_local_response", "contrast_delta",
                                  "image_auroc", "image_aupr"]
    macro = table.groupby(["job", "pair"], as_index=False)[macro_measures].mean().assign(
        held_out_modality="overall_modality_macro")
    domain = table.groupby(["job", "pair", "target_domain_group"], as_index=False)[macro_measures].mean()
    domain["held_out_modality"] = domain.target_domain_group + "_modality_macro"
    macro = pd.concat([macro, domain], ignore_index=True)
    correlate(macro).to_csv(out / "posthoc_macro_correlations.csv", index=False)
    selector_diagnostics(macro, selected).to_csv(out / "original_selection_vs_random_macro.csv", index=False)
    # Point-matched 2x2 effects: all nine locations, one screen seed; no seed CI.
    effects = []
    weights = dict(source_with_medical_target=[1, 0, -1, 0], source_with_general_target=[0, 1, 0, -1],
                   target_with_medical_source=[1, -1, 0, 0], target_with_general_source=[0, 0, 1, -1],
                   source_average=[.5, .5, -.5, -.5], target_average=[.5, -.5, .5, -.5],
                   interaction=[1, -1, -1, 1])
    for (stage, block, held), g in table.groupby(["source_stage", "target_block", "held_out_modality"]):
        wide = g.set_index("pair").reindex(["MM", "MG", "GM", "GG"])
        for measure in ["net_local_response", "normal_spatial_contrast", "aoss", "image_auroc"]:
            for effect, w in weights.items():
                effects.append(dict(source_stage=stage, target_block=block, held_out_modality=held,
                                    measure=measure, effect=effect, value=float(np.dot(wide[measure], w))))
    pd.DataFrame(effects).to_csv(out / "point_matched_effects.csv", index=False)
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for ax, pair in zip(axes, ["MM", "MG", "GM", "GG"]):
        g = table[table.pair == pair]
        for held in MODALITIES:
            v = g[g.held_out_modality == held]
            ax.scatter(v.net_local_response, v.image_auroc, s=20, label=held)
        ax.axhline(.5, color="grey", ls="--", lw=.8)
        ax.set(title=pair, xlabel="Source net response")
    axes[0].set_ylabel("Held-out modality AUROC")
    axes[-1].legend(fontsize=6, loc="best")
    fig.suptitle("Original screening grid; post-hoc association, no reselection", fontsize=10)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(out / f"response_vs_real_auroc.{suffix}", dpi=180)
    plt.close(fig)
    lines = ["# Original screening response / selection audit", "",
        "36 original screening points, 180 checkpoints, 221184 paired source rows; 900 original AOSS checks passed.",
        "No training, no reselection, no target test/valid cache access during GPU audit.",
        "Existing stitchmap aggregate target metrics are read only by this post-hoc CPU analysis.",
        "", "- Equal datasets within source modality, equal source modalities within fold. OCT is one modality.",
        "- Post-hoc target macro tables weight five target modalities equally; radiology/non-radiology use three/two.",
        "  These diagnostic weights differ from the six-dataset main table; main results remain unchanged.",
        "- All nine locations retained; screening seed 11 / 4 epochs / 500 normal images per source modality only.",
        "- Same input images recur across folds; no independent-patient or seed uncertainty claim.",
        "- Net response is diagnostic. Correlations are descriptive; no proxy is fitted or promoted to a selector.",
        "- Random comparator is the exact uniform expectation over nine points, not another trained run.",
        "- Oracle is a post-hoc upper bound; original selected_configs.json remains unchanged.",
        "- Original global mean AOSS aggregates five folds: excluded modality appears in other folds' source data.",
        "  Target test labels were not used, but global selection is not fold-exclusive. This boundary is retained.",
        "- Point-matched effects control stitch location; they do not remove pretraining data/objective/recipe confounds.",
        "- Perturbation kind is inferred. Synthetic response is not clinical anomaly sensitivity.",
        "", "See posthoc_correlations.csv, original_selection_vs_random.csv and point_matched_effects.csv."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    require(all(sha(Path(p)) == digest for p, digest in digests.items()), "Input changed during analysis")
    (out / "audit.json").write_text(json.dumps(dict(status="complete", n_rows=len(f), n_invalid_regions=int((~f.valid_regions).sum()),
        aoss_replay_checks=900, max_replay_abs_delta=float(replay.delta.abs().max()),
        n_posthoc_cells=len(table), selection_updated=False, test_cache_read=False,
        existing_target_metrics_read=True, input_sha256=digests), indent=2), encoding="utf-8")
    print(out / "report.md")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--input", type=Path, default=Path("results/followup/screen_response_audit"))
    args = p.parse_args()
    root = args.root.resolve()
    analyze(root, (root / args.input).resolve())


if __name__ == "__main__":
    main()
