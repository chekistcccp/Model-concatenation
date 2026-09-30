"""Read-only, CPU-only analysis of the locked experiment; never selects/retrains.

Run: python -m src.analyze_results --root .
Outputs only results/analysis and results/figures. Dependencies: numpy, pandas,
scipy, matplotlib, PyYAML. Raw artifacts are hashed before and after analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata, spearmanr
import yaml

PAIRS = ["MM", "MG", "GM", "GG"]
METRICS = ["image_auroc", "image_aupr", "pixel_auroc", "pixel_aupr"]
EFFECTS = {
    "source_with_medical_target": [1, 0, -1, 0],
    "source_with_general_target": [0, 1, 0, -1],
    "target_with_medical_source": [1, -1, 0, 0],
    "target_with_general_source": [0, 0, 1, -1],
    "source_average": [.5, .5, -.5, -.5],
    "target_average": [.5, -.5, .5, -.5],
    "interaction": [1, -1, -1, 1],
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pair(obj):
    return obj["source_backbone"][0].upper() + obj["target_backbone"][0].upper()


def digest(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def summarize(frame, keys, metrics=METRICS):
    rows = []
    for key, part in frame.groupby(keys, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        for metric in metrics:
            if metric not in part:
                continue
            vals = part[metric].dropna()
            if len(vals):
                rows.append(dict(zip(keys, key), metric=metric, mean=vals.mean(),
                                 sd=vals.std(ddof=1), minimum=vals.min(), maximum=vals.max(),
                                 n_seeds=len(vals)))
    return pd.DataFrame(rows)


def macro(frame, keys):
    # Aggregate within a seed BEFORE describing between-seed variability.
    return frame.groupby(keys + ["pair", "seed"], as_index=False)[METRICS].mean()


def effect_summary(frame, keys):
    rows = []
    for metric in ["image_auroc", "image_aupr"]:
        wide = frame.pivot(index=keys + ["seed"], columns="pair", values=metric)[PAIRS]
        if wide.isna().any().any():
            raise ValueError("Incomplete matched-seed effect cells")
        for name, weights in EFFECTS.items():
            values = wide.mul(weights).sum(axis=1).rename("effect").reset_index()
            for key, part in values.groupby(keys, sort=False):
                if not isinstance(key, tuple):
                    key = (key,)
                rows.append(dict(zip(keys, key), metric=metric, effect=name,
                                 mean=part.effect.mean(), sd=part.effect.std(ddof=1),
                                 minimum=part.effect.min(), maximum=part.effect.max(),
                                 n_seeds=len(part)))
    return pd.DataFrame(rows)


def exact_spearman(x, y, permutations):
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom == 0:
        return np.nan, np.nan
    observed = float(rx @ ry / denom)
    # Two-sided exact permutation of nine configuration ranks, including ties.
    permuted = (ry[permutations] * rx).sum(axis=1) / denom
    return observed, float(np.mean(np.abs(permuted) >= abs(observed) - 1e-12))


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(order), 0, -1)))
    return out


def md(frame, columns, digits=4):
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in frame.iterrows():
        cells = [f"{row[c]:.{digits}f}" if isinstance(row[c], (float, np.floating)) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()
    root = args.root.resolve()
    cfg = yaml.safe_load((root / "configs/experiment.yaml").read_text(encoding="utf-8"))
    results = root / cfg["paths"]["results_dir"]
    out, figs = results / "analysis", results / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    raw_paths = sorted(p for p in results.rglob("*") if p.is_file() and out not in p.parents and figs not in p.parents)
    raw_paths += [root / "configs/experiment.yaml"] + sorted((root / "src").glob("*.py"))
    raw_paths += [p for p in (root / cfg["paths"]["cache_dir"]).glob("*_audit.json")]
    if (root / "run.log").exists():
        raw_paths.append(root / "run.log")
    before = digest(raw_paths)
    checks, anomalies = [], []

    def check(name, ok, detail="", severity="error"):
        checks.append(dict(check=name, status="PASS" if ok else severity.upper(), detail=detail))

    def save(name, frame):
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")

    datasets = cfg["data"]["datasets"]
    modalities = cfg["data"]["modality_map"]
    mods = sorted(set(modalities.values()))
    seeds = cfg["final"]["seeds"]
    grid = {f"{s}_to_{t}_s{stage}_b{block}" for s, t in cfg["models"]["pairs"]
            for stage in cfg["screen"]["source_stages"] for block in cfg["screen"]["target_blocks"]}
    audit_path = root / cfg["paths"]["cache_dir"] / "data_audit.json"
    log = (root / "run.log").read_text(encoding="utf-8", errors="replace") if (root / "run.log").exists() else ""
    data_audit, provenance = {}, "unavailable"
    if audit_path.exists():
        data_audit, provenance = read(audit_path), str(audit_path)
    elif log.lstrip().startswith("{"):
        try:
            candidate, _ = json.JSONDecoder().raw_decode(log.lstrip())
            if set(datasets).issubset(candidate):
                data_audit, provenance = candidate, "run.log leading JSON (not independently verified on server)"
        except ValueError:
            pass
    audit_rows = []
    for ds in datasets:
        info = data_audit.get(ds, {})
        tr, te = info.get("splits", {}).get("train", {}), info.get("splits", {}).get("test", {})
        audit_rows.append(dict(dataset=ds, modality=modalities[ds], train_normal=tr.get("normal"),
                               test_normal=te.get("normal"), test_abnormal=te.get("abnormal"),
                               test_with_mask=te.get("with_mask"), root=info.get("root"), provenance=provenance))
        check(f"data_counts:{ds}", all(v is not None and v > 0 for v in [tr.get("normal"), te.get("normal"), te.get("abnormal")]), provenance, "warning")
    audit_df = pd.DataFrame(audit_rows)
    save("data_integrity", audit_df)
    model_path = root / cfg["paths"]["cache_dir"] / "model_audit.json"
    model_ready = False
    if model_path.exists():
        model = read(model_path)
        model_ready = all(model.get(group, {}).get(k, {}).get("ready", False)
                          for group in ("sources", "targets") for k in ("medical", "general"))
    check("four_models_ready", model_ready, "four READY flags observed (file readiness, not independent weight verification)" if model_ready else "model_audit.json missing or not all READY", "warning")
    check("log_no_traceback", "Traceback (most recent call last)" not in log, "NumPy non-writable tensor and torch_dtype warnings remain in log", "warning")
    tables, aoss_records = {}, []
    objects = {}
    for phase in ("screen", "final", "stitchmap", "ablation"):
        paths = sorted((results / phase).glob("*.json"))
        objects[phase] = []
        all_rows = []
        for path in paths:
            obj = read(path)
            objects[phase].append(obj)
            rows = obj.get("rows", [])
            ident = obj["config"]
            if phase in ("screen", "stitchmap"):
                check(f"filename:{path.name}:{phase}", path.stem == ident)
            if phase == "screen":
                check(f"no_target_evaluation:{ident}", not rows and np.isnan(obj["mean_image_auroc"]), "NaN mean_image_auroc is expected for skipped target evaluation")
                check(f"screen_metadata:{ident}", obj["seed"] == cfg["screen"]["seed"] and obj["adapter"] == cfg["screen"]["adapter"] and ident.endswith(f"_s{obj['source_stage']}_b{obj['target_block']}"))
            if phase in ("final", "stitchmap"):
                check(f"dataset_coverage:{phase}:{path.name}", len(rows) == len(datasets) and {r["dataset"] for r in rows} == set(datasets))
                check(f"mean_consistency:{phase}:{path.name}", np.isclose(obj["mean_image_auroc"], np.mean([r["image_auroc"] for r in rows])))
            for r in rows:
                r = dict(r, config=ident, pair=pair(obj), file=path.name)
                all_rows.append(r)
                check(f"row_identity:{phase}:{path.name}:{r['dataset']}:{r['score_mode']}", pair(r) == pair(obj) and r["target_modality"] == modalities[r["dataset"]])
                expected_group = "radiology" if modalities[r["dataset"]] in {"brain_mri", "liver_ct", "chest_xray"} else "non_radiology"
                check(f"row_config:{phase}:{path.name}:{r['dataset']}:{r['score_mode']}", ident.endswith(f"_s{r['source_stage']}_b{r['target_block']}") and r["domain_group"] == expected_group and r["seed"] == obj.get("seed", cfg["screen"]["seed"]))
                for metric in METRICS:
                    if metric in r:
                        check(f"finite_range:{phase}:{path.name}:{r['dataset']}:{r['score_mode']}:{metric}", np.isfinite(r[metric]) and 0 <= r[metric] <= 1)
                if phase == "final":
                    for metric in ("image_auroc", "pixel_auroc"):
                        if r.get(metric, 1) < .5:
                            anomalies.append(dict(kind="below_chance", phase=phase, file=path.name, dataset=r["dataset"], metric=metric, value=r[metric]))
                if data_audit and "n_test" in r:
                    counts = data_audit[r["dataset"]]["splits"]["test"]
                    check(f"n_test:{phase}:{path.name}:{r['dataset']}:{r['score_mode']}", r["n_test"] == counts["normal"] + counts["abnormal"])
            ar = obj.get("aoss_rows", [])
            if phase != "stitchmap":
                check(f"aoss_coverage:{phase}:{path.name}", len(ar) == len(mods) and {r["target_modality"] for r in ar} == set(mods))
            for r in ar:
                vals = [r[k] for k in ("normal_discrepancy", "perturb_in", "perturb_out", "local_sensitivity", "aoss", "train_loss")]
                check(f"aoss_finite:{phase}:{path.name}:{r['target_modality']}", np.isfinite(vals).all() and 0 <= r["train_loss"] <= 2.2 and r["normal_discrepancy"] > 0)
                check(f"aoss_formula:{phase}:{path.name}:{r['target_modality']}", np.isclose(r["local_sensitivity"], r["perturb_in"] - r["perturb_out"]) and np.isclose(r["aoss"], r["local_sensitivity"] / (r["normal_discrepancy"] + 1e-8)))
                aoss_records.append(dict(r, phase=phase, file=path.name, config=ident, pair=pair(obj)))
        tables[phase] = pd.DataFrame(all_rows)
        if phase in ("screen", "stitchmap"):
            check(f"{phase}_grid", len(paths) == len(grid) and {o["config"] for o in objects[phase]} == grid)
    aoss = pd.DataFrame(aoss_records)
    save("aoss_and_loss_audit", aoss)
    # Descriptive IQR flags within phase/pair; no observation is removed.
    for key, part in aoss.groupby(["phase", "pair"]):
        for metric in ["train_loss", "aoss", "normal_discrepancy"]:
            q1, q3 = part[metric].quantile([.25, .75])
            flagged = part[(part[metric] < q1 - 1.5 * (q3 - q1)) | (part[metric] > q3 + 1.5 * (q3 - q1))]
            for _, row in flagged.iterrows():
                anomalies.append(dict(kind="IQR_flag_not_exclusion", phase=key[0], file=row.file, dataset=row.target_modality, metric=metric, value=row[metric]))
    screen = aoss[aoss.phase == "screen"].groupby(["pair", "config"], as_index=False).aoss.mean()
    selected_payload = read(results / "selected_configs.json")
    selected = {pair(o): o["config"] for o in selected_payload["selected"]}
    check("selection_metric", selected_payload["selection_metric"] == "source_only_AOSS")
    check("selected_coverage", len(selected_payload["selected"]) == 4 and set(selected) == set(PAIRS))
    for p in PAIRS:
        part = screen[screen.pair == p]
        chosen = part[part.config == selected.get(p)]
        check(f"locked_AOSS_top1:{p}", len(chosen) == 1 and np.isclose(chosen.aoss.iloc[0], part.aoss.max()))
        stored = next(o for o in selected_payload["selected"] if pair(o) == p)
        check(f"stored_selection_aoss:{p}", np.isclose(stored["mean_aoss"], chosen.aoss.iloc[0]))
    final = tables["final"]
    check("final_jobs", len(objects["final"]) == 12)
    check("final_cells", len(final) == 72 and not final.duplicated(["pair", "seed", "dataset", "score_mode"]).any())
    for p in PAIRS:
        part = final[final.pair == p]
        check(f"final_selected:{p}", set(part.config) == {selected[p]} and set(part.seed) == set(seeds))
    expected_final = {(p, s, d) for p in PAIRS for s in seeds for d in datasets}
    check("final_exact_cartesian_product", set(zip(final.pair, final.seed, final.dataset)) == expected_final)
    expected_ablation = {(n, "mlp") for n in cfg["ablation"]["calibration_sizes"]} | {(cfg["final"]["n_per_source_modality"], a) for a in cfg["ablation"]["adapter_types"]}
    ab = tables["ablation"]
    ab_keys = ["n_per_source_modality", "adapter", "dataset", "score_mode"]
    check("ablation_cartesian_product", not ab.duplicated(ab_keys).any() and set(ab[ab_keys].itertuples(index=False, name=None)) == {(n, a, d, m) for n, a in expected_ablation for d in datasets for m in cfg["ablation"]["score_modes"]})
    screen_csv = pd.read_csv(results / "screen_summary.csv").set_index("config")
    check("screen_csv_configs", len(screen_csv) == len(grid) and set(screen_csv.index) == grid)
    check("screen_csv_aoss", np.allclose(screen_csv.loc[screen.config, "mean_aoss"], screen.aoss))
    check("final_protocol", set(final.score_mode) == {"contrast_topk"} and set(final.epochs) == {cfg["final"]["epochs"]} and set(final.n_per_source_modality) == {cfg["final"]["n_per_source_modality"]} and set(final.adapter) == {cfg["final"]["adapter"]})
    for phase, name in [("final", "final_results.csv"), ("ablation", "ablation_results.csv")]:
        exported = pd.read_csv(results / name)
        keys = ["dataset", "source_backbone", "target_backbone", "source_stage", "target_block", "seed", "score_mode", "adapter", "n_per_source_modality"]
        left, right = tables[phase].sort_values(keys), exported.sort_values(keys)
        check(f"csv_agrees:{name}", len(left) == len(right) and left[keys].reset_index(drop=True).equals(right[keys].reset_index(drop=True)) and all(np.allclose(left[m], right[m], equal_nan=True) for m in METRICS if m in left))
    save("integrity_checks", pd.DataFrame(checks))
    if any(c["status"] == "ERROR" for c in checks):
        raise ValueError("Integrity failure: inspect results/analysis/integrity_checks.csv; no inferential report generated")
    for metric in METRICS:
        if metric not in final:
            final[metric] = np.nan
    dataset_summary = summarize(final, ["dataset", "pair"])
    prevalence = audit_df[["dataset", "test_normal", "test_abnormal"]].copy()
    prevalence["prevalence"] = prevalence.test_abnormal / (prevalence.test_normal + prevalence.test_abnormal)
    ap_context = dataset_summary[dataset_summary.metric == "image_aupr"].merge(prevalence[["dataset", "prevalence"]], on="dataset")
    ap_context["mean_aupr_minus_prevalence"] = ap_context["mean"] - ap_context.prevalence
    save("aupr_prevalence_context", ap_context)
    mod_seed = macro(final, ["target_modality"])
    dom_seed = macro(final, ["domain_group"])
    overall = macro(final.assign(domain_group="overall"), ["domain_group"])
    dom_seed = pd.concat([dom_seed, overall], ignore_index=True)
    modality_summary = summarize(mod_seed, ["target_modality", "pair"])
    domain_summary = summarize(dom_seed, ["domain_group", "pair"])
    # Supplement: equal modality weight (OCT is one modality, not two).
    mod_seed["domain_group"] = mod_seed.target_modality.map(lambda m: "radiology" if m in {"brain_mri", "liver_ct", "chest_xray"} else "non_radiology")
    dom_mod = pd.concat([macro(mod_seed, ["domain_group"]), macro(mod_seed.assign(domain_group="overall"), ["domain_group"])])
    effects = pd.concat([effect_summary(final, ["dataset"]).rename(columns={"dataset": "scope"}).assign(level="dataset"),
                         effect_summary(mod_seed, ["target_modality"]).rename(columns={"target_modality": "scope"}).assign(level="modality"),
                         effect_summary(dom_seed, ["domain_group"]).rename(columns={"domain_group": "scope"}).assign(level="domain")], ignore_index=True)
    domain_diff = dom_seed[dom_seed.domain_group != "overall"].pivot(index=["pair", "seed"], columns="domain_group", values="image_auroc")
    domain_diff = (domain_diff.radiology - domain_diff.non_radiology).rename("image_auroc").reset_index()
    for name, frame in [("dataset_summary", dataset_summary), ("modality_summary", modality_summary), ("domain_summary", domain_summary), ("domain_modality_weighted_summary", summarize(dom_mod, ["domain_group", "pair"])), ("pretraining_effects", effects), ("domain_difference", summarize(domain_diff, ["pair"], ["image_auroc"])), ("final_seed_metrics", final), ("domain_seed_metrics", dom_seed)]:
        save(name, frame)
    for _, row in dataset_summary[dataset_summary.metric == "image_auroc"].iterrows():
        if row.maximum - row.minimum > .05:
            anomalies.append(dict(kind="seed_range_gt_0.05", phase="final", file=row.pair, dataset=row.dataset, metric=row.metric, value=row.maximum - row.minimum))
    save("anomalies", pd.DataFrame(anomalies))
    sm = tables["stitchmap"].merge(screen.rename(columns={"aoss": "selection_aoss"}), on=["pair", "config"], validate="many_to_one")
    local = aoss[aoss.phase == "screen"][["pair", "config", "target_modality", "aoss"]].rename(columns={"aoss": "fold_aoss"})
    sm = sm.merge(local, on=["pair", "config", "target_modality"], validate="many_to_one")
    save("stitchmap_joined", sm)
    permutations = np.array(list(itertools.permutations(range(9))), dtype=np.int16)
    corr_rows, regrets, rank_rows = [], [], []
    for p in PAIRS:
        part = sm[sm.pair == p]
        scopes = [("overall", "overall", part)]
        scopes += [("domain", g, v) for g, v in part.groupby("domain_group")]
        scopes += [("modality", g, v) for g, v in part.groupby("target_modality")]
        scopes += [("dataset", g, v) for g, v in part.groupby("dataset")]
        for level, scope, sub in scopes:
            points = sub.groupby("config", as_index=False)[["selection_aoss", "fold_aoss", "image_auroc"]].mean()
            points["aoss_rank"] = points.selection_aoss.rank(ascending=False, method="min")
            points["auroc_rank"] = points.image_auroc.rank(ascending=False, method="min")
            chosen = points[points.config == selected[p]].iloc[0]
            best = points.loc[points.image_auroc.idxmax()]
            regrets.append(dict(pair=p, level=level, scope=scope, selected_config=selected[p], selected_auroc=chosen.image_auroc, selected_auroc_rank=chosen.auroc_rank, oracle_config=best.config, oracle_auroc=best.image_auroc, regret=best.image_auroc - chosen.image_auroc, mean_grid_auroc=points.image_auroc.mean(), selected_minus_grid_mean=chosen.image_auroc - points.image_auroc.mean()))
            rank_rows.extend(points.assign(pair=p, level=level, scope=scope, locked=points.config == selected[p]).to_dict("records"))
            for predictor in (["selection_aoss", "fold_aoss"] if level in ("dataset", "modality") else ["selection_aoss"]):
                rho, exact_p = exact_spearman(points[predictor], points.image_auroc, permutations)
                corr_rows.append(dict(pair=p, level=level, scope=scope, predictor=predictor, n_configs=len(points), rho=rho, p_exact=exact_p, p_asymptotic=spearmanr(points[predictor], points.image_auroc).pvalue))
    correlations = pd.DataFrame(corr_rows)
    correlations["p_holm_family"] = np.nan
    for _, part in correlations.groupby(["level", "predictor"]):
        correlations.loc[part.index, "p_holm_family"] = holm(part.p_exact)
    regret = pd.DataFrame(regrets)
    save("aoss_correlation", correlations)
    save("stitch_regret", regret)
    save("stitch_ranks", pd.DataFrame(rank_rows))
    budget_gap = final.groupby(["pair", "dataset"], as_index=False).image_auroc.mean().rename(columns={"image_auroc": "final_mean_auroc"})
    locked_sm = sm[sm.apply(lambda r: r.config == selected[r.pair], axis=1)]
    budget_gap = budget_gap.merge(locked_sm[["pair", "dataset", "image_auroc"]].rename(columns={"image_auroc": "screen_auroc"}), on=["pair", "dataset"])
    budget_gap["final_minus_screen"] = budget_gap.final_mean_auroc - budget_gap.screen_auroc
    save("training_budget_gap", budget_gap)
    # Fixed stitch sensitivity: all four pairs evaluated at each same grid point.
    fixed = sm.groupby(["source_stage", "target_block", "pair"], as_index=False).image_auroc.mean()
    fixed_rows = []
    for (stage, block), part in fixed.groupby(["source_stage", "target_block"]):
        v = part.set_index("pair").loc[PAIRS, "image_auroc"].to_numpy()
        for name, weights in EFFECTS.items():
            fixed_rows.append(dict(source_stage=stage, target_block=block, effect=name, image_auroc=float(v @ weights)))
    save("fixed_stitch_effects_posthoc", pd.DataFrame(fixed_rows))
    abl = tables["ablation"].groupby(["n_per_source_modality", "adapter", "score_mode"], as_index=False)[["image_auroc", "image_aupr"]].mean()
    save("ablation_summary", abl)
    baseline = pd.DataFrame(read(results / "baselines.json")["rows"])
    baseline["pair"] = baseline.apply(pair, axis=1)
    baseline = baseline.merge(dataset_summary[dataset_summary.metric == "image_auroc"][["dataset", "pair", "mean"]], on=["dataset", "pair"])
    baseline["final_minus_baseline_auroc"] = baseline["mean"] - baseline.image_auroc
    save("baseline_comparison", baseline)

    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "savefig.dpi": 170})
    colors = ["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"]

    def figure_save(fig, name):
        fig.savefig(figs / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(figs / f"{name}.png", bbox_inches="tight")
        plt.close(fig)

    def comparison(summary, key, order, name):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for i, p in enumerate(PAIRS):
            data = summary[(summary.pair == p) & (summary.metric == "image_auroc")].set_index(key).loc[order]
            ax.errorbar(np.arange(len(order)) + (i - 1.5) * .16, data["mean"], yerr=data.sd, fmt="o", capsize=3, label=p, color=colors[i])
        ax.axhline(.5, color="gray", ls="--", lw=1)
        ax.set(xticks=np.arange(len(order)), xticklabels=order, ylabel="Image AUROC (mean ± seed SD)", ylim=(.2, .85))
        ax.legend(ncol=4)
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        figure_save(fig, name)

    comparison(dataset_summary, "dataset", datasets, "main_2x2_auroc")
    comparison(modality_summary, "target_modality", mods, "modality_2x2_auroc")
    comparison(domain_summary, "domain_group", ["radiology", "non_radiology", "overall"], "domain_comparison")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, p, color in zip(axes, PAIRS, colors):
        points = sm[sm.pair == p].groupby("config", as_index=False)[["selection_aoss", "image_auroc", "source_stage", "target_block"]].mean()
        ax.scatter(points.selection_aoss, points.image_auroc, color=color)
        for _, row in points.iterrows():
            offset = (-8, 8) if p == "GG" and row.source_stage == 2 and row.target_block == 3 else (2, 3)
            if p == "GG" and row.source_stage == 2 and row.target_block == 6:
                offset = (-8, -13)
            ax.annotate(f"s{int(row.source_stage)}b{int(row.target_block)}", (row.selection_aoss, row.image_auroc), fontsize=7, xytext=offset, textcoords="offset points")
        ax.margins(x=.17, y=.13)
        chosen = points[points.config == selected[p]]
        ax.scatter(chosen.selection_aoss, chosen.image_auroc, s=130, facecolors="none", edgecolors="black")
        r = correlations[(correlations.pair == p) & (correlations.level == "overall")].iloc[0]
        ax.set(title=f"{p}: rho={r.rho:.2f}", xlabel="Five-fold mean AOSS", ylabel="Macro dataset AUROC")
    fig.tight_layout()
    figure_save(fig, "aoss_vs_auroc")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ranks = pd.DataFrame(rank_rows)
    for p, color in zip(PAIRS, colors):
        pts = ranks[(ranks.pair == p) & (ranks.level == "overall")].sort_values("aoss_rank")
        axes[0].plot(pts.aoss_rank, pts.auroc_rank, "o-", color=color, label=p, alpha=.8)
    axes[0].set(xlabel="AOSS rank (1 = best)", ylabel="Target AUROC rank (1 = best)", xticks=range(1, 10), yticks=range(1, 10))
    axes[0].invert_yaxis()
    axes[0].legend(ncol=2)
    r = regret[regret.level == "overall"].set_index("pair").loc[PAIRS]
    axes[1].bar(PAIRS, r.regret, color=colors)
    for i, (_, row) in enumerate(r.iterrows()):
        axes[1].text(i, row.regret + .002, f"{row.regret:.4f}\nrank {int(row.selected_auroc_rank)}/9", ha="center", fontsize=9)
    axes[1].set(ylabel="Top-1 AUROC regret (screen budget)", ylim=(0, .11))
    fig.tight_layout()
    figure_save(fig, "stitch_selection_regret")
    for p in PAIRS:
        points = sm[sm.pair == p].groupby(["source_stage", "target_block"])[["selection_aoss", "image_auroc"]].mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
        for ax, metric in zip(axes, ["selection_aoss", "image_auroc"]):
            matrix = points.pivot(index="source_stage", columns="target_block", values=metric)
            im = ax.imshow(matrix, cmap="viridis", **({"vmin": .35, "vmax": .65} if metric == "image_auroc" else {}))
            for i, stage in enumerate(matrix.index):
                for j, block in enumerate(matrix.columns):
                    mark = "*" if selected[p].endswith(f"_s{stage}_b{block}") else ""
                    text_color = "black" if im.norm(matrix.iloc[i,j]) > .55 else "white"
                    ax.text(j, i, f"{matrix.iloc[i,j]:.3f}{mark}", ha="center", va="center", color=text_color, fontsize=11)
            ax.set(title=f"{p}: {metric}", xticks=range(3), xticklabels=["3 (4th)", "6 (7th)", "9 (10th)"], yticks=range(3), yticklabels=[1, 2, 3], xlabel="Tail slice index (first retained block)", ylabel="CNN stage")
            fig.colorbar(im, ax=ax, shrink=.8)
        fig.tight_layout()
        figure_save(fig, f"stitchmap_{p}")
    effect_names = list(EFFECTS)
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, scope in enumerate(["radiology", "non_radiology", "overall"]):
        part = effects[(effects.level == "domain") & (effects.scope == scope) & (effects.metric == "image_auroc")].set_index("effect").loc[effect_names]
        ax.errorbar(part["mean"], np.arange(len(part)) + (j - 1) * .2, xerr=part.sd, fmt="o", capsize=3, label=scope)
    ax.axvline(0, c="gray", ls="--")
    ax.set(yticks=range(len(effect_names)), yticklabels=effect_names, xlabel="AUROC effect (matched-seed mean ± SD)")
    ax.legend()
    fig.tight_layout()
    figure_save(fig, "pretraining_effects")
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, part in abl[abl.adapter == "mlp"].groupby("score_mode"):
        ax.plot(part.n_per_source_modality, part.image_auroc, "o-", label=f"MLP {mode}")
    for _, row in abl[(abl.adapter == "linear") & (abl.score_mode == "contrast_topk")].iterrows():
        ax.scatter(row.n_per_source_modality, row.image_auroc, marker="x", s=80, label="linear contrast_topk")
    ax.set(xlabel="Normal calibration samples per source modality", ylabel="Macro dataset AUROC", title="MM ablation: seed 11, 6 epochs (post-hoc)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figure_save(fig, "ablation")

    lines = ["# 实验结果审计与分析", "",
             "本报告由 `python -m src.analyze_results --root .` 从原始 JSON 重建。只写 analysis/figures；不运行 selection、不重训、不改变 contrast_topk、0.05 top-k、数据划分或候选网格。所有异常值保留。", "",
             "## 完整性与证据边界", "",
             f"结构检查：36 screening × 5 folds、36 stitchmap × 6 datasets、12 final jobs × 6 datasets；{len(final)} 个 final cells。screen 的 mean_image_auroc=NaN 与空 rows 是跳过 target evaluation 的预期行为，不是训练 NaN。",
             f"数据审计来源：{provenance}。以下为记录中的计数，无法据此重新验证服务器图片、标签及 mask 内容。", "",
             md(audit_df, ["dataset", "train_normal", "test_normal", "test_abnormal", "test_with_mask"]), "",
             ("model_audit.json 已收到，四模型记录均 READY；model_status 检查的是文件存在/大小及 target config，不是权重哈希或完整加载报告。" if model_ready else "模型 audit 缺失或未全部 READY；不能宣称四模型权重独立核验通过。") + " checkpoint/feature cache 未提供独立复核证据。NumPy non-writable tensor 警告不等价于已发生数据损坏，当前分析未修改运行时代码。",
             "RESC 仅 764 个 test mask，等于异常图数；Brain/liver 全 test 有 mask。这与当前评估补零逻辑相容，但 mask 匹配仍待原始文件检查。", "",
             "## Selection protocol 审查", "",
             "锁定配置逐一等于五个 fold 的 source-only AOSS 算术平均 Top-1：MM/MG/GM=s2b3，GG=s2b9。screen 不含 target metrics；selected_configs 中 posthoc AUROC 为空。当前代码的 selection 排序不引用 target test，分析亦未改写锁定文件。",
             "**关键限制：**每个 fold 的训练/AOSS 排除了自身 target modality，但现有 select_configs 将所有五个 fold 的 AOSS 平均后选一个全局配置。某 modality 会作为其他 folds 的 source，因此全局选择会间接依赖该 modality 的 normal train/perturb 数据。代码支持“未用 target test 选点”，却不能支持全局选点也完全不接触 held-out modality 的更强 strict leave-one-modality-out 声明。本次如实报告并保持协议不变；不能用事后 fold-wise 选点替换主结果。",
             "target_block 是 layers[target_block:] 的切片起点；3/6/9 分别从第 4/7/10 个 block 开始保留。", "",
             "## 主结果", "",
             "AUROC/AUPR 为 0–1；以下 mean ± sample SD (ddof=1)，n=3 seeds。SD 仅描述训练随机性，不是患者置信区间。先在每个 seed 内作 dataset macro，再跨 seeds 汇总；不按图像数量 pooled。OCT 的两个数据集在 modality 表先平均，domain_modality_weighted_summary 为额外等 modality 权重敏感性分析。", ""]
    for metric in ["image_auroc", "image_aupr"]:
        display = dataset_summary[dataset_summary.metric == metric].copy()
        display["value"] = display.apply(lambda r: f"{r['mean']:.4f} ± {r.sd:.4f}", axis=1)
        display = display.pivot(index="dataset", columns="pair", values="value").reindex(datasets)[PAIRS].reset_index()
        lines += [f"### {metric}", "", md(display, ["dataset"] + PAIRS), ""]
    lines += [md(domain_summary[domain_summary.metric == "image_auroc"], ["domain_group", "pair", "mean", "sd"]), "",
              "## Medical-pretraining effects", "",
              "按同 seed 差值后汇总；source/target average 为两条件简单效应的平均，interaction=MM−MG−GM+GG。这些是已选配置的系统差异，GG stitch 点不同；不能当作纯预训练数据的因果效应。fixed_stitch_effects_posthoc.csv 提供相同 stitch 点的 screen-budget 敏感性分析，不反向选择最终配置。", "",
              md(effects[(effects.level == "domain") & (effects.metric == "image_auroc")], ["scope", "effect", "mean", "sd"]), "",
              "解释：radiology 中 source medical 平均效应为 −0.0449，target medical 平均效应为 +0.0147；non-radiology 中分别为 +0.0571 和 −0.0452。interaction 在两组方向相反（−0.0703 / +0.0731），在 overall 中相互抵消至 +0.0014。不能把整体近零解释为各模态均无效应；也不能凭三个 seed 的 SD 宣称跨患者显著。", "",
              "## AOSS 与真实 AUROC", "",
              "主相关性每 pair n=9 configurations：五-fold mean AOSS 对六-dataset macro AUROC。p_exact 为全部 9! 排列的双侧精确秩置换 p，p_holm_family 在四 pair 内校正。共享数据与相关网格使这些 p 仅作探索性参考，不代表跨任务泛化证据。分层结果按 level/predictor 分别 Holm 校正；数据集/模态层同时报告全局 selection AOSS 与该 fold AOSS，不混合为 54 个独立样本。", "",
              md(correlations[correlations.level == "overall"], ["pair", "rho", "p_exact", "p_asymptotic", "p_holm_family"]), "",
              "## Stitch-selection regret", "",
              "regret = 同一 screening 训练预算下的 post-hoc oracle AUROC − 已锁定配置 AUROC；rank 1 最好，ties 用 min rank。oracle 仅是诊断上界，不替换选择。不能拿 8-epoch/1000-sample final 与 4-epoch/500-sample oracle 相减。", "",
              md(regret[regret.level == "overall"], ["pair", "selected_auroc", "selected_auroc_rank", "oracle_auroc", "regret", "selected_minus_grid_mean"]), "",
              "## 异常值及限制", "",
              f"anomalies.csv 记录 {len(anomalies)} 个描述性标记：低于 0.5 的 final image/pixel AUROC、seed range>0.05、phase/pair 内 IQR 异常。阈值仅用于审计，不删除数据、不翻转 anomaly score 方向。",
              "值得优先检查：GM/OCT2017 AUROC=0.2949±0.0523；GM/RESC seed SD=0.1051；Brain 除 GM 外均低于 0.40 左右。MM 选点的 screen macro AUROC=0.5641，增加训练预算后的 final mean=0.5290；这是不同训练预算及 seed 汇总的描述性比较，不能直接断言过拟合。training_budget_gap.csv 保留所有 dataset/pair 差值。",
              "RSNA 异常比例约 95.46%，Brain 82.77%，OCT2017 75%；AUPR 高并不自动意味着识别好，应对照 prevalence。没有 per-image/patient scores，不能补造病例级 bootstrap、DeLong 或患者配对显著性。仅有最后一轮 train_loss，不能确认完整收敛曲线。",
              "MM/MG/GM/GG 的整体表现接近随机且具有模态差异；不能写成医学预训练普遍更优。AOSS 在不同 pair 上方向不一致，当前结果不支持普遍有效的 stitch 预测器。sanity baseline 和 ablation 已作为补充表输出，不能用 post-hoc 指标改选主结果。", "",
              "建议首先归档服务器 data/model audit、manifest、checkpoint 与运行版本证据，并导出固定配置的 per-image scores 供误例检查；在明确全局跨-fold 选择的声明边界前，不扩大模型矩阵或更高分辨率实验。", "",
              "## 产物", "",
              "analysis/*.csv 包括完整性、数据计数、异常、主结果、效应、分层相关性、秩、regret、同点效应和 ablation/baseline。figures/*.pdf 与同名 PNG 为可复用图表。input_hashes.json 记录分析前后原始文件 SHA256，不修改旧 REPORT.md。", ""]
    (out / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    after = digest(raw_paths)
    if before != after:
        raise RuntimeError("Raw inputs changed while analyzing")
    (out / "input_hashes.json").write_text(json.dumps({"unchanged": True, "sha256": before, "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}, indent=2), encoding="utf-8")
    print(f"Analysis complete: {out}; {len(checks)} checks; raw inputs unchanged")
    print(domain_summary[domain_summary.metric == "image_auroc"].to_string(index=False))
    print(correlations[correlations.level == "overall"].to_string(index=False))
    print(regret[regret.level == "overall"].to_string(index=False))


if __name__ == "__main__":
    main()
