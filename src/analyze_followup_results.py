"""CPU-only audit/analysis of transferred follow-ups; never selects or trains."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .analyze_results import PAIRS, effect_summary, pair, summarize
from .followup_io import DATASETS, FOCUS, common_jobs, read_json, validate_common_result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_manifest(directory, root):
    m = read_json(directory / "run_manifest.json")
    require(m["status"] == "complete" and m["no_reselection"]
            and m["original_inputs_unchanged"], f"Incomplete/changed run: {directory}")
    require(m["main_score"] == "contrast_topk" and m["topk_fraction"] == .05, "Score changed")
    for name, expected in m["output_sha256"].items():
        p = (directory / name).resolve()
        require(p.is_relative_to(directory.resolve()), "Manifest path escapes input directory")
        require(p.is_file() and sha(p) == expected, f"Missing/changed transfer: {name}")
    # Verify available original artifacts; absent server arrays/weights are not claimed checked.
    checked, absent, line_endings, receipt_differences = [], [], [], []
    for name, expected in m["original_sha256"].items():
        p = root / name
        if p.is_file():
            if sha(p) == expected:
                checked.append(name)
            elif p.suffix in {".py", ".yaml"} and hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected:
                line_endings.append(name)
            elif name.startswith("results/predictions/") and p.name in {"evaluation.json", "export_manifest.json"}:
                receipt_differences.append(name)
            else:
                raise ValueError(f"Original artifact differs: {name}")
        else:
            absent.append(name)
    return m, dict(output_hashes=len(m["output_sha256"]), original_checked=len(checked),
                   original_not_available=absent, line_endings_only=line_endings,
                   prediction_receipt_hash_differences=receipt_differences, git_head=m["git_head"])


def common_analysis(directory, root):
    cfg = yaml.safe_load((directory / "inputs/experiment.yaml").read_text(encoding="utf-8"))
    selected = read_json(directory / "inputs/selected_configs.json")
    require(sha(directory / "inputs/selected_configs.json") == sha(root / "results/selected_configs.json"),
            "Locked selection differs")
    jobs = common_jobs(cfg, selected)
    manifest = read_json(directory / "run_manifest.json")
    for job in jobs:
        if job["reference"]: job["reference"] = job["reference"].replace("\\", "/")
    require(manifest["jobs"] == jobs, "Common job plan differs from locked point union")
    counts = {r["dataset"]: r["n_test"] for r in read_json(next((root / "results/final").glob("*seed*.json")))["rows"]}
    rows, aoss = [], []
    for job in jobs:
        p = directory / (job["job"] + ".json")
        result = read_json(p)
        validate_common_result(cfg, job, result, counts)
        if job["reuse_original"]:
            require(sha(p) == sha(root / "results/final" / p.name), "Reused final changed")
        for r in result["rows"]:
            require(0 <= r["image_auroc"] <= 1 and 0 <= r["image_aupr"] <= 1, "Invalid metric")
            rows.append(dict(r, pair=pair(result)))
        for r in result["aoss_rows"]:
            require(all(np.isfinite(r[k]) for k in ["normal_discrepancy", "local_sensitivity", "aoss"]), "Invalid AOSS")
            aoss.append(dict(r, pair=pair(result), seed=job["seed"], target_block=job["target_block"]))
    frame = pd.DataFrame(rows)
    scopes = [frame.assign(scope=frame.dataset), frame.assign(scope=frame.domain_group), frame.assign(scope="overall")]
    macros = pd.concat(scopes).groupby(["target_block", "scope", "pair", "seed"], as_index=False)[["image_auroc", "image_aupr"]].mean()
    summary = summarize(macros, ["target_block", "scope", "pair"], ["image_auroc", "image_aupr"])
    effects = effect_summary(macros, ["target_block", "scope"])
    wide = macros.pivot(index=["scope", "pair", "seed"], columns="target_block", values="image_auroc")
    delta = (wide[9] - wide[3]).rename("block9_minus_block3").reset_index()
    return frame, macros, summary, effects, delta, pd.DataFrame(aoss)


def map_statistics(z):
    raw, contrast, top = z["discrepancy"], z["contrast"], z["topk_mask"].astype(bool)
    require(raw.ndim == 3 and raw.shape[1:] == (16, 16), "Map geometry changed")
    require(np.isfinite(raw).all() and np.isfinite(z["scores"]).all(), "Nonfinite map")
    med = np.sort(raw.reshape(len(raw), -1), axis=1)[:, 127]
    require(np.allclose(med, z["patch_median"], atol=1e-7, rtol=0), "Wrong lower median")
    require(np.allclose(contrast, np.maximum(raw - med[:, None, None], 0), atol=1e-7, rtol=0), "Wrong contrast")
    require((top.sum((1, 2)) == 13).all(), "Wrong top-k count")
    expected = np.sort(contrast.reshape(len(raw), -1), axis=1)[:, -13:].mean(1)
    require(np.allclose(expected, z["scores"], atol=1e-6, rtol=0), "Map score mismatch")
    require(np.allclose((contrast * top).sum((1, 2)) / 13, expected, atol=1e-6, rtol=0), "Top-k mask mismatch")
    border = np.ones((16, 16), bool)
    border[1:-1, 1:-1] = False
    output = []
    for i in range(len(raw)):
        gt = z["ground_truth_mask"][i].astype(bool)
        valid = bool(z["has_ground_truth_mask"][i]) and gt.any()
        output.append(dict(record_index=int(z["record_indices"][i]), label=int(z["labels"][i]),
            score=float(expected[i]), raw_mean=float(raw[i].mean()), median=float(med[i]),
            border_topk_fraction=float((top[i] & border).sum() / 13),
            gt_available=bool(z["has_ground_truth_mask"][i]), gt_nonempty=bool(gt.any()),
            gt_area_fraction=float(gt.mean()) if valid else np.nan,
            topk_gt_precision=float((top[i] & gt).sum() / 13) if valid else np.nan,
            gt_patch_recall=float((top[i] & gt).sum() / gt.sum()) if valid else np.nan))
    return output


def diagnostic_analysis(directory, root):
    cases = pd.read_csv(directory / "cases.csv")
    require(not cases.duplicated(["dataset", "record_index"]).any(), "Duplicate cases")
    require(len(cases) == read_json(directory / "run_manifest.json")["n_cases"], "Case count mismatch")
    jobs = read_json(directory / "run_manifest.json")["jobs"]
    require(len(jobs) == 12 and len({j["job"] for j in jobs}) == 12, "Incomplete diagnostic jobs")
    cfg = yaml.safe_load((directory / "inputs/experiment.yaml").read_text(encoding="utf-8"))
    expected_jobs = {j["job"] for j in common_jobs(cfg, read_json(directory / "inputs/selected_configs.json")) if j["reuse_original"]}
    require({j["job"] for j in jobs} == expected_jobs, "Diagnostic jobs differ from locked selection")
    maps, sources, replay = [], [], []
    for job in jobs:
        p = directory / "jobs" / job["job"]
        receipt = read_json(p / "evaluation.json")
        require(receipt["job"] == job["job"] and receipt["no_training"] and receipt["no_reselection"], "Receipt identity")
        for key, file in [("original_final_sha256", root / "results/final" / (job["job"] + ".json")),
                          ("selected_configs_sha256", root / "results/selected_configs.json"),
                          ("case_plan_sha256", directory / "case_plan.json")]:
            require(receipt[key] == sha(file), f"Receipt hash differs: {key}")
        require(receipt["replay_matches_original"] and receipt["aoss_replay_matches_original"], "Failed replay")
        require(len(receipt["comparison"]) == 6 and len(receipt["aoss_comparison"]) == 25, "Incomplete replay")
        for r in receipt["comparison"] + receipt["aoss_comparison"]:
            require(r["within_tolerance"] and abs(r["replayed"] - r["original"]) <= 1e-6, "Replay differs")
            replay.append(dict(r, job=job["job"]))
        require(len(receipt["rows"]) == 3 and all(r["max_score_abs_delta"] <= 1e-6 for r in receipt["rows"]), "Score replay differs")
        ident = dict(pair=pair(job), seed=job["seed"], job=job["job"])
        for ds in FOCUS:
            subset = cases[cases.dataset == ds].set_index("record_index")
            with np.load(p / f"{ds}_maps.npz", allow_pickle=False) as z:
                require(set(z["record_indices"]) == set(subset.index), "Map case set differs")
                for r in map_statistics(z):
                    case = subset.loc[r["record_index"]]
                    require(r["label"] == case.label, "Case label differs")
                    maps.append(dict(r, **ident, dataset=ds, reason=case.reason))
        for source_file in p.glob("source_*.csv"):
            f = pd.read_csv(source_file)
            held = source_file.stem.removeprefix("source_")
            require((f.held_out_modality == held).all() and (f.source_modality != held).all(), "Held-out contamination")
            expected_ds = set(DATASETS) - set({"brain_mri": ["Brain"], "liver_ct": ["liver"], "oct": ["RESC", "OCT2017"], "chest_xray": ["RSNA"], "pathology": ["camelyon16"]}[held])
            require(set(f.dataset) == expected_ds and (f.groupby("dataset").size() == 256).all(), "Incomplete source rows")
            require(not f.duplicated(["dataset", "record_index"]).any(), "Duplicate source records")
            require(((f.n_mask_in + f.n_mask_out) == 256).all(), "Source mask geometry")
            require(np.isfinite(f[["normal_mean", "pert_mean", "normal_contrast_topk", "pert_contrast_topk"]]).all().all(), "Invalid source statistics")
            sources.append(f.assign(**ident))
        require(len(list(p.glob("source_*.csv"))) == 5, "Missing source folds")
    return cases, pd.DataFrame(maps), pd.concat(sources, ignore_index=True), pd.DataFrame(replay)


def figures(summary, effects, cases, directory, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, scope in zip(axes, ["overall", "radiology", "non_radiology"]):
        for offset, block in [(-.12, 3), (.12, 9)]:
            f = summary[(summary.scope == scope) & (summary.target_block == block) & (summary.metric == "image_auroc")].set_index("pair").loc[PAIRS]
            ax.errorbar(np.arange(4) + offset, f["mean"], yerr=f.sd, fmt="o", capsize=3, label=f"s2b{block}")
        ax.set_xticks(range(4), PAIRS)
        ax.set_title(scope)
        ax.axhline(.5, color="grey", linestyle=":")
    axes[0].set_ylabel("Dataset-macro AUROC (mean +/- seed SD)")
    axes[-1].legend()
    fig.tight_layout(); fig.savefig(out / "common_stitch.png", dpi=180); plt.close(fig)
    # Deterministic examples: first record in each original random/hard stratum, seed 11.
    for ds, focus in FOCUS.items():
        job = f"{focus[0]}_to_{focus[1]}_s2_b3_seed11"
        selected = []
        for label in [0, 1]:
            for reason in ["random", "hard"]:
                f = cases[(cases.dataset == ds) & (cases.label == label) & cases.reason.str.contains(reason)].sort_values("record_index")
                if len(f): selected.append(f.iloc[0])
        with np.load(directory / "jobs" / job / f"{ds}_maps.npz") as z:
            fig, axes = plt.subplots(len(selected), 4, figsize=(10, 2.5 * len(selected)), squeeze=False)
            vmax = float(z["discrepancy"].max())
            for row, c in enumerate(selected):
                i = list(z["record_indices"]).index(c.record_index)
                axes[row, 0].imshow(plt.imread(directory / c.image_file))
                axes[row, 0].set_title(f"{ds} #{c.record_index}, y={c.label}\n{c.reason}")
                for col, key in [(1, "discrepancy"), (2, "contrast")]:
                    im = axes[row, col].imshow(z[key][i], vmin=0, vmax=vmax, cmap="magma")
                    axes[row, col].set_title(key)
                axes[row, 3].imshow(z["topk_mask"][i], vmin=0, vmax=1, cmap="Greys")
                gt = z["ground_truth_mask"][i]
                if z["has_ground_truth_mask"][i] and gt.any() and not gt.all():
                    axes[row, 3].contour(gt, levels=[.5], colors=["red"], linewidths=1)
                axes[row, 3].set_title(f"Top-13; GT red if present\nscore={z['scores'][i]:.3f}")
                for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
            fig.suptitle(f"Post-hoc selected cases; {job}; raw/contrast share scale")
            fig.tight_layout(rect=[0, 0, .94, .96])
            fig.colorbar(im, cax=fig.add_axes([.95, .2, .015, .6]))
            fig.savefig(out / f"spatial_{ds}.png", dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path, default=Path("results/followup"))
    args = parser.parse_args()
    root = args.root.resolve()
    inp = (root / args.input).resolve()
    out = root / "results/analysis/followup_results"
    require(not out.is_relative_to(inp), "Analysis output overlaps inputs")
    manifests, audit = {}, {}
    for stage in ["diagnostics", "common_stitch"]:
        manifests[stage], audit[stage] = audit_manifest(inp / stage, root)
    frame, macros, summary, effects, delta, aoss = common_analysis(inp / "common_stitch", root)
    cases, maps, source, replay = diagnostic_analysis(inp / "diagnostics", root)
    out.mkdir(parents=True, exist_ok=True)
    outputs = dict(common_rows=frame, common_seed_macros=macros, common_summary=summary,
                   common_effects=effects, block_deltas=delta, common_aoss=aoss,
                   spatial_cases=maps, replay_checks=replay)
    # Per-image source diagnostic means, not a reconstruction of authoritative AOSS.
    keys = ["pair", "seed", "held_out_modality", "dataset", "perturbation_kind_inferred"]
    metrics = ["normal_mean", "pert_mean", "normal_contrast_topk", "pert_contrast_topk", "local_sensitivity", "mean_delta", "contrast_delta"]
    outputs["source_components_by_kind"] = source.groupby(keys, as_index=False)[metrics].mean()
    outputs["source_counts"] = source.groupby(keys, as_index=False).agg(n=("record_index", "size"), valid_local=("local_sensitivity", "count"))
    # Random and hard strata deliberately overlap when the predeclared sampling overlaps.
    spatial = pd.concat([maps[maps.reason.str.contains(r)].assign(stratum=r) for r in ["random", "hard"]])
    outputs["spatial_summary"] = spatial.groupby(["dataset", "pair", "seed", "label", "stratum"], as_index=False)[["score", "raw_mean", "median", "border_topk_fraction", "gt_area_fraction", "topk_gt_precision", "gt_patch_recall"]].mean()
    for name, data in outputs.items(): data.to_csv(out / f"{name}.csv", index=False)
    figures(summary, effects, cases, inp / "diagnostics", out)
    audit.update(cases=len(cases), common_rows=len(frame), spatial_maps=len(maps), source_rows=len(source), replay_checks=len(replay), max_replay_delta=float(replay.delta.abs().max()))
    # Detect input changes during analysis as well as transfer corruption.
    for stage in manifests: audit_manifest(inp / stage, root)
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    report = ["# Fixed-point and spatial follow-up analysis", "", "Transfer/metric audit passed with provenance qualifications below. Both predeclared points are reported; no ranking, reselection, training or score changes.",
              f"24 common jobs / {len(frame)} dataset rows; {len(cases)} selected cases / {len(maps)} maps; {len(source)} source diagnostic rows.",
              f"Replay checks: {len(replay)}; maximum absolute metric/AOSS difference: {audit['max_replay_delta']:.3g}.",
              "", "## AUROC mean and seed SD", "", "|Scope|Point|MM|MG|GM|GG|", "|---|---|---|---|---|---|"]
    for scope in ["overall", "radiology", "non_radiology"]:
        for block in [3, 9]:
            f = summary[(summary.scope == scope) & (summary.target_block == block) & (summary.metric == "image_auroc")].set_index("pair")
            report.append(f"|{scope}|s2b{block}|" + "|".join(f"{f.loc[p, 'mean']:.4f} +/- {f.loc[p, 'sd']:.4f}" for p in PAIRS) + "|")
    report += ["", "## Matched-seed effects (AUROC units)", "", "|Scope|Point|Effect|Mean|Seed SD|", "|---|---|---|---|---|"]
    selected_effects = effects[(effects.scope.isin(["overall", "radiology", "non_radiology"])) & (effects.metric == "image_auroc") & effects.effect.isin(["source_average", "target_average", "interaction"])]
    for r in selected_effects.itertuples():
        report.append(f"|{r.scope}|s2b{r.target_block}|{r.effect}|{r.mean:+.4f}|{r.sd:.4f}|")
    report += ["", "## Interpretation boundaries", "", "- Dataset macro is computed within each seed before seed SD. Seeds are matched for effects. Seed SD is not a patient or image confidence interval.",
               "- source_average=(MM+MG-GM-GG)/2; target_average=(MM+GM-MG-GG)/2; interaction=MM-MG-GM+GG. All conditional effects are in common_effects.csv.",
               "- Fixed points are post-hoc controls, not replacements for locked main results. Backbone contrasts do not isolate a pure causal pretraining effect.",
               "- Spatial summaries describe selected cases only. Random and hard strata are separate; overlapping cases occur in both strata. They do not estimate full-test localization performance.",
               "- Missing/empty downsampled masks have undefined localization ratios, not zero precision. Border means the outer one-patch ring (60/256 patches); no threshold was fitted.",
               "- Source perturbation types are inferred from current generator/index, not historical operation provenance. Per-image means are not authoritative AOSS; replay checks use original aggregation.",
               "- Large server arrays/weights were stat-checked by the runner, not independently content-hashed here. Unavailable original artifacts are listed in audit.json. Legacy seed provenance and global across-fold AOSS selection limitations remain.",
               f"- Original prediction receipt hash differences: {len(audit['diagnostics']['prediction_receipt_hash_differences'])}. These local metadata receipts cannot authenticate the server run; original prediction CSVs and final metrics are checked independently. See audit.json. Code/config CRLF-only differences are separately recorded."]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k not in manifests}, indent=2))
    print(out)


if __name__ == "__main__":
    main()
