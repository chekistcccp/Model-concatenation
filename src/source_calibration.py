"""Exploratory source-normal calibration; original selection/scoring stay intact.

Fit reads only locked identities and source diagnostic CSVs. Evaluate is separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from .analyze_results import effect_summary, pair, summarize
from .followup_io import common_jobs, read_json
from .prediction_io import sha256


VERSION = "source_normal_ecdf_v1"
COMPONENTS = {"mean": "normal_mean", "contrast": "normal_contrast_topk"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fit_reference(source, held_out, modality_map):
    """Only normal-map components enter the reference; no target scores/labels."""
    expected = {d for d, m in modality_map.items() if m != held_out}
    require(set(source.dataset) == expected, "Incomplete source datasets or held-out contamination")
    require((source.held_out_modality == held_out).all(), "Wrong held-out fold")
    require(all(source.source_modality == source.dataset.map(modality_map)), "Source modality mismatch")
    require(not source.duplicated(["dataset", "record_index"]).any(), "Duplicate source observations")
    require((source.groupby("dataset").size() == 256).all(), "Expected 256 cached normal observations per dataset")
    reference = {}
    for modality, group in source.groupby("source_modality"):
        reference[modality] = {}
        for dataset, rows in group.groupby("dataset"):
            require(set(rows.record_index) == set(range(256)), "Source record indices differ")
            reference[modality][dataset] = {}
            for component, column in COMPONENTS.items():
                values = rows[column].to_numpy(float)
                require(np.isfinite(values).all(), "Nonfinite normal component")
                reference[modality][dataset][component] = np.sort(values).tolist()
    require(len(reference) == 4 and held_out not in reference, "Need four non-target source modalities")
    return reference


def mid_ecdf(reference, values):
    reference, values = np.asarray(reference, float), np.asarray(values, float)
    require(reference.ndim == 1 and len(reference) > 0 and np.isfinite(reference).all()
            and (np.diff(reference) >= 0).all() and np.isfinite(values).all(), "Invalid ECDF input")
    return (np.searchsorted(reference, values, side="left")
            + np.searchsorted(reference, values, side="right")) / (2 * len(reference))


def calibrated_scores(reference, means, contrasts):
    """Equal dataset weights inside modality, then equal modality weights."""
    scores = {}
    for component, values in [("mean", means), ("contrast", contrasts)]:
        scores[component] = np.mean([
            np.mean([mid_ecdf(r[component], values) for r in datasets.values()], axis=0)
            for datasets in reference.values()], axis=0)
    return dict(source_mean_ecdf=scores["mean"], source_contrast_ecdf=scores["contrast"],
                source_equal_fusion=.5 * (scores["mean"] + scores["contrast"]))


def atomic_json(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    temporary.replace(path)


def inputs(root, diagnostics):
    cfg_path = root / "configs/experiment.yaml"
    selected_path = root / "results/selected_configs.json"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    jobs = [j for j in common_jobs(cfg, read_json(selected_path)) if j["reuse_original"]]
    manifest_path = diagnostics / "run_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest["status"] == "complete" and manifest["no_reselection"]
            and manifest["original_inputs_unchanged"], "Need complete unchanged diagnostics")
    require({j["job"] for j in manifest["jobs"]} == {j["job"] for j in jobs}, "Locked job set differs")
    require(manifest["original_sha256"]["results/selected_configs.json"] == sha256(selected_path), "Selection changed")
    require(yaml.safe_load((diagnostics / "inputs/experiment.yaml").read_text(encoding="utf-8")) == cfg, "Config changed")
    hashes = {str(p.resolve()): sha256(p) for p in [cfg_path, selected_path, manifest_path, Path(__file__)]}
    return cfg, jobs, manifest, hashes


def fit(root, diagnostics, out):
    cfg, jobs, manifest, hashes = inputs(root, diagnostics)
    calibrators = {}
    for job in jobs:
        calibrators[job["job"]] = {}
        for held in sorted(set(cfg["data"]["modality_map"].values())):
            relative = f"jobs/{job['job']}/source_{held}.csv"
            p = diagnostics / relative
            require(sha256(p) == manifest["output_sha256"][relative], f"Changed source export: {relative}")
            hashes[str(p.resolve())] = sha256(p)
            # Deliberately exclude perturbed-map columns and target-score files.
            source = pd.read_csv(p, usecols=["dataset", "source_modality", "held_out_modality", "record_index", *COMPONENTS.values()])
            calibrators[job["job"]][held] = fit_reference(source, held, cfg["data"]["modality_map"])
    payload = dict(version=VERSION, posthoc_exploratory=True, selection_unchanged=True,
                   formula="0.5 * source_mean_ecdf + 0.5 * source_contrast_ecdf",
                   weighting="equal modalities; equal datasets within each modality; midrank ECDF",
                   source_hashes=hashes, jobs=jobs, modality_map=cfg["data"]["modality_map"], calibrators=calibrators)
    require(all(sha256(Path(p)) == h for p, h in hashes.items()), "Input changed during fit")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "calibration.json"
    if path.exists():
        require(read_json(path) == payload, "Existing frozen calibration differs; do not overwrite this experiment")
    else:
        atomic_json(path, payload)
    print(f"Frozen 60 source-only references: {path}; no target scores or labels loaded")


def evaluate(root, diagnostics, out):
    frozen = out / "calibration.json"
    plan = read_json(frozen)
    require(plan["version"] == VERSION, "Calibration version mismatch")
    require(all(sha256(Path(p)) == h for p, h in plan["source_hashes"].items()), "Frozen source/code input changed")
    manifest = read_json(diagnostics / "run_manifest.json")
    hashes = dict(plan["source_hashes"], **{str(frozen.resolve()): sha256(frozen)})
    rows, saturation = [], []
    for job in plan["jobs"]:
        name = job["job"]
        final_path = root / "results/final" / f"{name}.json"
        require(sha256(final_path) == manifest["original_sha256"][f"results/final/{name}.json"], "Original final changed")
        hashes[str(final_path.resolve())] = sha256(final_path)
        original = {r["dataset"]: r for r in read_json(final_path)["rows"]}
        for dataset, held in plan["modality_map"].items():
            p = root / "results/predictions" / name / f"{dataset}.csv"
            require(sha256(p) == manifest["original_sha256"][f"results/predictions/{name}/{dataset}.csv"], "Prediction hash differs")
            hashes[str(p.resolve())] = sha256(p)
            data = pd.read_csv(p)
            require(len(data) == original[dataset]["n_test"] and set(data.label) == {0, 1}, "Test labels/counts differ")
            require((data.dataset == dataset).all() and (data.seed == job["seed"]).all()
                    and (data.config == job["config"]).all(), "Prediction identity differs")
            scores = calibrated_scores(plan["calibrators"][name][held], data.discrepancy_mean, data.score)
            scores = dict(original_contrast=data.score.to_numpy(), **scores)
            export = data[["record_index", "image_path", "label"]].assign(**scores)
            directory = out / "predictions" / name
            directory.mkdir(parents=True, exist_ok=True)
            export.to_csv(directory / f"{dataset}.csv", index=False)
            for method, values in scores.items():
                auc = float(roc_auc_score(data.label, values))
                ap = float(average_precision_score(data.label, values))
                if method == "original_contrast":
                    require(abs(auc - original[dataset]["image_auroc"]) <= 1e-6 and abs(ap - original[dataset]["image_aupr"]) <= 1e-6, "Original replay differs")
                rows.append(dict(dataset=dataset, domain_group=original[dataset]["domain_group"], pair=pair(job), seed=job["seed"],
                                 method=method, image_auroc=auc, image_aupr=ap))
                saturation.append(dict(job=name, dataset=dataset, method=method, n=len(values),
                                       unique_scores=len(np.unique(values)), fraction_zero=float(np.mean(values == 0)), fraction_one=float(np.mean(values == 1))))
    frame = pd.DataFrame(rows)
    scopes = pd.concat([frame.assign(scope=frame.dataset), frame.assign(scope=frame.domain_group), frame.assign(scope="overall")])
    macros = scopes.groupby(["method", "scope", "pair", "seed"], as_index=False)[["image_auroc", "image_aupr"]].mean()
    summary = summarize(macros, ["method", "scope", "pair"], ["image_auroc", "image_aupr"])
    effects = effect_summary(macros, ["method", "scope"])
    wide = macros.pivot(index=["scope", "pair", "seed"], columns="method", values="image_auroc")
    deltas = wide.subtract(wide.original_contrast, axis=0).drop(columns="original_contrast").reset_index()
    for file, table in dict(metrics=frame, seed_macros=macros, summary=summary, effects=effects,
                            paired_auroc_deltas=deltas, saturation=pd.DataFrame(saturation)).items():
        table.to_csv(out / f"{file}.csv", index=False)
    require(all(sha256(Path(p)) == h for p, h in hashes.items()), "Input changed during evaluation")
    atomic_json(out / "evaluation_manifest.json", dict(status="complete", version=VERSION, exploratory=True,
        selection_unchanged=True, input_sha256=hashes, n_metric_rows=len(frame),
        output_sha256={str(p.relative_to(out)): sha256(p) for p in out.rglob("*.csv")}))
    report = ["# Source-normal calibration: exploratory only", "", "Fixed source-only equal fusion; all four methods reported. Original main score and selection unchanged.",
              "No target score/label enters fitting. Development was informed by previously observed test results: this is not confirmatory held-out evidence.",
              "Calibration normals can overlap adapter training; no independent source validation claim. ECDF clipping can introduce ties; see saturation.csv.",
              "Global cross-fold AOSS selection limitations and legacy checkpoint provenance remain. Seed SD is not a patient-level CI.",
              "", "|Scope|Method|Pair|AUROC mean|Seed SD|", "|---|---|---|---|---|"]
    for r in summary[(summary.metric == "image_auroc") & summary.scope.isin(["overall", "radiology", "non_radiology"])].itertuples():
        report.append(f"|{r.scope}|{r.method}|{r.pair}|{r.mean:.4f}|{r.sd:.4f}|")
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Complete: {len(frame)} metric rows; {out / 'report.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--diagnostics", type=Path, default=Path("results/followup/diagnostics"))
    parser.add_argument("--stage", choices=["fit", "evaluate"], required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    diagnostics = (root / args.diagnostics).resolve()
    out = root / "results/exploratory/source_normal_ecdf_v1"
    require(not out.is_relative_to(diagnostics), "Output overlaps diagnostics")
    (fit if args.stage == "fit" else evaluate)(root, diagnostics, out)


if __name__ == "__main__":
    main()
