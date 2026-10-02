"""Read-only CPU audit/summary of source-only paired perturbation responses."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MEASURES = ["normal_spatial_contrast", "raw_local_sensitivity", "delta_in", "delta_out",
            "net_local_response", "mean_delta", "contrast_delta"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(frame, extra_keys=()):
    """Equal datasets inside modality, then equal source modalities and folds."""
    keys = ["pair", "seed", "held_out_modality", "source_modality", *extra_keys]
    ds = frame.groupby(keys + ["dataset"], as_index=False)[MEASURES].mean()
    mod = ds.groupby(keys, as_index=False)[MEASURES].mean()
    fold = mod.groupby(["pair", "seed", "held_out_modality", *extra_keys], as_index=False)[MEASURES].mean()
    seed = fold.groupby(["pair", "seed", *extra_keys], as_index=False)[MEASURES].mean()
    summary = seed.groupby(["pair", *extra_keys])[MEASURES].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(c) for c in summary.columns]
    return mod, fold, seed, summary.reset_index()


def pretraining_effects(seed):
    weights = {"source_with_medical_target": [1, 0, -1, 0],
               "source_with_general_target": [0, 1, 0, -1],
               "target_with_medical_source": [1, -1, 0, 0],
               "target_with_general_source": [0, 0, 1, -1],
               "source_average": [.5, .5, -.5, -.5],
               "target_average": [.5, -.5, .5, -.5], "interaction": [1, -1, -1, 1]}
    rows = []
    for measure in MEASURES:
        wide = seed.pivot(index="seed", columns="pair", values=measure).reindex(columns=["MM", "MG", "GM", "GG"])
        require(not wide.isna().any().any(), "Incomplete paired-seed effect cells")
        for name, w in weights.items():
            values = wide.mul(w).sum(axis=1)
            rows.append(dict(measure=measure, effect=name, mean=values.mean(), seed_sd=values.std(),
                             minimum=values.min(), maximum=values.max(), n_seeds=len(values)))
    return pd.DataFrame(rows)


def analyze(root, inp):
    mpath = inp / "run_manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    require(manifest["status"] == "complete" and manifest["original_inputs_unchanged"]
            and manifest["no_training"] and manifest["no_reselection"]
            and not manifest["target_test_read"] and not manifest["target_valid_read"], "Incomplete or changed protocol")
    digests = {str(mpath): sha(mpath)}
    for name, digest in manifest["output_sha256"].items():
        p = (inp / name).resolve()
        require(p.is_relative_to(inp.resolve()) and p.is_file() and sha(p) == digest, f"Missing/changed output: {name}")
        digests[str(p)] = digest
    jobs = manifest["jobs"]
    require(len(jobs) == 12 and len({j["job"] for j in jobs}) == 12, "Incomplete locked jobs")
    frames = []
    for job in jobs:
        for held in ["brain_mri", "chest_xray", "liver_ct", "oct", "pathology"]:
            name = f"jobs/{job['job']}/source_{held}.csv"
            require(name in manifest["output_sha256"], "Missing paired fold")
            f = pd.read_csv(inp / name)
            mapping = dict(Brain="brain_mri", liver="liver_ct", RESC="oct", OCT2017="oct", RSNA="chest_xray", camelyon16="pathology")
            require((f.job == job["job"]).all() and (f.seed == job["seed"]).all()
                    and (f.held_out_modality == held).all() and (f.source_modality != held).all(), "Identity or held-out contamination")
            require((f.source_backbone == job["source_backbone"]).all() and (f.target_backbone == job["target_backbone"]).all(), "Backbone identity")
            require(set(f.dataset) == {d for d, v in mapping.items() if v != held}
                    and (f.source_modality == f.dataset.map(mapping)).all(), "Incomplete source datasets")
            require((f.groupby("dataset").size() == 256).all() and not f.duplicated(["dataset", "record_index"]).any(), "Source rows duplicated/missing")
            require(all(set(g.record_index) == set(range(256)) for _, g in f.groupby("dataset")), "Source indices differ")
            require(((f.n_mask_in + f.n_mask_out) == 256).all(), "Mask geometry changed")
            require((f.n_mask_in >= 0).all() and (f.n_mask_out >= 0).all(), "Invalid region counts")
            valid = (f.n_mask_in > 0) & (f.n_mask_out > 0)
            require((valid == f.valid_regions).all(), "Mask validity flag differs")
            require(np.isfinite(f.loc[valid, MEASURES]).all().all(), "Nonfinite valid response")
            require(f.loc[~valid, ["net_local_response", "normal_spatial_contrast", "raw_local_sensitivity"]].isna().all().all(), "Missing region incorrectly zero-filled")
            require(np.allclose(f.loc[valid, "raw_local_sensitivity"],
                    f.loc[valid, "normal_spatial_contrast"] + f.loc[valid, "net_local_response"], atol=1e-8, rtol=0), "Paired decomposition failed")
            for value, a, b in [("normal_spatial_contrast", "normal_in", "normal_out"),
                                ("raw_local_sensitivity", "perturb_in", "perturb_out"),
                                ("delta_in", "perturb_in", "normal_in"),
                                ("delta_out", "perturb_out", "normal_out"),
                                ("net_local_response", "delta_in", "delta_out")]:
                require(np.allclose(f.loc[valid, value], f.loc[valid, a] - f.loc[valid, b], atol=1e-8, rtol=0), "Regional arithmetic differs")
            frames.append(f.assign(pair=job["source_backbone"][0].upper() + job["target_backbone"][0].upper()))
    frame = pd.concat(frames, ignore_index=True)
    require(len(frame) == manifest["n_source_rows"] == 73728, "Incomplete response matrix")
    replay = pd.read_csv(inp / "aoss_replay.csv")
    require(len(replay) == 300 and replay.within_tolerance.all() and np.isfinite(replay.delta).all()
            and (replay.delta.abs() <= 1e-6).all(), "Original AOSS replay differs")
    require(not replay.duplicated(["job", "held_out_modality", "component"]).any(), "Duplicate replay checks")
    expected_replay = {(j["job"], h, c) for j in jobs for h in ["brain_mri", "chest_xray", "liver_ct", "oct", "pathology"]
                       for c in ["normal_discrepancy", "perturb_in", "perturb_out", "local_sensitivity", "aoss"]}
    require(set(replay[["job", "held_out_modality", "component"]].itertuples(index=False, name=None)) == expected_replay, "Missing replay cells")
    for job in jobs:
        final_path = root / "results/final" / f"{job['job']}.json"
        require(sha(final_path) == manifest["original_sha256"][f"results/final/{job['job']}.json"], "Original final differs")
        final = json.loads(final_path.read_text(encoding="utf-8"))
        for r in replay[replay.job == job["job"]].itertuples():
            ref = next(v for v in final["aoss_rows"] if v["target_modality"] == r.held_out_modality)
            require(abs(r.original - ref[r.component]) < 1e-12 and abs(r.delta - (r.replayed - r.original)) < 1e-12, "Replay reference differs")
    valid = frame[frame.valid_regions].copy()
    out = root / "results/analysis/paired_response_audit"
    out.mkdir(parents=True, exist_ok=True)
    for label, keys in [("all", ()), ("by_kind", ("perturbation_kind_inferred",)),
                        ("by_inferred_membership", ("training_membership_inferred",))]:
        for name, table in zip(["modality", "fold", "seed", "summary"], summarize(valid, keys)):
            table.to_csv(out / f"{label}_{name}.csv", index=False)
    counts = frame.groupby(["pair", "seed", "held_out_modality", "dataset", "perturbation_kind_inferred"], as_index=False).agg(
        n=("record_index", "size"), valid_regions=("valid_regions", "sum"))
    counts.to_csv(out / "counts.csv", index=False)
    _, _, seeds, summary = summarize(valid)
    pretraining_effects(seeds).to_csv(out / "pretraining_effects.csv", index=False)
    lines = ["# Source paired-response audit", "", "Original checkpoint/selection/AOSS/main scoring unchanged.",
             "Diagnostic identity: raw_local_sensitivity = normal_spatial_contrast + net_local_response.",
             "Equal datasets within modality, equal four source modalities within fold, equal five folds within seed; SD over 3 seeds.",
             "", "|Pair|Normal spatial contrast|Raw local sensitivity|Net response|Net seed SD|", "|---|---|---|---|---|"]
    for r in summary.itertuples():
        lines.append(f"|{r.pair}|{r.normal_spatial_contrast_mean:.5f}|{r.raw_local_sensitivity_mean:.5f}|{r.net_local_response_mean:.5f}|{r.net_local_response_std:.5f}|")
    lines += ["", "Perturbation kinds and training membership are inferred, not historical provenance.",
              "Per-image averages are not a reconstruction of original AOSS aggregation. Net response is not a new selection metric.",
              "Membership strata may omit empty groups; consult counts and training_membership.csv. No patient CI or significance claim.",
              "Conditional source/target effects and interaction: pretraining_effects.csv. Original GG uses a different cut, so contrasts mix backbone and stitch-location differences.",
              "Large server arrays/weights stat-only. Source response does not establish sensitivity to clinical anomalies."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    require(all(sha(Path(p)) == h for p, h in digests.items()), "Input changed during analysis")
    (out / "audit.json").write_text(json.dumps(dict(status="complete", n_rows=len(frame), n_invalid_regions=int((~frame.valid_regions).sum()),
        aoss_replay_checks=len(replay), max_replay_abs_delta=float(replay.delta.abs().max()), input_sha256=digests), indent=2), encoding="utf-8")
    print(out / "report.md")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--input", type=Path, default=Path("results/followup/paired_response_audit"))
    args = p.parse_args()
    root = args.root.resolve()
    analyze(root, (root / args.input).resolve())


if __name__ == "__main__":
    main()
