import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
import torch

from src.analyze_screen_response import analyze, selector_diagnostics, summarize_grid, validate_frame
from src.screen_response_io import COMPONENTS, MAPPING, MODALITIES, grid_jobs, screen_plan, validate_screen


ROOT = Path(__file__).resolve().parents[1]


def setup_plan():
    cfg = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    selected = dict(selection_metric="source_only_AOSS", selected=[
        dict(config=f"{s}_to_{t}_s2_b{9 if s == t == 'general' else 3}",
             source_backbone=s, target_backbone=t, source_stage=2,
             target_block=9 if s == t == "general" else 3)
        for s, t in cfg["models"]["pairs"]])
    return cfg, selected


def reference(job):
    aoss = dict(normal_discrepancy=.1, perturb_in=.3, perturb_out=.1,
                local_sensitivity=.2, aoss=.2 / (.1 + 1e-8), train_loss=.1)
    return dict(**{k: v for k, v in job.items() if k != "job"}, adapter="mlp", rows=[],
                aoss_rows=[dict(target_modality=h, **aoss) for h in MODALITIES])


class ScreenResponseTests(unittest.TestCase):
    def test_grid_uses_only_original_screen_budget(self):
        cfg, selected = setup_plan()
        jobs = grid_jobs(cfg, selected)
        self.assertEqual(len(jobs), 36)
        self.assertEqual({j["seed"] for j in jobs}, {11})
        bad = copy.deepcopy(cfg); bad["screen"]["epochs"] = 8
        with self.assertRaisesRegex(ValueError, "screening budget"):
            grid_jobs(bad, selected)
        ref = reference(jobs[0]); ref["rows"] = [dict(image_auroc=.9)]
        with self.assertRaisesRegex(ValueError, "target evaluation"):
            validate_screen(ref, jobs[0])

    def test_missing_original_checkpoint_cannot_trigger_training(self):
        cfg, selected = setup_plan()
        with tempfile.TemporaryDirectory() as temp:
            cfg["paths"]["results_dir"] = temp
            directory = Path(temp) / "screen"; directory.mkdir()
            for job in grid_jobs(cfg, selected):
                (directory / f"{job['job']}.json").write_text(json.dumps(reference(job)))
            with self.assertRaisesRegex(ValueError, "no retraining"):
                screen_plan(cfg, selected)

    def test_oct_weighting_and_negative_responses_are_preserved(self):
        rows = []
        for ds, mod, value in [("RESC", "oct", -1), ("OCT2017", "oct", -1),
                               ("liver", "liver_ct", 1), ("RSNA", "chest_xray", 1), ("camelyon16", "pathology", 1)]:
            rows.append(dict(job="j", pair="MM", source_stage=2, target_block=3,
                held_out_modality="brain_mri", source_modality=mod, dataset=ds,
                **{k: value for k in ["normal_spatial_contrast", "raw_local_sensitivity", "delta_in", "delta_out",
                                      "net_local_response", "mean_delta", "contrast_delta"]}))
        ds, mod, fold, point = summarize_grid(pd.DataFrame(rows))
        self.assertEqual(fold.net_local_response.iloc[0], .5)
        self.assertEqual(fold.net_negative_fraction.iloc[0], .25)

    def test_random_regret_is_exact_expectation_not_reselection(self):
        cfg, selected = setup_plan()
        jobs = [j for j in grid_jobs(cfg, selected) if j["source_backbone"] == j["target_backbone"] == "medical"]
        table = pd.DataFrame([dict(job=j["job"], pair="MM", held_out_modality="oct", image_auroc=i / 10)
                              for i, j in enumerate(jobs)])
        original = copy.deepcopy(selected)
        result = selector_diagnostics(table, selected).iloc[0]
        self.assertAlmostEqual(result.random_expected_auroc, .4)
        self.assertAlmostEqual(result.random_expected_regret, .4)
        self.assertEqual(selected, original)

    def test_export_uses_checkpoint_stage_and_never_heldout_cache(self):
        from src.screen_response_audit import export_fold
        cfg, selected = setup_plan()
        job = next(j for j in grid_jobs(cfg, selected) if j["source_stage"] == 3)
        records = [dict(image=f"image/{i}.png") for i in range(256)]
        paths = []
        def array(path, **kwargs):
            paths.append(str(path))
            if Path(path).name.startswith("mask_"):
                mask = np.zeros((256, 256), dtype=np.uint8); mask[:, :10] = 1
                return mask
            return np.zeros((256, 1), dtype=np.float16)
        with tempfile.TemporaryDirectory() as temp, patch("src.screen_response_audit.read_records", return_value=records), \
                patch("src.screen_response_audit.np.load", side_effect=array), \
                patch("src.screen_response_audit.discrepancy", return_value=torch.full((256, 256), .1)):
            count = export_fold(cfg, job, "oct", None, None, torch.device("cpu"), Path(temp) / "fold.csv")
        self.assertEqual(count, 4 * 256)
        self.assertTrue(all("RESC" not in p and "OCT2017" not in p for p in paths))
        self.assertTrue(all("_s3.npy" in p for p in paths if "_source_" in p))
        self.assertTrue(all("perturb" in p and "test" not in p and "valid" not in p for p in paths))

    def test_full_export_audit_and_heldout_contamination(self):
        cfg, selected = setup_plan()
        jobs = grid_jobs(cfg, selected)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); inp = root / "results/followup/screen_response_audit"
            references = inp / "references/screen"; references.mkdir(parents=True)
            (inp / "references/experiment.yaml").write_text(yaml.safe_dump(cfg))
            (inp / "references/selected_configs.json").write_text(json.dumps(selected))
            originals = {"configs/experiment.yaml": hashlib.sha256((inp / "references/experiment.yaml").read_bytes()).hexdigest(),
                         "results/selected_configs.json": hashlib.sha256((inp / "references/selected_configs.json").read_bytes()).hexdigest()}
            checks = []
            first_frame = None
            for job in jobs:
                ref = reference(job); p = references / f"{job['job']}.json"
                p.write_text(json.dumps(ref))
                originals[f"results/screen/{job['job']}.json"] = hashlib.sha256(p.read_bytes()).hexdigest()
                metric_rows = [dict(dataset=ds, target_modality=h, score_mode="contrast_topk", n_test=10,
                    image_auroc=.2 + job["source_stage"] * .05 + job["target_block"] * .01, image_aupr=.5,
                    **{k: job[k] for k in ["source_backbone", "target_backbone", "source_stage", "target_block", "seed"]})
                    for ds, h in MAPPING.items()]
                path = root / "results/stitchmap" / f"{job['job']}.json"; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(config=job["config"], source_backbone=job["source_backbone"],
                                                target_backbone=job["target_backbone"], rows=metric_rows)))
                for h in MODALITIES:
                    frames = []
                    for ds, mod in MAPPING.items():
                        if mod == h: continue
                        f = pd.DataFrame(dict(record_index=np.arange(256),
                            image_path=[f"{ds}/{i}.png" for i in range(256)],
                            perturbation_kind_inferred=np.array(["intensity", "blur", "patch_copy"])[(2026 + np.arange(256)) % 3]))
                        for k, v in dict(**job, held_out_modality=h, source_modality=mod, dataset=ds,
                            n_mask_in=10, n_mask_out=246, valid_regions=True, normal_in=.1, normal_out=.1,
                            perturb_in=.3, perturb_out=.1, normal_spatial_contrast=0., raw_local_sensitivity=.2,
                            delta_in=.2, delta_out=0., net_local_response=.2, normal_mean=.1,
                            perturb_mean=.1 + .2 * 10 / 256, mean_delta=.2 * 10 / 256,
                            normal_contrast_topk=.05, perturb_contrast_topk=.25, contrast_delta=.2).items(): f[k] = v
                        frames.append(f)
                    frame = pd.concat(frames, ignore_index=True)
                    path = inp / "jobs" / job["job"] / f"source_{h}.csv"; path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_csv(path, index=False)
                    if first_frame is None: first_frame = (frame, job, h)
                    for c in COMPONENTS:
                        value = ref["aoss_rows"][0][c]
                        checks.append(dict(job=job["job"], held_out_modality=h, component=c,
                                           original=value, replayed=value, delta=0., within_tolerance=True))
            pd.DataFrame(checks).to_csv(inp / "aoss_replay.csv", index=False)
            manifest = dict(status="complete", protocol="original_screen_diagnostic", original_inputs_unchanged=True,
                no_training=True, no_reselection=True, target_test_read=False, target_valid_read=False, original_score="contrast_topk",
                topk_fraction=.05, batch_size=256, replay_tolerance=1e-6, jobs=jobs, n_source_rows=221184,
                original_sha256=originals, output_sha256={p.relative_to(inp).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                                        for p in inp.rglob("*") if p.is_file()})
            (inp / "run_manifest.json").write_text(json.dumps(manifest))
            analyze(root, inp)
            report = json.loads((root / "results/analysis/screen_response_audit/audit.json").read_text())
            self.assertEqual(report["n_posthoc_cells"], 180)
            self.assertEqual(report["aoss_replay_checks"], 900)
            self.assertFalse(report["selection_updated"])
            self.assertFalse((root / "results/selected_configs.json").exists())
            frame, job, h = first_frame
            bad = frame.copy(); bad.loc[0, "source_modality"] = h
            with self.assertRaisesRegex(ValueError, "Held-out contamination"):
                validate_frame(bad, job, h)


if __name__ == "__main__":
    unittest.main()
