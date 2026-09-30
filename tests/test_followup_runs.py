import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from src.followup_io import case_indices, common_jobs, protocol_guard, validate_common_result
from src.followup_worker import run_job, source_components, spatial_maps
from src.run_followup import common_command, inputs_unchanged, main, run_parallel
from src.train_eval import evaluate_dataset


def configuration():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/experiment.yaml").read_text(encoding="utf-8"))


def selection():
    items = []
    for source in ["medical", "general"]:
        for target in ["medical", "general"]:
            block = 9 if source == target == "general" else 3
            items.append(dict(config=f"{source}_to_{target}_s2_b{block}", source_backbone=source,
                              target_backbone=target, source_stage=2, target_block=block))
    return dict(selection_metric="source_only_AOSS", selected=items)


def result(cfg, job):
    identity = {k: job[k] for k in ["config", "source_backbone", "target_backbone", "source_stage", "target_block", "seed"]}
    identity["adapter"] = "mlp"
    rows = []
    for ds in cfg["data"]["datasets"]:
        rows.append(dict({k: v for k, v in identity.items() if k != "config"}, dataset=ds,
                         target_modality=cfg["data"]["modality_map"][ds], score_mode="contrast_topk",
                         epochs=8, n_per_source_modality=1000, n_test=4, image_auroc=.75,
                         image_aupr=.8, aoss=1., final_train_loss=.1))
    aoss = [dict(target_modality=m, normal_discrepancy=.1, perturb_in=.3, perturb_out=.2,
                 local_sensitivity=.1, aoss=1.) for m in set(cfg["data"]["modality_map"].values())]
    return dict(identity, rows=rows, aoss_rows=aoss)


class Adapter(torch.nn.Module):
    def forward(self, x):
        return None, x


class Tail(torch.nn.Module):
    def forward(self, prefix, patches):
        return patches


def arrays(directory, records, perturb=False):
    directory.mkdir(parents=True)
    (directory / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    d = np.full((len(records), 16), .1)
    d[:, -1] = [.2, .7, .3, .8][:len(records)]
    x = np.stack([1-d, np.sqrt(1-(1-d)**2)], axis=-1).astype(np.float16)
    y = np.zeros_like(x)
    y[:, :, 0] = 1
    if perturb:
        for prefix in ["normal", "pert"]:
            np.save(directory / f"{prefix}_source_medical_s2.npy", x)
            np.save(directory / f"{prefix}_target_medical.npy", y)
        mask = np.zeros((len(records), 4, 4), dtype=np.uint8)
        mask[:, -1, -1] = 1
        np.save(directory / "mask_medical.npy", mask)
    else:
        np.save(directory / "source_medical_s2.npy", x)
        np.save(directory / "target_medical.npy", y)


class FollowupRunTests(unittest.TestCase):
    def test_common_matrix_reuses_twelve_and_does_not_rank_test_metrics(self):
        cfg, selected = configuration(), selection()
        before = copy.deepcopy(selected)
        jobs = common_jobs(cfg, selected)
        self.assertEqual(len(jobs), 24)
        self.assertEqual(sum(j["reuse_original"] for j in jobs), 12)
        self.assertEqual({(j["source_stage"], j["target_block"]) for j in jobs}, {(2,3),(2,9)})
        self.assertEqual(selected, before)
        for item in selected["selected"]:
            item.update(posthoc_mean_image_auroc=1e9, mean_aoss=-1e9)
        self.assertEqual(jobs, common_jobs(cfg, selected))

    def test_common_points_and_protocol_drift_are_rejected(self):
        cfg = configuration()
        for key, value in [("seeds", [11,44,55]), ("epochs", 9), ("topk_fraction", .1), ("lr", .002)]:
            bad = copy.deepcopy(cfg)
            bad["final"][key] = value
            with self.assertRaises(ValueError):
                protocol_guard(bad)
        selected = selection()
        selected["selected"][-1]["target_block"] = 6
        with self.assertRaises(ValueError):
            common_jobs(cfg, selected)
        bad = copy.deepcopy(cfg)
        bad["data"]["modality_map"]["OCT2017"] = "oct_other"
        with self.assertRaises(ValueError):
            protocol_guard(bad)

    def test_existing_common_output_must_match_budget_and_all_folds(self):
        cfg = configuration()
        job = common_jobs(cfg, selection())[0]
        good = result(cfg, job)
        counts = {d:4 for d in cfg["data"]["datasets"]}
        validate_common_result(cfg, job, good, counts)
        for field, value in [("epochs", 6), ("score_mode", "raw_topk"), ("n_test", 5), ("image_auroc", float("nan"))]:
            bad = copy.deepcopy(good)
            bad["rows"][0][field] = value
            with self.assertRaises(ValueError):
                validate_common_result(cfg, job, bad, counts)

    def test_case_sampling_is_reproducible_balanced_and_deduplicated(self):
        labels = np.repeat([0,1], 40)
        scores = np.column_stack([np.arange(80), np.arange(80)[::-1]])
        cases = case_indices(labels, scores)
        self.assertEqual(cases, case_indices(labels, scores))
        self.assertEqual(len({r["record_index"] for r in cases}), len(cases))
        for label in [0,1]:
            own = [r for r in cases if r["label"] == label]
            self.assertEqual(sum("random" in r["reason"] for r in own), 10)
            self.assertEqual(sum("hard" in r["reason"] for r in own), 5)
            self.assertLessEqual(len(own), 15)
        with self.assertRaises(ValueError):
            case_indices(labels, np.full((80,3), np.nan))

    def test_spatial_export_matches_unchanged_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(paths=dict(cache_dir=tmp), final=dict(batch_size=64, topk_fraction=.05),
                       models=dict(targets=dict(medical=dict(input_size=56))))
            records = [dict(image=f"/{i}.png", label=i%2, mask=None) for i in range(4)]
            directory = Path(tmp) / "features/Toy/test"
            arrays(directory, records)
            target = SimpleNamespace(config=SimpleNamespace(patch_size=14))
            out = Path(tmp) / "maps"
            out.mkdir()
            cases = [dict(dataset="Toy", record_index=i) for i in range(4)]
            actual = spatial_maps(cfg, target, Adapter(), Tail(), "medical", "medical", 2, "Toy", cases, torch.device("cpu"), out)
            original = evaluate_dataset(cfg, target, "medical", "medical", Adapter(), Tail(), "Toy", 2, torch.device("cpu"), .05, batch_size=128)[0]
            self.assertEqual(actual["image_auroc"], original["image_auroc"])
            self.assertEqual(actual["image_aupr"], original["image_aupr"])
            with np.load(out / "Toy_maps.npz", allow_pickle=False) as saved:
                np.testing.assert_allclose(saved["scores"], [.1,.6,.2,.7], atol=1e-3)
                np.testing.assert_allclose(saved["contrast"], np.maximum(saved["discrepancy"]-saved["patch_median"][:,None,None],0))
                np.testing.assert_array_equal(saved["topk_mask"].sum((1,2)), 1)

    def test_source_diagnostics_exclude_both_oct_datasets_together(self):
        cfg = configuration()
        with tempfile.TemporaryDirectory() as tmp:
            cfg["paths"]["cache_dir"] = tmp
            for ds in cfg["data"]["datasets"]:
                records = [dict(image=f"/{ds}/{i}.png", label=0) for i in range(4)]
                arrays(Path(tmp) / "perturb" / ds, records, perturb=True)
            output = Path(tmp) / "source.csv"
            source_components(cfg, Adapter(), Tail(), "medical", "medical", 2, "oct", torch.device("cpu"), output)
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(r["source_modality"] != "oct" for r in rows))
            self.assertFalse({"OCT2017","RESC"} & {r["dataset"] for r in rows})
            self.assertTrue(all(r["kind_evidence"].startswith("inferred") for r in rows))

    def test_diagnostic_worker_only_restores_original_checkpoints(self):
        cfg = configuration()
        with tempfile.TemporaryDirectory() as tmp:
            cfg["paths"]["results_dir"] = tmp
            selected = selection()
            job = common_jobs(cfg, selected)[0]
            reference = result(cfg, job)
            ref = Path(tmp) / "final" / f"{job['job']}.json"
            ref.parent.mkdir()
            ref.write_text(json.dumps(reference))
            (Path(tmp) / "selected_configs.json").write_text(json.dumps(selected))
            root = ref.parent / "checkpoints" / job["job"]
            root.mkdir(parents=True)
            for modality in set(cfg["data"]["modality_map"].values()):
                torch.save(dict(adapter={}, source_name="medical", target_name="medical", source_stage=2,
                                target_block=3, adapter_type="mlp", target_modality=modality), root / f"{modality}.pt")
            out = Path(tmp) / "followup/diagnostics/jobs" / job["job"]
            plan = out.parent.parent / "case_plan.json"
            plan.parent.mkdir(parents=True)
            plan.write_text(json.dumps(dict(cases=[])))
            before = ref.read_bytes()
            def maps(*args, **kwargs):
                return {k: next(r for r in reference["rows"] if r["dataset"] == args[7])[k]
                        for k in ["dataset","score_mode","n_test","image_auroc","image_aupr"]}
            def aoss(*args):
                return {k: .1 if k in {"normal_discrepancy","local_sensitivity"} else .3 if k == "perturb_in" else .2 if k == "perturb_out" else 1.
                        for k in ["normal_discrepancy","local_sensitivity","perturb_in","perturb_out","aoss"]}
            with patch("src.followup_worker.load_target", return_value=object()), patch("src.followup_worker.build_stitch_modules", return_value=(Adapter(),Tail())), patch("src.followup_worker.spatial_maps", side_effect=maps), patch("src.followup_worker.source_components"), patch("src.followup_worker.compute_aoss", side_effect=aoss), patch("src.train_eval.train_adapter", side_effect=AssertionError("training forbidden")), patch("src.pipeline.select_configs", side_effect=AssertionError("selection forbidden")):
                run_job(cfg, str(ref), str(plan), out, torch.device("cpu"))
            self.assertEqual(ref.read_bytes(), before)
            evaluation = json.loads((out / "evaluation.json").read_text())
            self.assertTrue(evaluation["no_training"] and evaluation["no_reselection"] and evaluation["replay_matches_original"])

    def test_spatial_replay_detects_score_drift_even_if_ranks_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(paths=dict(cache_dir=tmp), final=dict(batch_size=64, topk_fraction=.05),
                       models=dict(targets=dict(medical=dict(input_size=56))))
            records = [dict(image=f"/{i}.png", label=i%2, mask=None) for i in range(4)]
            arrays(Path(tmp) / "features/Toy/test", records)
            out = Path(tmp) / "maps"
            out.mkdir()
            prediction = Path(tmp) / "scores.csv"
            with prediction.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["image_path","label","score"])
                writer.writeheader()
                writer.writerows(dict(image_path=r["image"],label=r["label"],score=s+.01)
                                 for r,s in zip(records,[.1,.6,.2,.7]))
            with self.assertRaisesRegex(RuntimeError, "scores differ"):
                spatial_maps(cfg, SimpleNamespace(config=SimpleNamespace(patch_size=14)), Adapter(), Tail(), "medical", "medical", 2,
                             "Toy", [dict(dataset="Toy",record_index=i) for i in range(4)], torch.device("cpu"), out,
                             prediction_reference=prediction)

    def test_common_command_retains_budget_and_isolated_destination(self):
        cfg = configuration()
        job = next(j for j in common_jobs(cfg, selection()) if not j["reuse_original"])
        output = "results/followup/common_stitch/job.json"
        command = common_command(cfg, "configs/experiment.yaml", job, output)
        for key, value in [("--epochs","8"), ("--n-per-modality","1000"), ("--score-modes","contrast_topk"), ("--output",output)]:
            self.assertEqual(command[command.index(key)+1], value)
        self.assertNotIn("--skip-target-eval", command)

    def test_check_only_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = configuration()
            cfg["paths"]["results_dir"] = str(Path(tmp) / "results")
            config = Path(tmp) / "cfg.yaml"
            config.write_text(yaml.safe_dump(cfg))
            with patch("sys.argv", ["run_followup","--stage","diagnostics","--config",str(config),"--check-only"]), patch("src.run_followup.preflight", return_value=([],[],[],{})), patch("src.run_followup.run_parallel", side_effect=AssertionError("workers forbidden")):
                main()
            self.assertFalse(Path(cfg["paths"]["results_dir"]).exists())

    def test_scheduler_terminates_all_other_children_on_failure(self):
        class Process:
            def __init__(self, failed):
                self.failed, self.terminated = failed, False
            def poll(self):
                return 1 if self.failed else -15 if self.terminated else None
            def terminate(self):
                self.terminated = True
            def wait(self, timeout=None):
                return self.poll()
        first, second = Process(True), Process(False)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.run_followup.subprocess.Popen", side_effect=[first,second]), patch("src.run_followup.time.sleep"):
                with self.assertRaises(RuntimeError):
                    run_parallel([("a",["fake"]),("b",["fake"])], [0,1], Path(tmp))
            self.assertTrue(second.terminated)

    def test_original_input_mutation_is_detected(self):
        from src.prediction_io import sha256
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected.json"
            path.write_text("original")
            hashes = {str(path):sha256(path)}
            self.assertTrue(inputs_unchanged(hashes, {}))
            path.write_text("changed")
            self.assertFalse(inputs_unchanged(hashes, {}))


if __name__ == "__main__":
    unittest.main()
