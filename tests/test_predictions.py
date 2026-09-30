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

from src.prediction_io import (build_plan, checkpoint_identity, compare_metrics,
                               sha256, validate_final, write_predictions)
from src.train_eval import evaluate_dataset


def fixture():
    cfg = dict(data=dict(datasets=["Toy"], modality_map={"Toy": "oct"}),
               final=dict(seeds=[11], adapter="mlp", epochs=8, n_per_source_modality=1000, topk_fraction=.05))
    item = dict(config="medical_to_medical_s2_b3", source_backbone="medical", target_backbone="medical", source_stage=2, target_block=3)
    row = dict(dataset="Toy", target_modality="oct", score_mode="contrast_topk", seed=11, adapter="mlp", epochs=8, n_per_source_modality=1000,
               source_backbone="medical", target_backbone="medical", source_stage=2, target_block=3, image_auroc=.75, image_aupr=.8, n_test=4)
    final = dict(item, seed=11, adapter="mlp", rows=[row])
    selected = dict(selection_metric="source_only_AOSS", selected=[item])
    return cfg, selected, final


class PredictionTests(unittest.TestCase):
    def test_locked_selection_ignores_posthoc_metrics(self):
        cfg, selected, final = fixture()
        args = ("medical", "medical", 2, 3, 11, "mlp", .05, ["contrast_topk"])
        validate_final(cfg, selected, final, *args)
        selected["selected"][0]["posthoc_mean_image_auroc"] = 1e9
        validate_final(cfg, selected, final, *args)
        selected["selected"][0]["config"] = "medical_to_medical_s1_b9"
        with self.assertRaises(ValueError):
            validate_final(cfg, selected, final, *args)

    def test_protocol_drift_is_rejected(self):
        cfg, selected, final = fixture()
        for seed, fraction, modes in [(22, .05, ["contrast_topk"]), (11, .1, ["contrast_topk"]), (11, .05, ["raw_topk"])]:
            with self.assertRaises(ValueError):
                validate_final(cfg, selected, final, "medical", "medical", 2, 3, seed, "mlp", fraction, modes)
        final["rows"][0]["epochs"] = 6
        with self.assertRaises(ValueError):
            validate_final(cfg, selected, final, "medical", "medical", 2, 3, 11, "mlp", .05, ["contrast_topk"])

    def test_checkpoint_wrong_fold_or_stitch_rejected(self):
        ckpt = dict(adapter={}, source_name="medical", target_name="medical", source_stage=2, target_block=3, adapter_type="mlp", target_modality="oct")
        args = ("medical", "medical", 2, 3, "mlp", "oct")
        checkpoint_identity(ckpt, *args)
        for field, value in [("target_modality", "brain_mri"), ("source_stage", 1), ("target_name", "general")]:
            bad = dict(ckpt, **{field: value})
            with self.assertRaises(ValueError):
                checkpoint_identity(bad, *args)

    def test_metric_mismatch_and_count_are_detected(self):
        _, _, final = fixture()
        old = final["rows"]
        self.assertTrue(all(r["within_tolerance"] for r in compare_metrics(old, old, 1e-6)))
        actual = copy.deepcopy(old)
        actual[0]["image_auroc"] += .01
        self.assertFalse(all(r["within_tolerance"] for r in compare_metrics(actual, old, 1e-6)))
        actual[0]["n_test"] += 1
        with self.assertRaises(ValueError):
            compare_metrics(actual, old, 1e-6)

    def test_csv_preserves_score_precision_and_record_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "pred.csv"
            records = [dict(image="/b.png", label=1), dict(image="/a.png", label=0)]
            scores = [.12345678901234567, .00000000023456789]
            stats = [dict(discrepancy_mean=.1, discrepancy_median=.1, discrepancy_max=.2)] * 2
            write_predictions(p, records, scores, dict(dataset="Toy"), stats)
            with p.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["image_path"] for r in rows], [r["image"] for r in records])
            self.assertEqual([float(r["score"]) for r in rows], scores)
            with self.assertRaises(ValueError):
                write_predictions(p, records, [np.nan, .1], dict(dataset="Toy"), stats)

    def test_real_evaluation_scores_unchanged_by_export(self):
        class Adapter(torch.nn.Module):
            def forward(self, x):
                return None, x

        class Tail(torch.nn.Module):
            def forward(self, prefix, patches):
                return patches

        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(paths=dict(cache_dir=tmp), models=dict(targets=dict(medical=dict(input_size=56))))
            root = Path(tmp) / "features/Toy/test"
            root.mkdir(parents=True)
            records = [dict(image=f"/{i}.png", label=i % 2, mask=None) for i in range(4)]
            (root / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records))
            d = np.full((4, 16), .1)
            d[:, -1] = [.2, .7, .3, .8]
            features = np.stack([1 - d, np.sqrt(1 - (1 - d) ** 2)], axis=-1).astype(np.float16)
            ref = np.zeros_like(features)
            ref[:, :, 0] = 1
            np.save(root / "source_medical_s2.npy", features)
            np.save(root / "target_medical.npy", ref)
            target = SimpleNamespace(config=SimpleNamespace(patch_size=14))
            args = (cfg, target, "medical", "medical", Adapter(), Tail(), "Toy", 2, torch.device("cpu"), .05)
            original = evaluate_dataset(*args, batch_size=2)
            csv_path = Path(tmp) / "pred.csv"
            before = sha256(root / "records.jsonl")
            exported = evaluate_dataset(*args, batch_size=2, prediction_path=csv_path,
                                         prediction_metadata=dict(dataset="Toy"), evaluate_pixels=False)
            self.assertEqual(original, exported)
            self.assertEqual(before, sha256(root / "records.jsonl"))
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            np.testing.assert_allclose([float(r["score"]) for r in rows], [.1, .6, .2, .7], atol=1e-3)
            self.assertEqual(exported[0]["image_auroc"], 1)
            np.save(root / "target_medical.npy", ref[:3])
            with self.assertRaisesRegex(ValueError, "length mismatch"):
                evaluate_dataset(*args)

    def test_plan_requires_saved_checkpoints_and_never_writes(self):
        cfg, selected, final = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            result, cache, model = Path(tmp) / "results", Path(tmp) / "cache", Path(tmp) / "model"
            cfg["paths"] = dict(results_dir=str(result), cache_dir=str(cache))
            cfg["models"] = dict(pairs=[["medical", "medical"]], targets=dict(medical=dict(local=str(model))))
            (result / "final").mkdir(parents=True)
            (result / "selected_configs.json").write_text(json.dumps(selected))
            job = final["config"] + "_seed11"
            (result / "final" / f"{job}.json").write_text(json.dumps(final))
            before = {str(p): sha256(p) for p in result.rglob("*") if p.is_file()}
            with self.assertRaisesRegex(FileNotFoundError, "no training fallback"):
                build_plan(cfg)
            checkpoint = result / "final/checkpoints" / job
            checkpoint.mkdir(parents=True)
            (checkpoint / "oct.pt").write_bytes(b"test presence only")
            test = cache / "features/Toy/test"
            test.mkdir(parents=True)
            (test / ".done").touch()
            (test / "records.jsonl").write_text("\n".join(json.dumps(dict(image=f"/test{i}.png", dataset="Toy", modality="oct", split="test", label=i % 2)) for i in range(4)))
            np.save(test / "source_medical_s2.npy", np.zeros((4, 1)))
            np.save(test / "target_medical.npy", np.zeros((4, 1)))
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "model.safetensors").touch()
            jobs = build_plan(cfg)
            self.assertEqual(jobs[0]["job"], job)
            self.assertTrue(all(sha256(p) == h for p, h in before.items()))
            self.assertFalse((result / "predictions").exists())

    def test_saved_replay_uses_existing_checkpoint_without_training_or_aoss(self):
        from src.worker import run_saved_eval

        cfg, selected, final = fixture()
        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

        with tempfile.TemporaryDirectory() as tmp:
            results, cache = Path(tmp) / "results", Path(tmp) / "cache"
            cfg["paths"] = dict(results_dir=str(results), cache_dir=str(cache))
            cfg["project"] = dict(allow_tf32=False)
            cfg["runtime"] = dict(deterministic=False)
            job = final["config"] + "_seed11"
            checkpoint_root = results / "final/checkpoints" / job
            checkpoint_root.mkdir(parents=True)
            reference = results / "final" / f"{job}.json"
            reference.write_text(json.dumps(final))
            selection = results / "selected_configs.json"
            selection.write_text(json.dumps(selected))
            records = cache / "features/Toy/test/records.jsonl"
            records.parent.mkdir(parents=True)
            records.write_text("{}")
            config = Path(tmp) / "config.yaml"
            config.write_text(yaml.safe_dump(cfg))
            ckpt = dict(adapter={}, source_name="medical", target_name="medical", source_stage=2, target_block=3, adapter_type="mlp", target_modality="oct")
            torch.save(ckpt, checkpoint_root / "oct.pt")
            pred = results / "predictions" / job
            before = {str(p): sha256(p) for p in [reference, selection, checkpoint_root / "oct.pt", records]}
            args = SimpleNamespace(prediction_dir=str(pred), reference_final=str(reference), checkpoint_root=str(checkpoint_root),
                                   source_name="medical", target_name="medical", source_stage=2, target_block=3,
                                   seed=11, adapter="mlp", topk_fraction=.05, score_modes=["contrast_topk"], gpu=0,
                                   batch_size=64, config=str(config), replay_tolerance=1e-6)
            def evaluate(*pos, **kwargs):
                self.assertFalse(kwargs["evaluate_pixels"])
                self.assertEqual(kwargs["prediction_metadata"]["seed"], 11)
                Path(kwargs["prediction_path"]).parent.mkdir(parents=True, exist_ok=True)
                Path(kwargs["prediction_path"]).write_text("synthetic prediction receipt")
                return [dict(dataset="Toy", score_mode="contrast_topk", n_test=4, image_auroc=.75, image_aupr=.8)]
            with patch("src.worker.load_target", return_value=object()), patch("src.worker.build_stitch_modules", return_value=(Module(), Module())), patch("src.worker.evaluate_dataset", side_effect=evaluate), patch("src.worker.train_adapter", side_effect=AssertionError("training forbidden")), patch("src.worker.compute_aoss", side_effect=AssertionError("AOSS forbidden")):
                result = run_saved_eval(cfg, args)
            self.assertTrue(result["replay_matches_original"])
            self.assertTrue(result["no_reselection"])
            self.assertTrue(all(sha256(p) == h for p, h in before.items()))


if __name__ == "__main__":
    unittest.main()
