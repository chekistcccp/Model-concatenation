"""Small real-module tests; no model downloads or scientific GPU claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torchvision.models import resnet50
from transformers import Dinov2Config, Dinov2Model

from src.cache_features import _normalize
from src.composed_detector import (ComposedAnomalyDetector, DirectFeatureTail, load_bundle,
                                   save_bundle, source_prefix, tensor_digest)
from src.models import build_stitch_modules
from src.method_controls import evaluate_phase, fit, initialize, training_records
from src.analyze_method_controls import summaries, ARMS, PAIRS, DATASETS


def fixture():
    spec = dict(kind="radimagenet_resnet50", input_size=28, channel_order="bgr",
                normalization_mean=[.5]*3, normalization_std=[.5]*3, stage_channels=[512,1024,2048])
    cfg = dict(models=dict(sources=dict(medical=spec), targets=dict(medical=dict(input_size=28,
               normalization_mean=[.4]*3, normalization_std=[.3]*3))), project=dict(compile_tail=False),
               runtime=dict(amp=False, deterministic=True),
               final=dict(epochs=2, batch_size=2, lr=.001, weight_decay=.0001))
    target = Dinov2Model(Dinov2Config(image_size=28, patch_size=14, hidden_size=16,
                         num_hidden_layers=3, num_attention_heads=4, intermediate_size=32)).eval()
    for p in target.parameters(): p.requires_grad_(False)
    job = dict(source_backbone="medical", target_backbone="medical", source_stage=1,
               target_block=1, seed=11, job="fixture_seed11")
    return cfg, target, job


class CompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_live_path_shares_inherited_parameters_and_matches_legacy(self):
        cfg, target, job = fixture()
        source = torch.nn.Module(); source.backbone = torch.nn.Sequential(*list(resnet50(weights=None).children())[:9])
        source.eval()
        adapter, tail = build_stitch_modules(cfg, target, "medical", "medical", 1, 1, "mlp")
        model = ComposedAnomalyDetector(source_prefix(source, 1), adapter, target,
                    cfg["models"]["sources"]["medical"], cfg["models"]["targets"]["medical"], 1)
        rgb = torch.rand(2,3,28,28)
        with torch.inference_mode():
            fmap = source.backbone[:6](_normalize(rgb, model.source_spec))
            prefix, patches = adapter(fmap)
            expected = tail(prefix, patches)
            reference = target(_normalize(rgb, model.target_spec)).last_hidden_state[:,1:]
            discrepancy = 1 - torch.nn.functional.cosine_similarity(expected.float(), reference.float(), dim=-1)
        actual = model(rgb)
        torch.testing.assert_close(actual["discrepancy"].flatten(1), discrepancy, rtol=0, atol=0)
        self.assertIs(model.tail.layers[0], target.encoder.layer[1])
        self.assertFalse(hasattr(model.tail, "target"))
        counts = model.parameter_counts()
        self.assertEqual(counts["total_unique"], counts["cnn_prefix"]+counts["adapter"]+counts["full_reference"])
        self.assertTrue(all(not p.requires_grad for p in model.prefix.parameters()))
        with self.assertRaises(ValueError): model(rgb+2)

    def test_self_contained_bundle_round_trip_without_original_weights(self):
        cfg, target, job = fixture()
        cfg["models"]["sources"]["medical"]["checkpoint"] = "does-not-exist.pt"
        source = torch.nn.Module(); source.backbone = torch.nn.Sequential(*list(resnet50(weights=None).children())[:9])
        source.eval()
        adapter, _ = build_stitch_modules(cfg, target, "medical", "medical", 1, 1, "mlp")
        model = ComposedAnomalyDetector(source_prefix(source, 1), adapter, target,
                    cfg["models"]["sources"]["medical"], cfg["models"]["targets"]["medical"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"
            save_bundle(path, model, cfg, job, "brain_mri", {})
            restored, payload = load_bundle(path, torch.device("cpu"))
            self.assertIn("patch_position", payload["states"]["adapter"])
            rgb = torch.rand(2,3,28,28)
            torch.testing.assert_close(restored(rgb)["score"], model(rgb)["score"], rtol=0, atol=0)
            with self.assertRaises(FileExistsError): save_bundle(path, model, cfg, job, "brain_mri", {})
            payload["states"]["adapter"]["prefix_tokens"].add_(1)
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "Corrupt"): load_bundle(path, torch.device("cpu"))

    def test_matched_initialization_normal_training_and_no_tail_effective_count(self):
        cfg, target, job = fixture()
        a, tail = initialize(cfg, target, job, "oct", "matched_tail", torch.device("cpu"))
        b, direct = initialize(cfg, target, job, "oct", "no_tail", torch.device("cpu"))
        self.assertEqual(tensor_digest(a.state_dict()), tensor_digest(b.state_dict()))
        self.assertIsInstance(direct, DirectFeatureTail)
        self.assertFalse(hasattr(direct, "layers"))
        before = tensor_digest(b.state_dict())
        x = np.random.default_rng(1).normal(size=(4,512,4,4)).astype("float16")
        y = np.random.default_rng(2).normal(size=(4,4,16)).astype("float16")
        proof = fit(cfg, b, direct, x, y, 11, torch.device("cpu"))
        self.assertEqual(proof["optimizer_steps"], 4)
        self.assertNotEqual(before, tensor_digest(b.state_dict()))
        self.assertEqual(proof["effective_trainable_parameters"], sum(p.numel() for n,p in b.named_parameters() if n != "prefix_tokens"))
        self.assertTrue(all(p.grad is None for p in target.parameters()))


class MethodProtocolTests(unittest.TestCase):
    def test_entire_oct_modality_excluded_and_order_matches_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            mapping = dict(Brain="brain_mri", liver="liver_ct", RESC="oct", OCT2017="oct", RSNA="chest_xray", camelyon16="pathology")
            cfg = dict(paths=dict(cache_dir=tmp), final=dict(n_per_source_modality=4), data=dict(datasets=list(mapping), modality_map=mapping))
            for ds, mod in mapping.items():
                if mod == "oct": continue  # Should never need either excluded dataset.
                p=Path(tmp)/"features"/ds/"train/records.jsonl";p.parent.mkdir(parents=True)
                p.write_text("\n".join(json.dumps(dict(image=f"{ds}/{i}", modality=mod, label=0, split="train")) for i in range(7)))
            rows = training_records(cfg, dict(seed=11), "oct")
            self.assertEqual(len(rows), 16)
            self.assertEqual({r["dataset"] for r in rows}, {"Brain","liver","RSNA","camelyon16"})

    def test_target_evaluation_blocked_before_all_source_training(self):
        with tempfile.TemporaryDirectory() as tmp, patch("src.method_controls.load_target") as loading:
            with self.assertRaisesRegex(ValueError, "Finish all source training"):
                evaluate_phase({}, [], Path(tmp), torch.device("cpu"))
            loading.assert_not_called()

    def test_macro_and_matched_tail_contrast_are_paired_by_seed(self):
        rows=[]
        for ai,arm in enumerate(ARMS):
            for pair in PAIRS:
                s,t=pair.split("_to_")
                for seed in [11,22,33]:
                    for i,ds in enumerate(DATASETS):
                        rows.append(dict(arm=arm, source_backbone=s, target_backbone=t, seed=seed, dataset=ds,
                                         image_auroc=.5+ai*.01+i*.001, image_aupr=.1+ai*.01))
        seed_rows, summary, diffs = summaries(rows)
        r=next(r for r in summary if r["arm"]=="matched_tail" and r["pair"]==PAIRS[0] and r["group"]=="overall" and r["metric"]=="image_auroc")
        self.assertAlmostEqual(r["mean"], .5125)
        d=next(r for r in diffs if r["contrast"]=="matched_tail_minus_no_tail")
        self.assertAlmostEqual(d["delta"], -.01)
        self.assertEqual(r["n_seeds"], 3)


if __name__ == "__main__": unittest.main()
