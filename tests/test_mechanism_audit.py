import unittest
from unittest.mock import patch

import torch
from transformers import Dinov2Config, Dinov2Model

from src.mechanism_audit import gradient_probe, loss_components, sample_indices, source_datasets, tail_identity
from src.models import StitchAdapter, TargetTail, _prefix_and_patch_position


class MechanismTests(unittest.TestCase):
    def model(self):
        # Synthetic local unit-test model only; production always loads actual local weights.
        torch.manual_seed(17)
        model = Dinov2Model(Dinov2Config(image_size=28, patch_size=14, hidden_size=24,
                                       num_hidden_layers=12, num_attention_heads=4, mlp_ratio=2)).eval()
        for p in model.parameters(): p.requires_grad_(False)
        return model

    def test_real_dinov2_layer_interface_and_final_norm(self):
        model = self.model()
        expected, checks = tail_identity(model, torch.randn(2, 3, 28, 28))
        self.assertEqual(tuple(expected.shape), (2, 4, 24))
        self.assertTrue(all(r["within_tolerance"] for r in checks))
        self.assertEqual([r["first_human_block"] for r in checks], [4, 7, 10])
        self.assertTrue(all(not layer._forward_pre_hooks for layer in model.encoder.layer))

    def test_wrong_cut_is_detected(self):
        model = self.model()
        with patch("src.mechanism_audit.TargetTail", side_effect=lambda t, c: TargetTail(t, c + 1)):
            _, checks = tail_identity(model, torch.randn(2, 3, 28, 28))
        self.assertFalse(all(r["within_tolerance"] for r in checks))

    def test_gradient_reaches_adapter_without_changing_weights(self):
        model = self.model()
        prefix, pos = _prefix_and_patch_position(model, 28)
        adapter = StitchAdapter(8, 24, 2, prefix, pos)
        tail = TargetTail(model, 3).eval()
        before = {k: v.clone() for k, v in adapter.state_dict().items()}
        result = gradient_probe(adapter, tail, torch.randn(2, 8, 3, 3), torch.randn(2, 4, 24))
        self.assertGreater(result["adapter_gradient_norm"], 0)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in adapter.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_unfrozen_backbone_rejected(self):
        model = self.model()
        next(model.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "not frozen"):
            gradient_probe(torch.nn.Linear(1, 1), TargetTail(model, 3), None, None)

    def test_oct_fold_excludes_both_datasets(self):
        mapping = dict(Brain="brain_mri", RESC="oct", OCT2017="oct", liver="liver_ct")
        self.assertEqual(source_datasets(mapping, "oct"), ["Brain", "liver"])
        with self.assertRaises(ValueError): source_datasets(mapping, "unknown")

    def test_pairing_control_detects_destroyed_correspondence(self):
        data = torch.eye(4).reshape(4, 1, 4).expand(-1, 3, -1)
        self.assertLess(float(loss_components(data, data)[2]), 1e-6)
        self.assertGreater(float(loss_components(data.roll(1, 0), data)[2]), 1)
        self.assertEqual(len(set(sample_indices(100))), 16)
        with self.assertRaises(ValueError): sample_indices(15)


if __name__ == "__main__":
    unittest.main()
