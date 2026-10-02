import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from src.paired_response_audit import paired_regions, reconstructed_training_images
from src.train_eval import load_source_arrays


class PairedResponseTests(unittest.TestCase):
    def test_normal_spatial_contrast_is_not_perturbation_response(self):
        normal = [[.8, .8, .1, .1]]
        r = paired_regions(normal, normal, [[1, 1, 0, 0]])[0]
        self.assertAlmostEqual(r["raw_local_sensitivity"], .7)
        self.assertAlmostEqual(r["normal_spatial_contrast"], .7)
        self.assertAlmostEqual(r["net_local_response"], 0)

    def test_local_and_global_change_are_distinguished(self):
        normal = [[.2, .2, .1, .1]]
        local = paired_regions(normal, [[.5, .5, .1, .1]], [[1, 1, 0, 0]])[0]
        self.assertAlmostEqual(local["net_local_response"], .3)
        global_change = paired_regions(normal, [[.5, .5, .4, .4]], [[1, 1, 0, 0]])[0]
        self.assertAlmostEqual(global_change["delta_in"], .3)
        self.assertAlmostEqual(global_change["delta_out"], .3)
        self.assertAlmostEqual(global_change["net_local_response"], 0)

    def test_negative_response_is_preserved(self):
        r = paired_regions([[.4, .4, .1, .1]], [[.2, .2, .1, .1]], [[1, 1, 0, 0]])[0]
        self.assertLess(r["net_local_response"], 0)

    def test_empty_regions_are_undefined(self):
        for mask in [[[0, 0]], [[1, 1]]]:
            r = paired_regions([[.1, .2]], [[.2, .3]], mask)[0]
            self.assertFalse(r["valid_regions"])
            self.assertIsNone(r["net_local_response"])

    def test_invalid_inputs_rejected(self):
        for normal, pert, mask in [([[np.nan, .2]], [[.2, .3]], [[1, 0]]),
                                    ([[.1, .2]], [[.2]], [[1, 0]]),
                                    ([[.1, .2]], [[.2, .3]], [[2, 0]])]:
            with self.assertRaises(ValueError): paired_regions(normal, pert, mask)

    def test_membership_exactly_reconstructs_original_loader_with_truncation(self):
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/experiment.yaml").read_text(encoding="utf-8"))
        cfg = copy.deepcopy(cfg)
        cfg["final"]["n_per_source_modality"] = 5  # Exercise ceil + concatenate/truncate.
        records = {ds: [{"image": f"{ds}_{i}"} for i in range(9)] for ds in cfg["data"]["datasets"]}
        base = {ds: i * 100 for i, ds in enumerate(cfg["data"]["datasets"])}
        def array(path):
            ds = path.parent.parent.name
            return (np.arange(9) + base[ds]).reshape(9, 1)
        for held in ["brain_mri", "oct"]:
            reconstructed = reconstructed_training_images(cfg, held, 11, records)
            with patch("src.train_eval._load_memmap", side_effect=array):
                x, _ = load_source_arrays(cfg, held, "medical", "medical", 2, 5, 11)
            encoded = {base[ds] + int(image.rsplit("_", 1)[1]) for ds, names in reconstructed.items() for image in names}
            self.assertEqual(set(x.ravel()), encoded)
            if held == "oct":
                self.assertNotIn("RESC", reconstructed)
                self.assertNotIn("OCT2017", reconstructed)


if __name__ == "__main__":
    unittest.main()
