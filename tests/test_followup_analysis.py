import unittest

import numpy as np

from src.analyze_followup_results import map_statistics


class SpatialAuditTests(unittest.TestCase):
    def fixture(self):
        raw = np.arange(256, dtype=np.float32).reshape(1, 16, 16) / 256
        contrast = np.maximum(raw - 127 / 256, 0)
        top = np.zeros_like(raw, dtype=np.uint8)
        top.reshape(-1)[-13:] = 1
        return dict(discrepancy=raw, contrast=contrast, topk_mask=top,
                    patch_median=np.array([127 / 256]), scores=np.array([contrast.reshape(-1)[-13:].mean()]),
                    record_indices=np.array([5]), labels=np.array([1]),
                    ground_truth_mask=np.zeros_like(top), has_ground_truth_mask=np.array([False]))

    def test_lower_median_and_missing_mask(self):
        row = map_statistics(self.fixture())[0]
        self.assertEqual(row["median"], 127 / 256)
        self.assertTrue(np.isnan(row["topk_gt_precision"]))

    def test_average_median_is_rejected(self):
        z = self.fixture()
        z["patch_median"] = np.array([127.5 / 256])
        with self.assertRaisesRegex(ValueError, "lower median"):
            map_statistics(z)

    def test_wrong_hotspots_are_rejected(self):
        z = self.fixture()
        z["topk_mask"] = np.flip(z["topk_mask"], axis=1)
        with self.assertRaisesRegex(ValueError, "mask mismatch"):
            map_statistics(z)

    def test_nonempty_gt_and_empty_gt_are_distinct(self):
        z = self.fixture()
        z["has_ground_truth_mask"][:] = True
        self.assertTrue(np.isnan(map_statistics(z)[0]["topk_gt_precision"]))
        z["ground_truth_mask"][:] = z["topk_mask"]
        row = map_statistics(z)[0]
        self.assertEqual(row["topk_gt_precision"], 1)
        self.assertEqual(row["gt_patch_recall"], 1)


if __name__ == "__main__":
    unittest.main()
