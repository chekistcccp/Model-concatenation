import unittest
import numpy as np
import pandas as pd

from src.source_calibration import calibrated_scores, fit_reference, mid_ecdf


class SourceCalibrationTests(unittest.TestCase):
    def test_midrank_ties_and_tails(self):
        np.testing.assert_allclose(mid_ecdf([1, 2, 2, 3], [0, 1, 2, 4]), [0, .125, .5, 1])

    def test_modality_weighting_not_pooled_dataset_weighting(self):
        ref = {"oct": {"a": {"mean": [0], "contrast": [0]}, "b": {"mean": [0], "contrast": [0]}},
               "brain": {"c": {"mean": [2], "contrast": [2]}}}
        result = calibrated_scores(ref, [1], [1])
        self.assertEqual(result["source_equal_fusion"][0], .5)

    def test_score_is_independent_of_other_test_images(self):
        ref = {"m": {"d": {"mean": [0, 1, 2], "contrast": [1, 2, 3]}}}
        a = calibrated_scores(ref, [.4], [2])["source_equal_fusion"][0]
        b = calibrated_scores(ref, [.4, -999, 999], [2, 999, -999])["source_equal_fusion"][0]
        self.assertEqual(a, b)

    def fixture(self):
        mapping = dict(A="a", B="b", C="c", D="d", E="target", F="target")
        rows = [dict(dataset=d, source_modality=mapping[d], held_out_modality="target", record_index=i,
                     normal_mean=i / 256, normal_contrast_topk=i / 512) for d in "ABCD" for i in range(256)]
        return pd.DataFrame(rows), mapping

    def test_excludes_entire_target_modality(self):
        frame, mapping = self.fixture()
        self.assertEqual(set(fit_reference(frame, "target", mapping)), set("abcd"))
        frame.loc[0, "dataset"] = "F"
        with self.assertRaisesRegex(ValueError, "contamination"):
            fit_reference(frame, "target", mapping)

    def test_synthetic_perturbations_do_not_fit_reference(self):
        frame, mapping = self.fixture()
        expected = fit_reference(frame, "target", mapping)
        frame["pert_mean"] = np.nan
        frame["local_sensitivity"] = 999
        self.assertEqual(fit_reference(frame, "target", mapping), expected)

    def test_nonfinite_source_rejected(self):
        frame, mapping = self.fixture()
        frame.loc[0, "normal_mean"] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            fit_reference(frame, "target", mapping)


if __name__ == "__main__":
    unittest.main()
