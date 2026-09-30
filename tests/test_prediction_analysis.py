import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.analyze_predictions import bootstrap_metrics, rank_contribution, weighted_metrics


class PredictionAnalysisTests(unittest.TestCase):
    def test_weighted_auc_and_ap_match_expanded_tied_samples(self):
        labels = np.array([0, 1, 1, 0, 1, 0])
        score = np.array([.1, .2, .2, .2, .8, .9])
        for weights in [np.ones(6, dtype=int), np.array([2, 0, 3, 2, 1, 0]), np.array([0, 1, 0, 1, 2, 3])]:
            y, s = np.repeat(labels, weights), np.repeat(score, weights)
            auc, ap = weighted_metrics(labels, score, weights)
            self.assertAlmostEqual(auc, roc_auc_score(y, s))
            self.assertAlmostEqual(ap, average_precision_score(y, s))

    def test_bootstrap_keeps_pairing_and_stratification(self):
        labels = np.array([0, 0, 1, 1])
        score = np.array([.1, .2, .8, .9])
        result = bootstrap_metrics(labels, np.column_stack([score, score, -score]), 40, np.random.default_rng(11))
        np.testing.assert_allclose(result[:, 0], result[:, 1])
        np.testing.assert_allclose(result[:, 0, 0], 1)
        np.testing.assert_allclose(result[:, 2, 0], 0)

    def test_contribution_means_equal_auc_for_both_labels(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        score = np.array([.1, .5, .5, .9, .8, .2])
        contribution = rank_contribution(labels, score)
        for label in [0, 1]:
            self.assertAlmostEqual(contribution[labels == label].mean(), roc_auc_score(labels, score))


if __name__ == "__main__":
    unittest.main()
