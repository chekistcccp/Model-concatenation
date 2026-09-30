"""Small numerical regressions for estimands, pairing and rank inference."""
import itertools
import unittest

import numpy as np
import pandas as pd

from src.analyze_results import EFFECTS, PAIRS, effect_summary, exact_spearman, holm, summarize


class AnalysisTests(unittest.TestCase):
    def test_effects_pair_seeds_before_sd(self):
        rows = []
        values = np.array([[.8, .6, .5, .4], [.6, .5, .4, .3], [.7, .55, .45, .35]])
        for seed, vals in zip([11, 22, 33], values):
            for pair, value in zip(PAIRS, vals):
                rows.append(dict(scope="example", seed=seed, pair=pair, image_auroc=value, image_aupr=value))
        frame = pd.DataFrame(rows).sample(frac=1, random_state=1)
        result = effect_summary(frame, ["scope"])
        for name, weights in EFFECTS.items():
            row = result[(result.effect == name) & (result.metric == "image_auroc")].iloc[0]
            expected = values @ weights
            self.assertAlmostEqual(row["mean"], expected.mean())
            self.assertAlmostEqual(row.sd, expected.std(ddof=1))
            self.assertEqual(row.n_seeds, 3)

    def test_incomplete_pair_fails(self):
        frame = pd.DataFrame([dict(scope="x", seed=s, pair=p, image_auroc=.5, image_aupr=.5)
                              for s in [11, 22] for p in PAIRS if (s, p) != (22, "MM")])
        with self.assertRaises(ValueError):
            effect_summary(frame, ["scope"])

    def test_exact_permutation_with_ties_and_constant(self):
        perms = np.array(list(itertools.permutations(range(3))))
        rho, p = exact_spearman([1, 2, 3], [1, 2, 3], perms)
        self.assertAlmostEqual(rho, 1)
        self.assertAlmostEqual(p, 2 / 6)
        rho, p = exact_spearman([1, 1, 2], [1, 1, 2], perms)
        self.assertAlmostEqual(rho, 1)
        self.assertAlmostEqual(p, 2 / 6)
        self.assertTrue(np.isnan(exact_spearman([1, 1, 1], [1, 2, 3], perms)[0]))

    def test_holm_and_sample_sd(self):
        np.testing.assert_allclose(holm([.03, .01, .8, np.nan]), [.06, .03, .8, np.nan])
        result = summarize(pd.DataFrame({"pair": ["MM"] * 3, "image_auroc": [.4, .5, .6]}), ["pair"], ["image_auroc"]).iloc[0]
        self.assertAlmostEqual(result.sd, .1)
        self.assertEqual(result.n_seeds, 3)


if __name__ == "__main__":
    unittest.main()
