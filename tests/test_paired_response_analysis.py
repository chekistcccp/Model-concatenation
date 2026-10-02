import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.analyze_paired_response import MEASURES, analyze, summarize


class PairedAnalysisTests(unittest.TestCase):
    def test_two_oct_datasets_do_not_double_modality_weight(self):
        rows = []
        for ds, mod, value in [("RESC", "oct", 1), ("OCT2017", "oct", 1),
                               ("liver", "liver_ct", 0), ("RSNA", "chest_xray", 0), ("camelyon16", "pathology", 0)]:
            rows.append(dict(pair="MM", seed=11, held_out_modality="brain_mri", source_modality=mod,
                             dataset=ds, **{k: value for k in MEASURES}))
        _, _, seeds, _ = summarize(pd.DataFrame(rows))
        self.assertEqual(seeds.net_local_response.iloc[0], .25)

    def test_full_matrix_audit_and_tampered_index_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inp = root / "results/followup/paired_response_audit"
            inp.mkdir(parents=True)
            mapping = dict(Brain="brain_mri", liver="liver_ct", RESC="oct", OCT2017="oct", RSNA="chest_xray", camelyon16="pathology")
            jobs, checks, originals = [], [], {}
            for source, target, seed in itertools.product(["medical", "general"], ["medical", "general"], [11, 22, 33]):
                block = 9 if source == target == "general" else 3
                name = f"{source}_to_{target}_s2_b{block}_seed{seed}"
                jobs.append(dict(job=name, source_backbone=source, target_backbone=target, seed=seed))
                aoss = dict(normal_discrepancy=.1, perturb_in=.9, perturb_out=.1, local_sensitivity=.8, aoss=8.)
                final = root / "results/final" / f"{name}.json"
                final.parent.mkdir(parents=True, exist_ok=True)
                final.write_text(json.dumps(dict(aoss_rows=[dict(target_modality=h, **aoss) for h in set(mapping.values())])))
                originals[f"results/final/{name}.json"] = hashlib.sha256(final.read_bytes()).hexdigest()
                for held in sorted(set(mapping.values())):
                    rows = []
                    for ds, mod in mapping.items():
                        if mod == held: continue
                        for i in range(256):
                            rows.append(dict(job=name, seed=seed, source_backbone=source, target_backbone=target,
                                held_out_modality=held, source_modality=mod, dataset=ds, record_index=i,
                                image_path=f"{ds}/{i}.png", n_mask_in=10, n_mask_out=246, valid_regions=True,
                                normal_in=.7, normal_out=.1, perturb_in=.9, perturb_out=.1,
                                normal_spatial_contrast=.6, raw_local_sensitivity=.8,
                                delta_in=.2, delta_out=0, net_local_response=.2,
                                mean_delta=.2 * 10 / 256, contrast_delta=.2,
                                perturbation_kind_inferred=["blur", "patch_copy", "intensity"][i % 3], training_membership_inferred=i % 2 == 0))
                    p = inp / "jobs" / name / f"source_{held}.csv"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(rows).to_csv(p, index=False)
                    for component, value in aoss.items():
                        checks.append(dict(job=name, held_out_modality=held, component=component,
                                           original=value, replayed=value, delta=0, within_tolerance=True))
            pd.DataFrame(checks).to_csv(inp / "aoss_replay.csv", index=False)
            m = dict(status="complete", original_inputs_unchanged=True, no_training=True, no_reselection=True,
                target_test_read=False, target_valid_read=False, jobs=jobs, n_source_rows=73728,
                original_sha256=originals, output_sha256={p.relative_to(inp).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inp.rglob("*.csv")})
            mp = inp / "run_manifest.json"
            mp.write_text(json.dumps(m))
            analyze(root, inp)
            summary = pd.read_csv(root / "results/analysis/paired_response_audit/all_summary.csv")
            np.testing.assert_allclose(summary.net_local_response_mean, .2)
            p = next((inp / "jobs").rglob("*.csv"))
            frame = pd.read_csv(p)
            frame.loc[0, "record_index"] = 999  # Not a duplicate, but violates the locked record index set.
            frame.to_csv(p, index=False)
            m["output_sha256"][p.relative_to(inp).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
            mp.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "indices"):
                analyze(root, inp)


if __name__ == "__main__":
    unittest.main()
