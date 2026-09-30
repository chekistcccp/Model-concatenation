import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from src.audit_followup import overlap_table, subset
from src.verify_manifest_duplicates import verify


class FollowupTests(unittest.TestCase):
    def test_overlap_counts_keys_not_all_rows(self):
        f = pd.DataFrame(dict(dataset=["x"]*4, split=["train", "test", "test", "valid"], filename=["1", "1", "2", "2"]))
        result = overlap_table(f, "filename")
        row = result[(result.split_a == "train") & (result.split_b == "test")].iloc[0]
        self.assertEqual(row.shared_keys, 1)
        self.assertEqual(row.rows_b, 1)

    def test_subset_is_deterministic_and_ordered(self):
        f = pd.DataFrame({"n": range(100)})
        a, b = subset(f, 8, 11), subset(f, 8, 11)
        self.assertTrue(a.equals(b))
        self.assertTrue(a.n.is_monotonic_increasing)
        self.assertEqual(len(a), 8)
        self.assertEqual(len(subset(f, 1000, 11)), 100)

    def test_same_name_does_not_imply_same_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = []
            for split, color in [("train", "red"), ("valid", "red"), ("test", "blue")]:
                p = Path(tmp) / split / "a.png"
                p.parent.mkdir()
                Image.new("RGB", (3, 4), color).save(p)
                records.append(dict(dataset="example", split=split, image=str(p)))
            result = verify(records, {"example"})
            self.assertEqual(result["candidate_pairs"], 3)
            self.assertEqual(result["verified_same_rgb_pixels"], 1)
            self.assertEqual(result["verified_same_bytes"], 1)
            Path(records[0]["image"]).unlink()
            result = verify(records, {"example"})
            self.assertEqual(len(result["errors"]), 1)
            self.assertEqual(sum(p["same_bytes"] is None for p in result["pairs"]), 2)


if __name__ == "__main__":
    unittest.main()
