import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from src.run_followup import main, validate_existing_common_plan


class CommonResumeGuardTests(unittest.TestCase):
    def fixture(self, out):
        out.mkdir(parents=True)
        (out / "toy_seed11.json").write_text("preserved result", encoding="utf-8")
        r = dict(stage="common_stitch", jobs=[dict(job="toy_seed11")],
                 plan_sha256="original-plan", original_sha256={"src/worker.py": "original-sha"},
                 versions={"torch": "original-version"}, python="original-python")
        (out / "run_manifest.json").write_text(json.dumps(r), encoding="utf-8")
        return r

    def check(self, out, r, **changes):
        args = dict(plan_hash="new-plan", jobs=r["jobs"], hashes=r["original_sha256"],
                    versions=r["versions"], python=r["python"])
        args.update(changes)
        return validate_existing_common_plan(out, **args)

    def test_recorded_identical_plan_and_empty_directory_are_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "common_stitch"
            self.check(out, dict(jobs=[], original_sha256={}, versions={}, python="same"))
            r = self.fixture(out)
            self.check(out, r, plan_hash=r["plan_sha256"])

    def test_changed_protected_input_is_named_and_nothing_is_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "common_stitch"
            r = self.fixture(out)
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            with self.assertRaisesRegex(ValueError, "Changed protected input: src/worker.py"):
                self.check(out, r, hashes={"src/worker.py": "changed-sha"})
            self.assertEqual(before, {p.name: p.read_bytes() for p in out.iterdir()})

    def test_environment_and_locked_matrix_differences_are_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "common_stitch"
            r = self.fixture(out)
            with self.assertRaises(ValueError) as error:
                self.check(out, r, versions={"torch": "new-version"}, python="new-python", jobs=[])
            message = str(error.exception)
            for expected in ["Package torch", "original-version", "new-version", "Python:", "Locked job matrix"]:
                self.assertIn(expected, message)

    def test_missing_manifest_and_unexplained_plan_drift_stay_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "common_stitch"
            r = self.fixture(out)
            with self.assertRaisesRegex(ValueError, "full historical plan hash differs"):
                self.check(out, r)
            (out / "run_manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "Missing provenance manifest"):
                self.check(out, r)

    def test_check_only_and_execution_both_reject_drift_without_workers_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "results/followup/common_stitch"
            r = self.fixture(out)
            r.update(python=sys.version, versions={"torch": "old"})
            (out / "run_manifest.json").write_text(json.dumps(r), encoding="utf-8")
            cfg = dict(paths=dict(results_dir=str(root / "results")))
            config = root / "experiment.yaml"
            config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            for flag in [[], ["--check-only"]]:
                with self.subTest(flag=flag), patch("sys.argv", ["run_followup", "--stage", "common_stitch", "--config", str(config)] + flag), \
                     patch("src.run_followup.preflight", return_value=(r["jobs"], [], [], {})), \
                     patch("src.run_followup.protected_inputs", return_value=(r["original_sha256"], {})), \
                     patch("src.run_followup.importlib.metadata.version", return_value="new"), \
                     patch("src.run_followup.run_parallel", side_effect=AssertionError("workers forbidden")):
                    with self.assertRaisesRegex(ValueError, "Package torch"):
                        main()
                self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
