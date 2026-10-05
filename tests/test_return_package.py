"""Read-only package integrity checks, including persistent failure receipts."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from src.audit_return_package import audit


ROOT = Path(__file__).resolve().parents[1]


def fixture(path, *, stage="method_controls", data=b"scores", expected=None,
            status="complete", output="metrics.json", extra=()):
    prefix = f"results/followup/{stage}/" if stage else ""
    manifest = dict(status=status, git_head="test-commit", versions=dict(torch="fixture"),
                    python="test-python", output_sha256={output: expected or hashlib.sha256(data).hexdigest()})
    with tarfile.open(path, "w:gz") as package:
        for name, payload in [(prefix + "run_manifest.json", json.dumps(manifest).encode()),
                              (prefix + "metrics.json", data)]:
            member = tarfile.TarInfo(name); member.size = len(payload)
            package.addfile(member, io.BytesIO(payload))
        for member in extra:
            package.addfile(member, io.BytesIO(b"x" * member.size) if member.isfile() else None)


class ReturnPackageTests(unittest.TestCase):
    def test_valid_package_records_scope_environment_and_hashes_without_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "return.tar.gz"; fixture(path)
            before = path.read_bytes(); report = audit(path, ["method_controls"])
            self.assertTrue(report["ok"])
            self.assertIn("not scientific protocol", report["scope"])
            self.assertEqual(report["archive_sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(report["stages"][0]["verified_outputs"], 1)
            self.assertEqual(report["stages"][0]["git_head"], "test-commit")
            self.assertEqual(report["stages"][0]["versions"], {"torch": "fixture"})
            self.assertEqual(list(Path(tmp).iterdir()), [path])
            self.assertEqual(path.read_bytes(), before)

    def test_cli_missing_stage_exits_nonzero_after_writing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "return.tar.gz"; fixture(path, stage="common_stitch")
            output = Path(tmp) / "audit/report.json"
            result = subprocess.run([sys.executable, "-m", "src.audit_return_package", "--archive", str(path),
                                     "--require-stage", "method_controls", "--output", str(output)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["ok"])
            self.assertEqual(report["missing_required_stages"], ["method_controls"])
            self.assertTrue(report["stages"][0]["complete"])

    def test_corrupted_missing_and_incomplete_outputs_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "return.tar.gz"
            for kwargs, field in [({"expected": "0" * 64}, "hash_mismatches"),
                                  ({"output": "missing.json"}, "missing_outputs"),
                                  ({"status": "failed"}, "errors")]:
                with self.subTest(kwargs=kwargs):
                    fixture(path, **kwargs); report = audit(path)
                    self.assertFalse(report["ok"])
                    self.assertTrue(report["stages"][0][field])

    def test_unsafe_members_links_and_duplicates_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "return.tar.gz"
            entries = [tarfile.TarInfo(name) for name in ["/absolute", "../escape", "a/../escape", "a\\escape", "C:/escape",
                "results/followup/method_controls/metrics.json", "./results/followup/method_controls/metrics.json"]]
            for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE]:
                member = tarfile.TarInfo("link"); member.type = kind; member.linkname = "metrics.json"; entries.append(member)
            for member in entries:
                with self.subTest(name=member.name, kind=member.type):
                    fixture(path, extra=[member]); report = audit(path)
                    self.assertFalse(report["ok"])
                    self.assertTrue(report["errors"])

    def test_manifest_cannot_reference_outside_its_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "return.tar.gz"
            for output in ["../metrics.json", "/metrics.json", "sub\\metrics.json", "C:/metrics.json"]:
                with self.subTest(output=output):
                    fixture(path, output=output)
                    self.assertFalse(audit(path)["ok"])

    def test_top_level_manifest_is_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "return.tar.gz"; fixture(path, stage="")
            report = audit(path)
            self.assertTrue(report["ok"])
            self.assertEqual(report["stages"][0]["stage"], ".")
            self.assertEqual(report["stages"][0]["verified_outputs"], 1)


if __name__ == "__main__":
    unittest.main()
