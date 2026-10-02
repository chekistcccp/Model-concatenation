"""Launcher control-flow tests; fixtures do not represent a GPU research run."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
if os.name == "nt":
    candidate = Path("C:/Program Files/Git/bin/bash.exe")
    if candidate.is_file(): BASH = str(candidate)


@unittest.skipUnless(BASH, "Bash unavailable")
class ScreenLauncherTests(unittest.TestCase):
    def launch(self, mode):
        # All writes and packaging remain inside this verified, fresh temp directory.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            shutil.copyfile(ROOT / "run_screen_response_audit.sh", root / "run_screen_response_audit.sh")
            fake = root / "fixture-python.sh"
            fake.write_text('#!/usr/bin/env bash\nif [[ "$1" == "-c" ]]; then exec "$REAL_PYTHON" "$@"; fi\nexec "$REAL_PYTHON" "$FIXTURE_HELPER" "$@"\n', encoding="utf-8")
            fake.chmod(0o755)
            helper = root / "fixture_helper.py"
            helper.write_text('''import hashlib,json,os,sys
from pathlib import Path
args=sys.argv[1:]
if args==["--version"]: print("fixture");sys.exit(0)
if args[:2]==["-m","src.screen_response_audit"]:
 if "--check-only" in args:
  sys.exit(7 if os.environ["FIXTURE_MODE"]=="preflight_failure" else 0)
 p=Path("results/followup/screen_response_audit");p.mkdir(parents=True)
 files={}
 for i in range(219):
  q=p/(str(i)+".csv");q.write_text("fixture\\n")
  files[q.name]=hashlib.sha256(q.read_bytes()).hexdigest()
 (p/"run_manifest.json").write_text(json.dumps(dict(status="complete",original_inputs_unchanged=True,
  no_training=True,no_reselection=True,target_test_read=False,target_valid_read=False,
  jobs=[dict(job=str(i)) for i in range(36)],n_source_rows=221184,output_sha256=files)))
 sys.exit(0)
if args[:2]==["-m","src.analyze_screen_response"]:
 p=Path("results/analysis/screen_response_audit");p.mkdir(parents=True)
 (p/"report.md").write_text("fixture; no GPU result")
 sys.exit(0)
raise RuntimeError(args)
''', encoding="utf-8")
            if mode == "existing":
                p = root / "results/followup/screen_response_audit/run_manifest.json"
                p.parent.mkdir(parents=True)
                p.write_text('{"status":"failed","evidence":"keep"}')
            env = dict(os.environ, PYTHON_BIN=fake.as_posix(), REAL_PYTHON=Path(sys.executable).as_posix(),
                       FIXTURE_HELPER=helper.as_posix(), FIXTURE_MODE=mode, GIT_DIR=(ROOT / ".git").as_posix())
            result = subprocess.run([BASH, "run_screen_response_audit.sh"], cwd=root, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=45)
            archives = list((root / "results/transfer").glob("*.tar.gz"))
            self.assertEqual(len(archives), 1, result.stdout)
            with tarfile.open(archives[0]) as archive:
                names = archive.getnames()
                logs = [x for x in names if x.endswith("launcher.log")]
                self.assertEqual(len(logs), 1)
                log = archive.extractfile(logs[0]).read().decode()
                manifest = archive.extractfile("screen_response_audit/run_manifest.json").read().decode() if mode != "preflight_failure" else None
            self.assertIn("[TRANSFER]", result.stdout)
            if mode == "success":
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn("[SUCCESS]", log)
                self.assertTrue(any(n.endswith("analysis/report.md") for n in names))
                self.assertEqual(json.loads(manifest)["status"], "complete")
            elif mode == "preflight_failure":
                self.assertEqual(result.returncode, 7, result.stdout)
                self.assertIn("Launcher exit code: 7", log)
            else:
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertEqual(json.loads(manifest), dict(status="failed", evidence="keep"))

    def test_success_hash_verification_analysis_copy_and_package(self): self.launch("success")
    def test_preflight_failure_retains_exit_code_and_log(self): self.launch("preflight_failure")
    def test_existing_audit_is_preserved_and_packaged(self): self.launch("existing")


if __name__ == "__main__":
    unittest.main()
