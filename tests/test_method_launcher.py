"""Bash failure/success packaging tests, including mandatory weight exclusion."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
BASH=shutil.which("bash")
if os.name=="nt" and Path("C:/Program Files/Git/bin/bash.exe").is_file(): BASH="C:/Program Files/Git/bin/bash.exe"


@unittest.skipUnless(BASH, "Bash unavailable")
class MethodLauncherTests(unittest.TestCase):
    def launch(self, mode, gpu_setting=None, expected_gpus="auto"):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            shutil.copyfile(ROOT/"run_method_controls.sh", root/"run_method_controls.sh")
            fake=root/"fixture-python.sh"
            fake.write_text('#!/usr/bin/env bash\nexec "$REAL_PYTHON" "$FIXTURE_HELPER" "$@"\n', encoding="utf-8");fake.chmod(0o755)
            helper=root/"fixture_helper.py"
            helper.write_text('''import json,os,sys
from pathlib import Path
a=sys.argv[1:]; mode=os.environ["FIXTURE_MODE"]
if a==["--version"]: print("fixture");sys.exit(0)
if a[:2]==["-m","src.method_controls"]:
 assert not mode.startswith("existing"), "existing experiment must never start workers"
 assert a[a.index("--gpus")+1]==os.environ["EXPECTED_GPUS"],a
 if "--check-only" in a: sys.exit(7 if mode=="preflight_failure" else 0)
 p=Path("results/followup/method_controls");p.mkdir(parents=True)
 (p/"run_manifest.json").write_text(json.dumps(dict(status="complete")))
 (p/"metrics.json").write_text("{}")
 for sub in ["checkpoints","bundles"]:
  q=p/sub;q.mkdir();(q/"exclude.pt").write_text("keep on server");(q/"exclude.tmp").write_text("partial")
 sys.exit(0)
if a[:2]==["-m","src.analyze_method_controls"]:
 if mode in ["analysis_failure","existing","existing_corrupt"]: sys.exit(8)
 p=Path("results/analysis/method_controls");p.mkdir(parents=True);(p/"report_CN.md").write_text("fixture only")
 sys.exit(0)
raise RuntimeError(a)
''', encoding="utf-8")
            if mode.startswith("existing"):
                p=root/"results/followup/method_controls";p.mkdir(parents=True)
                old_manifest=dict(status="failed" if mode=="existing" else "complete",evidence="preserve")
                (p/"run_manifest.json").write_text(json.dumps(old_manifest))
                (p/"keep.pt").write_text("retain")
            env=dict(os.environ, PYTHON_BIN=fake.as_posix(),REAL_PYTHON=Path(sys.executable).as_posix(),
                     FIXTURE_HELPER=helper.as_posix(),FIXTURE_MODE=mode,GIT_DIR=(ROOT/".git").as_posix(),
                     EXPECTED_GPUS=expected_gpus)
            env.pop("GPUS",None);env.pop("GPU_ID",None)
            if gpu_setting:env[gpu_setting[0]]=gpu_setting[1]
            result=subprocess.run([BASH,"run_method_controls.sh"], cwd=root,env=env,capture_output=True,text=True,timeout=45)
            archives=list((root/"results/transfer").glob("*.tar.gz"));self.assertEqual(len(archives),1,result.stdout+result.stderr)
            with tarfile.open(archives[0]) as archive:
                names=archive.getnames()
                self.assertFalse(any(n.endswith((".pt",".tmp")) for n in names))
                self.assertTrue(any(n.endswith("launcher.log") for n in names))
                if mode.startswith("existing"):
                    m=json.loads(archive.extractfile("method_controls/run_manifest.json").read())
                    self.assertEqual(m,old_manifest)
                    self.assertEqual(json.loads((root/"results/followup/method_controls/run_manifest.json").read_text()),old_manifest)
                    self.assertTrue((root/"results/followup/method_controls/keep.pt").is_file())
                if mode in ["success","existing_complete"]: self.assertTrue(any(n.endswith("analysis/report_CN.md") for n in names))
            self.assertEqual(result.returncode,dict(success=0,preflight_failure=7,analysis_failure=8,existing=2,existing_complete=0,existing_corrupt=2)[mode])
            self.assertIn("[TRANSFER]",result.stdout)
            if mode=="success": self.assertTrue((root/"results/followup/method_controls/bundles/exclude.pt").is_file())

    def test_success_excludes_weights_and_copies_analysis(self): self.launch("success")
    def test_failed_preflight_is_packaged(self): self.launch("preflight_failure")
    def test_failed_analysis_retains_results(self): self.launch("analysis_failure")
    def test_existing_evidence_preserved(self): self.launch("existing")
    def test_complete_existing_evidence_is_validated_and_repackaged_without_workers(self): self.launch("existing_complete")
    def test_claimed_complete_but_invalid_evidence_is_preserved_and_rejected(self): self.launch("existing_corrupt")
    def test_explicit_multi_gpu_setting_forwarded(self): self.launch("success",("GPUS","0,2,3"),"0,2,3")
    def test_legacy_gpu_id_still_works(self): self.launch("success",("GPU_ID","2"),"2")


if __name__=="__main__": unittest.main()
