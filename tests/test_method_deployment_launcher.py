"""Actual Bash packaging/reuse behavior with a process stub; no GPU claim."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

from test_method_launcher import BASH, ROOT


@unittest.skipUnless(BASH, 'Bash unavailable')
class DeploymentLauncherTests(unittest.TestCase):
    def test_success_repack_and_failure_preserve_outputs_and_exclude_weights(self):
        for mode in ['success', 'existing_complete', 'existing_failed']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp).resolve();shutil.copyfile(ROOT/'run_method_deployment.sh',root/'run_method_deployment.sh')
                fake=root/'python.sh';fake.write_text('#!/usr/bin/env bash\nexec "$REAL_PYTHON" "$HELPER" "$@"\n');fake.chmod(0o755)
                helper=root/'helper.py'
                helper.write_text('''import os,sys
from pathlib import Path
a=sys.argv[1:];mode=os.environ['MODE'];p=Path('results/followup/method_deployment')
if a[:2]==['-m','src.method_deployment']:
 assert mode=='success','Existing outputs must never launch workers'
 assert a[a.index('--gpus')+1]=='0,2'
 p.mkdir(parents=True);(p/'run_manifest.json').write_text('evidence');(p/'keep.pt').write_text('server only')
elif a[:2]==['-m','src.analyze_method_deployment']:
 if mode=='existing_failed':sys.exit(8)
 q=Path('results/analysis/method_deployment');q.mkdir(parents=True);(q/'report_CN.md').write_text('fixture')
else:raise RuntimeError(a)
''')
                if mode.startswith('existing'):
                    p=root/'results/followup/method_deployment';p.mkdir(parents=True)
                    (p/'run_manifest.json').write_text('evidence');(p/'keep.pt').write_text('server only')
                env=dict(os.environ,PYTHON_BIN=fake.as_posix(),REAL_PYTHON=Path(sys.executable).as_posix(),
                         HELPER=helper.as_posix(),MODE=mode,GPUS='0,2',GIT_DIR=(ROOT/'.git').as_posix())
                r=subprocess.run([BASH,'run_method_deployment.sh'],cwd=root,env=env,capture_output=True,text=True,timeout=45)
                self.assertEqual(r.returncode,2 if mode=='existing_failed' else 0,r.stdout+r.stderr)
                self.assertIn('[TRANSFER]',r.stdout)
                archives=list((root/'results/transfer').glob('*.tar.gz'));self.assertEqual(len(archives),1)
                with tarfile.open(archives[0]) as package:
                    self.assertEqual(package.extractfile('method_deployment/run_manifest.json').read(),b'evidence')
                    self.assertFalse(any(n.endswith(('.pt','.tmp')) for n in package.getnames()))
                self.assertEqual((root/'results/followup/method_deployment/keep.pt').read_text(),'server only')


if __name__=='__main__':unittest.main()
