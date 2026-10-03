"""CPU scheduling/process tests; scientific budgets and global phase barriers."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from src.followup_io import common_jobs
from src.method_schedule import command, parse_gpus, run_phases, run_worker, worker_path
from src.prediction_io import sha256
from src.run_followup import run_parallel, write_json


ROOT=Path(__file__).resolve().parents[1]
MODALITIES=["brain_mri","chest_xray","liver_ct","oct","pathology"]


def jobs_fixture():
    cfg=yaml.safe_load((ROOT/"configs/experiment.yaml").read_text(encoding="utf-8"))
    selected=dict(selection_metric="source_only_AOSS",selected=[])
    for s,t in cfg["models"]["pairs"]:
        b=9 if s==t=="general" else 3
        selected["selected"].append(dict(config=f"{s}_to_{t}_s2_b{b}",source_backbone=s,target_backbone=t,source_stage=2,target_block=b))
    return cfg,[j for j in common_jobs(cfg,selected) if j["reuse_original"]]


class MethodSchedulingTests(unittest.TestCase):
    def test_visible_gpu_indices_auto_and_invalid_plans(self):
        self.assertEqual(parse_gpus("auto",4),[0,1,2,3])
        self.assertEqual(parse_gpus("0,2",4),[0,2])
        for value,count in [("",4),("auto",0),("0,0",4),("4",4),("-1",4),("0,x",4),("0,",4)]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):parse_gpus(value,count)

    def test_job_command_keeps_protocol_in_config(self):
        c=command("configs/experiment.yaml","train",dict(job="medical_to_medical_s2_b3_seed11"))
        self.assertEqual(c[:4],[sys.executable,"-m","src.method_controls","--config"])
        self.assertIn("--job",c);self.assertNotIn("--skip-target-eval",c)
        self.assertNotIn("--epochs",c);self.assertNotIn("--lr",c)

    def simulate(self, out, fail_train=False):
        cfg,jobs=jobs_fixture();manifest=dict(status="running",jobs=jobs,gpus=[0,2])
        phases=[]
        def scheduling(commands,gpus,logs):
            phase=logs.name;phases.append(phase)
            self.assertEqual(gpus,[0,2]);self.assertEqual(len(commands),12)
            if phase!="evaluate":self.assertFalse((out/"training_completed.json").exists())
            else:
                self.assertEqual(len(json.loads((out/"training_completed.json").read_text())["rows"]),180)
                self.assertIn("training_completed_sha256",json.loads((out/"run_manifest.json").read_text()))
            for i,(name,c) in enumerate(commands):
                self.assertEqual(c[c.index("--worker-phase")+1],phase)
                if phase=="compose":
                    data=dict(rows=[dict(job=name,map_max_abs_delta=0) for _ in range(24)])
                elif phase=="train":
                    rows=[]
                    for arm in ["matched_tail","no_tail","untrained_adapter"]:
                        for held in MODALITIES:
                            p=out/"checkpoints"/arm/name/f"{held}.pt";p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name+arm+held)
                            rows.append(dict(arm=arm,job=name,held_out=held,checkpoint_sha256=sha256(p)))
                    data=dict(rows=rows,samples=[dict(job=name,held_out=h,samples=[]) for h in MODALITIES])
                else:
                    data=dict(rows=[dict(job=name) for _ in range(24)],aoss_rows=[dict(job=name) for _ in range(20)],
                              replays=[dict(job=name) for _ in range(12)])
                write_json(worker_path(out,phase,name),dict(status="complete",phase=phase,job=name,gpu=gpus[i%2],data=data,
                           device_name="fixture",device_total_memory=1,compute_capability=[0,0]))
                if fail_train and phase=="train":raise RuntimeError("fixture training worker failed")
        with patch("src.method_schedule.run_parallel",side_effect=scheduling):
            if fail_train:
                with self.assertRaisesRegex(RuntimeError,"training worker failed"):
                    run_phases(cfg,jobs,"config",out,[0,2],manifest,{}, {})
            else:
                composition,metrics=run_phases(cfg,jobs,"config",out,[0,2],manifest,{}, {})
                self.assertEqual((len(composition),len(metrics)),(288,288))
        return phases

    def test_all_training_finishes_before_any_evaluation_is_scheduled(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.simulate(Path(tmp)),["compose","train","evaluate"])

    def test_training_failure_retains_outputs_and_never_schedules_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            self.assertEqual(self.simulate(out,True),["compose","train"])
            self.assertFalse((out/"training_completed.json").exists());self.assertFalse((out/"metrics.json").exists())
            self.assertTrue(any(out.rglob("*.pt")))

    def test_worker_wrong_phase_is_rejected_before_touching_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(paths=dict(results_dir=tmp));out=Path(tmp)/"followup/method_controls"
            write_json(out/"run_manifest.json",dict(status="running",phase="source_training",gpus=[0]))
            with patch("torch.cuda.set_device") as setting:
                with self.assertRaisesRegex(ValueError,"coordinator phase"):
                    run_worker(cfg,"config","evaluate","job",0)
                setting.assert_not_called()

    def test_real_processes_overlap_across_gpus_but_never_share_one_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);helper=out/"fixture.py"
            helper.write_text('''import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--root");p.add_argument("--job");p.add_argument("--gpu");a=p.parse_args()
r=Path(a.root);lock=r/("gpu"+a.gpu+".lock")
with lock.open("x") as f:f.write(a.job)
start=time.monotonic();time.sleep(.6);end=time.monotonic()
(r/(a.job+".json")).write_text(json.dumps(dict(gpu=int(a.gpu),start=start,end=end)))
lock.unlink()
''',encoding="utf-8")
            commands=[(str(i),[sys.executable,str(helper),"--root",str(out),"--job",str(i)]) for i in range(4)]
            run_parallel(commands,[0,2],out/"logs")
            rows=[json.loads((out/f"{i}.json").read_text()) for i in range(4)]
            self.assertEqual({r["gpu"] for r in rows},{0,2})
            overlaps=[]
            for i,a in enumerate(rows):
                for b in rows[i+1:]:
                    overlap=max(a["start"],b["start"])<min(a["end"],b["end"])
                    if a["gpu"]==b["gpu"]:self.assertFalse(overlap)
                    else:overlaps.append(overlap)
            self.assertTrue(any(overlaps))


if __name__=="__main__":unittest.main()
