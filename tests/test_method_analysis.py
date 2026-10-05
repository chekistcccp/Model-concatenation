"""Complete synthetic return matrix; checks protocol/metric validation, not efficacy."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from src.analyze_method_controls import ARMS, DATASETS, MAPPING, validate, run, contrast_summary
from src.followup_io import common_jobs
from src.prediction_io import sha256
from src.method_schedule import worker_path


ROOT=Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding="utf-8")


def fixture(out):
    cfg=yaml.safe_load((ROOT/"configs/experiment.yaml").read_text(encoding="utf-8"))
    selected=dict(selection_metric="source_only_AOSS",selected=[])
    for s,t in cfg["models"]["pairs"]:
        b=9 if s==t=="general" else 3
        selected["selected"].append(dict(config=f"{s}_to_{t}_s2_b{b}",source_backbone=s,target_backbone=t,source_stage=2,target_block=b))
    jobs=[j for j in common_jobs(cfg,selected) if j["reuse_original"]]
    write(out/"references/config.json",cfg);write(out/"references/selected_configs.json",selected)
    metrics=[];aoss=[];training=[];sampling=[];composition=[];weights={}
    for j in jobs:
        original=[]
        for ds in DATASETS:
            orig=dict(dataset=ds,source_backbone=j["source_backbone"],target_backbone=j["target_backbone"],
                      seed=j["seed"],target_modality=MAPPING[ds],source_stage=2,target_block=j["target_block"],
                      adapter="mlp",epochs=8,n_per_source_modality=1000,score_mode="contrast_topk",n_test=2,image_auroc=1.,image_aupr=1.)
            original.append(orig)
            for arm in ARMS:
                metrics.append(dict(orig,arm=arm,job=j["job"],held_out=MAPPING[ds]))
                p=out/"predictions"/arm/j["job"]/f"{ds}.csv";p.parent.mkdir(parents=True,exist_ok=True)
                with p.open("w",encoding="utf-8",newline="") as f:
                    fields=["record_index","image_path","label","score","dataset","arm","job","seed","held_out","score_mode","topk_fraction"]
                    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                    for i in [0,1]:w.writerow(dict(record_index=i,image_path=f"{ds}/{i}",label=i,score=i*.2,
                        dataset=ds,arm=arm,job=j["job"],seed=j["seed"],held_out=MAPPING[ds],score_mode="contrast_topk",topk_fraction=.05))
        write(out/"references"/f"{j['job']}.json",dict(config=j["config"],source_backbone=j["source_backbone"],target_backbone=j["target_backbone"],
            source_stage=2,target_block=j["target_block"],seed=j["seed"],adapter="mlp",rows=original))
        for held in sorted(set(MAPPING.values())):
            take=[]
            for mod in sorted(set(MAPPING.values())-{held}):
                ds=[d for d in DATASETS if MAPPING[d]==mod]
                for d in ds:
                    take += [dict(dataset=d,source_modality=mod,record_index=i,image=f"{d}/{i}") for i in range(1000//len(ds))]
            sampling.append(dict(job=j["job"],held_out=held,samples=take))
            for arm in ARMS:
                aoss.append(dict(arm=arm,job=j["job"],held_out=held,normal_discrepancy=.1,
                                 perturb_in=.3,perturb_out=.1,local_sensitivity=.2,aoss=.2/(.1+1e-8)))
                if arm!="original":
                    training.append(dict(arm=arm,job=j["job"],held_out=held,n_normal_images=4000,
                                         optimizer_steps=0 if arm=="untrained_adapter" else 504,
                                         initial_adapter_sha256="fixture",checkpoint_sha256="fixture"))
                    weights[f"checkpoints/{arm}/{j['job']}/{held}.pt"]="fixture"
            for ds in DATASETS:
                if MAPPING[ds]==held:continue
                bundle=(j["seed"]==11 and held=="brain_mri" and ds=="camelyon16")
                composition.append(dict(job=j["job"],held_out=held,source_dataset=ds,n_images=2,map_max_abs_delta=0,
                                        bundle_roundtrip_delta=0 if bundle else ""))
        if j["seed"]==11:weights[f"bundles/{j['job']}_brain_mri.pt"]="fixture"
    write(out/"training_samples.json",dict(rows=sampling))
    write(out/"training_completed.json",dict(rows=training,no_target_data_used=True,no_reselection=True))
    write(out/"metrics.json",dict(rows=metrics,aoss_rows=aoss,replays=[dict(within_tolerance=True)]*144))
    with (out/"composition_checks.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(composition[0]));w.writeheader();w.writerows(composition)
    m=dict(status="complete",original_inputs_unchanged=True,no_reselection=True,score_unchanged=True,
           original_budget_unchanged=True,training_uses_target_data=False,training_uses_synthetic_anomalies=False,
           jobs=jobs,arms=ARMS,n_metric_rows=288,n_composition_checks=288,weights_sha256=weights)
    m["output_sha256"]={str(p.relative_to(out)).replace("\\","/"):sha256(p) for p in out.rglob("*") if p.is_file()}
    write(out/"run_manifest.json",m)
    return m


class MethodAnalysisTests(unittest.TestCase):
    def test_paired_seed_variation_and_missing_plot_dependency_preserve_tables(self):
        rows = [dict(pair='p', group='overall', metric='image_auroc', contrast='matched_tail_minus_no_tail',
                     datasets='Brain', delta=v) for v in [-.1, -.2, -.3]]
        summary = contrast_summary(rows)[0]
        self.assertAlmostEqual(summary['mean_delta'], -.2)
        self.assertAlmostEqual(summary['paired_seed_sd'], .1)
        self.assertEqual(summary['n_positive_seeds'], 0)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'input'; directory.mkdir(); fixture(directory)
            out = Path(tmp) / 'output'
            with patch.dict('sys.modules', {'matplotlib': None}):
                run(directory, out)
            self.assertFalse(json.loads((out/'audit.json').read_text())['plots_generated'])
            self.assertTrue((out/'contrast_summary.csv').is_file())
            with (out/'method_summary.csv').open() as f:
                groups = {r['group'] for r in csv.DictReader(f)}
            self.assertTrue(set(DATASETS) <= groups)

    def test_full_matrix_recomputes_predictions_and_rejects_contamination_or_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);m=fixture(out)
            self.assertEqual(len(m["output_sha256"]),306)
            _,data=validate(out);self.assertEqual(len(data["rows"]),288)
            # New distributed receipts retain compatibility with prior serial returns.
            for phase in ["compose","train","evaluate"]:
                for i,j in enumerate(m["jobs"]):
                    write(worker_path(out,phase,j["job"]),dict(status="complete",phase=phase,job=j["job"],gpu=[0,2][i%2],
                        device_name="CPU fixture, not a GPU run",device_total_memory=1,compute_capability=[0,0]))
                    log=out/"logs"/phase/f"{j['job']}.log";log.parent.mkdir(parents=True,exist_ok=True);log.write_text("fixture")
            m.update(scheduling="one_job_per_gpu_v1",gpus=[0,2],training_completed_sha256=sha256(out/"training_completed.json"))
            m["output_sha256"]={str(p.relative_to(out)).replace("\\","/"):sha256(p) for p in out.rglob("*") if p.is_file() and p.name!="run_manifest.json"}
            write(out/"run_manifest.json",m)
            self.assertEqual(len(m["output_sha256"]),378)
            validate(out)
            p=out/"training_samples.json";original=p.read_bytes();s=json.loads(original)
            s["rows"][0]["samples"][0]["source_modality"]=s["rows"][0]["held_out"]
            write(p,s);m["output_sha256"]["training_samples.json"]=sha256(p);write(out/"run_manifest.json",m)
            with self.assertRaisesRegex(ValueError,"contamination"):validate(out)
            p.write_bytes(original);m["output_sha256"]["training_samples.json"]=sha256(p)
            p=out/"metrics.json";s=json.loads(p.read_text());s["rows"][0]["image_auroc"]=.2
            write(p,s);m["output_sha256"]["metrics.json"]=sha256(p);write(out/"run_manifest.json",m)
            with self.assertRaisesRegex(ValueError,"replay differs"):validate(out)


if __name__=="__main__":unittest.main()
