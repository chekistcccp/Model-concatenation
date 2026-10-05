"""RGB scientific guards tested with small real modules, not server efficacy."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from transformers import Dinov2Config, Dinov2Model

from src.cache_features import load_rgb_raw
from src.composed_detector import ComposedAnomalyDetector, DirectFeatureTail
from src.method_deployment import evaluate_rgb, PROTOCOL, checkpoint_path
from src.models import build_stitch_modules
from src.prediction_io import write_predictions, sha256
from src.analyze_method_deployment import validate


class DeploymentTests(unittest.TestCase):
    def test_rgb_direct_control_uses_frozen_full_detector_and_original_targets(self):
        old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                spec = dict(input_size=224, channel_order='rgb', normalization_mean=[.5]*3,
                            normalization_std=[.5]*3, stage_channels=[4]*3)
                cfg = dict(paths=dict(results_dir=str(root/'results'), cache_dir=str(root/'cache')),
                           models=dict(sources=dict(medical=spec), targets=dict(medical=spec)), project=dict(compile_tail=False))
                target = Dinov2Model(Dinov2Config(image_size=224, patch_size=14, hidden_size=16,
                             num_hidden_layers=2, num_attention_heads=4, intermediate_size=32)).eval()
                adapter, _ = build_stitch_modules(cfg, target, 'medical', 'medical', 1, 1, 'mlp')
                model = ComposedAnomalyDetector(torch.nn.Conv2d(3,4,14,stride=14), adapter, target, spec, spec, 1)
                model.tail = DirectFeatureTail(target)
                job = dict(job='fixture_seed11', pair='medical_to_medical', source_backbone='medical',
                           target_backbone='medical', seed=11, source_stage=1, target_block=1)
                records = []
                for i in range(3):
                    path = root/f'{i}.png'; mask = root/f'mask{i}.png'
                    Image.fromarray(np.random.default_rng(i).integers(0,255,(224,224,3),dtype=np.uint8)).save(path)
                    pixels = np.zeros((224,224),dtype=np.uint8)
                    if i: pixels[50:90,50:90]=255
                    Image.fromarray(pixels).save(mask)
                    records.append(dict(image=str(path), mask=str(mask), label=int(i>0)))
                cache = root/'cache/features/Brain/test';cache.mkdir(parents=True)
                (cache/'records.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
                rgb = torch.stack([load_rgb_raw(r['image'],224) for r in records])
                scores = model(rgb)['score'].tolist()
                cached = root/'results/followup/method_controls/predictions/no_tail/fixture_seed11/Brain.csv'
                meta=dict(arm='no_tail', dataset='Brain',job=job['job'],seed=11,held_out='brain_mri',score_mode='contrast_topk',topk_fraction=.05)
                write_predictions(cached, records, [v+.001 for v in scores], meta,
                                  [dict(discrepancy_mean=0,discrepancy_median=0,discrepancy_max=0)]*3)
                before = {n: p.detach().clone() for n,p in model.named_parameters()}
                row,_ = evaluate_rgb(model,cfg,job,'no_tail','Brain',root/'out',torch.device('cpu'))
                self.assertEqual(row['n_test'],3);self.assertEqual(row['n_pixel_images'],3)
                self.assertEqual(row['live_parity_max_abs_delta'],0)
                self.assertAlmostEqual(row['score_mean_abs_delta'],.001,places=7)
                self.assertIn('pixel_aupr',row)
                for n,p in model.named_parameters():torch.testing.assert_close(p,before[n],rtol=0,atol=0)
                with cached.open() as f:rows=list(csv.DictReader(f));fields=list(rows[0])
                rows[0]['image_path']='different-target'
                with cached.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
                with self.assertRaisesRegex(ValueError,'identity/order'):
                    evaluate_rgb(model,cfg,job,'no_tail','Brain',root/'out',torch.device('cpu'))
        finally:torch.set_num_threads(old_threads)

    def test_checkpoint_routes_never_retrain_and_full_return_rejects_changed_scores_or_protocol(self):
        from test_method_analysis import fixture
        from src.analyze_method_controls import ARMS, DATASETS, MAPPING
        self.assertTrue(PROTOCOL['no_training'] and PROTOCOL['no_reselection'])
        self.assertEqual(checkpoint_path(Path('prior'),dict(checkpoint_root='final',job='j'),'oct','original'),Path('final/oct.pt'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prior=root/'prior';prior.mkdir();p=fixture(prior)
            out=root/'out';out.mkdir()
            refs=out/'references';refs.mkdir()
            shutil.copyfile(prior/'run_manifest.json',refs/'method_manifest.json')
            shutil.copyfile(prior/'metrics.json',refs/'method_metrics.json')
            shutil.copyfile(prior/'references/config.json',refs/'config.json')
            data=json.loads((prior/'metrics.json').read_text())
            for r in data['rows']:
                r.update(cached_image_auroc=r['image_auroc'],cached_image_aupr=r['image_aupr'],
                         live_parity_max_abs_delta=0,grid=16,score_mean_abs_delta=0,score_max_abs_delta=0)
                source=prior/'predictions'/r['arm']/r['job']/f"{r['dataset']}.csv"
                target=out/'predictions'/r['arm']/r['job']/f"{r['dataset']}.csv";target.parent.mkdir(parents=True,exist_ok=True)
                with source.open() as f:pred=list(csv.DictReader(f))
                for x in pred:x['cached_score']=x['score']
                with target.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(pred[0]));w.writeheader();w.writerows(pred)
            costs=[dict(arm=a,job=j['job'],held_out=h,latency_median_ms=1,latency_p95_ms=2,
                        resident_allocated_bytes=3,peak_allocated_bytes=4,total_unique=10,cnn_prefix=2,adapter=3,full_reference=5)
                   for a in ARMS for j in p['jobs'] for h in set(MAPPING.values())]
            payload=dict(rows=data['rows'],cost_rows=costs)
            (out/'metrics.json').write_text(json.dumps(payload))
            (out/'inference_costs.csv').write_text('fixture')
            for j in p['jobs']:
                w=out/'workers'/f"{j['job']}.json";w.parent.mkdir(exist_ok=True)
                w.write_text(json.dumps(dict(status='complete',job=j['job'],gpu=0,device_name='CPU fixture',device_total_memory=1,
                    rows=[r for r in payload['rows'] if r['job']==j['job']],cost_rows=[r for r in costs if r['job']==j['job']])))
                log=out/'logs'/f"{j['job']}.log";log.parent.mkdir(exist_ok=True);log.write_text('fixture')
            m=dict(status='complete',protocol=PROTOCOL,original_inputs_unchanged=True,arms=ARMS,jobs=p['jobs'],gpus=[0],
                   input_sha256={'results/followup/method_controls/run_manifest.json':sha256(prior/'run_manifest.json')})
            m['output_sha256']={x.relative_to(out).as_posix():sha256(x) for x in out.rglob('*') if x.is_file()}
            self.assertEqual(len(m['output_sha256']),317)
            path=out/'run_manifest.json';path.write_text(json.dumps(m));validate(out)
            m['protocol']=dict(PROTOCOL,no_training=False);path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'protocol'):validate(out)
            m['protocol']=PROTOCOL
            prediction=out/'predictions/original'/p['jobs'][0]['job']/'Brain.csv'
            with prediction.open() as f:pred=list(csv.DictReader(f))
            pred[0]['score']='.9';pred[1]['score']='.1'
            with prediction.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(pred[0]));w.writeheader();w.writerows(pred)
            m['output_sha256'][prediction.relative_to(out).as_posix()]=sha256(prediction);path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'reproduce metrics'):validate(out)


if __name__=='__main__':unittest.main()
