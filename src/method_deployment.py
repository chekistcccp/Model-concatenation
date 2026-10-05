"""Read-only RGB evaluation of the four frozen method arms; no training/selection."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys

import numpy as np
import torch
import yaml

from .analyze_method_controls import ARMS, DATASETS, MAPPING, validate
from .cache_features import _normalize, load_rgb_raw
from .common import configure_torch
from .composed_detector import ComposedAnomalyDetector, DirectFeatureTail, source_prefix
from .followup_io import read_json, read_records
from .method_controls import initialize, new_checkpoint, require, write_csv
from .method_schedule import parse_gpus
from .models import load_source, load_target, target_num_prefix
from .paired_response_audit import versions
from .prediction_io import checkpoint_identity, sha256, write_predictions
from .run_followup import inputs_unchanged, run_parallel, write_json
from .train_eval import _load_mask, _safe_auc, discrepancy


BATCH = 16
WARMUP = 5
REPEATS = 30
PROTOCOL = dict(no_training=True, no_reselection=True, score_unchanged=True,
                original_budget_unchanged=True, global_selection_fold_exclusive=False,
                input_size=224, batch_size=BATCH, benchmark_batch_size=1,
                benchmark_warmup=WARMUP, benchmark_repeats=REPEATS,
                benchmark_scope='complete RGB tensor detector including normalization; excludes decode/H2D',
                pixel_grid=16, original_results_overwritten=False)


def prior_root(cfg):
    return Path(cfg['paths']['results_dir']) / 'followup/method_controls'


def output_root(cfg):
    return Path(cfg['paths']['results_dir']) / 'followup/method_deployment'


def checkpoint_path(prior, job, held, arm):
    return (Path(job['checkpoint_root']) / f'{held}.pt' if arm == 'original'
            else new_checkpoint(prior, arm, job, held))


def preflight(cfg, config):
    prior = prior_root(cfg)
    manifest, _ = validate(prior)
    require(cfg == read_json(prior / 'references/config.json'), 'Config differs from completed method controls')
    require(manifest['versions'] == versions() and manifest['python'] == platform.python_version(),
            'Restore the completed method-control environment; do not upgrade dependencies')
    # Hash original module/config/checkpoint inputs again; new analysis modules
    # are not part of the old scientific execution identity.
    require(inputs_unchanged(manifest['original_sha256'], manifest['array_weight_stats']),
            'Protected original inputs/cache changed since method controls')
    hashes = dict(manifest['original_sha256'])
    hashes[str(prior / 'run_manifest.json')] = sha256(prior / 'run_manifest.json')
    for name, digest in {**manifest['output_sha256'], **manifest['weights_sha256']}.items():
        require(sha256(prior / name) == digest, f'Method artifact differs/missing: {name}')
        hashes[str(prior / name)] = digest
    hashes[str(Path(config))] = sha256(Path(config))
    for name in ['method_deployment.py', 'analyze_method_deployment.py']:
        p = Path(__file__).with_name(name); hashes[str(p)] = sha256(p)
    stats = dict(manifest['array_weight_stats'])
    for ds in DATASETS:
        records = read_records(Path(cfg['paths']['cache_dir']) / 'features' / ds / 'test/records.jsonl')
        require(records and all(r['dataset'] == ds and r['modality'] == MAPPING[ds] and r['split'] == 'test'
                                for r in records), f'Wrong target records: {ds}')
        for r in records:
            for value in [r['image'], r.get('mask')]:
                if value:
                    p = Path(value)
                    require(p.is_file(), f'Missing target image/mask: {p}; no preparation fallback')
                    stats[str(p)] = dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
    return manifest['jobs'], hashes, stats


@torch.inference_mode()
def benchmark(model, rgb, device):
    for _ in range(WARMUP):
        model(rgb)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    values = []
    for _ in range(REPEATS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); model(rgb); end.record()
        torch.cuda.synchronize(device)
        values.append(float(start.elapsed_time(end)))
    return dict(latency_median_ms=float(np.median(values)), latency_p95_ms=float(np.percentile(values, 95)),
                resident_allocated_bytes=baseline, peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device), **model.parameter_counts())


@torch.inference_mode()
def evaluate_rgb(model, cfg, job, arm, dataset, out, device):
    prior = prior_root(cfg)
    root = Path(cfg['paths']['cache_dir']) / 'features' / dataset / 'test'
    records = read_records(root / 'records.jsonl')
    with (prior / 'predictions' / arm / job['job'] / f'{dataset}.csv').open(encoding='utf-8', newline='') as f:
        cached = list(csv.DictReader(f))
    require(len(cached) == len(records) and all(p['image_path'] == r['image'] and int(p['label']) == int(r['label'])
                for p, r in zip(cached, records)), 'Target identity/order differs from completed controls')
    scores, map_stats, ps, pl = [], [], [], []
    has_masks = any(r.get('mask') for r in records)
    parity = None
    for start in range(0, len(records), BATCH):
        batch = records[start:start+BATCH]
        rgb = torch.stack([load_rgb_raw(r['image'], 224) for r in batch]).to(device)
        prediction = model(rgb)
        d = prediction['discrepancy'].flatten(1)
        require(torch.isfinite(d).all().item() and torch.isfinite(prediction['score']).all().item(), 'Nonfinite RGB inference')
        if start == 0:
            # Separately execute inherited modules on the same two raw images.
            # This check is an online-path parity check, not cached equivalence.
            x = _normalize(rgb[:2], model.source_spec).to(next(model.prefix.parameters()).dtype)
            y = _normalize(rgb[:2], model.target_spec).to(next(model.reference.parameters()).dtype)
            ref = model.reference(y, return_dict=True).last_hidden_state[:, target_num_prefix(model.reference):]
            expected = discrepancy(model.adapter, model.tail, model.prefix(x), ref, device)
            # Same two-image batch on both paths avoids batch-dependent GPU rounding.
            actual = model(rgb[:2])['discrepancy'].flatten(1)
            parity = float((expected-actual).abs().max().item())
            require(parity <= 1e-6, 'Registered RGB detector differs from live inherited module path')
        scores.extend(prediction['score'].cpu().tolist())
        for a, b, c in zip(d.mean(1).cpu().tolist(), d.median(1).values.cpu().tolist(), d.max(1).values.cpu().tolist()):
            map_stats.append(dict(discrepancy_mean=a, discrepancy_median=b, discrepancy_max=c))
        maps = prediction['anomaly_map'].cpu().numpy()
        for i, rec in enumerate(batch):
            mask = _load_mask(rec.get('mask'), 16)
            if mask is None and has_masks and int(rec['label']) == 0:
                mask = np.zeros((16, 16), dtype=np.uint8)
            if mask is not None:
                pl.append(mask.ravel()); ps.append(maps[i].ravel())
    metadata = dict(arm=arm, dataset=dataset, job=job['job'], pair=job['pair'], seed=job['seed'],
                    held_out=MAPPING[dataset], source_stage=job['source_stage'], target_block=job['target_block'],
                    score_mode='contrast_topk', topk_fraction=.05)
    prediction_path = out / 'predictions' / arm / job['job'] / f'{dataset}.csv'
    write_predictions(prediction_path, records, scores, metadata, map_stats)
    with prediction_path.open(encoding='utf-8', newline='') as f:
        exported = list(csv.DictReader(f))
    write_csv(prediction_path, [dict(p, cached_score=old['score']) for p, old in zip(exported, cached)])
    labels = [int(r['label']) for r in records]
    auroc, ap = _safe_auc(labels, scores)
    old_scores = np.array([float(p['score']) for p in cached])
    old_auc, old_ap = _safe_auc(labels, old_scores)
    row = dict(arm=arm, job=job['job'], source_backbone=job['source_backbone'], target_backbone=job['target_backbone'],
               seed=job['seed'], held_out=MAPPING[dataset], dataset=dataset, n_test=len(records),
               score_mode='contrast_topk', image_auroc=auroc, image_aupr=ap,
               cached_image_auroc=old_auc, cached_image_aupr=old_ap,
               score_mean_abs_delta=float(np.mean(np.abs(np.array(scores)-old_scores))),
               score_max_abs_delta=float(np.max(np.abs(np.array(scores)-old_scores))), live_parity_max_abs_delta=parity,
               n_pixel_images=len(pl), grid=16)
    if pl:
        row['pixel_auroc'], row['pixel_aupr'] = _safe_auc(np.concatenate(pl), np.concatenate(ps))
    return row, records[0]


def worker(cfg, job_name, gpu):
    out, prior = output_root(cfg), prior_root(cfg)
    m = read_json(out / 'run_manifest.json')
    require(m['status'] == 'running' and gpu in m['gpus'] and m['protocol'] == PROTOCOL, 'Invalid coordinator authorization')
    require(cfg == read_json(out / 'references/config.json') and m['versions'] == versions(), 'Worker config/environment differs')
    jobs = [j for j in m['jobs'] if j['job'] == job_name]
    require(len(jobs) == 1, 'Worker outside locked job matrix')
    job = jobs[0]
    for p in [Path(__file__), Path(__file__).with_name('analyze_method_deployment.py'), prior / 'run_manifest.json']:
        require(sha256(p) == m['input_sha256'][str(p)], f'Worker input changed: {p}')
    path = out / 'workers' / f'{job_name}.json'
    require(not path.exists(), 'Existing worker receipt; refuse overwrite/resume')
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    configure_torch(cfg['project'].get('allow_tf32', True))
    props = torch.cuda.get_device_properties(gpu)
    receipt = dict(status='running', job=job_name, gpu=gpu, device_name=props.name,
                   device_total_memory=props.total_memory, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write_json(path, receipt)
    try:
        source, target = load_source(cfg, job['source_backbone'], device), load_target(cfg, job['target_backbone'], device)
        prefix = source_prefix(source, job['source_stage'])
        del source  # Unused CNN suffix must not inflate detector memory accounting.
        metrics, costs = [], []
        for held in sorted(set(MAPPING.values())):
            for arm in ARMS:
                weight = checkpoint_path(prior, job, held, arm)
                require(sha256(weight) == m['input_sha256'][str(weight)], 'Frozen checkpoint changed')
                adapter, _ = initialize(cfg, target, job, held, arm, device)
                ckpt = torch.load(weight, map_location='cpu', weights_only=False)
                if arm == 'original':
                    checkpoint_identity(ckpt, job['source_backbone'], job['target_backbone'], job['source_stage'], job['target_block'], 'mlp', held)
                else:
                    require((ckpt['arm'], ckpt['job'], ckpt['held_out']) == (arm, job_name, held), 'Control checkpoint identity differs')
                adapter.load_state_dict(ckpt['adapter'], strict=True)
                del ckpt
                model = ComposedAnomalyDetector(prefix, adapter, target, cfg['models']['sources'][job['source_backbone']],
                                               cfg['models']['targets'][job['target_backbone']], job['target_block'])
                if arm == 'no_tail':
                    model.tail = DirectFeatureTail(target)
                model.eval()
                first = None
                for ds in DATASETS:
                    if MAPPING[ds] != held:
                        continue
                    print(f'[RGB] {job_name} {arm} {ds}', flush=True)
                    row, rec = evaluate_rgb(model, cfg, job, arm, ds, out, device)
                    metrics.append(row)
                    if first is None:
                        first = rec
                rgb = load_rgb_raw(first['image'], 224).unsqueeze(0).to(device)
                torch.cuda.empty_cache()
                costs.append(dict(arm=arm, job=job_name, held_out=held, **benchmark(model, rgb, device)))
                del model, adapter, rgb
        receipt.update(status='complete', rows=metrics, cost_rows=costs)
    except BaseException as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(path, receipt)
        raise
    write_json(path, receipt)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/experiment.yaml')
    ap.add_argument('--gpus', default='auto'); ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--check-only', action='store_true'); ap.add_argument('--job')
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.job:
        require(not args.check_only, 'Invalid worker options'); worker(cfg, args.job, args.gpu); return
    jobs, hashes, stats = preflight(cfg, args.config)
    gpus = parse_gpus(args.gpus, torch.cuda.device_count())
    print(f'[preflight] {len(jobs)} locked jobs; four frozen arms; all raw targets; no training/selection', flush=True)
    if args.check_only:
        return
    out = output_root(cfg)
    require(not out.exists() or not any(out.iterdir()), 'Existing RGB evaluation; refuse overwrite/resume')
    out.mkdir(parents=True, exist_ok=True)
    m = dict(status='running', protocol=PROTOCOL, jobs=jobs, arms=ARMS, gpus=gpus,
             input_sha256=hashes, array_image_stats=stats, versions=versions(), python=platform.python_version(),
             git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    write_json(out / 'run_manifest.json', m)
    try:
        (out / 'references').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prior_root(cfg) / 'references/config.json', out / 'references/config.json')
        shutil.copyfile(prior_root(cfg) / 'run_manifest.json', out / 'references/method_manifest.json')
        shutil.copyfile(prior_root(cfg) / 'metrics.json', out / 'references/method_metrics.json')
        commands = [(j['job'], [sys.executable, '-m', 'src.method_deployment', '--config', args.config, '--job', j['job']]) for j in jobs]
        def stop(signum, frame):
            raise SystemExit(128+signum)
        previous = signal.signal(signal.SIGTERM, stop)
        try:
            run_parallel(commands, gpus, out / 'logs')
        finally:
            signal.signal(signal.SIGTERM, previous)
        receipts = [read_json(out / 'workers' / f"{j['job']}.json") for j in jobs]
        require(all(r['status'] == 'complete' and r['job'] == j['job'] and r['gpu'] in gpus for r, j in zip(receipts, jobs)), 'Incomplete RGB workers')
        rows, costs = [r for w in receipts for r in w['rows']], [r for w in receipts for r in w['cost_rows']]
        require(len(rows) == 288 and len(costs) == 240, 'Incomplete RGB/cost matrix')
        require(inputs_unchanged(hashes, stats), 'Original inputs changed during RGB evaluation')
        write_json(out / 'metrics.json', dict(rows=rows, cost_rows=costs))
        write_csv(out / 'inference_costs.csv', costs)
        m.update(status='complete', original_inputs_unchanged=True, n_metric_rows=len(rows),
                 output_sha256={p.relative_to(out).as_posix(): sha256(p) for p in out.rglob('*') if p.is_file() and p.name != 'run_manifest.json'})
    except BaseException as exc:
        m.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(out / 'run_manifest.json', m); raise
    write_json(out / 'run_manifest.json', m)


if __name__ == '__main__':
    main()
