"""Audit RGB predictions independently; preserve cached main results and selection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .analyze_method_controls import ARMS, DATASETS, PAIRS, MAPPING, contrast_summary, plot_results, summaries, write_csv
from .prediction_io import sha256


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(directory):
    from .method_deployment import PROTOCOL
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    m = read(directory / 'run_manifest.json')
    require(m['status'] == 'complete' and m['original_inputs_unchanged'] and m['protocol'] == PROTOCOL
            and m['arms'] == ARMS and len(m['jobs']) == 12, 'Invalid/incomplete RGB protocol')
    require(len(m['output_sha256']) == 317, 'Incomplete RGB return output matrix')
    for name, digest in m['output_sha256'].items():
        require(sha256(directory / name) == digest, f'RGB output hash differs: {name}')
    prior = read(directory / 'references/method_manifest.json')
    require(prior['status'] == 'complete' and prior['jobs'] == m['jobs'] and prior['arms'] == ARMS
            and prior['no_reselection'] and prior['score_unchanged'], 'Changed locked prior method matrix')
    require(sha256(directory / 'references/config.json') == prior['output_sha256']['references/config.json'],
            'Locked experiment config reference differs')
    require(sha256(directory / 'references/method_manifest.json') ==
            next(v for k, v in m['input_sha256'].items() if k.replace('\\', '/').endswith('method_controls/run_manifest.json')),
            'Prior method manifest reference differs')
    reference = read(directory / 'references/method_metrics.json')
    require(sha256(directory / 'references/method_metrics.json') == prior['output_sha256']['metrics.json'],
            'Cached metric reference differs')
    data = read(directory / 'metrics.json')
    jobs = {j['job']: j for j in m['jobs']}
    for j in m['jobs']:
        worker = read(directory / 'workers' / f"{j['job']}.json")
        require(worker['status'] == 'complete' and worker['job'] == j['job'] and worker['gpu'] in m['gpus']
                and worker['device_name'] and worker['device_total_memory'] > 0, 'Incomplete GPU hardware/worker receipt')
        require(worker['rows'] == [r for r in data['rows'] if r['job'] == j['job']]
                and worker['cost_rows'] == [r for r in data['cost_rows'] if r['job'] == j['job']], 'Worker and aggregate differ')
    cells, ordering = set(), {}
    for r in data['rows']:
        j = jobs[r['job']]
        cell = (r['arm'], r['job'], r['dataset'])
        require(cell not in cells and r['held_out'] == MAPPING[r['dataset']] and r['seed'] == j['seed']
                and r['source_backbone'] == j['source_backbone'] and r['target_backbone'] == j['target_backbone']
                and r['score_mode'] == 'contrast_topk' and 0 <= r['live_parity_max_abs_delta'] <= 1e-6
                and r['grid'] == 16, 'Duplicate/invalid RGB identity/parity')
        cells.add(cell)
        require(all(np.isfinite(r[k]) and 0 <= r[k] <= 1 for k in ['pixel_auroc', 'pixel_aupr'] if k in r),
                'Invalid pixel metric receipt')
        with (directory / 'predictions' / r['arm'] / r['job'] / f"{r['dataset']}.csv").open(encoding='utf-8', newline='') as f:
            p = list(csv.DictReader(f))
        require(len(p) == r['n_test'] and all(int(x['record_index']) == i and x['job'] == r['job']
                and x['arm'] == r['arm'] and x['dataset'] == r['dataset'] and int(x['seed']) == r['seed']
                and x['held_out'] == r['held_out'] and x['score_mode'] == 'contrast_topk'
                and float(x['topk_fraction']) == .05 for i, x in enumerate(p)), 'RGB prediction identity/order differs')
        labels = np.array([int(x['label']) for x in p])
        scores = np.array([float(x['score']) for x in p]); cached = np.array([float(x['cached_score']) for x in p])
        require(set(labels) == {0, 1} and np.isfinite(scores).all() and np.isfinite(cached).all(), 'Invalid RGB labels/scores')
        identity = [(x['image_path'], int(x['label'])) for x in p]
        require(r['dataset'] not in ordering or ordering[r['dataset']] == identity, 'Targets unpaired across arms/seeds')
        ordering[r['dataset']] = identity
        require(len(set(x[0] for x in identity)) == len(identity), 'Duplicated target images')
        old = next(x for x in reference['rows'] if (x['arm'], x['job'], x['dataset']) == cell)
        for name, func in [('image_auroc', roc_auc_score), ('image_aupr', average_precision_score)]:
            require(abs(func(labels, scores)-r[name]) <= 1e-12 and abs(func(labels, cached)-r['cached_'+name]) <= 1e-12
                    and abs(old[name]-r['cached_'+name]) <= 1e-12, 'RGB/cached predictions do not reproduce metrics')
        require(abs(np.mean(np.abs(scores-cached))-r['score_mean_abs_delta']) <= 1e-12
                and abs(np.max(np.abs(scores-cached))-r['score_max_abs_delta']) <= 1e-12, 'RGB/cache difference receipt differs')
    require(cells == {(a, j, d) for a in ARMS for j in jobs for d in DATASETS}, 'Incomplete RGB target matrix')
    cost_cells = set()
    for r in data['cost_rows']:
        cell = (r['arm'], r['job'], r['held_out'])
        require(cell not in cost_cells and r['latency_median_ms'] > 0 and r['latency_p95_ms'] >= r['latency_median_ms']
                and r['peak_allocated_bytes'] >= r['resident_allocated_bytes'] > 0
                and r['total_unique'] == r['cnn_prefix']+r['adapter']+r['full_reference'], 'Invalid inference cost receipt')
        cost_cells.add(cell)
    require(cost_cells == {(a, j, h) for a in ARMS for j in jobs for h in set(MAPPING.values())}, 'Incomplete detector cost matrix')
    return m, data


def run(directory, out):
    m, data = validate(directory)
    out.mkdir(parents=True, exist_ok=True)
    seed, summary, differences = summaries(data['rows']); contrasts = contrast_summary(differences)
    for name, rows in [('seed_macros', seed), ('method_summary', summary), ('paired_seed_differences', differences),
                       ('contrast_summary', contrasts), ('inference_costs', data['cost_rows'])]:
        write_csv(out / f'{name}.csv', rows)
    drift = [dict(arm=r['arm'], job=r['job'], dataset=r['dataset'], seed=r['seed'],
                  image_auroc_delta=r['image_auroc']-r['cached_image_auroc'],
                  image_ap_delta=r['image_aupr']-r['cached_image_aupr'],
                  score_mean_abs_delta=r['score_mean_abs_delta'], score_max_abs_delta=r['score_max_abs_delta']) for r in data['rows']]
    write_csv(out / 'rgb_cache_differences.csv', drift)
    plots = plot_results(summary, contrasts, out)
    text = ('# 冻结拼接模型全目标 RGB 推理审计\n\n'
            '固定四 pair、三个 seed、四 arm、原选点、224 输入和 contrast_topk；未训练、未选点。'
            '全部目标原图进入完整 CNN prefix + adapter + suffix/reference 检测器；cached 主结果保留。\n\n'
            '先按六 dataset 等权，再汇总三个 seed；SD 不是患者 CI。pixel 指标保持原 16×16 mask 规则，'
            '不能称原始分辨率病变定位；无 mask 数据不补造 mask。在线路径一致性只核对每 dataset 开头两张图。\n\n'
            '成本计完整 RGB tensor 前向：归一化、CNN prefix、接口、继承 suffix 和完整 reference；'
            '不含解码/传输，不冒称临床端到端延迟。batch=1、5 warmup、30 次 CUDA event；'
            '显存记当前 resident、peak allocated/reserved。各 worker 同卡串行，外部 GPU 进程仍可影响时延。\n\n'
            '|arm|MM image AUROC|MG|GM|GG|\n|---|---|---|---|---|\n')
    for arm in ARMS:
        take = [next(r for r in summary if (r['arm'], r['pair'], r['group'], r['metric']) == (arm, p, 'overall', 'image_auroc')) for p in PAIRS]
        text += '|' + arm + '|' + '|'.join(f"{r['mean']:.4f} ± {r['seed_sd']:.4f}" for r in take) + '|\n'
    text += '\n数值差异只报告，不提高容差、翻转分数、换评分或覆盖历史结果。全目标 map 未返回，本地独立复算图像指标；pixel/成本来自服务器 receipts。\n'
    (out / 'report_CN.md').write_text(text, encoding='utf-8')
    (out / 'audit.json').write_text(json.dumps(dict(status='complete', n_metrics=len(data['rows']),
          n_cost_rows=len(data['cost_rows']), local_gpu_execution=False, selection_updated=False, plots_generated=plots), indent=2), encoding='utf-8')
    print(f'[RGB analysis] Verified: {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=Path('results/followup/method_deployment'))
    ap.add_argument('--output', type=Path, default=Path('results/analysis/method_deployment'))
    args = ap.parse_args(); run(args.input, args.output)


if __name__ == '__main__':
    main()
