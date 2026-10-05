"""Validate and summarize the predeclared method comparison; never select models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .prediction_io import sha256


ARMS = ["original", "matched_tail", "no_tail", "untrained_adapter"]
DATASETS = ["Brain", "liver", "RESC", "OCT2017", "RSNA", "camelyon16"]
PAIRS = ["medical_to_medical", "medical_to_general", "general_to_medical", "general_to_general"]
MAPPING = dict(Brain="brain_mri", liver="liver_ct", RESC="oct", OCT2017="oct", RSNA="chest_xray", camelyon16="pathology")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def validate(directory):
    m = read(directory / "run_manifest.json")
    require(m["status"] == "complete" and m["original_inputs_unchanged"] and m["no_reselection"]
            and m["score_unchanged"] and m["original_budget_unchanged"] and not m["training_uses_target_data"]
            and not m["training_uses_synthetic_anomalies"], "Invalid method experiment protocol/receipt")
    require(len(m["jobs"]) == 12 and m["arms"] == ARMS and m["n_metric_rows"] == 288
            and len(m["weights_sha256"]) == 184, "Incomplete method experiment")
    scheduling = m.get("scheduling")
    require(scheduling in [None, "one_job_per_gpu_v1"], "Unknown GPU scheduling protocol")
    expected_outputs = 378 if scheduling else 306
    require(len(m["output_sha256"]) == expected_outputs and {"metrics.json", "training_completed.json", "training_samples.json", "composition_checks.csv", "references/config.json"}
            <= set(m["output_sha256"]), "Incomplete returned metrics/predictions/references")
    for name, digest in m["output_sha256"].items():
        require(sha256(directory / name) == digest, f"Output hash differs: {name}")
    if scheduling:
        from .method_schedule import collect
        require(m["gpus"] and len(set(m["gpus"])) == len(m["gpus"]), "Duplicate/empty GPU plan")
        for phase in ["compose", "train", "evaluate"]:
            collect(directory, phase, m["jobs"], m["gpus"])
            for job in m["jobs"]:
                require(f"workers/{phase}/{job['job']}.json" in m["output_sha256"]
                        and f"logs/{phase}/{job['job']}.log" in m["output_sha256"], "Missing GPU worker evidence/log")
        require(sha256(directory / "training_completed.json") == m["training_completed_sha256"],
                "Training receipt differs from evaluation barrier")
    training = read(directory / "training_completed.json")
    require(training["no_target_data_used"] and training["no_reselection"] and len(training["rows"]) == 180,
            "Missing all-training-before-target-evaluation receipt")
    init = {}
    cells = set()
    for r in training["rows"]:
        cell = (r["arm"], r["job"], r["held_out"])
        require(cell not in cells and r["arm"] in ARMS[1:], "Duplicate/invalid training cell")
        cells.add(cell)
        require(r["n_normal_images"] == 4000 and
                r["optimizer_steps"] == (0 if r["arm"] == "untrained_adapter" else 8 * 63), "Training budget differs")
        key = (r["job"], r["held_out"])
        require(key not in init or init[key] == r["initial_adapter_sha256"], "Matched initializations differ")
        init[key] = r["initial_adapter_sha256"]
    require(len(init) == 60, "Incomplete folds")
    expected_training = {(a, j["job"], h) for a in ARMS[1:] for j in m["jobs"]
                         for h in ["brain_mri", "chest_xray", "liver_ct", "oct", "pathology"]}
    require(cells == expected_training, "Training matrix identities differ")
    for r in training["rows"]:
        require(m["weights_sha256"].get(f"checkpoints/{r['arm']}/{r['job']}/{r['held_out']}.pt") == r["checkpoint_sha256"],
                "Control weights changed after training receipt")
    expected_weights = {f"checkpoints/{a}/{j}/{h}.pt" for a,j,h in expected_training}
    expected_weights |= {f"bundles/{j['job']}_brain_mri.pt" for j in m["jobs"] if j["seed"] == 11}
    require(set(m["weights_sha256"]) == expected_weights, "Wrong control/bundle weight matrix")
    samples = read(directory / "training_samples.json")["rows"]
    require(len(samples) == 60 and {(r["job"], r["held_out"]) for r in samples} == set(init), "Missing exact source sampling manifest")
    for fold in samples:
        take = fold["samples"]
        require(len(take) == 4000 and len({r["image"] for r in take}) == 4000
                and all(r["dataset"] in MAPPING and r["source_modality"] == MAPPING[r["dataset"]]
                        and r["source_modality"] != fold["held_out"] for r in take), "Held-out contamination/duplicate source sample")
        for mod in set(MAPPING.values()) - {fold["held_out"]}:
            require(sum(r["source_modality"] == mod for r in take) == 1000, "Unequal source-modality sample budget")
    with (directory / "composition_checks.csv").open(encoding="utf-8", newline="") as f:
        composition = list(csv.DictReader(f))
    require(len(composition) == m["n_composition_checks"] == 288 and
            all(np.isfinite(float(r["map_max_abs_delta"])) and float(r["map_max_abs_delta"]) <= 1e-6 for r in composition),
            "Incomplete or failed live parameter-composition check")
    require({(r["job"], r["held_out"], r["source_dataset"]) for r in composition} ==
            {(j["job"], h, d) for j in m["jobs"] for h in set(MAPPING.values()) for d in DATASETS if MAPPING[d] != h}
            and all(int(r["n_images"]) == 2 for r in composition), "Wrong live source-fold matrix")
    bundles = [r for r in composition if r["bundle_roundtrip_delta"] != ""]
    require(len(bundles) == 4 and all(float(r["bundle_roundtrip_delta"]) <= 1e-6 for r in bundles), "Bundle round trip failed")
    data = read(directory / "metrics.json")
    from .followup_io import common_jobs
    from .prediction_io import compare_metrics, validate_final
    cfg = read(directory / "references/config.json")
    selected = read(directory / "references/selected_configs.json")
    locked = [j for j in common_jobs(cfg, selected) if j["reuse_original"]]
    require({j["job"] for j in locked} == {j["job"] for j in m["jobs"]}, "Changed locked main jobs")
    for job in m["jobs"]:
        original = read(directory / "references" / f"{job['job']}.json")
        validate_final(cfg, selected, original, job["source_backbone"], job["target_backbone"],
                       job["source_stage"], job["target_block"], job["seed"], "mlp", .05, ["contrast_topk"])
        replay = compare_metrics([r for r in data["rows"] if r["arm"] == "original" and r["job"] == job["job"]], original["rows"], 1e-6)
        require(all(r["within_tolerance"] for r in replay), "Independent original main replay differs")
    require(len(data["rows"]) == 288 and len(data["aoss_rows"]) == 240
            and len(data["replays"]) == 144 and all(r["within_tolerance"] for r in data["replays"]), "Main replay/comparison incomplete")
    require({(r["arm"], r["job"], r["held_out"]) for r in data["aoss_rows"]} ==
            {(a, j["job"], h) for a in ARMS for j in m["jobs"] for h in set(MAPPING.values())}, "Incomplete source-only AOSS matrix")
    for r in data["aoss_rows"]:
        require(np.isfinite([r[k] for k in ["normal_discrepancy", "perturb_in", "perturb_out", "local_sensitivity", "aoss"]]).all()
                and abs(r["local_sensitivity"]-r["perturb_in"]+r["perturb_out"]) <= 1e-10
                and abs(r["aoss"]-r["local_sensitivity"]/(r["normal_discrepancy"]+1e-8)) <= 1e-10, "Source AOSS arithmetic differs")
    seen = set()
    for r in data["rows"]:
        pair = r["source_backbone"] + "_to_" + r["target_backbone"]
        cell = (r["arm"], pair, r["seed"], r["dataset"])
        require(cell not in seen and r["score_mode"] == "contrast_topk"
                and np.isfinite([r["image_auroc"], r["image_aupr"]]).all(), "Duplicate/nonfinite/wrong-score metric")
        seen.add(cell)
    expected = {(a, p, s, d) for a in ARMS for p in PAIRS for s in [11, 22, 33] for d in DATASETS}
    require(seen == expected, "Method metric cells differ from declared matrix")
    # Independently recompute every returned image metric; labels only enter here.
    from sklearn.metrics import average_precision_score, roc_auc_score
    ordering = {}
    for r in data["rows"]:
        path = directory / "predictions" / r["arm"] / r["job"] / f"{r['dataset']}.csv"
        with path.open(encoding="utf-8", newline="") as f:
            predictions = list(csv.DictReader(f))
        require(len(predictions) == r["n_test"] and all(int(p["record_index"]) == i and
                p["dataset"] == r["dataset"] and p["arm"] == r["arm"] and p["job"] == r["job"]
                and int(p["seed"]) == r["seed"] and p["held_out"] == r["held_out"]
                and p["score_mode"] == "contrast_topk" and float(p["topk_fraction"]) == .05
                for i, p in enumerate(predictions)), "Prediction identity/count/order differs")
        labels = np.array([int(p["label"]) for p in predictions])
        scores = np.array([float(p["score"]) for p in predictions])
        require(set(labels) == {0, 1} and np.isfinite(scores).all(), "Invalid prediction label/score")
        identity = [(p["image_path"], int(p["label"])) for p in predictions]
        key = r["dataset"]
        require(key not in ordering or ordering[key] == identity, "Unpaired targets across arms/seeds/pairs")
        ordering[key] = identity
        require(abs(roc_auc_score(labels, scores) - r["image_auroc"]) <= 1e-12
                and abs(average_precision_score(labels, scores) - r["image_aupr"]) <= 1e-12,
                "Returned prediction metrics do not reproduce aggregate")
    return m, data


def summaries(rows):
    seed_rows, differences = [], []
    groups = dict(overall=DATASETS, radiology=["Brain", "liver", "RSNA"],
                  non_radiology=["RESC", "OCT2017", "camelyon16"])
    groups.update({d: [d] for d in DATASETS})
    for arm in ARMS:
        for pair in PAIRS:
            for seed in [11, 22, 33]:
                subset = [r for r in rows if r["arm"] == arm and r["seed"] == seed
                          and r["source_backbone"] + "_to_" + r["target_backbone"] == pair]
                for group, datasets in groups.items():
                    take = [r for r in subset if r["dataset"] in datasets]
                    for metric in ["image_auroc", "image_aupr", "pixel_auroc", "pixel_aupr"]:
                        valid = [r for r in take if metric in r and np.isfinite(r[metric])]
                        if valid:
                            seed_rows.append(dict(arm=arm, pair=pair, seed=seed, group=group, metric=metric,
                                                  value=float(np.mean([r[metric] for r in valid])),
                                                  datasets="|".join(sorted(r["dataset"] for r in valid))))
    for pair in PAIRS:
        for seed in [11, 22, 33]:
            for group in groups:
                for metric in ["image_auroc", "image_aupr", "pixel_auroc", "pixel_aupr"]:
                    take = {r["arm"]: r for r in seed_rows if (r["pair"], r["seed"], r["group"], r["metric"]) == (pair, seed, group, metric)}
                    if len(take) != 4:
                        continue
                    require(len({r["datasets"] for r in take.values()}) == 1, "Unequal metric coverage")
                    for control in ["no_tail", "untrained_adapter", "original"]:
                        differences.append(dict(pair=pair, seed=seed, group=group, metric=metric,
                                                contrast=f"matched_tail_minus_{control}",
                                                delta=take["matched_tail"]["value"]-take[control]["value"],
                                                datasets=take[control]["datasets"]))
    summary = []
    for keys in sorted({(r["arm"], r["pair"], r["group"], r["metric"], r["datasets"]) for r in seed_rows}):
        vals = [r["value"] for r in seed_rows if (r["arm"], r["pair"], r["group"], r["metric"], r["datasets"]) == keys]
        summary.append(dict(zip(["arm", "pair", "group", "metric", "datasets"], keys), mean=float(np.mean(vals)), seed_sd=float(np.std(vals, ddof=1)), n_seeds=len(vals)))
    return seed_rows, summary, differences


def contrast_summary(differences):
    result = []
    for key in sorted({(r['pair'], r['group'], r['metric'], r['contrast'], r['datasets']) for r in differences}):
        values = [r['delta'] for r in differences
                  if (r['pair'], r['group'], r['metric'], r['contrast'], r['datasets']) == key]
        result.append(dict(zip(['pair', 'group', 'metric', 'contrast', 'datasets'], key),
                           mean_delta=float(np.mean(values)), paired_seed_sd=float(np.std(values, ddof=1)),
                           n_positive_seeds=sum(v > 0 for v in values), n_seeds=len(values)))
    return result


def plot_results(summary, contrasts, out):
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        if exc.name != 'matplotlib':
            raise
        print('[analysis] matplotlib unavailable; validated tables/report saved; plots skipped.')
        return False
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, metric in zip(axes, ['image_auroc', 'image_aupr']):
        for i, arm in enumerate(ARMS):
            take = [next(r for r in summary if r['arm'] == arm and r['pair'] == p and r['group'] == 'overall' and r['metric'] == metric) for p in PAIRS]
            ax.errorbar(np.arange(4) + (i-1.5)*.13, [r['mean'] for r in take], yerr=[r['seed_sd'] for r in take],
                        fmt='o', capsize=3, label=arm)
        ax.set_xticks(range(4), ['MM', 'MG', 'GM', 'GG'])
        ax.set_ylabel(metric); ax.set_title('Six datasets; mean ± seed SD')
        if metric == 'image_auroc':
            ax.axhline(.5, ls='--', color='gray', linewidth=1)
    axes[0].legend(fontsize=8)
    fig.savefig(out / 'method_comparison.png', dpi=180); fig.savefig(out / 'method_comparison.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, metric in zip(axes, ['image_auroc', 'pixel_auroc']):
        values = np.full((len(DATASETS), len(PAIRS)), np.nan)
        for i, d in enumerate(DATASETS):
            for j, p in enumerate(PAIRS):
                take = [r for r in contrasts if r['pair'] == p and r['group'] == d and r['metric'] == metric
                        and r['contrast'] == 'matched_tail_minus_no_tail']
                if take:
                    values[i, j] = take[0]['mean_delta']
        bound = max(.01, float(np.nanmax(np.abs(values)))) if np.isfinite(values).any() else .01
        handle = ax.imshow(np.ma.masked_invalid(values), cmap='RdBu', vmin=-bound, vmax=bound)
        for i, j in np.ndindex(values.shape):
            ax.text(j, i, f'{values[i,j]:+.3f}' if np.isfinite(values[i,j]) else 'no mask', ha='center', va='center', fontsize=8)
        ax.set_xticks(range(4), ['MM', 'MG', 'GM', 'GG']); ax.set_yticks(range(6), DATASETS)
        ax.set_title(f'Tail minus direct reconstruction: {metric}')
        fig.colorbar(handle, ax=ax, shrink=.7)
    fig.savefig(out / 'tail_dataset_effects.png', dpi=180); fig.savefig(out / 'tail_dataset_effects.pdf'); plt.close(fig)
    return True


def run(directory, out):
    m, data = validate(directory)
    out.mkdir(parents=True, exist_ok=True)
    seed_rows, summary, differences = summaries(data["rows"])
    write_csv(out / "seed_macros.csv", seed_rows)
    write_csv(out / "method_summary.csv", summary)
    write_csv(out / "paired_seed_differences.csv", differences)
    contrasts = contrast_summary(differences)
    write_csv(out / 'contrast_summary.csv', contrasts)
    plots = plot_results(summary, contrasts, out)
    text = """# 参数拼接新模型：固定方法对照

主对照为 matched_tail − no_tail：同初始 adapter、同正常 source 样本/顺序、同优化预算，差别是继承的 Transformer 后段。
matched_tail − untrained_adapter 检查 NFFA 学习作用。matched_tail − original 只检查新逐 fold RNG 与历史连续 RNG 的差异，不能称方法改进。
无后段对照保留固定输出 LayerNorm；prefix token 无梯度，有效参数数列于 training_completed.json。
正常 source 训练完成后才统一评价全部 target；未训练合成扰动，没有更新 AOSS、选点或评分。
原主 AUROC/AP 重放通过，原结果未覆盖。上述比较仅限现有锁定点，不代表完整 grid。
四 pair 的原 GG cut 不同；医学预训练因果效应仍应参考共同 stitch 对照。
global AOSS 跨 fold 信息边界保留；不能宣称全局选择严格排除了每一目标模态。
±为三个 seed 的 SD；没有患者 CI。有 mask 的 pixel 指标覆盖列单独报告。
完整 bundle/checkpoint 留在服务器；返回包仅含指标、预测、引用、哈希和日志。返回包无法本地重放权重，应核对服务器成功 receipt。
部署验证仅为正常 source 抽样；不替代全部目标图像端到端性能/延迟验证。
缓存与新鲜图像路径的差异单独报告，不因 live 路径一致便宣称旧 cache 精确等价。
多卡运行时，每张卡同时一个 pair/seed 任务，三阶段之间设全局等待；各任务实际 GPU、设备型号及日志在 workers/ 与 logs/。
"""
    text += '\n## 六数据集图像 AUROC（均值 ± seed SD）\n\n|arm|MM|MG|GM|GG|\n|---|---|---|---|---|\n'
    for arm in ARMS:
        take = [next(r for r in summary if (r['arm'], r['pair'], r['group'], r['metric']) ==
                     (arm, p, 'overall', 'image_auroc')) for p in PAIRS]
        text += '|' + arm + '|' + '|'.join(f"{r['mean']:.4f} ± {r['seed_sd']:.4f}" for r in take) + '|\n'
    text += '\n## 预定配对主对照\n\n|pair|指标|后段−直接重建|配对 seed SD|正向 seed 数|覆盖|\n|---|---|---|---|---|---|\n'
    for r in contrasts:
        if r['group'] == 'overall' and r['contrast'] == 'matched_tail_minus_no_tail':
            coverage = r['datasets'].replace('|', ', ')
            text += f"|{r['pair']}|{r['metric']}|{r['mean_delta']:+.4f}|{r['paired_seed_sd']:.4f}|{r['n_positive_seeds']}/{r['n_seeds']}|{coverage}|\n"
    text += '\n负差异和单数据集失败完整保留；不翻转分数、不换选点、不挑最佳 arm。三个 seed 的方向一致不等于患者层面的显著性。逐数据集和 radiology 分层见 method_summary.csv 与 contrast_summary.csv。\n'
    if not plots:
        text += '\n服务器未安装 matplotlib，本次跳过绘图；表格和科学核验完整。可在本地相同返回包重生成图，不需重训或升级服务器环境。\n'
    (out / "report_CN.md").write_text(text, encoding="utf-8")
    (out / "audit.json").write_text(json.dumps(dict(status="complete", selection_updated=False,
        server_original_inputs_unchanged=True, local_weight_replay=False, n_metrics=len(data["rows"]),
        n_output_hashes=len(m["output_sha256"]), plots_generated=plots), indent=2), encoding="utf-8")
    print(f"[analysis] Verified method comparison: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/followup/method_controls"))
    ap.add_argument("--output", type=Path, default=Path("results/analysis/method_controls"))
    args = ap.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
