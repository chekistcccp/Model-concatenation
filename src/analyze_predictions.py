"""Read-only post-hoc analysis of locked-final per-image exports.

python -m src.analyze_predictions --root . --bootstrap 1000
Writes only results/analysis/predictions and results/figures/predictions.
No selection, training, threshold optimization or replacement scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
import yaml

from .analyze_results import EFFECTS, PAIRS, digest, md, pair


def weighted_metrics(labels, score, weights):
    """Exact empirical AUROC/AP with integer multiplicities, including ties."""
    groups = np.unique(score, return_inverse=True)[1]
    positive = np.bincount(groups, weights=weights * labels)
    negative = np.bincount(groups, weights=weights * (1 - labels), minlength=len(positive))
    return grouped_metrics(positive, negative)


def grouped_metrics(positive, negative):
    np_, nn = positive.sum(), negative.sum()
    below = np.cumsum(negative) - negative
    auc = np.dot(positive, below + .5 * negative) / (np_ * nn)
    above_pos = np.cumsum(positive[::-1])
    above_total = np.cumsum((positive + negative)[::-1])
    precision = np.divide(above_pos, above_total, out=np.zeros_like(above_pos), where=above_total > 0)
    ap = np.dot(positive[::-1], precision) / np_
    return auc, ap


def bootstrap_metrics(labels, scores, count, rng):
    """Shared stratified image resamples across all pair/seed columns."""
    labels = np.asarray(labels, dtype=int)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    groups = [np.unique(scores[:, j], return_inverse=True)[1] for j in range(scores.shape[1])]
    draws = np.empty((count, scores.shape[1], 2))
    for b in range(count):
        weights = np.zeros(len(labels))
        weights[pos] = rng.multinomial(len(pos), np.full(len(pos), 1 / len(pos)))
        weights[neg] = rng.multinomial(len(neg), np.full(len(neg), 1 / len(neg)))
        for j, group in enumerate(groups):
            n = int(group.max()) + 1
            positive = np.bincount(group, weights=weights * labels, minlength=n)
            negative = np.bincount(group, weights=weights * (1 - labels), minlength=n)
            draws[b, j] = grouped_metrics(positive, negative)
    return draws


def rank_contribution(labels, scores):
    """Each image's correctly ordered normal/abnormal pair fraction; no cutoff."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    result = np.empty(len(labels))
    normal, abnormal = np.sort(scores[labels == 0]), np.sort(scores[labels == 1])
    for label, opposite in [(1, normal), (0, abnormal)]:
        own = scores[labels == label]
        fractions = (np.searchsorted(opposite, own, "left") + np.searchsorted(opposite, own, "right")) / (2 * len(opposite))
        result[labels == label] = fractions if label == 1 else 1 - fractions
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()
    if args.bootstrap < 100:
        ap.error("--bootstrap must be >=100")
    root = args.root.resolve()
    cfg_path = root / "configs/experiment.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    results = root / cfg["paths"]["results_dir"]
    pred, out, figs = results / "predictions", results / "analysis/predictions", results / "figures/predictions"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    input_paths = sorted(p for p in pred.rglob("*") if p.is_file())
    input_paths += sorted((results / "final").glob("*.json")) + [results / "selected_configs.json", cfg_path]
    input_paths += [root / "cache/manifest.jsonl"] if (root / "cache/manifest.jsonl").is_file() else []
    before = digest(input_paths)
    manifest = json.loads((pred / "export_manifest.json").read_text(encoding="utf-8"))
    selected = json.loads((results / "selected_configs.json").read_text(encoding="utf-8"))
    checks, frames, metric_rows, distributions, diagnostics, subtypes = [], {}, [], [], [], []

    def check(name, valid, note=""):
        checks.append(dict(check=name, status="PASS" if valid else "FAIL", note=note))

    def save(name, frame):
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")

    check("export_complete", manifest["status"] == "complete" and manifest["original_inputs_unchanged"])
    check("posthoc_only", manifest["posthoc_only"] and manifest["selection_policy"] == "reuse_locked_selection_without_ranking")
    check("csv_count", len(list(pred.rglob("*.csv"))) == 72)
    check("evaluation_count", len(list(pred.glob("*/evaluation.json"))) == 12)
    config_note = "exact file bytes"
    for rel, expected in manifest["original_sha256"].items():
        path = root / rel
        valid = path.is_file() and digest([path])[str(path)] == expected
        if path == cfg_path and not valid:
            git_cfg = subprocess.check_output(["git", "show", f"{manifest['git_head']}:configs/experiment.yaml"], cwd=root)
            valid = hashlib.sha256(git_cfg).hexdigest() == expected and yaml.safe_load(git_cfg) == cfg
            config_note = "server Git config hash matches; local YAML identical, working-copy line endings differ"
        check(f"original_hash:{rel}", valid, config_note if path == cfg_path else "exact bytes")
    dataset_meta = {}
    if (root / "cache/manifest.jsonl").is_file():
        for line in (root / "cache/manifest.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["split"] == "test":
                dataset_meta[r["image"]] = r
    expected_jobs = {f"{i['config']}_seed{s}" for i in selected["selected"] for s in cfg["final"]["seeds"]}
    check("job_coverage", {j["job"] for j in manifest["jobs"]} == expected_jobs and len(manifest["jobs"]) == 12)
    for job in manifest["jobs"]:
        directory = pred / job["job"]
        evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
        original = json.loads((root / job["reference"]).read_text(encoding="utf-8"))
        check(f"replay_receipt:{job['job']}", evaluation["replay_matches_original"] and evaluation["no_reselection"] and all(c["within_tolerance"] for c in evaluation["comparison"]))
        check(f"receipt_identity:{job['job']}", evaluation["config"] == job["config"] and evaluation["seed"] == job["seed"] and pair(evaluation) == pair(job))
        check(f"receipt_original_hash:{job['job']}", evaluation["original_final_sha256"] == before[str(root / job["reference"])])
        for ds in cfg["data"]["datasets"]:
            path = directory / f"{ds}.csv"
            d = pd.read_csv(path, float_precision="round_trip")
            proof = next(x for x in evaluation["provenance"] if x["dataset"] == ds)
            check(f"csv_hash:{job['job']}:{ds}", before[str(path)] == proof["prediction_sha256"])
            check(f"finite:{job['job']}:{ds}", np.isfinite(d[["score", "discrepancy_mean", "discrepancy_median", "discrepancy_max"]]).all().all())
            check(f"score_bounds:{job['job']}:{ds}",
                  d.score.ge(-1e-6).all() and d.discrepancy_mean.between(-1e-6, 2 + 1e-6).all()
                  and d.discrepancy_median.between(-1e-6, 2 + 1e-6).all()
                  and d.discrepancy_max.between(-1e-6, 2 + 1e-6).all()
                  and d.discrepancy_max.ge(d.discrepancy_mean - 1e-6).all()
                  and d.score.le(d.discrepancy_max - d.discrepancy_median + 1e-6).all())
            check(f"labels_order:{job['job']}:{ds}", set(d.label) == {0, 1} and d.record_index.tolist() == list(range(len(d))) and not d.image_path.duplicated().any())
            identity = dict(dataset=ds, modality=cfg["data"]["modality_map"][ds], config=job["config"], seed=job["seed"],
                            source_backbone=job["source_backbone"], target_backbone=job["target_backbone"],
                            source_stage=job["source_stage"], target_block=job["target_block"], adapter=cfg["final"]["adapter"],
                            score_mode="contrast_topk", topk_fraction=cfg["final"]["topk_fraction"])
            check(f"csv_identity:{job['job']}:{ds}", all(d[k].eq(v).all() for k, v in identity.items()))
            domain = "radiology" if cfg["data"]["modality_map"][ds] in {"brain_mri", "liver_ct", "chest_xray"} else "non_radiology"
            check(f"csv_pair_domain:{job['job']}:{ds}", d.pair.eq(f"{job['source_backbone']}_to_{job['target_backbone']}").all() and d.domain_group.eq(domain).all())
            if dataset_meta:
                check(f"manifest_labels:{job['job']}:{ds}", all(p in dataset_meta and dataset_meta[p]["label"] == l and dataset_meta[p]["dataset"] == ds for p, l in zip(d.image_path, d.label)))
            old = next(r for r in original["rows"] if r["dataset"] == ds)
            auc, ap_ = roc_auc_score(d.label, d.score), average_precision_score(d.label, d.score)
            check(f"recomputed_metrics:{job['job']}:{ds}", len(d) == old["n_test"] and abs(auc - old["image_auroc"]) <= 1e-6 and abs(ap_ - old["image_aupr"]) <= 1e-6)
            metric_rows.append(dict(dataset=ds, pair=pair(job), seed=job["seed"], auroc=auc, aupr=ap_, auroc_delta=auc-old["image_auroc"], aupr_delta=ap_-old["image_aupr"], n_images=len(d)))
            frames[(ds, pair(job), job["seed"])] = d
            for label, part in d.groupby("label"):
                for statistic in ["score", "discrepancy_mean", "discrepancy_median", "discrepancy_max"]:
                    distributions.append(dict(dataset=ds, pair=pair(job), seed=job["seed"], label=label, statistic=statistic,
                                              count=len(part), mean=part[statistic].mean(), sd=part[statistic].std(),
                                              q05=part[statistic].quantile(.05), q25=part[statistic].quantile(.25), median=part[statistic].median(),
                                              q75=part[statistic].quantile(.75), q95=part[statistic].quantile(.95)))
            # For fixed k <= half the patch count, all top-k entries are >= median.
            # Thus raw_topk = contrast_topk + median exactly; diagnostic only.
            grid = cfg["models"]["targets"][job["target_backbone"]]["input_size"] // cfg["models"]["targets"][job["target_backbone"]]["patch_size"]
            k = max(1, round(grid * grid * cfg["final"]["topk_fraction"]))
            check(f"derived_raw_identity_condition:{job['job']}:{ds}", k <= (grid * grid + 1) // 2)
            for name, values in [("primary_contrast_topk", d.score), ("derived_raw_topk", d.score + d.discrepancy_median), ("mean", d.discrepancy_mean), ("median", d.discrepancy_median), ("max", d.discrepancy_max)]:
                diagnostics.append(dict(dataset=ds, pair=pair(job), seed=job["seed"], diagnostic=name, auroc=roc_auc_score(d.label, values), posthoc_only=True))
            if ds == "OCT2017":
                classes = d.image_path.map(lambda p: PurePosixPath(p).name.split("-")[0])
                for subtype in ["CNV", "DME", "DRUSEN"]:
                    mask = classes.isin(["NORMAL", subtype])
                    subtypes.append(dict(pair=pair(job), seed=job["seed"], subtype=subtype, auroc=roc_auc_score(d.loc[mask, "label"], d.loc[mask, "score"]), n_normal=int((classes == "NORMAL").sum()), n_abnormal=int((classes == subtype).sum())))
    save("integrity_checks", pd.DataFrame(checks))
    if any(c["status"] == "FAIL" for c in checks):
        raise ValueError("Prediction integrity failed; no inferential report generated")
    metrics = pd.DataFrame(metric_rows)
    save("recomputed_metrics", metrics)
    save("score_distributions", pd.DataFrame(distributions))
    diag = pd.DataFrame(diagnostics)
    save("scoring_diagnostics", diag)
    diag_mean = diag.groupby(["dataset", "pair", "diagnostic"], as_index=False).auroc.mean()
    save("scoring_diagnostic_mean", diag_mean)
    subtype_df = pd.DataFrame(subtypes)
    save("oct_subtype_metrics", subtype_df)
    stable_rows, correlations, bootstrap_draws, summary, effect_rows = [], [], {}, [], []
    sample_rows, macro_rows = [], []
    seeds = cfg["final"]["seeds"]
    rng = np.random.default_rng(args.seed)
    for ds in cfg["data"]["datasets"]:
        base = frames[(ds, "MM", seeds[0])]
        paths, labels = base.image_path.to_numpy(), base.label.to_numpy()
        sample_rows.append(dict(dataset=ds, n_images=len(labels), n_normal=int((labels == 0).sum()),
                                n_abnormal=int((labels == 1).sum()), abnormal_fraction=labels.mean()))
        columns = []
        for p in PAIRS:
            values, contributions = [], []
            for seed in seeds:
                d = frames[(ds, p, seed)]
                if not np.array_equal(d.image_path, paths) or not np.array_equal(d.label, labels):
                    raise ValueError("Cannot do paired resampling: image order/labels differ across pair or seed")
                values.append(d.score.to_numpy())
                contributions.append(rank_contribution(labels, d.score))
            contribution = np.column_stack(contributions)
            for label in [0, 1]:
                candidates = pd.DataFrame(dict(image_path=paths[labels == label], label=label, mean_pair_order_contribution=contribution[labels == label].mean(1),
                                               min_contribution=contribution[labels == label].min(1), max_contribution=contribution[labels == label].max(1),
                                               below_half_all_seeds=(contribution[labels == label] < .5).all(1)))
                stable_rows.extend(candidates.sort_values("mean_pair_order_contribution").head(20).assign(dataset=ds, pair=p).to_dict("records"))
            for a, b in itertools.combinations(range(len(seeds)), 2):
                for label in [-1, 0, 1]:
                    mask = np.ones(len(labels), dtype=bool) if label == -1 else labels == label
                    correlations.append(dict(dataset=ds, pair=p, seed_a=seeds[a], seed_b=seeds[b], label=label, rho=spearmanr(values[a][mask], values[b][mask]).statistic))
            columns.extend(values)
        print(f"[predictions analysis] {ds}: {len(labels)} images, paired image bootstrap {args.bootstrap}", flush=True)
        draws = bootstrap_metrics(labels, np.column_stack(columns), args.bootstrap, rng)
        draws = draws.reshape(args.bootstrap, len(PAIRS), len(seeds), 2).mean(axis=2)
        bootstrap_draws[ds] = draws
        for i, p in enumerate(PAIRS):
            original_metrics = metrics[(metrics.dataset == ds) & (metrics.pair == p)]
            for j, name in enumerate(["auroc", "aupr"]):
                lower, upper = np.quantile(draws[:, i, j], [.025, .975])
                summary.append(dict(dataset=ds, pair=p, metric=name, mean=original_metrics[name].mean(), seed_sd=original_metrics[name].std(),
                                    image_bootstrap_lower=lower, image_bootstrap_upper=upper, n_images=len(labels), n_seeds=len(seeds), n_bootstrap=args.bootstrap))
    save("dataset_summary_with_image_intervals", pd.DataFrame(summary))
    correlation_df = pd.DataFrame(correlations)
    save("cross_seed_rank_correlation", correlation_df)
    correlation_summary = correlation_df.groupby(["dataset", "pair", "label"], as_index=False).agg(
        min_rho=("rho", "min"), mean_rho=("rho", "mean"), max_rho=("rho", "max"))
    save("cross_seed_rank_summary", correlation_summary)
    samples = pd.DataFrame(sample_rows)
    save("test_sample_counts", samples)
    save("error_case_candidates", pd.DataFrame(stable_rows))
    scopes = {ds: [ds] for ds in cfg["data"]["datasets"]}
    scopes.update(radiology=[d for d in cfg["data"]["datasets"] if cfg["data"]["modality_map"][d] in {"brain_mri", "liver_ct", "chest_xray"}],
                  non_radiology=[d for d in cfg["data"]["datasets"] if cfg["data"]["modality_map"][d] not in {"brain_mri", "liver_ct", "chest_xray"}], overall=cfg["data"]["datasets"])
    for scope, ds_list in scopes.items():
        all_draws = np.stack([bootstrap_draws[d] for d in ds_list]).mean(axis=0)
        draws = all_draws[:, :, 0]
        by_seed = metrics[metrics.dataset.isin(ds_list)].groupby(["pair", "seed"]).auroc.mean().unstack("pair")[PAIRS]
        if scope in {"radiology", "non_radiology", "overall"}:
            for j, name in enumerate(["auroc", "aupr"]):
                seed_values = metrics[metrics.dataset.isin(ds_list)].groupby(["pair", "seed"])[name].mean().unstack("pair")[PAIRS]
                for i, p in enumerate(PAIRS):
                    lower, upper = np.quantile(all_draws[:, i, j], [.025, .975])
                    macro_rows.append(dict(scope=scope, pair=p, metric=name, mean=seed_values[p].mean(), seed_sd=seed_values[p].std(),
                                           image_bootstrap_lower=lower, image_bootstrap_upper=upper, n_datasets=len(ds_list), n_bootstrap=args.bootstrap))
        for name, weights in EFFECTS.items():
            observed = by_seed.to_numpy() @ weights
            lower, upper = np.quantile(draws @ weights, [.025, .975])
            effect_rows.append(dict(scope=scope, effect=name, mean=observed.mean(), seed_sd=observed.std(ddof=1),
                                    image_bootstrap_lower=lower, image_bootstrap_upper=upper, n_bootstrap=args.bootstrap))
    effects = pd.DataFrame(effect_rows)
    save("paired_pretraining_effect_intervals", effects)
    macro_summary = pd.DataFrame(macro_rows)
    save("macro_summary_with_image_intervals", macro_summary)
    save("oct_subtype_summary", subtype_df.groupby(["pair", "subtype"], as_index=False).agg(mean=("auroc", "mean"), seed_sd=("auroc", "std")))
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    colors = ["#2166ac", "#67a9cf", "#ef8a62", "#b2182b"]

    def figure_save(fig, name):
        for ext in ["pdf", "png"]:
            fig.savefig(figs / f"{name}.{ext}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    for kind in ["roc", "pr", "distribution"]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        fig.suptitle(f"{'ROC' if kind == 'roc' else 'Precision-recall' if kind == 'pr' else 'Primary score distribution'}: solid / dashed / dotted = seeds " + " / ".join(map(str, seeds)), fontsize=10)
        for ax, ds in zip(axes.flat, cfg["data"]["datasets"]):
            for color, p in zip(colors, PAIRS):
                for i, seed in enumerate(seeds):
                    d = frames[(ds, p, seed)]
                    if kind == "roc":
                        x, y, _ = roc_curve(d.label, d.score)
                    elif kind == "pr":
                        y, x, _ = precision_recall_curve(d.label, d.score)
                    else:
                        continue
                    ax.plot(x, y, color=color, lw=1, alpha=.65, ls=["-", "--", ":"][i], label=p if i == 0 else None)
            if kind == "distribution":
                p = "GM" if ds in ["OCT2017", "RESC"] else "MM"
                for i, seed in enumerate(seeds):
                    d = frames[(ds, p, seed)]
                    for label, color in [(0, "#2166ac"), (1, "#b2182b")]:
                        values = np.sort(d.loc[d.label == label, "score"])
                        ax.plot(values, np.arange(1, len(values)+1)/len(values), color=color, ls=["-", "--", ":"][i], label=f"{'normal' if label == 0 else 'abnormal'}" if i == 0 else None)
                ax.set(xlabel="Primary contrast_topk score", ylabel="Empirical CDF", title=f"{ds}: {p} (three seeds)")
            elif kind == "roc":
                ax.plot([0, 1], [0, 1], c="gray", ls=":", lw=1)
                ax.set(xlabel="False positive rate", ylabel="True positive rate", title=ds, xlim=(0,1), ylim=(0,1))
            else:
                ax.axhline(frames[(ds, "MM", seeds[0])].label.mean(), color="gray", ls=":")
                ax.set(xlabel="Recall", ylabel="Precision", title=ds, xlim=(0,1), ylim=(0,1))
            ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        figure_save(fig, kind)
    chosen = [("OCT2017", "GM"), ("RESC", "GM"), ("Brain", "MM")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    diagnostics_order = ["primary_contrast_topk", "derived_raw_topk", "mean", "median", "max"]
    for ax, (ds, p) in zip(axes, chosen):
        for i, seed in enumerate(seeds):
            data = diag[(diag.dataset == ds) & (diag.pair == p) & (diag.seed == seed)].set_index("diagnostic").loc[diagnostics_order]
            ax.plot(range(5), data.auroc, "o-", label=f"seed {seed}")
        ax.axhline(.5, color="gray", ls=":")
        ax.set(xticks=range(5), xticklabels=["contrast", "raw*", "mean*", "median*", "max*"], title=f"{ds}: {p}", ylabel="AUROC (diagnostic only)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    figure_save(fig, "score_component_diagnostics")
    summary_df = pd.DataFrame(summary)
    lines = ["# 已锁定 final 的逐图分数分析", "",
             "本报告复用主实验的 contrast_topk、固定配置及三 seeds。所有新比较是 post-hoc 诊断；不选点、不选 seed、不优化阈值、不替换主评分。", "",
             "## 完整性", "",
             f"72 CSV / 12 evaluation JSON / 12 jobs 完整。{len(checks)} 项检查通过；CSV 哈希与服务器 receipt 一致。"
             + ("图像路径/标签与本地 test manifest 一致。" if dataset_meta else "未提供本地 test manifest，只核对 CSV 的标签及各 pair/seed 配对一致性。")
             + f"重算 AUROC/AUPR 与原 final 最大绝对差为 {max(metrics.auroc_delta.abs().max(), metrics.aupr_delta.abs().max()):.3g}。服务器 Git HEAD={manifest['git_head']}。配置核验：{config_note}。", "",
             f"六 datasets 的同一图像在不同 pair/seed 中顺序和标签相同，因此可以作逐图配对诊断；共 {metrics.n_images.sum():,} 条导出分数，并非这么多独立患者。", "",
             "## 主结果与不确定性", "",
             "mean/seed_sd 仍按三个独立训练 seeds 汇总 AUROC/AUPR，不融合三个 score，也不挑最优 seed。image bootstrap 按 normal/abnormal 分层抽样，同一次抽样对四 pair/三 seeds 共享，再平均各 seed 指标；domain/overall 在每次重采样中作 dataset macro。", "",
             f"使用 {args.bootstrap} 次抽样，随机种子 {args.seed}。95% percentile interval 条件于当前训练模型，假设图像独立；slice/patch 相关及缺少可靠 patient IDs 使它不能当作患者级 CI。seed SD 另列，两个不确定性来源未合并，不做患者显著性声明。", "",
             md(summary_df[summary_df.metric == "auroc"], ["dataset", "pair", "mean", "seed_sd", "image_bootstrap_lower", "image_bootstrap_upper"]), "",
             "### Domain / overall dataset macro", "",
             md(macro_summary[macro_summary.metric == "auroc"], ["scope", "pair", "mean", "seed_sd", "image_bootstrap_lower", "image_bootstrap_upper"]), "",
             "### 异常比例与 AUPR", "",
             "这里的 AUPR 是 average precision（非梯形积分 PR-AUC）。PR 图灰线为各 dataset 的异常比例；随机排序的总体 AP 基准为该比例，有限样本期望可能有偏差。高异常比例可产生很高 AP，因此不凭 AP 的绝对大小跨 dataset 比较。", "",
             md(samples, ["dataset", "n_images", "n_normal", "n_abnormal", "abnormal_fraction"]), "",
             md(summary_df[summary_df.metric == "aupr"], ["dataset", "pair", "mean", "seed_sd", "image_bootstrap_lower", "image_bootstrap_upper"]), "",
             "## Medical-pretraining effects", "",
             "配对图像抽样下的 AUROC 差值与 interaction 区间见 paired_pretraining_effect_intervals.csv。比较的是各自已选配置的系统表现；不是纯预训练数据的因果效应。", "",
             md(effects[effects.scope.isin(["radiology", "non_radiology", "overall"])], ["scope", "effect", "mean", "seed_sd", "image_bootstrap_lower", "image_bootstrap_upper"]), "",
             "## Score components 与反向排序", "",
             "主 score 仍是 contrast_topk。原始 patch 数为 256、top-k=13，前 13 个 discrepancy 不低于 median，因此 raw_topk 可以由已导出的 score+discrepancy_median 精确恢复（仅浮点舍入差）。恢复 raw/mean/median/max 的 AUROC 只用于解释信号组成；星号诊断不得替换主结果。", ""]
    for ds, p in chosen:
        lines += [f"### {ds} / {p}", "", md(diag_mean[(diag_mean.dataset == ds) & (diag_mean.pair == p)], ["diagnostic", "auroc"]), ""]
    lines += ["## OCT2017 异常类别", "",
              "从文件名前缀划分 CNV/DME/DRUSEN；每个子类都与相同 NORMAL 对照比较，不能当成三个独立数据集。", "",
              md(subtype_df.groupby(["pair", "subtype"], as_index=False).agg(mean=("auroc", "mean"), seed_sd=("auroc", "std")), ["pair", "subtype", "mean", "seed_sd"]), "",
              "## 跨 seed 稳定性与误例", "",
              "下表汇总全部图像在三个 seed 两两比较中的排序相关；完整 CSV 还含 normal/abnormal 分层相关。高 rho 不代表正向诊断能力，低 rho 提示模型间排序波动，不能通过挑选 seed 消除。", "",
              md(correlation_summary[correlation_summary.label == -1], ["dataset", "pair", "min_rho", "mean_rho", "max_rho"]), "",
              "cross_seed_rank_correlation.csv 给出所有图像及分 label 的 Spearman rho（无假定独立配置的 p 值）。error_case_candidates.csv 为每 dataset/pair/label 最难排序的 20 张图：对 abnormal 计算其高于 normal 的配对贡献，对 normal 计算 abnormal 高于它的贡献，再跨 seeds 汇总。不是使用 test 调出的分类阈值，也不是已确认的临床误诊。路径供服务器原图检查；此处没有原始图像，不能推断病灶或成像伪影。", "",
              "## 结论与下一步", "",
              "逐图重放已验证聚合结果。以下观察为 post-hoc 描述，不能反向改变 selection：", ""]
    for ds, p in chosen:
        component = diag_mean[(diag_mean.dataset == ds) & (diag_mean.pair == p)].set_index("diagnostic").auroc
        lines += [f"- {ds}/{p}：contrast AUROC={component['primary_contrast_topk']:.4f}，mean={component['mean']:.4f}，median={component['median']:.4f}；相对峰值与整体 discrepancy 产生不同排序。"]
    gm_subtypes = subtype_df[subtype_df.pair == "GM"]
    lines += [f"- OCT2017/GM：CNV/DME/DRUSEN 各 seed AUROC 范围为 {gm_subtypes.auroc.min():.4f}–{gm_subtypes.auroc.max():.4f}；"
              + ("九个组合均低于 0.5，反向排序并非仅由一个异常子类驱动。" if (gm_subtypes.auroc < .5).all() else "各子类/seed 的方向并不完全一致，不能宣称共同反向。"), "",
              "信号组成与异常标签之间的失配尚不能直接归因为 subtraction 的因果影响、模型 bug 或病灶形态。mean/median 的正向排序也可能含非病灶混杂因素；需要原图及局部 discrepancy 检查来进一步解释。当前产物可支持具体误例核查，无需再为分数导出重训。", "",
              "AOSS 与真实 AUROC 的相关性、同 screening 预算下的 selection regret 继续见 [聚合分析](../analysis_report.md)；本次使用原选点，不重新执行排序。", "",
              "保持本轮主配置/评分与 source-only selection。不根据这些 test 诊断换点、翻转 score 或选更好超参数；若后续要验证新的 scoring/training 假设，需要另行记录、用 source-only 证据决定方案，并保留当前实验作为原结果。", "",
              "现有选点对同一 pair 的五个 held-out folds 作全局平均 AOSS：每个 fold 的训练/AOSS 排除了其 held-out modality，但该 modality 会出现在其他 fold 的 source 数据中。因此全局选点可能间接使用其 normal-domain 信息，不能把当前实现表述为全局 selection 完全不接触该 target modality。此次分析如实披露这一边界，没有修改协议。", "",
              "所有分析结果与图表只保留本地；仅分析代码、测试和使用说明同步仓库。", ""]
    (out / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    if before != digest(input_paths):
        raise RuntimeError("Input artifacts changed during analysis")
    (out / "input_hashes.json").write_text(json.dumps(dict(inputs_unchanged=True, sha256=before, bootstrap=args.bootstrap, random_seed=args.seed,
        versions=dict(numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, matplotlib=matplotlib.__version__)), indent=2), encoding="utf-8")
    print(f"Complete: {out}; input artifacts unchanged", flush=True)


if __name__ == "__main__":
    main()
