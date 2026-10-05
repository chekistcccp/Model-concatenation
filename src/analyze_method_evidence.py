"""Reorganize returned v1 evidence around the parameter-composition method.

CPU-only, descriptive. Writes ignored reports/tables/figures; never selects a
configuration, changes a score, or trains. Run after paired-response analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yaml

from .followup_io import common_jobs
from .prediction_io import sha256, validate_final


PAIRS = {("medical", "medical"): "MM", ("medical", "general"): "MG",
         ("general", "medical"): "GM", ("general", "general"): "GG"}
GROUPS = dict(overall=["Brain", "liver", "RESC", "OCT2017", "RSNA", "camelyon16"],
              radiology=["Brain", "liver", "RSNA"], non_radiology=["RESC", "OCT2017", "camelyon16"])


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def verify_returned(directory, n_outputs):
    m = read(directory / "run_manifest.json")
    if not (m["status"] == "complete" and m["original_inputs_unchanged"] and m["no_reselection"]
            and len(m["output_sha256"]) == n_outputs):
        raise ValueError(f"Incomplete returned evidence: {directory}")
    for name, digest in m["output_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Returned evidence hash differs: {name}")
    return m


def macros(rows):
    frame = pd.DataFrame(rows)
    frame["pair"] = [PAIRS[s, t] for s, t in zip(frame.source_backbone, frame.target_backbone)]
    values = []
    for group, datasets in GROUPS.items():
        for (pair, seed), cell in frame[frame.dataset.isin(datasets)].groupby(["pair", "seed"]):
            for metric in ["image_auroc", "image_aupr", "pixel_auroc", "pixel_aupr"]:
                if metric not in cell:
                    continue
                take = cell.dropna(subset=[metric])
                if len(take):
                    values.append(dict(pair=pair, seed=seed, group=group, metric=metric,
                                       value=float(take[metric].mean()), datasets="|".join(sorted(take.dataset))))
    return pd.DataFrame(values)


def effects(frame):
    result = []
    for (group, metric, seed), cell in frame.groupby(["group", "metric", "seed"]):
        v = dict(zip(cell.pair, cell.value))
        if len(v) != 4: continue
        terms = dict(source_given_medical_target=v["MM"]-v["GM"], source_given_general_target=v["MG"]-v["GG"],
                     target_given_medical_source=v["MM"]-v["MG"], target_given_general_source=v["GM"]-v["GG"],
                     source_average=(v["MM"]-v["GM"]+v["MG"]-v["GG"])/2,
                     target_average=(v["MM"]-v["MG"]+v["GM"]-v["GG"])/2,
                     interaction=v["MM"]-v["GM"]-v["MG"]+v["GG"])
        result += [dict(group=group, metric=metric, seed=seed, effect=k, value=float(x)) for k, x in terms.items()]
    return pd.DataFrame(result)


def run(root, followup=None, output=None, as_of="2026-10-03"):
    result = root / "results"
    followup = followup if followup is not None else result / "transfer"
    out = output if output is not None else result / "analysis/method_review_20261003"
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((root / "configs/experiment.yaml").read_text(encoding="utf-8"))
    selected = read(result / "selected_configs.json")
    jobs = [j for j in common_jobs(cfg, selected) if j["reuse_original"]]
    inputs = [root / "configs/experiment.yaml", result / "selected_configs.json"]
    rows = []
    for job in jobs:
        p = result / "final" / f"{job['job']}.json"; inputs.append(p)
        r = read(p)
        validate_final(cfg, selected, r, job["source_backbone"], job["target_backbone"], job["source_stage"],
                       job["target_block"], job["seed"], "mlp", .05, ["contrast_topk"])
        rows += r["rows"]
    if len(rows) != 72: raise ValueError("Incomplete original main table")
    seed = macros(rows)
    seed.to_csv(out / "main_seed_macros.csv", index=False)
    summary = seed.groupby(["pair", "group", "metric", "datasets"]).value.agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(out / "main_summary.csv", index=False)
    effects(seed).to_csv(out / "locked_pretraining_effects.csv", index=False)
    for name, count in [("mechanism_audit",4), ("paired_response_audit",62), ("common_stitch",53)]:
        verify_returned(followup / name, count)
        inputs.append(followup / name / "run_manifest.json")
    common_rows = []
    for p in sorted((followup / "common_stitch").glob("*_seed*.json")):
        common_rows += [dict(row, target_block=read(p)["target_block"]) for row in read(p)["rows"]]
    if len(common_rows) != 144: raise ValueError("Incomplete common-position comparison")
    common_effects = pd.concat([effects(macros([r for r in common_rows if r["target_block"]==block])).assign(target_block=block)
                               for block in [3,9]], ignore_index=True)
    common_effects.to_csv(out / "common_position_effects.csv", index=False)
    selection = []
    for (source, target), pair in PAIRS.items():
        x, y, names = [], [], []
        for stage in [1,2,3]:
            for block in [3,6,9]:
                name=f"{source}_to_{target}_s{stage}_b{block}"
                a=result/"screen"/f"{name}.json"; b=result/"stitchmap"/f"{name}.json"
                inputs += [a,b]
                screen, metric = read(a), read(b)
                if screen["rows"] != [] or len(screen["aoss_rows"]) != 5: raise ValueError("Screen protocol differs")
                names.append(name); x.append(np.mean([r["aoss"] for r in screen["aoss_rows"]]))
                y.append(np.mean([r["image_auroc"] for r in metric["rows"] if r["score_mode"]=="contrast_topk"]))
        chosen=next(r["config"] for r in selected["selected"] if r["source_backbone"]==source and r["target_backbone"]==target)
        index=names.index(chosen)
        selection.append(dict(pair=pair, spearman=float(spearmanr(x,y).statistic),
                              selected_config=chosen, selected_auroc=y[index], oracle_upper_bound=max(y),
                              regret=max(y)-y[index], uniform_random_expectation=float(np.mean(y)),
                              selected_minus_uniform=y[index]-np.mean(y)))
    pd.DataFrame(selection).to_csv(out / "original_selection_evidence.csv", index=False)
    align = pd.read_csv(followup / "mechanism_audit/alignment.csv")
    align["pair"] = align.job.str.extract(r"^(medical_to_medical|medical_to_general|general_to_medical|general_to_general)")[0].map({f"{s}_to_{t}":p for (s,t),p in PAIRS.items()})
    align["seed"] = align.job.str.extract(r"seed(\d+)$")[0].astype(int)
    a = align.groupby(["pair","seed","held_out","source_modality","control"]).nffa_loss.mean().reset_index()
    a = a.groupby(["pair","seed","held_out","control"]).nffa_loss.mean().reset_index()
    a = a.groupby(["pair","seed","control"]).nffa_loss.mean().reset_index()
    a.to_csv(out / "normal_alignment_seed.csv", index=False)
    response_path = result / "analysis/paired_response_audit/all_seed.csv"
    inputs.append(response_path)
    response = pd.read_csv(response_path)
    response.to_csv(out / "source_response_seed.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(12,3.8),layout="constrained")
    pairs=["MM","MG","GM","GG"]
    for ax,metric in zip(axes[:2],["image_auroc","pixel_auroc"]):
        take=[summary[(summary.pair==p)&(summary.group=="overall")&(summary.metric==metric)].iloc[0] for p in pairs]
        ax.errorbar(range(4),[r["mean"] for r in take],yerr=[r["std"] for r in take],fmt="o",capsize=4)
        ax.set_xticks(range(4),pairs);ax.set_ylabel(metric);ax.axhline(.5,color="gray",ls="--",linewidth=1)
        ax.set_title("6 datasets" if metric=="image_auroc" else "Masked dataset subset")
    take=response.groupby("pair").net_local_response.agg(["mean","std"])
    axes[2].errorbar(range(4),take.loc[pairs,"mean"],yerr=take.loc[pairs,"std"],fmt="o",capsize=4)
    axes[2].set_xticks(range(4),pairs);axes[2].set_ylabel("Source synthetic net response");axes[2].set_title("Mechanism evidence; not clinical AUROC")
    fig.suptitle("Parameter-stitching method v1: mean ± SD of 3 seeds",fontsize=12)
    fig.savefig(out/"method_evidence.png",dpi=180);fig.savefig(out/"method_evidence.pdf");plt.close(fig)
    lines=["# 参数拼接零样本异常检测：按方法主线重整", "", f"{as_of}。核心目标是设计参数模块拼接的新模型；医学预训练矩阵为方法验证。",
           "", "## 方法与目前证据", "", "F=T_suffix∘A_phi∘C_prefix；继承参数冻结，仅正常 source 学习 phi。完整 detector 仍需 frozen reference R，原 contrast_topk 不变。",
           "", "|Pair|六 dataset image AUROC ± seed SD|有 mask 子集 pixel AUROC ± seed SD|", "|---|---:|---:|"]
    for p in pairs:
        take=summary[(summary.pair==p)&(summary.group=="overall")].set_index("metric")
        lines.append(f"|{p}|{take.loc['image_auroc','mean']:.4f} ± {take.loc['image_auroc','std']:.4f}|{take.loc['pixel_auroc','mean']:.4f} ± {take.loc['pixel_auroc','std']:.4f}|")
    lines += ["", "原六 dataset image discrimination 较弱；较高 pixel AUROC 只覆盖有 mask 子集且受背景不平衡影响。两者不能互相替代，不据已看 test 改评分。",
              "正常 matched 对齐优于 shifted-image、冻结尾部接口与梯度路径及 source 配对净响应已有证据；这些验证方法可运行及输入依赖，不能单独证明真实病变泛化。",
              "", "## 选择环节：原 screening 预算", "", "|Pair|九点 Spearman|原选点 regret|原选点−均匀随机期望|", "|---|---:|---:|---:|"]
    for r in selection: lines.append(f"|{r['pair']}|{r['spearman']:.3f}|{r['regret']:.4f}|{r['selected_minus_uniform']:.4f}|")
    lines += ["", "oracle 只作事后上界；没有改选。screen 为 4 epochs/500，final 为 8 epochs/1000，不能混算。global 五折 AOSS 的跨 fold 信息边界仍需披露。",
              "", "## supporting analyses", "", "locked_pretraining_effects.csv 包含 source/target 条件效应、平均效应和 interaction；其 GG cut 与其他 pair 不同。common_position_effects.csv 在原 s2b3/s2b9 同点复算，radiology/non-radiology 均保留。",
              "这些结果说明参数来源/模态/位置会影响方法，不能替代方法增量或归因为所有医学预训练。source 响应分层与目标域性能分层不同。",
              "", "## 下一轮与发表缺口", "", "首要是完整模型实体及 inherited suffix 的同预算必要性：固定 original/matched_tail/no_tail/untrained_adapter，全部训练完成后统一评估。见仓库 docs/METHOD_PLAN_CN.md，运行 bash run_method_controls.sh。",
              "随后补强公平 target-free 基线、固定 map 的 AUPRO/AP、独立来源及 patient/slide 分组统计、完整 detector 效率。若普通 no_tail 重建同样有效，需发展新的方法机制，不能继续堆积医学效应或代理诊断作为算法贡献。",
              "正常流形瓶颈/结构约束与 source-only 合成正常化目标仅为待固定的研究方向；本轮未新增 loss，未把 AOSS 样本用于训练。已看 BMAD test 不能反复用于研发搜索，新的确认性数据需独立保留。",
              "", "2026 正式近邻及来源：仓库 docs/PUBLICATION_GAP_PLAN_2026_CN.md。尤其需对照 CVPR Revisiting Model Stitching 的最终特征匹配与 self-stitch 控制，以及 VisualAD/AnomalyVFM/PDD、FoundAD 等异常检测方法。",
              "拼接、轻量接口和残差本身已有近邻；需显示模块组合的实质增量。few-shot/目标正常训练结果不混入零样本主表，没有统一发表 AUROC 门槛，也不宣称首次。",
              "", "## 完整性及限制", "", "重验原 12 jobs/72 rows；mechanism 4、paired 62、common 53 个返回输出 hash。paired 73728 行/300 replay 与既有 CPU audit 一致。",
              "本地没有服务器完整 cache/权重，不能替服务器宣称所有原输入字节一致；大数组/旧 target 权重仍为 stat 保护。患者标识不足，seed SD 不是患者 CI。本报告仅汇总历史 v1 证据，新增 GPU 方法对照须另用 analyze_method_controls 验证，不能从本报告判断服务器是否跑过。",
              "本报告、表格、图和 hash 都保持忽略，不同步仓库。原主结果、选择、预算和评分没有改变。"]
    (out/"report_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/"input_hashes.json").write_text(json.dumps({str(p):sha256(p) for p in inputs},indent=2),encoding="utf-8")
    print(out/"report_CN.md")


if __name__ == "__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("."))
    ap.add_argument("--followup",type=Path,help="Returned follow-up directory; defaults to results/transfer")
    ap.add_argument("--output",type=Path,help="Independent analysis destination")
    ap.add_argument("--as-of",default="2026-10-03",help="Report date supplied by the analyst")
    args=ap.parse_args();root=args.root.resolve()
    run(root, (root/args.followup).resolve() if args.followup else None,
        (root/args.output).resolve() if args.output else None, args.as_of)
