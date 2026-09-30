"""Further read-only audit of returned server manifests and aggregate results.

python -m src.audit_followup --root .
Filename overlaps are candidates, NOT verified content duplicates or patient IDs.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from .analyze_results import digest, md, pair


def subset(frame, limit, seed):
    if limit <= 0 or len(frame) <= limit:
        return frame.copy()
    return frame.iloc[np.sort(np.random.default_rng(seed).choice(len(frame), limit, replace=False))].copy()


def overlap_table(frame, key):
    rows = []
    for ds, group in frame.groupby("dataset"):
        for a, b in itertools.combinations(["train", "valid", "test"], 2):
            left, right = group[group.split == a], group[group.split == b]
            shared = set(left[key].dropna()) & set(right[key].dropna())
            rows.append(dict(dataset=ds, split_a=a, split_b=b, key=key, shared_keys=len(shared),
                             rows_a=int(left[key].isin(shared).sum()), rows_b=int(right[key].isin(shared).sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    root = ap.parse_args().root.resolve()
    cfg = yaml.safe_load((root / "configs/experiment.yaml").read_text(encoding="utf-8"))
    cache, results = root / cfg["paths"]["cache_dir"], root / cfg["paths"]["results_dir"]
    out = results / "analysis"
    out.mkdir(exist_ok=True, parents=True)
    paths = [cache / "manifest.jsonl", cache / "data_audit.json", cache / "model_audit.json", root / "configs/experiment.yaml"]
    paths += sorted((root / "src").glob("*.py"))
    paths += [p for p in results.rglob("*.json") if out not in p.parents]
    before = digest(paths)
    manifest = pd.read_json(cache / "manifest.jsonl", lines=True)
    audit = json.loads((cache / "data_audit.json").read_text(encoding="utf-8"))
    models = json.loads((cache / "model_audit.json").read_text(encoding="utf-8"))
    manifest["filename"] = manifest.image.map(lambda x: PurePosixPath(x).name)
    checks = []

    def save(name, frame):
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")

    def check(name, ok, note):
        checks.append(dict(check=name, status="PASS" if ok else "FAIL", note=note))

    check("unique_image_paths", not manifest.image.duplicated().any(), f"{len(manifest)} manifest rows")
    check("dataset_coverage", set(manifest.dataset) == set(cfg["data"]["datasets"]), "six datasets")
    check("normal_only_train", manifest.loc[manifest.split == "train", "label"].eq(0).all(), "all manifest training rows")
    counts = []
    for ds, group in manifest.groupby("dataset"):
        for split, g in group.groupby("split"):
            counts.append(dict(dataset=ds, split=split, normal=int(g.label.eq(0).sum()), abnormal=int(g.label.eq(1).sum()), with_mask=int(g['mask'].notna().sum()), normal_with_mask=int((g.label.eq(0) & g['mask'].notna()).sum()), abnormal_with_mask=int((g.label.eq(1) & g['mask'].notna()).sum())))
            values = counts[-1]
            check(f"counts:{ds}:{split}", all(values[k] == audit[ds]["splits"][split][k] for k in ("normal", "abnormal", "with_mask")), "manifest vs data_audit exact")
    count_df = pd.DataFrame(counts)
    save("manifest_counts", count_df)
    normal, abnormal = {"good", "normal", "healthy"}, {"ungood", "bad", "anomaly", "anomalous", "abnormal", "disease", "diseased"}
    label_errors, mask_errors = [], []
    for row in manifest.itertuples(index=False):
        path = PurePosixPath(row.image)
        parts = {s.lower() for s in path.parts}
        label = 1 if parts & abnormal else (0 if parts & normal else None)
        if label != row.label or (parts & abnormal and parts & normal):
            label_errors.append(dict(image=row.image, stored_label=row.label, path_label=label))
        expected_root = PurePosixPath(audit[row.dataset]["root"])
        if not path.is_relative_to(expected_root) or row.modality != cfg["data"]["modality_map"][row.dataset]:
            label_errors.append(dict(image=row.image, stored_label=row.label, path_label="root_or_modality_mismatch"))
        if pd.notna(row.mask):
            mask = PurePosixPath(row.mask)
            # Dataset/split/class branch must agree; only img/label component differs.
            if mask.stem != path.stem or mask.parent.parent != path.parent.parent or not mask.is_relative_to(expected_root):
                mask_errors.append(dict(image=row.image, mask=row.mask))
    check("path_label_and_root_consistency", not label_errors, "path semantics only; no pixel inspection")
    check("mask_stem_and_branch_consistency", not mask_errors, "path semantics only; no mask content inspection")
    masked = manifest[manifest['mask'].notna()]
    check("mask_not_shared_by_different_images", not masked.groupby("mask").image.nunique().gt(1).any(), "exact mask paths")
    save("manifest_label_errors", pd.DataFrame(label_errors, columns=["image", "stored_label", "path_label"]))
    save("manifest_mask_errors", pd.DataFrame(mask_errors, columns=["image", "mask"]))
    oct_rows = manifest[manifest.dataset == "OCT2017"].copy()
    oct_rows["filename_class"] = oct_rows.filename.str.split("-").str[0]
    check("OCT_filename_class_label", ((oct_rows.filename_class == "NORMAL") == (oct_rows.label == 0)).all(), "CNV/DME/DRUSEN are abnormal; NORMAL is normal")
    save("oct_class_counts", oct_rows.groupby(["split", "filename_class", "label"]).size().rename("count").reset_index())
    overlaps = overlap_table(manifest, "filename")
    save("split_filename_overlaps", overlaps)
    candidates = []
    for ds, group in manifest.groupby("dataset"):
        shared = group.groupby("filename").split.nunique()
        shared = shared[shared > 1].index
        candidates.append(group[group.filename.isin(shared)])
    save("filename_overlap_candidates", pd.concat(candidates, ignore_index=True))
    # These are filename tokens, deliberately not asserted to be patient identifiers.
    token_rows = manifest[manifest.dataset.isin(["Brain", "OCT2017"])].copy()
    token_rows["candidate_subject_token"] = token_rows.apply(lambda r: PurePosixPath(r.image).stem.split("_")[0] if r.dataset == "Brain" else PurePosixPath(r.image).stem.split("-")[1], axis=1)
    token_overlaps = overlap_table(token_rows, "candidate_subject_token")
    save("candidate_subject_token_overlaps", token_overlaps)
    # Reproduce current code's sampling from manifest order, without loading torch.
    # Actual cache records are absent, so this is conditional reconstruction only.
    samples, train_cache = [], {}
    for ds in cfg["data"]["datasets"]:
        normal_rows = manifest[(manifest.dataset == ds) & (manifest.split == "train") & (manifest.label == 0)]
        train_cache[ds] = subset(normal_rows, cfg["data"]["cache_train_limit_per_dataset"], cfg["project"]["seed"])
        perturb = subset(normal_rows, cfg["data"]["perturb_samples_per_dataset"], cfg["project"]["seed"] + 17)
        test_names = set(manifest[(manifest.dataset == ds) & (manifest.split == "test")].filename)
        for kind, part in [("feature_cache", train_cache[ds]), ("perturb_cache", perturb)]:
            samples.append(dict(dataset=ds, kind=kind, seed=cfg["project"]["seed"], n=len(part), test_filename_matches=int(part.filename.isin(test_names).sum())))
    sample_df = pd.DataFrame(samples)
    save("reconstructed_cache_overlap", sample_df)
    groups = {}
    for ds in cfg["data"]["datasets"]:
        groups.setdefault(cfg["data"]["modality_map"][ds], []).append(ds)
    source_rows = []
    for phase in ["screen", "final"]:
        n = cfg[phase]["n_per_source_modality"]
        for seed in ([cfg[phase]["seed"]] if phase == "screen" else cfg[phase]["seeds"]):
            for mod, datasets in groups.items():
                portions = [subset(train_cache[ds], min(int(np.ceil(n / len(datasets))), len(train_cache[ds])), seed + 101 * i + len(mod)) for i, ds in enumerate(datasets)]
                actual = pd.concat(portions).iloc[:n]
                for ds, part in actual.groupby("dataset"):
                    test_names = set(manifest[(manifest.dataset == ds) & (manifest.split == "test")].filename)
                    source_rows.append(dict(phase=phase, source_modality=mod, dataset=ds, seed=seed, n=len(part), test_filename_matches=int(part.filename.isin(test_names).sum()), excluded_when_target_modality=mod))
    source_df = pd.DataFrame(source_rows)
    save("reconstructed_training_overlap", source_df)
    model_rows = []
    for group in ["sources", "targets"]:
        for name, obj in models[group].items():
            model_rows.append(dict(role=group, backbone=name, ready=obj["ready"], repo_id=obj.get("repo_id", "manual RadImageNet"), evidence="file existence/size and target config; not weight hash or load report"))
    save("model_evidence", pd.DataFrame(model_rows))
    check("models_marked_ready", all(r["ready"] for r in model_rows), "readiness only")
    save("followup_integrity_checks", pd.DataFrame(checks))
    if any(r["status"] == "FAIL" for r in checks):
        raise ValueError("Manifest consistency failed; see followup_integrity_checks.csv")

    final = pd.DataFrame([r for p in sorted((results / "final").glob("*.json")) for r in json.loads(p.read_text(encoding="utf-8"))["rows"]])
    final["pair"] = final.apply(pair, axis=1)
    final_mean = final.groupby(["pair", "dataset"], as_index=False).image_auroc.mean().rename(columns={"image_auroc": "final_auroc"})
    base = pd.DataFrame(json.loads((results / "baselines.json").read_text(encoding="utf-8"))["rows"])
    base["pair"] = base.apply(pair, axis=1)
    comp = final_mean.merge(base, on=["pair", "dataset"])
    comp["delta"] = comp.final_auroc - comp.image_auroc
    comp_summary = comp.groupby(["pair", "baseline"], as_index=False).agg(final_auroc=("final_auroc", "mean"), baseline_auroc=("image_auroc", "mean"), delta=("delta", "mean"), datasets_improved=("delta", lambda x: int((x > 0).sum())))
    save("baseline_macro_diagnostic", comp_summary)
    ablation_rows = []
    for p in sorted((results / "ablation").glob("*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        for r in obj["rows"]:
            ablation_rows.append(dict(r, file=p.name))
    ab = pd.DataFrame(ablation_rows)
    score_wide = ab.pivot(index=["n_per_source_modality", "adapter", "dataset"], columns="score_mode", values="image_auroc").reset_index()
    score_wide["contrast_minus_raw"] = score_wide.contrast_topk - score_wide.raw_topk
    save("score_mode_diagnostic", score_wide)
    budget = ab[ab.score_mode == "contrast_topk"].groupby(["n_per_source_modality", "adapter"], as_index=False).agg(auroc=("image_auroc", "mean"), aoss=("aoss", "mean"), train_loss=("final_train_loss", "mean"))
    # AOSS/loss use five folds equally; OCT datasets must not count twice.
    folds = ab[ab.score_mode == "contrast_topk"].drop_duplicates(["file", "target_modality"])
    fold_means = folds.groupby(["n_per_source_modality", "adapter"], as_index=False)[["aoss", "final_train_loss"]].mean()
    budget = budget.drop(columns=["aoss", "train_loss"]).merge(fold_means, on=["n_per_source_modality", "adapter"])
    save("calibration_budget_diagnostic", budget)
    figs = results / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    mlp = budget[budget.adapter == "mlp"]
    for ax, metric, label in zip(axes, ["final_train_loss", "aoss", "auroc"], ["Alignment loss (five-fold mean)", "AOSS (five-fold mean)", "Image AUROC (six-dataset macro)"]):
        ax.plot(mlp.n_per_source_modality, mlp[metric], "o-")
        ax.set(xlabel="Normal samples per source modality", ylabel=label, xticks=[250, 500, 1000, 2000])
        ax.grid(alpha=.2)
    fig.suptitle("MM, MLP, seed 11, 6 epochs: post-hoc diagnostic")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(figs / f"alignment_aoss_auroc_diagnostic.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    gm_oct = pd.DataFrame([r for p in sorted((results / "stitchmap").glob("general_to_medical_*.json")) for r in json.loads(p.read_text(encoding="utf-8"))["rows"] if r["dataset"] == "OCT2017"])
    save("gm_oct_stitch_diagnostic", gm_oct)
    lines = ["# 服务器审计文件补充分析", "",
             "新增 data_audit.json、model_audit.json、manifest.jsonl；未收到独立服务器 Git HEAD、pip freeze、preflight_report 或 cache records/checkpoint。未修改数据、指标方向、训练或选择协议。", "",
             "## 已核实的结构事实", "",
             f"manifest 共 {len(manifest):,} 条，无重复完整 image 路径；全部 split 的 normal/abnormal/mask 计数与 data_audit 精确一致。train 全为 normal；目录标签、modality、dataset root 一致；mask 与图像同 stem 且同 dataset/split/class 分支，无同一路径 mask 被多图复用。以上不能代替图像/标注内容检查。",
             "四模型均记录 READY。依据 model_status 实现，这证明服务器当时找到了权重文件（及 target config），不证明权重语义、哈希、完整 load 结果或实际运行版本。此前“model audit 缺失”已解除，但独立权重核验仍未完成。", "",
             md(count_df, ["dataset", "split", "normal", "abnormal", "normal_with_mask", "abnormal_with_mask"]), "",
             "## 新发现：跨 split 文件名重合", "",
             md(overlaps[overlaps.shared_keys > 0], ["dataset", "split_a", "split_b", "shared_keys", "rows_a", "rows_b"]), "",
             "OCT2017：213/242=88.02% 正常 test 图文件名与 normal train 相同，另有 5 个 train/valid 同名。Brain：10 个异常图文件名在 valid/test 重合，未发现 train/test 同名。camelyon16 使用 0.png 等局部编号，跨目录同名不足以推断重复。所有候选保存在 filename_overlap_candidates.csv；须对服务器真实文件作 SHA256/像素比对才可确认内容相同。",
             "OCT 文件名前缀核验：NORMAL 均为 label=0，CNV/DME/DRUSEN 均为 label=1；test 三个异常类别各 242 张。未发现 OCT 标签在 manifest 层面整体反转，因此不能把 GM/OCT 低 AUROC 简单归因为目录标签反了。",
             "Brain/OCT 的候选 subject token 重合另存 candidate_subject_token_overlaps.csv。这只是文件名启发式，不是已确认患者身份或患者泄漏证据。", "",
             "## 对当前协议的实际影响", "",
             "held-out OCT fold 的 train_adapter/compute_aoss 同时排除 RESC 和 OCT2017，因此上述 train/test 同名不能直接解释为 OCT adapter 用 OCT train 训练。其他 folds 使用 OCT source，而全局选择平均五 fold AOSS，仍存在前轮指出的跨 fold 选择依赖。valid 不参与当前训练/选择，因此 Brain 的 valid/test 重合目前不等于直接使用验证集选点。",
             "若同名文件经哈希确认内容相同，且运行时 cache records 与下面重建一致，则全局 AOSS 可能经其他 folds 的 source 数据间接接触 OCT test 图像内容，即使代码从未为 selection 打开 test 路径。因此在哈希及 cache 核实前，必须限定“target-test 未用于选点”的声明为代码路径层面的观察。",
             "以下按当前本地代码、manifest 顺序和配置重建采样；缺少服务器 cache records 与版本证据，不能声称是运行时采样记录。文件名匹配亦不等于内容重复。", "",
             md(sample_df[sample_df.dataset == "OCT2017"], ["dataset", "kind", "n", "test_filename_matches"]), "",
             md(source_df[source_df.dataset == "OCT2017"], ["phase", "seed", "n", "test_filename_matches", "excluded_when_target_modality"]), "",
             "## 异常结果定位", "",
             "固定 final 与原有 sanity baselines 的 dataset macro 比较（无重新选点；baseline 本身不依赖 adapter seed）：", "",
             md(comp_summary, ["pair", "baseline", "final_auroc", "baseline_auroc", "delta", "datasets_improved"]), "",
             "GM/OCT2017 final AUROC=0.2949，而 general source dispersion=0.7166、medical target dispersion=0.7044；相同测试集的两个独立表示都能提供正向信号。问题更值得沿 stitched disagreement、选点和训练过程诊断，而不是直接断言数据集不可分。", "",
             "同一 screening 预算的 GM/OCT2017 在 s1b9 达到 0.6666，锁定 s2b3 为 0.2796；这是明确的配置敏感性，但不能据 target test 替换已选点。四 pair 的 final macro 均低于各自 source dispersion；当前结果尚未建立 stitching 相对这些简单 baseline 的整体优势。", "",
             md(gm_oct, ["source_stage", "target_block", "image_auroc", "image_aupr"]), "",
             "以下 MM ablation 固定 seed=11、epochs=6：训练损失与 AOSS 按五 modality 平均，AUROC 按六 dataset macro。数据量变化会同时改变优化步数，不能当作干净的单因素样本量因果实验。", "",
             md(budget, ["n_per_source_modality", "adapter", "auroc", "aoss", "final_train_loss"]), "",
             "500 normal samples 的较好 test AUROC 只能作 post-hoc 诊断，不能因此把主配置改选为 500；更低 alignment loss/更高 synthetic AOSS 不保证真实 AUROC 更高。score_mode_diagnostic.csv 给出同一次 adapter 的 raw/contrast/robust 对比；不据此翻转 score 或重选评分方式。", "",
             "## 下一步的最小补证", "",
             "当前不需要重新跑训练矩阵。服务器运行 `python -m src.verify_manifest_duplicates`，返回 results/analysis/server_duplicate_checks.json；该脚本只读候选图片并计算文件/解码像素哈希，不训练、不选点。另归档 Git HEAD、pip freeze，以及 cache/features/OCT2017/train/records.jsonl、cache/perturb/OCT2017/records.jsonl。其次固定已有 final checkpoint 导出 per-image label/score，核查 GM/OCT2017 与 GM/RESC 的 score 分布和误例，并检查 mask 像素值。若未来要求完全隔离 target modality 的选择，需要单独记录协议修订，不能把新结果覆盖到这轮。", ""]
    (out / "followup_report.md").write_text("\n".join(lines), encoding="utf-8")
    assert before == digest(paths), "Inputs changed during audit"
    (out / "followup_input_hashes.json").write_text(json.dumps({"unchanged": True, "sha256": before}, indent=2), encoding="utf-8")
    print(f"Follow-up audit complete: {len(checks)} checks; {len(manifest)} records; raw inputs unchanged")
    print(sample_df[sample_df.dataset == "OCT2017"].to_string(index=False))
    print(source_df[source_df.dataset == "OCT2017"].to_string(index=False))
    print(budget.to_string(index=False))


if __name__ == "__main__":
    main()
