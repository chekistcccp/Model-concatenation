# 已锁定 final 的逐图分数导出

下一步仅补充诊断数据：用现有 final checkpoint 重新评估，导出逐图分数。
四个 MM/MG/GM/GG pair、锁定 stitch 点、三 seeds、224 输入、contrast_topk 和
topk_fraction=0.05 均沿用原实验。训练、AOSS、全局选点及候选网格没有改动。

## 服务器运行

在原实验仓库根目录，激活原来的 Python 环境后：

```bash
git pull --ff-only
python -m src.export_predictions --config configs/experiment.yaml --check-only
GPUS=0,1,2,3 bash run.sh predictions
```

单 GPU 可用 `GPUS=0 bash run.sh predictions`。检查失败时补齐所指明的原始
文件；此入口不会重新生成 manifest/cache、下载模型或调用训练/selection。
不应为补分数运行 `bash run.sh all` 或 `final`。

前置文件：

- `results/selected_configs.json`，保持原来的 source_only_AOSS Top-1。
- `results/final/<config>_seed<seed>.json`，共 12 个原始 final job。
- `results/final/checkpoints/<config>_seed<seed>/<target_modality>.pt`，每 job 五个。
- `cache/features/<dataset>/test/` 的 `.done`、records、所选 source stage 和 target 数组。
- 原有两个 Transformer 的本地权重与 config；导出使用 full target 与原 adapter/tail。

## 产物与一致性检查

仅写 `results/predictions/`（不提交到 Git）：

```text
results/predictions/
├── export_manifest.json
└── <config>_seed<seed>/
    ├── Brain.csv
    ├── liver.csv
    ├── RESC.csv
    ├── OCT2017.csv
    ├── RSNA.csv
    ├── camelyon16.csv
    └── evaluation.json
```

总计 72 个逐图 CSV、12 个评估 JSON 和一份导出清单。CSV 保留缓存记录顺序与
浮点精度，含 image_path、label、score、dataset、modality、pair、seed、config、
score_mode、topk_fraction，以及 discrepancy mean/median/max。后者仅用于诊断，
不改变主评分。`record_index` 对应当前 test cache records 行号。

每个 checkpoint 校验 source/target、stage/block、adapter、held-out modality，
adapter state 严格加载。legacy checkpoint 未存 seed，因此 seed 依据原 final JSON
及 checkpoint 目录核对，不能声称由 checkpoint 内容独立验证。

CSV 的分数就是原 evaluate_dataset 的 contrast_topk；图像指标与原 final JSON
逐 dataset 对照，AUROC/AUPR 绝对差超过 1e-6 时该 job 报错，保留诊断输出。
该阈值只判定重放一致性，不用于选点或调参。导出不重复计算 pixel 指标，既有
pixel 结果保留在原 final 中。若出现差异，先查 checkpoint/cache/model 版本及
数值环境，不覆盖原始结果，也不放宽阈值来声称重放成功。

export_manifest.status 必须为 complete，且 original_inputs_unchanged=true；
每 job 的 replay_matches_original 也必须为 true。失败输出不能作为完整复现。
重新运行会重放所有 jobs；不会仅凭旧 CSV 存在就跳过检查。

导出记录 Git HEAD、库版本、checkpoint/records/CSV 哈希，以及配置、锁定文件和
原 final 的哈希。这些证明本次重放输入，不能追溯补造原训练时缺失的来源记录。

## 取回后分析与研究设计边界

取回整个 `results/predictions/` 目录即可进行固定模型的 score 分布、ROC/PR、
误例排序和跨 seed 比较。缺少可靠 patient IDs 时，不能把逐图（slice/patch）
bootstrap 说成患者级置信区间。

CPU 分析命令（无需模型权重或 feature cache）：

```bash
pip install -r requirements-analysis.txt
python -m src.analyze_predictions --root . --bootstrap 1000
```

除 predictions 外还需保留原 `results/final/*.json`、`selected_configs.json` 与
实验配置；已有 `cache/manifest.jsonl` 时逐图核对标签。入口先检查 receipt、
哈希和原指标，再输出 `results/analysis/predictions/analysis_report.md` 及
`results/figures/predictions/` 下的 PDF/PNG。配置跨系统仅有换行差异时，核对
服务器 Git 版本中的原始字节及解析后的 YAML 内容，不能跳过内容一致性检查。

报告分别列出训练 seed SD、固定模型下配对图像 bootstrap 区间、dataset macro、
异常比例与 AUPR、score components、OCT 子类、跨 seed 秩相关和误例候选。
bootstrap 对所有 pair/seed 使用相同图像抽样，不把 seeds 当成新增患者。
诊断用 raw top-k 可从当前 contrast score 加 patch median 恢复；仅适用于当前
top-k 不超过一半 patch 数的条件，不能据此改变评分。所有输出保持在被忽略的
results/ 中，不上传报告、预测或图表。

所有结果均属 post-hoc 诊断，不能用 target test 改选 stitch、调整 top-k、翻转
score、挑选 seed 或把最好的校准量替换主结果。现有全局跨-fold AOSS 平均的
声明边界继续如实保留；本次导出不修改 selection protocol。后续若需修订该
协议，须另记为新实验，不能覆盖当前研究结果。

## 本地验证

```bash
python -m unittest discover -s tests
python -m compileall -q src tests
```

测试包含同一路径上开启/关闭导出的图像指标相等、分数及行序保真、错误
checkpoint/fold、评分漂移、缺失 checkpoint 无重训 fallback 等。本地 synthetic
CPU 测试不代替服务器真实 GPU/权重集成验证。
