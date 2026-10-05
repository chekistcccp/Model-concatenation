# MedStitch-ZS / Model-concatenation

研究主线：以预训练 CNN/Transformer 的参数模块拼接形成新模型，实现零样本异常检测。
医学预训练矩阵用于验证和解释方法。最新固定方法对照与服务器一键入口见
[docs/METHOD_PLAN_CN.md](docs/METHOD_PLAN_CN.md)：`GPUS=0,1,2,3 bash run_method_controls.sh`。
原 AOSS、选点、预算和评分保留；具体分析结果与模型权重不提交 Git。

当前下一步在成功 paired-response 的环境运行：

```bash
git pull --ff-only
GPUS=0,1,2,3 bash run_method_controls.sh
```

已有完整方法产物时，此入口验证后只重新打包；失败产物保留并停止。下载终端
`[TRANSFER]` 指向的 method_controls_run_*.tar.gz，权重留服务器。不能只根据
followup_results.tar.gz 的包名判断其含有当前方法对照证据。返回前可用
`python -m src.audit_return_package --archive <包路径> --require-stage method_controls --output results/analysis/method_return_audit.json`
检查阶段和哈希，再用 analyze_method_controls 核验科学协议与指标。

当前使用方式已经简化为：

> **你只下载两个 Google Drive 原始压缩包/压缩包组，分别直接放入 `data/` 和 `model/`。不要解压、不要改名。其余模型自动通过 ModelScope 下载。**

完整说明：

**[PREPARE_EXPERIMENT_CN.md](PREPARE_EXPERIMENT_CN.md)**

后续使用 Codex 继续处理实验与结果时，请优先阅读：

**[docs/CODEX_HANDOFF_CN.md](docs/CODEX_HANDOFF_CN.md)**

## 你手工准备什么

### data/

从 BMAD 官方 Google Drive 下载整理好的数据。Google Drive 给你的一个或多个 zip，全部直接放进：

```text
data/
```

例如：

```text
data/
├── BMAD-001.zip
├── BMAD-002.zip
└── ...
```

无需解压。

### model/

从 RadImageNet 官方 Google Drive 下载 PyTorch pretrained models 原始压缩包，直接放进：

```text
model/
```

例如：

```text
model/
└── <原始 RadImageNet PyTorch 压缩包>.zip
```

无需解压，也无需寻找 ResNet50。

## 程序自动完成什么

数据：

```text
data/*.zip
→ data/_extracted/
→ 自动识别 Brain/liver/RESC/OCT2017/RSNA/camelyon16
```

RadImageNet：

```text
model/*.zip
→ model/_manual_extracted/
→ 自动查找 resnet50_torch.pt / ResNet50 .pt/.pth
→ model/radimagenet_resnet50/resnet50_torch.pt
```

其他模型自动通过 ModelScope：

```text
timm/resnet50.a1_in1k
microsoft/rad-dino
facebook/dinov2-base
```

## 一次性运行

安装依赖：

```bash
pip install -r requirements.txt
```

4×3090：

```bash
GPUS=0,1,2,3 bash run.sh
```

单卡：

```bash
GPUS=0 bash run.sh
```

完整流程：

```text
raw BMAD archive
      ↓
auto extract + discover
      ↓
raw RadImageNet archive
      ↓
auto extract + locate ResNet50
      ↓
ModelScope download remaining 3 models
      ↓
shared FP16 feature cache
      ↓
36 AOSS screening jobs
      ↓
4 pair Top-1 × 3 seeds
      ↓
post-hoc stitch map + ablation
      ↓
REPORT.md
```

## 第一次推荐检查

```bash
bash run.sh prepare
```

然后：

```bash
bash run.sh models
GPUS=0 bash run.sh cache
```

确认正常后再：

```bash
GPUS=0,1,2,3 bash run.sh
```

## 当前 2×2 模型设计

| CNN | Transformer | 组合 |
|---|---|---|
| RadImageNet-ResNet50 | RAD-DINO | MM |
| RadImageNet-ResNet50 | DINOv2-Base | MG |
| ImageNet-ResNet50 | RAD-DINO | GM |
| ImageNet-ResNet50 | DINOv2-Base | GG |

详细下载地址和压缩包处理逻辑请看 **PREPARE_EXPERIMENT_CN.md**。

## 已锁定实验的只读结果分析

在仓库根目录运行（不需要 GPU，不调用 selection 或训练）：

```bash
pip install -r requirements-analysis.txt
python -m src.analyze_results --root .
python -m unittest discover -s tests -p "test_analysis.py"
```

读取原始 JSON 与配置，先检查完整性，再生成 `results/analysis/analysis_report.md`、
CSV 和 `results/figures/` PDF/PNG。缺失或无效核心结果会中止统计；原始结果和
`selected_configs.json` 不改写，SHA256 留档。数据审计缺失时尝试读取 `run.log`
开头的审计 JSON，并标明来源；模型权重验证不以下载日志替代。

统计使用每 seed 内的 dataset macro 和跨 seed 的 sample SD；regret 仅比较同一
screening 预算。无病例级 scores 时不生成患者 CI。报告明确披露当前跨 fold
平均 AOSS 的全局选点边界，保持现有 zero-shot selection 实现不变。

取回服务器 `cache/manifest.jsonl` 与 data/model audit 后，运行
`python -m src.audit_followup --root .` 生成补充审计报告。它区分文件名重合与
已确认内容重复，并按现有代码重建采样（不冒充运行时 cache records）。
服务器可运行 `python -m src.verify_manifest_duplicates`，仅对 Brain/OCT2017
跨 split 同名候选计算文件及 RGB 像素哈希，输出
`results/analysis/server_duplicate_checks.json`；不需要 GPU 或重训。

下一步使用已锁定 final checkpoint 导出逐图分数：

```bash
python -m src.export_predictions --config configs/experiment.yaml --check-only
GPUS=0,1,2,3 bash run.sh predictions
```

详见 [逐图分数导出说明](docs/PREDICTION_REPLAY_CN.md)。仅重放已有 12 个 final jobs，
输出到 `results/predictions/`；缺失 checkpoint 或重放指标不一致会失败，
不重训、不重选配置。图像评分保持 contrast_topk / 0.05。

取回完整 predictions 后运行 `python -m src.analyze_predictions --root .`，先验证
导出哈希/标签/指标，再生成配对图像 bootstrap 区间、分数/ROC/PR、跨 seed 秩
相关和误例候选。区间条件于固定模型，不能当作患者级 CI；所有附加 scoring
比较仅为 post-hoc 诊断，不能替换主评分。输出仅在 results/ 下，不上传仓库。

历史附属入口为 `bash run.sh diagnostics`（固定 checkpoint 的空间/source 诊断）
和 `bash run.sh common_stitch`（固定 s2b3/s2b9 对照；复用原 12 jobs，补训 12 jobs）。
执行前使用 `python -m src.run_followup --stage <stage> --check-only`。
命令、预算与只取回 `results/followup/` 的清单见
[补充实验运行与取回说明](docs/FOLLOWUP_EXPERIMENTS_CN.md)。原主实验和选点不变。
当前参数拼接方法研究优先运行本页开头的 run_method_controls.sh。
