# MedStitch-ZS / Model-concatenation

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
