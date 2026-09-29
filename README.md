# MedStitch-ZS / Model-concatenation

本项目研究 **跨架构 model stitching 是否可以形成 zero-shot medical anomaly signal**。

当前版本采用严格的 2×2 预训练域对照：

| CNN source | Transformer target | 角色 |
|---|---|---|
| RadImageNet-ResNet50 | RAD-DINO | MM：医学→医学，主模型 |
| RadImageNet-ResNet50 | DINOv2-Base | MG |
| ImageNet-ResNet50 | RAD-DINO | GM |
| ImageNet-ResNet50 | DINOv2-Base | GG |

CNN 都是 ResNet50 系列；Transformer 都是 DINOv2-Base / ViT-B14，因此可以重点分析 **medical pretraining** 而不是把架构差异混进结果。

## 最重要的准备规则

请先阅读：

**[PREPARE_EXPERIMENT_CN.md](PREPARE_EXPERIMENT_CN.md)**

当前只需要你手工准备 Google Drive 上的两个项目：

```text
1. BMAD 整合数据
2. RadImageNet PyTorch ResNet50
```

其余模型由代码自动通过 ModelScope 下载：

```text
timm/resnet50.a1_in1k
microsoft/rad-dino
facebook/dinov2-base
```

## 你手工准备的目录

```text
data/BMAD/
├── Brain/
├── liver/
├── RESC/
├── OCT2017/
├── RSNA/
└── camelyon16/

model/radimagenet_resnet50/
└── resnet50_torch.pt
```

其他 model/ 子目录不需要你创建。

## 环境

推荐 Python 3.11。

先安装适合你的 CUDA / RTX 3090 的 PyTorch，然后：

```bash
pip install -r requirements.txt
```

## 1. 检查手工文件

```bash
bash run.sh prepare
```

如果只准备了 Google Drive 内容，正常状态可能是：

```text
data ready: True
models ready: False

source medical: READY
source general: MISSING (will auto-download from ModelScope)
target medical: MISSING (will auto-download from ModelScope)
target general: MISSING (will auto-download from ModelScope)
```

## 2. 自动下载 ModelScope 模型

```bash
bash run.sh models
```

自动保存为：

```text
model/resnet50_a1_in1k/
model/rad_dino/
model/dinov2_base/
```

再次检查后应为：

```text
data ready: True
models ready: True
```

## 3. 第一次建议单卡建立缓存

```bash
GPUS=0 bash run.sh cache
```

缓存同时生成：

```text
source_medical_s1/s2/s3
source_general_s1/s2/s3
target_medical
target_general
```

完整四个 backbone 对每张图只运行一次，后续 stitching 实验只读 FP16 memmap。

## 4. 一条命令跑全部实验

4× RTX 3090：

```bash
GPUS=0,1,2,3 bash run.sh
```

如果自动模型尚未下载，完整流程会先通过 ModelScope 补齐，再继续。

完整流程：

```text
BMAD audit
    ↓
检查 RadImageNet 手工权重
    ↓
ModelScope 自动下载 3 个模型
    ↓
共享 FP16 feature cache
    ↓
4 pairs × 3 ResNet stages × 3 DINOv2 cuts
= 36 AOSS screening configs
    ↓
每个 pair 选 Top-1
    ↓
4 configs × 3 seeds final
    ↓
36-point post-hoc stitchability map
    ↓
MM 主模型 ablation
    ↓
CSV + REPORT.md
```

## Stitch source

两个 ResNet50 的候选 stage：

```text
s1 = layer2: 28×28×512
s2 = layer3: 14×14×1024
s3 = layer4:  7× 7×2048
```

## Stitch target

RAD-DINO 与 DINOv2-Base 都是 DINOv2-Base / ViT-B14：

```text
input = 224×224
patch = 14
grid = 16×16
hidden = 768
12 transformer blocks
```

候选 cut：

```text
block 3 / 6 / 9
```

DINOv2 positional embedding 会根据 224 输入显式插值。

## Strict zero-shot protocol

五个 modality folds：

- Brain MRI
- Liver CT
- OCT（RESC + OCT2017）
- Chest X-ray
- Pathology

目标 modality 的 train/valid 不用于训练和选模；AOSS 只使用其他 source-domain normal images + synthetic perturbations。目标 test 只在配置锁定后评价。

## 输出

```text
results/
├── baselines.json
├── screen/
├── screen_summary.csv
├── selected_configs.json
├── final/
├── stitchmap/
├── ablation/
├── final_results.csv
├── ablation_results.csv
└── REPORT.md
```

详细下载地址与目录以 **PREPARE_EXPERIMENT_CN.md** 为准。
