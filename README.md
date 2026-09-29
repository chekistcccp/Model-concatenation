# MedStitch-ZS / Model-concatenation

本项目研究 **跨架构模型裁剪拼接（model stitching）是否可以直接形成 zero-shot medical anomaly signal**。

当前版本已改为更严格的 **2×2 医学/通用预训练对照**：

| CNN source | Transformer target | 角色 |
|---|---|---|
| RadImageNet-ResNet50 | RadioDINO-S/16 | 主模型：医学→医学 |
| RadImageNet-ResNet50 | DINO ViT-S/16 | 医学 CNN 对照 |
| ImageNet-ResNet50 | RadioDINO-S/16 | 医学 Transformer 对照 |
| ImageNet-ResNet50 | DINO ViT-S/16 | 完全通用对照 |

CNN 两边都是 ResNet50，Transformer 两边都是 ViT-S/16，因此比上一版“不同架构同时变化”的比较更适合研究 **medical pretraining 本身对 stitchability 的影响**。

## 必看：完整中文准备说明

请先阅读：

**[PREPARE_EXPERIMENT_CN.md](PREPARE_EXPERIMENT_CN.md)**

其中列出了：

- BMAD 官方下载位置；
- 4 个模型的官方来源；
- 必须手工下载的具体文件；
- 每个文件应放到哪个目录；
- 如何检查 Git-LFS 假权重；
- 第一次运行顺序。

## 推荐目录

```text
data/BMAD/
├── Brain/
├── liver/
├── RESC/
├── OCT2017/
├── RSNA/
└── camelyon16/

model/
├── radimagenet_resnet50/
│   └── resnet50_torch.pt
├── imagenet_resnet50/
│   └── resnet50-11ad3fa6.pth
├── radiodino_s16/
│   ├── config.json
│   └── model.safetensors
└── dino_vits16/
    ├── config.json
    └── model.safetensors
```

当前版本 **不自动下载模型**。这样可以固定模型版本、完全离线运行，并避免服务器网络或模型仓库变化影响实验复现。

## 环境

推荐 Python 3.11。

先安装与你 CUDA 匹配的 PyTorch，再执行：

```bash
pip install -r requirements.txt
```

依赖已经缩减为 PyTorch / torchvision / timm / safetensors 及常规科学计算库，不再要求 ModelScope 或 Transformers。

## 第一步：检查准备状态

```bash
bash run.sh prepare
```

会生成：

```text
cache/preflight_report.json
cache/data_audit.json
cache/model_audit.json
```

目标状态：

```text
data ready:   True
models ready: True
```

## 第二步：建议先单卡建立缓存

```bash
GPUS=0 bash run.sh cache
```

缓存会同时保存：

```text
source_medical_s1/s2/s3
source_general_s1/s2/s3
target_medical
target_general
```

四个 backbone 对每张图只运行一次。

## 第三步：一条命令完成全部实验

4× RTX 3090：

```bash
GPUS=0,1,2,3 bash run.sh
```

单卡：

```bash
GPUS=0 bash run.sh
```

流程：

```text
BMAD audit
    ↓
manual weight validation
    ↓
shared FP16 feature cache
    ↓
4 source-target pairs × 9 stitch points
= 36 source-only AOSS screening jobs
    ↓
每个 pair 选 Top-1
    ↓
4 configs × 3 seeds final experiment
    ↓
36-point post-hoc stitchability map
    ↓
MM 主模型 ablation
    ↓
CSV + REPORT.md
```

## 为什么速度仍然可控

虽然从 18 个配置增加到 36 个，但最贵的完整 backbone forward 已经缓存。

训练时只运行：

```text
cached ResNet stage
    ↓
small adapter
    ↓
ViT tail blocks
```

4×3090 使用 experiment-level parallelism，不使用 DDP/FSDP/NCCL。

## Stitch source

两个 ResNet50 的候选 stage 完全一致：

```text
s1 = layer2: 28×28×512
s2 = layer3: 14×14×1024
s3 = layer4:  7× 7×2048
```

## Stitch target

两个 Transformer 都是 ViT-S/16：

```text
224×224 input
14×14 patch grid
384 hidden dimension
12 blocks
```

候选 cut：

```text
block 3
block 6
block 9
```

其中最自然的接口是：

```text
ResNet layer3
14×14×1024
    ↓
projection + adapter
    ↓
14×14×384
    ↓
ViT tail
```

## Strict zero-shot protocol

五个 modality folds：

- brain_mri：Brain
- liver_ct：liver
- oct：RESC + OCT2017
- chest_xray：RSNA
- pathology：camelyon16

当某 modality 为 target 时：

- 不使用 target train；
- 不使用 target valid；
- 不使用 target test 做 stitch selection；
- AOSS 只由其余 source-domain normal + synthetic perturbation 计算；
- 配置锁定后才读取 target test。

## AOSS

```text
AOSS =
local synthetic perturbation sensitivity
/
normal representation discrepancy
```

每个 MM/MG/GM/GG pair 独立排序。

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

REPORT 会自动给出 medical/general CNN 与 medical/general Transformer 的 2×2 结果，并分别汇总 radiology 与 non-radiology。

详细下载和目录说明请以 **PREPARE_EXPERIMENT_CN.md** 为准。
