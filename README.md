# MedStitch-ZS / Model-concatenation

面向 **zero-shot medical anomaly detection** 的跨架构模型裁剪拼接实验框架。项目将 DINOv3 ConvNeXt-Tiny 的前部与 DINOv3 ViT-S/16 的后部拼成单一可执行网络，并研究 **CNN–Transformer stitchability / representation disagreement 是否可直接作为医学异常信号**。

项目按照 **1–4 × RTX 3090 (24 GB)** 优化，默认最多使用 4 卡。多卡不做 DDP，而是同时运行不同数据集、stitch configuration 与随机种子，避免通信开销。

## 1. 你只需要做两件事

### 1.1 准备 Python 环境

建议 Python 3.11。先安装与你机器 CUDA 匹配的 PyTorch，然后：

```bash
pip install -r requirements.txt
```

代码要求 `transformers >= 4.56`，因为 DINOv3 的 Hugging Face/Transformers 架构支持从该版本开始可直接加载。

### 1.2 把 BMAD 数据放进 `data/`

数据由用户手动下载，本项目 **不会自动下载数据集**。推荐直接使用 BMAD 官方整理后的 6 个数据集：

```text
data/
├── Brain/
├── liver/
├── RESC/
├── OCT2017/
├── RSNA/              # 也接受 Chest-RSNA/ 或 Chest/
└── camelyon16/        # 也接受 camelyon16_256/
```

大小写不敏感。代码也会递归寻找这些目录，因此外面多套一层 `BMAD/` 也可以。

BMAD 官方典型结构：

```text
Brain/
├── train/good/img/*.png
├── valid/good/img/*.png
├── valid/Ungood/img/*.png
├── valid/Ungood/anomaly_mask/*.png
├── test/good/img/*.png
├── test/Ungood/img/*.png
└── test/Ungood/anomaly_mask/*.png
```

Camelyon16 等 image-level 数据可以是：

```text
camelyon16/
├── train/good/*.png
├── valid/good/*.png
├── valid/Ungood/*.png
├── test/good/*.png
└── test/Ungood/*.png
```

如果 `data/` 顶层放的是 `.zip/.tar/.tar.gz/.tgz`，`run.sh` 会先解压到 `data/_extracted/`，再递归识别数据。大数据已经手工解压时，不需要保留压缩包。

## 2. 一条命令运行全部实验

```bash
bash run.sh
```

等价于：

1. 审查数据并生成 `cache/data_audit.json`、`cache/manifest.jsonl`；
2. 使用 **ModelScope SDK** 自动下载两个模型到 `model/`；
3. 一次性缓存 ConvNeXt stage1/2/3 与完整 ViT patch features；
4. 缓存小规模 synthetic perturbation 特征供 AOSS 使用；
5. 9 个 stitch points 只用 source-domain AOSS 快速 screening；
6. **不查看 target test AUROC**，按 AOSS 自动锁定 Top-2 configuration；
7. Top-2 × 3 seeds 正式 leave-one-modality-out zero-shot target-test 实验；
8. 锁定配置后，复用 screening checkpoint 对 9 个 stitch points 做 post-hoc stitchability map（用于 AOSS–AUROC 相关分析，不参与选模）；
9. calibration size / adapter / anomaly-score 消融；
10. 自动整理 CSV 与 `results/REPORT.md`。

也支持断点式执行：

```bash
bash run.sh prepare
bash run.sh cache
bash run.sh screen
bash run.sh final
bash run.sh stitchmap
bash run.sh ablation
bash run.sh report
```

所有阶段都是幂等的：已完成的缓存和 JSON 结果默认跳过。

## 3. ModelScope 权重

所有模型权重只通过 ModelScope 自动下载，不使用 Hugging Face Hub 下载：

- `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`
- `facebook/dinov3-vits16-pretrain-lvd1689m`

最终目录：

```text
model/
├── dinov3_convnext_tiny/
└── dinov3_vits16/
```

`src/models.py` 使用：

```python
from modelscope import snapshot_download
snapshot_download(repo_id=..., local_dir=...)
```

下载完成后，Transformers 只从本地目录 `local_files_only=True` 加载，因此实验过程中不会偷偷访问其他模型站点。

> DINOv3 使用其自己的模型许可。开展研究与发布时请核对并遵守模型许可及 BMAD/原始数据集许可。

## 4. 为什么实验速度比较快

### 4.1 Backbone 只跑一次

最耗时的 ConvNeXt / 完整 ViT forward 不会为每一个 stitch configuration 重复运行。

第一次 cache 阶段把下面的 FP16 feature 写成 `.npy` memory-mapped arrays：

```text
cache/features/<dataset>/<split>/
├── s1.npy       # 28 × 28 × 192
├── s2.npy       # 14 × 14 × 384
├── s3.npy       #  7 ×  7 × 768
├── vit.npy      # 196 × 384 final ViT patch tokens
└── records.jsonl
```

之后的 adapter training / evaluation 只读这些缓存。

### 4.2 训练不再重复跑 CNN 和完整 ViT

训练路径是：

```text
cached ConvNeXt stage feature
          ↓
   Stitch Adapter
          ↓
     ViT tail only
          ↓
match cached full-ViT final tokens
```

例如 `Stage2 → Block9` 只需要运行 ViT 的最后 3 个 blocks。

### 4.3 Screening → Top-2

默认只先跑：

```text
source stage ∈ {1, 2, 3}
target block ∈ {3, 6, 9}
```

即 9 个 configurations、1 seed、较小 calibration set。筛选阶段只依据 source-domain AOSS；锁定 Top-2 后才对目标测试集跑 3 seeds。

### 4.4 4 卡采用 experiment-level parallelism

默认检测最多 4 张 GPU，并把独立任务分发给空闲卡：

```text
GPU0 -> dataset/config A
GPU1 -> dataset/config B
GPU2 -> dataset/config C
GPU3 -> dataset/config D
```

不使用 DDP/FSDP，因此没有梯度同步和 NCCL 通信成本。

指定卡：

```bash
GPUS=0,1,2,3 bash run.sh
```

只使用单卡：

```bash
GPUS=0 bash run.sh
```

## 5. Stitching 模型

ConvNeXt 的三个候选 source stages：

```text
s1: 28×28×192 -> pool -> 14×14 -> projection -> 384
s2: 14×14×384 -> native match
s3:  7× 7×768 -> projection -> upsample -> 14×14×384
```

ViT-S/16 在 224×224 输入下产生 14×14=196 个 patch tokens，hidden dimension=384。

MLP adapter：

```text
CNN feature
   ↓ spatial/channel adaptation
196×384
   ↓ LayerNorm
384 -> 768 -> 384
   ↓ residual
stitched patch tokens
   ↓
ViT block j ... block 11
```

CLS + register prefix tokens由原始 ViT 初始化，但作为 adapter 的少量可训练参数；ViT tail 权重全部冻结。

## 6. NFFA training objective

完整 ViT 最终 patch tokens：

[
Z^T = T(x)
]

stitched network：

[
Z^S = T_{j:11}(A(C_i(x)))
]

训练仅使用 **source modalities 的 normal images**：

[
L_{patch}=1-operatorname{cos}(Z^S,Z^T)
]

[
L=L_{patch}+0.1L_{global}
]

目标模态的 train/valid 图像不会用于 adapter 训练。

## 7. Strict leave-one-modality-out protocol

BMAD 六个数据集组织成五个 modality folds：

```text
brain_mri : Brain
liver_ct  : liver
oct       : RESC + OCT2017
chest_xray: RSNA
pathology : camelyon16
```

例如 target=`liver_ct` 时，adapter 只从 Brain/OCT/RSNA/Camelyon normal train images 学习；liver 只用于最终 test。

因此主实验是 **cross-modality zero-shot**，而不是 target-normal unsupervised AD。

## 8. AOSS

缓存阶段还会对每个 source dataset 的少量 normal images 生成 deterministic synthetic perturbations：

- local intensity shift
- local blur
- local patch copy

候选 stitch point 的 normal discrepancy：

[
D_N=mathbb{E}[1-cos(Z^S,Z^T)]
]

扰动区域内外差：

[
S_{local}=D_{in}-D_{out}
]

AOSS：

[
AOSS=rac{S_{local}}{D_N+epsilon}
]

`results/screen_summary.csv` 先按 **source-only AOSS** 排名并锁定配置。正式配置锁定之后，`stitchmap` 阶段才复用已保存的 screening adapter 对全部 9 个拼接点做 target-test post-hoc 分析，`REPORT.md` 再计算 AOSS 与 AUROC 的 Spearman 相关；该 AUROC 不参与模型选择。

## 9. Zero-shot anomaly score

目标图像无需 anomaly labels / target-normal memory bank。

每个 patch：

[
d_p=1-cos(Z^S_p,Z^T_p)
]

默认 image-level score 使用 spatial contrast：

[
s(x)=mean(Top5%(d-median(d)))
]

这样可部分消除跨模态图像的全局 representation shift，突出局部 disagreement。

消融自动比较：

- `raw_topk`
- `contrast_topk`
- `robust_topk`

## 10. 快速内部 baselines

为了在第一轮实验中快速判断课题是否成立，缓存完成后会零额外模型训练地跑：

- `vit_patch_dispersion`
- `convnext_s2_dispersion`
- `direct_s2_vit_gap`

结果保存于 `results/baselines.json`。

正式投稿前仍建议根据论文目标补充公开方法（例如当时可稳定复现的 zero-shot medical AD baselines），但这不会影响本项目的主实验代码和缓存。

## 11. 输出

```text
results/
├── baselines.json
├── screen/
│   ├── s1_b3.json
│   ├── ...
│   └── s3_b9.json
├── screen_summary.csv
├── selected_configs.json
├── stitchmap/          # post-hoc，绝不参与选模
├── final/
│   ├── *.json
│   └── checkpoints/
├── ablation/
├── final_results.csv
├── ablation_results.csv
└── REPORT.md
```

正式 final jobs 会保存每个 target modality 对应的 **adapter-only checkpoint**；冻结的 DINOv3 权重仍从 `model/` 加载，不重复保存。

## 12. 默认参数为什么这样设

默认配置位于 `configs/experiment.yaml`：

- 输入 224×224：与 ViT-S/16 的 14×14 patch grid 对齐；
- FP16 feature cache：显著降低磁盘和 I/O；
- train cache 每数据集最多 2500 normal samples：覆盖 2000-sample calibration ablation；
- screening：500/source-modality、4 epochs；
- final：1000/source-modality、8 epochs；
- batch size=128（adapter stage）；
- feature extraction batch size=64；
- AMP 开启；
- TF32 开启；
- 只保存 adapter checkpoints。

如 3090 显存不足，优先将 `cache.batch_size` 从 64 调到 32；adapter training 通常显存压力很小。

## 13. 断点续跑与清缓存

`run.sh` 会检测 `.done` 文件和已有结果，因此中断后直接再次：

```bash
bash run.sh
```

即可继续。

若修改了图像预处理、模型或 cache 结构，应删除：

```bash
rm -rf cache/features cache/perturb
```

若只修改训练超参数，不需要重建 feature cache。

---

当前代码目标是先用最少工程成本验证论文核心假设：**真正的 CNN→Transformer network surgery 能否在完全不接触目标模态训练数据的情况下，将 cross-architecture disagreement 转化为异常信号，并由 AOSS 自动找到更适合异常检测的拼接点。**
