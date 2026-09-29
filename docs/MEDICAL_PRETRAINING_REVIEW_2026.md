# 2026 医学预训练模型调研与最终实验选择

更新日期：2026-09-29

## 最终选择

为了同时满足：

1. 医学预训练研究价值；
2. 尽量匹配架构；
3. Google Drive 内容由用户手工准备；
4. 其余模型尽可能由 ModelScope 自动下载；
5. 1–4 × RTX 3090 可快速完成；

当前采用：

### CNN source

- medical：RadImageNet-ResNet50
  - 官方 PyTorch 权重来自 Google Drive，用户手工下载。
- general：ImageNet ResNet50 A1
  - ModelScope：`timm/resnet50.a1_in1k`
  - 代码自动下载。

### Transformer target

- medical：RAD-DINO
  - ModelScope：`microsoft/rad-dino`
  - DINOv2-Base / ViT-B14
  - 代码自动下载。
- general：DINOv2-Base
  - ModelScope：`facebook/dinov2-base`
  - 与 RAD-DINO 保持 DINOv2-Base 架构。
  - 代码自动下载。

四组：

```text
MM = RadImageNet ResNet50 -> RAD-DINO
MG = RadImageNet ResNet50 -> DINOv2-Base
GM = ImageNet ResNet50    -> RAD-DINO
GG = ImageNet ResNet50    -> DINOv2-Base
```

---

## 为什么不再使用 RadioDINO-S/16 作为默认 target

RadioDINO 在科学上很合适，但当前主要公开分发在 Hugging Face。

用户希望：

> Google Drive 必须手动下载的内容手动准备，其他模型尽量由 ModelScope 自动下载。

RAD-DINO 与 DINOv2-Base 在 ModelScope 上都有可用模型页，因此更适合当前实际实验环境。

同时 RAD-DINO 与通用 DINOv2-Base 使用相同的大体架构族，可以构成更明确的 medical/general target comparison。

---

## 为什么 RadImageNet 仍然保留手工下载

RadImageNet 官方仓库明确将 PyTorch pretrained models 发布在 Google Drive：

https://github.com/BMEII-AI/RadImageNet

因此该模型遵循用户要求，由用户手工下载官方 PyTorch 包。

项目只需要：

```text
model/radimagenet_resnet50/resnet50_torch.pt
```

---

## 为什么 BMAD 也手工下载

BMAD 官方仓库直接提供整理后的六个 anomaly-detection 数据集 Google Drive：

https://github.com/DorisBao/BMAD

因此 BMAD 由用户手工下载并放到：

```text
data/BMAD/
```

---

## 当前论文问题

当前设计可用于研究：

> How do source-side and target-side medical pretraining alter cross-architecture stitchability and zero-shot anomaly sensitivity?

主要分析：

1. MM / MG / GM / GG 的总体性能；
2. source medical-pretraining 主效应；
3. target medical-pretraining 主效应；
4. medical-medical interaction；
5. radiology vs non-radiology；
6. AOSS 与真实 AUROC 的相关性；
7. 参数、FLOPs、显存和延迟。

详细准备方式请看：

```text
PREPARE_EXPERIMENT_CN.md
```
