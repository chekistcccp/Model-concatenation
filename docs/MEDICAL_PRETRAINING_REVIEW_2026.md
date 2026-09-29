# 2026 医学预训练模型调研与最终实验选择

更新日期：2026-09-29

## 最终结论

为了避免“模型架构变化”和“预训练域变化”同时发生，当前项目采用一个更严格的 2×2 对照设计。

### CNN source

- **medical**：RadImageNet-ResNet50
- **general**：ImageNet-ResNet50

二者都是 ResNet50。

### Transformer target

- **medical**：RadioDINO-S/16
- **general**：DINO ViT-S/16

二者都是 ViT-Small/16，224 输入、384 维 token。

因此四组实验是：

1. medical CNN → medical ViT（MM，主模型）
2. medical CNN → general ViT（MG）
3. general CNN → medical ViT（GM）
4. general CNN → general ViT（GG）

这样可以把 source-side medical pretraining 与 target-side medical pretraining 的影响拆开分析。

---

## RadImageNet-ResNet50

官方仓库：

https://github.com/BMEII-AI/RadImageNet

RadImageNet 包含约 135 万 CT、MRI 和超声医学图像，覆盖多种解剖部位和病理标签。

官方提供 PyTorch ResNet50 权重，并在官方 notebook 中以 ResNet50 backbone 方式严格加载。

当前代码使用：

```text
model/radimagenet_resnet50/resnet50_torch.pt
```

作为医学 CNN source。

---

## RadioDINO-S/16

官方模型：

https://huggingface.co/Snarcy/RadioDino-s16

官方代码/项目：

https://github.com/Snarci/Radio-DINO

RadioDINO-S/16：

- ViT-Small
- patch size 16
- hidden size 384
- 约 21.7M 参数
- DINO 自监督训练
- 预训练数据为 RadImageNet（CT/MRI/US）

当前代码使用：

```text
model/radiodino_s16/config.json
model/radiodino_s16/model.safetensors
```

作为主医学 Transformer target。

---

## 为什么不再把 RAD-DINO 设为默认主模型

RAD-DINO 是质量很高的医学视觉模型，但其主要预训练域是胸片，并且架构是 DINOv2 ViT-B/14。

如果拿它直接与 DINOv3 ViT-S/16 比较，会同时改变：

- 预训练域；
- 模型规模；
- patch size；
- hidden dimension；
- Transformer 实现。

这会削弱“医学预训练本身是否改善 stitching”的因果解释。

RAD-DINO 更适合作为后续额外 external medical backbone，而不是当前最核心的 controlled comparison。

---

## 为什么 RadioDINO 更适合本项目

RadioDINO-S/16 与标准 DINO ViT-S/16 在主要结构上可以对齐：

```text
ViT-S/16
224×224
14×14 patch grid
384 hidden dimension
```

因此可以保持：

```text
same architecture
different pretraining domain
```

这比比较不同大小、不同 patch size 的 foundation model 更适合 model stitching 研究。

---

## 为什么 CNN 也改成配对 ResNet50

之前使用 DINOv3 ConvNeXt-Tiny 作为 general CNN source，而医学 CNN 候选是 RadImageNet-ResNet50。

这样会把：

```text
ConvNeXt vs ResNet
```

和：

```text
general vs medical pretraining
```

混在一起。

当前改成：

```text
ImageNet ResNet50
vs
RadImageNet ResNet50
```

二者架构一致。

因此整个论文可以真正组织为一个 2×2 pretraining-domain study，而不是多个不完全匹配 backbone 的经验比较。

---

## 当前主科学问题

最终建议论文围绕：

> Does domain-specific medical pretraining alter cross-architecture stitchability, and can this change be exploited as a zero-shot medical anomaly signal?

展开。

重点结果：

1. MM / MG / GM / GG 四组 zero-shot AUROC；
2. radiology 与 non-radiology 分组；
3. AOSS 对真实 anomaly performance 的预测能力；
4. source medical pretraining 的主效应；
5. target medical pretraining 的主效应；
6. medical-medical pairing 是否出现 interaction / synergy；
7. 参数量、FLOPs、显存与延迟。

---

## 详细模型下载与目录

请以仓库根目录：

```text
PREPARE_EXPERIMENT_CN.md
```

为唯一执行说明。
