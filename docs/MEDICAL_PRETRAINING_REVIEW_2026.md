# 2026 医学预训练模型调研与本项目选择

更新日期：2026-09-29

## 结论

本项目不再只使用通用 DINOv3。默认将 **RAD-DINO (microsoft/rad-dino) 作为医学预训练 Transformer target**，并保留 DINOv3 ViT-S/16 作为 general-vision control。

这样设计的研究问题从“CNN 和 Transformer 是否可以拼接用于医学异常检测？”升级为“医学域预训练是否会改变跨架构 stitchability，以及这种改变是否在 radiology 与 non-radiology 模态间具有不同规律？”

主实验仍严格采用 ModelScope 自动下载权重。

## RAD-DINO — 当前主医学 target

- Model: microsoft/rad-dino
- Architecture: DINOv2-Base / ViT-B/14
- Hidden dimension: 768
- 12 transformer blocks
- Medical pretraining: self-supervised chest radiographs
- Published work: Exploring scalable medical image encoders beyond text supervision, Nature Machine Intelligence, 2025
- Model card: https://huggingface.co/microsoft/rad-dino
- ModelScope: https://modelscope.cn/models/microsoft/rad-dino

优点：已确认 ModelScope 可用；是真正医学图像自监督视觉编码器；结构适合 block-level stitching；ViT-B 规模可在 RTX 3090 上进行 adapter-only 实验。

限制：主要训练域是 chest X-ray，因此不能把它描述成覆盖所有医学模态的通用 foundation model。报告中应将 Brain/Liver/Chest 的 radiology 结果与 OCT/Pathology 的 non-radiology 结果分开。

## RadioDINO — 科学上非常合适，但目前不作为默认模型

RadioDINO-S/16 基于 RadImageNet 约 1.35M CT/MRI/ultrasound 图像进行医学自监督训练，384 hidden dimension，和原 DINOv3 ViT-S 尺度接近，非常适合 architecture-matched comparison。

- Model: https://huggingface.co/Snarcy/RadioDino-s16
- Project: https://github.com/unica-visual-intelligence-lab/OmniRad

当前没有将其设为默认，是因为本项目要求权重全部使用 ModelScope SDK 自动下载，而本轮调研没有验证到稳定公开的 ModelScope 镜像。

## OmniRad — 2026 年很值得后续加入

OmniRad 是 2026 radiology foundation model，约 1.2M medical images，重点强调 CT/MRI/X-ray/ultrasound 跨模态 transfer，ViT-S 级别且单卡友好。

- Project: https://github.com/unica-visual-intelligence-lab/OmniRad

它的 domain coverage 比 RAD-DINO 更符合 BMAD 的 Brain MRI / Liver CT / Chest X-ray 组合，但当前同样没有作为默认模型，因为尚未验证到符合本项目约束的稳定 ModelScope 下载源。

## RadImageNet — 医学 CNN source 的自然候选

RadImageNet 包含约 1.35M 医学图像，覆盖 CT、MRI、ultrasound 等模态，并公开 ResNet50、DenseNet121 等 CNN 权重。

- Project: https://github.com/BMEII-AI/RadImageNet

它是把 CNN source 也医学化的自然选择。但当前版本不直接切换，因为本项目要求所有模型通过 ModelScope 自动下载，而官方权重主要经项目提供的外部下载位置发布；本轮调研未验证到可稳定依赖的 ModelScope RadImageNet CNN repo。

因此本版本采用：General DINOv3 ConvNeXt front + Medical RAD-DINO Transformer tail。研究上可解释为 grafting a general local-feature extractor into a medical-domain transformer representation space。

## BiomedCLIP / MedSigLIP

它们具有更广泛 biomedical / multimodal medical coverage，尤其适合 pathology、ophthalmology 等 non-radiology 模态。但本项目核心是 CNN→Transformer block-level surgery、中间层 feature alignment 与低成本 3090 实验；VLM/CLIP 会同时引入 text supervision 与不同训练目标，因此第一轮不作为默认 backbone。

## 为什么不把所有通用模型全部替换成医学模型

医学预训练并非所有模态和任务上必然优于通用预训练，预训练域与目标模态的匹配关系本身就是需要测量的问题。

因此使用 controlled comparison：

same CNN source / same BMAD splits / same NFFA / same AOSS / same anomaly score / same stitch candidates；仅 target 改为 RAD-DINO medical-pretrained 或 DINOv3 ViT-S general-pretrained。

这样医学预训练本身就是可测量的实验变量，而不会和异常检测头、数据划分等因素混在一起。

## 输入分辨率选择

RAD-DINO 的公开配置使用 518 级输入，patch size=14；518/14=37，即 1369 patch tokens。默认快速实验用 224，224/14=16，即 256 patch tokens。

DINOv2 absolute positional embeddings 支持插值，因此本项目默认以 224 进行大规模 screening/final experiments，显著缩短 3090 实验时间。若核心结果成立，建议最后仅对最佳医学 stitch configuration 增加 RAD-DINO 518 resolution ablation。

## 当前代码协议

缓存一次写入 s1.npy、s2.npy、s3.npy、target_medical.npy (RAD-DINO) 和 target_general.npy (DINOv3 ViT-S)。每张原图只解码一次，再应用各自模型正确的 normalization。

Screening 为 2 target backbones × 3 CNN stages × 3 Transformer cuts = 18 configurations。每个 target 单独按 source-only AOSS 排名，并各取 Top-2，不使用 target-test AUROC 做模型选择。

## 推荐论文主结果组织

1. Medical target main result: RAD-DINO on all BMAD modalities
2. General-pretraining controlled comparison: DINOv3 ViT-S
3. Domain-group analysis: radiology (Brain MRI/Liver CT/Chest X-ray) vs non-radiology (OCT/Pathology)
4. Stitchability analysis: AOSS vs real zero-shot AUROC, medical vs general
5. Efficiency: parameters, active tail blocks, GFLOPs, latency, VRAM

这比简单声明“医学模型更好”更有研究价值，因为结果本身可以回答医学域预训练在何种模态上改善或破坏 cross-architecture compatibility。