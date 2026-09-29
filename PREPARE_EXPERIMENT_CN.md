# MedStitch-ZS 实验准备说明（Google Drive 手动 + ModelScope 自动）

本版本的准备规则非常简单：

> **Google Drive 上必须手动下载的内容由你准备；其余预训练模型全部由代码通过 ModelScope 自动下载。**

因此你实际只需要手工准备 **BMAD 数据** 和 **RadImageNet-ResNet50 权重**。

---

# 1. 最终目录

推荐最终目录：

```text
Model-concatenation/
├── data/
│   └── BMAD/
│       ├── Brain/
│       ├── liver/
│       ├── RESC/
│       ├── OCT2017/
│       ├── RSNA/
│       └── camelyon16/
│
├── model/
│   ├── radimagenet_resnet50/
│   │   └── resnet50_torch.pt          # 你手工放
│   ├── resnet50_a1_in1k/              # ModelScope 自动下载
│   ├── rad_dino/                       # ModelScope 自动下载
│   └── dinov2_base/                    # ModelScope 自动下载
│
├── cache/
├── results/
└── run.sh
```

---

# 2. 你必须手工下载的内容

## 2.1 BMAD 整合数据 —— Google Drive

BMAD 官方仓库：

https://github.com/DorisBao/BMAD

BMAD 官方 README 提供的 Google Drive：

https://drive.google.com/drive/folders/1AC-wWZl_K18CWL2eIxUScoSOoxT4IBuw?usp=sharing

下载其中整理好的六个数据集：

```text
Brain
liver
RESC
OCT2017
RSNA
camelyon16
```

放到：

```text
data/BMAD/
```

最终：

```text
data/BMAD/Brain/
data/BMAD/liver/
data/BMAD/RESC/
data/BMAD/OCT2017/
data/BMAD/RSNA/
data/BMAD/camelyon16/
```

这六个数据集覆盖：

- Brain：脑 MRI
- liver：肝 CT
- RESC：OCT
- OCT2017：OCT
- RSNA：胸片
- camelyon16：病理

不需要下载 BMAD 官方训练好的 14 个 anomaly detection checkpoint。

---

## 2.2 RadImageNet-ResNet50 —— Google Drive

RadImageNet 官方仓库：

https://github.com/BMEII-AI/RadImageNet

官方 README 中的 PyTorch pretrained models：

https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view?usp=sharing

下载 PyTorch 模型包并解压。

本项目需要其中的：

```text
resnet50_torch.pt
```

最终放置为：

```text
model/radimagenet_resnet50/resnet50_torch.pt
```

这是唯一需要你手工准备的模型权重。

如果下载包中的 ResNet50 文件已经叫 `resnet50_torch.pt`，直接复制即可；不要重新转换 state_dict。

代码会按 RadImageNet 官方 PyTorch notebook 的 Backbone 格式严格加载。

---

# 3. 不需要你手工下载的模型

下面三个模型都由代码通过 ModelScope SDK 自动下载。

只要服务器可以访问 ModelScope 即可。

---

## 3.1 通用 CNN：ImageNet ResNet50

ModelScope ID：

```text
timm/resnet50.a1_in1k
```

ModelScope：

https://modelscope.cn/models/timm/resnet50.a1_in1k

自动保存到：

```text
model/resnet50_a1_in1k/
```

用途：

```text
general CNN source
```

它和 RadImageNet source 都属于 ResNet50 系列，输出 stage 维度统一为：

```text
s1 = layer2: 28×28×512
s2 = layer3: 14×14×1024
s3 = layer4:  7× 7×2048
```

---

## 3.2 医学 Transformer：RAD-DINO

ModelScope ID：

```text
microsoft/rad-dino
```

ModelScope：

https://modelscope.cn/models/microsoft/rad-dino

自动保存到：

```text
model/rad_dino/
```

用途：

```text
medical Transformer target
```

架构：

```text
DINOv2-Base / ViT-B/14
hidden dimension = 768
12 blocks
patch size = 14
```

RAD-DINO 是医学自监督视觉模型，主要预训练于胸部 X-ray。

---

## 3.3 通用 Transformer：DINOv2-Base

ModelScope ID：

```text
facebook/dinov2-base
```

ModelScope：

https://modelscope.cn/models/facebook/dinov2-base

自动保存到：

```text
model/dinov2_base/
```

用途：

```text
general Transformer target
```

它和 RAD-DINO 使用相同的 DINOv2-Base / ViT-B14 架构，因此可以形成更严格的：

```text
same architecture
different pretraining domain
```

对照。

---

# 4. 当前 2×2 实验设计

最终四组：

| CNN source | Transformer target | 缩写 | 作用 |
|---|---|---|---|
| RadImageNet-ResNet50 | RAD-DINO | MM | 主模型：医学→医学 |
| RadImageNet-ResNet50 | DINOv2-Base | MG | 医学 CNN + 通用 ViT |
| ImageNet-ResNet50 | RAD-DINO | GM | 通用 CNN + 医学 ViT |
| ImageNet-ResNet50 | DINOv2-Base | GG | 通用→通用 |

这样可以分别研究：

1. CNN 前端医学预训练是否有效；
2. Transformer 后端医学预训练是否有效；
3. 医学→医学是否存在额外协同；
4. 收益是否集中在 MRI / CT / X-ray，而在 OCT / Pathology 上不同。

---

# 5. 你真正需要手工准备的最简清单

只需要：

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

除此之外的三个模型不要手工下载。

---

# 6. 安装环境

建议 Python 3.11。

先安装与你 CUDA 匹配的 PyTorch，然后：

```bash
pip install -r requirements.txt
```

依赖中包含：

```text
modelscope
transformers
timm
safetensors
```

---

# 7. 第一步：只检查你手工准备得对不对

执行：

```bash
bash run.sh prepare
```

这个阶段不会主动下载 ModelScope 模型。

它会检查：

- BMAD 六个目录是否存在；
- RadImageNet `resnet50_torch.pt` 是否存在且不是空文件；
- 三个自动模型当前是否已经存在。

生成：

```text
cache/preflight_report.json
cache/data_audit.json
```

如果你刚 clone 仓库，正常可能看到：

```text
data ready: True
models ready: False

source medical: READY
source general: MISSING
target medical: MISSING
target general: MISSING
```

这并不是错误，因为后三个模型本来就应该自动下载。

---

# 8. 第二步：自动下载 ModelScope 模型

确认 BMAD + RadImageNet 已准备好后：

```bash
bash run.sh models
```

代码会自动下载：

```text
timm/resnet50.a1_in1k
microsoft/rad-dino
facebook/dinov2-base
```

分别保存到：

```text
model/resnet50_a1_in1k/
model/rad_dino/
model/dinov2_base/
```

下载后生成：

```text
cache/model_audit.json
```

并再次执行准备检查。

理想状态：

```text
data ready: True
models ready: True
ready_for_full_run: true
```

---

# 9. 也可以直接运行完整实验

如果 BMAD 和 RadImageNet 已准备好，你不必单独执行 models：

```bash
GPUS=0,1,2,3 bash run.sh
```

完整流程发现自动模型缺失时，会先使用 ModelScope 下载，然后继续实验。

---

# 10. 第一次推荐先单卡跑 cache

建议：

```bash
GPUS=0 bash run.sh cache
```

成功后应看到类似：

```text
cache/features/Brain/train/
├── source_medical_s1.npy
├── source_medical_s2.npy
├── source_medical_s3.npy
├── source_general_s1.npy
├── source_general_s2.npy
├── source_general_s3.npy
├── target_medical.npy
├── target_general.npy
└── records.jsonl
```

这里：

```text
source_medical = RadImageNet-ResNet50
source_general = ModelScope ImageNet-ResNet50
target_medical = ModelScope RAD-DINO
target_general = ModelScope DINOv2-Base
```

一旦 cache 成功，后面 36 个 stitch screening 配置不会再重复运行完整四个 backbone。

---

# 11. 正式四卡运行

```bash
GPUS=0,1,2,3 bash run.sh
```

主要流程：

```text
BMAD audit
      ↓
检查手工 RadImageNet
      ↓
ModelScope 自动补齐 3 个模型
      ↓
共享 FP16 feature cache
      ↓
MM / MG / GM / GG
×
3 source stages
×
3 target cuts
=
36 AOSS screening jobs
      ↓
每个 pair 选 Top-1
      ↓
4 configs × 3 seeds
      ↓
post-hoc stitchability map
      ↓
MM 消融
      ↓
REPORT.md
```

---

# 12. 分辨率说明

RAD-DINO 和 DINOv2-Base 都是 patch14。

为了节省 RTX 3090 实验时间，默认输入：

```text
224×224
```

所以：

```text
224 / 14 = 16
16×16 = 256 patch tokens
```

两套 target 使用相同 token grid。

原始 DINOv2/RAD-DINO 可使用更大输入，代码通过 DINOv2 positional embedding interpolation 支持 224 输入。

建议第一轮全部使用 224；如果核心结果成立，再对最佳 MM 配置补高分辨率消融。

---

# 13. 下载失败时怎么处理

如果 ModelScope 访问失败，不需要重新下载 Google Drive 数据。

只需再次：

```bash
bash run.sh models
```

已经完整存在的模型会跳过，缺失模型继续下载。

模型统一保存在项目自己的 `model/`，不会依赖用户 home 目录里的随机缓存位置。

---

# 14. 一句话版本

你手动准备：

```text
Google Drive:
1. BMAD
2. RadImageNet PyTorch ResNet50
```

代码自动准备：

```text
ModelScope:
1. timm/resnet50.a1_in1k
2. microsoft/rad-dino
3. facebook/dinov2-base
```

然后：

```bash
bash run.sh prepare
bash run.sh models
GPUS=0 bash run.sh cache
GPUS=0,1,2,3 bash run.sh
```
