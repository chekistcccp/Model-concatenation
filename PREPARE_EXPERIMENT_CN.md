# MedStitch-ZS 实验准备说明（完整中文版）

> 当前版本 **不再自动下载模型**。请先手工准备 BMAD 数据和 4 个预训练权重，然后运行 `bash run.sh prepare` 检查。

## 1. 最终应该准备成什么样

项目根目录最终推荐为：

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
│   │   └── resnet50_torch.pt
│   ├── imagenet_resnet50/
│   │   └── resnet50-11ad3fa6.pth
│   ├── radiodino_s16/
│   │   ├── config.json
│   │   └── model.safetensors
│   └── dino_vits16/
│       ├── config.json
│       └── model.safetensors
│
├── cache/
├── results/
├── configs/
├── src/
└── run.sh
```

四个模型全部准备好后，代码完全离线运行，不会再访问 Hugging Face、ModelScope 或 PyTorch 下载服务器。

---

# 2. 数据：BMAD

## 2.1 官方来源

BMAD 官方仓库：

https://github.com/DorisBao/BMAD

官方整理后的数据 Google Drive：

https://drive.google.com/drive/folders/1AC-wWZl_K18CWL2eIxUScoSOoxT4IBuw?usp=sharing

BMAD 包含六个重组数据集，覆盖五种医学模态：

- Brain：脑 MRI
- liver：肝脏 CT
- RESC：OCT
- OCT2017：OCT
- RSNA：胸片
- camelyon16：数字病理

请将它们统一放在：

```text
data/BMAD/
```

即：

```text
data/BMAD/Brain/
data/BMAD/liver/
data/BMAD/RESC/
data/BMAD/OCT2017/
data/BMAD/RSNA/
data/BMAD/camelyon16/
```

代码仍兼容旧版直接放在 `data/` 下，但论文实验建议统一使用上述标准目录。

## 2.2 内部目录

BMAD 官方常见结构例如：

```text
Brain/
├── train/
│   └── good/
│       └── img/*.png
├── valid/
│   ├── good/
│   └── Ungood/
└── test/
    ├── good/
    │   └── img/*.png
    └── Ungood/
        ├── img/*.png
        └── anomaly_mask/*.png
```

Camelyon16 等只有图像级标签的数据可以是：

```text
camelyon16/
├── train/good/*.png
├── valid/good/*.png
├── valid/Ungood/*.png
├── test/good/*.png
└── test/Ungood/*.png
```

无需自行重新划分 train/test。

---

# 3. 模型 1：RadImageNet-ResNet50（医学 CNN，主 source）

## 3.1 为什么用它

这是当前主实验中的医学 CNN 前端，与通用 ImageNet-ResNet50 保持相同 ResNet50 架构，只改变预训练域。

RadImageNet 官方包含约 135 万 CT、MRI、超声医学图像。

官方仓库：

https://github.com/BMEII-AI/RadImageNet

官方 PyTorch 权重：

https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view?usp=sharing

## 3.2 需要准备的文件

下载官方 PyTorch 权重包，解压后找到 ResNet50 权重。

代码期望最终文件：

```text
model/radimagenet_resnet50/resnet50_torch.pt
```

如果下载包中的 ResNet50 文件名不同，请将对应的官方 ResNet50 PyTorch 权重重命名为：

```text
resnet50_torch.pt
```

不要转换 state_dict，不要重新保存。

代码按照 RadImageNet 官方 PyTorch notebook 的 `Backbone(nn.Sequential(...))` 格式进行严格加载。

---

# 4. 模型 2：RadioDINO-S/16（医学 Transformer，主 target）

## 4.1 为什么用它

RadioDINO-S/16 是医学域自监督 ViT-S/16：

- ViT-Small
- patch size = 16
- hidden dimension = 384
- 约 21.7M 参数
- 在 RadImageNet 约 135 万 CT/MRI/超声图像上进行医学自监督预训练

这使其和通用 DINO ViT-S/16 在架构上高度匹配，适合做公平的 medical-vs-general pretraining 对照。

模型主页：

https://huggingface.co/Snarcy/RadioDino-s16

文件列表：

https://huggingface.co/Snarcy/RadioDino-s16/tree/main

## 4.2 需要下载的文件

只需要手工下载：

```text
config.json
model.safetensors
```

放到：

```text
model/radiodino_s16/
├── config.json
└── model.safetensors
```

注意：`model.safetensors` 应该是几十 MB 级别（约 87 MB），不是几十或几百字节。

如果你通过 git clone 得到的是约 100 多字节的 Git-LFS/Xet 指针文件，说明权重并没有真正下载，必须从网页点击 Download 或使用完整的 Hugging Face 下载工具取得真实文件。

---

# 5. 模型 3：ImageNet-ResNet50（通用 CNN 对照）

为了让医学 CNN 与通用 CNN 保持相同架构，使用 TorchVision ResNet50 ImageNet-1K V2 权重。

官方权重地址：

https://download.pytorch.org/models/resnet50-11ad3fa6.pth

下载后放到：

```text
model/imagenet_resnet50/resnet50-11ad3fa6.pth
```

不要修改文件名。

该权重约 98 MB。

---

# 6. 模型 4：DINO ViT-S/16（通用 Transformer 对照）

使用和 RadioDINO-S/16 相同的 ViT-S/16 架构，只将预训练域改为 ImageNet。

Hugging Face / timm 官方镜像：

https://huggingface.co/timm/vit_small_patch16_224.dino

文件列表：

https://huggingface.co/timm/vit_small_patch16_224.dino/tree/main

手工下载：

```text
config.json
model.safetensors
```

放到：

```text
model/dino_vits16/
├── config.json
└── model.safetensors
```

`model.safetensors` 应约 87 MB。

---

# 7. 为什么现在需要 4 个模型

当前论文设计是一个 2×2 因子实验：

| CNN source | Transformer target | 缩写 | 作用 |
|---|---|---|---|
| RadImageNet-ResNet50 | RadioDINO-S/16 | MM | 主模型：医学→医学 |
| RadImageNet-ResNet50 | DINO ViT-S/16 | MG | 医学 CNN + 通用 Transformer |
| ImageNet-ResNet50 | RadioDINO-S/16 | GM | 通用 CNN + 医学 Transformer |
| ImageNet-ResNet50 | DINO ViT-S/16 | GG | 完全通用对照 |

这样可以分别研究：

1. CNN 前端的医学预训练是否有贡献；
2. Transformer 后端的医学预训练是否有贡献；
3. 两者同时医学预训练是否有协同效应；
4. 医学预训练的收益是否只发生在 MRI/CT/X-ray，而不一定发生在 OCT/病理。

相比之前使用不同 CNN/Transformer 架构的比较，这个设计更干净。

---

# 8. 代码中的 stage 定义

两个 CNN 都是 ResNet50，因此 stitch source 完全一致：

```text
s1 = layer2: 28×28×512
s2 = layer3: 14×14×1024
s3 = layer4:  7× 7×2048
```

两个 Transformer 都是 ViT-S/16：

```text
224 / 16 = 14
patch grid = 14×14
hidden dimension = 384
12 transformer blocks
```

因此最自然的拼接位置是：

```text
ResNet layer3
14×14×1024
       ↓
1×1 projection / MLP
       ↓
14×14×384
       ↓
ViT tail
```

代码仍会自动筛选：

```text
source stage = 1 / 2 / 3
target cut   = block 3 / 6 / 9
```

每个 source-target pair 共 9 个配置，四组共 36 个 screening 配置。

---

# 9. 第一次运行前检查

环境：

```bash
pip install -r requirements.txt
```

建议 Python 3.11，并先单独安装与你 CUDA 匹配的 PyTorch。

然后执行：

```bash
bash run.sh prepare
```

该步骤不会下载任何模型。

它会生成：

```text
cache/preflight_report.json
cache/data_audit.json
cache/model_audit.json
```

终端应该最终显示：

```text
data ready:   True
models ready: True
```

如果缺失，会明确显示是哪一个：

```text
source medical: MISSING
source general: READY
target medical: MISSING
target general: READY
```

---

# 10. 模型文件检查规则

代码不只检查“文件名存在”。

对于模型权重，必须：

- 文件存在；
- 文件大小至少 1 MB；
- RadioDINO / DINO 的 config.json 必须存在；
- 实际加载时 state_dict 必须与模型结构匹配。

因此 Git-LFS 指针文件不会被误认为真实模型。

RadImageNet 权重采用严格结构检查；不会使用 `strict=False` 悄悄忽略大量不匹配参数。

---

# 11. 推荐第一次只跑缓存

准备完成后：

```bash
GPUS=0 bash run.sh cache
```

成功后应看到：

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

这说明四个 backbone 全部加载正确。

---

# 12. 正式运行

4× RTX 3090：

```bash
GPUS=0,1,2,3 bash run.sh
```

单卡：

```bash
GPUS=0 bash run.sh
```

完整流程：

```text
prepare data manifest
        ↓
validate 4 manual checkpoints
        ↓
cache 2 CNN + 2 ViT features once
        ↓
36 AOSS screening configs
        ↓
每个 MM/MG/GM/GG pair 选 Top-1
        ↓
4 configs × 3 seeds final experiment
        ↓
post-hoc 36-point stitchability map
        ↓
主 MM 模型消融
        ↓
REPORT.md / CSV
```

---

# 13. 加速策略

四个 backbone 在每幅图上只运行一次。

缓存之后：

```text
cached ResNet stage
        ↓
small StitchAdapter
        ↓
ViT tail only
```

不再反复运行完整 ResNet50 和完整 ViT。

四张 3090 使用 experiment-level parallelism：

```text
GPU0 -> config A
GPU1 -> config B
GPU2 -> config C
GPU3 -> config D
```

没有 DDP/NCCL 通信开销。

---

# 14. 磁盘空间

建议至少：

```text
100 GB
```

更稳妥：

```text
150 GB
```

主要用于 BMAD、四个权重和 FP16 feature cache。

---

# 15. 最简准备清单

## 数据

```text
data/BMAD/
├── Brain/
├── liver/
├── RESC/
├── OCT2017/
├── RSNA/
└── camelyon16/
```

## 模型

```text
model/radimagenet_resnet50/resnet50_torch.pt

model/imagenet_resnet50/resnet50-11ad3fa6.pth

model/radiodino_s16/config.json
model/radiodino_s16/model.safetensors

model/dino_vits16/config.json
model/dino_vits16/model.safetensors
```

准备完后：

```bash
bash run.sh prepare
GPUS=0 bash run.sh cache
GPUS=0,1,2,3 bash run.sh
```

即可。
