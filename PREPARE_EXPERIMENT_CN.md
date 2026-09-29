# MedStitch-ZS 实验准备说明（只放原始压缩包）

这版代码已经按你的要求改成：

> **你只负责把 Google Drive 下载下来的原始压缩包放到 `data/` 和 `model/` 根目录。不要解压、不要改名、不要手工整理。其余模型全部由代码通过 ModelScope 自动下载。**

---

# 1. 你真正需要手工下载的只有两类

## 1.1 BMAD 数据 —— Google Drive

BMAD 官方仓库：

https://github.com/DorisBao/BMAD

官方整理后的数据 Google Drive：

https://drive.google.com/drive/folders/1AC-wWZl_K18CWL2eIxUScoSOoxT4IBuw?usp=sharing

BMAD 官方说明显示该 benchmark 包含六个重组数据集：

```text
Brain
liver
RESC
OCT2017
RSNA
camelyon16
```

覆盖：

- Brain MRI
- Liver CT
- Retinal OCT
- Chest X-ray
- Digital histopathology

### 你要做什么

从 Google Drive 下载 BMAD 后，无论浏览器最终给你一个 zip，还是多个 zip，都直接放进：

```text
data/
```

例如：

```text
data/
├── BMAD.zip
```

或者 Google Drive 把大文件夹拆成多个压缩包：

```text
data/
├── BMAD-001.zip
├── BMAD-002.zip
├── BMAD-003.zip
└── ...
```

**不要解压。**

**不要手工建立 `data/BMAD/`。**

**不要修改压缩包文件名。**

程序会自动：

```text
data/*.zip / *.tar / *.tar.gz / *.tgz / *.tar.bz2 / *.tbz2
        ↓
data/_extracted/
        ↓
递归搜索
        ↓
Brain / liver / RESC / OCT2017 / RSNA / camelyon16
        ↓
生成 cache/manifest.jsonl
```

如果压缩包内部还有额外的 Google Drive 外层目录也没有关系，代码递归识别。

---

## 1.2 RadImageNet PyTorch 模型包 —— Google Drive

RadImageNet 官方仓库：

https://github.com/BMEII-AI/RadImageNet

官方 PyTorch pretrained models Google Drive：

https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view?usp=sharing

RadImageNet 官方提供：

- ResNet50
- DenseNet121
- InceptionResNetV2
- InceptionV3

本项目只使用其中的 **ResNet50**。

### 你要做什么

把从 Google Drive 下载到的原始 PyTorch 模型压缩包直接放进：

```text
model/
```

例如：

```text
model/
└── <Google Drive 原始下载文件>.zip
```

文件名是什么都不重要。

**不要解压。**

**不要手工寻找 resnet50_torch.pt。**

**不要重命名。**

代码会自动：

```text
model/*.zip / *.tar / *.tar.gz / *.tgz / *.tar.bz2 / *.tbz2
        ↓
model/_manual_extracted/
        ↓
递归查找 resnet50_torch.pt
        ↓
如果没有精确文件名，则查找名称含 resnet50 的 .pt/.pth
        ↓
自动生成规范路径
model/radimagenet_resnet50/resnet50_torch.pt
```

生成规范文件时优先使用硬链接；如果文件系统不允许硬链接，则自动复制。

因此你不需要知道 Google Drive 包内部的目录结构。

---

# 2. 不需要你手工下载的三个模型

下面三个模型代码会自动通过 ModelScope 下载。

## 2.1 通用 CNN

ModelScope ID：

```text
timm/resnet50.a1_in1k
```

自动保存到：

```text
model/resnet50_a1_in1k/
```

ModelScope：

https://modelscope.cn/models/timm/resnet50.a1_in1k

---

## 2.2 医学 Transformer

ModelScope ID：

```text
microsoft/rad-dino
```

自动保存到：

```text
model/rad_dino/
```

ModelScope：

https://modelscope.cn/models/microsoft/rad-dino

用途：

```text
medical DINOv2-Base target
```

---

## 2.3 通用 Transformer

ModelScope ID：

```text
facebook/dinov2-base
```

自动保存到：

```text
model/dinov2_base/
```

ModelScope：

https://modelscope.cn/models/facebook/dinov2-base

用途：

```text
general DINOv2-Base target
```

---

# 3. 因此你开始实验前只需要这样

仓库目录：

```text
Model-concatenation/
├── data/
│   ├── <BMAD Google Drive 原始压缩包 1>.zip
│   ├── <BMAD Google Drive 原始压缩包 2>.zip   # 如果 Google Drive 拆包
│   └── ...
│
├── model/
│   └── <RadImageNet Google Drive 原始 PyTorch 压缩包>.zip
│
├── cache/
├── results/
└── run.sh
```

就够了。

你不需要提前得到：

```text
data/BMAD/
model/radimagenet_resnet50/
model/rad_dino/
model/dinov2_base/
model/resnet50_a1_in1k/
```

这些目录都由程序产生或自动下载。

---

# 4. 第一次运行

安装环境：

```bash
pip install -r requirements.txt
```

推荐 Python 3.11，并先安装与你 CUDA 匹配的 PyTorch。

然后：

```bash
bash run.sh prepare
```

这一步会：

1. 扫描 `data/` 原始压缩包；
2. 自动解压 BMAD；
3. 自动定位六个数据集；
4. 扫描 `model/` 原始压缩包；
5. 自动解压 RadImageNet；
6. 自动定位 ResNet50 权重；
7. 不下载 ModelScope 模型；
8. 输出准备报告。

生成：

```text
cache/preflight_report.json
cache/data_audit.json
```

如果 Google Drive 两个内容都放对了，而 ModelScope 模型还没下载，正常应该类似：

```text
data ready: True
models ready: False

source medical: READY
source general: MISSING (will auto-download from ModelScope)
target medical: MISSING (will auto-download from ModelScope)
target general: MISSING (will auto-download from ModelScope)
```

---

# 5. 自动下载 ModelScope 模型

执行：

```bash
bash run.sh models
```

自动下载：

```text
timm/resnet50.a1_in1k
microsoft/rad-dino
facebook/dinov2-base
```

保存到：

```text
model/resnet50_a1_in1k/
model/rad_dino/
model/dinov2_base/
```

完成后：

```text
data ready: True
models ready: True
ready_for_full_run: true
```

---

# 6. 也可以跳过 models 步骤

只要你已经把两个 Google Drive 原始包分别放入：

```text
data/
model/
```

就可以直接：

```bash
GPUS=0,1,2,3 bash run.sh
```

程序会按顺序：

```text
自动解压 BMAD
       ↓
自动解压 RadImageNet
       ↓
自动定位 RadImageNet ResNet50
       ↓
ModelScope 自动下载其余三个模型
       ↓
建立 feature cache
       ↓
执行全部实验
```

---

# 7. 推荐第一次仍然分三步

最稳妥：

```bash
bash run.sh prepare
bash run.sh models
GPUS=0 bash run.sh cache
```

如果 cache 成功，再：

```bash
GPUS=0,1,2,3 bash run.sh
```

---

# 8. 自动产生的目录

数据：

```text
data/
├── 你下载的原始压缩包
└── _extracted/
    └── ...
```

模型：

```text
model/
├── 你下载的 RadImageNet 原始压缩包
├── _manual_extracted/
│   └── ...
├── radimagenet_resnet50/
│   └── resnet50_torch.pt
├── resnet50_a1_in1k/
├── rad_dino/
└── dinov2_base/
```

不要删除原始压缩包。

后续重复运行时，程序通过压缩包大小和修改时间识别是否已经完成解压，不会每次重复解压。

---

# 9. 支持的原始压缩包格式

当前自动解压支持：

```text
.zip
.tar
.tar.gz
.tgz
.tar.bz2
.tbz2
```

Google Drive 浏览器下载文件夹通常会生成 zip，因此 BMAD 可以直接使用浏览器下载结果。

---

# 10. 当前实验模型

四组 2×2：

| CNN source | Transformer target | 简称 |
|---|---|---|
| RadImageNet-ResNet50 | RAD-DINO | MM |
| RadImageNet-ResNet50 | DINOv2-Base | MG |
| ImageNet-ResNet50 | RAD-DINO | GM |
| ImageNet-ResNet50 | DINOv2-Base | GG |

其中：

```text
manual Google Drive:
BMAD
RadImageNet-ResNet50 package

automatic ModelScope:
timm/resnet50.a1_in1k
microsoft/rad-dino
facebook/dinov2-base
```

---

# 11. 最简版

你只做：

```text
下载 BMAD Google Drive 原始 zip
→ 放 data/

下载 RadImageNet PyTorch Google Drive 原始压缩包
→ 放 model/
```

然后：

```bash
GPUS=0,1,2,3 bash run.sh
```

其余全部由代码完成。
