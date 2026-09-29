# 实验准备说明（中文）

本文件只回答一件事：**运行实验前，你需要准备什么，以及放到哪里。**

项目根目录假设为：

    Model-concatenation/

最终推荐目录：

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
    │   ├── dinov3_convnext_tiny/
    │   ├── rad_dino/
    │   └── dinov3_vits16/
    │
    ├── cache/
    ├── results/
    ├── configs/
    ├── src/
    └── run.sh

---

## 一、你必须准备的数据：BMAD

本项目只需要准备 **一套 BMAD 整合数据**。

需要的六个 BMAD 子集：

1. Brain
2. liver
3. RESC
4. OCT2017
5. RSNA
6. camelyon16

推荐全部放在：

    data/BMAD/

因此完整目录应为：

    data/BMAD/Brain/
    data/BMAD/liver/
    data/BMAD/RESC/
    data/BMAD/OCT2017/
    data/BMAD/RSNA/
    data/BMAD/camelyon16/

代码优先使用这个目录。

为了兼容旧数据，如果你已经把它们直接放在：

    data/Brain/
    data/liver/
    ...

代码也仍然可以递归识别，但新实验建议统一使用 data/BMAD/。

---

## 二、BMAD 内部目录最低要求

代码需要：

- train 中的 normal/good 图像，用于 source-domain normal calibration；
- test 中的 normal + abnormal 图像，用于最终评价；
- 如果数据集提供 anomaly mask，则自动做 pixel-level evaluation；
- valid 不是主实验必须项。

推荐 BMAD 风格：

    Brain/
    ├── train/
    │   └── good/
    │       └── img/
    │           └── *.png
    ├── valid/
    │   ├── good/
    │   └── Ungood/
    └── test/
        ├── good/
        │   └── img/
        │       └── *.png
        └── Ungood/
            ├── img/
            │   └── *.png
            └── anomaly_mask/
                └── *.png

也接受没有 img/ 这一层的结构，例如：

    camelyon16/
    ├── train/good/*.png
    ├── test/good/*.png
    └── test/Ungood/*.png

正常类别目录可识别：

    good
    normal
    healthy

异常类别目录可识别：

    Ungood
    bad
    anomaly
    anomalous
    abnormal
    disease
    diseased

mask 目录可识别：

    anomaly_mask
    mask
    masks
    label
    labels
    ground_truth
    gt

支持图像格式：

    .png
    .jpg
    .jpeg
    .bmp
    .tif
    .tiff

---

## 三、数据集目录别名

代码会自动处理以下别名：

    Brain        -> Brain
    liver        -> liver
    RESC         -> RESC
    OCT2017      -> OCT2017

胸片：

    RSNA
    Chest-RSNA
    Chest

都映射为：

    RSNA

病理：

    camelyon16
    camelyon16_256

都映射为：

    camelyon16

但为了避免混淆，推荐最终统一改成：

    data/BMAD/RSNA/
    data/BMAD/camelyon16/

---

## 四、你需要的三个模型

当前实验需要三个预训练模型。

### 模型 1：CNN source

用途：

    CNN front / source network

ModelScope ID：

    facebook/dinov3-convnext-tiny-pretrain-lvd1689m

推荐目录：

    model/dinov3_convnext_tiny/

作用：

- 提取 ConvNeXt stage1/stage2/stage3；
- 与医学 Transformer / 通用 Transformer 进行跨架构 stitching；
- 只在 feature-cache 阶段运行完整模型。

---

### 模型 2：医学 Transformer target（主实验）

用途：

    primary medical target

ModelScope ID：

    microsoft/rad-dino

推荐目录：

    model/rad_dino/

模型性质：

    RAD-DINO
    DINOv2 ViT-B/14
    hidden dimension = 768
    12 transformer blocks
    medical self-supervised pretraining

该模型是当前论文主实验中的医学预训练 Transformer。

---

### 模型 3：通用 Transformer target（控制实验）

用途：

    general-pretraining control

ModelScope ID：

    facebook/dinov3-vits16-pretrain-lvd1689m

推荐目录：

    model/dinov3_vits16/

模型性质：

    DINOv3 ViT-S/16
    hidden dimension = 384
    12 transformer blocks

该模型用于和 RAD-DINO 做受控比较。

---

## 五、模型权重是否必须手工下载？

**不必须。**

推荐方式：

1. 你手工准备 BMAD；
2. model/ 可以一开始为空；
3. 运行：

       bash run.sh models

代码会通过 ModelScope 自动下载缺失模型到固定目录。

也可以直接：

    GPUS=0,1,2,3 bash run.sh

如果模型缺失，完整流程也会自动下载。

---

## 六、如果你希望提前手工准备模型

可以使用 ModelScope Python SDK：

    python - <<'PY'
    from modelscope import snapshot_download

    snapshot_download(
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        local_dir="model/dinov3_convnext_tiny",
    )

    snapshot_download(
        "microsoft/rad-dino",
        local_dir="model/rad_dino",
    )

    snapshot_download(
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        local_dir="model/dinov3_vits16",
    )
    PY

这样之后实验全程只从本地读取模型。

---

## 七、模型目录最低完整性要求

每个模型目录至少必须包含：

    config.json

以及至少一个模型权重文件，例如：

    model.safetensors

或者 safetensors 分片：

    model-00001-of-000xx.safetensors
    model-00002-of-000xx.safetensors
    ...

或者：

    pytorch_model.bin

程序现在会同时检查：

1. config.json 是否存在；
2. 是否存在 .safetensors 或 .bin 权重；
3. 只有两者都满足才视为模型 READY。

因此一个只有空目录或只下载了一半的模型不会被错误识别为已完成。

---

## 八、不需要准备的模型文件

本项目直接使用 AutoModel，不需要 tokenizer。

因此通常不要求：

    tokenizer.json
    tokenizer_config.json
    vocab.json
    merges.txt

项目也没有依赖 AutoImageProcessor，所以图像 normalization 在代码中显式完成。

如果 ModelScope snapshot 自动带有其他文件，可以保留，不需要删除。

---

## 九、推荐准备顺序

### Step 1：clone 仓库

    git clone https://github.com/chekistcccp/Model-concatenation.git
    cd Model-concatenation

### Step 2：安装环境

建议 Python 3.11。

先安装适合你的 CUDA / RTX 3090 的 PyTorch，然后：

    pip install -r requirements.txt

### Step 3：准备 BMAD

放置为：

    data/BMAD/Brain/
    data/BMAD/liver/
    data/BMAD/RESC/
    data/BMAD/OCT2017/
    data/BMAD/RSNA/
    data/BMAD/camelyon16/

### Step 4：先检查，不下载模型

运行：

    bash run.sh prepare

它会生成：

    cache/preflight_report.json
    cache/data_audit.json

并在终端明确显示：

    data ready: True/False
    models ready: True/False

以及：

    missing datasets: ...
    model source: READY/MISSING
    model target:medical: READY/MISSING
    model target:general: READY/MISSING

prepare 阶段不会主动下载模型。

### Step 5：准备模型

如果你没有手工准备：

    bash run.sh models

程序会通过 ModelScope 下载三个模型。

下载后会生成：

    cache/model_audit.json

### Step 6：再次检查

    bash run.sh prepare

理想状态：

    data ready: True
    models ready: True

此时：

    cache/preflight_report.json

中的：

    ready_for_full_run

应该为：

    true

### Step 7：运行全部实验

四卡：

    GPUS=0,1,2,3 bash run.sh

单卡：

    GPUS=0 bash run.sh

---

## 十、如果服务器不能访问 ModelScope

提前在可联网机器执行模型下载，然后完整复制下面三个目录到实验服务器：

    model/dinov3_convnext_tiny/
    model/rad_dino/
    model/dinov3_vits16/

在离线服务器上：

    AUTO_DOWNLOAD_MODELS=0 bash run.sh prepare

然后：

    AUTO_DOWNLOAD_MODELS=0 GPUS=0,1,2,3 bash run.sh

如果某个模型缺失，程序会直接报出缺失的具体目录，不会访问网络。

---

## 十一、AUTO_DOWNLOAD_MODELS 参数

默认：

    AUTO_DOWNLOAD_MODELS=1

含义：

    缺少模型 -> 自动从 ModelScope 下载

强制离线：

    AUTO_DOWNLOAD_MODELS=0

含义：

    缺少模型 -> 直接报错

例如：

    AUTO_DOWNLOAD_MODELS=0 GPUS=0 bash run.sh cache

适合模型已经提前复制到服务器的场景。

---

## 十二、磁盘空间建议

建议项目磁盘至少预留：

    100 GB

更稳妥：

    150 GB

空间主要用于：

1. BMAD 原始数据；
2. 三个模型；
3. FP16 feature cache；
4. perturbation cache；
5. checkpoints/results。

缓存会显著减少 GPU 重复计算，因此不建议为了节省几十 GB 而关闭 feature cache。

---

## 十三、第一次建议不要直接跑全部实验

第一次部署建议：

    bash run.sh prepare

确认数据目录。

然后：

    bash run.sh models

确认模型。

然后先单卡：

    GPUS=0 bash run.sh cache

如果成功生成例如：

    cache/features/Brain/train/s1.npy
    cache/features/Brain/train/s2.npy
    cache/features/Brain/train/s3.npy
    cache/features/Brain/train/target_medical.npy
    cache/features/Brain/train/target_general.npy

说明：

- DINOv3 ConvNeXt
- RAD-DINO
- DINOv3 ViT
- preprocessing
- ModelScope/local loading
- FP16 cache

这一整条链路已经跑通。

之后再：

    GPUS=0,1,2,3 bash run.sh

已有 cache 会自动跳过，不会重算。

---

## 十四、最终你真正需要准备的内容

最简清单：

### 必须手工准备

    data/BMAD/
      Brain/
      liver/
      RESC/
      OCT2017/
      RSNA/
      camelyon16/

### 可以让程序自动准备

    model/dinov3_convnext_tiny/
    model/rad_dino/
    model/dinov3_vits16/

### 不需要手工准备

    cache/
    results/
    logs/

程序会自动创建。

---

## 十五、准备完成后的标准目录

    Model-concatenation/
    ├── data/
    │   └── BMAD/
    │       ├── Brain/
    │       ├── liver/
    │       ├── RESC/
    │       ├── OCT2017/
    │       ├── RSNA/
    │       └── camelyon16/
    ├── model/
    │   ├── dinov3_convnext_tiny/
    │   │   ├── config.json
    │   │   └── *.safetensors / *.bin
    │   ├── rad_dino/
    │   │   ├── config.json
    │   │   └── *.safetensors / *.bin
    │   └── dinov3_vits16/
    │       ├── config.json
    │       └── *.safetensors / *.bin
    ├── cache/
    ├── results/
    └── run.sh

完成这个目录之后，实验入口始终只有：

    GPUS=0,1,2,3 bash run.sh
