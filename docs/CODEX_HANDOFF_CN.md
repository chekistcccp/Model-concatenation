
# MedStitch-ZS / Model-concatenation：Codex 实验交接文件

> 面向后续 Codex 继续处理实验运行、结果汇总、统计分析与论文图表。
>
> 主线澄清：2026-10-03（按用户最新明确要求）；历史运行记录保留
> 仓库：chekistcccp/Model-concatenation
> 本交接文件创建前代码 HEAD：05b7756ae812098b131b175421addf855bd2231e

---

## 0. 交接目标

本文件汇总此前围绕本项目完成的研究设计、代码实现、数据/模型准备规则、最近真实运行错误与修复，以及后续结果处理建议。

后续 Codex 应优先遵循：

1. 保持当前 strict zero-shot protocol 不变。
2. 先验证实验产物完整性，再做统计。
3. 优先设计和验证由预训练参数模块拼接形成的新零样本异常检测模型。
4. 以同预算组件对照检验拼接、正常对齐和继承后段是否带来真实异常检测收益。
5. medical/general source、target effect、interaction 和 radiology 分层属于方法解释与泛化验证。
6. AOSS 属于方法的 source-only 选择环节；相关性和 regret 是选择有效性的证据。
7. 新方法变体独立记录，原选点、预算、评分、主结果不覆盖；不根据已看 target test 调方法。

不要在没有记录的情况下改变数据划分、target selection protocol、候选 stitch grid 或主评分方式。

---

# 1. 当前研究问题

核心问题：

> 设计一种参数模块拼接方法，将已有预训练模型的部分参数与小型可训练接口组成新模型，在不使用目标模态训练/验证数据的情况下实现零样本异常检测。

此定位依据用户 2026-10-03 的明确澄清，优先于历史章节中将医学预训练效应
列为核心目标的表述。医学/通用参数矩阵用于检验和解释所提出的方法，不能替代
方法贡献。现有 NFFA + CNN-prefix→adapter→Transformer-suffix + reference discrepancy
是方法 v1；还需要直接证明真实检测价值及组件必要性。

“参数拼接”的具体实现是继承并连接可执行网络模块：新路径参数由 CNN prefix 参数、
adapter 参数和 Transformer suffix 参数构成。训练仅更新 adapter。完整异常检测器
仍包含 frozen reference Transformer，用于原差异评分；新 stitched backbone 与完整
detector 的依赖、参数量和推理成本必须分别描述。

zero-shot 仍指当前 leave-one-modality-out 的 adapter 训练边界。历史 global 五折
平均 AOSS 的跨 fold 信息边界见第 38–40 节；不能因主线纠正便宣称全局选择严格
fold-exclusive，也不改变历史选择文件。

当前不是训练传统监督分类器或分割器。

基本结构：

~~~text
frozen CNN source
    ↓
selected intermediate feature
    ↓
small trainable StitchAdapter
    ↓
frozen Transformer tail
    ↓
stitched representation
~~~

同时存在完整 frozen Transformer：

~~~text
input
  ↓
full frozen Transformer
  ↓
reference representation
~~~

在正常 source-domain 图像上训练 adapter，使 stitched final patch representation 接近 full target final patch representation。

测试时 stitched representation 与 full target representation 的 patch-wise disagreement 作为 anomaly signal。

---

# 2. 当前最终 2×2 backbone 设计

## 2.1 CNN source

medical source：

- RadImageNet-ResNet50
- checkpoint：model/radimagenet_resnet50/resnet50_torch.pt
- input：224
- channel order：BGR
- mean：[0.5, 0.5, 0.5]
- std：[0.5, 0.5, 0.5]
- stage channels：[512, 1024, 2048]

general source：

- ModelScope ID：timm/resnet50.a1_in1k
- local：model/resnet50_a1_in1k
- architecture：resnet50.a1_in1k
- input：224
- RGB / ImageNet normalization
- stage channels：[512, 1024, 2048]

CNN stitch stages：

~~~text
s1 = ResNet layer2 = 28×28×512
s2 = ResNet layer3 = 14×14×1024
s3 = ResNet layer4 =  7× 7×2048
~~~

## 2.2 Transformer target

medical target：

- microsoft/rad-dino
- ModelScope 自动下载到 model/rad_dino
- 当前按 DINOv2-Base / ViT-B14 使用
- patch size：14
- hidden dim：768
- 12 blocks
- experiment input：224
- patch grid：16×16
- mean：[0.5307, 0.5307, 0.5307]
- std：[0.2583, 0.2583, 0.2583]

general target：

- facebook/dinov2-base
- ModelScope 自动下载到 model/dinov2_base
- 同为 DINOv2-Base / ViT-B14

## 2.3 四组 pair

| Source | Target | Pair | 作用 |
|---|---|---|---|
| RadImageNet ResNet50 | RAD-DINO | MM | 历史主模型；所提方法的医学参数条件 |
| RadImageNet ResNet50 | DINOv2-Base | MG | 医学 source 对照 |
| ImageNet ResNet50 | RAD-DINO | GM | 医学 target 对照 |
| ImageNet ResNet50 | DINOv2-Base | GG | 通用对照 |

结果分析不要只做 MM vs GG。

source-side medical pretraining effect：

~~~text
MM - GM
MG - GG
~~~

target-side medical pretraining effect：

~~~text
MM - MG
GM - GG
~~~

interaction 可按：

~~~text
(MM - MG) - (GM - GG)
~~~

或等价形式分析。

---

# 3. 数据与 modality

当前使用 BMAD 六个子数据集：

| Dataset | Modality |
|---|---|
| Brain | brain_mri |
| liver | liver_ct |
| RESC | oct |
| OCT2017 | oct |
| RSNA | chest_xray |
| camelyon16 | pathology |

domain group：

~~~text
radiology:
  brain_mri
  liver_ct
  chest_xray

non_radiology:
  oct
  pathology
~~~

RESC 与 OCT2017 在 leave-one-modality-out 中共同属于 OCT modality。

---

# 4. Strict zero-shot / leave-one-modality-out protocol

这是论文最重要的 protocol 之一。

当某 modality 为 target 时：

- target train 不用于 adapter training；
- target valid 不用于 stitch selection；
- target test 不用于 AOSS 或 hyperparameter selection；
- adapter training 只使用其他 modalities 的 normal train images；
- AOSS 只使用其他 modalities 的 normal images + synthetic perturbations；
- stitch 配置由 source-only AOSS 锁定后才读取 target test。

后续结果分析必须明确：

> target test was not used for stitch-point selection.

---

# 5. Adapter training：NFFA

当前训练目标可简称 NFFA（Normal Final Feature Alignment）。

训练路径：

~~~text
cached CNN feature
    ↓
StitchAdapter
    ↓
frozen target Transformer tail
    ↓
predicted final patch representations
~~~

目标是 full frozen target encoder final patch representations。

loss：

~~~text
patch cosine loss
+
0.1 × global cosine loss
~~~

Transformer body frozen，只训练 StitchAdapter。

---

# 6. StitchAdapter

处理流程：

~~~text
CNN fmap
 ↓
adaptive pooling / interpolation
 ↓
target patch grid
 ↓
1×1 projection to target hidden dim
 ↓
LayerNorm + residual MLP
 ↓
add target patch positional embedding
 ↓
prepend target CLS-derived prefix
 ↓
frozen Transformer tail
~~~

adapter 类型：

- mlp：screen/final 默认
- linear：ablation

---

# 7. Candidate stitch grid

source stage：

~~~text
1, 2, 3
~~~

target cut：

~~~text
3, 6, 9
~~~

四个 pair 总计：

~~~text
4 × 3 × 3 = 36 screening configurations
~~~

重要实现细节：

target_block 当前是 Python slicing index，TargetTail 使用 layers[target_block:]。

因此论文中的 block 编号必须在出图/写作前统一 0-based implementation 与人类常用 1-based 表述，避免把代码 block 3 错写成真正的“第 3 层”。

---

# 8. AOSS：source-only stitch selection

synthetic perturbation：

1. local intensity change
2. local Gaussian blur
3. local patch copy

计算：

~~~text
normal_discrepancy = normal image discrepancy

perturb_in  = perturbation-region discrepancy
perturb_out = outside-region discrepancy

local_sensitivity = perturb_in - perturb_out

AOSS = local_sensitivity / (normal_discrepancy + eps)
~~~

目的：

- normal image disagreement 尽量低；
- perturbation region disagreement 相对外部显著提高。

每个 MM/MG/GM/GG pair 独立按 AOSS 排序。

---

# 9. Screening / final 配置

screen：

~~~yaml
source_stages: [1, 2, 3]
target_blocks: [3, 6, 9]
seed: 11
n_per_source_modality: 500
epochs: 4
batch_size: 64
lr: 0.002
weight_decay: 0.0001
adapter: mlp
topk_fraction: 0.05
~~~

final：

~~~yaml
top_configs_per_pair: 1
seeds: [11, 22, 33]
n_per_source_modality: 1000
epochs: 8
batch_size: 64
lr: 0.001
weight_decay: 0.0001
adapter: mlp
topk_fraction: 0.05
~~~

正式 final：

~~~text
4 pair × Top-1 × 3 seeds = 12 jobs
~~~

---

# 10. Anomaly scoring

patch discrepancy：

~~~text
d = 1 - cosine(stitched_patch, full_target_patch)
~~~

当前 score modes：

- raw_topk
- contrast_topk（默认主结果）
- robust_topk

contrast_topk：

~~~text
median = median(d)
z = max(d - median, 0)
score = mean(top-k(z))
~~~

默认 topk_fraction = 0.05。

---

# 11. 当前指标

image level：

- AUROC
- AUPR

有 mask 时：

- pixel AUROC
- pixel AUPR

尚未实现：

- AUPRO / PRO

如果后续论文需要标准 anomaly segmentation 指标，建议增加 AUPRO，但不能用 target validation 调参。

---

# 12. 当前 baseline

当前代码里的 baseline 主要是 sanity baseline：

- target_patch_dispersion
- source_s2_dispersion

它们不是 publication-grade SOTA baseline。

若主结果成立，后续再补强 baseline。不要在当前核心流程尚未跑通时先扩大 baseline 工作量。

---

# 13. 数据下载与自动解压

用户只负责 Google Drive 原始下载文件。

## 13.1 BMAD

原始 zip/tar 直接放 data/。

支持：

- .zip
- .tar
- .tar.gz
- .tgz
- .tar.bz2
- .tbz2

代码自动解压至：

~~~text
data/_extracted/
~~~

并支持嵌套压缩包递归解压。

## 13.2 RadImageNet

Google Drive 下载得到的原始 PyTorch 模型包直接放 model/。

自动解压到：

~~~text
model/_manual_extracted/
~~~

递归寻找：

优先：

~~~text
resnet50_torch.pt
~~~

fallback：

~~~text
filename contains resnet50
extension is .pt or .pth
~~~

之后自动 materialize：

~~~text
model/radimagenet_resnet50/resnet50_torch.pt
~~~

优先 hardlink，失败时 copy。

---

# 14. ModelScope 自动下载

以下模型不要求用户手工下载：

~~~text
timm/resnet50.a1_in1k
microsoft/rad-dino
facebook/dinov2-base
~~~

本地目录：

~~~text
model/resnet50_a1_in1k/
model/rad_dino/
model/dinov2_base/
~~~

bash run.sh models 会下载缺失项。

完整 bash run.sh 在缺失时也会尝试自动补齐。

---

# 15. 最近一次真实运行错误与修复

真实服务器日志显示六个 BMAD 原始压缩包都成功进入自动解压：

~~~text
Brain_AD.zip
Chest-AD.zip
Histopathology_AD.zip
Liver_AD.zip
Retina_OCT2017_AD.zip
Retina_RESC_AD.zip
~~~

但旧版目录识别只接受精确 canonical 名：

~~~text
Brain
liver
RESC
OCT2017
RSNA
camelyon16
~~~

而自动解压目录带哈希与官方后缀，例如：

~~~text
3918150265_Brain_AD
3b257ef473_Liver_AD
~~~

因此旧代码错误报告 Brain、liver 缺失。

这不是数据缺失，而是名称标准化 bug。

## 15.1 已修复官方 BMAD 名称

当前 src/data.py 支持：

~~~text
Brain_AD          -> Brain
Liver_AD          -> liver
Chest-AD          -> RSNA
Histopathology_AD -> camelyon16
Retina_OCT2017_AD -> OCT2017
Retina_RESC_AD    -> RESC
~~~

并处理：

- 10-char extraction hash prefix
- _AD suffix
- - / _ 差异
- extra wrapper directories

相关修复提交：

~~~text
a37db2c85d448abce11a772d0f785cb9ab2eb886
05b7756ae812098b131b175421addf855bd2231e
~~~

## 15.2 split 目录递归

旧逻辑只看 dataset_root 的直接子目录。

现已支持 wrapper 内的最近 train / valid / test 目录。

---

# 16. 数据标签与 mask 识别

normal aliases：

~~~text
good
normal
healthy
~~~

abnormal aliases：

~~~text
ungood
bad
anomaly
anomalous
abnormal
disease
diseased
~~~

mask aliases：

~~~text
anomaly_mask
mask
masks
label
labels
ground_truth
gt
~~~

当前 mask matching 规则：

~~~text
mask stem == image stem
~~~

这是已知风险。

如果真实 BMAD 使用 xxx.png 对 xxx_mask.png，当前可能匹配失败，需要基于 data_audit.json 与实际目录修正。

---

# 17. Feature cache

目标：完整 backbone 每张图只跑一次。

典型：

~~~text
cache/features/<dataset>/<split>/
├── source_medical_s1.npy
├── source_medical_s2.npy
├── source_medical_s3.npy
├── source_general_s1.npy
├── source_general_s2.npy
├── source_general_s3.npy
├── target_medical.npy
├── target_general.npy
├── records.jsonl
└── .done
~~~

perturb cache：

~~~text
cache/perturb/<dataset>/
├── normal_source_medical_s*.npy
├── pert_source_medical_s*.npy
├── normal_source_general_s*.npy
├── pert_source_general_s*.npy
├── normal_target_medical.npy
├── pert_target_medical.npy
├── normal_target_general.npy
├── pert_target_general.npy
├── mask_medical.npy
├── mask_general.npy
├── records.jsonl
└── .done
~~~

dtype：FP16 NumPy memmap。

---

# 18. GPU 调度

目标机器：1–4 × RTX 3090 24GB。

不是 DDP/FSDP。

采用 experiment-level subprocess parallelism：

~~~text
GPU0 -> config A
GPU1 -> config B
GPU2 -> config C
GPU3 -> config D
~~~

当前：

~~~text
cache batch = 32
screen/final batch = 64
~~~

若 OOM，按顺序降低：

~~~text
cache: 32 -> 16 -> 8
screen/final/ablation: 64 -> 32 -> 16
~~~

不要首先改模型、protocol、数据量或 stitch grid。

---

# 19. Pipeline stages

run.sh 支持：

~~~text
all
prepare
models
cache
screen
final
stitchmap
ablation
report
~~~

推荐真实服务器：

~~~bash
git pull
bash run.sh prepare
bash run.sh models
GPUS=0 bash run.sh cache
GPUS=0,1,2,3 bash run.sh
~~~

---

# 20. 各 stage 含义

prepare：

- 自动解压 data/ 原始包
- 识别 BMAD
- 生成 manifest / audit
- 自动解压 model/ 手工包
- 定位 RadImageNet ResNet50
- 检查自动模型状态
- 不主动下载 ModelScope 模型

主要输出：

~~~text
cache/preflight_report.json
cache/data_audit.json
cache/manifest.jsonl
~~~

models：

- 验证手工 RadImageNet
- 自动下载其余三个 ModelScope 模型
- 输出 cache/model_audit.json

cache：

- frozen 2 CNN + 2 Transformer 特征缓存
- synthetic perturbation cache

screen：

- 36 configs
- source-only AOSS
- 默认不读取 target test 做 selection

final：

- 每 pair AOSS Top-1
- 3 seeds
- target test evaluation

stitchmap：

- 36 screening 点 post-hoc target test evaluation
- 只用于研究 AOSS 与真实性能关系
- 不能反过来改变 final selection

ablation：

当前主 MM：

- calibration size
- linear vs MLP
- raw / contrast / robust top-k

report：

- CSV / Markdown 汇总

---

# 21. 结果目录

预期：

~~~text
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
~~~

---

# 22. Codex 收到真实实验结果后的第一步

不要直接看平均 AUROC 写结论。

## 22.1 数据完整性

读取 cache/data_audit.json。

确认每个 dataset：

- train normal > 0
- test normal > 0
- test abnormal > 0
- mask 数量合理
- OCT 两数据集均存在
- path 没有错误指向 wrapper

建议输出：

~~~text
dataset
modality
train_normal
test_normal
test_abnormal
test_with_mask
~~~

## 22.2 模型完整性

读取 cache/model_audit.json。

确认四模型 READY。

任何模型加载不匹配都不能 silently fallback 到随机初始化。

## 22.3 screening 完整性

理论值：

~~~text
4 pair × 9 configs = 36 screen json
~~~

检查：

- 每 pair 是否正好 9 个；
- 每 config 是否含全部 5 target modality AOSS rows；
- AOSS 是否 NaN / inf；
- train loss 是否异常；
- selected_configs 是否每 pair 1 个；
- selection 是否只使用 AOSS。

## 22.4 final 完整性

理论值：

~~~text
4 pair × 3 seeds = 12 final jobs
~~~

检查所有 pair / seed / dataset 是否齐全。

---

# 23. 主结果分析

## Table A：dataset-level main results

行：

- Brain
- liver
- RESC
- OCT2017
- RSNA
- camelyon16

列：

- MM AUROC / AUPR
- MG AUROC / AUPR
- GM AUROC / AUPR
- GG AUROC / AUPR

每项优先：

~~~text
mean ± SD across 3 seeds
~~~

如有病例级 score，再做 bootstrap 95% CI。

## Table B：domain-group summary

分：

~~~text
radiology
non_radiology
overall
~~~

同时给 macro dataset average，避免只做 sample-size-weighted pooled metric。

## Table C：medical pretraining effects

每 dataset / modality 计算：

~~~text
source effect with medical target:
MM - GM

source effect with general target:
MG - GG

target effect with medical source:
MM - MG

target effect with general source:
GM - GG

interaction:
(MM - MG) - (GM - GG)
~~~

优先报告 effect size + uncertainty，不只给 P value。

---

# 24. 统计注意事项

当前只有 6 datasets、5 modalities、3 seeds。

不要把 seed 当作独立病人样本扩大 n。

如果只有 aggregate JSON：

- 不能伪造 patient-level bootstrap；
- 不能做声称为病人级配对的显著性检验。

如要严格统计，建议补 per-image score export：

~~~text
image_path
label
score
dataset
modality
pair
seed
config
~~~

保存到：

~~~text
results/predictions/*.csv
~~~

之后才能规范做：

- paired bootstrap
- ROC CI
- DeLong / permutation 等
- error-case inspection

---

# 25. AOSS 验证

stitchmap 对 36 个 screening configs 做 post-hoc target evaluation。

核心分析：

~~~text
AOSS vs target-test AUROC
~~~

每 pair：

- Spearman rho
- P value
- scatter
- AOSS rank vs AUROC rank
- AOSS-selected Top-1 在真实 AUROC 排名中的位置

建议增加：

~~~text
Top-1 regret =
oracle best AUROC - AOSS-selected AUROC
~~~

这是非常重要的模型选择质量指标。

---

# 26. 建议图表

Figure 1：方法图

CNN stage → adapter → Transformer tail → disagreement。

Figure 2：2×2 main comparison

MM/MG/GM/GG，按 modality。

Figure 3：stitchability heatmaps

每 pair 一张：

~~~text
rows = source stage 1/2/3
cols = target block 3/6/9
cell = target-test AUROC
~~~

同时可做 AOSS heatmap。

Figure 4：AOSS vs AUROC。

Figure 5：source effect / target effect / interaction。

Figure 6：ablation。

---

# 27. 当前代码已知风险

## 27.1 CI 不是完整 runtime validation

GitHub Actions 当前主要是 Python compileall。

它能证明语法通过，但不能证明：

- ModelScope snapshot 结构完全匹配 loader；
- RAD-DINO 的 transformers 内部接口在服务器版本完全一致；
- 3090 24GB 一定不 OOM；
- RadImageNet state_dict 与 wrapper 100% 匹配；
- BMAD 所有内部 label/mask 命名都已覆盖。

第一次 GPUS=0 bash run.sh cache 才是真正 integration test。

## 27.2 RAD-DINO domain bias

RAD-DINO 是医学模型，但主要医学预训练域偏胸片。

不能预设它对 CT/MRI/OCT/pathology 都优于 general DINOv2。

这应作为实验问题而不是先验结论。

## 27.3 224 input 是速度折中

当前 target 统一 224。

如果主结果成立，再考虑最佳 MM 配置补更高分辨率。

不要在 screening 全矩阵阶段扩大分辨率。

## 27.4 mask stem matching

当前 image stem == mask stem。

需要真实 audit 验证。

## 27.5 baseline 仍弱

当前 baseline 只适合 sanity check。

## 27.6 缺少 per-image score export

这是后续做严格统计最值得优先补的功能之一。

---

# 28. Codex 推荐工作顺序

Phase 1：不改实验，先审查

读取：

~~~text
cache/data_audit.json
cache/model_audit.json
results/screen_summary.csv
results/selected_configs.json
results/final_results.csv
results/ablation_results.csv
results/REPORT.md
~~~

Phase 2：统计

建议生成：

~~~text
results/analysis/
├── dataset_summary.csv
├── modality_summary.csv
├── domain_summary.csv
├── pretraining_effects.csv
├── aoss_correlation.csv
├── stitch_regret.csv
└── analysis_report.md
~~~

Phase 3：图表

建议生成：

~~~text
results/figures/
├── main_2x2_auroc.pdf
├── domain_comparison.pdf
├── aoss_vs_auroc.pdf
├── stitchmap_MM.pdf
├── stitchmap_MG.pdf
├── stitchmap_GM.pdf
├── stitchmap_GG.pdf
└── ablation.pdf
~~~

Phase 4：只有主结果成立后扩展

优先级：

1. per-image score export
2. stronger baselines
3. AUPRO
4. higher-resolution MM
5. efficiency profiling
6. optional extra medical backbone

---

# 29. 再次发生 BMAD 识别错误时

不要要求用户手工改名。

优先检查：

~~~text
data/_extracted/
cache/data_audit.json
cache/preflight_report.json
~~~

并增强：

- _normalized_dataset_name()
- _dataset_roots()
- _find_split_dir()
- _label_from_path()
- _mask_map()

项目约束始终是：

> 用户只需把 Google Drive 原始压缩包放到 data/ 和 model/，代码自己适配官方包结构。

---

# 30. 模型加载报错时

RadImageNet：

先看：

~~~text
model/_manual_extracted/
model/radimagenet_resnet50/resnet50_torch.pt
~~~

打印 state_dict key 前若干项。

不要第一时间改成 strict=False。

timm ResNet50：

检查 ModelScope snapshot 中真实权重文件以及 key 与 resnet50.a1_in1k 是否一致。

RAD-DINO / DINOv2：

先验证：

~~~text
AutoModel.from_pretrained(local_dir, local_files_only=True)
~~~

再验证：

~~~text
target.encoder.layer
target.layernorm
target.embeddings.interpolate_pos_encoding
~~~

是否符合服务器 transformers 版本。

---

# 31. 论文叙事建议

如果真实结果支持：

1. cross-architecture stitching 可形成 anomaly-sensitive disagreement；
2. medical pretraining 会改变 stitchability；
3. source-side 与 target-side medical pretraining 作用不同；
4. 作用具有 modality dependence；
5. source-only AOSS 可以在不访问 target test 的情况下预测较优 stitch；
6. 由此形成 training-light / zero-shot target adaptation 的 anomaly framework。

不要先验写：

> medical pretraining is always better

更稳妥的研究表述：

> medical pretraining alters cross-architecture compatibility and anomaly sensitivity in a modality-dependent manner

前提是结果真实支持。

---

# 32. 当前状态摘要

截至本交接文件创建时：

- 2×2 medical/general backbone 设计已实现；
- BMAD Google Drive 原始压缩包自动递归解压已实现；
- RadImageNet 原始模型压缩包自动递归解压和 ResNet50 自动定位已实现；
- 另外三个模型走 ModelScope 自动下载；
- BMAD 官方 *_AD 命名已兼容；
- wrapper split 目录递归识别已兼容；
- 36-point AOSS screening 已实现；
- 每 pair Top-1 + 3-seed final 已实现；
- shared FP16 feature cache 已实现；
- stitchmap / ablation / report 已实现；
- 最新代码已通过 GitHub Python static-check；
- 曾出现一次真实 BMAD 路径识别错误，已依据真实日志修复；
- 尚未确认完整 GPU cache + full final run 已成功结束；
- 后续必须以真实服务器输出为准，不能把设计假设写成实验结果。

---

# 33. 给 Codex 的一句话任务定义

> 先把当前实验完整跑通并验证结果产物；随后围绕 MM/MG/GM/GG 的 source effect、target effect、interaction、radiology/non-radiology 差异，以及 AOSS→真实 AUROC 的预测能力做严格统计与可视化；在主结果成立前不要改变 zero-shot selection protocol。

# 34. 仓库同步约定（用户要求）

- 后续代码、测试、依赖与使用文档修改完成后，提交并推送到远程仓库。
- 不同步分析结果：`results/` 下的报告、CSV、图表及哈希记录保留在本地。
- 原始运行日志、数据、模型、cache 审计及 manifest 不随代码提交。
- 同步失败须明确报告，不把本地提交当作远程同步成功。
- 分析入口为 `python -m src.analyze_results --root .`；补充 manifest 审计为
  `python -m src.audit_followup --root .`；服务器图片重复核验为
  `python -m src.verify_manifest_duplicates`。这些命令不改变实验选择协议。

# 35. 固定配置的下一步诊断

已增加独立 `predictions` 入口，按原 selected_configs 与 final JSON 重放保存的
final checkpoints，导出逐图分数，不改变原训练或 source-only AOSS 排序。
服务器命令、文件要求、指标一致性检查及声明边界见
[PREDICTION_REPLAY_CN.md](PREDICTION_REPLAY_CN.md)。

仅代码、测试与此类使用说明同步仓库；所有 predictions、报告和实际运行证据
仍放在被 Git 忽略的 results/ 下，不能提交。

取回完整 predictions 后，CPU 分析入口为
`python -m src.analyze_predictions --root . --bootstrap 1000`，依赖安装见
`requirements-analysis.txt`。完整性检查通过后生成固定模型下的配对图像区间、
dataset/domain macro、ROC/PR、分数诊断及误例候选；缺少可靠患者 ID 时不能
声称患者级 CI。使用说明继续见 `PREDICTION_REPLAY_CN.md`，不得用诊断指标
重选配置或替换 contrast_topk 主结果。

# 36. 独立补充实验的运行与取回

新增 `diagnostics` 和 `common_stitch` 入口，均绕过主 pipeline 的 preparation
和 selection。前者只重放固定 checkpoint；后者固定使用原锁定点并集 s2b3/s2b9，
复制 12 个原 final JSON，仅补训练缺少的 12 个 job（原 final 预算不变）。
两个点均报告，不能根据 test 改选最优点或替代原主结果。

命令、检查、产物和取回清单见 [FOLLOWUP_EXPERIMENTS_CN.md](FOLLOWUP_EXPERIMENTS_CN.md)。
只新增被忽略的 `results/followup/` 产物，不修改原 selected_configs、screen、final，
不重建 cache 或下载模型。所有代码/文档修改同步仓库；结果、地图、原图、checkpoint、
日志和取回压缩包不能提交 Git。大型缓存的 stat 检查不等价于完整内容哈希。

# 37. 历史探索：评分标定（已暂停）

新增独立 CPU 入口 `python -m src.source_calibration --stage fit/evaluate`（两步分别运行）。
详见 [RESEARCH_REVIEW_V1_CN.md](RESEARCH_REVIEW_V1_CN.md)。保持原主结果、AOSS 和选点不变，
只用非目标 modality 正常 source 分数拟合整体/局部分量的等权经验 CDF 标定。
所有四种分量/融合对照全部报告，不根据 test 选择权重或方法；因方案受既有 test
诊断启发，结果必须标记为探索性，不能声称新的未见测试验证。
输出在忽略目录 results/exploratory/，不得提交。代码/测试/方法说明同步仓库。

用户纠正研究方向后，此探索不再作为改进主线；本节是后续工作记录，不是原研究目的。
不得据此将方法研究预先改写为仅分析负面结果，亦不得替换原 contrast_topk 主结果。

# 38. 恢复原研究主线：机制审计

按第 1、5、8、31 节研究 CNN→Transformer 正常对齐、异常敏感 disagreement 和
source-only AOSS。先验证真实 target 中间层接回 tail 的恒等性、目标缓存一致性、
原 adapter 的正常图像配对优势和梯度路径，再决定需要修复或补充的环节。
入口为 `python -m src.mechanism_audit --check-only` / `--gpu 0`。
见 [MECHANISM_AUDIT_CN.md](MECHANISM_AUDIT_CN.md)。不训练、不更新权重、不重选，
不读取 test/valid cache，原数据与主结果保持不变。真实 GPU 验证必须以取回产物为准。

# 39. 原方法第二环：配对扰动响应

机制审计取回后，下一项为固定 final checkpoint 的 source 配对扰动分解。
用 normal_in/out 分离原有空间差异和扰动净响应，并以原 compute_aoss 独立重放
300 个分量检查。净响应只作机制诊断，不能替代 AOSS 或改变选点。
一键入口为 `bash run_paired_response_audit.sh`；取回 CPU 分析为
`python -m src.analyze_paired_response --input results/transfer/paired_response_audit`。
见 [PAIRED_RESPONSE_AUDIT_CN.md](PAIRED_RESPONSE_AUDIT_CN.md)。
原 zero-shot selection、主评分和结果保留；训练样本成员与扰动类型推断须披露限制。

# 40. 2026 文献对照后：原 screening 全网格配对诊断

固定 final 点配对实验取回后，继续连接第二环（扰动新增响应）与第三环
（source-only AOSS 的 stitch-selection 能力）。检索已核验 2026 主会/期刊原文，
方法与后续优先级见 [PUBLICATION_GAP_PLAN_2026_CN.md](PUBLICATION_GAP_PLAN_2026_CN.md)；
具体结果与发表差距报告留在 results/analysis，不提交 Git。

一键入口 `bash run_screen_response_audit.sh`，只复用 36 原 screening job / 180
checkpoint：原 seed=11、4 epochs/500 normal per source modality，不补训练、不重选。
导出全网格配对分解并重放 900 原 AOSS 分量，随后 CPU 与既有 stitchmap 事后对照。
净响应只作诊断；原 score、grid、预算、selected_configs 和 final 保留。
原 global 五折选点的跨 fold 信息边界继续披露，不能声称本补充实验已修复该问题。
公平基线、组件对照、定位/外部验证仍是后续研究内容，不能称此入口已完成全部。

# 41. 用户澄清后的当前方法主线与下一轮（2026-10-03）

第 1 节已按用户明确研究目的改写：设计参数模块拼接的方法，组成新模型并实现
零样本异常检测。医学 source/target effect、interaction、radiology 分层与 AOSS
诊断用于验证和解释方法，不替代方法增量。第 38–40 节为历史补充记录；第 40 节
的全网格审计保留为可选附属实验，下一轮先运行 `bash run_method_controls.sh`。

固定方案见 [METHOD_PLAN_CN.md](METHOD_PLAN_CN.md)。新增实际注册 CNN prefix、
adapter 和共享 Transformer suffix/reference 的 ComposedAnomalyDetector；完整
bundle 不依赖原权重路径/下载，保留原 RGB 预处理、reference 和 contrast_topk。
新 stitched backbone 与完整双分支 detector 分别报告参数与运行依赖。

四 arm：原 60 checkpoint 重放、matched_tail 60 同预算训练、no_tail 60 同初始
正常 source 训练、untrained_adapter 60 无优化控制。仅 normal source，全部新
控制训练固定后再统一 target 评价；新 AOSS 只报告，不重选。原 final/selected、
原预算与主结果不覆盖。逐 fold 重置 RNG 与历史原 worker 连续 RNG 的差异明确披露。

服务器保留成功 mechanism/paired 产物、原 cache、模型权重、正常 source 原图和
已有目标 masks。原 AOSS global 五折的信息边界保留，不声称已修复为 fold-exclusive。
手动 git pull 后在成功 paired 环境运行；退出打印的 transfer 包放本地
results/transfer/ 并解压。新 checkpoint/bundle .pt 留在服务器，不进返回包/Git。
CPU 接收分析入口 `python -m src.analyze_method_controls --input results/transfer/method_controls`。

已有结果按方法主线重整的 CPU 入口为 `python -m src.analyze_method_evidence`，
输出 results/analysis/method_review_20261003/；仅核验/汇总，不选择或训练。
本轮九篇正式 2026 文献见 PUBLICATION_GAP_PLAN_2026_CN.md。强公平基线、
AUPRO/外部验证、新训练目标与全目标端到端效率尚未完成，不得将单元测试当 GPU 结果。
代码/测试/方法文档同步仓库；数字、报告、图、日志、权重及运行包保持忽略。

# 42. 方法对照多卡独立调度（2026-10-03）

按用户要求，`GPUS=0,1,2,3 bash run_method_controls.sh` 将 pair×seed 独立任务
分配给指定 GPU；每卡同时一个进程，空闲后接下一个任务。未指定 GPUS 时默认
auto 使用全部可见 GPU；GPUS=0 或旧 GPU_ID=0 仍支持单卡。CUDA_VISIBLE_DEVICES
重映射后按可见索引指定，不能把物理卡号混入可见索引。

研究设计、正常 source 样本、seed、初始化、预算、AOSS、评分和原选择全部保留。
同一任务内所有组件对照同卡执行。实体验证完成→全部 source 训练完成并验证
180 个控制 checkpoint→统一 target 评估，阶段之间设全局等待。不能哪个任务
训练完就提前查看 target 并调整未完成训练的任务。

coordinator 独占全局 manifest/summary 写入；worker 按 phase/job 写独立 receipts、
预测、checkpoint 和日志。任一失败停止其余进程，保留失败证据，不自动重训/续跑。
返回包仍排除所有 .pt，新增 workers/ 和 logs/；代码/文档同步，分析结果不提交。

# 43. 旧 common_stitch 计划冲突的定位（2026-10-03）

`run.log` 中 predictions 和 diagnostics 完成后，旧 common_stitch 被已有输入
计划 hash 检查阻止。这不证明 GPU 或新方法调度故障；也不能绕过保护复用来源
不同/不完整的结果。新增差异说明，并让 common_stitch 的 --check-only 执行同一
已有计划、结果和 checkpoint 检查；失败时不写入 manifest，不启动训练。

错误报告可比较的文件哈希键、包版本、Python、job 矩阵和旧/新 plan hash。
无法解释的 plan 差异继续拒绝，不自动删结果、归档或重训。详见
[FOLLOWUP_EXPERIMENTS_CN.md](FOLLOWUP_EXPERIMENTS_CN.md)。当前主线仍运行
run_method_controls.sh；已完整取回的历史共同点结果无需重新运行来推进方法实验。

# 44. 2026 文献复核与返回阶段检查（2026-10-05）

PUBLICATION_GAP_PLAN_2026_CN.md 已扩充为十三篇正式 2026 工作，新增最直接的
CVPR Revisiting Model Stitching In the Foundation Model Era（最终特征匹配、自拼接
容量控制）以及 SubspaceAD、Spatial-FAD、WALDO。不要将模型拼接或最终特征匹配
本身作为首次贡献；方法主张须有正常 source-only 检测收益与继承后段必要性证据。
文献数字不可替代同协议重跑，目标正常/few-shot/无标签目标选择的信息权限单列。

新增只读 src.audit_return_package：检测归档路径、阶段、完成状态和 output_sha256；
--require-stage method_controls 缺失时先保存报告再非零退出。它不替代科学协议
及指标检查，也不解压覆盖原始结果。返回路径见 METHOD_PLAN_CN.md。

run_method_controls.sh 对已有完整方法产物调用 CPU 验证后重新打包，不启动
训练/评估 worker；验证失败保留并停止，仍不允许覆盖/续跑。下载其明确指定的
method_controls_run_*.tar.gz。src.analyze_method_evidence 增加 --followup、--output、
--as-of 参数，便于独立复算不同批次的历史 v1 证据，不写原输入或重新选点。

后续顺序：完整方法四 arm → 同信息权限强基线/自拼接与融合机制控制 → 固定
map 定位/完整推理效率 → 冻结方案后的独立来源验证。发现 no_tail 同样有效时，
需继续发展方法，不能用医学因子分析或 AOSS 诊断替代算法增量。所有数字、图、
报告和审计 JSON 保持 results/ 忽略；代码、测试和方法说明同步仓库。

# 45. 完成组件对照后的原图模型验证（2026-10-05）

最新 method_controls_run 返回阶段必须实际核验，不沿用历史旧包缺阶段的判断。
独立 CPU 验证先检查所有 hash/科学边界并从全部 prediction CSV 重算指标。具体
完整性、异常值、四 arm 数字、逐 dataset 定位和文献差距保存在 results/analysis/。
不能凭 matched_tail 源域 loss/AOSS 上升宣称检测收益，也不能删负对照或低于随机的结果。

服务器 matplotlib 缺失只影响图；analyze_method_controls 现在保存表格和报告后
明确跳过图，audit.json 标记 plots_generated。已有完整方法产物重新运行 launcher
只验证、分析、打包，不重训。archive 审计改为单次流式解压，避免每个 hash 回读 gzip。

新增 run_method_deployment.sh：使用完成 method_controls 的同环境和已冻结四 arm，
全部目标 RGB 图前向与完整 detector 成本，原选点/AOSS/训练/评分不变。每卡一个
pair×seed job，保留原图/旧 cache 差异；原图只做 stat 稳定性检查。输出完全独立。
详见 METHOD_DEPLOYMENT_PLAN_CN.md。返回 method_deployment_run_*.tar.gz，放本地
results/transfer/ 解压，运行 src.analyze_method_deployment；数字/图不进 Git。

预期 288 metric/prediction cells、240 cost rows、12 worker/log、317 output hashes。
成本计完整 RGB tensor 前向，排除解码/传输；定位仍原 16×16 网格，不冒称高分辨率
AUPRO。若实际原图对照仍无后段增量，转入已隔离 target 的 source-only 接口开发，
固定方案后用新外部数据确认；新颖性、容量匹配基线、临床定位和患者统计仍须补足。
