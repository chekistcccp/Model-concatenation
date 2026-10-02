
# MedStitch-ZS / Model-concatenation：Codex 实验交接文件

> 面向后续 Codex 继续处理实验运行、结果汇总、统计分析与论文图表。
>
> 更新时间：2026-09-30
> 仓库：chekistcccp/Model-concatenation
> 本交接文件创建前代码 HEAD：05b7756ae812098b131b175421addf855bd2231e

---

## 0. 交接目标

本文件汇总此前围绕本项目完成的研究设计、代码实现、数据/模型准备规则、最近真实运行错误与修复，以及后续结果处理建议。

后续 Codex 应优先遵循：

1. 保持当前 strict zero-shot protocol 不变。
2. 先验证实验产物完整性，再做统计。
3. 优先分析 medical/general pretraining 的 source effect、target effect 和 interaction。
4. 区分 radiology 与 non-radiology。
5. 验证 AOSS 是否能在不访问 target test 的情况下预测较好的 stitch。
6. 在真实结果成立后，再决定是否扩展分辨率、baseline 或模型矩阵。

不要在没有记录的情况下改变数据划分、target selection protocol、候选 stitch grid 或主评分方式。

---

# 1. 当前研究问题

核心问题：

> 医学域预训练是否会改变 CNN→Transformer 的跨架构 stitchability，并且这种变化能否被利用为 zero-shot medical anomaly signal？

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
| RadImageNet ResNet50 | RAD-DINO | MM | 主模型 |
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
