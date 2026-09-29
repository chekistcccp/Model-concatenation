# MedStitch-ZS / Model-concatenation

面向 **zero-shot medical anomaly detection** 的跨架构模型裁剪拼接实验框架。当前版本已经加入医学预训练 controlled study：

- CNN source：DINOv3 ConvNeXt-Tiny（通用视觉）
- Primary medical target：RAD-DINO / DINOv2 ViT-B/14（医学自监督，胸片域）
- General control target：DINOv3 ViT-S/16

核心研究问题不再只是“CNN+Transformer 能否拼接”，而是：**医学域预训练是否改变 cross-architecture stitchability，并能否把这种表示失配转化为 zero-shot anomaly signal？**

项目按 1–4 × RTX 3090 (24 GB) 优化。最多 4 卡采用 experiment-level parallelism，不做 DDP。

## 开始实验前先看这里

**完整的数据与模型准备说明： [PREPARE_EXPERIMENT_CN.md](PREPARE_EXPERIMENT_CN.md)**

推荐准备目录：

    data/BMAD/
      Brain/
      liver/
      RESC/
      OCT2017/
      RSNA/
      camelyon16/

模型固定目录：

    model/
      dinov3_convnext_tiny/
      rad_dino/
      dinov3_vits16/

模型可以手工提前准备，也可以让程序通过 ModelScope 自动下载。

准备检查（不下载模型）：

    bash run.sh prepare

自动下载缺失模型：

    bash run.sh models

单卡先建立缓存：

    GPUS=0 bash run.sh cache

四卡执行全部实验：

    GPUS=0,1,2,3 bash run.sh

准备检查会生成：

    cache/preflight_report.json

详细的 2026 医学预训练模型调研见：docs/MEDICAL_PRETRAINING_REVIEW_2026.md

## 1. 数据

数据由用户手工下载，本项目不会自动下载数据集。推荐使用 BMAD 官方整理后的六个数据集，统一放在：

    data/BMAD/
    ├── Brain/
    ├── liver/
    ├── RESC/
    ├── OCT2017/
    ├── RSNA/              # 也接受 Chest-RSNA / Chest
    └── camelyon16/        # 也接受 camelyon16_256

代码会优先使用 data/BMAD/，同时兼容旧版直接放在 data/ 下的目录。顶层 zip/tar/tgz 仍可自动解压。

## 2. 环境

推荐 Python 3.11。先安装与你 CUDA 匹配的 PyTorch，再执行：

    pip install -r requirements.txt

## 3. 一条命令完成全部实验

4×3090：

    GPUS=0,1,2,3 bash run.sh

单卡：

    GPUS=0 bash run.sh

也支持分阶段断点运行：

    bash run.sh prepare     # 检查数据/模型，不下载模型
    bash run.sh models      # 仅准备/下载模型
    bash run.sh cache
    bash run.sh screen
    bash run.sh final
    bash run.sh stitchmap
    bash run.sh ablation
    bash run.sh report

默认 bash run.sh 会依次执行：

1. 数据审查与 manifest 生成；
2. ModelScope 自动下载所有模型到 model/；
3. FP16 feature cache；
4. 18 个 stitch configurations 的 source-only AOSS screening；
5. medical/general 两个 target 各锁定 Top-2；
6. Top-2 × 3 seeds 正式 leave-one-modality-out zero-shot 实验；
7. 锁定模型后再做全部 stitch points 的 target-test post-hoc stitchability map；
8. medical target 消融实验；
9. 自动汇总 CSV 与 REPORT.md。

## 4. 所有模型均通过 ModelScope 自动下载

当前模型：

    model/
    ├── dinov3_convnext_tiny/
    ├── rad_dino/
    └── dinov3_vits16/

对应 ModelScope repo：

- facebook/dinov3-convnext-tiny-pretrain-lvd1689m
- microsoft/rad-dino
- facebook/dinov3-vits16-pretrain-lvd1689m

代码使用 modelscope.snapshot_download(..., local_dir=...)。下载完成后 Transformers 只从 local_files_only=True 加载。

## 5. 为什么主结果使用 RAD-DINO，但 CNN source 暂时保留 DINOv3 ConvNeXt

RAD-DINO 是真正医学自监督视觉编码器，因此医学预训练应该进入主实验。但它主要由胸片训练，不能假设对 OCT/病理等所有 BMAD 模态都一定更好，所以本项目同时保留通用 DINOv3 target 形成 controlled comparison。

RadImageNet ResNet 等医学 CNN 是很自然的 source 候选，但当前项目要求全部权重通过 ModelScope SDK 自动下载；本轮调研没有验证到稳定可依赖的 ModelScope RadImageNet CNN repo。因此当前版本不通过其他下载源偷偷引入权重。

结果会自动区分：

- radiology：Brain MRI / Liver CT / Chest X-ray
- non-radiology：OCT / Pathology

这样可以直接分析医学预训练的收益是否依赖目标模态。

## 6. RAD-DINO 的分辨率与速度策略

RAD-DINO 是 ViT-B/14，原始公开配置使用 518 级输入。全量 518 会产生约 1369 patch tokens，显著增加 3090 上的 attention 成本。

因此默认大规模实验使用 224：

- RAD-DINO：16×16 = 256 patch tokens，hidden=768
- DINOv3 ViT-S：14×14 = 196 patch tokens，hidden=384

DINOv2 absolute positional embedding 在代码中显式插值到当前输入网格。若主结果成立，建议仅对最终最佳 RAD-DINO stitch configuration 增加 518 分辨率验证，而不是让全部 18 个 screening jobs 都用 518。

## 7. 加速：每张图只解码一次、所有 backbone feature 只提取一次

cache 阶段对同一张原图只解码一次，然后分别应用 source / medical target / general target 的正确 normalization，写入 FP16 .npy memmap：

    cache/features/<dataset>/<split>/
    ├── s1.npy
    ├── s2.npy
    ├── s3.npy
    ├── target_medical.npy
    ├── target_general.npy
    └── records.jsonl

synthetic perturbation 也只生成一次，然后分别缓存两个 target 的输出和相应 patch-grid mask。

之后 18 个 stitching configurations 都不再重复运行 ConvNeXt 或完整 target encoder，只读取 memmap 并运行：

    cached CNN stage -> StitchAdapter -> target tail blocks -> alignment/disagreement

这是实验加速的核心。

## 8. 旧缓存自动失效

如果你曾经运行过只含 DINOv3 的旧版代码，cache 中可能已有 .done。新版会额外检查 target_medical.npy / target_general.npy 和医学 perturbation cache；缺失时会自动判定旧缓存 stale 并重建，不会错误复用。

## 9. Stitching 结构

候选 source stages：

- s1：约 28×28×192
- s2：约 14×14×384
- s3：约 7×7×768

每个 source feature 会自适应 resize 到目标 Transformer patch grid，并通过 1×1 projection + LayerNorm + MLP/Linear adapter 映射到 target hidden dimension。

RAD-DINO target 使用 DINOv2 absolute positional embedding；由于 stitching 绕过了原始 patch embedding，代码会显式插入插值后的 patch position。DINOv3 target 则保留原始 RoPE 逻辑。

候选 target cuts：block 3 / 6 / 9。

总 screening 数：

    2 target backbones × 3 source stages × 3 target cuts = 18

## 10. Strict leave-one-modality-out zero-shot

五个 modality folds：

- brain_mri：Brain
- liver_ct：liver
- oct：RESC + OCT2017
- chest_xray：RSNA
- pathology：camelyon16

当某 modality 为 target 时，adapter 只使用其余 modalities 的 normal train feature。目标模态的 train/valid 图像不参与训练、AOSS 选模或超参数选择。

## 11. NFFA

完整 target encoder 的最终 patch feature记为 Z_T；stitched network 输出为 Z_S。

训练 loss：

    L_patch = 1 - cosine(Z_S, Z_T)
    L = L_patch + 0.1 * L_global

只训练 StitchAdapter 与 prefix tokens；target tail 权重冻结。

## 12. AOSS 与无泄漏选模

对 source-domain normal image 生成 local intensity shift / blur / patch-copy perturbation。

每个 stitch configuration 计算：

    AOSS = local_perturbation_sensitivity / normal_discrepancy

medical 和 general target **分别**按 AOSS 排名并各取 Top-2。

target-test AUROC 不参与 configuration selection。只有配置锁定后，stitchmap 阶段才复用 screening checkpoint 对全部 18 个点做 post-hoc target-test evaluation，用于 AOSS–真实 AUROC 的相关性分析。

## 13. Zero-shot anomaly score

测试时 patch anomaly signal 是 stitched target representation 与完整 target representation 的 cosine disagreement。

默认 image score 使用 spatial contrast + top-5% pooling，以削弱全局 modality shift，突出局部异常。

消融同时测试：

- raw_topk
- contrast_topk
- robust_topk

## 14. 4×3090 并行策略

不做 DDP/FSDP。每张卡独立运行一个 config/fold/seed job：

    GPU0 -> medical_s1_b3
    GPU1 -> medical_s1_b6
    GPU2 -> medical_s1_b9
    GPU3 -> general_s1_b3

任务结束后调度器立即给该 GPU 分配下一个 job。

默认 adapter batch size=64，cache extraction batch size=48，优先保证 RAD-DINO ViT-B tail 在 24GB 3090 上稳定。

## 15. 输出

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

REPORT.md 会自动生成 medical vs general 的 radiology / non-radiology 分组结果，以及每个 target 内 AOSS 与 post-hoc target AUROC 的 Spearman 相关。

## 16. 当前研究解释

当前主方法最好表述为：

> A general CNN local-feature frontend is surgically grafted into a medically pretrained transformer representation space, and cross-architecture compatibility is exploited as a zero-shot anomaly signal.

而 DINOv3 general target 是必要的受控对照。最终论文不应预先宣称 RAD-DINO 一定更好；如果它只在 radiology 上获益、在 OCT/Pathology 上下降，这本身就是有价值的 domain-compatibility 结果。

## 17. 注意

代码已经针对 Transformers 的 DINOv2/DINOv3 接口做了双路径适配，但仓库提交环境没有 RTX 3090/CUDA 与 ModelScope 权重缓存，因此首次在你的服务器运行时仍应先执行：

    GPUS=0 bash run.sh prepare
    GPUS=0 bash run.sh cache

确认模型与 cache forward 正常后再启动 4 卡全流程。出现运行日志后可以继续针对实际 Transformers/ModelScope 版本做兼容修正。