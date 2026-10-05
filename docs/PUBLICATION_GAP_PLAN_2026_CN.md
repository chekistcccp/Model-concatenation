# 参数拼接零样本异常检测方法：2026 文献对照

检索核验日期：2026-10-05（保留 2026-10-03 的研究计划）。此文件记录文献与方法计划；具体实验数字、
图表、完整性检查和发表差距判断保存在被忽略的 results/analysis/ 下。
继续遵守 CODEX_HANDOFF_CN.md 第 1、4、5、8、25、31、38、39 节。

按用户 2026-10-03 澄清，研究目的是设计参数模块拼接方法，让已有预训练模型
组成一个能进行零样本异常检测的新模型。医学预训练效应是方法验证与解释。
当前 NFFA + CNN prefix→adapter→Transformer suffix + reference discrepancy 是方法 v1。
后续首要实验改为新模型实体与同预算组件对照，见 [METHOD_PLAN_CN.md](METHOD_PLAN_CN.md)。
保留原 selection、训练预算和评分，不以已看 target test 调方法。

## 已核验的 2026 主会和期刊工作

会议年份与初次 arXiv 年份分开核实。未把 ICLR 投稿、workshop 或预印本当作主会接收。

|工作与一次来源|设置与相关性|对本研究的启示（作者判断）|
|---|---|---|
|[Revisiting Model Stitching In the Foundation Model Era](https://openaccess.thecvf.com/content/CVPR2026/html/Mai_Revisiting_Model_Stitching_In_the_Foundation_Model_Era_CVPR_2026_paper.html)，CVPR 2026，41342–41351；[全文](https://arxiv.org/html/2603.12433v3)|冻结异构 VFM，通过小接口继承前后模块；最终特征匹配后再做任务训练，并以相同接口的 self-stitch 控制容量|这是最直接的拼接近邻；“异构参数模块+最终特征对齐”已有先例。当前方法贡献需落在正常 source-only 异常检测、异常敏感性、继承 suffix 的必要性和 source-only 选点的实际价值。其分类/分割任务训练不能直接移入本项目|
|[PDD](https://openaccess.thecvf.com/content/CVPR2026/html/Lu_PDD_Manifold-Prior_Diverse_Distillation_for_Medical_Anomaly_Detection_CVPR_2026_paper.html)，CVPR 2026；[全文](https://arxiv.org/html/2603.07142v1)|以异构冻结教师、流形对齐和双学生处理医学异常；使用目标数据集正常训练，不能与跨模态 zero-shot 数字直接比较|异构编码器加特征差异已有相近工作；需要证明 stitch/tail、医学 source/target 和 source-only 选点的独立作用|
|[VisualAD](https://openaccess.thecvf.com/content/CVPR2026/html/Hou_VisualAD_Language-Free_Zero-Shot_Anomaly_Detection_via_Vision_Transformer_CVPR_2026_paper.html)，CVPR 2026；[全文](https://arxiv.org/html/2603.07952v1)|冻结视觉 Transformer 中学习正常/异常 tokens；用工业辅助数据监督，跨工业/医学数据集评估|视觉-only、冻结 backbone、轻量模块本身不够构成新颖性；适合作为外部 zero-shot 比较，但辅助异常标签与分辨率必须披露|
|[AnomalyVFM](https://openaccess.thecvf.com/content/CVPR2026/html/Fucka_AnomalyVFM_--_Transforming_Vision_Foundation_Models_into_Zero-Shot_Anomaly_Detectors_CVPR_2026_paper.html)，CVPR 2026；[全文](https://arxiv.org/html/2601.20524v2)|以合成正常/异常图和掩码监督低秩适配，将 VFM 变为 zero-shot detector，比较多个 backbone、数据与适配策略|合成扰动响应不能替代真实病变泛化；外部比较须披露合成异常监督，不能直接加入原 normal-only 训练|
|[Grounding Functional Similarity by Invariance-Aware Model Stitching](https://proceedings.mlr.press/v306/athanasiadis26a.html)，ICML 2026|指出前向可拼接不等于共享功能/不变性，以前向和反向兼容性检验表征|正常 NFFA loss 低或 matched 优于 shuffled 仅说明对齐；异常敏感性、输入依赖性及检测价值仍须独立证据。原 ICLR 投稿版本不另算一篇接收论文|
|[FoundAD](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9e47a0bc530cc88b09b7670d2c130a29-Abstract-Conference.html)，ICLR 2026；[正式全文](https://openreview.net/pdf?id=YRrlJ8oVEH)，[代码](https://github.com/ymxlzgy/FoundAD)|冻结视觉编码器、训练非线性投影将合成异常特征映回正常流形，以残差检测；目标正常样本 1/2/4-shot。预印本首发 2025|与轻量接口+差异检测很接近。必须证明跨架构继承的 suffix 的作用；其 few-shot 数字不能放入严格 target-free 主表。source-only 迁移对照须独立固定方案|
|[DNP-ConFormer](https://papers.miccai.org/miccai-2026/0296-Paper0431.html)，MICCAI 2026；[全文](https://papers.miccai.org/miccai-2026/paper/0431_paper.pdf)|原型引导重建、EMA 编码器与多样性对齐；使用数据集正常训练，报告重复运行及定位指标|公开评审明确关注异常图依据、backbone 公平性、基线覆盖和泛化范围；这些是实验设计依据，不是所有 venue 的强制清单|
|[AUCp](https://pubmed.ncbi.nlm.nih.gov/42009338/)，IEEE TMI 2026，45(7):3720–3733，DOI 10.1109/TMI.2026.3684946；[全文](https://arxiv.org/html/2606.08742v1)|利用正常训练数据与无标签混合评估数据选择模型，并研究代理指标与真 AUC 的关系|source-only AOSS 需要相关性、regret 与简单比较，不能只展示公式；AUCp 使用无标签评估分布，不能直接替代禁止 target 数据参与选择的 AOSS|
|[UniTransAD](https://pubmed.ncbi.nlm.nih.gov/42430321/)，IEEE TMI 2026，45(9):4876–4891，DOI 10.1109/TMI.2026.3711975；[作者代码](https://github.com/zhibaishouheilab/UniTransAD)|脑 MRI 翻译与多层差异检测，Brain-OmniA 汇集多个病理与序列来源。这里只依据摘要和作者仓库，不声称已读完付费全文|多模态 BMAD 并不自动证明外部医院或病种泛化；需独立来源与病例标识。其设置也不是当前 leave-one-modality-out|
|[Q-Former Autoencoder](https://openaccess.thecvf.com/content/WACV2026/html/Dalmonte_Q-Former_Autoencoder_A_Modern_Framework_for_Medical_Anomaly_Detection_WACV_2026_paper.html)，WACV 2026；[补充材料](https://openaccess.thecvf.com/content/WACV2026/supplemental/Dalmonte_Q-Former_Autoencoder_A_WACV_2026_supplemental.pdf)|冻结 DINO/DINOv2 等特征，经瓶颈及重建产生医学异常信号；补充材料明确四个 BMAD 数据集分别训练模型|已覆盖冻结特征、轻量桥接和多尺度重建；可参考重建/瓶颈消融与定位证据。原始数值只能作为目标域正常训练的参考，不能填入 zero-shot 主表|
|[SubspaceAD](https://openaccess.thecvf.com/content/CVPR2026/html/Lendering_SubspaceAD_Training-Free_Few-Shot_Anomaly_Detection_via_Subspace_Modeling_CVPR_2026_paper.html)，CVPR 2026|冻结 DINOv2、用少量目标正常图拟合 PCA，或在 batched zero-shot 设置使用目标批次，按子空间残差检测|需要简单统计/单 encoder 控制。原 few-shot 和目标批次设置均不能当严格 target-free；source-only PCA 迁移对照需另立固定方案|
|[Spatial-FAD](https://papers.miccai.org/miccai-2026/0135-Paper3052.html)，MICCAI 2026，Bridging Vision Foundation Model Priors with CLIP for Spatial-aware Few-shot Anomaly Detection in Medical Images|CLIP 与 DINOv3 空间先验、适配器和原型记忆；支持集含目标正常/异常及 lesion mask，属于 few-shot|医学双模型互补已有近邻；需区分拼接和简单融合，并匹配支持数据、滑窗预算、参数与推理成本|
|[WALDO](https://papers.miccai.org/miccai-2026/1147-Paper0474.html)，MICCAI 2026，Wasserstein-Aligned Localisation for VLM-Based Distributional OOD Detection in Medical Imaging|无优化训练，但依赖健康参考池、DINOv3 检索及 VLM API，主要研究病灶定位|training-free 不等于无目标参考。关注病变大小、假阳性、完整推理成本；其定位指标不能直接替代本项目 image AUROC|

以上来源的原文与公开代码确认发表/实验设置；“本研究需要什么”是对照后提出的
研究判断，并非论文作者给本项目的指令。文献没有给出统一发表 AUROC 门槛。

### 2026-10-05 复核：创新主张与复现边界

以上共十三篇正式 2026 主会/期刊工作。另有直接相关的
[BSMAD](https://openaccess.thecvf.com/content/CVPR2026W/Med-Reasoner/html/Lin_BSMADBridging_Semantic_and_Structural_Manifolds_for_Robust_Cross-Modality_Medical_Anomaly_CVPRW_2026_paper.html)
为 CVPR 2026 **Workshop**，不计入主会；完整监督协议仍需另核验。

不能将“首次异构模型拼接”“首次最终特征对齐”作为当前创新主张。
Mai 等的最终特征匹配与当前 NFFA 接近，接口和数学形式的差异应逐项说明。
要主张新零样本异常检测方法，需证明正常对齐如何保留异常差异，以及 inherited
suffix 相对 direct reconstruction、未训练接口、同 encoder 自拼接/投影、简单
双 encoder 融合的增量。后三类仍需独立固定对照方案；本轮没有擅自扩展训练矩阵。
原 matched_tail/no_tail/untrained_adapter 首先提供可检验的组件必要性证据。

PDD [作者仓库](https://github.com/OxygenLu/PDD) 本次只有页面/README 等，未发现
可运行训练实现；不能安排成即时复现基线。DNP-ConFormer 正式版使用 OCT2017、
APTOS2019、ISIC2018 和 in-house BrainMRI，五次运行；其定位指标来自私有 BrainMRI，
仓库 Br35H 脚本不能冒充该正式数据。UniTransAD 的 tune.py 使用 validation
选择 checkpoint/后处理，不可直接移入原 AOSS。VisualAD 的辅助真实异常/掩码与
AnomalyVFM 的合成异常/掩码都要披露，不能当 normal-only 同预算组件控制。

## 可选附属实验：原 screening 网格的机制—选择分析

本节入口保留为方法解释，不是本次方法研发的首要下一轮。新的首要入口为
`bash run_method_controls.sh`；不要求先完成本节审计。

新增独立入口，不经过原 pipeline：

```bash
bash run_screen_response_audit.sh
```

先手动 git pull，在已成功运行 paired-response 的原环境执行。
默认 GPU_ID=0；可用 `GPU_ID=1 bash run_screen_response_audit.sh`。
PYTHON_BIN 可指定该环境的 Python 绝对路径；脚本不安装依赖或自动切换环境。

前置文件必须保留：

- configs/experiment.yaml 与 results/selected_configs.json；
- results/screen/ 的原 36 JSON 和 checkpoints/ 下 180 个原检查点；
- cache/perturb/ 六个数据集的 normal/pert source s1/s2/s3、target、mask 和 records；
- model/rad_dino 与 model/dinov2_base；
- results/followup/paired_response_audit/ 的完整产物；
- results/stitchmap/ 的原 36 JSON，仅在 GPU source 审计完成后由 CPU 分析读取。

预检查核验原网格、seed=11、screen 4 epochs/500 normal per source modality、
冻结目标版本、前置哈希和缓存。旧 checkpoint 没保存历史训练预算/样本清单，
当前配置和产物身份检查不能补造历史证明；重放原 AOSS 是额外一致性证据。
缺少检查点时停止，不训练或用 final 8 epochs/1000 的检查点替代。

计算全部 36 点 × 5 held-out folds。每 fold 只用其非目标 source modalities，
OCT 同时排除 RESC/OCT2017；在原 normal/pert/mask 上导出配对响应。
预计 221184 行、180 个 source CSV、900 项原 AOSS 分量重放检查。
固定 batch=256、容差 1e-6；任何失败保留证据，不放宽容差。
原 AOSS 的 batch/区域加权与 per-image 配对诊断权重有别，分别保存。

CPU 汇总依次按 dataset→source modality→fold 等权，并保留三类推断扰动分层。
负响应完整保留，区域缺失为空值；单个 screening seed 不报告 seed SD 或患者 CI。
对原已有 stitchmap 仅做事后关联：

- 每 pair/held-out modality 的 AOSS、正常 discrepancy、原局部敏感性、正常空间基线、
  净响应、contrast score 增量与真实 AUROC 的九点 Spearman；
- overall、radiology/non-radiology 的 target-modality 等权事后汇总；与六 dataset
  等权主表的权重不同，主结果不变；
- 原锁定配置与九点均匀随机配置的精确期望表现/regret，及仅作上界的 oracle；
- 全九个共同 stitch 位置的 source/target 条件效应与 interaction；
- 净响应与真实 AUROC 散点，所有 pair 与 fold 都保留。

不拟合代理权重，不输出新 selected_configs，不将净响应升级为选点指标。
原 global 五折平均 AOSS 的边界继续披露：本 fold 排除的 modality 会出现在其他
fold 的 source 数据中；没有使用 target test labels，不等于全局选择严格 fold-exclusive。
原主结果保留，不把该边界悄悄修成新协议。

原始输出：results/followup/screen_response_audit/。
分析输出：results/analysis/screen_response_audit/。
退出后打印 `[TRANSFER] results/transfer/screen_response_run_日期时间_后缀.tar.gz`；
成功与常规失败均尝试打包。断电/SIGKILL 不保证打包。
下载到本地 results/transfer/，在该目录解压；压缩包还含 launcher 日志和成功汇总。
取回后可在本地复算：

```bash
python -m src.analyze_screen_response --input results/transfer/screen_response_audit
```

已有输出时拒绝覆盖/续跑。失败时取回包先诊断，不删除 manifest 后盲目重跑。
本地没有服务器完整 cache/target 权重，本轮代码测试不等价于真实 GPU 检查。

## 后续研究内容与顺序

1. **第二环和第三环的连接（可选附属入口）。** 检查同预算全网格的 source 净响应、
   正常对齐与真实异常表现是否相关，AOSS 是否优于随机期望。失败也保留全部结果；
   分辨是代理选点失效、正常对齐与异常分辨脱钩，还是原评分的局部化限制。
2. **机制对照。** 预先固定未经训练 adapter、图像对应打乱/空间打乱、无 tail 的
   直接 feature reconstruction 对照；在正常 source 上先检验必要性，再按既有 target
   评价。现有 linear/MLP 消融训练预算与 final 有差异，不能当作同预算组件因果对照。
   新增同预算 matched_tail/no_tail、untrained_adapter 与原模型重放入口，详见
   METHOD_PLAN_CN.md；必须以完整取回 GPU receipts 为准。其他 shuffle/self-stitch 对照仍未新增。
3. **公平基线。** 主比较优先加入无需 target 正常样本的 WinCLIP，以及 VisualAD/
   AnomalyVFM 的公开辅助训练模型。逐项记录辅助异常标签、预训练数据、分辨率、
   backbone、参数和算力；PDD/DNP/QFAE 或 target-normal PatchCore 属不同设置，另列参考。
   source-only PatchCore 可作迁移对照，但须明确 memory 来自非目标模态。
4. **病变定位和误差机制。** 在已有原评分图上补 AUPRO（固定 FPR 范围/连通区域规则）、
   pixel AP、病变大小/正常解剖背景/成像来源分层和预定抽样案例。test 只用于评估，
   不据 F1 最优阈值、分辨率或 score mode 改写主结果。
5. **泛化与统计。** 增加至少一个独立来源或新病种的确认性评价；在原 adapter 冻结且
   该目标模态未参与相应 fold 的前提下运行，不回头调整 AOSS。得到可信 patient/slide ID
   后做分组配对不确定性；切片、patch、跨 fold 重复图像和 seed 不能冒充独立患者。
6. **医学预训练的归因。** 现 2×2 与同点比较能分开 source/target/interaction，
   仍混有预训练数据规模、部位覆盖、优化配方和目标模型归一化差异。先限定为这四个
   公共模型的预训练条件效应；更广泛结论需要匹配配方/初始化和额外医学模型复制。
7. **资源与复现。** 同硬件测实际两编码器+adapter+tail 全链路延迟、显存/参数；
   单独列 cache 时间。以后新增训练保存输入 manifest、预算、环境、权重哈希。

基线接入、外部数据和定位评估仍需独立固定方案，不把整张清单视为已完成。
当前新增方法组件训练/部署验证，须取得并核验真实 GPU 产物。研究主张首先是参数拼接
新模型的零样本异常检测能力；医学效应和选点证据围绕此主张，不替代方法增量。
不能据上述检索宣称“首次参数拼接异常检测”，该优先权需要更广年份/术语检索。
代码、测试和方法计划同步 Git；所有结果/图表/日志/数据/cache/权重保持忽略。

## 组件结果完整后的执行更新（2026-10-05）

新的方法判断必须依据真实取回的四 arm，而不能沿用旧包缺阶段结论。独立分析
保留 dataset 负结果、图像与定位的差别以及三 seed 方向；数值留 results/analysis/。
继承后段不优于 direct reconstruction 时，医学效应或 source AOSS 不能弥补方法
增量缺失。Mai 等 self-stitch 与 ICML 不变性分析所提示的容量/功能性证据仍欠缺。

新增可提交下一实验为全目标 RGB 推理与完整 detector 成本，固定四 arm/原选点/
原评分，详见 METHOD_DEPLOYMENT_PLAN_CN.md。该证据确认真实新模型的表现，不自动
解决新颖性。后续正常流形瓶颈/结构约束需先独立 source-only development，并与
FoundAD/PDD/QFAE 区分，再固定容量基线与新外部数据；不直接把已看 target 改成开发集。
