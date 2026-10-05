# 参数模块拼接新模型：固定方法实验方案

2026-10-03 按用户澄清制定。主线：**设计参数模块拼接方法，组成新模型，实现
零样本异常检测**。具体数值、图表、判断和运行日志仅保存在 results/analysis/。
2026 年十三篇已核验近邻及协议区别见 [PUBLICATION_GAP_PLAN_2026_CN.md](PUBLICATION_GAP_PLAN_2026_CN.md)。

## 方法 v1 与边界

新路径 F(x)=T_suffix(A_phi(C_prefix(x)))。其参数由冻结 CNN prefix、可训练接口
phi 和冻结 Transformer suffix 组成。仅正常 source 上 NFFA 训练 phi；完整 frozen
Transformer R(x) 作为 reference。原 patch 差异 1−cos(F_patch,R_patch)，原 top5%
contrast_topk 不变。新 stitched backbone 与包含 reference 的完整 detector 分别描述。

原四 pair、候选 grid、AOSS、selected_configs、final 预算、224 输入、评分和主结果
全部保留。整目标模态排除，尤其 RESC/OCT2017 一并排除。global 五折平均 AOSS
使本 fold 排除的模态出现在其他 fold 的 source 中；历史选择的信息边界必须披露。
本轮继承该选择，不宣称全局严格 fold-exclusive，不暗改选择协议。

## 本轮实际实现

- **新模型实体。** ComposedAnomalyDetector 注册实际选定 CNN prefix、adapter、
  Transformer suffix 和 reference；suffix 与 reference 共享参数对象。RGB 输入经
  原归一化得到 map/score。bundle 含全部实际权重、位置 buffer、配置和模块哈希，
  严格加载只构造拓扑，不读取原权重路径或下载。报告 unique 参数、suffix 和 reference。
- **original。** 复用四 pair×三 seed×五 folds 的 60 个原 adapter；144 项原
  image AUROC/AP 与 final JSON 重放，容差 1e-6，失败停止解释，不覆盖主结果。
- **matched_tail。** 原结构另训 60 adapter；每 fold 重置 seed，以匹配组件对照。
  原 worker 的 RNG 跨 folds 连续推进；不能宣称该重训与历史 checkpoint 完全相同。
- **no_tail。** 另训 60 adapter，直接预测 reference final features，保留同 adapter
  和冻结输出 LayerNorm，移除全部 Transformer blocks。损失、样本、采样顺序、初始化、
  shuffle 和优化预算与 matched_tail 相同。prefix token 无梯度，有效参数数另报。
  这是真正训练的控制模型，不能用训练后临时拔掉 tail 来替代。
- **untrained_adapter。** 60 个同初始 adapter + pretrained suffix，不执行优化。
  检验正常 source 对齐学习作用；AOSS 只报告，不据此重选。

训练仅正常 source：每模态 1000、四模态共 4000；8 epochs、batch64、AdamW
lr=.001、wd=.0001；原 patch cosine +0.1 global cosine；seed11/22/33。
共 120 次新训练、60 个未训练控制。保存确切 source record indices/paths、预算、
初始 hash、步数、有效梯度参数、loss history 和 checkpoint hash。全部 180 控制
checkpoint 固定并验证 hash 后才统一进行新 target 评价；不按结果挑 arm。

预定主 contrast：matched_tail−no_tail；次 contrast：matched_tail−untrained_adapter。
matched_tail−original 只检验历史 RNG/复现敏感性，不能当作方法改进。
所有 pair/dataset/fold/seed 保留。六 dataset 等权 image macro；pixel 覆盖另列。
要证明真实检测/定位收益，source loss/AOSS 变化只能提供解释。

全部原 60 adapter 做 fresh 正常 source 检查：每非目标 dataset 两张固定图，288 组
live parity 要在 1e-6 内匹配原在线路径。fresh source/target/cache/map 差异另报；
在线一致不代表旧 cache 精确重现。四 pair 的 seed11/brain_mri fold 导出完整 bundle
并检验 round trip。此 source 抽样不能替代全目标图像端到端评价或实际延迟测量。

## 运行和返回

服务器手动 git pull，激活成功 paired-response 的原环境，不自动升级依赖：

```bash
GPUS=0,1,2,3 bash run_method_controls.sh
```

PYTHON_BIN 可指定该环境 Python 绝对路径。GPUS 指定可见 CUDA 索引；不指定时默认
auto 使用全部可见 GPU，单卡可用 GPUS=0；旧 GPU_ID=0 仍兼容。设置了
CUDA_VISIBLE_DEVICES 时，GPUS 使用其重编号后的 0、1 等索引。

调度单位为 pair×seed（共 12 jobs），每张卡同时一个独立进程；该任务内五 folds
及 matched_tail/no_tail/untrained_adapter 同卡依次运行，不改变 batch、epoch、
样本、初始化或随机 seed。模型实体验证、source 训练、target 评估分三阶段，各阶段
全部 12 任务完成才推进；全体 180 控制 checkpoint 验哈希后才开始任何目标评估。
空闲卡动态接下一个任务。任一 worker 失败即停止其余 worker，保留产物和日志；
SIGTERM/中断也走停止与失败记录流程，SIGKILL/断电仍无法保证。
每阶段每任务实际 GPU、设备型号、显存与日志在 workers/ 和 logs/。GPU 算术差异
可能影响浮点结果，原指标仍必须通过既定重放容差。多进程同时占用主机内存及读
cache，资源不足可减少 GPUS，不通过缩小 batch 或训练预算回退。时长依机器而定。
保留原 final/selected、completed mechanism/paired artifacts、cache/train/perturb/test、
目标 masks、完整 CNN/ViT 权重和正常 source 原图。无下载/建 cache/训练补缺回退。
无需先运行 screen-response 网格审计。

原始产物：results/followup/method_controls/；分析：results/analysis/method_controls/。
预期 288 dataset metric rows、240 fold AOSS rows、144 original replay checks，
180 控制 checkpoint 和四 bundle。CPU 分析核对所有输出 hash，独立从 288 prediction
CSV 重算 AUROC/AP、核对身份/顺序/覆盖，生成同 seed contrast 表和图。
多卡调度新增 36 个 worker receipts 和 36 份任务日志；分析同时核验其完整性。

结束下载 `[TRANSFER] results/transfer/method_controls_run_日期时间_后缀.tar.gz`，
放本地 results/transfer/ 并在该目录解压。包含指标、预测、引用、hash、分析和日志；
**.pt、数据、cache 和原权重留服务器，不装入返回包，不进 Git**。返回包不含权重，
本地无法实际重放 bundle；独立权重重放是服务器的成功 receipt，不冒称本地已运行。
本地可重算：

```bash
python -m src.analyze_method_controls --input results/transfer/method_controls
```

成功及常规失败都尝试打包；断电/SIGKILL 不保证。已有产物先调用完整 CPU 验证：
验证通过则仅重生成分析并重新打包，不启动 worker；失败则保留并打包证据，拒绝
覆盖/续跑/重训。原 manifest、指标、预测与全部服务器新权重保留；原数据、代码
和结果文件内容/大数组 stat 均受保护。

返回前确认拿到的是终端 `[TRANSFER]` 指向的 **method_controls_run_*.tar.gz**，
包中应有 method_controls/run_manifest.json、training_completed.json、metrics.json、
predictions/、workers/ 和 logs/。不能只按历史 followup_results.tar.gz 包名认定阶段完整。
可先在本地只读审计压缩包，不解压：

```bash
python -m src.audit_return_package --archive results/transfer/实际方法运行包.tar.gz \
  --require-stage method_controls --output results/analysis/method_return_audit.json
```

缺阶段、失败状态或哈希不符会保留审计 JSON 并非零退出。此命令只检查归档完整性；
解压后的 analyze_method_controls 仍负责训练矩阵、选点协议与指标的科学核验。

## 方法改进与发表研究的后续顺序

1. 先检验新路径、继承 suffix 和正常对齐是否带来真实异常检测收益。如果 no_tail
   同样好，不能把普通特征重建收益归功于 stitching；若均弱，需继续设计方法。
2. 若正常对齐成立而异常分离弱，独立设计正常流形瓶颈/结构约束接口，或 source-only
   配对合成扰动的正常化目标；逐项区分 FoundAD/PDD 已有机制。先在 source-only
   development 固定方案，保留原 v1/AOSS/评分，不反复用已看 BMAD test 搜索。
   新来源目标数据留作确认。本轮没有新增该 loss，也没有把 AOSS 扰动样本拿去训练。
3. 加入 target-free training-free 与 2026 视觉 zero-shot 强基线，及 source-only
   feature reconstruction/单 encoder 适配。辅助异常标签、backbone、分辨率和算力
   单列；原 few-shot/target-normal 论文数字只作不同设置参考。
4. 固定原 map 的 AUPRO/pixel AP、病变大小、背景误报和病例图；独立来源/病种验证，
   有可信 patient/slide ID 后做分组配对 CI。病例/像素重复不能扩大独立患者样本量。
5. AOSS 与真实 AUROC、相对均匀随机 regret 属辅助方法证据；原 screen-response
   入口保留为可选审计。医学 2×2/source/target/interaction、radiology 分层和共同点
   对照用于解释方法；限定为现四组权重条件，避免泛化为所有医学预训练。
6. 同硬件测完整两分支+suffix 推理时延、显存、unique 参数，单列 cache 成本。

本方案不保证发表，也不宣称新训练/强基线/外部验证已经完成。
代码、测试与方法文档同步仓库；具体分析结果、图、日志、权重和运行包保持忽略。
