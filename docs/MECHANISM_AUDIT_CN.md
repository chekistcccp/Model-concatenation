# 回到研究主线：NFFA 与 TargetTail 机制审计

目标来自原交接第 1、5、8、25、31 节：验证跨架构对齐产生异常敏感 disagreement
的前提，再解释医学 source/target 的作用和 AOSS 选点能力。本轮不引入新评分、
新 backbone 或新的选择指标；先排查原方法是否按设计工作。

## 检验顺序与解释

1. **Target→Target 恒等重放**：在正常 train 图像上运行完整真实 Transformer，
   用 pre-hook 捕获代码 cut 3/6/9 的实际层输入，再交给原 TargetTail。
   输出应还原完整模型的 final-normalized patch representations。
   检查 prefix、cut 边界、末层归一化和实际 transformers 接口。固定容差为
   `atol=rtol=1e-3`；失败保留证据并停止，不放宽容差或自动改模型。
   cut 3 对应保留人类编号第 4 层起，不能写成“第 3 层”。
2. **目标缓存一致性**：每 dataset 固定取两个正常 train 图像，用原预处理即时提取，
   对照 cached target features，报告最大绝对差和 cosine loss。
   原 cache batch 与当前 batch 不同，数值差须结合精度分析；不自动重建缓存。
   这一步不声称核验 CNN cache，也不证明旧 cache 生成时图片从未变化。
3. **NFFA 对应关系**：原 12 个 final job / 60 个 checkpoint，每个 held-out fold
   只读其他 modalities 的正常 train 特征。每 dataset 固定 16 个均匀索引样本，
   比较真实配对与 dataset 内循环错配的 patch/global/总 NFFA loss。
   若 matched 没有优于错配，需要检查图像对应关系、信息保留或表示退化；
   单一小子集不能证明完整收敛或异常检测有效。可能与训练图像重叠，不能叫独立验证集。
4. **梯度路径**：每 checkpoint 用首个 source dataset 的两个正常样本执行一次
   backward，验证 adapter 梯度有限且非零、所有 backbone 冻结且无梯度；
   不创建 optimizer、不做参数更新，之后清空 adapter 梯度。
5. **AOSS 分量核算**：核对原 final 保存的分量和比值公式，输出分母、局部敏感性、
   AOSS 的完整表；不是重新计算 AOSS、不是替换 source-only 选点。

整体接口检查覆盖六个 dataset，但不训练/拟合任何参数；对每个 checkpoint 的
NFFA 和梯度检查严格排除其 held-out modality，OCT 同时排除 RESC 与 OCT2017。
不读取 test/valid cache 或 target 逐图分数。为核对身份会读取原 final JSON，
其中已有 test 汇总值不参与样本、配置或任何参数的选择。

## 服务器运行

保留原实验环境，不升级依赖。先更新代码，再运行独立命令：

```bash
git pull --ff-only
python -m src.mechanism_audit --check-only
python -m src.mechanism_audit --gpu 0
```

单 GPU 顺序运行、最多 16 张缓存特征一个 batch；不需重跑 all/final/screen。
需要原正常 train cache、原 final checkpoint、两种 target 本地权重，以及所抽取
正常 train 原图；不下载、重建或回退训练。check-only 不写文件、不加载真实模型。
小型 CPU 单元测试不能替代服务器真实权重、FP16 和当前环境的运行验证。

结果目录：`results/followup/mechanism_audit/`，包含 interfaces.csv、alignment.csv、
gradients.csv、aoss_components.csv 和 run_manifest.json。源码、checkpoint、原图、
配置/records 记录 SHA256，大型 cache/权重只检查 size/mtime，结束再次核对。
已有 run_manifest 时拒绝覆盖；若失败先取回已有输出，不删除后盲目重跑。

打包并取回（失败时也取回）：

```bash
mkdir -p results/transfer
tar -czf results/transfer/mechanism_audit.tar.gz -C results/followup mechanism_audit
```

下载到本地 `results/transfer/`，在该目录解压，得到
`results/transfer/mechanism_audit/run_manifest.json` 等文件。结果不提交 Git。

## 下一步决策

- 接口恒等失败：先修复可重现实现问题，明确旧结果受影响范围，保持同一协议重跑必要部分。
- 接口通过但缓存不一致：排查预处理、权重、精度与历史环境，不能先修改评分掩盖问题。
- 缓存一致但配对优势弱：研究 NFFA 是否只学到平均表示，以及 source/target medical
  pretraining 是否改变这一现象；补训练过程观测须另留结果，不覆盖原实验。
- 正常对齐成立后，结合已取回的 source 扰动分量和真实异常分数分析敏感性与 AOSS；
  仍报告全部 pair 和 modality，不把解释性分析用于重选 stitch。

此前 ECDF 评分融合暂停，不是本轮研究路线。不能把原方法研究预先改写为负面结果论文，
也不能预设 MM 必须胜出；由上述验证决定下一步需要修复或检验的环节。
