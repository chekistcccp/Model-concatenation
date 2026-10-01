# 研究复盘与独立改进方案 v1

> 状态：已暂停的历史探索。用户纠正研究方向后，此文不再代表后续实验优先级。
> 原研究目标以 CODEX_HANDOFF_CN.md 第 1、5、8、31 节为准；下一步执行
> [MECHANISM_AUDIT_CN.md](MECHANISM_AUDIT_CN.md) 的原方法机制审计。
> 保留此文和独立代码仅为追溯已做过的探索，不将其结果替换原主实验。

## 研究问题与现有证据

研究问题仍是医学预训练如何改变 CNN→Transformer stitchability，以及这种差异
能否成为 zero-shot medical anomaly signal。保留 MM/MG/GM/GG 因子设计、六数据集、
五种 leave-one-modality-out 分组、三个 seed、冻结 backbone 和正常 source adapter 对齐。
不转向监督分类/分割，不扩模型和分辨率，不用 target test 改选 stitch。

目前主结果接近随机、具有明显模态差异；共同 stitch 对照表明分组效应并非完全由
GG 的层位置不同造成，但效应大小受层位置影响。AOSS 在四 pair 上预测能力不一致。
空间和逐图诊断提示：图内中位数扣除可能移除整体异常信号，正常结构也可产生局部
高 disagreement。合成扰动敏感性未必迁移到真实疾病，AOSS 分母也会影响排序。
这些是可检验的解释，不能写成已经证明的因果机制或普遍优势。

## 本轮实现：source 正常分布标定的整体/局部等权分数

一次只检验评分环节，原 adapter、AOSS、选点、训练预算和主表保持不变。
不基于已观察到的 AUROC 修改 AOSS 或重选层。新增方法是探索性补充，不替代
contrast_topk 主评分，也不是新的确认性测试。

对于每个原 final checkpoint 的 held-out modality：

1. 使用 diagnostics 的 `source_<modality>.csv` 中 **normal_mean** 和
   **normal_contrast_topk**；不使用 perturbed 分量、target 数据或 target 标签拟合。
2. 每个 source dataset 各 256 个正常样本分别形成经验 CDF。相等值用 midrank：
   `F(x)=(count(v<x)+count(v<=x))/(2*n)`。
3. 在同一 modality 内对 dataset CDF 等权平均，再对四个 source modalities 等权。
   RESC/OCT2017 不因两个数据集而获得两倍 modality 权重；目标是 OCT 时两者均排除。
4. 固定 `score = 0.5 * F_mean(mean_disagreement) + 0.5 * F_contrast(contrast_topk)`。
   不学习权重，不反转方向，不搜索阈值，不按数据集使用不同公式。
5. 同时完整报告 original_contrast、source_mean_ecdf、source_contrast_ecdf、
   source_equal_fusion 四种方法；不在四者中按 test 挑“最终方法”。

先执行 fit 并保存不可静默覆盖的 60 个 source reference，再独立 evaluate。
fit 不加载 target prediction CSV；evaluate 才读取标签计算 AUROC/AP。
输入/源码 SHA256、原分数重放、输出 CSV 哈希均记录；已有 calibration 与输入不同
时拒绝覆盖。不要删除锁文件后反复试公式；新假设应明确另立版本。

该设计由已经观察的 test 结果启发，因此即使计算上 source-only，也不能宣称本次
改进是在未见数据上得到的确认性 zero-shot 优势。原全局跨-fold AOSS 信息边界仍需披露。
正常标定样本可能与 adapter 训练重叠，不声称独立 source validation。
经验 CDF 在尾部会饱和、产生 ties，输出 saturation.csv；不根据饱和或 AUROC 再调权重。

## 运行和产物

不需要新 GPU 实验，复用已有 predictions 与 diagnostics：

```bash
python -m src.source_calibration --stage fit --diagnostics results/followup/diagnostics
python -m src.source_calibration --stage evaluate --diagnostics results/followup/diagnostics
```

本地取回目录则使用 `--diagnostics results/transfer/diagnostics`。两步应在同一仓库位置
连续执行，原始输入须保留。依赖为 requirements-analysis.txt，入口不下载权重、
不生成 cache、不训练，也不调用 pipeline selection。

所有输出在 `results/exploratory/source_normal_ecdf_v1/`：calibration.json、
evaluation_manifest.json、72 个逐图 CSV、全指标、seed macro/SD、配对 AUROC 差值、
source/target 条件效应与 interaction、radiology/non-radiology 分层、饱和率和报告。
若在服务器执行，取回整个该目录；不必取模型。所有结果不得提交 Git。

## 事先确定的解释规则与后续顺序

- 四 pair、所有 seed/dataset 均报告，无效/负面结果保留；不排除不利数据集。
- 等权融合若仅部分数据集改善，不能声称普遍有效；若整体差异更有用，仍需回答
  stitching 相对已有 frozen-feature dispersion baseline 的增量价值。
- 三个 seed 的 SD 不是患者 CI；不把相关的多个分数当作独立重复。
- 本轮不改 AOSS：下一步若确有必要，应独立设计 source-only 内层验证，区分
  local sensitivity、normal discrepancy 和比值的作用，并在新测试数据上确认。
  这属于后续新协议，不能回写当前选点结果。
- 不预先改变论文研究定位。应先检查 NFFA、TargetTail 接口和 AOSS 的原技术链条，
  再依据证据决定方法修复及受控实验；不得预设 MM 必须优于 GG。
