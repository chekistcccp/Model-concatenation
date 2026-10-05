# 冻结方法模型的全目标原图验证

方案固定于 2026-10-05。研究主线仍为继承预训练参数模块，构成新的零样本异常检测模型。
本轮检验真实 RGB 输入上的检测能力和完整检测器成本，不新增训练或模型选择。
具体取回数字、异常值、论文差距判断和图保存在 results/analysis/，不提交 Git。

## 前置条件与运行

保留完成的 results/followup/method_controls/ 全部产物及其 180 个控制 checkpoint、
四个 bundle、原 final/checkpoints、原模型、cache、目标原图和真实 mask。
使用完成 method_controls 的同一 Python/依赖环境；不安装或升级依赖。手动同步代码：

```bash
git pull --ff-only
# 可选：重新验证并打包已有方法实验。修复缺少 matplotlib 的后处理，不会重训。
GPUS=4,5,6,7 bash run_method_controls.sh
# 下一轮实际实验：使用可见 CUDA 索引，每卡同时一个独立 pair×seed 任务。
GPUS=4,5,6,7 bash run_method_deployment.sh
```

未设置 GPUS 时默认使用全部可见卡。若 CUDA_VISIBLE_DEVICES=4,5,6,7，则 GPUS 应为
0,1,2,3。单卡 GPUS=0；PYTHON_BIN 支持成功环境 Python 的绝对路径。多进程需要
主机内存和原图读盘带宽；不通过改变固定 batch 或省略目标数据回退。

缺少 matplotlib 时，CPU 验证仍保存完整表格、报告和 audit.json，明确标记图未生成。
有该库的本地环境可从返回包重画图，不必更改服务器环境或重做训练。

## 固定实验矩阵与信息边界

- 原四 pair×seed11/22/33×五 held-out modality，四 arm 为 original、matched_tail、
  no_tail、untrained_adapter。使用各自原已冻结 checkpoint，不按目标表现挑 arm。
- 目标六 dataset 的全部 test 原图，严格按原 cache records 的身份、顺序、标签和
  mask。原 224 输入、归一化、top5% contrast_topk 不变。OCT 两个数据集用同一排除
  整个 OCT 模态的 adapter。目标标签仅进入指标评价，不训练、选点或校准。
- 四 arm 都从 RGB 经实际 CNN prefix、接口、后段/reference 前向；no_tail 使用其
  独立训练接口和固定输出 LayerNorm。原 checkpoint、配置、选择和 cache 只读。
- 每 dataset 开头两张原图比较注册模型与独立执行模块的 live discrepancy，容差
  1e-6；不把此抽样称为全部图像的独立数值重放。全目标实际前向的预测均导出。
- cached 与 RGB 数值差异分别记录，无强行归零、提高容差、翻转分数或替换历史主表。
  原 global AOSS 跨 fold 信息边界继续披露，不声称全局严格 fold-exclusive。
- 目标推理 batch=16；pixel AUROC/AP 继续原 16×16 patch-grid mask 规则，只在真实
  mask 覆盖的样本上算，不冒称原始分辨率定位或完整 AUPRO。

## 成本与完整性

每 fold/arm 测量完整 RGB tensor 检测器：归一化、CNN prefix、adapter、继承后段和
完整 reference。batch=1，五次 warmup，三十次 CUDA event；记录中位数/p95、当前
resident 和 peak allocated/reserved 显存、unique 参数。解码、CPU/GPU 传输不在此
计时内，不能称完整临床管线延迟。suffix 与 reference 的共享参数不重复计数，
unused CNN suffix 从 GPU 移除。其他外部 GPU 作业会影响时延，最好使用空闲卡。

coordinator 检查完成的原科学产物、全部冻结 checkpoint 的哈希和同环境；输入代码、
选择、配置、产物哈希及大 cache/原图 stat 前后不变。原图仅做 stat 稳定性检查，
不是所有原图的内容哈希证明。12 个 worker 各写独立日志和 receipt；失败停止同批
其余进程，保留证据。已有完整输出 CPU 核验后只重新打包；不完整输出拒绝覆盖/续跑。

预期 288 组 dataset 指标与 prediction CSV、240 行完整 detector 成本、12 worker
receipts、12 logs、317 个已声明返回产物哈希。图像指标和 RGB/cache 差异可在本地
独立从 CSV 重算；pixel、权重和成本的实际运行以服务器 receipt 为依据。

## 返回

下载终端明确打印的 `[TRANSFER] .../method_deployment_run_*.tar.gz`，放到本地
results/transfer/ 并在该目录解压。包含 method_deployment/ 和 launcher 日志、分析。
数据、cache、权重留服务器；所有结果和图保持 Git 忽略。取回后：

```bash
python -m src.audit_return_package --archive results/transfer/实际返回包.tar.gz \
  --require-stage method_deployment --output results/analysis/deployment_package_audit.json
python -m src.analyze_method_deployment --input results/transfer/method_deployment \
  --output results/analysis/deployment_independent_review
```

## 方法研发下一阶段

本轮不是算法创新的替代品。若继承 suffix 在原图推理上仍无检测增量，应进入独立
source-only 方法开发，而非继续扩大已看目标测试集的调参：

1. 待检验假设：普通全特征匹配可能把异常也重建，后段的空间混合可能削弱局部
   分离。现有指标不足以断言这些原因，需要输入干预、正常/异常差异及结构响应证据。
2. 候选方案：正常流形约束的瓶颈 stitch 接口，保留 CNN prefix→接口→继承后段
   路径；研究正常区域的结构保持与过度重建抑制。与 FoundAD 的正常化投影、QFAE
   瓶颈重建及 PDD 的异构蒸馏明确区分。瓶颈本身不是已证明的新颖性。
3. 仅用相应 held-out fold 的非目标正常 source 建 development 划分；完整排除目标
   模态。结构/损失及预算在 source development 固定，原 AOSS、锁定点和主评分保留。
   新变体另立输出。不得根据已看 BMAD target 表选择瓶颈宽度、权重或 seed。
4. 同时预注册容量匹配的 self-stitch/单 encoder、direct reconstruction、简单融合
   和 source-only PCA 对照；再加入 VisualAD/AnomalyVFM 等公开强基线，分列辅助
   异常监督权限。原 target-normal/few-shot 数字不能当同权限 zero-shot 实验。
5. 新方案冻结后增加独立来源确认性数据、可信 patient/slide ID 分组不确定性，
   原分辨率 AUPRO/pixel AP、病变大小及背景误报。不要只靠三 seed SD 宣称显著。

以上瓶颈/基线/新来源阶段尚未实现或运行，本次代码只完成可直接提交的冻结原图
推理实验。先解决实际模型证据边界，再固定独立方法方案；不能把可部署性称为性能创新。
