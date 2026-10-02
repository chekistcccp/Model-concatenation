# 原方法第二环：source 扰动的配对响应

研究目标依交接第 1、5、8、25、31 节：正常对齐后的 CNN→Transformer
disagreement 是否对异常变化敏感，以及医学 source/target 是否改变这种能力。
本轮承接已通过的机制审计，保持原 12 个 final job / 60 个 checkpoint、五模态
排除关系、原 source-only AOSS、已锁定选点和 contrast_topk 主评分。

## 预先固定的检验

复用原 cache/perturb 中每 dataset 256 个正常/扰动配对样本和原 mask。
每个 checkpoint 只读取其 held-out modality 以外的数据；OCT 同时排除两个数据集。
不读取 test/valid cache、不训练、不更新权重、不修改选择流程。

对于同一图像、同一 mask：

```text
normal_spatial_contrast = normal_in - normal_out
raw_local_sensitivity  = perturb_in - perturb_out
delta_in               = mean(d_pert - d_normal | mask)
delta_out              = mean(d_pert - d_normal | outside)
net_local_response     = delta_in - delta_out

raw_local_sensitivity = normal_spatial_contrast + net_local_response
```

该分解用于判断原局部敏感性有多少来自正常空间结构，有多少来自扰动新增变化。
净响应、原始响应和负响应均完整保存；缺少区域时相应统计为空值，不填成 0。
净响应不是新的 AOSS 或选点指标，不用于重新排序，也不替换任何主结果。

每个 fold 另外使用原 `compute_aoss` 函数及原 batch=256 独立重放原五个分量，
与原 final 比较固定绝对容差 1e-6，共 300 项检查；失败留存证据并停止。
CSV 的 per-image 均值与原 AOSS 的 batch/区域加权不同，不能混同。

根据当前原训练采样代码、cache records 和 seed 重建训练图像成员集合，输出成员
标记和剩余 normal 数量。旧 checkpoint 未保存历史样本清单，因此成员信息属于
推断，不能声称已经证明训练外泛化。本轮不新增训练或另造验证集。
扰动类型也仅按原生成器/索引推断，不作为已核验历史操作 ID。

## 一键运行与取回

用户手动 git pull 后，在原实验环境中运行：

```bash
bash run_paired_response_audit.sh
```

默认可见 GPU 0，也可用 `GPU_ID=1 bash run_paired_response_audit.sh`。
保留已完成的 `results/followup/mechanism_audit/`；其运行版本、输入哈希与权重/cache
stat 将作为前置证据。若当前环境与机制审计不同，预检查拒绝混用，请恢复其环境。
脚本不会安装依赖、切换环境、git pull、下载权重、重建 cache 或运行主 pipeline。
计算顺序为预检查 → 配对重放 → 原 AOSS 重放 → CPU 汇总 → 日志与结果打包。

原始输出在 `results/followup/paired_response_audit/`，含 60 个 source CSV、
aoss_replay.csv、training_membership.csv、run_manifest.json。期望 73728 行配对诊断。
CPU 汇总在 `results/analysis/paired_response_audit/`：全部、按推断扰动类型及推断
训练成员分层的 modality/fold/seed 表、seed 均值/SD、计数和 report.md。
pretraining_effects.csv 给出四组条件 source/target effect、平均效应和 interaction。
原 GG 选点与其他 pair 不同，必须披露该混合因素；这些结果不是共同点的因果对比。
每 modality 内 dataset 等权，每 fold 内四个 source modalities 等权，再平均五 fold。
种子 SD 不是患者 CI；同一图像在多个 fold 中重复，不能当作独立观测扩大 n。

已有配对输出时拒绝覆盖。成功或失败结束都会尝试打包，并打印 `[TRANSFER]` 路径：
`results/transfer/paired_response_run_日期时间_后缀.tar.gz`。
包内包含 paired_response_audit/ 和本次 launcher 目录；成功完成的汇总放在
launcher 目录的 analysis/ 下。机器断电或 SIGKILL 不保证退出打包。

下载压缩包到本地 results/transfer 并在该目录解压。取回后也可重新运行 CPU 分析：

```bash
python -m src.analyze_paired_response --input results/transfer/paired_response_audit
```

结果、日志、压缩包、权重、数据与 cache 不提交 Git；仅代码、测试和方法说明同步。

## 结果如何指导下一步

- 原始局部响应高、净响应弱：说明正常区域基线对该诊断有较大贡献，须先定位
  这种贡献与医学预训练/模态的关系，不能直接据 test 改 AOSS。
- 净响应随合成扰动增强，但真实异常表现仍弱：继续检查合成到疾病变化的迁移，
  结合已锁定真实结果解释，不事后调权重或重选层。
- 配对 AOSS 重放失败：先检查运行环境/缓存/输入身份，保留失败结果，不放宽容差。
- 本轮只覆盖原选定 final 点。若后续要解释九点 selection regret，应另做同一
  screening 预算的九点分解，不能拿 final 点诊断代表完整网格。
