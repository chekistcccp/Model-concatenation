# 下一轮实验：运行与取回文件

这两个入口是独立补充实验，保持原四 pair、五个 held-out modalities、三 seeds、
224 输入、contrast_topk/0.05，以及原 selected_configs 不变。
不调用 prepare、cache 重建、模型下载或 select_configs，不覆盖原 screen/final。
不根据目标 test 的诊断选择更好的点、评分或 seed。现有跨-fold 全局 AOSS 的
正常域信息边界仍需披露；这里不会修订原 selection protocol。

## 1. 先运行诊断，不训练

在服务器**原实验仓库根目录**，激活原实验环境：

```bash
git pull --ff-only
python -m src.run_followup --stage diagnostics --check-only
GPUS=0,1,2,3 bash run.sh diagnostics
```

单 GPU 将最后一行改为 `GPUS=0 bash run.sh diagnostics`。不需要重新安装或升级
原 GPU 依赖。`--check-only` 不写文件；缺失原始输入会明确失败，不补数据或重训。
本地 CPU 测试不能代替服务器真实模型重放校验。

前置输入：已有 12 个 final JSON、60 个 checkpoint、完整 test/perturb cache，
两种 Transformer 的原本地权重，以及完整 `results/predictions/`。
Brain/RESC/OCT2017 的所选原图及有标注时的 mask 必须仍在服务器原位置。
**不必重新跑 predictions**：本轮已经取回并验证过的导出即可。

此入口复用原 checkpoint：

- 诊断 Brain/MM、RESC/GM、OCT2017/GM 的异常；同一批图比较全部四 pair/三 seeds。
- 每 dataset/label 固定随机抽 10 张，加入 5 张难排序样例；随机与误例分别标记，
  重合去重，最多 90 张图。样例抽取仅用于 post-hoc 检查，不能估计无偏总体性能。
- 保存 224 RGB 样例图、已有 mask、16×16 discrepancy/contrast map、top-k mask。
- 在三个 dataset 的完整 test 上逐图核对原 CSV 分数，并重算 image AUROC/AP；
  分数和指标的绝对差均须不超过 1e-6，不能用排名一致替代数值一致。
- 使用各 fold 的 source-only perturb cache 输出分量；OCT held-out 时同时排除
  RESC/OCT2017。原 AOSS 仍调用未经修改的 compute_aoss，重放差异须不超过 1e-6。
- 历史 perturb records 没存操作 ID；类型由原生成器配方与记录索引推断并明确标记，
  不能冒充独立验证过的历史类型来源。没有修改或重新生成 perturb cache。

输出：

```text
results/followup/diagnostics/
├── run_manifest.json
├── case_plan.json
├── cases.csv
├── cases/<dataset>/*.png
├── jobs/<config>_seed<seed>/
│   ├── evaluation.json
│   ├── <dataset>_maps.npz
│   ├── <dataset>_components.csv
│   └── source_<held_out_modality>.csv
├── inputs/                         # 原配置、选点、12 final JSON、已有审计/manifest 快照
└── logs/<job>.log
```

地图与 PNG 尚未形成病灶解释；取回后用它们作对照图和局部统计。缺失 mask 会在
NPZ 中明确标注，不能把未知标注当成正常区域。top-k mask 的并列值位置只反映
数值算子的选择，不能单凭该位置声明唯一热点。

## 2. 再运行共同 stitch 对照

诊断通过后，或已决定按固定矩阵补齐共同点对照时：

```bash
python -m src.run_followup --stage common_stitch --check-only
GPUS=0,1,2,3 bash run.sh common_stitch
```

这一步**会新增 adapter 训练**，但不会重选主配置。固定点来自原 AOSS 锁定点的
并集 s2b3/s2b9；两个点、四 pair、三 seeds 全部报告，不挑表现最好的点。
block 数字沿用原代码的 Python slicing index。

共 24 个 job：复用 12 个原 final JSON，只训练缺少的 12 个，共 60 次 fold
adapter 训练。使用原 final 的每 source modality 1000 normal、8 epochs、MLP、
lr=0.001、weight_decay=0.0001、seeds=11/22/33 和 contrast_topk/0.05。
采样与 NFFA 训练仍调用原 worker/train_adapter；目标 modality 不参与其训练/AOSS。
保持原 frozen backbone，不扩展分辨率或候选网格。

输出：

```text
results/followup/common_stitch/
├── run_manifest.json
├── <config>_seed<seed>.json         # 24 个，原 12 个按字节复制
├── checkpoints/<job>/<modality>.pt # 仅新增 12 个 job 的 checkpoint
├── inputs/
└── logs/<job>.log
```

完成的补充 job 仅在输入计划一致、指标/预算/五 folds 和 checkpoint 齐全时复用。
失败或中断的未完成 job 会从该 job 重新训练；不使用部分训练状态替代完整结果。
若旧输出属于不同输入，入口拒绝混用，应单独归档那次补充运行后再执行。

## 3. 如何判断完成

两阶段都检查 `run_manifest.json`：`status` 必须是 `complete`，
`original_inputs_unchanged` 和 `no_reselection` 必须为 true。
diagnostics 的 12 个 evaluation JSON 还须同时满足 `replay_matches_original`
和 `aoss_replay_matches_original` 为 true。失败时保留 manifest 与 logs 返回排查，
不要放宽阈值、覆盖原 final 或换点重跑来消除异常。

manifest 保存配置、原结果、checkpoint、records、代码与产物哈希。
大型 feature arrays 和 Transformer 权重仅检查 size/mtime，不声称有完整 SHA256
来源验证；legacy checkpoint 的 seed 来源限制继续保留。
原图按 cache records 路径对应；当前原图哈希不能追溯补证历史 cache
编码时的像素内容。继续使用原服务器上保留的原始数据与实验环境。

## 4. 从服务器取回哪些目录

**取回是复制到本地分析，不是提交 Git。**

| 内容 | 取回本地 | 同步 Git |
|---|---|---|
| results/followup/diagnostics/ 整个目录 | 必须，包含 inputs、cases、jobs、logs | 否 |
| results/followup/common_stitch/ 除 checkpoints 外全部 | 跑完对照后必须 | 否 |
| common_stitch/checkpoints/ | 留在服务器；之后重放需要时再取 | 否 |
| data/、model/、cache/features/、cache/perturb/ | 本轮无需整批取回，服务器必须保留 | 否 |
| 原 results/predictions/、screen/、final/、stitchmap/、ablation/ | 已取回的无需重复复制或覆盖 | 否 |
| src/、tests/、run.sh、docs/ 的代码改动 | 通过 git pull 获取 | 是 |

便于一次下载，可在服务器打包（包含运行日志与输入快照，排除训练权重）：

```bash
mkdir -p results/transfer
tar --exclude='*/checkpoints' -czf results/transfer/followup_results.tar.gz results/followup
```

下载 `results/transfer/followup_results.tar.gz`，在本地仓库根目录解压，使其恢复
`results/followup/` 层级即可。本轮新目录不要覆盖原 results；不必把模型或缓存
打包。只跑 diagnostics 时，同一个打包命令也适用。

不要对结果或压缩包使用 `git add -f`。不要为这两项实验运行 `bash run.sh all`
或 `final`；已有 final 入口会再次调用选点，不适合补充实验。
# 取回后的 CPU 分析

保持取回文件原样；若 `diagnostics/` 和 `common_stitch/` 解压在
`results/transfer/`，运行：

```bash
python -m src.analyze_followup_results --root . --input results/transfer
```

若保留服务器目录层次，改为 `--input results/followup`。依赖使用
`requirements-analysis.txt`。入口核对传输哈希、原始 final、锁定选点、训练预算、
重放指标、空间分数与 source 模态排除，再报告两个固定点的配对 seed 效应。
输出仅在 `results/analysis/followup_results/`，不能提交仓库。

原始 prediction 元数据 receipt 若与服务器记录不同，会单独记录为来源限制；
CSV 分数和 final 的哈希不允许不同。Windows 配置/源码仅 CRLF 差异单列记录。
病例图是预定 random/hard 子集的事后诊断，不能当作全测试集定位评估；
不根据这些分析重选 stitch 或替换主分数。
