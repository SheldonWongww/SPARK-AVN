# EAM 在 AVN 上的固定范围强度搜索实验报告

更新日期：2026-08-01

状态：探索性超参数开发报告。SMT+Audio single-source 的 24 组固定范围强度
网格全部完成，并在服务器运行结束时通过逐任务校验；但本地未同步逐运行
`manifest.json`，Source checkpoint 与数据资产 provenance 也不完整，因此当前
结果不能直接进入论文正式主对比表。

## 1. 结论摘要

本报告解析批次
`eam_intensity_grid/eam-intensity-grid-avn-val-rerun-v2-seed0` 的全部结果。

1. **24 组运行完整且内部协议一致。** 所有任务均完成 2000 episodes，
   `runner_exitcode=0`、`exitcode=0`、`validation=ok`；每组均有 9 项最终聚合
   指标和 40 个 50-episode 诊断窗口，未发现 NaN、OOM 或 Traceback。
2. **按 SPL 主排序、SR 次排序，开发候选是 job 3。** `LR=1e-8`、
   `UPDATE_INTERVAL=32` 达到 SR/SPL/SoftSPL=55.40/29.8248/36.7086，
   相对 Source 分别提高 1.25/0.4019/0.1981 个百分点；DTG 降低 0.1870，
   NDTG 降低 0.012667，SNA 提高 1.4668 个百分点。
3. **收益并非全指标一致。** job 3 的 NA 从 153.172 增至 155.725，SWS
   从 3.20% 降至 2.70%。24 组中只有 job 3 同时超过 Source 的 SR、SPL 和
   SoftSPL，且没有配置取得更低 NA。
4. **SR/SPL Pareto 前沿只有两个点。** job 3 提供最高 SPL；job 7
   (`LR=1e-7, interval=32`) 提供最高 SR=55.90%，但 SPL=29.2430、
   SoftSPL=35.7377，均低于 Source。
5. **主要规律是控制累计更新强度。** LR 增大时平均参数漂移从 0.001659
   增至 0.038371，平均 SR/SPL 同步下降；更新间隔从 1 增至 32 时，平均漂移
   从 0.028605 降至 0.006664，平均 SPL 从 28.2877 升至 28.9404。
6. **没有硬坍塌，但存在过适配和过置信。** 现有末段风险启发式在 24 组中
   均未触发；然而最强配置 `1e-5/interval=1` 的漂移达到 0.072146，平均动作
   熵降至 0.27876，SPL 降至 27.0791，NA 增至 182.266。
7. **正信号较弱，必须复验。** 最优点位于最低 LR 与最大更新间隔的双重边界，
   且来自同一 seed-0 流上的 24 次选择。当前证据支持冻结一个开发候选，不能
   证明 EAM 已稳定优于 Source。

## 2. 实验协议

### 2.1 搜索空间

| 变量 | 取值 |
|---|---|
| EAM learning rate | `1e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5` |
| 动作步更新间隔 | `1, 4, 16, 32` |
| 适配范围 | `full_transformer_plus_head` |
| 笛卡尔积 | `6 × 4 = 24` |

适配范围为 SMT state Transformer 与 action-distribution head，共 36 个张量、
1,057,284 个标量；所有 pre-Transformer 感知与状态编码器保持冻结。Source
分支始终冻结，只有从 Source 初始化的辅助分支被更新。EAM 不使用 success 或
其他二值 episode feedback，属于无监督 TTA。

### 2.2 固定控制量

- SMT+Audio、single-source、val split、seed 0、2000 episodes；
- 20 scenes × 100 episodes、global shuffle，与 Source/Tent 使用同一 episode
  stream；
- sample 动作、单进程、`EVAL.USE_CKPT_CONFIG=False`；
- continual adaptation：`EPISODIC=False, STEPS=1`；
- `CONFIDENCE_SCALE=0.4, MEMORY_SIZE=32, BATCH_SIZE=8`；
- Adam，`betas=(0.9, 0.999)`、`weight_decay=0`、不裁剪梯度。

重放单位为 action step。当前样本先进入更新后的 reservoir，再形成 replay
batch；缓冲区不足 8 步时只用当前样本，之后从 reservoir 取 7 个历史项并显式
追加当前项。当前动作由优化前的决策分布确定，优化只影响后续动作。这些语义及
更新次数、重放步数、参数范围均由 launcher 的逐任务校验检查。

## 3. 完整性与哈希核验

| 检查项 | 结果 |
|---|---:|
| 计划/完成 | 24/24 |
| 唯一 LR×interval 配置 | 24 |
| runner/综合 exitcode 为 0 | 24/24 |
| `validation=ok` | 24/24 |
| 9 项最终聚合指标完整 | 24/24 |
| 40 个诊断窗口完整 | 24/24 |
| manifest 路径指针 | 24 |
| 本地可解析 manifest | 0 |
| 当前可进入正式结果表 | 0 |

最终指标取日志末尾的 9 条 `Average episode ...` 聚合行；
`[TTA] episodes=2000 window=...` 只是最后 50 个 episodes，不应误作全程结果。

批次锁定 Git commit `4d70c032be02f83e5e298083281bb714c2c13e38`，启动时
`tracked_worktree_dirty=0`。以下摘要既通过服务器 preflight，也已在当前工作区
重新计算并确认一致：

| 资产 | SHA256 |
|---|---|
| SMT+Audio single-source checkpoint | `8007dc0de8b0e994244d4f2fdb4a642bcc6213b4e9694568c93b10141f53ef03` |
| canonical val dataset index | `838532d8e10064dd2bccbdbb7e75b8ca7cb5c4e7a3db579c3b40cfab18081c80` |
| episode stream order | `07f327590ccee2999b3f6bcb2fc412f39d9802cf932b14933fd0bdd9e5ca380c` |
| episode stream content | `dd411c4aafaf626b2848d20b92d1832ea46a5380c57107043fc639996837fdf2` |

Source 对照使用同一 checkpoint、seed、episode 数、动作协议和 stream
order/content 摘要，但 Source 批次位于 commit `48ea6285...`，并明确标记为
`legacy_checkpoint_provenance_incomplete`。因此这是协议和二进制指纹高度对齐
的探索性比较，不是同一代码快照上的正式配对实验。

## 4. 与 Source 的比较

### 4.1 SPL-first 最优配置

| 配置 | Reward | SR↑ | SPL↑ | SoftSPL↑ | DTG↓ | NDTG↓ | NA↓ | SNA↑ | SWS↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source | 11.480265 | 54.15 | 29.4229 | 36.5105 | 5.0005 | 0.306263 | 153.1720 | 42.9696 | 3.20 |
| job 3：`1e-8/u32` | **11.606735** | **55.40** | **29.8248** | **36.7086** | **4.8135** | **0.293596** | 155.7250 | **44.4364** | 2.70 |
| Δ | +0.126470 | +1.25 | +0.4019 | +0.1981 | -0.1870 | -0.012667 | +2.5530 | +1.4668 | -0.50 |

24 个配置相对 Source 的计数为：

| 判据 | 配置数 |
|---|---:|
| SR 更高 | 4 |
| SPL 更高 | 1 |
| SR 与 SPL 同时更高 | 1 |
| SoftSPL 更高 | 1 |
| DTG 更低 | 4 |
| NDTG 更低 | 5 |
| NA 更低 | 0 |
| SNA 更高 | 5 |

### 4.2 SR/SPL Pareto 前沿

| job | LR | interval | SR | SPL | SoftSPL | DTG | NA | 相对参数漂移 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | `1e-8` | 32 | 55.40 | **29.8248** | **36.7086** | 4.8135 | 155.7250 | 0.000391 |
| 7 | `1e-7` | 32 | **55.90** | 29.2430 | 35.7377 | 4.8430 | 161.6780 | 0.001966 |

job 7 的 SR 比 job 3 高 0.50 个百分点，但其 SPL 比 Source 低 0.1799、
SoftSPL 低 0.7728 个百分点；因此不符合 SPL-first 选择规则。

## 5. 超参数边际规律

### 5.1 Learning rate

| LR | 平均 SR | 平均 SPL | 平均 SoftSPL | 平均 DTG | 平均 NA | 平均漂移 |
|---:|---:|---:|---:|---:|---:|---:|
| `1e-8` | **54.5375** | **29.3486** | **36.2661** | **4.9689** | **161.3726** | **0.001659** |
| `1e-7` | 54.0375 | 28.7135 | 35.4254 | 5.0880 | 168.8315 | 0.005312 |
| `3e-7` | 53.2750 | 28.6117 | 35.5221 | 5.1124 | 171.4951 | 0.008642 |
| `1e-6` | 52.5125 | 28.6102 | 35.6442 | 5.2081 | 172.2666 | 0.014431 |
| `3e-6` | 52.3750 | 28.3211 | 35.2482 | 5.2583 | 177.8649 | 0.021232 |
| `1e-5` | 51.9375 | 27.9859 | 35.3332 | 5.1689 | 176.8326 | 0.038371 |

最低 LR 的边际均值最好，但其平均 SPL 仍比 Source 低 0.0743 个百分点。
这说明 job 3 是一个孤立的弱正点，而不是整个 `1e-8` 区域都稳定优于 Source。

### 5.2 更新间隔

| interval | 平均 SR | 平均 SPL | 平均 SoftSPL | 平均 DTG | 平均 NA | 平均漂移 | 平均更新数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 52.5583 | 28.2877 | 35.3642 | 5.1687 | 175.4206 | 0.028605 | 350,832.5 |
| 4 | 52.5417 | 28.4809 | 35.4849 | 5.1860 | 173.0217 | 0.015564 | 86,495.2 |
| 16 | 53.4667 | 28.6849 | 35.5775 | 5.1539 | 170.0370 | 0.008932 | 21,253.7 |
| 32 | **53.8833** | **28.9404** | **35.8662** | **5.0278** | **167.2962** | **0.006664** | **10,453.8** |

更大的间隔整体降低漂移并改善导航效率，但 interval=32 的边际 SR/SPL 仍低于
Source。当前证据支持“减少累计更新预算”，不支持把单点最优解释成普遍增益。

## 6. 更新、漂移与坍塌诊断

### 6.1 最优配置

job 3 共记录 311,450 个动作步，理论更新尝试数为
`floor(311450/32)=9732`，实际完成 9,731 次，仅 1 次因无可靠 replay 样本跳过。
最终 replay size 为 32，重放步数为 77,856，与 launcher 的预期公式完全一致。

其关键诊断为：

- 相对参数漂移：0.000391，即相对范数约 0.0391%；
- 平均动作熵：0.339526；平均最大动作概率：0.866047；
- Source gate rate：0.704174；auxiliary gate rate：0.740353；
- 平均/最后训练损失：0.046955/0.021732；
- 最后四个窗口平均 SR/SPL：56.50/30.5154。

### 6.2 全网格行为

沿用项目现有的末段风险启发式——最后四个 50-episode 窗口平均 SR 低于 25%，
或比全程 SR 低超过 15 个百分点——24 组均未触发。该结果只说明没有明显的
末段硬坍塌，不等于所有配置稳定有效。

| 变量 | 与 SR 的 Pearson r | 与 SPL 的 Pearson r |
|---|---:|---:|
| 相对参数漂移 | -0.682 | -0.756 |
| 平均动作熵 | +0.783 | +0.673 |
| 平均最大动作概率 | -0.814 | -0.708 |

全网格共有 2,814,365 次更新尝试，仅跳过 154 次，跳过率 0.00547%。Source
gate rate 约为 65.6%--70.6%，但 replay batch 中只需存在一个可靠样本即可更新，
所以当前门控主要筛选 loss 中的样本，并没有实质限制 optimizer step 数。

最强配置 job 20 (`1e-5/u1`) 完成 364,526 次更新，漂移为 0.072146，平均熵
0.27876、平均最大动作概率 0.90293，最后训练损失仅 `1.87e-7`；其 SR/SPL 为
51.40/27.0791，NA 为 182.266。它表现为持续过适配和置信饱和，而不是运行时
数值失败或突然的末段坍塌。

## 7. 证据边界

当前结果必须保留以下限制：

- 24 个 `manifest.path` 均指向原服务器 `/data1/.../manifest.json`，当前工作区
  可解析数为 0；因此无法本地复核硬件、完整有效配置和逐运行 manifest 内容。
- `avn/checkpoints/manifests/imported_pretrained.yaml` 仍将 Source checkpoint
  标为 `provenance_incomplete`；训练命令、训练 seed 和 checkpoint 选择依据
  尚未恢复。
- `avn/data/manifests/datasets.yaml` 状态为 `partial`，场景、音频与 RIR 等资产
  的完整来源和摘要仍未登记。
- Source 与 EAM 不在同一 Git commit；虽然 checkpoint 与 episode stream
  摘要完全相同，仍不能替代同提交确认运行。
- 全部搜索只有 seed 0，且从 24 个点中选择最大值，存在 multiple-comparisons
  selection bias。本地没有逐 episode JSON，无法做配对显著性分析。
- EAM 与 ENMuS FSTTA 批次并行运行，每张 GPU 峰值并发 10 个任务；现有
  wall-clock 日志不能用于正式时延比较。
- `avn/experiments/eam_intensity_grid.yaml` 仍描述旧的 72-job v1 搜索并标记
  `planned`；实际权威协议是本批次 `batch.env`、`grid.csv` 与 fixed-scope v2
  launcher。该文档差异不改变已运行配置，但正式登记前应修正。

## 8. 下一步建议

1. **冻结而非宣称胜出。** 暂将 job 3 的 `LR=1e-8, interval=32` 冻结为
   SMT+Audio single-source 开发候选，不写入正式主表。
2. **先做确认运行。** 在同一最终 commit 上隔离运行 Source 与 job 3，保存
   本地可解析的 manifest、逐 episode stats、diagnostics、硬件和环境信息；至少
   增加重复 seed 或独立开发流，以估计 sample 动作和持续适配的方差。
3. **补齐 provenance。** 恢复 Source checkpoint 的训练来源和选择依据，补齐
   canonical val 数据及场景/声音/RIR 资产版本与摘要。
4. **确认后再迁移。** 不重新调参，直接把冻结配置测试到 SMT+Audio
   multi-source 和 ENMuS single/multi，检验跨 setting 与跨模型泛化。
5. **若继续第二阶段，只研究更弱更新区域。** 最优点位于强度边界，后续可在
   独立 dev stream 上小范围测试 interval 64/128、相邻更低 LR，以及 confidence
   与 replay 控制；不应继续扩展已经显示退化的高 LR 高频更新区域。
6. **正式时延单独测量。** 使用单 GPU、无并发、固定预热和重复运行，不复用
   本次共享资源的 wall-clock 时间。

## 9. 证据清单

- 启动脚本：`avn/scripts/run_eam_intensity_grid.sh`
- 旧实验定义：`avn/experiments/eam_intensity_grid.yaml`
- 批次元数据：
  `avn/results/logs/eam_intensity_grid/eam-intensity-grid-avn-val-rerun-v2-seed0/batch.env`
- 批次状态：
  `avn/results/logs/eam_intensity_grid/eam-intensity-grid-avn-val-rerun-v2-seed0/SUMMARY`
- 24 组配置：
  `avn/results/logs/eam_intensity_grid/eam-intensity-grid-avn-val-rerun-v2-seed0/grid.csv`
- 逐运行日志：同批次目录下 `jobs/<run_tag>/console.log`
- 联合调度记录：`avn/results/logs/parallel_fstta_eam/avn-val-rerun-v2-seed0/`
- Source 对照：`avn/results/logs/source_reval/source-reval-v1-seed0/metrics.csv`
- checkpoint provenance：`avn/checkpoints/manifests/imported_pretrained.yaml`
- dataset provenance：`avn/data/manifests/datasets.yaml`
