# EAM 在 AVN 上的超参数搜索实验报告

更新日期：2026-08-04

状态：已完成 42 次 single-source 超参数开发运行，并将两个模型各自
冻结的配置直接迁移到 multi-source，新增 2 次完整主表复验。
44/44 次导航运行均完成并通过各自 launcher 校验。EAM 四个场景
结果已按当前研究决定冻结；但本地仍未同步逐运行 `manifest.json`，
Source checkpoint 与数据资产 provenance 也不完整，因此主表中继续
标记为 provisional (`[P]`)。

## 1. 结论摘要

本报告联合解析以下三批实验：

- `eam_intensity_grid/eam-intensity-grid-avn-val-rerun-v2-seed0`：24 组
  SMT+Audio 第一阶段强度网格；
- `eam_boundary_grid/eam-boundary-joint-v1-seed0`：SMT+Audio 和 ENMuS 各
  9 组弱更新边界网格。
- `eam_main/eam-main-multi-v1-seed0`：两个模型各 1 组
  multi-source 冻结配置复验。

1. **44 组运行全部完成。** 第一阶段 24/24、边界阶段 18/18、
   multi-source 复验 2/2 均为
   `runner_exitcode=0`、`exitcode=0`、`validation=ok`，每组都完成同一
   场景下的 canonical val、seed-0、2000-episode stream。
2. **SMT+Audio 最佳配置更新为 boundary job 5。** `LR=1e-8`、
   `UPDATE_INTERVAL=128` 达到 SR/SPL/SoftSPL=56.20/30.5805/37.4332，
   相对 Source 提高 2.05/1.1576/0.9227 个百分点，NA 降低 3.089。
   该点同时是边界网格的最高 SR 和最高 SPL，替代第一阶段 job 3。
3. **ENMuS 最佳配置是 boundary job 11。** `LR=3e-9`、
   `UPDATE_INTERVAL=128` 达到 SR/SPL/SoftSPL=68.15/36.9083/40.6694，
   相对 Source 提高 1.60/0.8603/0.4509 个百分点，NDTG 降低
   0.002491，NA 降低 1.3345。该点也同时取得 ENMuS 网格最高 SR/SPL。
4. **两个模型的单点最优都位于低累计更新强度区，但最佳 LR 不同。** 两者
   都选中 interval 128；SMT+Audio 选中 `1e-8`，ENMuS 选中更低的
   `3e-9`。边际均值并不随 interval 单调改善，因此这是 LR×interval 交互，
   不应把 `u128` 或一个统一 LR 单独解释为普适最优。
5. **边界正向区域比第一阶段更密集。** SMT+Audio 9 组中有 6 组同时
   超过 Source SR/SPL，5 组还同时超过 SoftSPL；ENMuS 9 组中有
   7 组同时超过 SR/SPL，2 组进一步超过 SoftSPL。这比原先
   SMT+Audio 24 组中仅 1 个三指标双胜点更有说服力。
6. **最佳配置均超过当前 Tent 的 SR/SPL。** SMT+Audio job 5 相对
   Tent 提高 0.40 SR/0.7573 SPL；ENMuS job 11 提高 0.80 SR/0.2226 SPL。
   但这些均是同一开发 stream 上搜索后的最大值，不等于独立确认结果。
7. **multi-source 迁移呈现模型依赖性。** SMT+Audio 相对 Source
   提高 1.60 SR / 0.7086 SPL，并高于当前 Tent/FSTTA 的 SR/SPL；
   ENMuS 只提高 0.65 SR / 0.0104 SPL，且低于 Tent/FSTTA。这两项
   都是未在 multi-source 重新选参的直接迁移结果。
8. **证据资格仍为 provisional。** 实验均只有一个 episode 顺序，
   本地缺失逐运行 manifest 和完整 checkpoint 训练来源。SMT+Audio 日志
   保留 40 个诊断窗口，ENMuS compact 日志则没有等价的 50-episode
   窗口，所以尚不能宣称统计显著或已排除末段退化。

## 2. 第一阶段 SMT+Audio 强度网格协议

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

### 2.3 弱更新边界网格

第一阶段最优点落在最低 LR 和最大 interval 的边界，因此第二阶段只扩展弱更新
区域，并同时测试两个模型：

| 变量 | 取值 |
|---|---|
| 模型 | `smt_audio, enmus` |
| EAM learning rate | `3e-9, 1e-8, 3e-8` |
| 动作步更新间隔 | `32, 64, 128` |
| 适配范围 | `full_transformer_plus_head` |
| 每模型组合数 | `3 × 3 = 9` |
| 总运行数 | `2 × 9 = 18` |

SMT+Audio 仍适配 36 个张量、1,057,284 个标量；ENMuS 适配相同语义范围下的
71 个张量、3,499,012 个标量。其他控制量保持不变：single-source canonical
val、seed 0、2000 episodes、continual adaptation、step-level replay、
`CONFIDENCE_SCALE=0.4`、`MEMORY_SIZE=32`、`BATCH_SIZE=8`、Adam、
`betas=(0.9, 0.999)`、零 weight decay、无梯度裁剪。SMT+Audio 使用 sample
动作；ENMuS 使用其原生 `deterministic=False` 采样路径。

## 3. 完整性与哈希核验

| 检查项 | 第一阶段 SMT+Audio | 边界阶段 SMT+Audio | 边界阶段 ENMuS |
|---|---:|---:|---:|
| 计划/完成 | 24/24 | 9/9 | 9/9 |
| runner/综合 exitcode 为 0 | 24/24 | 9/9 | 9/9 |
| `validation=ok` | 24/24 | 9/9 | 9/9 |
| 9 项最终聚合指标完整 | 24/24 | 9/9 | 9/9 |
| 40 个 50-episode 窗口 | 24/24 | 9/9 | 0/9 |
| manifest 路径指针 | 24/24 | 9/9 | 9/9 |
| 本地可解析 manifest | 0 | 0 | 0 |
| 当前可进入正式结果表 | 0 | 0 | 0 |

最终指标取日志末尾的 9 条 `Average episode ...` 聚合行；
`[TTA] episodes=2000 window=...` 只是最后 50 个 episodes，不应误作全程结果。

第一阶段锁定 Git commit `4d70c032be02f83e5e298083281bb714c2c13e38`；边界
阶段锁定 `f27e257cf9d5dc323919b1632fe51c1b369d9ee3`，两个批次启动时均为
clean tracked worktree。以下摘要既通过服务器 preflight，也已在当前工作区
重新计算并确认一致：

| 资产 | SHA256 |
|---|---|
| SMT+Audio single-source checkpoint | `8007dc0de8b0e994244d4f2fdb4a642bcc6213b4e9694568c93b10141f53ef03` |
| ENMuS single-source checkpoint | `4f37a377cc7fcb888c545850c91883560a908ba5366072df787e4c8238ecefcd` |
| canonical val dataset index | `838532d8e10064dd2bccbdbb7e75b8ca7cb5c4e7a3db579c3b40cfab18081c80` |
| episode stream order | `07f327590ccee2999b3f6bcb2fc412f39d9802cf932b14933fd0bdd9e5ca380c` |
| episode stream content | `dd411c4aafaf626b2848d20b92d1832ea46a5380c57107043fc639996837fdf2` |

Source 对照使用同一 checkpoint、seed、episode 数、动作协议和 stream
order/content 摘要，但 Source 批次位于 commit `48ea6285...`，并明确标记为
`legacy_checkpoint_provenance_incomplete`。因此这是协议和二进制指纹高度对齐
的探索性比较，不是同一代码快照上的正式配对实验。

边界网格中的 SMT+Audio `1e-8/u32` 与第一阶段 job 3 在九项最终聚合指标的
六位小数、40 个窗口指标和动作计数上完全复现；低层浮点诊断存在极小差异，
因此这是行为级复现而不是逐字节完全相同。

## 4. 第一阶段：与 Source 的比较

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

## 5. 第一阶段：超参数边际规律

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

## 6. 第一阶段：更新、漂移与坍塌诊断

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

## 7. 弱更新边界网格结果

### 7.1 SMT+Audio：9 组完整结果

SR、SPL、SoftSPL、SNA、SWS 均按百分制展示。

| job | LR | interval | Reward | DTG↓ | NDTG↓ | SR↑ | SPL↑ | SoftSPL↑ | NA↓ | SNA↑ | SWS↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | `3e-9` | 32 | 11.503640 | 4.9800 | 0.305841 | 55.25 | 29.6007 | 36.5734 | 152.8845 | 43.8634 | 2.70 |
| 1 | `3e-9` | 64 | 11.445755 | 5.0680 | 0.309571 | 55.20 | 30.1683 | 36.7721 | 153.3730 | 43.9615 | 2.55 |
| 2 | `3e-9` | 128 | 11.627285 | 4.8735 | 0.296505 | 55.10 | 29.8003 | 37.0500 | 152.1700 | 43.5536 | 3.00 |
| 3 | `1e-8` | 32 | 11.606735 | 4.8135 | 0.293596 | 55.40 | 29.8248 | 36.7086 | 155.7250 | 44.4364 | 2.70 |
| 4 | `1e-8` | 64 | 11.416944 | 5.0105 | 0.301159 | 54.00 | 29.6856 | 36.8676 | 154.0040 | 43.7220 | 2.10 |
| **5** | **`1e-8`** | **128** | **11.834155** | **4.8325** | **0.295396** | **56.20** | **30.5805** | **37.4332** | **150.0830** | **45.2148** | **2.30** |
| 6 | `3e-8` | 32 | 11.496705 | 4.9290 | 0.297968 | 55.25 | 29.6096 | 36.2621 | 159.1780 | 43.5891 | 2.50 |
| 7 | `3e-8` | 64 | 11.476295 | 4.9745 | 0.297633 | 54.35 | 29.1510 | 36.5362 | 156.6690 | 42.3931 | 2.75 |
| 8 | `3e-8` | 128 | 11.429615 | 5.0020 | 0.303584 | 54.25 | 29.0087 | 36.5035 | 156.0870 | 42.8299 | 2.20 |

job 5 同时取得该模型 9 点网格的最高 SR 和最高 SPL，SPL-first 与 SR-first
选择完全一致。相对第一阶段 job 3，它提高 0.80 SR、0.7557 SPL 和 0.7246
SoftSPL，NA 减少 5.642，因此第一阶段候选已被替代。

### 7.2 ENMuS：9 组完整结果

| job | LR | interval | Reward | DTG↓ | NDTG↓ | SR↑ | SPL↑ | SoftSPL↑ | NA↓ | SNA↑ | SWS↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | `3e-9` | 32 | 14.641322 | 3.2910 | 0.205939 | 67.20 | 36.2279 | 39.8299 | 165.5160 | 51.0093 | 2.40 |
| 10 | `3e-9` | 64 | 14.390012 | 3.4785 | 0.213549 | 66.60 | 36.3140 | 40.0385 | 168.3970 | 50.9732 | 2.15 |
| **11** | **`3e-9`** | **128** | **14.859267** | **3.2590** | **0.193446** | **68.15** | **36.9083** | **40.6694** | **162.9215** | **51.8355** | **2.20** |
| 12 | `1e-8` | 32 | 14.511631 | 3.2630 | 0.206178 | 66.60 | 36.0880 | 40.0279 | 167.7850 | 50.8154 | 3.00 |
| 13 | `1e-8` | 64 | 14.449851 | 3.4190 | 0.211713 | 66.45 | 35.9897 | 39.8923 | 166.8630 | 50.0558 | 2.35 |
| 14 | `1e-8` | 128 | 14.679277 | 3.2620 | 0.202161 | 67.45 | 36.2479 | 39.9434 | 164.6205 | 51.2208 | 2.55 |
| 15 | `3e-8` | 32 | 14.691966 | 3.2815 | 0.203257 | 67.95 | 36.4763 | 40.0625 | 167.4015 | 51.2602 | 2.85 |
| 16 | `3e-8` | 64 | 14.673356 | 3.2730 | 0.208509 | 67.25 | 36.7044 | 40.4719 | 167.1125 | 51.3625 | 2.60 |
| 17 | `3e-8` | 128 | 14.325456 | 3.4670 | 0.218243 | 66.10 | 35.3750 | 39.0529 | 170.0025 | 49.9647 | 2.30 |

job 11 同样同时取得 ENMuS 9 点网格的最高 SR 和最高 SPL，且在 Reward、
NDTG、NA、SNA、SoftSPL 上也是网格最佳。该结果支持模型级调参：ENMuS
需要比 SMT+Audio 更低的 `3e-9` LR。

### 7.3 最佳配置与 Source、Tent 的比较

| 模型/方法 | SR↑ | SPL↑ | SoftSPL↑ | DTG↓ | NDTG↓ | NA↓ |
|---|---:|---:|---:|---:|---:|---:|
| SMT+Audio Source | 54.15 | 29.4229 | 36.5105 | 5.0005 | 0.306263 | 153.1720 |
| SMT+Audio Tent | 55.80 | 29.8232 | 36.8647 | 4.8190 | 0.293822 | 153.6960 |
| **SMT+Audio EAM job 5** | **56.20** | **30.5805** | **37.4332** | 4.8325 | 0.295396 | **150.0830** |
| ENMuS Source | 66.55 | 36.0480 | 40.2185 | **3.2115** | 0.195937 | 164.2560 |
| ENMuS Tent | 67.35 | 36.6857 | 40.3085 | 3.2565 | 0.200093 | 166.1920 |
| **ENMuS EAM job 11** | **68.15** | **36.9083** | **40.6694** | 3.2590 | **0.193446** | **162.9215** |

| 模型 | 相对 Source ΔSR / ΔSPL | 相对 Tent ΔSR / ΔSPL |
|---|---:|---:|
| SMT+Audio | +2.05 / +1.1576 | +0.40 / +0.7573 |
| ENMuS | +1.60 / +0.8603 | +0.80 / +0.2226 |

SMT+Audio 9 组中有 6 组同时超过 Source SR/SPL，其中 5 组还超过 Source
SoftSPL；ENMuS 分别为 7 组和 2 组。若改用冻结统一 Tent 作为更强参照，两个
模型都只有各自的最佳点同时超过 Tent SR/SPL。这表明正向区域相对 Source 较宽，
但超过已调 Tent 的余量仍小，不能仅凭开发流最大值断言显著优越。

### 7.4 边际规律与 LR×interval 交互

| 模型 | 分组 | 水平 1 SR/SPL | 水平 2 SR/SPL | 水平 3 SR/SPL |
|---|---|---:|---:|---:|
| SMT+Audio | LR | `3e-9`: 55.183/29.856 | `1e-8`: **55.200/30.030** | `3e-8`: 54.617/29.256 |
| SMT+Audio | interval | `32`: **55.300**/29.678 | `64`: 54.517/29.668 | `128`: 55.183/**29.796** |
| ENMuS | LR | `3e-9`: **67.317/36.483** | `1e-8`: 66.833/36.109 | `3e-8`: 67.100/36.185 |
| ENMuS | interval | `32`: **67.250**/36.264 | `64`: 66.767/**36.336** | `128`: 67.233/36.177 |

两个模型的单点最优都位于 interval 128，但边际均值并不支持“interval 越大
越好”的单调结论。尤其 ENMuS 中 `3e-9/u128` 最好，而 `3e-8/u128` 最差；
最佳点来自 LR 与累计更新预算的交互。可迁移的结论是 EAM 需要弱更新，不能把
`u128` 脱离 LR 单独解释为普适最优。

### 7.5 诊断与稳定性

SMT+Audio job 5 共经历 300,166 个动作步，完成 2,345 次更新，无跳过；重放
18,760 个 action steps，最终相对参数漂移为 0.00010373。全程平均动作熵为
0.343573，Source/Aux gate rate 为 0.7146/0.7212。最后四个 50-episode 窗口
平均 SR/SPL 为 58.00/33.5017，没有触发既有的末段坍塌启发式。9 个
SMT+Audio 边界配置均未触发该启发式。

ENMuS compact console 只保留最终聚合指标，没有等价的 50-episode 窗口和
最终 adapter diagnostics，本地无法补做末段稳定性检查。实验 YAML 声明了
`require_no_late_stream_collapse`，但 launcher 实际校验只覆盖完成性、有限值、
参数范围和 replay 计数，并未实现尾段坍塌判据。因此 `validation=ok` 不能被
解释为 ENMuS 已通过末段稳定性验证。

## 8. multi-source 冻结配置复验

`eam-main-multi-v1-seed0` 在 commit
`99f46dfa3010ca41d0a2898db1e88e1934ae5bb9` 的 clean worktree 上完成。
批次 2/2 运行成功，2/2 通过 manifest、配置、参数范围、诊断和
2000-episode 统计校验。其 canonical multi-source 指纹为：

| 项目 | 值 |
|---|---|
| dataset index SHA256 | `45d8dbdea540e78b01b252a3958afc4657745374d45185100d731df6a6cb849d` |
| stream-order SHA256 | `cc2f1ce8319fae6a1313750c2b1235ac39985e8d2fe6270a70fda7b12d2a6525` |
| stream-content SHA256 | `deab5e0c91abeb999563927b6c80c05bc6dfcbdd94b455b815bd303386918f2d` |
| SMT+Audio checkpoint SHA256 | `c5c039a35da13a58a8f771738208603c93d3727dbebbea0a6dbfcccd16bdddd8` |
| ENMuS checkpoint SHA256 | `3b1ccc9421b8fd6b9bad8a165528fa2323161d8b3c74642be13b2a59a5ae0464` |

两个 job 都直接使用 single-source 搜索后冻结的配置，没有在
multi-source 上重新选参。SR、SPL、SoftSPL、SNA、SWS 按百分制展示：

| 模型 | LR | interval | Reward | DTG | NDTG | SR | SPL | SoftSPL | NA | SNA | SWS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SMT+Audio | `1e-8` | 128 | 5.716312 | 7.5095 | 0.538814 | 27.50 | 14.1282 | 25.3413 | 166.3175 | 20.2154 | 1.30 |
| ENMuS | `3e-9` | 128 | 7.193380 | 7.1725 | 0.550503 | 35.40 | 17.1222 | 25.1913 | 156.8105 | 26.4333 | 0.90 |

| 模型 | 相对 Source ΔSR/ΔSPL | 相对 Tent ΔSR/ΔSPL | 相对 FSTTA ΔSR/ΔSPL |
|---|---:|---:|---:|
| SMT+Audio | +1.60 / +0.7086 | +1.40 / +1.0643 | +1.25 / +0.6112 |
| ENMuS | +0.65 / +0.0104 | -0.15 / -0.0460 | -0.45 / -0.4262 |

SMT+Audio 在直接迁移后仍保持明确正收益，其 SR/SPL 也高于当前
Tent 和 FSTTA；但 SoftSPL 比 FSTTA 低 0.2619 个百分点。ENMuS 的
SR 小幅提高，SPL 只增加 0.0104 个百分点，并未超过 Tent/FSTTA。
因此可信结论是：EAM 的弱更新设定能迁移到 multi-source，但收益大小
明显依赖导航模型，不能由 SMT+Audio 的结果外推为统一大幅改善。

## 9. 最终冻结的模型级配置与结果

共同固定项如下：

- `scope=full_transformer_plus_head`；
- `CONFIDENCE_SCALE=0.4, MEMORY_SIZE=32, BATCH_SIZE=8`；
- `EPISODIC=False, STEPS=1`，action-step replay；
- Adam，`betas=(0.9, 0.999)`、`weight_decay=0`、`max_grad_norm=0`。

模型级差异只有当前搜索出的学习率：

| 模型 | job | LR | UPDATE_INTERVAL | single SR/SPL | multi SR/SPL |
|---|---:|---:|---:|---:|---:|
| SMT+Audio | 5 | `1e-8` | `128` | 56.20 / 30.5805 | 27.50 / 14.1282 |
| ENMuS | 11 | `3e-9` | `128` | 68.15 / 36.9083 | 35.40 / 17.1222 |

上述四项数值已登记到 AVN 主对比表。SMT+Audio 配置来自同一开发流上
32 个唯一 EAM 配置的累计选择（第一阶段 24 个加边界阶段 8 个新增点），
ENMuS 为 9 选 1，因此 single-source 最大值存在选择偏差；multi-source
未重新选参，是直接迁移确认。

## 10. 证据边界

当前结果必须保留以下限制：

- 44 个 `manifest.path` 均指向原服务器 `/data1/.../manifest.json`，当前工作区
  可解析数为 0；因此无法本地复核硬件、完整有效配置和逐运行 manifest 内容。
- `avn/checkpoints/manifests/imported_pretrained.yaml` 仍将 Source checkpoint
  标为 `provenance_incomplete`；训练命令、训练 seed 和 checkpoint 选择依据
  尚未恢复。
- `avn/data/manifests/datasets.yaml` 状态为 `partial`，场景、音频与 RIR 等资产
  的完整来源和摘要仍未登记。
- Source、Tent 与 EAM 不在同一 Git commit。checkpoint 与 episode stream
  digest 对齐可支持协议级探索比较，但不能替代同提交确认运行。
- 全部搜索只有 seed 0 和一个 episode 顺序。本地没有逐 episode stats，无法做
  配对置信区间或显著性分析；开发流上的多重比较会放大最大值。
- 边界批次每张 GPU 合并并发 5 个任务；第一阶段也与其他任务共享服务器资源，
  现有 wall-clock 日志不能用于正式时延比较。
- ENMuS 缺窗口诊断，且 launcher 未真正执行 YAML 声明的末段坍塌验证。
- `avn/experiments/eam_intensity_grid.yaml` 仍描述旧的 72-job v1 搜索并标记
  `planned`；实际第一阶段权威协议是 `batch.env`、`grid.csv` 与 fixed-scope v2
  launcher。该文档差异不改变已运行配置，但正式登记前应修正。

## 11. 下一步建议

1. **保持当前冻结配置。** job 5 和 job 11 已完成 single/multi-source
   登记，不再利用这两条流继续选参。
2. **补做同 commit Source 确认时再升级证据资格。** 当前 multi-source
   EAM 已在最终 commit 上复验，但 matched Source 来自旧 commit。若正式论文
   要求同代码快照，应在同一 commit 重评 Source，并同步可解析的
   manifest、逐 episode stats 和 diagnostics。
3. **补齐 provenance。** 恢复 Source checkpoint 的训练来源和选择依据，补齐
   canonical val 数据及场景、声音、RIR 资产版本与摘要。
4. **正式时延单独测量。** 使用单 GPU、无并发、固定预热和重复运行，不复用
   本次共享资源的 wall-clock 时间。

## 12. 证据清单

- 第一阶段启动脚本：`avn/scripts/run_eam_intensity_grid.sh`
- 第一阶段旧实验定义：`avn/experiments/eam_intensity_grid.yaml`
- 第一阶段批次：
  `avn/results/logs/eam_intensity_grid/eam-intensity-grid-avn-val-rerun-v2-seed0/`
- 边界实验定义：`avn/experiments/eam_boundary_grid.yaml`
- 边界启动器：`avn/scripts/run_eam_boundary_grid.py`
- 边界批次元数据：
  `avn/results/logs/eam_boundary_grid/eam-boundary-joint-v1-seed0/batch.json`
- 边界批次状态：
  `avn/results/logs/eam_boundary_grid/eam-boundary-joint-v1-seed0/SUMMARY.json`
- 边界 18 组配置与结果：
  `avn/results/logs/eam_boundary_grid/eam-boundary-joint-v1-seed0/grid.csv`、
  `avn/results/logs/eam_boundary_grid/eam-boundary-joint-v1-seed0/metrics.csv`
- multi-source 冻结配置启动器：`avn/scripts/run_eam_main_multi_source.py`
- multi-source 批次：
  `avn/results/logs/eam_main/eam-main-multi-v1-seed0/`
- multi-source 最终结果：
  `avn/results/logs/eam_main/eam-main-multi-v1-seed0/metrics.csv`
- 逐运行日志：边界批次下 `<model>/jobs/<run_tag>/console.log`
- Source 对照：`avn/results/logs/source_reval/source-reval-v1-seed0/metrics.csv`
- Tent 对照：`avn/results/AVN_MAIN_COMPARISON.md`
- checkpoint provenance：`avn/checkpoints/manifests/imported_pretrained.yaml`
- dataset provenance：`avn/data/manifests/datasets.yaml`
