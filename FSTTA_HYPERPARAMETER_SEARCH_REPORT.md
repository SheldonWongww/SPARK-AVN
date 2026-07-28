# FSTTA 在 AVN 上的超参数搜索实验报告

更新日期：2026-07-28

状态：探索性超参数开发报告。240 组运行均已完成并通过批次校验，但本地缺少
每个运行的 `manifest.json`，且 Source checkpoint 与数据资产 provenance 仍不完整，
因此当前结果不能直接进入论文正式主对比表。

## 1. 结论摘要

本报告解析了
`avn/results/logs/fstta_grid/fstta-smt-single-v1-seed0` 下的全部 240 组
FSTTA 日志，并使用相同 checkpoint、相同 2000 条 episode 及相同顺序的
SMT+Audio single-source Source 重评估作为参照。

1. **批次执行完整，但不是正式结果。** 240/240 组均为 `exitcode=0`、
   `runner_exitcode=0`、`validation=ok`，每组都有 9 项最终聚合指标和 40 个
   50-episode 诊断窗口；240 个配置、run tag 和最终指标向量均唯一。日志中虽有
   240 个 manifest 路径指针，但对应文件均未同步到本地，正式可用运行数为 0。
2. **按预先声明的 SPL 主指标选择，候选配置是 job 170：**
   `fast_lr=3e-7, M=16, slow_lr=1e-4, N=8`。其 SR/SPL 为
   56.00/29.5332，相对 Source 的 54.15/29.4229 提高 1.85/0.1103 个百分点。
   但 SoftSPL 下降 0.5387 个百分点，NA 增加 11.2755，SPL 增益很小，不能称为
   全指标稳定胜出。
3. **240 组中只有 2 组同时超过 Source 的 SR 与 SPL。** 22 组 SR 更高，
   2 组 SPL 更高，0 组 SoftSPL 更高；200 组 SR 低于 50%，147 组低于 25%，
   88 组低于 10%。当前搜索空间大部分区域会显著破坏导航策略。
4. **慢分支学习率是最强的经验控制变量。** `slow_lr` 从 `1e-4` 增至
   `3e-3` 时，平均 SR 从 48.75% 降至 5.07%，平均 SPL 从 24.4538% 降至
   2.4379%。`log10(slow_lr)` 与 SR/SPL 的 Pearson 相关为 -0.867/-0.859；
   该相关只说明本次网格中的伴随关系，不构成因果估计。
5. **更大的 N 和 M 总体更稳定。** `N=8`、`M=16` 分别减少慢锚更新频率和
   FAST 更新频率。最稳定的切片是 `slow_lr=1e-4, N=8`：20 组平均
   SR/SPL 为 54.115/27.6603，且没有触发本文的末段风险启发式；但该切片仍只有
   1 组 SPL 超过 Source。
6. **参数漂移与坍塌高度伴随。** 相对参数漂移与 SR/SPL 的相关为
   -0.941/-0.934。平均熵与 SR/SPL 反而正相关 0.981/0.986，说明失败运行通常
   进入了低熵、高置信度的错误策略；这不表示“保持高熵”本身会因果改善导航。
7. **当前 FSTTA 没有优于调参后的 Tent。** 同一开发流上，Tent 网格最佳
   SR/SPL 为 57.70/30.7582，均高于 FSTTA 的 56.00/29.5332。两者都是
   单 seed、同流选优结果，不能做统计显著性声明，但至少说明当前 FSTTA slow
   分支没有形成预期的稳定优势。

因此，现阶段最合理的结论不是“FSTTA 已在 AVN 上有效”，而是：**FSTTA 在一个
狭窄、低 slow learning rate、低漂移区域内存在弱正向信号，但慢锚更新在当前
AVN 移植中非常敏感，绝大多数配置会造成持续或末段策略坍塌。**

## 2. 数据范围与完整性

### 2.1 搜索空间

| 变量 | 取值 |
|---|---|
| FAST learning rate | `1e-8, 1e-7, 3e-7, 1e-6` |
| FAST 窗口 M（动作梯度数） | `2, 3, 4, 8, 16` |
| SLOW learning rate | `1e-4, 3e-4, 1e-3, 3e-3` |
| SLOW 窗口 N（episode 数） | `2, 4, 8` |
| 笛卡尔积 | `4 × 5 × 4 × 3 = 240` |

固定条件为 SMT+Audio、single-source、seed 0、2000 episodes、20 scenes ×
100 episodes、global shuffle、sample 动作、单进程、continual adaptation。
适配参数为最后 4 个 LayerNorm 的 affine 参数，共 8 个张量、2048 个标量。

固定的 FSTTA 参数为：`q=0.1, rho=0.95, tau=0.7, a=0.9, b=1.1`；FAST
和 SLOW 均使用 AdamW，`betas=(0.9, 0.99)`、`weight_decay=0`，最大梯度范数
为 1.0。FAST optimizer moment 在 episode 边界重置，SLOW optimizer moment
跨 N-episode 窗口保留。

### 2.2 批次完整性

| 检查项 | 结果 |
|---|---:|
| 计划/完成 | 240/240 |
| 唯一配置 | 240 |
| exitcode / runner_exitcode 为 0 | 240/240 |
| validation=ok | 240/240 |
| 9 项最终聚合指标完整 | 240/240 |
| 40 个诊断窗口完整 | 240/240 |
| manifest 路径指针 | 240 |
| 本地可解析 manifest | 0 |
| 当前可进入正式结果表 | 0 |

解析最终指标时使用每个日志末尾的 9 条 `Average episode ...`，而不是最后一条
`[TTA] episodes=2000 window=...`；后者只是最后 50 个 episode 的滑动窗口。
240 组日志未发现 NaN、Inf、Traceback、重复配置或重复最终指标向量。

### 2.3 Source 对齐情况

Source 与 240 个 FSTTA 运行在下列项目上完全一致：

- model、single-source setting、seed 0、2000 episodes、sample 动作；
- checkpoint SHA256：`8007dc0d...ef03`；
- stream order SHA256：`07f32759...380c`；
- stream content SHA256：`dd411c4a...fdf2`。

但 Source commit 为 `48ea6285...`，FSTTA commit 为 `a5ec8e08...`，并非同一
代码快照；同时 checkpoint manifest 标为 `provenance_incomplete`，数据资产
manifest 为 `partial`，Source 与 FSTTA 的 per-run manifest 均未同步到本地。
因此这里属于协议和二进制指纹高度对齐的探索性比较，不是形式化可复核结果。

## 3. 当前 AVN FSTTA 的实际语义

当前实现是一个以论文核心几何为基础、为 AVN 做过数值稳定化的
**FSTTA-style** 移植，不应描述为官方代码的逐行复现。

- 每个动作对预测动作分布做无监督熵最小化，不使用 success 等 episode
  feedback；动作由更新前分布采样，参数更新从下一动作开始生效。
- FAST 缓存 M 个动作梯度，利用低秩 SVD 对梯度协方差方向做逆方差加权，
  再按均值梯度范数校准并由 AdamW 更新。episode 末不足 M 个梯度会丢弃。
- 每个 episode 结束保存当前参数轨迹；累积 N 个 episode 后，SLOW 用相对旧锚
  的轨迹方向构造慢梯度，通过独立、持续的 AdamW 更新锚点，再将 FAST 参数
  snap 到新锚点。
- `q=0.1` 时最近一个 episode 对参考方向的有效权重约为 90%，所以增大 N 的
  主要作用是减少慢更新频率和改变轨迹 PCA 几何，而不是均匀平均更长历史。
- 每个 episode 重置 FAST buffer、方差历史和 FAST AdamW moment，但不重置
  已更新的模型参数；SLOW anchor 与其 AdamW moment 持续跨 episode 保留。

相对论文/公开实现的重要差异包括：AVN SMT+Audio 的 LayerNorm 维度和导航训练
方式与 DUET 不同；本实验使用 sample 而非 argmax；FAST learning rate 最高仅
`1e-6`，远低于论文的 `6e-4`；当前实现采用低秩截断、全局梯度裁剪、eval mode
以及 `weight_decay=0`。公开代码本身还存在未定义变量、慢模型未正确部署等问题，
因此本项目采用的是“论文思想对齐 + 可执行修正”的实现口径。

## 4. 与 Source 的总体比较

Source 的完整结果为：

| 方法 | SR↑ | SPL↑ | SoftSPL↑ | DTG↓ | NDTG↓ | NA↓ | SNA↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source | 54.15 | 29.4229 | 36.5105 | 5.0005 | 0.306263 | 153.1720 | 42.9696 |

240 个配置相对 Source 的计数为：

| 判据 | 配置数 | 占比 |
|---|---:|---:|
| SR 高于 Source | 22 | 9.17% |
| SPL 高于 Source | 2 | 0.83% |
| SR 与 SPL 同时更高 | 2 | 0.83% |
| SoftSPL 高于 Source | 0 | 0.00% |
| SR < 50% | 200 | 83.33% |
| SR < 25% | 147 | 61.25% |
| SR < 10% | 88 | 36.67% |

整个网格的最高 SoftSPL 仅为 36.0469，仍低于 Source 的 36.5105；整个网格
最低 NA 为 161.4405，也高于 Source 的 153.1720。换言之，即使最优候选在
SR/SPL 上有弱增益，也没有提升路径软效率或动作效率。

## 5. 最优配置与 Pareto 前沿

### 5.1 预声明规则下的推荐候选

实验 YAML 预先规定按 SPL 主排序、SR 次排序，并拒绝数值失败和末段坍塌。
在该规则下，job 170 是应冻结的开发候选：

| 配置 | fast LR | M | slow LR | N | SR | SPL | SoftSPL | DTG | NA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source | - | - | - | - | 54.15 | 29.4229 | 36.5105 | 5.0005 | 153.1720 |
| job 170 | 3e-7 | 16 | 1e-4 | 8 | **56.00** | **29.5332** | 35.9718 | 4.9295 | 164.4475 |
| Δ |  |  |  |  | +1.85 | +0.1103 | -0.5387 | -0.0710 | +11.2755 |

job 170 的最后四个 50-episode 窗口平均 SR 为 59.50%，最低为 50.00%；相对
参数漂移为 0.02209，250 次 SLOW 尝试全部成功。它是本次网格中唯一同时具有
最高 SPL、较高 SR、低漂移和良好末段行为的候选。

另一个同时超过 Source SR/SPL 的 job 49 为
`fast_lr=1e-8, M=16, slow_lr=1e-4, N=4`，SR/SPL 为 55.45/29.4339；
其 SPL 只高 0.0110 个百分点，并且被 job 170 在 SR/SPL 上同时支配。

### 5.2 SR/SPL Pareto 前沿

| job | fast LR | M | slow LR | N | SR | SPL | SoftSPL | 末四窗 SR 均值 | 参数漂移 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 170 | 3e-7 | 16 | 1e-4 | 8 | 56.00 | 29.5332 | 35.9718 | 59.50 | 0.02209 |
| 53 | 1e-8 | 16 | 3e-4 | 8 | 56.10 | 29.0397 | 35.4897 | 58.50 | 0.04537 |
| 86 | 1e-7 | 4 | 1e-4 | 8 | **56.45** | 28.8650 | 34.7425 | 55.00 | 0.02250 |

job 86 的 SR 最高，但 SPL 比 Source 低 0.5579 个百分点，因此不符合预声明的
SPL 优先选择标准。job 53 虽有较高 SR，但 SPL 仍低于 Source，且漂移约为
job 170 的两倍。

## 6. 超参数影响

### 6.1 SLOW learning rate：主导稳定性

| slow LR | 配置数 | 平均 SR | 平均 SPL | 平均 SoftSPL | 平均漂移 | 风险启发式触发数 |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 60 | 48.75 | 24.4538 | 31.0931 | 0.04472 | 21 |
| 3e-4 | 60 | 28.41 | 13.8078 | 21.3173 | 0.13995 | 54 |
| 1e-3 | 60 | 10.92 | 5.2231 | 12.7155 | 0.24412 | 60 |
| 3e-3 | 60 | 5.07 | 2.4379 | 9.1770 | 0.30025 | 60 |

最佳结果位于搜索空间最低的 `slow_lr=1e-4` 边界，而论文使用的 `1e-3` 在
AVN 上几乎全面坍塌。这表明当前 AVN 移植的慢分支更新幅度明显过大；不能把
VLN/DUET 的 slow learning rate 直接迁移到 SMT+Audio。

### 6.2 N：降低慢更新频率有效

| N | 配置数 | 平均 SR | 平均 SPL | 平均漂移 | 风险启发式触发数 |
|---:|---:|---:|---:|---:|---:|
| 2 | 80 | 17.57 | 8.5765 | 0.22056 | 75 |
| 4 | 80 | 23.34 | 11.4652 | 0.17959 | 65 |
| 8 | 80 | 28.95 | 14.4003 | 0.14663 | 55 |

N 增大时 SLOW 尝试次数由约 `2000/N` 降低，参数漂移与失败率同步下降。由于
`q=0.1` 极度偏向最近 episode，这里更可能是更新频率/轨迹几何效应，而不是
更长历史平均带来的收益。

### 6.3 M：稀疏 FAST 更新更稳定

| M | 配置数 | 平均 SR | 平均 SPL | 平均漂移 | 风险启发式触发数 |
|---:|---:|---:|---:|---:|---:|
| 2 | 48 | 16.56 | 7.9376 | 0.18939 | 43 |
| 3 | 48 | 20.09 | 9.7940 | 0.18682 | 43 |
| 4 | 48 | 21.74 | 10.6149 | 0.18165 | 40 |
| 8 | 48 | 26.45 | 13.0390 | 0.17410 | 38 |
| 16 | 48 | 31.59 | 16.0180 | 0.17934 | 31 |

更大的 M 减少 FAST optimizer step，并用更多动作梯度估计协方差，整体上更
稳定。这与 Tent 实验中“控制累计更新预算”比单纯追求更快熵下降更重要的观察
一致。由于 M 同时改变更新频率和梯度几何，当前网格不能将两种机制分离。

### 6.4 FAST learning rate：次于慢分支

| fast LR | 配置数 | 平均 SR | 平均 SPL | 平均漂移 |
|---:|---:|---:|---:|---:|
| 1e-8 | 60 | 27.20 | 13.4147 | 0.16540 |
| 1e-7 | 60 | 22.26 | 10.9980 | 0.18546 |
| 3e-7 | 60 | 21.89 | 10.7652 | 0.19011 |
| 1e-6 | 60 | 21.79 | 10.7449 | 0.18807 |

FAST learning rate 的边际相关较弱，且 job 170 使用的并非最小 fast LR。
这说明在当前双分支实现中，SLOW update 与 M/N 决定的更新预算优先于继续细化
FAST learning rate；不能从边际均值推断 `1e-8` 对所有条件都最优。

### 6.5 关键交互：slow LR × N

| slow LR | N=2 平均 SR/SPL | N=4 平均 SR/SPL | N=8 平均 SR/SPL |
|---:|---:|---:|---:|
| 1e-4 | 41.94 / 20.6271 | 50.18 / 25.0741 | **54.12 / 27.6603** |
| 3e-4 | 18.60 / 8.9455 | 27.43 / 13.2558 | 39.19 / 19.2222 |
| 1e-3 | 6.60 / 3.1269 | 10.62 / 5.0877 | 15.54 / 7.4548 |
| 3e-3 | 3.13 / 1.6066 | 5.14 / 2.4431 | 6.95 / 3.2641 |

`slow_lr=1e-4, N=8` 的 20 组中，12 组 SR 超过 Source，但只有 1 组 SPL
超过 Source。它是稳定区域，不等同于普遍有效区域。

## 7. 末段行为与内部诊断

### 7.1 “末段坍塌”启发式的口径

本文为开发筛选定义了一个风险启发式：最后四个 50-episode 窗口平均 SR
低于 25%，或比全程 SR 低超过 15 个百分点。240 组中 195 组触发；其中
190 组是末四窗均值低于 25%，79 组是相对全程下降超过 15 个百分点，两者
重合 74 组。

该指标**不是 FSTTA 论文定义的 collapse 指标**，也不能把 195 组都解释成
“后期才发生坍塌”：第一项会把全程持续低性能的配置也计入。报告中应称其为
“末段低性能/下降风险启发式”，正式持续 TTA 分析应使用逐 episode 配对轨迹、
滑窗置信区间和明确的 change-point 判据。

### 7.2 参数漂移、熵与策略坍塌

| 变量 | 与 SR 的 Pearson r | 与 SPL 的 Pearson r | 解释限制 |
|---|---:|---:|---|
| `log10(slow_lr)` | -0.867 | -0.859 | 网格伴随关系 |
| 相对参数漂移 | -0.941 | -0.934 | 漂移也可能是失败结果，而非唯一原因 |
| 平均动作熵 | +0.981 | +0.986 | 失败运行多为低熵错误策略 |
| 平均 LR scale | -0.927 | -0.919 | 与状态/梯度几何共同变化 |

最严重的失败运行并不是“不够自信”，而是持续把错误动作分布推向更低熵。
因此对 AVN 来说，单独的 entropy objective 不足以作为可靠的在线学习信号；
参数漂移阈值、Source anchor、恢复机制或导航时序一致性信号可能是必要约束。

### 7.3 SLOW 几何退化

57 个运行发生过 SLOW skip，共 3087 次：1765 次为 degenerate trajectory，
1322 次为 zero reference direction。所有运行均满足
`slow_updates + slow_skips = floor(2000/N)`；最终 FAST 参数与 SLOW anchor
的相对距离均为 0，说明 snap 语义正常执行。这些 skip 不是程序失败，而是当前
轨迹几何无法产生有效 slow direction 的信号。

## 8. 与 Tent 结果的关系

| 方法/配置 | SR | SPL | 相对 Source ΔSR/ΔSPL |
|---|---:|---:|---:|
| Source | 54.15 | 29.4229 | - |
| Tent 调参最佳 | 57.70 | 30.7582 | +3.55 / +1.3353 |
| FSTTA job 170 | 56.00 | 29.5332 | +1.85 / +0.1103 |

两项研究共同支持“AVN 的 TTA 首先是稳定性和更新预算问题”。Tent 通过极小
learning rate 或限制累计更新避免坍塌；FSTTA 虽引入 FAST/SLOW 几何，但当前
SLOW AdamW 仍能积累过大漂移。FSTTA 的优势不能只靠更复杂的梯度几何获得，
还需要显式约束慢锚步长与跨 episode 漂移。

这不是对 FSTTA 原论文结论的否定。论文使用 DUET/VLN、不同参数维度和动作
协议；当前实现也不是官方代码的逐位复现。合理结论是：**论文超参数和稳定更新
假设不能直接迁移到 SMT+Audio AVN，当前移植仍需任务级校准。**

## 9. 后续实验建议

### 9.1 当前应该冻结什么

按预声明选择规则，开发候选应冻结为：

```text
NORM_SCOPE=last_k_ln
LAST_K_LN=4
LR=3e-7
FSTTA.M=16
FSTTA.LR_SLOW=1e-4
FSTTA.N=8
EPISODIC=False
STEPS=1
```

该配置下一步应在**同一 Git commit** 下分别重跑 Source 与 FSTTA，并生成包含
配置、checkpoint、数据版本、seed、硬件和完整 SHA256 的本地 run manifest。
只有确认运行通过后，才可作为 FSTTA 主表候选；不能直接把网格最大值填入正式表。

### 9.2 主表扩展顺序

1. SMT+Audio single-source：冻结配置后的确认性复验；
2. SMT+Audio multi-source：不再重新调参，直接检验迁移；
3. ENMuS single/multi：沿用同一配置，判断跨模型泛化；
4. 若统一配置严重失败，再在独立 dev stream 上声明一个很小的模型级校准实验，
   不能在正式测试流上反复选参。

### 9.3 后续机制研究，而非继续扩大当前网格

- 当前最佳 `slow_lr` 落在最低边界，可在独立机制实验中测试更低的
  `1e-5, 3e-5, 1e-4`；
- 测试更大的 `N=8,16`，或将 slow step 归一化为相对 anchor 范数的受控步长；
- 加入 drift threshold、Source anchor interpolation 或失败回滚，验证能否保留
  FAST 的短期收益而避免 SLOW 累积漂移；
- 分离 M 的“更新频率”与“梯度协方差样本数”效应，避免把二者混为同一机制；
- 正式时延实验需单 GPU、无并发、固定预热与重复测量，当前 10 jobs/GPU 的
  wall-clock 日志不能用于论文时延表。

## 10. 证据与可复现性清单

- 批次定义：`avn/experiments/fstta_core_grid.yaml`
- 启动脚本：`avn/scripts/run_fstta_grid.sh`
- 批次元数据：
  `avn/results/logs/fstta_grid/fstta-smt-single-v1-seed0/batch.env`
- 批次状态：
  `avn/results/logs/fstta_grid/fstta-smt-single-v1-seed0/SUMMARY`
- 240 组配置：
  `avn/results/logs/fstta_grid/fstta-smt-single-v1-seed0/grid.csv`
- 逐运行日志：
  `avn/results/logs/fstta_grid/fstta-smt-single-v1-seed0/jobs/<run_tag>/console.log`
- Source 对照：
  `avn/results/logs/source_reval/source-reval-v1-seed0/metrics.csv`
- Tent 对照与前序稳定性分析：`TENT_HYPERPARAMETER_SEARCH_REPORT.md`

当前 `avn/experiments/fstta_core_grid.yaml` 仍写着 `status: planned`，与已经完成的
日志状态不一致；后续在正式登记该批次时应更新实验元数据，但不应借此把缺少
manifest 的开发日志升级为正式结果。
