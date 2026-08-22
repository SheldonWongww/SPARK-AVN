# R2R-CE 小规模超参数搜索结果分析

## 结论摘要

- 批次：`vln-r2r-ce-small-hparam-search-v1-seed0`
- 数据：R2R-CE v1.3 unified `val_seen`，ETPNav 与 BEVBert 均为 778 个
  canonical-order episodes。
- 预算：40 个 100-episode screening jobs，加 10 个 778-episode full jobs；
  Source 使用已有正式结果，没有重新执行。
- 本轮 50/50 TTA jobs 均成功并具有可解析指标。10 个 full jobs 和 2 个复用
  Source controls 的正式 manifest、immutable identity、结果文件大小与 SHA256
  均通过本地复核。
- 以 full SPL 为主指标，10 个模型-方法组合中只有 5 个严格优于 Source：
  ETPNav 的 EAM、ATENA，以及 BEVBert 的 Tent、EAM、ATENA。其余 5 个组合
  与 Source 的 SR/SPL 相同到输出精度；本轮并没有证明“每个方法都优于
  Source”。
- ATENA 是两个模型上最优的方法：ETPNav `+1.2609` SPL，BEVBert
  `+4.9298` SPL。EAM 在两个模型上均有正收益。BEVBert–Tent 有明确收益，
  ETPNav–Tent 没有改变导航结果。
- FSTTA 的入选配置不是“效果接近 Source”，而是实际 `updates=0`：`M=16`
  大于本批次观测到的单 episode 最大 15 个高层动作，所有 fast gradients 被
  episode reset 丢弃，24 次 slow attempts 也全部跳过。两个 full run 因而都是
  Source 等价的 no-op，必须补搜有效的较小 `M`。
- FeedTTA 确实完成了每 episode 一次更新，但当前 `1e-8..1e-7` 范围在
  screening 中没有改变任何 SR/SPL 轨迹；最终选择又因“较低漂移”tie-break
  选中 `1e-8`，full 仍与 Source 相同。它需要搜索 `1e-7` 与历史不稳定的
  `5e-6` 之间的中间学习率。
- EAM 与 BEVBert–ATENA 有早期收益减弱的 fixed-order warning，但没有达到
 预设的 severe late-collapse 判据；BEVBert–ATENA 的末四分位仍明显优于
  Source。

## 证据审计

本报告重新读取每个 job 的 `job.json`、`parameters.json`、`console.log`、
`exitcode`、`metrics.json` 与 `tta_diagnostics.json`，并检查 full run 的正式
manifest 和其中列出的结果 artifact。

| 检查项 | 结果 |
|---|---|
| Screening jobs | 40/40，均为 100 episodes，退出码均为 0 |
| Full jobs | 10/10，均为 778 episodes，退出码均为 0 |
| Source jobs | 0；复用 2 个既有 Source formal manifests |
| Stage summaries | 10/10 stages 均 `complete=true`、`terminal=true`、无 error |
| 参数与任务身份 | job、parameters、metrics 中 method/setting/run tag/参数一致 |
| 指标复核 | 从 console 重解析的聚合指标与 `metrics.json` 一致 |
| Diagnostics | 50/50 文件存在、SHA256 一致、episode 数正确、无非有限值 |
| Full formal evidence | 10/10 manifests 的 SHA256 与 immutable identity 正确 |
| Full result artifacts | 30/30 文件的大小与 SHA256 正确 |
| Source evidence | 2/2 formal manifests 和 4 个 aggregate/per-episode artifacts 正确 |
| Checkpoint/data/order identity | 10/10 full manifests 与复用 Source 清单一致 |
| Per-episode order | 10/10 full outputs 均为相同的 778-episode canonical order |

统一身份为：

- TTA Git commit：`4e4cbd1429bb2065ba564bb931318aebf0aa326f`
- 搜索 spec SHA256：
  `f5b1b68b7c843465c5f75a2b425866c35c5eb94b1e26f598f69becb2e252f2ed`
- Source ETPNav：SR `67.4807`，SPL `59.9694`
- Source BEVBert：SR `68.3805`，SPL `59.9207`

有一项执行协议偏差需要保留在 provenance 中：Tent、FSTTA、EAM、FeedTTA
阶段使用 spec 固定的单 worker；REVERIE 完成后，ATENA 的 stage manifests 如实
记录了 `max_workers=3, max_per_model=2`，而原 spec 和 joint manifest 固定为
`1+1` 共卡模式。ATENA 实测 screening 峰值为 22,666 MiB GPU、38.643 GiB
cgroup memory，full 峰值为 16,967 MiB、34.368 GiB，没有资源越界；每个 job
仍从 Source 独立加载并具有完整 formal manifest。其数值证据可用于本报告，但
正式归档时应把这次经人工批准的并发变更明确写入 run provenance，不能声称它是
原 immutable scheduler profile 的逐字执行。后续应使用单独、tracked 的
continuation spec 固化此类并发变化。

## Full `val_seen` 结果

括号外的差值均为相对同模型 Source 的百分点变化。FeedTTA 和 ATENA 使用
episode-level binary navigation feedback，以 `†` 标识；它们不应与无监督方法
混作相同监督预算的公平排名。

| 模型 | 方法 | Full 超参数 | SR | ΔSR | SPL | ΔSPL | nDTW | SDTW | 漂移 | 更新数 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETPNav | Source | — | 67.4807 | — | 59.9694 | — | 71.2848 | 56.8504 | — | 0 |
| ETPNav | Tent | `lr=1e-8, last_k_ln, k=4, I=1` | 67.4807 | 0.0000 | 59.9694 | 0.0000 | 71.2848 | 56.8504 | 4.27e-5 | 6,574 |
| ETPNav | FSTTA | `lf=3e-7, ls=1e-5, M/N=16/32` | 67.4807 | 0.0000 | 59.9694 | 0.0000 | 71.2848 | 56.8504 | 0 | 0 |
| ETPNav | EAM | `lr=1e-6, I=1, mem/b=32/8` | 67.4807 | 0.0000 | 60.8084 | +0.8390 | 71.6208 | 57.1194 | 5.66e-3 | 6,505 |
| ETPNav | FeedTTA† | `lr=1e-8, last_crossmodal` | 67.4807 | 0.0000 | 59.9694 | 0.0000 | 71.2848 | 56.8504 | 1.94e-5 | 778 |
| ETPNav | ATENA† | `lq=5e-7, ls=1e-8, lambda=.5, threshold=.3` | **67.7378** | **+0.2571** | **61.2303** | **+1.2609** | **72.1812** | **57.4345** | 4.85e-4 | 778 |
| BEVBert | Source | — | 68.3805 | — | 59.9207 | — | 68.1134 | 56.5927 | — | 0 |
| BEVBert | Tent | `lr=1.5625e-5, all-LN, I=1` | 69.1517 | +0.7712 | 62.2365 | +2.3158 | 70.8578 | 58.7623 | 1.90e-2 | 6,635 |
| BEVBert | FSTTA | `lf=3e-7, ls=1e-5, M/N=16/32` | 68.3805 | 0.0000 | 59.9207 | 0.0000 | 68.1134 | 56.5927 | 0 | 0 |
| BEVBert | EAM | `lr=1e-6, I=1, mem/b=64/8` | 68.8946 | +0.5141 | 61.2692 | +1.3485 | 69.5249 | 57.6499 | 4.01e-3 | 6,598 |
| BEVBert | FeedTTA† | `lr=1e-8, last_crossmodal` | 68.3805 | 0.0000 | 59.9207 | 0.0000 | 68.1134 | 56.5927 | 1.14e-5 | 778 |
| BEVBert | ATENA† | `lq=2e-6, ls=4e-8, lambda=.25, threshold=.1` | **71.4653** | **+3.0848** | **64.8505** | **+4.9298** | **71.7386** | **60.3383** | 1.75e-3 | 778 |

所有 full candidates 都满足预设的 `SR >= Source SR - 1.5pp` gate。不过，满足
gate 只代表“没有过度牺牲 SR”，并不等价于优于 Source。

### Full-stream 稳定性

以下统计把每个 TTA episode 与相同 canonical order 中的 Source episode 配对。
CI 是 2,000 次、block length 32 的 circular moving-block bootstrap；Q1/Q4 是
首末各 194 个 episode 的 SPL 差值。该分析只描述 seed 0 的固定顺序，不等价于
跨顺序或跨 seed 的稳健性。

| 模型 | 方法 | 全流 ΔSPL [95% CI] | Q1 ΔSPL | Q4 ΔSPL | Q4−Q1 | rolling-64 范围 | 判定 |
|---|---|---:|---:|---:|---:|---:|---|
| ETPNav | Tent | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | no-op，无 collapse |
| ETPNav | FSTTA | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | zero-update no-op |
| ETPNav | EAM | +0.8390 [-0.6014, +2.2846] | +2.2728 | -1.0942 | -3.3670 | [-4.5338, +6.8840] | warning，未达 severe |
| ETPNav | FeedTTA† | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 行为近似 no-op |
| ETPNav | ATENA† | +1.2609 [-0.0320, +2.6454] | +0.9970 | +1.0521 | +0.0551 | [-3.1539, +7.5270] | 无 collapse，收益边缘显著 |
| BEVBert | Tent | +2.3158 [+0.6388, +4.0031] | +2.3146 | +2.3110 | -0.0037 | [-6.3597, +11.4468] | 稳定正收益 |
| BEVBert | FSTTA | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | zero-update no-op |
| BEVBert | EAM | +1.3485 [-0.0329, +2.8460] | +2.2883 | -0.1852 | -2.4735 | [-4.1323, +8.0150] | warning，未达 severe |
| BEVBert | FeedTTA† | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | exact no-op |
| BEVBert | ATENA† | +4.9298 [+3.1346, +6.8885] | +6.9756 | +4.5612 | -2.4144 | [-1.1848, +13.0094] | 收益减弱 warning，但 Q4 仍为正 |

采用现有 2pp materiality rule，并在同一模型的 5 个方法内对相同检验做 Holm
校正后，没有一个 full winner 被判为 `severe_fixed_order_collapse`。ETPNav–EAM、
BEVBert–EAM 和 BEVBert–ATENA 的 early-to-late erosion 达到名义 2pp，因此保留
warning；这些 erosion 均未通过校正后的显著性条件。

## 100-episode screening 排名

Screening Source：ETPNav SR/SPL 为 `60.00/51.9644`，BEVBert 为
`68.00/59.1048`。`*` 表示被晋级到 full。所有 40 个候选均通过 SR floor。
由于 console 只输出四位小数，表中的 `0.0000` 表示与 Source 在记录精度内相同。

### ETPNav

Tent 固定 `update_interval=1, AdamW, weight_decay=0, episodic=false`。

| 排名 | Job | 参数 | SR | ΔSR | SPL | ΔSPL | 漂移 | Updates |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1* | 0 | `lr=1e-8, last_k_ln, k=4` | 60.00 | 0.00 | 51.9644 | 0.0000 | 6.15e-6 | 884 |
| 2 | 2 | `lr=3e-8, last_k_ln, k=4` | 60.00 | 0.00 | 51.9644 | 0.0000 | 2.08e-5 | 884 |
| 3 | 4 | `lr=1e-7, last_ln, k=1` | 60.00 | 0.00 | 51.9644 | 0.0000 | 6.98e-5 | 884 |
| 4 | 6 | `lr=1.5625e-5, all-LN` | 60.00 | 0.00 | 51.0383 | -0.9261 | 4.95e-3 | 889 |

FSTTA 固定论文更新机制、最后 4 个 LayerNorm、AdamW；`lf/ls` 分别表示
fast/slow learning rate。

| 排名 | Job | 参数 | SR | ΔSR | SPL | ΔSPL | 漂移 | Updates |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1* | 0 | `lf=3e-7, ls=1e-5, M/N=16/32` | 60.00 | 0.00 | 51.9644 | 0.0000 | 0 | **0** |
| 2 | 2 | `lf=1e-6, ls=3e-5, M/N=16/32` | 60.00 | 0.00 | 51.9644 | 0.0000 | 0 | **0** |
| 3 | 4 | `lf=3e-6, ls=1e-4, M/N=8/16` | 60.00 | 0.00 | 51.9644 | 0.0000 | 7.44e-4 | 61 |
| 4 | 6 | `lf=5e-4, ls=1e-3, M/N=4/32` | 60.00 | 0.00 | 51.9644 | 0.0000 | 6.78e-3 | 181 |

EAM 固定 `confidence_scale=.4, Adam`。

| 排名 | Job | 参数 | SR | ΔSR | SPL | ΔSPL | 漂移 | Updates |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1* | 6 | `lr=1e-6, I=1, mem/b=32/8` | 63.00 | +3.00 | 54.7458 | +2.7814 | 3.59e-3 | 877 |
| 2 | 0 | `lr=3e-9, I=32, mem/b=32/8` | 60.00 | 0.00 | 51.9644 | 0.0000 | 1.08e-6 | 27 |
| 3 | 2 | `lr=1e-8, I=16, mem/b=32/8` | 60.00 | 0.00 | 51.9644 | 0.0000 | 6.88e-6 | 55 |
| 4 | 4 | `lr=3e-8, I=8, mem/b=32/8` | 60.00 | 0.00 | 51.9644 | 0.0000 | 4.11e-5 | 110 |

FeedTTA 固定 `gamma=.99, p=.05, alpha=-.2`，并使用目标模型 native argmax。

| 排名 | Job | 参数 | SR | ΔSR | SPL | ΔSPL | 漂移 | Updates |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1* | 0 | `lr=1e-8, last_crossmodal` | 60.00 | 0.00 | 51.9644 | 0.0000 | 3.31e-6 | 100 |
| 2 | 2 | `lr=3e-8, last_crossmodal` | 60.00 | 0.00 | 51.9644 | 0.0000 | 1.00e-5 | 100 |
| 3 | 6 | `lr=3e-8, action_head` | 60.00 | 0.00 | 51.9644 | 0.0000 | 1.01e-5 | 100 |
| 4 | 4 | `lr=1e-7, last_crossmodal` | 60.00 | 0.00 | 51.9644 | 0.0000 | 3.33e-5 | 100 |

ATENA 固定 `lambda=.5, threshold=.3, self_loss_weight=.1, AdamW`。

| 排名 | Job | 参数 | SR | ΔSR | SPL | ΔSPL | 漂移 | Updates |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1* | 6 | `lq=5e-7, ls=1e-8` | 61.00 | +1.00 | 52.4169 | +0.4525 | 9.64e-5 | 100 |
| 2 | 0 | `lq=3e-8, ls=1e-8` | 60.00 | 0.00 | 51.9644 | 0.0000 | 5.60e-6 | 100 |
| 3 | 2 | `lq=1e-7, ls=1e-8` | 60.00 | 0.00 | 51.9644 | 0.0000 | 1.91e-5 | 100 |
| 4 | 4 | `lq=3e-7, ls=3e-8` | 60.00 | 0.00 | 51.7342 | -0.2302 | 5.66e-5 | 100 |

### BEVBert

Tent 固定 `update_interval=1, AdamW, weight_decay=0, episodic=false`。

| 排名 | Job | 参数 | SR | ΔSR | SPL | ΔSPL | 漂移 | Updates |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1* | 7 | `lr=1.5625e-5, all-LN` | 68.00 | 0.00 | 59.2110 | +0.1062 | 4.61e-3 | 931 |
| 2 | 1 | `lr=1e-8, last_k_ln, k=4` | 68.00 | 0.00 | 59.1048 | 0.0000 | 2.75e-6 | 934 |
| 3 | 3 | `lr=3e-8, last_k_ln, k=4` | 68.00 | 0.00 | 59.1048 | 0.0000 | 9.31e-6 | 934 |
| 4 | 5 | `lr=1e-7, last_ln, k=1` | 68.00 | 0.00 | 59.1048 | 0.0000 | 2.19e-5 | 934 |

FSTTA 固定项同 ETPNav。

| 排名 | Job | 参数 | SR | ΔSR | SPL | ΔSPL | 漂移 | Updates |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1* | 1 | `lf=3e-7, ls=1e-5, M/N=16/32` | 68.00 | 0.00 | 59.1048 | 0.0000 | 0 | **0** |
| 2 | 3 | `lf=1e-6, ls=3e-5, M/N=16/32` | 68.00 | 0.00 | 59.1048 | 0.0000 | 0 | **0** |
| 3 | 5 | `lf=3e-6, ls=1e-4, M/N=8/16` | 68.00 | 0.00 | 59.1048 | 0.0000 | 6.57e-4 | 68 |
| 4 | 7 | `lf=1.5e-4, ls=1e-3, M/N=1/16` | 67.00 | -1.00 | 58.9957 | -0.1091 | 9.25e-3 | 933 |

EAM 固定 `confidence_scale=.4, Adam`。

| 排名 | Job | 参数 | SR | ΔSR | SPL | ΔSPL | 漂移 | Updates |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1* | 7 | `lr=1e-6, I=1, mem/b=64/8` | 70.00 | +2.00 | 60.2009 | +1.0961 | 2.42e-3 | 928 |
| 2 | 1 | `lr=3e-9, I=32, mem/b=32/8` | 68.00 | 0.00 | 59.1048 | 0.0000 | 1.02e-6 | 29 |
| 3 | 3 | `lr=1e-8, I=16, mem/b=32/8` | 68.00 | 0.00 | 59.1048 | 0.0000 | 6.35e-6 | 58 |
| 4 | 5 | `lr=3e-8, I=8, mem/b=32/8` | 68.00 | 0.00 | 59.1048 | 0.0000 | 3.83e-5 | 116 |

FeedTTA 固定项同 ETPNav。

| 排名 | Job | 参数 | SR | ΔSR | SPL | ΔSPL | 漂移 | Updates |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1* | 1 | `lr=1e-8, last_crossmodal` | 68.00 | 0.00 | 59.1048 | 0.0000 | 3.47e-6 | 100 |
| 2 | 7 | `lr=3e-8, action_head` | 68.00 | 0.00 | 59.1048 | 0.0000 | 9.43e-6 | 100 |
| 3 | 3 | `lr=3e-8, last_crossmodal` | 68.00 | 0.00 | 59.1048 | 0.0000 | 1.05e-5 | 100 |
| 4 | 5 | `lr=1e-7, last_crossmodal` | 68.00 | 0.00 | 59.1048 | 0.0000 | 3.49e-5 | 100 |

ATENA 固定 `lambda=.25, threshold=.1, self_loss_weight=.1, AdamW`。

| 排名 | Job | 参数 | SR | ΔSR | SPL | ΔSPL | 漂移 | Updates |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1* | 7 | `lq=2e-6, ls=4e-8` | 71.00 | +3.00 | 62.8317 | +3.7269 | 5.58e-4 | 100 |
| 2 | 5 | `lq=3e-7, ls=3e-8` | 68.00 | 0.00 | 59.2472 | +0.1424 | 8.90e-5 | 100 |
| 3 | 1 | `lq=3e-8, ls=1e-8` | 68.00 | 0.00 | 59.1048 | 0.0000 | 8.99e-6 | 100 |
| 4 | 3 | `lq=1e-7, ls=1e-8` | 68.00 | 0.00 | 58.8758 | -0.2290 | 2.99e-5 | 100 |

## 参数影响与失败模式

### Tent

- ETPNav 的 `1e-8`、`3e-8` 和 `1e-7` 都没有改变 SR/SPL；`1.5625e-5`
  开始改变轨迹，但 screening SPL 下降 `0.9261pp`。低区间过弱，高锚点又偏强，
  且 `1e-7` 同时改成了 `last_ln`、高锚点改成 all-LN，当前设计无法把 LR 与
  scope 的作用分开。
- BEVBert 呈现相反结果：`1.5625e-5/all-LN` 在 screening 仅 `+0.1062pp`，
  但 full 达到 `+2.3158pp`，首末四分位收益几乎相同。该组合已经有效，无需
  为“所有模型统一更低 LR”而放弃它。

### FSTTA

- 两个模型的前两个低 LR 配置均为严格的 zero-update：100-episode screening
  中分别丢弃全部 884/934 个 fast gradients；full 中丢弃全部 6,574/6,705
  个，24 次 slow attempts 全部 skip。
- 根因是 `M=16` 与当前每 episode 最多 15 个高层动作、且 episode 间 reset 的
  组合，而不是低 LR 本身。调度器在四个指标完全相同时按低 drift 排名，恰好把
  no-op 晋级到了 full。
- `M=8` 候选确实发生 61/68 次 fast updates 和 6 次 slow updates，但尚不足以
  改变聚合导航指标；高强度候选在 ETPNav 仍无变化，在 BEVBert 已轻微变差。
  因此现有数据既没有证明 FSTTA 有效，也不能据此否定 FSTTA；首先必须让低 LR
  配置真正执行更新。

### EAM

- 三个极低 LR/低频更新点在两个模型上都是 Source 等价；`1e-6, I=1` 才进入
  有效区，并在 screening/full 都获益。
- ETPNav full 为 `+0.8390pp` SPL，但收益从 Q1 `+2.2728pp` 降至 Q4
  `-1.0942pp`；BEVBert 从 `+2.2883pp` 降至 `-0.1852pp`。这是 warning，
  尚非统计确认的 severe collapse。
- 按当前实验决策不再为 EAM 扩搜索；若将来做稳健性确认，应优先换 order/seed，
  而不是继续扩大超参数网格。

### FeedTTA

- 所有 8 个 screening runs 都完成 100 次 episode-end update，参数漂移随 LR
  上升，但 SR/SPL 及逐 episode success/SPL 与 Source 完全一致。
- full 的 `1e-8` 也完成 778 次更新。BEVBert 的全部逐 episode 记录与 Source
  完全相同；ETPNav 仅一个失败 episode 的 `steps_taken` 不同，success、SPL、
  nDTW、SDTW 等任务指标均未变化。
- 因此实现路径已经运行，但 `1e-8..1e-7` 对 argmax 动作边界过弱。当前
  tie-break 又选择最低漂移的 `1e-8`，等价于主动选择最接近 no-op 的点。

### ATENA

- 两个模型都是搜索上边界的历史锚点胜出，并且 screening 到 full 的收益扩大：
  ETPNav `+0.4525 -> +1.2609pp`，BEVBert `+3.7269 -> +4.9298pp`。
- Full 查询率分别为 ETPNav `596/778=76.61%`、BEVBert
  `587/778=75.45%`；`queries + self_label_episodes = 778`，replay feature
  reconstruction error 为 0，反馈与更新计数自洽。
- BEVBert 的收益强且 bootstrap CI 不跨 0；ETPNav 有稳定的首末段正收益，但
  整体 CI 下界为 `-0.032pp`，属于边缘而非强统计证据。
- 查询率约 75% 仍然较高。若研究目标包含 feedback efficiency，应在冻结 LR 后
  单独提高 threshold；不要把 threshold 与 LR 同时扩网格，否则无法解释收益来源。

## 是否需要补搜

| 组合 | 判定 | 理由 |
|---|---|---|
| ETPNav–Tent | **需要** | 低 LR no-op，高锚点已变差，中间约两数量级为空白 |
| BEVBert–Tent | 不需要 | Full SPL +2.3158pp，且首末四分位稳定 |
| ETPNav–FSTTA | **需要，最高优先级** | 被选配置 updates=0，当前 full 不能代表 FSTTA |
| BEVBert–FSTTA | **需要，最高优先级** | 同上；有效高强度点已出现轻微退化，需要低 LR+小 M |
| ETPNav–EAM | 不补搜索；可做顺序复核 | Full 正收益但有 late-stream warning |
| BEVBert–EAM | 不补搜索；可做顺序复核 | Full 正收益但有 late-stream warning |
| ETPNav–FeedTTA | **需要，最高优先级** | 778 次更新但任务指标 no-op，LR 区间过低 |
| BEVBert–FeedTTA | **需要，最高优先级** | 逐 episode 输出与 Source 完全一致 |
| ETPNav–ATENA | 可选边界补搜 | 已有 +1.2609pp，稳定，但优胜点在上边界且 CI 刚跨 0 |
| BEVBert–ATENA | 不需要 | +4.9298pp，CI 明确为正；仅在追求更优上限时补一个边界点 |

## 最小补搜建议

不重复 Source，不重跑 EAM 和 BEVBert–Tent。仍先用 canonical 前 100 episodes
筛选，再对每个受影响 cell 运行一个 778-episode finalist。

### 必做：15 screening + 5 full，合计 20 jobs

1. ETPNav–Tent：固定 `last_k_ln, k=4, update_interval=1`，只测试
   `lr={3e-7, 1e-6, 3e-6}`。这填补 `1e-7` 到 `1.5625e-5` 的巨大空档，并消除
   LR/scope 混杂。
2. ETPNav/BEVBert–FSTTA：每个模型测试三点：
   - `lf=3e-7, ls=1e-5, M=4, N=16`
   - `lf=1e-6, ls=3e-5, M=4, N=16`
   - `lf=1e-6, ls=3e-5, M=1, N=16`

   前两点在固定窗口下比较 LR，后两点在固定 LR 下比较 fast-update 密度。任何
   `updates=0` 的候选直接判无效，不再依靠低 drift 晋级。
3. ETPNav/BEVBert–FeedTTA：固定 `last_crossmodal, gamma=.99, p=.05,
   alpha=-.2, argmax`，测试 `lr={3e-7, 1e-6, 2e-6}`。旧实验的 `5e-6`
   已显示明显崩塌，这三个点正好覆盖当前 `1e-7` no-op 与 `5e-6` 过强之间。

若同一 cell 的 screening 候选在 SR/SPL 上再次完全并列，不应再自动晋级最低
漂移点：FSTTA 要求 `updates>0`；FeedTTA 应预先规定同时 full-confirm 最低和最高
安全强度两个点，或者至少晋级产生非零导航轨迹差异的最高安全点。该规则必须在
查看 full 结果前固定。

### 可选：ATENA 边界细化

- ETPNav：`(lq,ls)=(1e-6,2e-8),(2e-6,4e-8)`，其余保持
  `lambda=.5, threshold=.3`。
- BEVBert：已有结果足够强；若只追求上限，增加
  `(lq,ls)=(4e-6,8e-8)` 一个 screening 点即可。
- 只有新点在 screening 同时提高 SPL 且满足 SR floor，才追加 full run。

若要优化反馈成本，应另开正交实验：先固定上述获胜 LR，再把 threshold 向上调，
目标是在保持 SPL 收益的同时把 query rate 从约 75% 降低。该问题不应混入本轮
“学习率是否有效”的补搜。

## 最终判断

本轮最可信的正结果是 BEVBert–ATENA、BEVBert–Tent，其次是两个 EAM 和
ETPNav–ATENA。FSTTA 与 FeedTTA 的零增益不是“方法已经搜到最优但能力弱”的
证据：FSTTA 是窗口配置导致的零更新，FeedTTA 是低学习率加 tie-break 导致的
行为 no-op。下一轮应只补 ETPNav–Tent、两组 FSTTA、两组 FeedTTA；ATENA
边界搜索属于可选优化，EAM 无需继续扩搜。
