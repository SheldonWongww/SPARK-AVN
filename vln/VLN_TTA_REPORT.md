# VLN Test-Time Adaptation 实验报告

更新日期：2026-08-14

当前状态：**DUET、HAMT、GOAT、ETPNav、BEVBert 的 Source validation 与 test 轨迹已下载；HAMT-R2R 的论文最终 `vitbase-finetune-e2e` 替换评估也已跑完，但尚未补成可进入 formal Source 汇总的 run manifest。Tent、FSTTA、EAM、FeedTTA、ATENA 的固定顺序 `val_seen` 超参数搜索均已完成并下载，本文已按 SR 第一、SPL 第二的统一口径重排逐模型探索性最优与跨模型共享候选，并登记反馈预算和 late-collapse 审计。StreamVLN 因推理耗时暂缓。当前 TTA 数值仍属于调参证据，须通过零更新适配器一致性审计后才能进入正式主表。**

本文持续维护 R2R、REVERIE 和 R2R-CE 三个 benchmark 的 Source/TTA 对比、实验口径、论文参考值与后续小规模超参数方案。机器可读的本地结果见 [`results/source_baselines_20260810.json`](results/source_baselines_20260810.json)，论文 TTA 参考值见 [`results/legacy/published_tta_metrics.json`](results/legacy/published_tta_metrics.json)。

## 1. 证据范围与结果等级

- 五个已完成模型的原始日志位于 `results/source/grouped-source-20260810T080743Z/`：124 个文件、约 111 MiB、24 份 per-setting run manifest。该目录被 Git 忽略，不会提交 raw logs、预测文件或其他大产物。
- Tent、FSTTA、EAM 的调参证据分别位于 `results/logs/hparam_search/<method>/<batch>/` 与 `results/tuning/<method>/<batch>/`。本次从服务器同步 12,530 个允许文件，传输归档 SHA256 为 `0db2302515a3e349dfc0debe74dfde53c4dd41f5d7ccee2ed003fb22133a59a7`；归档明确排除了 checkpoints、predictions、videos、TensorBoard 和常见模型/数组二进制。这些目录均被 Git 忽略。
- 前三种方法的 post-hoc late-collapse 结果位于 `results/audits/hparam_search/`；服务器生成归档 SHA256 为 `cb88d4aba2c6efefeced425f5cb41931185778b3d5d7d987077317c7fb9e0490`。连续环境使用逐 episode 证据和 block bootstrap，离散环境只有 256-prefix 与 remainder 的粗粒度分解。
- FeedTTA 的日志、tuning evidence 与 late-collapse 审计另外同步了 3,743 个允许文件，传输归档 SHA256 为 `0a49a161928b0b60abaaa633dd7ba8eba1512754e436631cab2885680d1cd674`；使用与前三种方法相同的二进制和大产物排除规则。
- ATENA 的日志、tuning evidence 与 late-collapse 审计同步了 2,608 个允许文件，传输归档 SHA256 为 `f0b4b471e9c48f6f568d7263ae010abf832894047ebf0ad97b48fa87531acbef`；已排除 checkpoints、predictions、videos、TensorBoard 以及常见模型/数组二进制。stage1 的两个首次 OOM 尝试作为 `attempt-00` provenance 保留，降至最多 4 路并发后重试成功；表中任务数只统计 logical jobs，不重复计入归档尝试。
- 本报告中的“本地 Source”只来自状态为 `completed` 的 validation manifest。每份 manifest 均绑定运行 commit `0c6e38b`、完整命令、seed、checkpoint SHA256、dataset/episode-order SHA256 和 NVIDIA vGPU-32GB 硬件信息。
- “论文参考”不是本工作区复现结果，只用于核对趋势和规划超参数；不能与本地结果混称为同一协议下的结果。
- test split 不提供本地 ground truth，因而只登记提交文件状态，不填写本地 test 指标。正式 test 指标必须来自相应榜单。
- ATENA 和 FeedTTA 消费 episode 成功/失败二值反馈，必须标记为 **feedback-supervised TTA**；它们不能与 Tent、FSTTA、EAM 等无标签方法混称为 unsupervised TTA。

## 2. 固定评估协议

本工作区统一使用 `val_seen → val_unseen → test` 的 split 顺序，但每个 split 都从原始 checkpoint 启动独立新进程，**不会把 val_seen 的适应状态带入 val_unseen 或 test**。split 内部顺序由已跟踪的 episode-order manifest 固定：

| Benchmark | val_seen | val_unseen | test |
|---|---:|---:|---:|
| R2R | 1,021 | 2,349 | 4,173 |
| REVERIE | 1,423 | 3,521 | 6,292 |
| R2R-CE v1.3 | 778 | 1,839 | 3,408 |

正式对比采用单进程、单环境、batch size 1、seed 0 和固定顺序。Source 对 episode 顺序理论上不敏感；continual TTA 会受顺序影响，因此所有方法必须消费相同 manifest。论文式随机 shuffle 复验只能作为单独的 robustness 表，不能覆盖固定顺序主表。

## 3. 当前本地 Source 结果

表中比率均为百分数，`NE/TL` 单位为米。这里只展示主指标；nDTW、SDTW、CLS 及精确小数保存在机器汇总中。

### 3.1 R2R（离散环境）

| 模型 | val_seen SR | val_seen SPL | val_unseen SR | val_unseen SPL | test |
|---|---:|---:|---:|---:|---|
| DUET | 78.84 | 72.88 | 71.52 | 60.41 | 4,173 条轨迹，格式验证通过 |
| HAMT fixed-feature（已废弃） | 60.63 | 58.73 | 55.34 | 52.71 | 旧轨迹仅保留 provenance |
| HAMT e2e（论文最终权重；formal manifest 待补） | — | — | — | — | 4,173 条轨迹已生成 |
| GOAT | **84.82** | **80.05** | **78.12** | **67.58** | 4,173 条轨迹，格式验证通过 |

HAMT-R2R 的活动配置已切换到论文最终 `vitbase-finetune-e2e/best_val_unseen` checkpoint 和配套 `r2r.e2e.ft.22k` 特征。原始替换日志得到 val-seen `SR/SPL=75.61/72.18`、val-unseen `66.24/61.51`，三个 TTA 批次的 full-val `final_controls` 也重复得到 val-seen `75.61/72.18`。但独立 Source 目录尚缺 formal run manifest，因此这些值只作为 matched tuning evidence 使用，暂不写入上面的 formal Source 行；fixed-feature 行继续只保留 provenance。

### 3.2 REVERIE（离散环境 + 目标物体定位）

| 模型 | val_seen SR/SPL | val_seen RGS/RGSPL | val_unseen SR/SPL | val_unseen RGS/RGSPL | test |
|---|---:|---:|---:|---:|---|
| DUET | 71.75 / 63.94 | 57.41 / 51.14 | 46.98 / 33.73 | 32.15 / 23.03 | 6,292 条轨迹，格式验证通过 |
| HAMT | 43.29 / 40.19 | 27.20 / 25.18 | 32.95 / 30.20 | 18.92 / 17.28 | 6,292 条轨迹，格式验证通过 |
| GOAT | **80.74 / 73.44** | **64.93 / 58.82** | **53.82 / 37.52** | **39.17 / 27.00** | 6,292 条轨迹，格式验证通过 |

DUET 和 HAMT 与各自论文中的对应 Source 行基本一致。GOAT 本地值与 ATENA 论文采用的 GOAT Source 行一致，但与 GOAT 原论文表略有差异，说明比较时必须同时固定 checkpoint、sidecar 字典和评估实现，不能只按模型名称合并数值。

### 3.3 R2R-CE v1.3（连续环境）

| 模型 | val_seen NE/OSR/SR/SPL | val_unseen NE/OSR/SR/SPL | test |
|---|---:|---:|---|
| ETPNav | 3.58 / 74.42 / 67.48 / 59.97 | 4.83 / 62.59 / 55.90 / 48.30 | 3,408 条轨迹，修复后的验证器复核通过 |
| BEVBert | 3.62 / **75.32 / 68.38** / 59.92 | **4.53 / 66.45 / 58.24** / 48.26 | 3,408 条轨迹，修复后的验证器复核通过 |
| StreamVLN | — | — | 等待 Source 运行完成 |

这里使用统一的 R2R-CE v1.3 episode 起点和顺序。ETPNav、BEVBert 论文值来自其 v1.2 preprocessing，不能与本表做严格逐点比较。两个 CE test 轨迹在生成后被旧验证器错误地以“相邻 3D 位移必须不超过 0.25m”拒绝，因此原 run manifest 保留 `failed`；commit `79b40ac` 移除该错误限制后，原生轨迹均通过 3,408 episode 完整性检查。它们是可提交文件，但不是状态为 `completed` 的正式 test run。

## 4. 已发表 VLN-TTA 参考值

下表只列主要指标，数值来自 [FSTTA（ICML 2024）](https://proceedings.mlr.press/v235/gao24p.html)和 [ATENA（arXiv:2506.06630v1）](https://arxiv.org/abs/2506.06630v1)。FSTTA 原文通常报告 5 次 shuffled stream 均值；ATENA 报告 3 个随机 seed，并使用二值 episode feedback。`†` 表示 feedback-supervised，`‡` 表示 ATENA 作者的重实现而非 FSTTA 官方结果。

| Benchmark | Base | 方法 | val_seen | val_unseen | test_unseen |
|---|---|---|---:|---:|---:|
| R2R | DUET | FSTTA | SR/SPL 79 / 73 | 75 / 62 | — |
| R2R | DUET | ATENA† | 80 / 75 | 75 / 66 | — |
| R2R | GOAT | ATENA† | 85.01 / 80.13 | 79.01 / 69.30 | — |
| REVERIE | HAMT | Tent‡ | SR/SPL/RGSPL 43.43 / 40.78 / 25.81 | 30.56 / 28.23 / 14.48 | 23.73 / 21.78 / 10.82 |
| REVERIE | HAMT | FSTTA‡ | 42.87 / 39.56 / 24.58 | 32.89 / 30.51 / 17.20 | 30.39 / 26.65 / 13.61 |
| REVERIE | HAMT | ATENA† | 57.34 / 48.08 / 29.60 | 34.00 / 30.96 / 17.51 | 32.55 / 28.38 / 14.32 |
| REVERIE | DUET | Tent | 71.89 / 64.06 / 50.41 | 47.55 / 33.99 / 23.32 | 52.61 / 36.17 / 22.16 |
| REVERIE | DUET | FSTTA | 75.48 / 65.84 / 52.23 | 54.15 / 36.41 / 23.56 | 53.40 / 36.43 / 22.40 |
| REVERIE | DUET | ATENA† | 84.33 / 74.31 / 59.99 | 68.11 / 45.82 / 31.26 | 54.28 / 40.70 / 25.01 |
| REVERIE | GOAT | Tent‡ | 80.74 / 73.47 / 58.75 | 53.51 / 37.49 / 26.99 | 57.28 / 39.82 / 26.97 |
| REVERIE | GOAT | FSTTA‡ | 80.74 / 73.42 / 58.82 | 53.79 / 37.50 / 26.95 | 57.52 / 39.49 / 26.82 |
| REVERIE | GOAT | ATENA† | 83.35 / 76.45 / 61.60 | 67.66 / 53.15 / 39.80 | 62.03 / 46.82 / 31.54 |
| R2R-CE | ETPNav | FSTTA‡ | SR/SPL 66 / 59 | 57 / 49 | — |
| R2R-CE | ETPNav | ATENA† | 67 / 61 | 58 / 49 | — |
| R2R-CE | BEVBert | FSTTA | 69 / 60 | 60 / 51 | 60 / 50 |
| R2R-CE | BEVBert | ATENA† | 71 / 64 | 60 / 51 | — |

这些值只能在各自论文内部用于 Source→TTA 增益判断。例如 ATENA 在 REVERIE 上提升显著，但它使用人工/自预测的 episode 成败标签；FSTTA 在 DUET 上有效，在 ATENA 对 HAMT/GOAT 的重实现中则几乎无增益。两类结论回答的是不同监督条件下的问题，不应只按最高 SR 排名。

## 5. 本地 Source 与论文值的初步分析

1. **DUET 的复现最稳定。** R2R 的本地 val-unseen `71.52/60.41` 与论文四舍五入后的 `72/60` 一致；REVERIE 两个 validation split 与论文行一致，可作为第一批 TTA 集成基线。
2. **HAMT-R2R 的 checkpoint 变体偏差已经消除。** 旧 fixed-feature 模型在 val-unseen 为 `55.34/52.71`；新的 end-to-end 模型原始日志达到 `66.24/61.51`，并成为所有 HAMT-R2R TTA 的 matched Source。现在剩余的是补齐 formal Source manifest，而不是重跑模型。
3. **GOAT 是当前离散 Source 最强模型。** 它在 R2R 和 REVERIE 两个 validation split 上均领先，但复杂的 back-door/front-door sidecar 使 checkpoint identity 之外的资产 provenance 同样重要。
4. **R2R-CE 必须区分 v1.2 与 v1.3。** 当前 ETPNav、BEVBert 和 StreamVLN 被统一到 v1.3 顺序，适合本项目横向比较；原论文 v1.2 数值仅作背景。
5. **StreamVLN 不登记中间值。** 其单 episode 推理成本远高于其他模型，在 778/1,839 episode 完成前，任何滚动均值都不是正式 benchmark 结果。

## 6. TTA 实现与小规模超参数策略

### 6.1 论文默认锚点

| 方法 | 监督类型 | 首选论文设置 |
|---|---|---|
| Tent | 无监督 | AdamW；batch size 1 学习率 `0.001/64 = 1.5625e-5`；只更新 normalization affine 参数。需要明确采用 action-step 更新还是 ATENA 论文的 episode-end 版本。 |
| FSTTA | 无监督 | 离散环境：最后 4 个 LN，`M=3, N=4, lr_fast=6e-4, lr_slow=1e-3, q=0.1, rho=0.95, tau=0.7`，动态缩放截断 `[0.9,1.1]`；R2R-CE：`M=7, N=4, lr_fast=5e-4, lr_slow=1e-3`。 |
| EAM | 无监督 | 当前没有这三项 VLN benchmark 上与本项目模型一一对应的论文默认值；应标记为 VLN port，先做梯度/显存 smoke，再采用极小学习率邻域，不直接照搬 AVN winner。 |
| FeedTTA | 二值反馈 | episode-end REINFORCE；必须用 `FeedTTA†` 标记。VLN 上没有可直接套用到六模型的已验证默认点，先使用论文锚点并只搜索学习率/折扣的少量邻点。 |
| ATENA | 二值反馈 | 论文搜索 `delta ∈ {0.1,0.2,0.3}`、`lambda = 0.0…1.0`（步长 0.1）、学习率 `{5e-6,1e-6,5e-7}`；正式运行优先采用各 released script 的 model-specific 配置，并记录 query/self 学习率、binary-loss 权重和 feedback ratio。 |

FSTTA 发布代码的 REVERIE 锚点为 `lr_fast=6e-4`、`lr_slow=1e-3`、`M=3`、`N=4`。ATENA 发布脚本还给出 DUET-R2R 的 `lr_query=8e-7, lr_self=1e-7, lambda=0.75, delta=0.1` 和 DUET-REVERIE 的 `5e-6, 1e-7, 0.5, 0.1`；这些 model-specific 点比无约束大网格更适合作为本项目起点。

### 6.2 当前逐模型 R2R 超参数规则

上一轮分阶段搜索不足以回答“每个导航模型上该 TTA 方法能达到的最好效果”，
因此当前协议改为按 `导航模型 × benchmark × TTA 方法` 独立选择参数。第一批只
搜索 R2R 的 DUET、HAMT、GOAT；REVERIE 与 R2R-CE 后续分别建立自己的搜索，
不能直接把 R2R winner 当作已经验证的最终参数。跨模型共享配置仍可作为公平性
补充分析，但不再替代逐模型 winner。

1. 当前 R2R 搜索直接运行完整固定顺序的 1,021 个 `val_seen` episodes，不使用 smoke、256-prefix、分轮筛选或晋级。
2. 每种方法使用一次性笛卡尔积：Tent 40、FSTTA 81、EAM 320、FeedTTA 250、ATENA 100 个配置；三个模型合计 2,373 个 TTA jobs。
3. 每个模型/方法独立按 **SR 第一、SPL 第二、参数漂移更小、更新次数更少** 选择 winner，不要求同一方法在三个模型上使用相同参数。
4. 三个标准 argmax Source 是所有增益的唯一基线。FeedTTA 的三个 sampled no-update Source 只诊断采样策略差异，不参与正式增益计算。
5. 参数冻结后，`val_unseen`、test 和 order seeds `0/1/2` 只能消费对应模型/benchmark 的 winner，不能利用这些 split 重新选择超参数。
6. 每个 split 从原始 Source checkpoint 重新开始，并记录配置、checkpoint/dataset digest、episode 顺序、更新范围、更新次数、反馈比例、显存、内存和墙钟时间。
7. 无监督方法与二值反馈方法分表；FeedTTA/ATENA 必须带 `†` 并报告 feedback/query budget。
8. StreamVLN 暂不进入大规模 Cartesian 搜索；只在更合适的长时评估服务器上运行默认点和少量保守邻点。

### 6.3 冻结参数后的零更新适配器一致性审计

超参数搜索结束后、正式解释增益前，必须完成独立的 256-episode
canonical-prefix 审计。计划固定为 56 个任务：`5 方法 × 8 setting = 40`
个零写入适配器任务，外加 8 个 argmax Source 与 8 个 sampled Source。
FeedTTA 只与 sampled Source 配对，其余方法只与 argmax Source 配对。

这里的“零更新”不是把学习率设成 0，也不是绕开适配器。审计模式仍执行
loss、backward、FSTTA fast/slow 调度、EAM replay/gate、FeedTTA episode
feedback/SGR、ATENA query/self-label/replay 等完整控制流，只在最后的
optimizer/direct-copy 参数写入边界拦截并计数。一个组合只有同时满足以下
条件才通过：suppressed attempts 大于 0 且等于所有 write attempts，
`updates=0`、`slow_updates=0`、relative drift 精确为 0，完整部署模型（全部
parameter 与 buffer）适配前后 SHA256 完全相同；FSTTA fast/slow、EAM
reliable replay、FeedTTA feedback/policy-gradient、ATENA gate/self-prediction/
replay 均有正向执行证据；256 个 episode ID/顺序正确；action trajectory SHA256、
逐 episode 输出证据和聚合指标与 matched Source 精确一致。

审计使用独立 spec、job schema 和 `results/audits/adapter_parity/` 结果空间，
并绑定 audit commit、搜索阶段冻结参数文件、资产/环境 manifest、dataset 与
episode-order digest；每个 job 还生成不可变 formal run manifest，钉住精确
256 条 prefix manifest 及其 canonical parent，并在执行前后检查干净 Git
状态，同时要求 `verify_preflight.py --hash all` 通过。它只
证明“适配器在不允许写参数时与 Source 路径等价”，不产生可进入论文对比表
的 TTA 性能结果。

## 7. Tent、FSTTA、EAM、FeedTTA 与 ATENA 的 `val_seen` 超参数搜索

本节统一采用 **SR 第一、SPL 第二** 的分析口径。逐 setting 排名先比较
完整 `val_seen` 的 `ΔSR`，相同时再比较 `ΔSPL`；REVERIE 的 RGS/RGSPL
只作为目标定位补充指标。**所有方法均相对标准 argmax Source 报告结果**；
FeedTTA† 的 sampled no-update control 只保留为内部适配诊断。所有增益均为
固定 canonical order、seed 0 下的绝对百分点。

需要特别说明：历史调度器用 R2R/R2R-CE 的 SPL 和 REVERIE 的 RGSPL
完成分阶段晋级。因此下面是对每组 **5 个已完成 full-val finalist** 的
SR-first 事后重排，不是一次从原始搜索空间重新执行的 SR 驱动搜索。40 个
逐 setting winner 中有 13 个与原 scheduler rank 1 不同；原 rank 仍保留在
附表和 CSV 中以维护 provenance。

### 7.1 批次与完整性

五个方法批次的 `SUMMARY.json` 均为 `complete=true`、`terminal=true`、
`errors=[]`。Tent/FSTTA/EAM 使用 search spec SHA256
`389cd63a7d52f68040c920d41c9e0ead331189945e0c313e05f772f8bf645525`；
FeedTTA/ATENA 使用精简 spec SHA256
`24dee7711e9e3168c2340ffbace4678545ce8a8d4681cb5c0438cf36e1203219`。

| 方法 | Batch | Git commit | smoke | controls | stage1 | stage2 | stage3 | final controls | final | 总任务 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tent | `vln-tta-hparam-final-20260810T182823Z` | `9ad8a42` | 8/8 | 8/8 | 280/280 | — | — | 8/8 | 40/40 | 344 |
| FSTTA | `vln-tta-hparam-final-20260810T182823Z` | `9ad8a42` | 8/8 | 8/8 | 160/160 | 64/64 | 80/80 | 8/8 | 40/40 | 368 |
| EAM | `vln-val-seen-hparam-v1-seed0` | `25ead04` | 8/8 | 8/8 | 280/280 | 80/80 | 64/64 | 8/8 | 40/40 | 488 |
| FeedTTA† | `vln-val-seen-hparam-compact-v1-seed0` | `8572227` | 8/8 | 8/8 | 120/120 | 176/176 | — | 8/8 | 40/40 | 360 |
| ATENA† | `vln-val-seen-hparam-compact-v1-seed0` | `8572227` | 8/8 | 8/8 | 40/40 | 80/80 | 64/64 | 8/8 | 40/40 | 248 |

### 7.2 逐模型 SR-first 最优

下表先汇总每种方法的 8 个 SR-first winner。`SR > 0` 和 `SPL > 0`
均采用严格正增益；`0.00` 视为持平而非提升。FeedTTA†、ATENA† 的数值
属于 feedback-supervised TTA，不能与前三种无监督方法直接合并排名。

| 方法 | SR > 0 | 平均 ΔSR | 中位 ΔSR | SPL > 0 | 平均 ΔSPL | winner 改变 |
|---|---:|---:|---:|---:|---:|---:|
| Tent | 6/8 | +0.27 | +0.21 | 7/8 | +1.10 | 4/8 |
| FSTTA | 5/8 | +0.36 | +0.18 | 6/8 | +0.48 | 2/8 |
| EAM | 7/8 | +0.64 | +0.50 | 8/8 | +1.47 | 4/8 |
| FeedTTA† | 4/8 | +0.48 | -0.28 | 2/8 | -1.93 | 2/8 |
| ATENA† | 7/8 | +3.01 | +2.47 | 7/8 | +3.50 | 1/8 |

#### Tent

| Setting | ΔSR（主） | ΔSPL（次） | 核心参数 | 原 scheduler rank | 后段审计 |
|---|---:|---:|---|---:|---|
| DUET-R2R | -0.19 | +1.11 | `lr=1e-5, I=2` | 3 | `no_coarse_flag` |
| HAMT-R2R | +0.59 | +1.38 | `lr=0.0001, I=1` | 1 | `no_coarse_flag` |
| GOAT-R2R | +0.10 | +0.27 | `lr=1e-5, I=1` | 2 | `no_coarse_flag` |
| DUET-REVERIE | +0.14 | +1.02 | `lr=0.0001, I=16` | 2 | `no_coarse_flag` |
| HAMT-REVERIE | +0.28 | +1.02 | `lr=1.5625e-5, I=1` | 1 | `no_coarse_flag` |
| GOAT-REVERIE | +0.57 | +0.76 | `lr=3e-5, I=4` | 2 | `no_coarse_flag` |
| ETPNav-R2R-CE | -0.6427 | -0.0404 | `lr=3e-5, I=2` | 1 | `severe_fixed_order_collapse` |
| BEVBert-R2R-CE | +1.2853 | +3.2639 | `lr=0.0001, I=4` | 1 | `warning_fixed_order_degradation` |

Tent 在 6/8 个 setting 上提高 SR；DUET-R2R 虽提高 SPL `+1.11`，SR 仍下降 `0.19`，ETPNav-R2R-CE 则同时下降并出现 severe collapse。

#### FSTTA

| Setting | ΔSR（主） | ΔSPL（次） | 核心参数 | 原 scheduler rank | 后段审计 |
|---|---:|---:|---|---:|---|
| DUET-R2R | +0.20 | +0.18 | `lf=0.0006, ls=0.0003, M=8, N=4` | 1 | `no_coarse_flag` |
| HAMT-R2R | +0.00 | +0.12 | `lf=0.0018, ls=0.001, M=1, N=16` | 1 | `coarse_warning` |
| GOAT-R2R | -0.10 | -0.09 | `lf=0.0006, ls=0.001, M=3, N=4` | 1 | `no_coarse_flag` |
| DUET-REVERIE | -0.49 | +0.40 | `lf=0.0018, ls=0.001, M=4, N=32` | 1 | `no_coarse_flag` |
| HAMT-REVERIE | +2.11 | +2.30 | `lf=0.0018, ls=0.0003, M=2, N=4` | 1 | `no_coarse_flag` |
| GOAT-REVERIE | +0.15 | +0.19 | `lf=0.00018, ls=0.001, M=2, N=16` | 1 | `no_coarse_flag` |
| ETPNav-R2R-CE | +0.2571 | -0.0064 | `lf=0.0005, ls=0.001, M=4, N=8` | 2 | `no_fixed_order_collapse_signal` |
| BEVBert-R2R-CE | +0.7712 | +0.7234 | `lf=0.00015, ls=0.001, M=1, N=16` | 2 | `no_fixed_order_collapse_signal` |

FSTTA 在 5/8 个 setting 上提高 SR。ETPNav-R2R-CE 的 SR-first 配置为 `ΔSR=+0.2571`、`ΔSPL=-0.0064`，直接展示了 SR 优先时必须同时披露的路径效率代价。

#### EAM

| Setting | ΔSR（主） | ΔSPL（次） | 核心参数 | 原 scheduler rank | 后段审计 |
|---|---:|---:|---|---:|---|
| DUET-R2R | +0.49 | +1.66 | `lr=3e-6, I=1, c=0.6, mem/b=32/8` | 1 | `coarse_warning` |
| HAMT-R2R | +0.98 | +1.39 | `lr=3e-6, I=1, c=0.4, mem/b=32/8` | 1 | `coarse_warning` |
| GOAT-R2R | +0.00 | +0.60 | `lr=1e-5, I=4, c=0.4, mem/b=64/16` | 1 | `no_coarse_flag` |
| DUET-REVERIE | +0.84 | +2.13 | `lr=1e-6, I=1, c=0.4, mem/b=32/8` | 2 | `no_coarse_flag` |
| HAMT-REVERIE | +1.69 | +2.32 | `lr=1e-6, I=1, c=0.4, mem/b=16/4` | 4 | `no_coarse_flag` |
| GOAT-REVERIE | +0.22 | +1.60 | `lr=3e-6, I=4, c=0.5, mem/b=32/8` | 1 | `coarse_warning` |
| ETPNav-R2R-CE | +0.3856 | +0.6995 | `lr=1e-6, I=1, c=0.4, mem/b=16/4` | 2 | `no_fixed_order_collapse_signal` |
| BEVBert-R2R-CE | +0.5141 | +1.3485 | `lr=1e-6, I=1, c=0.4, mem/b=64/8` | 2 | `warning_fixed_order_degradation` |

EAM 有 7/8 个严格 SR 正增益，GOAT-R2R 持平；8/8 个 SPL 均为正。按 SR 重排后，HAMT-REVERIE 从 scheduler rank 4 升为第一，两个 CE 模型也都改用原 rank 2，说明旧 SPL/RGSPL 排名会掩盖更好的成功率。

#### FeedTTA†

| Setting | ΔSR（主） | ΔSPL（次） | 核心参数 | 原 scheduler rank | 后段审计 |
|---|---:|---:|---|---:|---|
| DUET-R2R | -1.46 | -2.58 | `lr=5e-6, γ=0.8, p=0.01, α=-0.05` | 2 | `coarse_warning` |
| HAMT-R2R | -5.29 | -6.85 | `lr=1e-5, γ=1, p=0.05, α=-0.2` | 1 | `coarse_warning` |
| GOAT-R2R | -0.69 | -0.06 | `lr=5e-6, γ=0.99, p=0.1, α=-0.2` | 1 | `no_coarse_flag` |
| DUET-REVERIE | +5.27 | +2.14 | `lr=1e-5, γ=0.8, p=0.1, α=-0.05` | 1 | `coarse_warning` |
| HAMT-REVERIE | +4.92 | +2.20 | `lr=5e-6, γ=0.99, p=0.1, α=-0.1` | 1 | `coarse_warning` |
| GOAT-REVERIE | +2.11 | -1.22 | `lr=1e-5, γ=1, p=0.05, α=-0.1` | 1 | `no_coarse_flag` |
| ETPNav-R2R-CE | -1.1568 | -8.2484 | `lr=5e-6, γ=0.99, p=0, α=-0.2` | 4 | `warning_fixed_order_degradation` |
| BEVBert-R2R-CE | +0.1285 | -0.8121 | `lr=5e-6, γ=0.8, p=0.05, α=-0.2` | 1 | `no_fixed_order_collapse_signal` |

以标准 argmax Source 为基线后，FeedTTA† 只有 4/8 个 setting 提高 SR，只有 2/8 个提高 SPL，正收益主要集中在 REVERIE。DUET-R2R 为 `ΔSR=-1.46, ΔSPL=-2.58`；ETPNav-R2R-CE 虽相对 sampled control 有改善，但相对标准 Source 仍为 `-1.1568/-8.2484`。

#### ATENA†

| Setting | ΔSR（主） | ΔSPL（次） | 核心参数 | 原 scheduler rank | 后段审计 |
|---|---:|---:|---|---:|---|
| DUET-R2R | +0.98 | +2.91 | `lq=3.2e-6, ls=4e-7, λ=0.75, δ=0.2, w=0.1` | 1 | `no_coarse_flag` |
| HAMT-R2R | +1.86 | +1.98 | `lq=3.2e-6, ls=4e-7, λ=0.25, δ=0.1, w=0.1` | 1 | `coarse_warning` |
| GOAT-R2R | -0.39 | -0.16 | `lq=8e-7, ls=1e-7, λ=0.75, δ=0.1, w=0.1` | 1 | `no_coarse_flag` |
| DUET-REVERIE | +5.62 | +5.58 | `lq=5e-6, ls=1e-7, λ=0.5, δ=0.1, w=0.25` | 1 | `coarse_warning` |
| HAMT-REVERIE | +8.22 | +5.61 | `lq=5e-6, ls=1e-7, λ=0.5, δ=0.2, w=0.25` | 2 | `coarse_warning` |
| GOAT-REVERIE | +4.43 | +5.85 | `lq=1e-5, ls=2e-7, λ=0.5, δ=0.1, w=0.25` | 1 | `coarse_warning` |
| ETPNav-R2R-CE | +0.2571 | +1.2609 | `lq=5e-7, ls=1e-8, λ=0.5, δ=0.3, w=0.1` | 1 | `no_fixed_order_collapse_signal` |
| BEVBert-R2R-CE | +3.0848 | +4.9298 | `lq=2e-6, ls=4e-8, λ=0.25, δ=0.1, w=0.1` | 1 | `warning_fixed_order_degradation` |

ATENA† 有 7/8 个严格 SR 正增益；GOAT-R2R 的最佳 finalist 仍为 `ΔSR=-0.39, ΔSPL=-0.16`。HAMT-REVERIE 按 SR 重排后选择 `δ=0.2`（原 scheduler rank 2），而不是 RGSPL 驱动的 `δ=0.1`。

### 7.3 旧分阶段批次的 benchmark 级共享参数事后分析

本小节只保留旧批次的历史分析，不定义当前搜索协议。同一 benchmark 的共同 tuple 必须在所有纳入模型上完成 full-val。有效性
先要求每个模型 `ΔSR > 0`，再最大化最弱/平均 `ΔSR`，最后比较最弱/平均
`ΔSPL`。增益顺序：R2R/REVERIE 为 DUET、HAMT、GOAT；R2R-CE 为
ETPNav、BEVBert。表中的无效行展示当前共同 finalist 中 SR-first 得分最高
的 tuple，不代表它可用于正式主表。

| 方法 | Benchmark | SR-first 最佳共同 tuple | 各模型 ΔSR（主） | 各模型 ΔSPL（次） | 判定 |
|---|---|---|---:|---:|---|
| Tent | R2R | `lr=1.5625e-5, I=1` | `-0.58/-0.39/-0.30` | `+1.23/-0.11/+0.04` | 无有效共享配置（DUET, HAMT, GOAT 未严格提高 SR） |
| Tent | REVERIE | `lr=1.5625e-5, I=1` | `-1.12/+0.28/+0.43` | `+0.87/+1.02/+1.25` | 无有效共享配置（DUET 未严格提高 SR） |
| Tent | R2R-CE | `lr=1.5625e-5, I=1` | `-1.1568/+0.7712` | `-0.5418/+2.3158` | 无有效共享配置（ETPNav 未严格提高 SR） |
| FSTTA | R2R | `lf=0.0006, ls=0.001, M=3, N=4` | `-7.34/-0.49/-0.10` | `-13.49/-0.28/-0.09` | 无有效共享配置（DUET, HAMT, GOAT 未严格提高 SR） |
| FSTTA | REVERIE | `lf=0.0006, ls=0.001, M=3, N=4` | `-1.69/+0.70/-0.14` | `-0.13/+0.93/-0.07` | 无有效共享配置（DUET, GOAT 未严格提高 SR） |
| FSTTA | R2R-CE | `lf=0.0005, ls=0.001, M=7, N=4` | `-0.1285/-0.3856` | `-0.3435/-0.5575` | 无有效共享配置（ETPNav, BEVBert 未严格提高 SR） |
| EAM | R2R | `lr=1e-6, I=1, c=0.4, mem/b=32/8` | `+0.00/-0.49/+0.00` | `+0.84/-0.28/+0.25` | 无有效共享配置（DUET, HAMT, GOAT 未严格提高 SR） |
| EAM | REVERIE | `lr=1e-6, I=1, c=0.4, mem/b=32/8` | `+0.84/+0.07/-1.47` | `+2.13/+1.03/-0.01` | 无有效共享配置（GOAT 未严格提高 SR） |
| EAM | R2R-CE | `lr=1e-6, I=1, c=0.4, mem/b=16/4` | `+0.3856/-0.3856` | `+0.6995/+0.6008` | 无有效共享配置（BEVBert 未严格提高 SR） |
| FeedTTA† | R2R | `lr=5e-6, γ=0.99, p=0.05, α=-0.2` | `-2.54/-6.27/-2.25` | `-2.38/-6.96/-2.62` | 无有效共享配置（DUET, HAMT, GOAT 未严格提高 SR） |
| FeedTTA† | REVERIE | `lr=5e-6, γ=0.99, p=0.05, α=-0.2` | `+1.05/-1.13/+0.29` | `-3.10/-3.37/-2.72` | 无有效共享配置（HAMT 未严格提高 SR） |
| FeedTTA† | R2R-CE | `lr=5e-6, γ=0.99, p=0.05, α=-0.2` | `-1.6709/-6.2982` | `-8.0751/-29.7992` | 无有效共享配置（ETPNav, BEVBert 未严格提高 SR） |
| ATENA† | R2R | `lq=8e-7, ls=1e-7, λ=0.75, δ=0.1, w=0.1` | `+0.30/+0.20/-0.39` | `+1.88/+0.50/-0.16` | 无有效共享配置（GOAT 未严格提高 SR） |
| ATENA† | REVERIE | `lq=5e-6, ls=1e-7, λ=0.5, δ=0.1, w=0.25` | `+5.62/+7.03/+3.80` | `+5.58/+4.58/+4.78` | **SR 有效；待 parity/多顺序审计** |
| ATENA† | R2R-CE | `lq=5e-7, ls=1e-8, λ=0.5, δ=0.2, w=0.1` | `-0.5141/+0.7712` | `+0.8134/+2.2479` | 无有效共享配置（ETPNav 未严格提高 SR） |

SR-first 且统一标准 Source 后，当前 15 个“方法 × benchmark”组合中只有
ATENA†-REVERIE 存在全模型 SR 严格为正的共同 full-val tuple。FeedTTA†
原先相对 sampled control 的 REVERIE 共享结论不再成立。**当前没有任何
无监督方法形成可直接冻结的 SR 有效共享配置。** 旧口径下 Tent-R2R/
Tent-REVERIE、EAM-REVERIE/
EAM-R2R-CE、ATENA-R2R-CE 等基于 SPL/RGSPL 的“有效”结论由本节取代。
例如 ATENA-R2R-CE 的共同 tuple 虽有 `ΔSPL=+0.8134/+2.2479`，ETPNav
的 `ΔSR=-0.5141`，因此不能冻结。

### 7.4 反馈预算、稳定性与解释边界

- FeedTTA† 的 40 个 full-val finalist 共消费 44,440 个二值反馈；8 个
  SR-first winner 共 8,888 个反馈（6,692 成功、2,196 失败）。整个搜索
  共消费 120,232 个反馈。FeedTTA† 运行本身使用 sampled policy，但正式
  结果统一相对标准 argmax Source；sampled no-update control 只进入诊断附表。
- ATENA† 的 40 个 finalist 在 44,440 episodes 中查询 30,335 次真实反馈
  （68.26%）。8 个 SR-first winner 查询 4,635/8,888（52.15%；3,000 成功、
  1,635 失败），另使用 4,253 个 self labels。全搜索查询 63,398/91,560
  （69.24%）。
- 40 个 SR-first winner 中，23 个无后段 flag，16 个为 warning，
  1 个为 severe collapse。唯一 severe 项仍是 Tent-ETPNav；因此
  SR-first 排名不能替代 late-collapse 与多顺序检查。
- 由于分阶段晋级最初不是按 SR 完成，当前“无共享配置”只对已完成的
  full-val finalist 交集成立；它不限制新的逐模型完整 Cartesian 搜索。
- 上述数值仍是调参证据。零更新适配器一致性审计通过前，不进入正式主表；
  无监督方法与 FeedTTA†/ATENA† 继续分表。

逐 setting 的全部 200 行、绝对 SR/SPL、REVERIE RGS/RGSPL、两个 rank、
完整参数 JSON、run tag 和 audit 标签见
[`VAL_SEEN_TOP5_BY_MODEL_BENCHMARK.md`](results/analysis/hparam_search/VAL_SEEN_TOP5_BY_MODEL_BENCHMARK.md)
及配套 CSV。

### 7.5 当前 R2R 完整 Cartesian 搜索（运行中）

活动协议由 `experiments/r2r_modelwise_cartesian_hparam_v2.json` 定义，batch ID
固定为 `vln-r2r-modelwise-cartesian-v2-seed0`。作业顺序为标准 Source、
FeedTTA sampled control、Tent、FSTTA、EAM、FeedTTA、ATENA；方法内部在
DUET、HAMT、GOAT 之间 round-robin。日志使用 benchmark-first 目录：

```text
results/logs/r2r/hparam_search/<batch>/<method>/
results/tuning/r2r/hparam_search/<batch>/<model>/<method>/jobs/<run-tag>/val_seen/
```

全部 2,379 个 jobs 都绑定同一 Git commit 和 spec SHA256。搜索结束后分别为
15 个 `模型 × 方法` 组合生成 SR-first `WINNER.json`/Top-5，再汇总为 batch
级 `WINNERS.json` 与 `FROZEN_HPARAMETERS.json`。本小节在完整结果同步回来后
补入新的绝对 SR/SPL、相对标准 Source 的增益、稳定性与反馈预算分析。

## 8. 后续更新流程

1. 完成并同步 §7.5 的 R2R 全 Cartesian 搜索；按模型/方法核对 1,021 episodes 完整性，并生成 SR-first Top-5 与 winner。
2. 对逐模型 winner 运行 §6.3 的零更新适配器一致性审计；未通过的 method/setting 不解释性能增益。
3. 参考 R2R 的响应面为 REVERIE 和 R2R-CE 缩小各自的 Cartesian 空间，但二者仍需独立 `val_seen` 搜索和独立 winner。
4. 无监督方法与二值反馈方法继续分表；ATENA/FeedTTA 附加报告真实反馈预算和 performance/query-rate Pareto。
5. 使用各模型/benchmark 冻结参数运行固定顺序 seed 0 的 `val_unseen` 和 test 轨迹；随后才运行 order seeds 1/2 的 robustness 附表。
6. StreamVLN Source 与默认 TTA 点移至更适合长时评估的服务器；只在完整 val split 通过后更新第 3.3 节。
7. 新的正式结果必须链接 per-run manifest；test 提交后记录 submission ID、榜单版本、提交时间和返回指标，不用论文 test 值替代本次提交值。

## 9. 相关文件

- 本地 Source 汇总：[`results/source_baselines_20260810.json`](results/source_baselines_20260810.json)
- 上游 Source 论文值：[`results/legacy/upstream_published_metrics.json`](results/legacy/upstream_published_metrics.json)
- FSTTA/ATENA 论文值：[`results/legacy/published_tta_metrics.json`](results/legacy/published_tta_metrics.json)
- 固定 episode 顺序说明：[`manifests/episode_order/README.md`](manifests/episode_order/README.md)
- Source 调度与复现说明：[`README.md`](README.md)
- 当前 R2R Cartesian spec：[`experiments/r2r_modelwise_cartesian_hparam_v2.json`](experiments/r2r_modelwise_cartesian_hparam_v2.json)
- 当前 R2R Cartesian runner：[`scripts/run_r2r_cartesian_hparam_search.py`](scripts/run_r2r_cartesian_hparam_search.py)
- Tent 调参日志：`results/logs/hparam_search/tent/vln-tta-hparam-final-20260810T182823Z/`
- FSTTA 调参日志：`results/logs/hparam_search/fstta/vln-tta-hparam-final-20260810T182823Z/`
- EAM 调参日志：`results/logs/hparam_search/eam/vln-val-seen-hparam-v1-seed0/`
- FeedTTA 调参日志：`results/logs/hparam_search/feedtta/vln-val-seen-hparam-compact-v1-seed0/`
- ATENA 调参日志：`results/logs/hparam_search/atena/vln-val-seen-hparam-compact-v1-seed0/`
- ATENA tuning evidence：`results/tuning/atena/vln-val-seen-hparam-compact-v1-seed0/`
- SR-first 逐 model-benchmark/method 完整 Top-5：[`results/analysis/hparam_search/VAL_SEEN_TOP5_BY_MODEL_BENCHMARK.md`](results/analysis/hparam_search/VAL_SEEN_TOP5_BY_MODEL_BENCHMARK.md)
- Late-collapse 审计：`results/audits/hparam_search/<method>/<batch>/late-collapse/`

原始 Source bundle、调参日志和 tuning outputs 均保留在本机 `vln/results/`，不进入 Git；Git 只跟踪本报告和紧凑的 provenance/协议文件。
