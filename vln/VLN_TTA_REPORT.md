# VLN Test-Time Adaptation 实验报告

更新日期：2026-08-10

当前状态：**DUET、HAMT、GOAT、ETPNav、BEVBert 的首轮 Source validation 已完成并下载到本机；对应 test 提交轨迹已生成。首轮 HAMT-R2R 使用了 fixed-feature checkpoint，现仅作为 provenance 保留，论文最终 `vitbase-finetune-e2e` checkpoint 的替换评估已准备。StreamVLN Source 仍未形成可登记的完整结果，因此本报告暂不填写其中间指标。尚无本工作区产生的 TTA 正式结果。**

本文持续维护 R2R、REVERIE 和 R2R-CE 三个 benchmark 的 Source/TTA 对比、实验口径、论文参考值与后续小规模超参数方案。机器可读的本地结果见 [`results/source_baselines_20260810.json`](results/source_baselines_20260810.json)，论文 TTA 参考值见 [`results/legacy/published_tta_metrics.json`](results/legacy/published_tta_metrics.json)。

## 1. 证据范围与结果等级

- 五个已完成模型的原始日志位于 `results/source/grouped-source-20260810T080743Z/`：124 个文件、约 111 MiB、24 份 per-setting run manifest。该目录被 Git 忽略，不会提交 raw logs、预测文件或其他大产物。
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
| HAMT e2e（论文最终权重） | — | — | — | — | 等待替换评估 |
| GOAT | **84.82** | **80.05** | **78.12** | **67.58** | 4,173 条轨迹，格式验证通过 |

HAMT-R2R 的活动配置已切换到论文最终 `vitbase-finetune-e2e/best_val_unseen` checkpoint 和配套 `r2r.e2e.ft.22k` 特征。上表 fixed-feature 行不再参与 Source/TTA 主比较；待 e2e 完整结果产生后填入新行。

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
2. **HAMT-R2R 正在消除 checkpoint 变体偏差。** 旧 fixed-feature 模型在 val-unseen 为 `55.34/52.71`，明显低于论文 end-to-end 模型的 `66/61`；它已退出主表。后续 HAMT TTA 只相对新的 e2e matched Source 计算增益。
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

### 6.2 本项目的小搜索规则

1. 每个“模型 × 方法 × benchmark”先运行 2-episode lifecycle smoke，再运行固定 `val_seen` prefix 的数值稳定性检查。
2. 论文默认配置始终保留为主候选；最多增加两个学习率邻点（`0.5×`、`2×`）。只有论文明确存在第二个关键参数时，再增加不超过三个值。
3. 256-episode canonical prefix 只用于分阶段筛选，并始终与相同前缀的 matched Source 比较。最后五个候选和独立的 `final_controls` Source 均在完整 `val_seen` canonical stream 上运行；winner 必须按这一全量 matched Source 约束选择并立即冻结。
4. `FROZEN_HPARAMETERS.json` 在固定顺序的完整 `val_seen` 决赛结束后生成，不依赖多顺序实验。主表使用 seed 0 固定顺序；计算允许时，后续 order seeds `0/1/2` 只能消费已经冻结的配置并形成 robustness 附表，不能重新定义 winner。FSTTA 的 5-shuffle、ATENA 的 3-seed 论文协议单独标注，不与 canonical 主表合并。
5. StreamVLN 只运行论文默认点和至多一个保守学习率邻点；prefix 建议不超过 64 episodes。除非默认点发生发散，不进行二维以上搜索。
6. 每个 split 从 Source checkpoint 重新开始，保存参数更新范围、可训练参数数、优化器状态策略、每 episode 更新次数、查询反馈比例、峰值显存和墙钟时间。
7. 无监督表与二值反馈表分开排名；若需要一张总表，ATENA/FeedTTA 必须带 `†` 并在表头说明监督预算。

## 7. 后续更新流程

1. StreamVLN Source 完成后，下载其完整日志和 run manifests；只在完整 val split 通过后更新第 3.3 节。
2. 为每个 TTA 方法建立 matched Source：相同 commit、checkpoint、dataset version、episode order、action policy 和 seed。
3. 先集成论文原生覆盖最好的组合：DUET+Tent/FSTTA/ATENA、ETPNav/BEVBert+FSTTA/ATENA；再扩展到 HAMT、GOAT、StreamVLN。
4. 新结果必须写入机器汇总，并链接 per-run manifest；console 中的中间均值不进入主表。
5. test 文件提交榜单后，记录 submission ID、榜单版本、提交时间和返回指标；不以论文 test 值替代本次提交值。
6. 每次更新报告时保留 Source、论文参考和 workspace reproduction 三种 provenance 标签，避免后续表格失去可追踪性。

## 8. 相关文件

- 本地 Source 汇总：[`results/source_baselines_20260810.json`](results/source_baselines_20260810.json)
- 上游 Source 论文值：[`results/legacy/upstream_published_metrics.json`](results/legacy/upstream_published_metrics.json)
- FSTTA/ATENA 论文值：[`results/legacy/published_tta_metrics.json`](results/legacy/published_tta_metrics.json)
- 固定 episode 顺序说明：[`manifests/episode_order/README.md`](manifests/episode_order/README.md)
- Source 调度与复现说明：[`README.md`](README.md)

原始 bundle 保留在本机 `vln/results/source/grouped-source-20260810T080743Z/`，不进入 Git。
