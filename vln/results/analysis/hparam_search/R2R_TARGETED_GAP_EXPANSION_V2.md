# R2R targeted gap expansion v2 正式分析

更新日期：2026-08-22

状态：**97/97 个完整 `val_seen` 搜索任务及其 formal run evidence 已核验；
本报告只冻结 canonical-order seed 0 的选参结论，不把它写成 publication-final
结果。**

## 1. 范围、协议与证据完整性

本报告分析 batch `vln-r2r-targeted-gap-expand-v2-seed0`。本轮没有重跑
Source，也没有扩展到其他模型/方法，只运行以下三个阶段：

```text
DUET Tent: 24 -> GOAT FeedTTA: 36 -> GOAT ATENA: 37
```

所有任务均使用 canonical-order seed 0、完整的 1,021-episode R2R
`val_seen`，每个 job 从 Source checkpoint 重新开始。Tent 属于无监督 TTA；
`FeedTTA†` 和 `ATENA†` 消费二值 episode feedback，不能与 Tent 合并成同一
监督类别排名。

- 运行 commit：`ef4f3e37fca3b3057cb2a2c365352b70633bd22b`。
- spec SHA256：
  `4087a1e0680b932c95cfbf9b1f5c7830917f87f6088d2cfe4b23caade706d3aa`。
- 复用 Source ledger SHA256：
  `15b9a9f72bb5bcaeb26fabc5abdf94cfffcc03778b4ea8f9ec10940d8044a3b4`。
- 三个 `SUMMARY.json` 均为 `complete=true, terminal=true`，validated 数分别为
  24、36、37；三个 `progress.json` 均为 0 failed、0 pending。
- 97 个 admin job tag 与 97 份 formal manifest 一一对应，没有 orphan、额外
  manifest 或 retry；所有 manifest 均为 `status=completed, exit_code=0`，且记录
  同一个运行 commit。
- 97 份 manifest 共声明 291 个 result artifacts。本地逐项重算后，291/291
  文件均存在，实际 byte size 和 SHA256 均与 manifest 一致，0 mismatch。
- 97 份 `tta_diagnostics.json` 均记录 1,021 episodes。36 个 FeedTTA job
  全部使用
  `r2r_submitted_trajectory_evaluator_success_every_episode`，且各有 1,021 次反馈
  和更新；37 个 ATENA job 全部使用
  `r2r_submitted_trajectory_evaluator_success_lazy_query`，各执行 1,021 次 query
  gate，且 `queries == feedback_observed_episodes`。

Source 沿用 parent campaign 的受跟踪证据，不计入上述 97 个任务。原 canonical
formal path 当前未单独复制到本工作区，但下载的 evidence copy 可用：DUET/GOAT
Source manifest SHA256 分别为
`e7adfa7bcab1c9b0c19b4f37b08c9c38de75cb247716c629504941612a87d9bc` 和
`6dcb148b179ea45bbef99ccc34ab0109be035b78dfe5e4fa5e48329eaed65b96`，均为
`completed, exit_code=0`，checkpoint 与 canonical stream digest 也和 ledger
一致。

实际并发与资源峰值如下；三个 phase 的 resource 时间区间互不重叠，符合严格
barrier 协议。

| phase | validated | 最大并发 | 峰值 GPU 显存 | 峰值 cgroup 内存 |
|---|---:|---:|---:|---:|
| DUET Tent | 24/24 | 10 | 18,567 MiB | 15.279 GiB |
| GOAT FeedTTA | 36/36 | 6 | 11,902 MiB | 11.591 GiB |
| GOAT ATENA | 37/37 | 5 | 26,499 MiB | 13.973 GiB |

## 2. Source、targeted-v1 与 v2 总览

主选择顺序固定为 `SR -> SPL -> lower parameter drift -> fewer updates`。由于
1/1,021 个 success 等于约 `0.09794 pp`，下表同时报告成功 episode 数。
本轮预注册的确认门槛比 v1 更严格：相对 Source 至少增加 2 个 success 且 SPL
不下降，或增加 1 个 success 且 SPL 至少提高 `0.10 pp`。

| 模型–方法 | Source | targeted-v1 | expansion-v2 SR-first | v2 相对 Source | v2 相对 v1 | v2 门槛 |
|---|---:|---:|---:|---:|---:|---|
| DUET–Tent | 805 / `78.84/72.88` | 805 / `78.84/72.93` | **806 / `78.94/73.08`** | `+1 / +0.10/+0.20` | `+1 / +0.10/+0.15` | 通过 |
| GOAT–FeedTTA† | 866 / `84.82/80.05` | 867 / `84.92/80.11` | **867 / `84.92/80.11`** | `+1 / +0.10/+0.06` | `0 / +0.00/+0.00` | 不通过 |
| GOAT–ATENA† | 866 / `84.82/80.05` | 867 / `84.92/80.24` | **867 / `84.92/80.24`** | `+1 / +0.10/+0.19` | `0 / +0.00/+0.00` | 通过 |

因此 v2 真正刷新导航指标的是 DUET–Tent。FeedTTA 找到相同指标下略低 drift
的机制点；ATENA 没有刷新 SR/SPL，但在保持主指标的同时刷新了 drift，并找到了
更省反馈的次级 Pareto 点。所有差异仍只是选参顺序上的点估计，不能据此声称统计
显著。

## 3. DUET–Tent：新 SR-first winner

v2 winner 为：

```text
lr=1e-5, update_interval=1
norm_scope=last_k_ln, last_k_ln=9
optimizer=AdamW, weight_decay=0, episodic=false
```

该点达到 `SR/SPL=78.94/73.08`，即 806 successes；相对 Source 为
`+0.10/+0.20 pp`，相对 v1 winner 为 `+0.10/+0.15 pp`。它执行 6,863 次更新，
relative parameter drift 为 `2.5274e-2`。v1 的 `k=4, lr=3e-5` 点执行
6,869 次更新，drift 为 `6.3936e-2`；v2 在增加 1 个 success 的同时把 drift
降低约 60.5%。

24 个新点中只有该点的 SR 高于 Source。更宽的 `k=29, lr=1e-5` 达到本阶段
最高 SPL `74.06`，但 SR 降到 `78.45`，比 Source 少 4 个 successes；它是明确
的 SR/SPL trade-off，不能取代 SR-first winner。该响应面说明收益来自 scope
深度和 LR 的配对，而不是单纯扩大 LayerNorm 解冻范围。

新 winner 达到本轮 `+1 success` 且 `+0.20 SPL` 的确认门槛，可以冻结后进入
order seeds 1/2；在完成该确认前，不能把这 1 个 success 的变化写成稳健 Tent
增益。

## 4. GOAT–FeedTTA†：指标持平，低 drift 决胜

v2 SR-first winner 为：

```text
lr=1e-6, gamma=.8
p=.1, alpha=-.1, scope_profile=action_head
action_selection=argmax, optimizer=Adam, eps=1e-5
```

它和 v1 的 `lr=1e-6, gamma=.8, p=.05, alpha=+.1` 点得到逐项相同的导航
聚合指标：867 successes、`SR/SPL=84.92/80.11`。两者都执行 1,021 次更新，
并消费 1,021 次 evaluator-endpoint 二值反馈，其中 867 次成功、154 次失败。

v2 点的 drift 为 `9.0274e-4`，v1 为 `9.1195e-4`，降低约 1.0%，因此按预注册
tie-breaker 成为 v2 winner。36 个 v2 点中有 8 个得到相同的
`84.92/80.11`，包括 no-SGR、正 scaling 和负 SGR 配置；这说明 seed 0 上存在
一条离散的等指标窄脊，但不足以证明负 SGR 本身带来性能收益。

相对 Source，本点仍只有 `+1 success/+0.06 SPL`，没有达到 v2 要求的
`+1 success/+0.10 SPL` 门槛。因此它应登记为“与 v1 指标持平的最低 drift
代表点”，而不是 v2 的新性能晋级；尤其不能因 tie-breaker 更优而宣称 FeedTTA
增益被加强。

## 5. GOAT–ATENA†：三个不同角色

ATENA 的主选择、反馈效率和高 SPL 支线回答不同问题，不能用单一“最佳点”合并。

| 角色 | 核心参数 `(lq, ls, lambda, delta)` | successes / SR / SPL | 相对 Source | query budget | drift |
|---|---|---:|---:|---:|---:|
| **SR-first winner** | `(3.75e-7, 3.515625e-8, 0, .15)` | **867 / 84.92 / 80.24** | `+1 / +0.10/+0.19` | 265/1,021 (25.95%) | `1.1521e-4` |
| **feedback-efficiency** | `(3e-7, 3.75e-8, 0, .20)` | **867 / 84.92 / 80.11** | `+1 / +0.10/+0.06` | **208/1,021 (20.37%)** | `8.6596e-5` |
| **high-SPL trade-off** | `(4.8e-6, 6e-7, .625, 0)` | 862 / 84.43 / **80.69** | `-4 / -0.39/+0.64` | 1,021/1,021 (100%) | `2.0020e-3` |

### 5.1 SR-first winner

主 winner 与 v1 winner 都是 867 successes、`84.92/80.24`。v1 参数为
`3.5e-7/4.375e-8, lambda=0, threshold=.15`，查询 264 次，drift
`1.2209e-4`；v2 通过 self-LR 解耦将 drift 降低约 5.6%，代价只是多查询 1 个
episode。v2 的 265 次 query 中 181 次反馈为成功，其余 756 episodes 使用
self label。该点满足 `+1 success/+0.19 SPL` 门槛，是 order robustness 的主
冻结候选。

### 5.2 Feedback-efficiency 点

机器规则是在距离主 winner 不超过 `0.10 pp SR` 和 `0.20 pp SPL` 的候选中
最小化 query rate。选出的点保持相同 867 successes，只把 SPL 从 `80.24`
降到 `80.11`，同时将查询从 265 降到 208，少 57 次、相对减少约 21.5%。其中
133 个 queried episodes 成功，813 episodes 使用 self label。

与 v1 的效率点 `84.92/80.20, queries=239` 相比，v2 再少查询 31 次，但多损失
`0.09 pp SPL`。因此它是反馈预算 Pareto 点，不是主导航性能 winner。

### 5.3 High-SPL 全查询支线

高 SPL 点把 SPL 提高到 `80.69`，但相对 Source 少 4 个 successes，相对主
winner 少 5 个；它还查询全部 1,021 episodes，drift 约为主 winner 的 17.4 倍。
这一结果在 corrected lazy evaluator endpoint 下复现了高-SPL方向，但本配置因
`threshold=0` 实际退化为全反馈上界。它只能作为“牺牲 SR 换路径效率”的独立
trade-off 报告，不能覆盖 SR-first winner，也不能作为低反馈 ATENA 的证据。

其余高-LR全查询点大多同时降低 SR/SPL，说明 `4.8e-6/.625` 是局部特例，不支持
继续把该支线解释为单调的高 LR 收益。

## 6. 最终判定与 publication 边界

1. **DUET–Tent 刷新 seed-0 winner 并通过确认门槛。** 冻结
   `last_k_ln=9, lr=1e-5, interval=1`，进入 matched Source 的 order seeds
   1/2。
2. **GOAT–FeedTTA† 没有刷新性能。** v2 只在相同指标下找到约 1% 更低 drift
   的负 SGR 点；它未通过加严后的晋级门槛。
3. **GOAT–ATENA† 保留三角色。** 主冻结点是 SR-first winner；效率点单独报告
   query/SPL trade-off；全查询高-SPL点只作上界和机制诊断。
4. 本批是用于选参的单一 canonical order seed 0。`+1 success` 不构成跨顺序
   稳健性或统计显著性，**不得把本报告数值直接作为 publication-final 主表结果**。
5. 只有在冻结参数后完成 order seeds 1/2、matched Source、零更新 parity 审计，
   才能报告顺序稳健性；随后可在 `val_unseen` 做一次独立确认，但不得用
   `val_unseen` 重新选参。
6. FeedTTA/ATENA 始终标记为 binary-feedback-supervised TTA，并同时报告实际
   feedback/query budget；不能与无监督 Tent 合并排名。

## 7. 证据位置

- 机器可执行 spec：
  [`r2r_targeted_gap_expansion_v2.json`](../../../experiments/r2r_targeted_gap_expansion_v2.json)
- 运行前设计：
  [`R2R_TARGETED_GAP_EXPANSION_V2_PLAN.md`](../../../experiments/R2R_TARGETED_GAP_EXPANSION_V2_PLAN.md)
- parent v1 分析：
  [`R2R_TARGETED_GAP_REFINEMENT_V1.md`](R2R_TARGETED_GAP_REFINEMENT_V1.md)
- 本地调度、逐 job metrics 与资源证据：
  `vln/results/logs/r2r/hparam_search/vln-r2r-targeted-gap-expand-v2-seed0/`
- 本地模型输出与 TTA diagnostics：
  `vln/results/tuning/r2r/hparam_search/vln-r2r-targeted-gap-expand-v2-seed0/`
- 97 份 formal manifests：
  `vln/results/runs/vln-r2r-targeted-gap-expand-v2-seed0-*/manifest.json`
