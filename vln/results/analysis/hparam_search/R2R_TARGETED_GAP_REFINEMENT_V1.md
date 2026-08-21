# R2R targeted gap refinement v1

更新日期：2026-08-21

## 1. 范围与完整性

本报告分析 batch `vln-r2r-targeted-gap-refine-v1-seed0`。该轮只运行三个
弱组合，没有重跑 Source 或 sampled control：

```text
DUET Tent: 9 -> GOAT FeedTTA: 11 -> GOAT ATENA: 9
```

- 29/29 个完整 `val_seen` 任务成功，0 failed、0 invalid、0 orphaned；每个
  任务均为固定顺序的 1,021 episodes。
- 运行 commit 为 `9c006fbd521f1bfab8f8f7b654109d4971e59f38`，spec SHA256 为
  `fec86947a209b524ed19517b108693c74504bc8685afee7305f6c80ea0a29fd9`。
- 三个阶段严格串行；并发上限分别为 10/6/5，实际同时运行最多 9/6/5。
- 29 份 formal manifest 已下载；其声明的 87 个 result artifacts 均通过本地
  文件大小和 SHA256 复核。
- Source 复用受跟踪 manifest：DUET 为 `78.84/72.88`，GOAT 为
  `84.82/80.05`。下文数值均为 SR/SPL，差值单位为百分点。

## 2. 为什么先修 ATENA

旧 `tta_r2r_episode_stats()` 只为 FeedTTA 返回正式 evaluator success；ATENA
则回退到 simulator stop 时的 `distance < 3`。GOAT/DUET 可能在 simulator
停止后重排提交终点，因此旧 ATENA 查询到的反馈不一定对应正式 SR。历史 25 个
全查询点中有 19 个发生不一致，`evaluator successes - feedback successes` 的
范围为 -1 到 +6。

本轮将 ATENA 改为惰性 evaluator callback：

- callback 使用提交轨迹的正式 evaluator endpoint；
- entropy gate 只有在主动 query 时才调用 callback；
- 未查询 episode 仍完全使用 self label，不额外读取 oracle；
- runner 要求 `queries == feedback_observed_episodes`、所有 1,021 episodes 均执行
  query gate，并校验 diagnostics 中的 endpoint 协议。

修正前后数值 anchor 都是 `4e-7/5e-8, lambda=0, threshold=.15`。其正式指标
均为 `84.92/80.24`，但旧 anchor 的 263 次 query 中记录 178 次成功，修正后为
181 次。这说明本顺序下最终轨迹恰好未变，但适配所消费的监督标签确实被纠正。

## 3. 最优结果

| 模型–方法 | Source | 本轮最优 | 相对 Source | 成功数 | 结论 |
|---|---:|---:|---:|---:|---|
| DUET–Tent | 78.84/72.88 | **78.84/72.93** | +0.00/+0.05 | 805→805 | 仅微小 SPL 增益，不晋级 |
| GOAT–FeedTTA† | 84.82/80.05 | **84.92/80.11** | +0.10/+0.06 | 866→867 | 首次同时超过 Source，边缘晋级 |
| GOAT–ATENA† | 84.82/80.05 | **84.92/80.24** | +0.10/+0.19 | 866→867 | 数值未超过旧最优，但协议已修正 |

`†` 表示使用二值 episode feedback。由于 `+0.10 SR` 只对应 1/1,021 个新增
成功 episode，FeedTTA 和 ATENA 的结果都必须经过 order seeds 1/2 才能视为
稳健增益。

### 3.1 DUET–Tent

最优参数：

```text
lr=3e-5, update_interval=1
norm_scope=last_k_ln, last_k_ln=4
optimizer=AdamW, weight_decay=0, episodic=false
```

该点执行 6,869 次更新，relative parameter drift 为 `6.39e-2`。它保持 805 个
成功 episode，但 SPL 只提高 0.05，低于预设的“持平 SR 时 SPL 至少 +0.10”门槛。
九个点中只有这个点不降低 SR；因此局部 LayerNorm scope 缓和了 all-LN 的
trade-off，却没有得到可推广的 Tent 增益。

| scope | LR | SR/SPL |
|---|---:|---:|
| `last_ln` | 3e-6 / 1e-5 / 3e-5 | 78.75/72.78 · 78.65/72.84 · 78.75/72.77 |
| `last_k_ln, k=3` | 3e-6 / 1e-5 / 3e-5 | 78.75/72.78 · 78.65/72.85 · 78.75/72.86 |
| `last_k_ln, k=4` | 3e-6 / 1e-5 / 3e-5 | 78.65/72.76 · 78.45/72.63 · **78.84/72.93** |

结论是继续对 DUET–Tent 做同类微调价值较低；不能为了数值更高而恢复不符合
当前协议的 `update_interval=16`。

### 3.2 GOAT–FeedTTA

SR/SPL-first winner：

```text
lr=1e-6, gamma=.8
p=.05, alpha=.1, scope_profile=action_head
action_selection=argmax, optimizer=Adam, eps=1e-5
```

该点消费 1,021 次反馈（867 成功、154 失败），执行 1,021 次更新，relative
parameter drift 为 `9.12e-4`。此前相同 LR、`gamma=.9` 只有 `84.82/80.06`；
本轮补上的 `gamma=.8` 配对点达到 `84.92/80.11`。另有两点得到完全相同指标：
`1.25e-6/.8` 和 `1.5e-6/.9`，winner 因漂移更小而选择 `1e-6/.8`。

响应面很窄：

- `5e-7` 对两个 gamma 都退化为 Source 的 `84.82/80.05`；
- `7.5e-7` 为 `84.82/80.06`；
- `1e-6/.8`、`1.25e-6/.8`、`1.5e-6/.9` 均为 `84.92/80.11`；
- 其余 `1.25e-6–2e-6` 点为 `84.82/80.05–80.06`。

因此这里不是“越小 LR 越好”，而是 `1e-6–1.5e-6` 内 gamma 与 LR 的窄耦合带。
该轮已经实现相对 Source 的双指标正增益，不建议继续在 seed 0 上加密；应直接
冻结 `1e-6/.8` 做 order robustness。

### 3.3 GOAT–ATENA

主 winner：

```text
lr_query=3.5e-7, lr_self=4.375e-8
mix_lambda=0, query_threshold=.15
self_loss_weight=.1, optimizer=AdamW, weight_decay=.01
```

该点为 `84.92/80.24`，查询 264/1,021 episodes（25.86%），其中 182 个反馈为
成功；其余 757 episodes 使用 self labels。它与修正后的旧 anchor
`4e-7/5e-8, threshold=.15` 数值完全相同，但漂移更小（`1.22e-4` 对
`1.37e-4`），因此成为新 winner。

反馈效率点是：

```text
lr_query=3.5e-7, lr_self=4.375e-8, threshold=.17
84.92/80.20, queries=239/1021 (23.41%)
```

它少查询 25 个 episode，保持相同 SR，只损失 0.04 SPL。如果强调反馈预算，可将
其作为次级 Pareto 点。`threshold=.13` 查询更多但 SR 退回 Source；
`lr_query=4.5e-7` 全部不如主区域。self LR 减半仍为 `84.92/80.23`，增至
`7.5e-8` 则降到 `84.62/79.96`，说明 self branch 不宜更激进。

## 4. 结论与后续

1. **DUET–Tent 不晋级。** 新 scope 只得到 `+0.05 SPL`，不足以证明有效。
2. **GOAT–FeedTTA 有效但增益很小。** 冻结 `lr=1e-6, gamma=.8`，运行 order
   seeds 1/2；若新增成功 episode 不稳定，不应宣称优于 Source。
3. **GOAT–ATENA 实现已纠正。** 主指标仍为 `84.92/80.24`；冻结
   `3.5e-7/4.375e-8, lambda=0, threshold=.15` 做 seeds 1/2，同时保留
   threshold `.17` 的反馈效率点。
4. robustness 完成前不要在 `val_unseen` 上继续选择参数；冻结后再进行一次独立
   `val_unseen` 评估。

## 5. 证据位置

- 机器可执行计划：
  [`r2r_targeted_gap_refinement_v1.json`](../../../experiments/r2r_targeted_gap_refinement_v1.json)
- 设计说明：
  [`R2R_TARGETED_GAP_REFINEMENT_PLAN.md`](../../../experiments/R2R_TARGETED_GAP_REFINEMENT_PLAN.md)
- 本地调度证据：
  `vln/results/logs/r2r/hparam_search/vln-r2r-targeted-gap-refine-v1-seed0/`
- 本地 compact outputs：
  `vln/results/tuning/r2r/hparam_search/vln-r2r-targeted-gap-refine-v1-seed0/`
- formal manifests：
  `vln/results/runs/vln-r2r-targeted-gap-refine-v1-seed0-*/manifest.json`
