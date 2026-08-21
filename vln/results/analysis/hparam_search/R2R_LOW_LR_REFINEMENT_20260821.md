# R2R 四方法低学习率补搜最终分析

更新时间：2026-08-21（Asia/Shanghai）

## 1. 执行与证据完整性

本报告分析 batch
`vln-r2r-low-lr-refine-prod-nosource-20260820` 在 R2R `val_seen`、固定
canonical order、seed 0 上的全部结果。该 batch 于 2026-08-21 09:12 CST
自然结束。

- 运行 commit：`a7fa28efc34970527461f03f60b959d1bff0be10`
- 搜索 spec SHA256：
  `9d2eb6b995c7369cb08053188c2b486f57691fb8225bb4b29a69f7763562b3f1`
- 13 个 phase，469/469 jobs validated；全部 `complete=true`、
  `terminal=true`，0 errors
- 469/469 formal manifests 均为 `status=completed`、`exit_code=0`，并绑定
  上述 commit
- 三个标准 argmax Source 直接复用父 batch 的 formal evidence；本批没有再次
  运行 Source
- 三份 Source manifest 的 SHA256 与
  [`r2r_reused_source_controls.json`](../../../manifests/r2r_reused_source_controls.json)
  完全一致
- 本地 batch log root 共 3,405 个 regular files（含 469 份 formal
  manifest），40 MB；tuning root 共 1,876 个 regular files，28 MB
- raw logs、tuning 产物和 formal run 目录均受 Git ignore 保护；本报告是可跟踪
  的紧凑分析产物

关键 aggregate 的 SHA256：

| 文件 | SHA256 |
|---|---|
| `PLAN.json` | `742cfcf23d3cdccd54a72f5a753ae10b092b693a43d05b696e8e0d534ed07201` |
| `WINNERS.json` | `8acde19370ec1dfdf6d22b23ee7786938f40c02e142c48d39050c4061484bdb0` |
| `FROZEN_HPARAMETERS.json` | `14402a2e99db8d1ca64767d47275b3221d439783968d6b3c6f43c1ab38d74159` |
| `TOP5.csv` | `8c98bde274e26f60fc0783a73df059c3a7d1fc33570ce668083eb4e4077b647c` |

完整搜索设计与父批结果见
[`R2R_CARTESIAN_V2_AND_LOCAL_REFINEMENT.md`](R2R_CARTESIAN_V2_AND_LOCAL_REFINEMENT.md)。
EAM 已在父批三个模型上同时提高 SR/SPL，本轮按计划没有补搜。

## 2. 判定规则

R2R `val_seen` 有 1,021 个 episodes。两位小数 SR 先重建为精确成功数：

```text
successes = round(SR * 1021 / 100)
```

每个模型/方法按以下顺序冻结 winner：成功数、SPL、较低 parameter drift、较少
updates。标准 Source 为：

| 模型 | 成功数 | SR | SPL |
|---|---:|---:|---:|
| DUET | 805 | 78.84 | 72.88 |
| HAMT | 772 | 75.61 | 72.18 |
| GOAT | 866 | 84.82 | 80.05 |

本报告区分两种“优于 Source”：

- **SR-first better**：成功数至少多 1；或成功数相同且 SPL 至少高 0.10 pp。
- **Pareto better**：除满足上述有效增益外，成功数和 SPL 都不能低于 Source。

未加限定地写“优于 Source”时，采用更严格的 Pareto 定义。FeedTTA 还必须与
同 action seed 的 sampled no-update control 比较；该 control 不能替代标准
argmax Source。

## 3. 冻结 winner 与 Source

`Δ成功` 和 `ΔSPL` 均相对同模型标准 Source。`SR-first/Pareto` 分别对应上节的
两种判定。

| 模型 | 方法 | 成功数 | SR/SPL | Δ成功 | ΔSPL | SR-first | Pareto |
|---|---|---:|---:|---:|---:|---|---|
| DUET | Tent | 805 | `78.84/72.88` | 0 | `+0.00` | 否 | 否 |
| DUET | FSTTA | 809 | `79.24/72.33` | +4 | `-0.55` | 是 | 否 |
| DUET | FeedTTA | 804 | `78.75/69.34` | -1 | `-3.54` | 否 | 否 |
| HAMT | Tent | 779 | `76.30/72.88` | +7 | `+0.70` | 是 | 是 |
| HAMT | FSTTA | 779 | `76.30/72.79` | +7 | `+0.61` | 是 | 是 |
| HAMT | FeedTTA | 723 | `70.81/66.45` | -49 | `-5.73` | 否 | 否 |
| GOAT | Tent | 867 | `84.92/80.32` | +1 | `+0.27` | 是 | 是 |
| GOAT | FSTTA | 867 | `84.92/80.21` | +1 | `+0.16` | 是 | 是 |
| GOAT | FeedTTA | 866 | `84.82/79.07` | 0 | `-0.98` | 否 | 否 |
| GOAT | ATENA | 867 | `84.92/80.24` | +1 | `+0.19` | 是 | 是 |

结果是：10 个激活的模型-方法 cell 中，6/10 为 SR-first better，5/10 为
Pareto better。因此“每个方法都应优于 Source”在当前 seed 0 `val_seen`
证据上被否定。所有三个 FeedTTA cell 都没有超过标准 Source；DUET Tent 仅与
Source 完全持平；DUET FSTTA 以 SPL 下降换取更高 SR。

这些都是调参 seed 上的点估计，不是统计显著性结论。尤其 GOAT 的 +1 success
只有 `0.097943 pp`，必须经过预注册的 order seeds 1、2 和冻结后的
`val_unseen` 确认。

## 4. 本轮是否真的找到新优胜点

父配置在本批的复跑与历史 parent 的 SR、SPL、drift、updates 完全一致，因此
不存在 batch drift。下表将本轮 winner 与同批 parent/overlap anchor 比较。

| 模型 | 方法 | 同批 anchor（成功数/SPL） | 本轮 winner（成功数/SPL） | 是否刷新 anchor |
|---|---|---:|---:|---|
| DUET | Tent | `803/72.81` | `805/72.88` | 形式上通过：`+2/+0.07` |
| DUET | FSTTA | `809/72.33` | 相同 | 否，winner 即 parent anchor |
| DUET | FeedTTA | `804/69.34` | 相同 | 否，winner 即 parent anchor |
| HAMT | Tent | `779/72.88` | 相同 | 否 |
| HAMT | FSTTA | `779/72.79` | 相同 | 否 |
| HAMT | FeedTTA | `723/66.45` | 相同 | 否 |
| GOAT | Tent | `867/80.32` | 相同 | 否 |
| GOAT | FSTTA | `867/80.21` | 相同 | 否 |
| GOAT | FeedTTA | `866/79.07` | 相同 | 否 |
| GOAT | ATENA | `866/80.13` | `867/80.24` | **是：`+1/+0.11`** |

DUET Tent 需要特殊解释：其同批 anchor 是 `lr=3e-7,
update_interval=1` 的低 LR 重叠点，不是父批全空间 winner。本轮 `lr=1e-9`
的 parameter drift 仅 `6.63e-7`，实质上是返回 no-adaptation Source；它虽然
通过相对 overlap anchor 的形式门槛，却没有证明 Tent 适配有效。父批真正的
DUET Tent winner 是 `lr=3e-5, update_interval=16`，结果 `805/73.05`，仍优于
本轮 `805/72.88`。

因此，若只按注册的同批 anchor gate，有 DUET Tent 和 GOAT ATENA 两个 cell
通过；若要求“刷新父批整体最佳且体现有效适配”，本轮只有 GOAT ATENA 真正
改进。

## 5. “Tent 最差、FSTTA 应更好”是否成立

只在同属无监督 TTA 的 Tent 与 FSTTA 间按固定 SR-first 顺序比较：

| 模型 | Tent（成功数/SPL） | FSTTA（成功数/SPL） | SR-first 胜者 |
|---|---:|---:|---|
| DUET | `805/72.88` | `809/72.33` | FSTTA，但存在 SPL trade-off |
| HAMT | `779/72.88` | `779/72.79` | Tent |
| GOAT | `867/80.32` | `867/80.21` | Tent |

所以该组合假设只在 DUET 成立，在 HAMT 和 GOAT 均不成立。HAMT 上 Tent 与
FSTTA 成功数相同，Tent 的 SPL 高 0.09，drift `0.00235` 也远低于 FSTTA 的
`0.03715`；GOAT 同样是 Tent 在相同成功数下 SPL 更高。不能据当前结果写
“Tent 是最差方法”。

FeedTTA 消费二值 episode feedback 且使用 sampled action policy，ATENA 也
消费二值反馈，不能把它们与无监督方法混成一个不加说明的方法排名。

## 6. Tent 参数区域

本轮固定 `update_interval=1`，只搜索 LR。

| 模型 | 冻结 LR | 成功数/SR/SPL | 区域结论 |
|---|---:|---:|---|
| DUET | `1e-9` | `805/78.84/72.88` | `1e-9..1e-7` 全部等同 Source；仅因 drift 最低选中 |
| HAMT | `3e-6` | `779/76.30/72.88` | 清晰单峰；`1e-6..1e-5` 是正收益区 |
| GOAT | `1e-5` | `867/84.92/80.32` | `<=1e-6` 等同 Source，`1e-5` 最佳 |

坏区也很清楚：`1.5625e-5` 在 HAMT 降至 768 successes、在 GOAT 降至
863；DUET 的 `1e-6..1.5625e-5` 虽提高 SPL 至最高 74.11，却将成功数降至
799--801。更低 LR 并非自动更好：过低时只是没有产生可见适配。

固定 interval 为 1 也没有刷新父批全空间 Tent 结果。父批 HAMT winner
`lr=1e-5, interval=2` 为 `782/73.17`，比本轮固定 interval 1 的 winner 多
3 个 successes、SPL 高 0.29；这应标记为 paper-compliance 与实测最优之间的
协议差异，而不是隐藏。

## 7. FSTTA 参数区域

| 模型 | 冻结 `(lr_fast, lr_slow, m, n)` | 成功数/SR/SPL | 本轮低 LR 网格结论 |
|---|---|---:|---|
| DUET | `(1.8e-3,3e-4,3,16)` | `809/79.24/72.33` | parent anchor 仍为 SR-first winner |
| HAMT | `(6e-4,3e-4,1,16)` | `779/76.30/72.79` | parent anchor 仍明显最好 |
| GOAT | `(1.8e-3,1e-3,3,8)` | `867/84.92/80.21` | 70 个更低 LR 点几乎全等同 Source |

有价值的新局部结构是：

- DUET：`n=4, lr_slow=1e-4` 是稳定好区。新低 LR 网格最佳点
  `(3e-4,1e-4,3,4)` 为 `807/79.04/72.90`，较 Source 多 2 successes 且
  SPL 高 0.02。`(1e-4,1e-4,3,4)` 得到相同指标。已有的 joint anchor
  `(1.8e-3,3e-4,8,4)` 为 `808/79.14/73.10`，仍是更好的 SR/SPL 平衡点。
- HAMT：新点中 `(3e-4,3e-5,1,16)` 最好，为 `775/75.91/72.49`；同一
  `lr_fast=3e-4,m=1,n=16` 配五个 slow LR 都为 775 successes，说明该局部
  区域稳健，但仍比 parent anchor 少 4 successes。
- GOAT：69/70 个新低 LR 点为 `866/80.05`，另一个仅 `866/80.06`。对 GOAT
  而言，本轮 LR 范围已经低到几乎不更新；数据不支持继续向更低方向搜索。

坏区主要是 DUET 的 `n=16` 配 `lr_slow=3e-5` 和较大 grid fast LR，最低
802 successes；HAMT 的部分低 fast/slow 组合为 771，已略差于 Source。
总的说，降低 FSTTA LR 在 DUET/HAMT 找到更平衡或更宽的 Source-positive
区域，但没有刷新三个父 winner。

## 8. FeedTTA：适配有效，但没有越过 argmax Source

FeedTTA 的 `action_selection=sample`，所以需要先拆出 action policy 的协议
损失。下表比较冻结 winner 与同批、同 action seed 的 sampled no-update
control。

| 模型 | sampled control（成功数/SR/SPL） | FeedTTA winner | 适配增益（成功数/SR/SPL） | 对 argmax Source |
|---|---:|---:|---:|---:|
| DUET | `790/77.38/67.43` | `804/78.75/69.34` | `+14/+1.37/+1.91` | `-1/-0.09/-3.54` |
| HAMT | `707/69.25/64.68` | `723/70.81/66.45` | `+16/+1.56/+1.77` | `-49/-4.80/-5.73` |
| GOAT | `853/83.55/78.67` | `866/84.82/79.07` | `+13/+1.27/+0.40` | `0/+0.00/-0.98` |

因此不能简单说 FeedTTA “没有作用”：三个模型都在数值上优于自己的 sampled
control；但也不能说它优于 Source，因为 sampled policy 的损失没有被适配完全
抵消，标准 argmax Source 在三个模型上都更好。

冻结区域均在本轮搜索的上侧或父 anchor：

| 模型 | 冻结 `(lr, gamma, p, alpha)` | 结论 |
|---|---|---|
| DUET | `(3e-6,.8,.05,-.05)` | 804 successes；本轮唯一接近 Source 的点 |
| HAMT | `(5e-6,.8,.1,-.2)` | parent anchor；最佳新点 `3e-6` 仍少 4 successes |
| GOAT | `(5e-6,.8,.1,-.1)` | parent anchor；最佳新低 LR 点只有 863 successes |

`1e-8/3e-8` 基本退化为 sampled control；继续单纯减小 LR 不能解决问题。
HAMT/GOAT 最佳点仍在 `5e-6` parent anchor，直接反驳“FeedTTA 在当前 R2R
上主要需要更低 LR”的假设。高侧也存在强交互：HAMT 在
`lr=3e-6, gamma=1.0, alpha=-.2` 时只有 692 successes；GOAT 的 paper
default anchor `(5e-6,.99,.05,-.2)` 只有 `843/77.43`。

## 9. ATENA：本轮唯一真正刷新的方法

GOAT ATENA 新 winner 为：

```text
lr_query=4e-7, lr_self=5e-8, mix_lambda=0.1,
query_threshold=0.15, self_loss_weight=0.1
```

它得到 `867/84.92/80.24`，相对 Source 为 `+1 success/+0.19 SPL`，相对
同批 parent anchor `866/80.13` 为 `+1/+0.11`。winner 使用 263 次真实反馈
查询，query rate 为 25.76%；parent anchor 使用 204 次，19.98%。

好区域集中在 `lr_query=3e-7..4e-7`、固定 8:1 query/self LR 比例和
`threshold=0.15`；`mix_lambda=0/.05/.1` 的导航指标几乎相同。`<=1e-7`
基本等同 Source；`threshold=0.25` 在较高 LR 下开始回落。core default
`(1e-6,1e-7,.5,.1)` 最差，仅 `861/84.33/79.85`。

### 9.1 feedback-efficiency 导出边界错误

`WINNER.json` 导出的 efficiency point 是
`(3e-7,3.75e-8,mix=.05,threshold=.2)`，结果 `867/80.11`、208 queries
（20.37%）。但生成代码使用未经容差保护的浮点比较：

```text
84.82 >= 84.92 - 0.1
```

右侧在二进制浮点中为 `84.82000000000001`，错误排除了刚好处于边界的点。
按精确 episode 数，一次 success 差值为 `0.097943 pp`；按计划“距 winner
不超过 0.1 pp SR、0.2 pp SPL”的原意，有 9 个并列的 158-query 最低查询
合格点。按原排序列出的第一个是：

```text
run: ...refine-12-atena-0002-goat-r2r-49998db9d3
lr_query=1e-8, lr_self=1.25e-9, mix_lambda=0,
query_threshold=0.25
result: 866/84.82/80.05
queries: 158/1021 = 15.48%
```

该点比 winner 少 1 success、SPL 低 0.19，正好在意图容差内。这个问题只影响
二级 feedback-efficiency point，不影响 SR-first winner、TOP5 或冻结参数。
在修复生成器并重建 aggregate 前，不应把导出的 208-query 点称为真正的
“最低查询近优点”。

## 10. 最终结论与后续门槛

1. 本轮 469/469 jobs 全部成功，日志和 formal evidence 已完整落到本地。
2. 低 LR 补搜没有让所有方法超过 Source：只有 5/10 winner 严格 Pareto
   better，FeedTTA 为 0/3。
3. “Tent 最差、FSTTA 在每个模型都应更好”不成立；HAMT、GOAT 均是 Tent
   在相同成功数下 SPL 更高。
4. 对父批整体最佳的真正刷新只有 GOAT ATENA。FSTTA 的低 LR 补搜提供了
   DUET/HAMT 的稳健正收益区域，但三个冻结 winner 仍是父 anchor；FeedTTA
   继续降 LR 没有收益。
5. FeedTTA 相对 matched sampled control 在三个模型上都有效；主要未决问题
   是 sampled action protocol 的损失及 action-seed 稳健性，而不是再向更低
   LR 盲目扩展。
6. 任何 +1 success 结果都不能直接写成稳定提升。下一步只有在不重新选参的
   前提下运行 order seeds 1、2，并最终在 `val_unseen` 验证，才可进入正式
   对比表。
