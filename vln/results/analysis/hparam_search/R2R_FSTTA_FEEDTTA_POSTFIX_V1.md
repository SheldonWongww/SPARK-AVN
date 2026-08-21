# R2R FSTTA / FeedTTA 论文对齐修正补搜结果

更新时间：2026-08-21（Asia/Shanghai）

## 1. 范围与证据完整性

本报告分析 batch `vln-r2r-fstta-feedtta-postfix-v1-seed0`。实验只运行
FSTTA 和 FeedTTA，没有重跑 Source；DUET、HAMT、GOAT 各自按
`FSTTA -> FeedTTA` 严格串行，在 canonical-order、seed-0 的 R2R
`val_seen` 上完整评估 1,021 episodes。

- 运行 commit：`b2e37dd387fd8df4543dcfcbac7c79592f88d739`
- 搜索 spec SHA256：
  `f979f7809265b9144a9ab75da1ede8871b5f93eefb1e87014a42a90011158490`
- 展开计划 SHA256：
  `63cb7073ad64f019ae38bc7199b5750a266c781ecbb90f0aa292d305fb293263`
- FSTTA：DUET/HAMT/GOAT 均为 12/12 validated；FeedTTA 均为
  14/14 validated；六个 phase 都是 `complete=true`、`errors=[]`
- 78 个根任务退出码均为 0；78 份 compact tuning result 与 78 份
  completed formal run manifest 已下载
- manifest 声明的 234 个紧凑 result artifacts 均通过本地 size/SHA256
  复核；method 数量为 FSTTA 36、FeedTTA 42
- 正式任务从 18:26:14 运行到 19:47:48，墙钟约 1 小时 21 分 34 秒

实际并发与资源峰值如下。GOAT-FSTTA 的生产上限为 14，但该 phase 只有
12 个候选，因此实际同时运行 12 个。

| Phase | 最大并发 | GPU 显存峰值 | GPU 利用率峰值 |
|---|---:|---:|---:|
| DUET-FSTTA | 12 | 22,217 MiB | 99% |
| DUET-FeedTTA | 6 | 16,042 MiB | 100% |
| HAMT-FSTTA | 11 | 25,454 MiB | 100% |
| HAMT-FeedTTA | 5 | 13,559 MiB | 100% |
| GOAT-FSTTA | 12 | 22,967 MiB | 100% |
| GOAT-FeedTTA | 6 | 16,378 MiB | 100% |

首个 DUET-FSTTA worker 曾在进入模型前因服务器默认 PATH 没有
`python3` 退出。该证据保存在 `attempt-00`；显式继承 DUET 环境 PATH 后，
同一个 logical job 以 `retry1` 完成全部 1,021 episodes。它不是算法失败，
也不计入 78 个正式任务的失败数。

## 2. 本轮协议修正

本轮相对旧 R2R 搜索有三项关键变化：

1. FeedTTA 恢复目标模型原生 argmax，不再把 AVN 适配时引入的 sampling
   强加给 VLN。
2. FeedTTA 在 episode 停止后用 evaluator 最终提交轨迹取得一次二值成功/
   失败反馈，并按模型搜索 `paper_full`、`last_crossmodal`、`action_head`
   三种更新范围。
3. FSTTA 的 FAST 方差历史改为整个 test stream 生命周期，不在每个 episode
   开始时清零。

FSTTA 不使用标签，属于无监督 TTA。FeedTTA 每个 episode 消费一次真实二值
反馈，必须写作 **FeedTTA†（feedback-supervised TTA）**，不能与 FSTTA
合并成无监督方法排名。

## 3. Source 与 SR-first winner

所有差值均为相对复用的标准 argmax Source 的绝对百分点。winner 按
`SR -> SPL -> lower drift -> fewer updates` 排序。

| 模型 | Source SR/SPL | FSTTA SR-first winner（ΔSR/ΔSPL） | FeedTTA† winner（ΔSR/ΔSPL） |
|---|---:|---:|---:|
| DUET | `78.84/72.88` | `79.24/72.34` (`+0.40/-0.54`) | **`79.33/73.88` (`+0.49/+1.00`)** |
| HAMT | `75.61/72.18` | **`76.40/72.97` (`+0.79/+0.79`)** | **`76.00/72.66` (`+0.39/+0.48`)** |
| GOAT | `84.82/80.05` | **`84.92/80.21` (`+0.10/+0.16`)** | `84.82/80.06` (`+0.00/+0.01`) |

不能把上表概括成“每个 winner 都优于 Source”：DUET 的 FSTTA SR-first
winner 牺牲了 SPL；GOAT 的 FeedTTA 只保持相同成功 episode 数，SPL 的
`+0.01` 也低于预注册的 `+0.10` 持平-SR 晋级门槛。

DUET-FSTTA 另有更合适的双指标 Pareto 点：

```text
lr_fast=1.8e-3, lr_slow=3e-4, M=8, N=4
SR/SPL=79.14/73.10, delta=+0.30/+0.22, relative drift=0.01750
```

因此，如果“优于 Source”要求 SR 和 SPL 同时上升，应冻结该 Pareto 点，
而不是自动生成的 SR-first winner。

## 4. FSTTA 结果

### 4.1 优胜配置与搜索区域

| 模型 | 推荐配置 | 解释 |
|---|---|---|
| DUET | `lf=1.8e-3, ls=3e-4, M=8, N=4` | 双指标均提高；SR-first 的 `lf=2.4e-3, ls=2e-4, M=3, N=16` 虽多 1 个 success，但 SPL 下降 0.54 pp |
| HAMT | `lf=8e-4, ls=2e-4, M=1, N=16` | 12 点中 8 点同时提高 SR/SPL；`lf=6e-4~8e-4, ls=2e-4~5e-4, M=1, N=16` 是稳定好区 |
| GOAT | `lf=1.8e-3, ls=1e-3, M=3, N=8` | 与 `lf=2.4e-3` 得到相同 SR/SPL；增益只有 1 个 success，必须多 seed 确认 |

跨模型最明显的坏区是高频 SLOW 更新与较高 slow LR 的组合。论文锚点
`lf=6e-4, ls=1e-3, M=3, N=4` 在 DUET 上降到 `71.50/59.39`
（`-7.34/-13.49`），在 HAMT 与 GOAT 上也低于 Source。该参数不能作为
VLN 三模型的共享默认值。

与上一轮 low-LR 搜索相比，DUET winner 为 `+0.00/+0.01`，HAMT 为
`+0.10/+0.18`，GOAT 完全相同。本轮三个旧 winner 的重跑 trajectory SHA、
updates、drift 和指标也逐项相同；新增收益主要来自新候选点，而不是方差历史
修正改变了旧轨迹。36 点中 32 点的 FAST LR scaler 还持续卡在上界 `b=1.1`，
后续应单独检查动态 scaler 是否实际提供了足够变化。

### 4.2 与 Tent 的关系

FSTTA 没有在三个模型上都严格胜过上一轮 Tent：采用 DUET 双指标点后，
DUET 和 HAMT 的 SR/SPL 都略高于 Tent；GOAT 与 Tent 的 SR 相同，但 SPL
低 0.11 pp。因此“FSTTA 理论上应全面好于 Tent”尚未被 seed-0 R2R 结果支持。

## 5. FeedTTA† 结果

### 5.1 修正前后

| 模型 | 旧 sampled low-LR | 本轮 argmax winner | 相对旧结果 | 本轮相对 Source |
|---|---:|---:|---:|---:|
| DUET | `78.75/69.34` | **`79.33/73.88`** | `+0.58/+4.54` | `+0.49/+1.00` |
| HAMT | `70.81/66.45` | **`76.00/72.66`** | `+5.19/+6.21` | `+0.39/+0.48` |
| GOAT | `84.82/79.07` | `84.82/80.06` | `+0.00/+0.99` | `+0.00/+0.01` |

旧 sampled zero-update control 相对标准 argmax Source 已分别损失：DUET
`-1.46/-5.45`、HAMT `-6.36/-7.50`、GOAT `-1.27/-1.38`。因此旧结果中
很大一部分退化来自动作协议混杂，而不是 FeedTTA 更新本身。本轮恢复原生
argmax 后，DUET 14 点中 12 点、HAMT 14 点中 3 点同时严格超过 Source；
GOAT 没有一个点同时严格超过 Source。

### 5.2 优胜配置与更新范围

| 模型 | winner | feedback 成功/失败 | relative drift |
|---|---|---:|---:|
| DUET | `last_crossmodal, lr=5e-6, gamma=.8, p=.05, alpha=+.1` | `810/211` | `0.00613572` |
| HAMT | `last_crossmodal, lr=2e-6, gamma=.9, p=.05, alpha=+.1` | `776/245` | `0.00333811` |
| GOAT | `action_head, lr=1e-6, gamma=.9, p=.05, alpha=+.1` | `866/155` | `0.00088220` |

三个模型的 `paper_full` 都不是最佳选择。在相同
`lr=5e-6, gamma=.9, p=.05, alpha=.1` 下：

| 模型 | `paper_full` | `last_crossmodal` | `action_head` |
|---|---:|---:|---:|
| DUET | `78.45/73.85` | `79.04/73.72` | `79.04/73.75` |
| HAMT | `73.65/70.18` | `75.61/72.48` | `75.61/72.18` |
| GOAT | `82.66/77.48` | `84.13/79.06` | `84.62/79.96` |

DUET 在 `3e-6~5e-6` 的窄范围较稳健；HAMT 只有
`2e-6 + last_crossmodal` 形成明确好区，`>=5e-6` 且 `gamma=.8` 会退化；
GOAT 必须把 LR 压到 `1e-6` 并只更新 action head，scope 越宽、LR 越高越差。
SGR 的 `p/alpha` 影响明显小于动作协议、scope 和 LR。

42 个候选总计消费 42,882 次二值 episode feedback，并执行 42,882 次更新；
它不是稀疏反馈方法。所有 42 点的反馈成功数都与 evaluator SR 对应的成功
episode 数一致，证明本轮反馈端点已经对齐。旧 DUET、GOAT winner 则分别
存在 11、8 个 episode 的端点错位。

## 6. 结论与后续门槛

1. **FSTTA**：DUET 应使用双指标 Pareto 点，HAMT 使用 SR-first winner；
   GOAT 只有 1 个 success 的增益。三个模型都可进入 order seeds 1/2 的
   matched-control 确认，但不能把 seed-0 调参值当作最终泛化结果。
2. **FeedTTA†**：DUET、HAMT 已达到至少增加 1 个 success 的预注册门槛，
   可进入 seeds 1/2；GOAT 的 success 数持平且 SPL 只 `+0.01`，不晋级。
3. 本轮说明 VLN FeedTTA 应使用目标导航模型原生 argmax 和 model-aware
   update scope；AVN 的原生 sampling 路径保持不变，不能反向用本结论修改。
4. 所有结果仍是在用于选参的 `val_seen`、单一固定顺序 seed 0 上获得；冻结
   参数后还需独立 `val_unseen` 确认，且不能把 `val_unseen` 回流用于选参。

## 7. 证据路径

- 搜索计划：
  [`R2R_FSTTA_FEEDTTA_POSTFIX_PLAN.md`](../../../experiments/R2R_FSTTA_FEEDTTA_POSTFIX_PLAN.md)
- 搜索 spec：
  [`r2r_fstta_feedtta_postfix_search_v1.json`](../../../experiments/r2r_fstta_feedtta_postfix_search_v1.json)
- 复用 Source 清单：
  [`r2r_reused_source_controls.json`](../../../manifests/r2r_reused_source_controls.json)
- 本地日志：`results/logs/r2r/hparam_search/vln-r2r-fstta-feedtta-postfix-v1-seed0/`
- 本地 compact results：
  `results/tuning/r2r/hparam_search/vln-r2r-fstta-feedtta-postfix-v1-seed0/`
- Formal run manifests：
  `results/runs/vln-r2r-fstta-feedtta-postfix-v1-seed0-*/manifest.json`

日志与 tuning results 保持本地，不提交 Git；本报告和 78 份 compact formal
manifest 保持可追踪，不包含数据、checkpoint、预测轨迹或原始日志。
