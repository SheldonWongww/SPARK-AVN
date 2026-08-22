# R2R-CE 超参数搜索与补搜最终分析

## 范围与证据

- Benchmark：R2R-CE v1.3 unified `val_seen`。
- 模型：ETPNav、BEVBert。
- 顺序：canonical order，`order_seed=0`；每个模型均为 778 episodes。
- 首轮批次：`vln-r2r-ce-small-hparam-search-v1-seed0`，40 个 100-episode
  screening jobs + 10 个 778-episode full jobs，共 50/50 成功。
- 定点补搜：`vln-r2r-ce-targeted-supplement-v1-seed0-rerun2`，15 个
  screening jobs + 4 个实际晋级的 full jobs，共 19/19 成功。
- Source 全部复用已有正式结果，没有重新执行。
- 本地原始证据包含首轮 50 份 exitcode、50 份 metrics 和 50 份
  `tta_diagnostics.json`；补搜 `RESULTS.json` SHA256 为
  `5de3bd288921299391a51044a30b2a4511b09a5061e894f07f1da8a084fe88cf`。

首轮的逐任务表、固定顺序稳定性分析和 bootstrap 结果见
[`R2R_CE_SMALL_HPARAM_SEARCH_V1.md`](R2R_CE_SMALL_HPARAM_SEARCH_V1.md)。本文不改变
该报告，因为补搜 spec 已绑定它的 SHA256；本文合并首轮和补搜后给出最终结论。

## 最终 `val_seen` 选择

SPL 为主指标、SR 为次指标，所有选择都只使用 `val_seen`。FeedTTA 和 ATENA
使用 episode-level binary navigation feedback，以 `†` 标识，不能与无监督方法
视为相同监督预算。

| 模型 | 方法 | 最终关键超参数 | SR | ΔSR | SPL | ΔSPL | 来源 |
|---|---|---|---:|---:|---:|---:|---|
| ETPNav | Source | — | 67.4807 | — | 59.9694 | — | 复用 control |
| ETPNav | Tent | `lr=3e-7, last4-LN, I=1` | 67.6093 | +0.1285 | 60.0980 | +0.1285 | 补搜 |
| ETPNav | FSTTA | `lf=3e-7, ls=1e-5, M/N=4/16` | 67.4807 | 0.0000 | 59.9694 | 0.0000 | 补搜 |
| ETPNav | EAM | `lr=1e-6, I=1, mem/b=32/8` | 67.4807 | 0.0000 | 60.8084 | +0.8390 | 首轮 |
| ETPNav | FeedTTA† | `lr=1e-8, last_crossmodal` | 67.4807 | 0.0000 | 59.9694 | 0.0000 | 首轮 |
| ETPNav | ATENA† | `lq=5e-7, ls=1e-8, lambda=.5, threshold=.3` | 67.7378 | +0.2571 | 61.2303 | +1.2609 | 首轮 |
| BEVBert | Source | — | 68.3805 | — | 59.9207 | — | 复用 control |
| BEVBert | Tent | `lr=1.5625e-5, all-LN, I=1` | 69.1517 | +0.7712 | 62.2365 | +2.3158 | 首轮 |
| BEVBert | FSTTA | `lf=3e-7, ls=1e-5, M/N=4/16` | 68.5090 | +0.1285 | 59.9985 | +0.0778 | 补搜 |
| BEVBert | EAM | `lr=1e-6, I=1, mem/b=64/8` | 68.8946 | +0.5141 | 61.2692 | +1.3485 | 首轮 |
| BEVBert | FeedTTA† | `lr=2e-6, last_crossmodal` | 68.6375 | +0.2571 | 60.2761 | +0.3554 | 补搜 |
| BEVBert | ATENA† | `lq=2e-6, ls=4e-8, lambda=.25, threshold=.1` | 71.4653 | +3.0848 | 64.8505 | +4.9298 | 首轮 |

最终 registry 的机器可读来源为
[`selected_winners.json`](../../final/r2r-ce/selected_winners.json) 和
[`registry.json`](../../final/r2r-ce/registry.json)。ETPNav–FSTTA 与
ETPNav–FeedTTA 仍未建立严格正收益，因此不能声称 10 个 model-method cell 全部
优于 Source。

## 补搜如何改变结论

### Tent

ETPNav 在 100-episode 前缀上测试 `3e-7/1e-6/3e-6` 时，SR/SPL 全部与 Source
相同，但相对参数漂移随 LR 近似线性增加：`2.71e-4/9.01e-4/2.69e-3`。最低点
在 full split 上产生 `+0.1285pp` SPL，说明短前缀的离散导航指标无法可靠区分
非常小的适配效应。BEVBert 则在更大的 `1.5625e-5`、all-LN 范围取得
`+2.3158pp`。因此 Tent 的 LR 和可训练 LayerNorm 范围具有明显模型依赖性；
`update_interval=1` 应继续固定，不作为搜索维度。

### FSTTA

首轮 `M=16` 大于观测到的单 episode 最大 15 个高层动作，两个 full run 均为
`updates=0` 的无效 no-op。补搜把 `M` 缩到 4 后，ETPNav/BEVBert full 分别执行
1,327/1,348 次更新，证明实现路径已真正工作；但收益分别为 `0.0000` 和
`+0.0778pp` SPL。

在补搜前缀中，`M=4` 时把 `(lf,ls)` 从 `(3e-7,1e-5)` 提到
`(1e-6,3e-5)`，指标不变而漂移约增至 3 倍；`M=1` 将更新数从约 190 提高到
约 900，仍未改变前缀指标。由此可见当前最重要的超参数首先是窗口是否允许实际
更新，其次才是 LR；单纯增大学习率或更新密度主要放大漂移，尚无收益证据。
若以后做方法机理复核，优先单独测试论文式 `M=3,N=4`，不要再使用 `M=16`，也
不要把 `M=1` 的退化窗口当作 FSTTA 的代表结果。

### EAM

首轮把 LR 与更新频率一起变化：极低 LR、低频更新均近似 Source；
`lr=1e-6,I=1` 才进入有效区，两个模型分别得到 `+0.8390/+1.3485pp` SPL。
这说明有效更新强度是主因，但现有设计不能完全分离 LR 和 interval 的主效应。
两个 full stream 都出现早期收益减弱的 warning，因此更长 split 上应避免继续提高
累计强度；后续优先做固定参数的顺序/稳定性复核，而不是扩大网格。

### FeedTTA

补搜固定 `last_crossmodal, gamma=.99, p=.05, alpha=-.2, argmax`，只提高 LR。
ETPNav 在 `3e-7/1e-6/2e-6` 上均执行 100 次更新，但 100 个 episode 的导航记录
变化数都是 0，因此没有候选晋级；最终仍保留首轮 Source 等价的 `1e-8`。
BEVBert 的导航记录变化数随 LR 为 `0/2/6`，`2e-6` 的 screening SPL
`+0.0824pp`，full SPL `+0.3554pp`。这表明 FeedTTA 存在明显的模型相关动作边界：
LR 是主要敏感因素，但“参数有漂移”不等于“策略动作发生变化”。ETPNav 更高 LR
靠近已有不稳定区，当前预算下不再追逐极窄边界。

### ATENA

两个模型的较高 query LR 候选均明显优胜，ETPNav/BEVBert 的 full SPL 分别提高
`+1.2609/+4.9298pp`。它对 query LR 最敏感，同时还受 query threshold 和
self/query LR 比例影响。当前查询率约 75%，若后续关注反馈成本，应固定已选 LR
后单独提高 threshold；不要把反馈效率实验混入效果主表选参。

## 后续决策

1. 当前定点补搜已经修复 ETPNav–Tent、两组 FSTTA 的 zero-update 问题，并为
   BEVBert–FeedTTA 找到正收益；不再开启第三轮 R2R-CE 网格，以免在同一
   `val_seen` 顺序上继续追逐微小波动。
2. 固化上述 10 个配置后，直接在 1,839-episode `val_unseen` canonical seed-0
   顺序上评测；`val_unseen` 只报告，不参与重选。
3. ETPNav–FSTTA、ETPNav–FeedTTA 的零增益应如实保留。论文式 FSTTA 窗口或更高
   FeedTTA LR 若要研究，应作为独立消融，而不是在看到 `val_unseen` 后回补主表。
4. 连续环境中累计更新次数远高于离散 VLN。迁移到其他长流设置时，优先控制
   “LR × 更新频率/窗口产生的有效更新数”，并同时查看参数漂移、动作变化数和
   分段性能，不能只看最终 SR/SPL。
