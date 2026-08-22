# REVERIE `val_seen` 小规模补充超参数搜索

## 目标与边界

本轮只修复 R2R 最优参数直接迁移到 REVERIE 后表现为负向或接近零的
model-method cell，不重新做全空间搜索。标准 argmax Source 和已经完成的 15 个
R2R-frozen transfer run 均复用，不重跑。EAM 在三个模型上的 RGSPL 均为正向，
DUET/HAMT 的 FeedTTA、ATENA 也已有明确收益，因此保留为 incumbent。

协议固定为 REVERIE `val_seen`、canonical order、model seed 0。筛选使用同一顺序的
前 256 个 episode，只负责晋级；正式候选从原始 Source checkpoint 启动新的独立
进程，在全部 1,423 个 episode 上重跑。筛选状态不能延续到正式运行。

Tent、FSTTA、EAM 属于无监督 TTA。FeedTTA 和 ATENA 只能消费官方 submitted
trajectory 在 endpoint reranking 和 object prediction 之后给出的二值导航成功反馈：
FeedTTA 每 episode 消费一次，ATENA 仅在 entropy gate 查询时消费；不得使用
grounding label 或 simulator distance fallback。

## 为什么只搜索 8 个 cell

现有 R2R-frozen transfer 的主指标 RGSPL 相对 Source 变化如下：

| 模型 | Tent | FSTTA | EAM | FeedTTA | ATENA |
|---|---:|---:|---:|---:|---:|
| DUET | +0.52 | +0.42 | +2.83 | +1.43 | +4.41 |
| HAMT | +0.43 | **-0.36** | +2.08 | +0.43 | +5.97 |
| GOAT | +0.23 | **-0.02** | +0.67 | +0.19 | +0.58 |

因此目标 cell 为：

- DUET：Tent、FSTTA；
- HAMT：FSTTA、FeedTTA；
- GOAT：Tent、FSTTA、FeedTTA、ATENA。

其余 7 个 cell 不产生新任务，最终表直接继承已认证 incumbent。EAM 不补搜，符合
此前“EAM 已稳定有效，不继续扩量”的决策。

## 候选设计

所有没有列出的超参数保持当前实现和方法定义不变。历史 REVERIE 搜索仅用于定位
可能的局部区域；其中旧 FeedTTA/ATENA 结果不能作为正式证据，所以相关点必须在
当前 argmax 与纠正后的 binary-feedback endpoint 下重新验证。

| Cell | 256-episode 候选 | 设计依据 |
|---|---:|---|
| DUET–Tent | 2 | 固定 `update_interval=1`，比较当前 `last_k_ln=9` 与 full-LN，学习率均为 `1.5625e-5`。 |
| DUET–FSTTA | 2 | 复测长 slow window 区域：`(lf,ls,M,N)=(1.8e-3,1e-3,4,32)` 与更低 fast LR 的 `(6e-4,1e-3,3,32)`。 |
| HAMT–FSTTA | 3 | 当前 drift 偏大且 RGSPL 负向；固定较短 fast window，比较历史峰值和两个逐级降低 LR 的点。 |
| HAMT–FeedTTA | 2 | 在当前 argmax 实现下复测旧结果提示的 `lr=5e-6, gamma=.99` 区域，只小幅改变 SGR 强度。 |
| GOAT–Tent | 2 | 固定 `update_interval=1`，只测 `1.5625e-5` 与 `3e-5`。 |
| GOAT–FSTTA | 2 | 检查低 fast LR 与长 slow window 两个已知 ridge。 |
| GOAT–FeedTTA | 2 | 保持 `action_head`，把 LR 从当前 `1e-6` 提到 `5e-6/1e-5`，并在当前 argmax 协议下复验。 |
| GOAT–ATENA | 3 | 在纠正后的 lazy-query endpoint 下复测 `5e-6/1e-5` query LR，并比较 threshold `0.1/0.2`。 |

完整数值由
[`reverie_val_seen_small_hparam_search_v1.json`](reverie_val_seen_small_hparam_search_v1.json)
唯一约束。

## 晋级与最终固化

每个目标 cell 的 256-episode 候选按 RGSPL 排序，依次用 RGS、SPL、SR、较小参数
漂移、较少更新次数打破并列。每个 cell 最多晋级 1 个新候选。

正式 1,423-episode 结果必须满足：

- 完整 episode 与 adapter accounting；
- 无监督方法没有 binary feedback；
- FeedTTA/ATENA 的反馈 endpoint、时序和计数正确；
- SR 不低于复用 Source 超过 1.0 个百分点；
- run manifest 绑定 Git commit、配置、checkpoint digest、dataset digest、顺序、seed、
  hardware，并认证 metrics 与 diagnostics artifact。

正式候选与同 cell 的 R2R-transfer incumbent 按 RGSPL、RGS、SPL、SR 比较。只有新
候选更好且通过 SR floor 时才替换 incumbent；否则保留 incumbent。未搜索的 7 个
cell 原样保留。最终输出还会显式记录每个 winner 是否严格优于 Source 的 RGSPL，
不会把未超过 Source 的结果描述成有效增益。

Source 与 incumbent 表中的指标不能只来自 JSON literal：runner 会从各自 formal
manifest 认证的 `valid.txt` 重新解析指标，并逐项与 registry/spec literal 比对后才允许
参与选择。新 full run 同样会重新认证 manifest 中列出的 metrics 与 diagnostics
artifact；缓存的 `metrics.json` 不能替代该检查。

## 执行与资源策略

执行顺序严格为 `DUET screening → DUET full → HAMT screening → HAMT full → GOAT
screening → GOAT full`。当前模型的筛选、晋级、正式运行和阶段汇总全部结束后才能进入
下一个模型，不允许不同导航模型重叠。一个模型内部让不同 TTA 方法并行，但同一种
方法同一时刻最多运行 1 个候选，避免同方法候选挤占并行槽：

| 模型 | 最大并发 | 同时涉及的方法 |
|---|---:|---|
| DUET | 2 | Tent、FSTTA |
| HAMT | 2 | FSTTA、FeedTTA |
| GOAT | 4 | Tent、FSTTA、FeedTTA、ATENA |

runner 使用共享 GPU reservation ledger 和 29,000 MiB 投影上限；真正启动前仍应先
确认服务器没有遗留任务。`BATCH.json` 会固定启动时的 Git commit；每个 model
phase、每次 attempt materialize 以及实际 launch 前都会重新比对当前 HEAD，resume
也执行相同检查，中途 pull 后将 fail closed。正式启动还会通过
`git ls-files --error-unmatch` 确认 runner 与传入 spec 均已被当前 HEAD 跟踪，未提交
的新脚本或临时 spec 不能绕过门禁；同时拒绝 `core/`、`tools/`、`vln/baselines/`、
`vln/navtta_vln/`、`vln/scripts/`、`vln/experiments/`、`vln/manifests/` 下任何未跟踪
文件。该完整检查会在每个 phase、materialize、launch、promotion、stage summary 和
最终 selection 前重复。退出码为 0 但证据验证失败的 attempt 只有在显式
`--retry-failed` 时才会归档旧 metrics/result/formal evidence 并创建新 attempt。
最终选择拒绝不足 15 个 cell 的输出。计划检查不会启动实验：

```bash
python3 vln/scripts/run_reverie_small_hparam_search.py --stage plan --verbose
```

人工审核并提交代码后，正式运行才可使用：

```bash
python3 vln/scripts/run_reverie_small_hparam_search.py \
  --stage all --confirm-reviewed
```

中断后必须使用同一个 batch ID 并显式 `--resume`；失败任务只有再加
`--retry-failed` 才会产生新 attempt。

## 预算

- 256-episode screening：18 个任务，4,608 episode evaluations；
- full `val_seen`：最多 8 个任务，11,384 episode evaluations；
- 总上限：26 个新任务，15,992 episode evaluations；
- Source：0 个新任务；incumbent：0 个重跑。

相比原 36 screening + 15 full 的方案，新任务从 51 降至 26，episode evaluations
从 30,561 降至 15,992，均减少约 49%。
