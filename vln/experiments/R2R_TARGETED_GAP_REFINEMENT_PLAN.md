# R2R targeted gap refinement

本轮只处理三组仍与 Source 持平或增益过小的组合，不重跑 Source，也不重新展开
其他模型或方法：

```text
DUET Tent (9) -> GOAT FeedTTA (11) -> GOAT ATENA (9)
```

正式预算为 29 个完整 `val_seen` 任务，每个任务固定 1,021 episodes、order seed 0，
模型/方法阶段之间设置严格 barrier。

## 设计依据

### DUET–Tent

固定论文协议要求的 `update_interval=1` 后，旧搜索的 all-LayerNorm 最优结果与
Source 完全相同：`78.84/72.88`。单独增加学习率只能形成提高 SPL、降低 SR 的
trade-off；`lr=3e-5` 为 `78.55/74.46`。DUET 会 detach 早期语言与全景缓存，
all-LayerNorm 名义上选择的许多层不在当前 entropy 图中，因此本轮保持更新频率、
优化器和损失不变，只缩小可训练 LayerNorm 范围：

```text
scope = last_ln, last_k_ln(k=3), last_k_ln(k=4)
lr    = 3e-6, 1e-5, 3e-5
```

共 9 个此前未运行的点。

### GOAT–FeedTTA

修正为原生 argmax、正确 evaluator feedback 和模型感知 scope 后，14 个可比点中
`action_head, lr=1e-6, gamma=.9` 最好，但只有 `84.82/80.06`，成功 episode 数
仍与 Source 相同。更宽 scope 和 `lr>=3e-6` 都显著退化，因此固定
`action_head, p=.05, alpha=.1`，只在最低学习率边界附近插值：

```text
lr    = 5e-7, 7.5e-7, 1e-6, 1.25e-6, 1.5e-6, 2e-6
gamma = .8, .9
```

删除已完成的 `(1e-6,.9)` 后共 11 个新点。

### GOAT–ATENA

旧最优 `4e-7/5e-8, lambda=0, threshold=.15` 为 `84.92/80.24`，只多成功一个
episode。审计同时发现旧实现向 ATENA 提供了 simulator stop 时的距离反馈，而
GOAT 会在 stop 后重排正式提交终点；25 个全查询历史点中有 19 个出现反馈成功数
与 evaluator 成功数不一致，差值范围为 -1 到 +6。

本轮先将 ATENA 改为惰性 evaluator callback：只有 entropy gate 发起 query 时才
读取提交轨迹的正式 success，未查询 episode 不访问 oracle。随后重跑旧数值 anchor，
并围绕唯一有效区域搜索：

```text
lr_query = 3.5e-7, 4.5e-7; lr_self/lr_query = 1/8
threshold = .13, .15, .17
anchor = 4e-7/5e-8, lambda=0, threshold=.15
self-LR ablation = 4e-7/{2.5e-8,7.5e-8}, lambda=0, threshold=.15
```

共 9 个任务，其中 8 个新点和 1 个修正实现后的协议 anchor。

## 并发与晋级

直接复用已经完成的显存实测，不再逐任务校准：DUET Tent 最多 10 并发、GOAT
FeedTTA 6 并发、GOAT ATENA 5 并发。由于各阶段分别只有 9/11/9 个任务，实际
调度分别为 9、6+5、5+4。

Source 为 DUET `78.84/72.88`、GOAT `84.82/80.05`。候选至少需要多成功一个
episode 且 SPL 不低于 Source，或者成功数持平但 SPL 提高至少 0.10 pp，才进入
后续 order seeds 1/2。ATENA 仅保持旧有 867 successes 仍视为边缘结果；明显提升
门槛为至少 868 successes。

## 启动

```bash
SPEC=vln/experiments/r2r_targeted_gap_refinement_v1.json
BATCH=vln-r2r-targeted-gap-refine-v1-seed0
python3 vln/scripts/run_r2r_local_refinement.py \
  --spec "$SPEC" --batch-id "$BATCH" --plan-only --print-commands
python3 vln/scripts/run_r2r_local_refinement.py \
  --spec "$SPEC" --batch-id "$BATCH" --resume --confirm-reviewed --gpu 0
python3 vln/scripts/run_r2r_local_refinement.py \
  --batch-id "$BATCH" --status
```

权威机器可执行配置为
[`r2r_targeted_gap_refinement_v1.json`](r2r_targeted_gap_refinement_v1.json)。
