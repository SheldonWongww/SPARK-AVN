# R2R targeted gap expansion v2

本轮继续只搜索三个弱组合，不重跑 Source，不改动其他模型/方法：

```text
DUET Tent (24) -> GOAT FeedTTA (36) -> GOAT ATENA (37)
```

总计 97 个完整 `val_seen` 任务。所有候选都与 corrected targeted-v1 精确去重；
FeedTTA 还与 postfix corrected 候选去重。ATENA 的历史高-SPL点允许在本轮重跑，
因为旧任务使用的是修正前的 simulator feedback，本轮使用惰性 evaluator feedback。

## DUET–Tent：24 点

固定 `update_interval=1`、AdamW、无 weight decay，不恢复历史上不符合当前协议的
`update_interval=16`。只扩 replay-connected LayerNorm 深度与 LR：

| `last_k_ln` | LR |
|---:|---|
| 4 | 2e-5, 4e-5, 5e-5, 7.5e-5, 1e-4 |
| 6 | 3e-6, 1e-5, 3e-5, 5e-5, 1e-4 |
| 9 | 1e-6, 3e-6, 1e-5, 3e-5, 5e-5 |
| 15 | 3e-7, 1e-6, 3e-6, 1e-5, 3e-5 |
| 29 | 3e-7, 1e-6, 3e-6, 1e-5 |

`k=6/9/15` 逐步覆盖最后 1/2/4 个 global cross-modal layers 与 heads；`k=29`
覆盖 local/global 两个仍在 replay graph 中的分支，同时避开被 detach 的早期语言和
全景缓存。scope 越宽，LR 上限越保守。

## GOAT–FeedTTA：36 点

固定 target-native argmax、`action_head`、Adam。30 点扩充 LR–gamma 面：

```text
lr={6.25e-7,8.75e-7,1.125e-6,1.375e-6,1.75e-6}
  × gamma={.8,.9}

lr={7.5e-7,1e-6,1.25e-6,1.5e-6,2e-6}
  × gamma={.70,.75,.85,.95}
```

主 profile 固定 `p=.05, alpha=.1`。另在三个已有 +1-success ridge
`(1e-6,.8)`、`(1.25e-6,.8)`、`(1.5e-6,.9)` 上各运行两种 SGR：
`(p,alpha)=(0,0)` 与 `(.1,-.1)`，共 6 点。scope 明显比 SGR 更重要，因此
不再扩宽 scope，也不做 SGR 大网格。

## GOAT–ATENA：37 点

全部使用修正后的 lazy evaluator feedback。

低 LR/query gate 共 20 点：五个 8:1 LR pair

```text
(2.75e-7,3.4375e-8), (3e-7,3.75e-8),
(3.25e-7,4.0625e-8), (3.75e-7,4.6875e-8),
(4.25e-7,5.3125e-8)
```

分别乘 `threshold={.14,.15,.17,.20}`，固定 `lambda=0`。

self-LR 解耦共 5 点：固定 `lr_query=3.75e-7, threshold=.15, lambda=0`，取
`lr_self={1.171875e-8,2.34375e-8,3.515625e-8,5.859375e-8,7.03125e-8}`。

高-SPL全查询支线共 12 点：

```text
(lr_query,lr_self)={
  (4.8e-6,6e-7), (5.6e-6,7e-7),
  (6.4e-6,8e-7), (7.2e-6,9e-7)
}
× lambda={.625,.75,.875}, threshold=0
```

该支线单独按“SR 不显著下降时最大化 SPL”解释，不能覆盖主 SR-first winner。

## 并发、门槛与启动

复用 targeted-v1 的实测上限：Tent 10 并发、FeedTTA 6 并发、ATENA 5 并发。
阶段间严格 barrier。新的晋级门槛提高为：至少多成功 2 个 episode 且 SPL 不低于
Source，或多成功 1 个同时 SPL 至少 `+0.10 pp`。

```bash
SPEC=vln/experiments/r2r_targeted_gap_expansion_v2.json
BATCH=vln-r2r-targeted-gap-expand-v2-seed0
PY=/root/autodl-tmp/conda/envs/duet/bin/python
"$PY" vln/scripts/run_r2r_local_refinement.py \
  --spec "$SPEC" --batch-id "$BATCH" --plan-only --print-commands
"$PY" vln/scripts/run_r2r_local_refinement.py \
  --spec "$SPEC" --batch-id "$BATCH" --resume --confirm-reviewed --gpu 0
```

权威机器可执行配置为
[`r2r_targeted_gap_expansion_v2.json`](r2r_targeted_gap_expansion_v2.json)。
