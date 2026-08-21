# R2R EAM / ATENA formal confirmation

本批只补齐五个已经冻结的 R2R `val_seen` cell，不重新搜索超参数：

```text
DUET EAM -> DUET ATENA -> HAMT EAM -> HAMT ATENA -> GOAT EAM
```

每个 phase 只有一个 1,021-episode、canonical order seed-0 任务。Source 继续复用
`vln/manifests/r2r_reused_source_controls.json`，不启动 Source、sampled control、
其他方法或其他 split。

## 冻结配置

| Phase | 配置 |
|---|---|
| DUET EAM | `lr=1e-5, confidence_scale=.5, memory_size=64, batch_size=8, update_interval=8` |
| DUET ATENA | `lr_query=1.6e-6, lr_self=2e-7, mix_lambda=.5, query_threshold=0` |
| HAMT EAM | `lr=1e-6, confidence_scale=.6, memory_size=32, batch_size=8, update_interval=8` |
| HAMT ATENA | `lr_query=3.2e-6, lr_self=4e-7, mix_lambda=.25, query_threshold=.1` |
| GOAT EAM | `lr=3e-6, confidence_scale=.3, memory_size=32, batch_size=8, update_interval=4` |

EAM 共同固定 `optimizer=Adam, max_grad_norm=0, episodic=false`。ATENA 共同固定
`self_loss_weight=.1, optimizer=AdamW, weight_decay=.01, max_grad_norm=0,
action_selection=argmax, episodic=false`。这些对象与 Cartesian v2 的
`FROZEN_HPARAMETERS.json` 一致；本批不得根据新指标改参数。

EAM 的旧值仅是确定性回归预期：DUET `80.31/74.75`、HAMT `76.69/73.40`、
GOAT `85.31/80.55`。若重跑不一致，保留新 formal run，但在进入结果登记册前先
审计代码或环境差异。DUET/HAMT 的旧 ATENA 指标使用错误反馈端点，不能作为数值
相等门槛，也不能在新运行失败时回填。

## Corrected ATENA 合约

机器 spec 显式设置 `protocol.enforce_corrected_result_contract=true`。runner 在每个
ATENA job 完成后、释放下一 phase barrier 前强制检查：

- 参数使用 target-native argmax；
- `binary_feedback_endpoint` 为
  `r2r_submitted_trajectory_evaluator_success_lazy_query`；
- `queries == feedback_observed_episodes`，并且 1,021 个 episode 全部执行 query gate；
- `queries + self_label_episodes == 1021`；
- queried/self success 数分别落在其 episode 预算内；
- query rate 与 feedback observation rate 都精确对应 `queries / 1021`；
- `query_threshold=0` 的 DUET job 必须查询全部 1,021 episodes。

任何一项失败都会生成 `POSTFIX_CONTRACT_ERROR.json`、阻断下游 phase，并要求使用
同一 batch 的 `--resume --retry-failed` 重试。不能把仅有 `status=completed` 的旧
ATENA manifest 当作 corrected evidence。

## 并发与证据

五个 phase 的 `production_cap` 均为 1，正式启动再加 `--max-workers 1` 作为全局
上限；因此不需要新的并发校准。EAM/ATENA prelaunch gate 分别为 24,000/23,500
MiB，观察到 30,000 MiB 时沿用 runner 的紧急终止线。严格 model/method barrier
禁止跨 phase 重叠。

完整任务不携带 `--episode-limit` 或 `--order-seed`，所以
`run_source_eval.sh` 会在 `vln/results/runs/` 创建、finalize 并立即验证 formal
manifest。最终必须得到恰好五份 `status=completed, exit_code=0` 的 manifest；
同步前还应在执行服务器上重新运行 manifest validator，并保存每份文件的 SHA256。

本 spec 复用通用 V1 schema。V1 要求声明
`feedtta_sampled_no_update_per_setting=true`；由于本批没有任何 FeedTTA phase，
该条件是空真，实际 phase 中没有 `feedtta_control`，控制任务数仍为 0。

## 测试与启动

提交并同步实现后先运行：

```bash
python3 vln/tests/test_r2r_eam_atena_formal_confirmation_plan.py
python3 vln/tests/test_r2r_local_refinement.py
python3 vln/tests/test_discrete_tta.py
python3 vln/tests/test_tta_config_cli.py
python3 tools/verify_layout.py
```

正式 plan 和执行使用新 batch，不能复用 Cartesian 或 targeted batch：

```bash
SPEC=vln/experiments/r2r_eam_atena_formal_confirmation_v1.json
BATCH=vln-r2r-eam-atena-formal-confirm-v1-seed0
PY=/root/autodl-tmp/conda/envs/duet/bin/python

"$PY" vln/scripts/run_r2r_local_refinement.py \
  --spec "$SPEC" --batch-id "$BATCH" --gpu 0 --max-workers 1 \
  --plan-only --print-commands

"$PY" vln/scripts/run_r2r_local_refinement.py \
  --spec "$SPEC" --batch-id "$BATCH" --gpu 0 --max-workers 1 \
  --resume --confirm-reviewed

"$PY" vln/scripts/run_r2r_local_refinement.py \
  --batch-id "$BATCH" --status --watch
```

仅在失败且没有下游 phase 已启动时重试：

```bash
"$PY" vln/scripts/run_r2r_local_refinement.py \
  --spec "$SPEC" --batch-id "$BATCH" --gpu 0 --max-workers 1 \
  --resume --retry-failed --confirm-reviewed
```

权威机器配置为
[`r2r_eam_atena_formal_confirmation_v1.json`](r2r_eam_atena_formal_confirmation_v1.json)。
完成本批只关闭五个 cell 的 manifest/ATENA-endpoint 缺口；这些结果仍是
`val_seen` seed-0 选择证据，不替代 order robustness 或冻结后的独立 split 评估。
