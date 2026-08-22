# R2R-CE 补搜后固化与 val_unseen 计划

## 执行边界

本阶段必须等待 `vln-r2r-ce-targeted-supplement-v1-seed0` 完整结束并下载
`RESULTS.json`、所有 full 结果和 formal manifests。补搜结束前不生成 winner
registry，也不启动 `val_unseen`；`val_unseen` 指标不参与任何超参数选择。

机器可读入口：

- 固化规则：`vln/experiments/r2r_ce_postsearch_finalization_v1.json`
- 固化工具：`vln/scripts/build_r2r_ce_final_registry.py`
- 冻结评估 runner：`vln/scripts/run_r2r_ce_val_unseen_frozen_eval.py`

## val_seen winner 固化

候选范围严格限定为首轮每个 cell 的 778-episode finalist，以及定点补搜对同一
cell 产生的最多一个 778-episode confirmation。不得加入 screening 指标，也不得
查看 `val_unseen` 后回改。

固化器不会把 `PROMOTION.json` / `CONFIRMATION.json` 当作自证摘要。它会校验
PLAN 和四个 PHASE 的 canonical order、job cap、资源限制与 spec 身份，逐项重建
15 个 screening job 和动态 full job，认证 job/config/result 路径及 retry 链，随后
从原始 100-episode 输出重放 `select_finalist`。每个目标 cell 必须具有可重放的
PROMOTION disposition；晋级项还必须具有匹配的 full job 与 CONFIRMATION，全部
未晋级的 full phase 则必须同时具有空 CONFIRMATION 和 `EMPTY_COMPLETE.json`。
补搜 runner 在同一个进程锁临界区内写完最终 `RESULTS.json`。

首轮 10 个 full finalist 的紧凑、可跟踪索引保存在
`vln/results/final/r2r-ce/initial_full_candidates.json`；其中同时固定原始
`FINAL_SELECTION.json` 的路径与 SHA256，避免依赖把 raw logs 提交到 Git。
固化时还会认证原始搜索 spec/commit、matched Source、winner membership、正式
manifest、aggregate 以及 formal-manifest 中的 `tta_diagnostics.json`；FSTTA 的
`updates` 只从该认证 diagnostics 读取。Source 的 aggregate 与逐 episode文件按
name、canonical path、size、SHA256 精确绑定到 formal manifest，并重算全部 778
episodes 的聚合值与覆盖范围。

10 个 cell 中，EAM、ATENA 和 BEVBert–Tent 直接沿用首轮 winner；ETPNav–Tent、
两组 FSTTA 和两组 FeedTTA 比较首轮与补搜 full 结果。排序使用 full SPL、SR；
满足精确并列时优先补搜的有效候选。所有候选须满足 `SR >= Source SR - 1.5pp`。
FSTTA 额外要求 `M<=15` 且补搜 confirmation 的 `updates>0`，因此首轮 `M=16`
zero-update 结果不能作为最终 winner；若补搜没有产生有效 FSTTA confirmation，
固化工具会失败并明确要求进一步补搜，不会制造 winner。

补搜证据齐全后运行：

```bash
python3 vln/scripts/build_r2r_ce_final_registry.py \
  --supplement-results \
  vln/results/logs/r2r-ce/hparam_search/vln-r2r-ce-targeted-supplement-v1-seed0/RESULTS.json \
  --write
```

该命令原子生成：

- `vln/results/final/r2r-ce/selected_winners.json`
- `vln/results/final/r2r-ce/registry.json`
- `vln/experiments/r2r_ce_val_unseen_frozen_eval_v1.json`

生成后需人工复核并提交；正式 runner 会要求这些依赖均已被 Git 跟踪且工作树无
tracked 修改。

registry 校验不是仅检查已生成 JSON：每次使用时都会从固定的初始
`FINAL_SELECTION`、补搜原始 job 证据、正式 manifests 和 selection policy
确定性重建参数、指标、eligibility 与 winner，再逐字重建 registry；任何偏差均
fail-closed。

## val_unseen Source 复用

`val_unseen` 为 canonical seed-0 顺序，共 1,839 episodes，order SHA256 为
`e5a86bf6…07a8a`。ETPNav 和 BEVBert 的 Source 已在 2026-08-10 完成，分别为：

| 模型 | SR | SPL |
|---|---:|---:|
| ETPNav | 55.8999 | 48.3048 |
| BEVBert | 58.2382 | 48.2588 |

两个原始 formal manifest 已从归档目录恢复到 canonical
`vln/results/runs/`，并由
`vln/manifests/r2r_ce_val_unseen_reused_source_controls.json` 固定 manifest、
checkpoint、dataset、order、aggregate 和逐 episode artifact 的 SHA256。后续预算
始终是 **10 TTA + 0 Source**；任何证据缺失均 fail-closed，禁止以重跑 Source
修补。

## 冻结 val_unseen 执行

顺序固定为 ETPNav 五种方法，再 BEVBert 五种方法；模型之间有严格 barrier。
模型内最多并发 3 个任务，所以每个模型自然形成 `3+2`，不会同时运行两个导航
模型。每次启动前都重新检查 GPU/cgroup：显存门槛 22,000 MiB、单任务预算
8,000 MiB、aggregate 上限 30,000 MiB；cgroup 分别为 55、20、75 GiB，启动间隔
15 秒。这沿用已经审查的 R2R-CE 三并发资源门槛。

正式执行固定唯一 batch ID，并持有覆盖整个 campaign 生命周期的进程锁。每次
snapshot/launch 都在共享 GPU launch guard 内登记显存与 cgroup reservation；
恢复时会先重新认领仍存活 worker 的 reservation，再放行启动握手。异常清理只在
本 scheduler 启动的 worker 已确认退出后释放其 reservation，不会释放 resume
接管的外部存量 worker。存活判定覆盖整个独立 process group：每个 attempt 的
后代进程继承唯一 token；即使外层 shell 异常退出，仍在运行的评测进程也不会被
误判为 orphan、释放 reservation 或触发重复 retry。

所有任务均由 Source checkpoint 独立初始化，固定 registry 中的 val_seen winner
参数，并省略 `--order-seed`（canonical seed 0 是 R2R-CE runner 默认协议）。正式
启动命令为：

```bash
python3 vln/scripts/run_r2r_ce_val_unseen_frozen_eval.py \
  --batch-id vln-r2r-ce-val-unseen-frozen-eval-v1-seed0 \
  --confirm-reviewed
```

恢复使用 `--resume`；只有失败或验证失败任务才使用
`--resume --retry-failed`，每项最多一次 retry，且 retry 会获得新 run tag。runner 验证 1,839-episode
diagnostics、FeedTTA/ATENA 反馈记账、FSTTA stream lifetime、aggregate artifact、
formal manifest identity、checkpoint/dataset/order digest，且不会写 promotion 或
selection 文件。

attempt 目录按 staging 后 rename 原子落盘；attempt 只能是连续的 `00,01`。重试
会保留旧目录及不可变 `validation_error.json`，并在新 job 中固定旧 job/error 的
SHA256。`--status` 是只读路径，不会与 scheduler 并发重写 metrics 或错误记录。
