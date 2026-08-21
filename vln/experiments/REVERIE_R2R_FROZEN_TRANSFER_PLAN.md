# REVERIE frozen transfer from R2R

本轮不是 REVERIE 超参数搜索。它把 R2R canonical-order seed-0 registry 中冻结的
3 个模型 × 5 种 TTA 配置原样迁移到 REVERIE `val_seen`，每个 cell 只运行一次，
不读取 REVERIE 指标重新选参。标准 argmax Source 不重跑，复用
[`reverie_reused_source_controls.json`](../manifests/reverie_reused_source_controls.json)
中的 1,423-episode 正式证据。三份 Source formal manifest 的原始字节固化在
`vln/results/runs/<run_id>/manifest.json`；runner 会校验文件 SHA256、完成状态、
模型/方法/seed、stream order 和 immutable identity。raw metric artifact 在
fresh clone 的 plan-only 阶段可缺失，但正式服务器启动时必须存在且通过 SHA256。

## 启动门禁

正式执行前必须先完成 R2R EAM/ATENA formal confirmation，并生成：

```text
vln/results/final/r2r/registry.json
schema=navtta.vln_r2r_final_registry.v1
registry_status=complete
```

runner 会逐 cell 比较 registry 的
`records.<r2r-setting>.<method>.parameters` 与本计划的冻结参数。缺 registry、状态
未完成、参数不一致都会 fail closed。每条 record 还必须包含最终 `run_tag`、
`SR/SPL`、formal manifest 路径及 SHA256；正式 job 的 transfer provenance 以该
registry record 覆盖计划阶段的临时 anchor。`--plan-only` 可以在 registry 生成前
使用。

## 反馈语义修正

REVERIE navigation success 不是 simulator 中的 `distance < 3`，而是正式提交终点
是否属于目标物体可见 viewpoint 集合。因此：

- DUET/GOAT 在 endpoint reranking 和最终 object prediction 完成后调用 evaluator；
- HAMT 在最终 navigation endpoint 和 object prediction 完成后调用 evaluator；
- FeedTTA 每个 episode eager 读取一次 submitted-trajectory navigation success；
- ATENA 只把同一 evaluator 暴露为 lazy callback，entropy gate 未 query 时不读取；
- 反馈字典只包含 `success`，绝不暴露 `rgs`/`rgspl`；
- REVERIE 上若未提供 submitted-trajectory callback，controller 会拒绝
  simulator-distance fallback。

最终结果同时报告 `SR/SPL/RGS/RGSPL`，但 FeedTTA/ATENA 的监督标签仅为 SR
对应的 navigation success。

## 冻结配置

参数来自 R2R 最终 registry；详细字段及原始 run tag 在机器可执行 spec 中。

| 方法 | DUET | HAMT | GOAT |
|---|---|---|---|
| Tent | `lr=1e-5,k=9,I=1` | `lr=3e-6,all-LN,I=1` | `lr=1e-5,all-LN,I=1` |
| FSTTA | `1.8e-3/3e-4,M8,N4` | `8e-4/2e-4,M1,N16` | `1.8e-3/1e-3,M3,N8` |
| EAM | `1e-5,I8,c=.5,64/8` | `1e-6,I8,c=.6,32/8` | `3e-6,I4,c=.3,32/8` |
| FeedTTA | `5e-6,.8,.05,+.1,last-crossmodal` | `2e-6,.9,.05,+.1,last-crossmodal` | `1e-6,.8,.1,-.1,action-head` |
| ATENA | `1.6e-6/2e-7,.5,0` | `3.2e-6/4e-7,.25,.1` | `3.75e-7/3.515625e-8,0,.15` |

Tent 的 update interval 全部固定为 1；FeedTTA 全部使用目标模型原生 argmax。

## 与 R2R-CE 并行时的 GPU 策略

REVERIE 和独立的 R2R-CE 搜索会同时推进，因此默认不假设独占 32-GiB GPU。
默认 profile 为 `shared_gpu_with_r2r_ce`：15 个 REVERIE job 全部进入队列，
但同一时刻只放行 1 个；每次启动前通过 `nvidia-smi` 重新检查空闲显存，按方法
分别保留 4/5/6/8/10 GiB 的最低空闲量。R2R-CE 调度端也必须保留该显存预算，
不能在 REVERIE job 启动后继续无界增并发。两个 scheduler 还会在共享 ledger 中为
active worker 持有显存/RAM reservation；即使 CUDA 冷启动尚未反映到
`nvidia-smi`，另一侧也按预留峰值判断，不依赖固定 sleep 猜测。

仍按方法顺序执行，并在方法之间设置严格 barrier：

```text
Tent (1) -> FSTTA (1) -> EAM (1) -> FeedTTA (1) -> ATENA (1)
```

这样 REVERIE 与 R2R-CE 在 campaign 级强制并行，而不是让任一 campaign 先跑完
再启动另一个。正式入口禁止 `exclusive_gpu` 和 CLI 超配；两个调度器共享同一个
per-GPU 启动锁，并在锁内重新测显存/cgroup RAM、启动 worker、等待 CUDA 显存
分配稳定。任何 job 失败或协议校验失败时，不会释放 REVERIE 的下一方法阶段。

## 命令

```bash
python3 vln/scripts/run_joint_reverie_r2r_ce.py --plan-only
python3 vln/scripts/run_joint_reverie_r2r_ce.py
python3 vln/scripts/run_joint_reverie_r2r_ce.py --status
```

两个 child 必须先完成数据/registry/batch preflight、取得 campaign-lifetime lock
并写 readiness ACK，联合入口收到双 ACK 后才放行正式 job。失败后仅在检查日志、
确认旧 scheduler/worker 均已退出后，通过联合入口增加
`--resume --retry-failed`；联合入口会核对原 joint identity，runner 会创建 `-retryN`
新 run tag，
不会覆盖失败证据。每个完成任务必须同时通过 1,423 episodes、正式 run manifest、
四项 REVERIE 指标和反馈端点/预算校验。

权威机器可执行配置为
[`reverie_r2r_frozen_transfer_v1.json`](reverie_r2r_frozen_transfer_v1.json)。
