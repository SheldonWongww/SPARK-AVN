# R2R-CE 小规模两阶段 TTA 超参数搜索计划

状态：已实现，尚未启动。实验只覆盖当前具有正式 TTA runner 的 ETPNav 和
BEVBert；StreamVLN runner 尚未就绪，因此本轮明确阻断，不生成占位任务或伪造
结果。

## 目标与预算

R2R-CE 是连续环境，一个 episode 包含的高层导航决策明显多于 R2R。旧搜索把
R2R/论文学习率直接迁移过来，ETPNav Tent 出现持续退化，FeedTTA 还混入了
sampled action control，无法回答 native policy 下低强度更新是否有效。本轮结合
R2R 的优胜区域和 AVN 的长流稳定性经验，主要下调学习率，并只为更新频率、参数
scope 等强影响因素保留极少量联合点。

本轮只执行 TTA，不重复执行 Source：

| 阶段 | episode | TTA 数量 | Source 执行数 | 用途 |
|---|---:|---:|---:|---|
| Screening | canonical prefix 100 | 40 | 0 | 每个模型×方法 4 点 |
| Full confirmation | 完整 val_seen 778 | 10 | 0 | 每个模型×方法只晋级 1 点 |
| 合计 | — | 50 | 0 | 不是旧方案的 68 个 full-val jobs |

Source 来自已完成的 `grouped-source-20260810T080743Z` ETPNav/BEVBert formal
manifests。两份证据都有完整 778 条逐 episode 指标。调度器按 tracked canonical
episode order 离线重算前 100 条控制指标，并用全部 778 条重算 full control。
两份 formal manifests 已按原字节固化到 `vln/results/runs/<run_id>/manifest.json`；
较大的逐 episode/aggregate artifacts 继续留在 ignored Source 目录并按 SHA256
验证。
正式 manifest、checkpoint、dataset、order、artifact size/SHA256、778 个 episode
ID 和 full aggregate 任一不一致都会在创建计划前 fail closed；不会退化成重新跑
Source。

## 搜索空间

每个模型使用相同的低强度主区域，并保留一个模型特定的历史/官方锚点用于校准。
锚点不强制晋级，因为 full-val 每个 cell 只有一个名额。

- Tent：`update_interval=1` 固定；主搜 `1e-8, 3e-8, 1e-7`，比较最后 4 个
  LayerNorm 与最后 1 个 LayerNorm；保留历史 `1.5625e-5/all-LN` 点。
- FSTTA：最后 4 个 LayerNorm 及论文机制固定；低强度三点成对取
  `(fast,slow)=(3e-7,1e-5),(1e-6,3e-5),(3e-6,1e-4)`，保持约 `0.03` 的
  fast/slow 比率，不做容易产生 `fast << slow` 极端点的无约束笛卡尔积。FAST
  每 `M` 个高层动作更新、SLOW 每 `N` 个 episode 更新，因此二者绝对 LR 不能
  直接要求相等；较大的 `M/N` 继续控制长轨迹累计更新。第四点保留各模型旧 CE
  最优附近配置。该范围以 AVN 的 `fast=3e-7, slow=1e-4` 正向点为下界依据，
  同时比旧 CE fast LR 再降低约 50–500 倍。
- EAM：AVN 的稳定区 `lr=3e-9,1e-8,3e-8` 与 `interval=32,16,8`；保留各模型
  `1e-6/interval=1` 历史点。replay size/batch 不做大网格。
- FeedTTA：恢复目标模型 native argmax，不再使用采样；主搜
  `lr=1e-8,3e-8,1e-7`，固定 `gamma=.99,p=.05,alpha=-.2`，并在中心学习率比较
  `last_crossmodal` 与 `action_head`。连续模型现已支持显式 model-aware scope。
- ATENA：只搜成对学习率 `(3e-8,1e-8)`、`(1e-7,1e-8)`、
  `(3e-7,3e-8)`，另保留模型历史锚点；ETPNav 固定
  `lambda=.5,threshold=.3`，BEVBert 固定 `lambda=.25,threshold=.1`。

FeedTTA 和 ATENA 消耗 binary episode navigation feedback，必须在报告中标记为
feedback-supervised；Tent、FSTTA、EAM 属于无监督 TTA。

## 晋级与拒绝规则

每个模型×方法独立排序。Screening 先要求 `SR >= matched Source SR - 1.5pp`，
再最大化 SPL，依次用 SR、较低参数漂移和较少更新次数打破平局。40 个 screening
job 必须全部成功且诊断完整，才允许生成 10 个 full jobs。Full 阶段重复相同
Source floor；不合格的唯一 finalist 不会被写成 winner。

本轮只使用 `val_seen` 做开发选择，不读取 `val_unseen` 或 test。固定 seed 0 与
canonical episode order；后续若需要 order robustness，必须先冻结 full winner，
不能在本轮预算内边搜边换顺序。Spec 显式设置
`order_robustness_enabled=false, final_order_seeds=[0]`，因此 `--with-orders` 会被
runner 拒绝，不会在 50-job 预算后自动追加 seed 1/2。

## 与 REVERIE 共卡运行

R2R-CE 与 REVERIE frozen-transfer campaign 必须由联合入口同时 detached 启动并
持续独立推进，不能把一个 campaign 的完成作为另一个的启动条件。两个调度器都
固定为一名 worker：`CE cap=1 + REVERIE cap=1`。CE 默认启动门限为：

- 启动前 GPU 已用不超过 16,000 MiB；按 CE job 8,000 MiB 估算，projected
  aggregate 不超过 26,000 MiB（服务器为 32 GiB vGPU）；
- 启动前 cgroup memory 不超过 55 GiB；按 CE job 20 GiB 估算，projected
  aggregate 不超过 75 GiB；
- 任一门限不满足时等待，不抢占或杀死 REVERIE；等待 3600 秒仍无法安全启动则
  fail closed。

这些是基于历史 CE smoke 峰值（约 3.9–6.7 GiB GPU）的保守初值。本联合批次禁止
通过 CLI 放宽门限或把任一并发提高到 2；如后续有新实测依据，必须修改并评审
tracked spec 后启动新批次。

## 文件与运行入口

- 搜索 spec：`vln/experiments/r2r_ce_small_hparam_search_v1.json`
- Source 复用清单：`vln/manifests/r2r_ce_reused_source_controls.json`
- 调度器：`vln/scripts/run_tta_hparam_search.py`
- 强制联合入口：`vln/scripts/run_joint_reverie_r2r_ce.py`

服务器同步到固定 commit 并确认两份 ignored Source artifacts 仍在原路径后，使用：

```bash
python3 vln/scripts/run_joint_reverie_r2r_ce.py --plan-only
python3 vln/scripts/run_joint_reverie_r2r_ce.py
python3 vln/scripts/run_joint_reverie_r2r_ce.py --status
```

联合入口内部固定以 `method=all` 排队；每个方法先完成 2 模型×4 点 screening，
再运行该方法的 2 个 full winner。两者分属独立批次、没有跨 campaign 阶段依赖，
但使用同一 launch authorization、per-GPU 启动锁、active reservation ledger 和
projected resource gate。联合入口把 CE spec 作为显式 `--spec` 参数固化在 argv 和
launch manifest 中；两个 child 完成自身 preflight、取得 campaign-lifetime lock 并
写 readiness ACK 后才会被同时放行。`--resume` 会校验原 joint identity，并拒绝仍
有旧 scheduler/worker 存活的批次。
