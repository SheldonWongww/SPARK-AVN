# R2R-CE val_seen 最小补搜计划（Targeted Supplement V1）

## 目标与边界

本计划只修复首轮 R2R-CE 小搜索中尚未形成有效证据的 5 个模型–方法组合：

| 模型 | 方法 | 补搜原因 |
|---|---|---|
| ETPNav | Tent | `1e-7` 以下为 no-op，`1.5625e-5/all-LN` 已变差，中间 LR 空白 |
| ETPNav | FSTTA | 首轮 full 获胜点 `M=16`，超过单回合最多 15 个高层动作，实际 `updates=0` |
| BEVBert | FSTTA | 同上，首轮 full 是 zero-update no-op |
| ETPNav | FeedTTA† | 778 次更新但任务轨迹基本不变，原 LR 上界过低 |
| BEVBert | FeedTTA† | 逐 episode 输出与 Source 完全一致，原 LR 上界过低 |

不运行 Source，不补 EAM、ATENA 和 BEVBert–Tent。FeedTTA 使用 episode-level
binary feedback，以 `†` 标记，结果只在同一模型–方法 cell 内选参，不和无监督
方法混作同一监督预算排名。

对应机器可读方案为
`vln/experiments/r2r_ce_targeted_supplement_v1.json`，执行入口为
`vln/scripts/run_r2r_ce_targeted_supplement.py`。当前状态是
`review_required_not_launched`；只有提交后、tracked worktree 干净且显式传入
`--confirm-reviewed` 才允许正式启动。

## 搜索点与预算

全部 screening 使用 seed 0 canonical order 的前 100 个 episode；每个通过预定
晋级规则的 cell 最多确认 1 个 778-episode full finalist。因此固定执行 15 个
screening jobs，full 为 0–5 个，总量最多 20 个。

| 模型–方法 | 固定项 | 3 个 screening 点 |
|---|---|---|
| ETPNav–Tent | `last_k_ln, k=4, I=1, AdamW, wd=0` | `lr={3e-7,1e-6,3e-6}` |
| ETPNav/BEVBert–FSTTA | 最后 4 个 LN、论文 fast/slow 机制、AdamW | `(lf,ls,M,N)={(3e-7,1e-5,4,16),(1e-6,3e-5,4,16),(1e-6,3e-5,1,16)}` |
| ETPNav/BEVBert–FeedTTA† | `last_crossmodal, gamma=.99, p=.05, alpha=-.2, argmax` | `lr={3e-7,1e-6,2e-6}` |

这些点逐字落实分析报告的“必做 15 screening + 最多 5 full”，没有加入可选的
ATENA 边界点，也没有扩大成新的笛卡尔积。

## 晋级规则

所有方法首先满足 `SR >= matched Source SR - 1.5pp`，以 SPL 为主指标、SR 为
次指标。

- FSTTA 在 spec 加载和结果晋级两处都 fail-closed：`M>15` 直接拒绝；
  `adapter.updates<=0` 直接拒绝。zero-update 候选不可能再因低漂移晋级。
- FeedTTA 若候选在 SPL/SR 上并列，先剔除对 canonical prefix 的逐 episode
  navigation record 完全没有变化的候选，再从非零变化候选中选最高安全 LR。
  若所有最高分候选仍是 no-op，则该 cell 不执行 full，而不是再次选择最低漂移
  的 no-op。
- R2R-CE screening 不保存原始几何轨迹，因此“轨迹变化”固定定义为 10 个
  evaluator navigation fields（steps、终点距离、success、path length、SPL、
  nDTW 等）的逐 episode 完整记录是否相对认证 Source 改变。runner 验证 100 个
  canonical episode 覆盖并固定这些字段，不能用参数漂移替代行为变化。
- Tent/FSTTA 在满足上述硬门槛后按 SPL、SR、较低参数漂移、较少更新依次排序。

晋级记录写入每个 screening phase 的 immutable `PROMOTION.json`。full job 从该
记录重建，保留 parent screening run tag。

## 顺序、隔离与并发

执行采用严格模型屏障，不允许两个导航模型重叠：

1. ETPNav screening：9 jobs，Tent/FSTTA/FeedTTA 按候选编号轮转排队，并发 3。
2. ETPNav full confirmation：0–3 jobs，并发最多 3。
3. BEVBert screening：6 jobs，FSTTA/FeedTTA 轮转排队，并发 3。
4. BEVBert full confirmation：0–2 jobs，并发最多 2。

每个 job 都是独立的 `run_source_eval.sh` 进程，并从相应模型的 immutable Source
checkpoint 重新加载；full 绝不延续 screening 的模型或优化器状态。runner 同时
写入 `restart_from_source_checkpoint=true`，full 解析时还会核对正式 manifest 的
checkpoint SHA256。

prefix 和 full 均使用 tracked canonical manifest、默认 model seed 0。prefix
协议禁止 `--order-seed`，所以命令不传该参数；这不是未固定顺序，而是
`run_source_eval.sh` 的 canonical seed-0 路径。job/phase/campaign manifest 都记录
order SHA256 `93f44aab…4bd94e`。

并发上限采用已在该 R2R-CE 服务器阶段验证过的 3-worker 保守上限；预启动门槛为
22,000 MiB，单 job 预留 8,000 MiB，aggregate 上限 30,000 MiB。cgroup memory
对应为 `55 + 20 <= 75 GiB`。本轮允许按总计划与 REVERIE frozen evaluation
并行：两个调度器使用同一 per-GPU launch lock 和 active reservation ledger。
R2R-CE 在锁内读取实际显存/cgroup 与对端尚未显现为实际占用的预约量，原子完成
资源判断、reservation 和 worker 启动，并继续持锁 15 秒等待 CUDA 分配可见。
每个 job 的 worker、runner 及 Python 后代继承唯一 token，恢复时通过 token 与
PGID/SID 重新发现进程并认领 reservation；即使两层 shell 异常退出，只要评测
后代仍存活就不会释放或 retry。任务完全终止后才释放，避免两个 campaign 同时
基于旧快照启动。`exitcode` 仅表示 worker 已写出状态；在 token 后代归零前不视为
phase terminal，因此模型 barrier 也不会提前放行。

## 正式证据与恢复

- Source 指标只从 `vln/manifests/r2r_ce_reused_source_controls.json` 引用；缺失或
  SHA256 不匹配时在规划前失败，Source 执行数始终为 0。
- 100-episode screening 是非正式筛选证据，但 job/config、canonical prefix、
  diagnostics 和逐 episode navigation record 均被持久化并校验。
- 每个 778-episode finalist 必须生成正式 run manifest；runner 校验 Git commit、
  config、Source checkpoint digest、dataset/order digest、硬件结果 artifacts 和
  immutable identity。
- campaign/phase/promotion/confirmation 文件均按当前 Git commit 和 spec SHA256
  固定。恢复必须使用 `--resume`；失败重试还需 `--retry-failed`，并会保留前一次
  job、日志、结果和 formal manifest。

仅生成计划（不会运行 GPU job）的命令示例：

```bash
python3 vln/scripts/run_r2r_ce_targeted_supplement.py \
  --batch-id vln-r2r-ce-targeted-supplement-v1-seed0 \
  --plan-only --print-commands
```

正式启动应在方案提交并再次人工确认后执行；本次实现任务不启动实验。
