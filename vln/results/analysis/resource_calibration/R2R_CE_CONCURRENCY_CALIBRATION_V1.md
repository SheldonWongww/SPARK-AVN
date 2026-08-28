# R2R-CE consistency-v2 并发校准记录

状态：`in_progress`。本文只记录资源校准，不改变 TTA 科学协议，也不把短前缀结果当作导航指标。

## 范围

- GPU：单张 NVIDIA vGPU-32GB，`32760 MiB`。
- cgroup 内存上限：`90 GiB`，无 swap。
- 模型：ETPNav、BEVBert。
- 方法：Tent、FSTTA、EAM、FeedTTA、ATENA。IDEA 暂不进入本轮。
- StreamVLN 仍被 consistency-v2 runner 明确阻塞，当前没有可校准的 TTA 搜索入口。

## 验收规则

1. 隔离单任务必须覆盖真实 TTA 更新；正式分组验证使用每 worker 100 episode 前缀。
2. 每秒采样进程级显存、整卡显存/GPU 利用率、job 进程树 RSS 和 cgroup memory。
3. 生产线为整卡显存 `< 29000 MiB`、cgroup memory `< 80 GiB`；`30000 MiB` 为紧急停止线。
4. 单任务投影使用 `1.05 × 最大历史峰值 × worker 数`。投影只产生待验证并发，不直接成为生产上限。
5. 待验证并发必须作为一个完整同时驻留组运行；所有 worker 完成加载、出现方法特定更新、持续稳定且零 OOM/零失败后才能批准。
6. 若目标并发失败，逐一降低 worker；不允许通过启动错峰把瞬时低活跃数解释为安全并发。

## 已有隔离单任务证据

下表取服务器历史 `RESOURCE_PROFILE.json` 中同一 model-method 的最大观测值。历史采样较稀疏，因此只用于确定分组压力测试目标。

| 模型 | 方法 | 最大单任务峰值 MiB | 5% 余量下的投影并发 | 投影总显存 MiB | 状态 |
|---|---|---:|---:|---:|---|
| ETPNav | Tent | 3860 | 7 | 28371 | 待分组验证 |
| ETPNav | FSTTA | 3858 | 7 | 28356 | 待分组验证 |
| ETPNav | EAM | 4818 | 5 | 25294 | 待分组验证 |
| ETPNav | FeedTTA | 4726 | 5 | 24812 | 待分组验证 |
| ETPNav | ATENA | 5902 | 4 | 24788 | 待分组验证 |
| BEVBert | Tent | 4346 | 6 | 27380 | 待分组验证 |
| BEVBert | FSTTA | 4322 | 6 | 27229 | 待分组验证 |
| BEVBert | EAM | 6426 | 4 | 26989 | 待分组验证 |
| BEVBert | FeedTTA | 6314 | 4 | 26519 | 待分组验证 |
| BEVBert | ATENA | 7302 | 3 | 23001 | 待分组验证 |

计算式为 `floor(29000 / (single_peak_mib × 1.05))`。这些值不是批准上限。

## 当前全流监控

监控对象：`consistency-v2-five-r2rce-recovery-20260828T080836Z`。正式 campaign 保持原 spec 的 3-worker 上限，旁路监控不修改进程、结果或仓库。

- 监控 screen：`navtta-r2rce-resource-calib-0828`
- 远端汇总：`/root/autodl-tmp/tmp/navtta-r2rce-resource-calibration-20260828/summary-v3.json`
- 监控工具：`vln/scripts/monitor_r2r_ce_resources.py`
- 第一阶段 ETPNav–Tent 的早期观测：3 个 job 的进程显存合计峰值 `9722 MiB`，整卡峰值 `12284 MiB`，cgroup 峰值约 `24.11 GiB`；单 Python 进程峰值 `3242 MiB`。该阶段尚未结束。

## 冻结条件

只有十个组合均获得完整 1 秒全流证据、随后各自通过投影并发的 100-episode 分组压力测试，才会：

1. 将投影值改为 `approved` 或按失败结果下调；
2. 创建 consistency 搜索 spec 的后继版本；
3. 同步更新 runner 的注册上限与测试；
4. 保留当前 v2 spec 和已运行证据，不原地改写。
