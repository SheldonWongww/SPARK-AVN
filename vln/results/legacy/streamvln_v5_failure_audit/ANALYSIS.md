# StreamVLN v5 故障与冻结参数审计

用户快照 `中间结果/streamvln_val_unseen_search_v5` 的 16 个搜索运行中，4 个 FSTTA 完成全部 1839 个 episode；EAM、FeedTTA 和 ATENA 均没有可选优的完整结果。逐条核对后，所有已有 episode 都符合统一 seed 0 顺序的完整流或前缀。

本目录只记录紧凑审计，不复制原始日志、图像或逐 episode 结果。详细配置、输入文件 SHA256、诊断、故障位置和冻结参数见 [summary.json](summary.json)。

快照的 campaign 声明 Git commit 为 `3da09b96b4dccc266ab15db9519fb8a9d3654cf9`。四份 FSTTA completion.json 引用了 manifest SHA256，但快照及本地 StreamVLN 结果中均未找到对应 manifest.json 原档，因此全部旧指标仍为 legacy，不能移入正式结果表。

| v5 配置 | 已完成 / 1839 | 状态 | 完整 SR (%) | 完整 SPL (%) |
|---|---:|---|---:|---:|
| atena-01 | 701 | 历史特征缺失异常 | — | — |
| atena-02 | 1420 | 快照时未完成，未见异常 | — | — |
| atena-03 | 1423 | 快照时未完成，未见异常 | — | — |
| atena-04 | 1420 | 快照时未完成，未见异常 | — | — |
| eam-01 | 1033 | SIGHUP 中断 | — | — |
| eam-02 | 1016 | SIGHUP 中断 | — | — |
| eam-03 | 1022 | SIGHUP 中断 | — | — |
| eam-04 | 1018 | SIGHUP 中断 | — | — |
| feedtta-01 | 0 | CLI 参数拒绝 | — | — |
| feedtta-02 | 0 | CLI 参数拒绝 | — | — |
| feedtta-03 | 0 | CLI 参数拒绝 | — | — |
| feedtta-04 | 0 | CLI 参数拒绝 | — | — |
| fstta-01 | 1839 | 完整，legacy | 56.49810 | 50.35649 |
| fstta-02 | 1839 | 完整，legacy | 57.85753 | 51.37406 |
| fstta-03 | 1839 | 完整，legacy | 57.04187 | 50.84974 |
| fstta-04 | 1839 | 完整，legacy | 56.93312 | 50.80705 |

## 已定位问题

- **FeedTTA 4/4 在启动前失败**：搜索配置传入 `scope_profile=configured_prefixes`，但 `streamvln_eval.py` 的 argparse 只接受 `paper_full / last_crossmodal / action_head`。没有执行任何 episode。
- **EAM 4/4 被外部 SIGHUP 中断**：约在 2026-09-09 06:44，torchrun 收到 signal 1。已完成的 episode 数分别为 1033、1016、1022、1018；日志没有显示 EAM 算法异常，也不能从日志确定信号发送者。
- **ATENA-01 在 `TbHJrupSAjP_171` 的 step 194 崩溃**：此前 701 个 episode 已完成。动作块跨越 step 192 的 context reset，残余动作执行到 194 后才再次生成。新 prompt 含 `<memory>`，历史图像拼接却仍要求当前 step 能被 32 整除，导致只传当前图像；`stream_video_vln.py:224` 访问 `memory_features[batch_idx][cur_mem_id]` 时遇到 None。问题在共用推理路径，不能仅给 ATENA 特判。
- **ATENA-02/03/04 的快照不是完成结果**：分别有 1420、1423、1420 个 episode，末尾仍处于推理中，未出现 traceback。只能标记未完成，不能推断崩溃原因。

## Tent 与 FSTTA 的冻结依据

Source/Tent 沿用已跟踪的 [compact-v4 审计](../streamvln_compact_v4_audit/summary.json) 作为历史参数依据，不将旧记录补写为正式结果。Tent01 和 Tent02 的完整 SPL/SR 持平，按低学习率打破平局，选择 **Tent01：Adam，LR 3e-8，last_ln，interval 1**。执行协议保留 `legacy_v4`；若更换为 `native_residual`，旧 val_unseen 指标不能作为该新路径的结果。

FSTTA 比较了所有可获得结果：v4 三组和 v5 四组。v1-v3 仅有配置，没有可用于排名的结果。下表按 SPL、SR 降序排列；不同 readout 协议的差异不归因于单一超参数。

| FSTTA 运行 | 协议 | fast / slow LR | M / N | SR (%) | SPL (%) |
|---|---|---|---|---:|---:|
| streamvln-vu-fstta-02-v5 | native_residual | 3e-07 / 9e-07 | 3 / 16 | 57.85753 | 51.37406 |
| streamvln-vu-fstta-03-v5 | native_residual | 3e-06 / 9e-06 | 3 / 4 | 57.04187 | 50.84974 |
| streamvln-vu-fstta-04-v5 | native_residual | 3e-06 / 9e-06 | 3 / 16 | 56.93312 | 50.80705 |
| streamvln-vu-compact-fstta-01-v4 | legacy_v4 | 3e-08 / 1e-07 | 3 / 4 | 56.87874 | 50.50408 |
| streamvln-vu-compact-fstta-02-v4 | legacy_v4 | 1e-07 / 3e-07 | 3 / 4 | 56.87874 | 50.50408 |
| streamvln-vu-compact-fstta-03-v4 | legacy_v4 | 3e-07 / 1e-06 | 4 / 16 | 56.82436 | 50.44990 |
| streamvln-vu-fstta-01-v5 | native_residual | 3e-07 / 9e-07 | 3 / 4 | 56.49810 | 50.35649 |

**冻结 FSTTA v5-02**：fast LR 3e-7、slow LR 9e-7、M=3、N=16、Q=0.1、last_ln、AdamW、native_residual；其余完整参数与三个工件摘要均保存在 `frozen_selections.fstta`。该点的 SPL=51.37406、SR=57.85753，在七个完整历史结果中均最高，无需重跑 FSTTA 搜索。

重跑时仅搜索 EAM、FeedTTA、ATENA，并从首个 episode 开始。中断前缀不含可恢复的完整 TTA 状态，不能靠跳过已有 episode 来继续适应流。val_seen 的 Source 和五种 TTA 使用同一 split 的统一 seed 0 顺序；TTA 参数只由 val_unseen 选择，val_seen 不参与搜索。新运行必须生成完整 provenance，旧快照继续保留为 legacy。
