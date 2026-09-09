# StreamVLN v6：修复后搜索与固定参数评测

运行规格为 [streamvln_val_unseen_search_v6.json](streamvln_val_unseen_search_v6.json)。新结果使用独立 `v6` 目录：只在 `val_unseen` 搜索 EAM、FeedTTA、ATENA，各 4 组、共 12 次完整评测；Tent 和 FSTTA 不再搜索。随后在 `val_seen` 运行 Source、Tent、FSTTA、EAM、FeedTTA、ATENA，五种 TTA 均固定各自 `val_unseen` 选出的参数。

## 日志诊断和修复

对用户提供的 `中间结果/streamvln_val_unseen_search_v5` 的审计见 [ANALYSIS.md](../results/legacy/streamvln_v5_failure_audit/ANALYSIS.md) 和 [summary.json](../results/legacy/streamvln_v5_failure_audit/summary.json)。

| 方法 | v5 实际情况 | 处理 |
|---|---|---|
| FeedTTA | 4 组均为 0 episode；`configured_prefixes` 被共享 argparse 的离散模型 scope choices 拒绝 | StreamVLN parser 明确接受自己的 `configured_prefixes`；回归测试将所有新搜索配置实际翻译并解析 |
| EAM | 4 组仅完成 1,033 / 1,016 / 1,022 / 1,018 条；torchrun 在同一时刻收到 `SIGHUP` | 属外部中断证据，没有算法异常栈；新调度器为每个作业创建独立 session，并用 `nohup` 或 tmux 启动调度器 |
| ATENA | 01 完成 701 条后在 `TbHJrupSAjP_171` 崩溃；02/03/04 只有 1,420 / 1,423 / 1,420 条，提供的日志没有其失败证据 | 修复长 action chunk 跨缓存重置后首次生成缺少历史帧；全部重新运行完整流 |
| FSTTA | 4 组均完成 1,839 条，episode 顺序一致，有 completion；提供的目录缺少原始 run manifest | 比较全部可用 v4/v5 完整结果后冻结 v5-02 参数，不再搜索；旧指标仍为 legacy |

ATENA 的具体触发过程是：第 192 步清空缓存后，已生成的 action chunk 仍有行动；直到第 194 步才再次 `generate()`。旧逻辑给新 prompt 加入 `<memory>`，但只在 `step % 32 == 0` 时追加历史图像，因此模型收到 memory placeholder 与 `memory_features=None`。现在 prompt 与历史图像都依照“新的生成 context”判断，并使用该段真正的起点 `time_ids[0]` 选取历史帧。正常第 32、64 等对齐边界的采样保持原样。修复适用于共同 evaluator，不将未发现失败的 Source 描述成失败实验。

## 参数、顺序和预算

全部使用 seed 0、单环境、greedy/argmax；每条候选、每个 split 都从同一初始 checkpoint 独立启动，模型、优化器和 replay 不跨作业继承。episode 顺序由已有 `vln/manifests/episode_order/r2r_vlnce_v1_3/{val_unseen,val_seen}.json` 固定。评测逐条核验实际 `(scene_id, episode_id)`，调度器再次核验结果的完整顺序、唯一性和最终聚合值；Source 使用同样的 seed 和 manifest。

| 方法 | val_unseen 配置来源 |
|---|---|
| Tent | 冻结 v4-01：LR `3e-8`、Adam、last LN、interval=1、`legacy_v4` readout；与 v4-02 完整结果持平时取较低 LR |
| FSTTA | 冻结 v5-02：fast LR `3e-7`、slow LR `9e-7`、M=3、N=16、`native_residual` readout；为全部 7 个可获得完整 v4/v5 配置中 SR/SPL 最佳 |
| EAM | LR `1e-6 / 3e-6` × confidence scale `0.4 / 0.8`；memory=32、batch=8、interval=1 |
| FeedTTA | LR `1e-6 / 3e-6` × gamma `0.7 / 0.95`；p=0.05、alpha=-0.2、SGR seed=0、Adam eps=1e-5 |
| ATENA | query LR `3e-7 / 3e-6` × query threshold `0.3 / 0.5`；self LR 为 query LR 的 0.1 倍，lambda=0.25、self loss weight=0.1 |

三种搜索方法均为 `native_residual` readout，继续使用 v5 已定的范围。Tent 保留其历史最佳配置的 `legacy_v4` 数值路径；这项差异保留在配置和报告中。完整配置（含所有 reset/optimizer 参数）以 JSON 为准。上述适应范围是第一个生成行动 token 的最后 RMSNorm FP32 副本，不能解释为更新整个 7B 模型。FeedTTA/ATENA 使用二元 episode 反馈，与无监督方法区分报告。

1. `smoke`：Source 和五种 TTA 各跑 `val_seen` seed 0 的前 2 条，共 6 个小作业（沿用公共 launcher 的 smoke 协议）。仅确认加载、parser、适应入口、输出和 manifest，不用其成绩选参；缓存边界问题另有无需模型的回归测试。
2. `search`：三方法各 4 组，每组完整 1,839 条。必须全部 12 组完成且有效，才按 **SPL → SR → 候选编号** 冻结 winner。部分结果、非有限数值、旧 v5 文件不会参与排名；有效但不涨点的结果保留。
3. `seen`：Source 加五种 TTA，各完整 778 条。Tent/FSTTA 来自历史选参审计，另外三种来自新 `selection.json`。`val_seen` 不能反向选参。
4. `all`：依次执行 smoke → search → seen。单 GPU 同时最多一个 StreamVLN 实例，4 卡最多 4 个评测进程。AVN 四卡任务和本任务应分别占用这 4 张卡。

历史 Source/Tent/FSTTA 的指标保留在 `results/legacy/`，只作为历史选参证据。新脚本不要求复制未跟踪的历史原始日志；通过 tracked 审计摘要 SHA256 绑定冻结参数。新正式结果必须具有 checkpoint/data/episode-order 摘要、seed、硬件、顶层 Git commit 和结果 artifacts 完整的 run manifest。

## 服务器运行

GitHub 的 `SPARK-AVN/main` 与当前 NavTTA 历史独立，本次完整 NavTTA 代码位于 `navtta-eval-fixes-20260909` 分支。已有 NavTTA 工作目录第一次切换：

```bash
cd /data1/wxy/code/NavTTA
git fetch https://github.com/SheldonWongww/SPARK-AVN.git navtta-eval-fixes-20260909
git switch -c navtta-eval-fixes-20260909 FETCH_HEAD
STREAM_PY=/data1/wxy/exp_data/NavTTA/vln/envs/streamvln/bin/python
"$STREAM_PY" -m pip install -e core
"$STREAM_PY" vln/scripts/run_streamvln_search_v6.py --stage all --gpus 0,1,2,3 --dry-run
```

已在该分支时，更新使用 `git pull --ff-only https://github.com/SheldonWongww/SPARK-AVN.git navtta-eval-fixes-20260909`。StreamVLN 使用上面的独立环境，不使用 AVN 的 `enmus` 环境。确认旧任务已经退出并释放 GPU 后启动：

```bash
mkdir -p vln/results/tuning/streamvln_val_unseen_search_v6
nohup "$STREAM_PY" -u vln/scripts/run_streamvln_search_v6.py \
  --stage all --gpus 0,1,2,3 \
  > vln/results/tuning/streamvln_val_unseen_search_v6/scheduler.log 2>&1 &
```

也可以在 tmux 中执行同一个 Python 命令。不要直接把依赖 SSH 终端的前台长任务作为后台作业遗留；v5 EAM 已观察到 SIGHUP 中断。新 evaluator session 隔离终端信号，调度器收到 SIGINT/SIGTERM 时只停止自己创建的评测进程组。

如 v6 中断，仍使用同一分支和同一代码版本，将未完成作业归档并从 episode 1 重启；已经校验成功的作业会自动跳过：

```bash
nohup "$STREAM_PY" -u vln/scripts/run_streamvln_search_v6.py \
  --stage all --gpus 0,1,2,3 --retry-incomplete \
  >> vln/results/tuning/streamvln_val_unseen_search_v6/scheduler.log 2>&1 &
```

不要用 `v5` 的 `--retry-incomplete` 执行本轮计划，也不要把旧目录重命名成 v6。continual TTA 不保存中间优化器/replay 状态，不能只跳过已经完成的 episode 继续后缀。修改运行代码后需要新的 campaign 版本，脚本会拒绝在旧 campaign 中混合 Git commit。

可单独执行 `--stage smoke`、`--stage search`、`--stage seen`；`seen` 会先验证全部搜索结果和选参记录。进度和汇总：

```bash
tail -f vln/results/tuning/streamvln_val_unseen_search_v6/scheduler.log
"$STREAM_PY" vln/scripts/run_streamvln_search_v6.py --stage report
```

目录分层：

```text
vln/results/tuning/streamvln_val_unseen_search_v6/
  scheduler.log                    总调度日志
  campaign.json                    固定代码版本和 spec 摘要
  configs/<run-tag>.json            每条作业的冻结参数、seed、order 摘要
  launcher_logs/<run-tag>.log       启动和运行日志
  jobs/<run-tag>/<split>/           TTA 逐 episode 结果、诊断、completion
  incomplete/<run-tag>-<time>/      中断重启前保存的原始证据
  historical_selection.json        Tent/FSTTA 历史选参绑定
  selection.json                   三种新搜索方法的 winner
  report.json / report.md           完整度、参数、指标及诊断汇总
vln/results/source/<source-run-tag>/streamvln-r2r-ce/<split>/
vln/results/smoke/<smoke-run-tag>/streamvln-r2r-ce/val_seen/
vln/results/runs/<run-id>/manifest.json
```

原始日志、图片、模型文件和数据不进入 Git。
