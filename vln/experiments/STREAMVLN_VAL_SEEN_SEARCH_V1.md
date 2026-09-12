# StreamVLN：四方法 val_seen 三点搜索

运行规格：[streamvln_val_seen_search_v1.json](streamvln_val_seen_search_v1.json)。
入口：`vln/scripts/run_streamvln_val_seen_search.py`。
默认只运行 FSTTA、EAM、FeedTTA、ATENA 各 3 个完整 `val_seen` 候选，合计
12 × 778 = 9,336 episodes。四个方法各独占一张 GPU，同方法候选按 01 → 02 → 03
串行执行。每个候选都启动新进程，从原始 checkpoint 初始化模型、优化器和 replay。

## 已有证据与设计理由

用户提供的 v6 归档审计位于
[ANALYSIS.md](../results/legacy/streamvln_v6_seen_search_audit/ANALYSIS.md)，原文件
SHA256、候选配置和紧凑诊断保存在同目录 `summary.json`，由本轮 spec 绑定摘要。

- v6 Source seen 终值为 SR/SPL `68.5090/62.5255`，只有 launcher aggregate。
  FSTTA 完整 778 条为 `67.7378/61.8658`，EAM 为 `67.3522/61.6537`。
- FeedTTA seen 只有 622 条，ATENA 只有 625 条；归档没有 terminal aggregate。
  因此不能把这两份前缀当作完整 seen 搜索失败，也不能据此前缀选参。
- v6 unseen 的 FeedTTA 在相同 LR 下 gamma=.7 均优于 .95；ATENA 的
  `3e-7/.3` 优于 `3e-6` 两组。EAM 的 c=.4 两组执行动作与原生策略相同，
  c=.8 才产生实际动作变化。FSTTA 已有数千次 FAST 和 48 次 SLOW 更新，
  因此不是未进入适应路径。
- R2R 论文对齐补搜支持 FSTTA 的 M=1/较稀疏 SLOW；FeedTTA 恢复 argmax
  后，`2e-6~5e-6`、gamma=.8/.9、alpha=+.1 在 DUET/HAMT 得到改善。
  R2R-CE 结果支持 EAM 的 LR=1e-6、memory=32/64，以及 ATENA 分别控制
  query/self LR。具体来源在审计中固定，不将不同模型的指标横向排名。
- AVN 的 FSTTA 长慢窗口和 EAM 低累计更新量支持保守分支；SMT+Audio 与
  ENMuS 的有效 LR 不同。SMT ATENA single-source 的 self-loss probe 有正向
  结果，但本地 multi-source 的 20 个候选都低于 matched Source SR。
  因此只转移机制经验，不照搬 AVN 的 sampled action、更新范围或绝对 LR。

这轮已参考 seen 和 unseen 历史结果，属于 **seen 超参数开发**。
新选中的 seen 成绩不是独立确认结果；未来复用已调过的 unseen 也不能称为
从未参与开发的 held-out 测试。新脚本不会自动再次跑 unseen。

## 十二组参数

表中顺序即实际执行顺序。全部 seed=0、greedy/argmax、continual、
`native_residual`、weight decay=0、max grad norm=1；只适应首个行动 token
的最后 RMSNorm FP32 副本，保持 v6 模型实现与缓存修复。

| 方法 / 默认 GPU | 01 | 02 | 03 |
|---|---|---|---|
| FSTTA / 0 | fast=`1e-7`, slow=`3e-7`, M/N=`3/32` | fast=`3e-7`, slow=`9e-7`, M/N=`3/32` | fast=`3e-7`, slow=`3e-6`, M/N=`1/16` |
| EAM / 1 | LR=`3e-7`, c=`.8`, memory=`32` | LR=`1e-6`, c=`.8`, memory=`32` | LR=`1e-6`, c=`.6`, memory=`64` |
| FeedTTA† / 2 | LR=`3e-7`, gamma=`.7`, alpha=`-.2` | LR=`1e-6`, gamma=`.7`, alpha=`-.2` | LR=`2e-6`, gamma=`.8`, alpha=`+.1` |
| ATENA† / 3 | query/self LR=`3e-7/3e-8`, threshold=`.15`, lambda=`.25`, self weight=`.1` | query/self LR=`5e-7/1e-8`, threshold=`.3`, lambda=`.5`, self weight=`.1` | query/self LR=`1e-7/1e-8`, threshold=`.5`, lambda=`.5`, self weight=`.25` |

FSTTA 保留 concordant FAST、FAST/SLOW 双分支、stream 生命周期的方差历史；
FAST 优化器逐 episode 重置，SLOW 优化器跨窗口持续。其余 q/rho/tau/a/b 沿用 v6。
EAM 固定 batch=8、interval=1、Adam；c=.6 相比 .8 更严格筛选低熵预测。
FeedTTA 固定 p=.05、SGR seed=0、`paper_main`、Adam eps=1e-5；正 alpha
分支明确是历史 VLN 中采用的梯度缩放变体，不能把它描述成负 alpha 的梯度反转。
ATENA 使用 AdamW、可被 replay 到达的 navigation scope。
完整 optimizer/reset/scope 参数以 JSON 为准。

† FeedTTA 每个 episode 使用一次二值成功反馈；ATENA 按查询路由获取二值反馈，
其余 episode 用 self-prediction。FSTTA/EAM 不读取二值反馈，报告保留此区别。

## 服务器更新与运行

代码推送到 GitHub `SheldonWongww/NavTTA` 和已有服务器使用的
`SheldonWongww/SPARK-AVN` 的 `navtta-eval-fixes-20260909` 分支。
已在该分支的服务器直接更新：

```bash
cd /data1/wxy/code/NavTTA
git pull --ff-only git@github.com:SheldonWongww/SPARK-AVN.git navtta-eval-fixes-20260909
STREAM_PY=/data1/wxy/exp_data/NavTTA/vln/envs/streamvln/bin/python
"$STREAM_PY" -m pip install -e core
"$STREAM_PY" vln/scripts/run_streamvln_val_seen_search.py --stage search --gpus 0,1,2,3 --dry-run
```

如果服务器尚未创建这个本地分支，先运行：

```bash
git fetch git@github.com:SheldonWongww/SPARK-AVN.git navtta-eval-fixes-20260909
git switch -c navtta-eval-fixes-20260909 FETCH_HEAD
```

GPU 空闲后，启动 12 个完整搜索作业：

```bash
mkdir -p vln/results/tuning/streamvln_val_seen_search_v1
nohup "$STREAM_PY" -u vln/scripts/run_streamvln_val_seen_search.py \
  --stage search --gpus 0,1,2,3 \
  > vln/results/tuning/streamvln_val_seen_search_v1/scheduler.log 2>&1 &
```

`--gpus 4,5,6,7` 表示依次将 FSTTA/EAM/FeedTTA/ATENA 放到物理 GPU 4/5/6/7。
必须传 4 个不同 GPU，映射写入 `campaign.json`。每张卡最多一个评测进程，
启动前检查现有 compute process，并与 v6 共享 GPU 锁。无需额外设置
`CUDA_VISIBLE_DEVICES`；launcher 会设置所选物理卡。

可先用 `--stage smoke` 跑四方法各前 2 条，只用于加载和链路检查。
`--stage all` 表示 4 个 smoke 作业后运行全部 12 个搜索作业；smoke 不参与排名。
默认 `--stage search` 无额外 Source/Tent 或 smoke 评测。

查看日志及汇总：

```bash
tail -f vln/results/tuning/streamvln_val_seen_search_v1/scheduler.log
"$STREAM_PY" vln/scripts/run_streamvln_val_seen_search.py --stage report
```

`configs/` 保存冻结配置；`launcher_logs/` 保存每个候选的运行日志；
`jobs/<run-tag>/val_seen/` 保存逐 episode 结果、诊断和 completion；
`report.json`、`report.md` 报告完整性、监督方式、SR/SPL 和历史 Source 差值；
全部 12 组完整且通过验证后生成 `selection.json`，按 **SPL → SR → 编号较小**
选择每方法一组。有效但没有提升的结果同样保留，不将 Source 差值当作筛除条件。

中断后沿用同一 commit、相同 GPU 映射和参数，重启命令为：

```bash
nohup "$STREAM_PY" -u vln/scripts/run_streamvln_val_seen_search.py \
  --stage search --gpus 0,1,2,3 --retry-incomplete \
  >> vln/results/tuning/streamvln_val_seen_search_v1/scheduler.log 2>&1 &
```

已认证完成的候选自动跳过；未完成候选的输出、manifest、日志归档到 `incomplete/`，
然后从第 1 条 episode 重新开始。continual TTA 没有保存中间优化器/replay 状态，
不能只继续运行剩余后缀。代码、配置或 GPU 映射变化会被 campaign 冻结记录拒绝。

新 run manifest 绑定顶层 commit、配置、checkpoint shards/vision 资产摘要、
dataset 版本、episode 顺序、seed、硬件和输出 artifact 摘要。历史 Source 对比
显式标为 legacy；不要求把本地 `中间结果/` 复制到服务器。原始数据、checkpoint、
日志均不进入 Git。
