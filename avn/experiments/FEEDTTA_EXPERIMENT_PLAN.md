# FeedTTA 在 AVN 上的实现与两阶段超参数实验方案

状态：实验开发方案。本文档定义 SMT+Audio 与 ENMuS 单声源 FeedTTA
超参数搜索，不直接登记论文主对比表结果。

## 1. 方法口径

FeedTTA 使用 episode 结束后的二元反馈进行一次 REINFORCE 更新。成功反馈为
`+1`，失败反馈为 `-1`。对于长度为 `T` 的轨迹，适应梯度为：

\[
\sum_{t=0}^{T-1}\gamma^{T-1-t}\mathcal F
\nabla_\theta\log\pi_\theta(a_t\mid s_t).
\]

当前实现使用逐动作梯度的在线折扣累加器，数学上与保存全部动作梯度后求和
等价，但内存复杂度从 `O(TD)` 降为 `O(D)`。动作必须从当前策略分布采样；
每个 episode 内参数保持不变，获得反馈后只更新一次。

AVN 中冻结音频、视觉等模态编码器和 critic，更新：

```text
net.smt_state_encoder
action_distribution
```

这对应论文“冻结视觉/语言编码器，从跨模态编码器开始更新”的任务移植。
SMT+Audio 预计更新 42 个参数张量、1,197,156 个标量；ENMuS 预计更新 77 个
参数张量、3,638,884 个标量。启动器会用运行 diagnostics 验证该范围。

FeedTTA 使用真实的 simulator `success` 作为完美二元 oracle，因此属于
`feedback-supervised TTA`，论文表格中必须标记为 `FeedTTA†`，不能描述为
无监督 TTA。

## 2. 论文歧义与本项目决定

1. 主文 Eq. (5) 仅对未被选中的梯度维度除以
   `alpha*p + 1-p`，附录 B.1 却把整个掩码梯度都除以该分母。当前基线严格
   跟随主文 Eq. (5)，运行 manifest 记录
   `main_text_eq5_unselected_coordinates_scaled`。不能在论文中声称这个主文
   公式严格无偏。
2. 论文规定 `alpha<0`，官方搜索范围也全部为负数，但 R2R 设置又写成
   `alpha=+0.1`。正数不会反转梯度。本项目把负 `alpha` 记为 SGR，把
   `alpha=0` 记为 gradient dropout，把正 `alpha` 明确记为 gradient scaling。
3. 论文未报告具体 `gamma` 和优化器细节。本项目以 AVN PPO 默认值
   `gamma=0.99`、Adam 为锚点，并在第一阶段显式搜索 `gamma`。

## 3. 两阶段共同协议

| 项目 | 固定设置 |
|---|---|
| 模型 | SMT+Audio、ENMuS |
| 场景 | single-source |
| split | canonical `val` |
| 数据流 | 20 scenes × 100 episodes，global shuffle |
| episode 顺序 | seed 0，与 Source/Tent 对齐 |
| 动作 | 从策略分布采样 |
| 适应方式 | continual，`EPISODIC=False` |
| 轨迹梯度 | 折扣和，不做长度归一化 |
| 参数范围 | 完整 state-fusion encoder + action head |
| 优化器 | Adam，betas=(0.9, 0.999)，eps=1e-5 |
| 正则项 | weight decay=0，gradient clipping 关闭 |
| SGR RNG | 独立生成器，`SGR_SEED=0` |
| 反馈 | 每个 episode 的 simulator success/failure |

正式搜索固定为 2,000 episodes，不允许修改 seed、数据 split、episode 顺序、
动作协议、解冻范围或轨迹归一化设置。短流程验证只能通过 `--smoke` 运行，
不能进入实验分析表。

## 4. 第一阶段：适应强度网格

目的：分别确定两个模型稳定的学习率和轨迹信用长度。SGR 固定为 REVERIE
val-unseen 的真实反转锚点：`p=0.05, alpha=-0.2`。

```text
LR    = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 5e-6]
gamma = [0.90, 0.95, 0.99, 1.0]
```

每个模型 `6 × 4 = 24` 组，两个模型共 48 组。对于约 160 步的 AVN 轨迹，
四个 gamma 的有效折扣质量约为 10、20、80、160 步，因此 gamma 不能直接从
VLN 设置机械迁移。

启动器：

```bash
python3 avn/scripts/run_feedtta_stage1.py --dry-run \
  --gpus 0,1,2,3 --jobs-per-gpu 2 \
  --batch-id feedtta-stage1-v1-seed0

screen -dmS avn_feedtta_stage1 \
  python3 avn/scripts/run_feedtta_stage1.py \
    --gpus 0,1,2,3 --jobs-per-gpu 2 \
    --batch-id feedtta-stage1-v1-seed0

screen -d -r avn_feedtta_stage1
```

`--jobs-per-gpu` 是 SMT+Audio 与 ENMuS 共用的单卡并发上限，可根据服务器资源
手动调整，不在代码中固定为 4 或 5。

选择规则对两个模型分别执行：首先要求 `SR >=` 对应 Source；在满足条件的
配置中以 SPL 最高为主，SR 次高、参数漂移更小依次作为 tie-breaker。如果没有
配置满足约束，则保留 Pareto frontier 并明确记录，不得静默改用单点最大 SR。
当前约束使用相同 canonical 流上的 Source SR：SMT+Audio 为 `0.5415`，ENMuS
为 `0.6655`。Stage-2 preflight 会从 Stage-1 `metrics.csv` 重算该规则并拒绝
与 winner 不一致的命令行配置；如果没有配置满足约束，Stage 2 会停止，需先
讨论并修订选择规则，而不会自动挑选一个结果。

## 5. 第二阶段：官方 SGR 网格和机制对照

第二阶段必须在第一阶段分析完成后启动。SMT+Audio 和 ENMuS 分别通过命令行
显式传入各自选定的 `LR/gamma`，且只能使用第一阶段搜索过的取值。启动器还
要求 `--stage1-batch-id`，并验证该 48-job batch 已完整结束、选择值确实存在于
validated Stage-1 数据中，同时把 Stage-1 `metrics.csv` 和 `SUMMARY.json` 的
SHA256 写入 Stage-2 batch provenance。两个阶段必须使用同一 Git commit、数据
索引及 episode 内容/顺序；如果中间修改 FeedTTA 代码，应重新运行第一阶段。

论文官方网格：

```text
p     = [0.01, 0.05, 0.1, 0.2, 0.3]
alpha = [-0.01, -0.025, -0.05, -0.075, -0.1, -0.2, -0.3]
```

每个模型 35 组，并增加四个互不重复的机制对照：

| variant | p | alpha | 含义 |
|---|---:|---:|---|
| `no_sgr` | 0 | -0.2 | 仅二元反馈 REINFORCE |
| `gradient_dropout` | 0.05 | 0 | 随机梯度 dropout |
| `gradient_scaling_0p05` | 0.05 | +0.05 | 正梯度缩放对照 |
| `gradient_scaling_0p1` | 0.05 | +0.1 | 论文 R2R 字面设置 |

每模型 39 组，两个模型共 78 组。下面数值只是命令格式示例，正式启动时必须
替换为第一阶段经审阅选定的模型级配置：

两阶段合计 126 组完整实验；smoke 运行不计入该数量。

```bash
python3 avn/scripts/run_feedtta_stage2.py --dry-run \
  --stage1-batch-id feedtta-stage1-v1-seed0 \
  --smt-audio-lr 1e-7 --smt-audio-gamma 0.99 \
  --enmus-lr 3e-8 --enmus-gamma 0.95 \
  --gpus 0,1,2,3 --jobs-per-gpu 2 \
  --batch-id feedtta-stage2-v1-seed0

screen -dmS avn_feedtta_stage2 \
  python3 avn/scripts/run_feedtta_stage2.py \
    --stage1-batch-id feedtta-stage1-v1-seed0 \
    --smt-audio-lr 1e-7 --smt-audio-gamma 0.99 \
    --enmus-lr 3e-8 --enmus-gamma 0.95 \
    --gpus 0,1,2,3 --jobs-per-gpu 2 \
    --batch-id feedtta-stage2-v1-seed0

screen -d -r avn_feedtta_stage2
```

第二阶段沿用第一阶段选择规则。最终冻结配置后，需要在最终 commit 上独立
复验 SMT+Audio/ENMuS 的 single-source 与 multi-source；搜索网格最大值不能
直接复制到主对比表。

## 6. Smoke、恢复和输出

每个阶段的 `--smoke --episodes 2` 只选择两个模型各一个论文锚点，用于验证
配置、参数范围、反馈更新和结果归档链路：

```bash
python3 avn/scripts/run_feedtta_stage1.py \
  --gpus 0,1,2,3 --jobs-per-gpu 1 \
  --smoke --episodes 2 --allow-dirty \
  --batch-id feedtta-stage1-smoke
```

正式实验要求服务器 tracked worktree 干净。运行中 launcher 每 10 秒检查 HEAD
与 tracked status；代码改变会终止调度。中断后确认没有旧 worker，再用完全相同
参数和 batch id 增加 `--resume`。已经完成的 job 只有在 manifest、配置覆盖、
checkpoint/data/stream SHA256、逐 episode 统计和 diagnostics 全部重新验证后才会
跳过。

输出位置：

```text
avn/results/logs/feedtta_stage1/<batch-id>/
avn/results/logs/feedtta_stage2/<batch-id>/
```

每个 batch 包含 `batch.json`、`grid.csv`、`scheduler.log`、`SUMMARY.json`、
`metrics.csv`；每个 job 单独保存 console、参数、exit code、manifest 指针和
`metrics.json`。每次评估在 `avn/results/runs/<run-id>/` 保存可跟踪的
`manifest.json`、`summary.json` 和精简 `diagnostics.json`；逐 episode stats、
TensorBoard 和原始输出位于其 `raw/` 子目录，不进入 Git。

当前导入的 Source checkpoint 训练 provenance 仍不完整；即使运行时 digest、
commit 和数据流均验证通过，结果在补齐 checkpoint 来源前仍应标记为
provisional。

## 7. 分析指标

除 AVN 的 SR、SPL、SoftSPL、DTG、NDTG、NA、SNA、SWS 和 reward 外，还应利用
相同 episode ID 的 Source 与 FeedTTA `val_stats_0.json` 计算论文指标：

```text
PSR = P(success_FeedTTA | success_Source)
CSR = P(success_FeedTTA | failure_Source)
ASR = (PSR + CSR) / 2
```

同时检查成功/失败反馈数、轨迹长度、实际 SGR 选择比例、动作 NLL、梯度范数和
相对参数漂移，避免只根据最终 SR/SPL 忽略中后段坍塌。
