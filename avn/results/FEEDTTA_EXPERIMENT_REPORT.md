# FeedTTA 在 AVN 上的实验进展

更新日期：2026-08-04

当前状态：**Stage 1 的 48 组导航运行已经完成；SMT+Audio 24 组 validated，
ENMuS 24 组保留参数计数告警但指标可用于 provisional 选参。Stage 2 的 78 组
配置已经按 Stage 1 结果冻结，尚未运行。**

本文用于持续记录 FeedTTA 在 AVN 上的实现口径、实验进度、搜索结果和最终
主表配置。详细搜索定义见
[`FEEDTTA_EXPERIMENT_PLAN.md`](../experiments/FEEDTTA_EXPERIMENT_PLAN.md)。

## 1. 研究范围与方法属性

- 导航模型：SMT+Audio、ENMuS。
- 开发场景：single-source、canonical `val`、seed 0、2,000 episodes。
- 后续正式复验：两个模型的 single-source 与 multi-source。
- FeedTTA 在每个 episode 结束后使用 simulator 提供的成功/失败二值反馈，
  因此属于 **feedback-supervised TTA**，不是严格的无监督 TTA。
- 论文主表中统一记为 `FeedTTA†`，并用脚注明确其使用二值 episode feedback。

## 2. 当前实现口径

当前实现遵循 FeedTTA 主文的核心流程：

1. 从当前策略分布中采样动作，不使用贪心 `argmax`。
2. episode 成功反馈为 `+1`，失败反馈为 `-1`。
3. 使用带折扣的 trajectory REINFORCE 梯度，不加入 baseline、advantage、熵损失
   或轨迹长度归一化。
4. episode 内模型参数保持不变，在 episode 结束后更新一次。
5. 使用 continual adaptation，固定 `EPISODIC=False`。
6. 冻结音频、视觉等模态编码器和 critic，只更新 state-fusion encoder 与
   action head。

为避免保存每个动作步的完整梯度，当前使用数学等价的在线折扣累加器，将轨迹
梯度内存复杂度由 `O(TD)` 降为 `O(D)`。

| 模型 | 更新模块 | 参数张量数 | 可更新参数量 |
|---|---|---:|---:|
| SMT+Audio | `net.smt_state_encoder` + `action_distribution` | 42 | 1,197,156 |
| ENMuS | `net.smt_state_encoder` + `action_distribution` | 77 | 3,638,884 |

SGR 默认锚点为 `p=0.05, alpha=-0.2`，并使用独立的 `SGR_SEED=0`，不污染策略
动作采样的随机数状态。

### 论文歧义的处理

- 主文 Eq. (5) 与附录 B.1 的归一化位置不一致。当前实现跟随主文 Eq. (5)，
  后续论文中不能将该公式描述为严格无偏。
- 论文定义 SGR 时要求 `alpha<0`，但 R2R 配置又写成 `alpha=+0.1`。正数只会
  缩放梯度而不会反转梯度，因此 `+0.1` 只作为机制对照，不作为默认 SGR。
- 原论文未明确报告 `gamma` 和完整优化器设置，因此第一阶段需要在 AVN 上搜索
  `gamma`；优化器固定为 Adam，避免同时扩大搜索维度。

## 3. 已完成工作

| 工作项 | 状态 | 说明 |
|---|---|---|
| FeedTTA 核心逻辑复核与修正 | 已完成 | 对齐动作采样、二值反馈、轨迹梯度和 episode-end update |
| SMT+Audio / ENMuS 参数冻结范围 | 已完成 | 启动器会在运行时校验张量数和参数量 |
| Stage 1 搜索脚本 | 已完成 | 48 组，两个模型联合调度 |
| Stage 2 搜索脚本 | 已更新 | 78 组，固定 Stage 1 winner，含官方 SGR 网格和四个机制对照 |
| 运行 provenance 与结果校验 | 已完成 | 校验 commit、checkpoint/data/stream digest、配置和 diagnostics |
| 调度器 dry-run | 已完成 | Stage 1 为 48 组，Stage 2 为 78 组，四卡分配通过 |
| 静态与结构检查 | 已完成 | `py_compile`、`git diff --check`、`tools/verify_layout.py` 通过 |
| GitHub 提交 | 已完成 | 开发锚点 commit：`f288da7` |
| 服务器 PyTorch 单元测试 | 待完成 | Mac 本机缺少 PyTorch 环境 |
| 2-episode 真实 smoke | 待完成 | 应在服务器最终运行 commit 上执行 |
| Stage 1 正式搜索 | 已完成 | SMT+Audio job 5；ENMuS provisional job 39 |
| Stage 2 正式搜索 | 待完成 | 两模型共 78 组 |
| single/multi-source 最终复验 | 待完成 | 搜索结束并冻结配置后执行 |

`f288da7` 只是当前开发锚点。由于后续还会合入 EAM 多声源实验相关改动，正式
FeedTTA 的 run manifest 应记录最终实际运行的 commit，而不能预先固定为该提交。

## 4. 两阶段超参数搜索

### Stage 1：适应强度与轨迹信用长度

固定 `p=0.05, alpha=-0.2`，搜索：

```text
LR    = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 5e-6]
gamma = [0.90, 0.95, 0.99, 1.0]
```

每个模型 `6 × 4 = 24` 组，两个模型共 **48 组**。

两个模型分别选择配置：先要求 `SR >= Source SR`，再按 SPL、SR、较低参数漂移
依次排序。如果没有任何配置达到 Source SR，则停止进入 Stage 2，先报告 Pareto
frontier 并重新讨论选择规则。

### Stage 2：SGR 官方网格与机制对照

分别固定两个模型在 Stage 1 选出的 `LR/gamma`，搜索：

```text
p     = [0.01, 0.05, 0.1, 0.2, 0.3]
alpha = [-0.01, -0.025, -0.05, -0.075, -0.1, -0.2, -0.3]
```

官方笛卡尔积为每模型 35 组，另增加四个对照：

| 对照 | p | alpha | 目的 |
|---|---:|---:|---|
| no SGR | 0 | -0.2 | 检查普通二值反馈 REINFORCE |
| gradient dropout | 0.05 | 0 | 区分反转与随机丢弃 |
| positive scaling | 0.05 | +0.05 | 检查正向梯度缩放 |
| literal R2R setting | 0.05 | +0.1 | 复现论文 R2R 的字面配置 |

每个模型共 39 组，两个模型共 **78 组**。两阶段总计 **126 组**正式搜索实验，
smoke 不计入该数量。

## 5. 固定实验协议

| 项目 | 设置 |
|---|---|
| split | canonical `val` |
| 数据流 | 20 scenes × 100 episodes，global shuffle |
| 顺序 | seed 0，与 Source/Tent 的主表流对齐 |
| 动作选择 | 从策略分布采样 |
| 适应模式 | continual，`EPISODIC=False` |
| episode 数 | 每组 2,000 |
| 优化器 | Adam，betas=(0.9, 0.999)，eps=1e-5 |
| weight decay / clipping | 0 / 关闭 |
| 轨迹梯度 | 折扣求和，不按长度归一化 |
| SGR RNG | 独立生成器，seed 0 |

当前用于 Stage 1 选择约束的 matched Source 候选值为：

| 模型 | Source SR | Source SPL | 状态 |
|---|---:|---:|---|
| SMT+Audio single-source | 54.15 | 29.4229 | provisional |
| ENMuS single-source | 66.55 | 36.0480 | provisional |

上述 Source 数值来自相同 seed-0 canonical 流，但 checkpoint 训练 provenance 尚未
补齐，因此继续标记为 provisional。

## 6. 后续执行顺序

1. 等当前 EAM 搜索结束，完成 EAM 多声源运行脚本及其他必要改动。
2. 将所有改动提交到同一个最终 commit；当前运行中的正式实验结束前不要在服务
   器工作区执行 `git pull`。
3. 服务器拉取最终 commit，安装 `core`，运行 FeedTTA 单元测试和 2-episode smoke。
4. 运行 Stage 1 的 48 组实验，检查完整性并分别冻结两个模型的 `LR/gamma`。
5. 在相同 checkpoint、数据和 episode 流上运行 Stage 2 的 78 组实验。调度和
   证据代码可以使用后续 commit；共享文件允许经过审计的 ATENA/Source-argmax
   分支变化，但启动器会比较 FeedTTA adapter 与实际 trainer 路径的归一化 AST，
   并严格比较专属配置和 runner。FeedTTA 行为变化时必须重跑 Stage 1。
6. 冻结最终配置后，独立复验 SMT+Audio/ENMuS 的 single-source 和
   multi-source，复验结果才可登记到 AVN 主对比表。
7. 整理 PSR、CSR、ASR、推理耗时、参数漂移和逐流稳定性，并更新本报告和飞书
   FeedTTA 工作表。

EAM 与 FeedTTA 可以在硬件资源允许时并行运行，但必须使用同一干净 commit，且
两个调度器的 `--jobs-per-gpu` 互不感知；启动前需要人工核算两者的合计并发数。

## 7. 结果与产物位置

正式日志目录：

```text
avn/results/logs/feedtta_stage1/<batch-id>/
avn/results/logs/feedtta_stage2/<batch-id>/
```

每个 batch 应包含 `batch.json`、`grid.csv`、`scheduler.log`、`SUMMARY.json` 和
`metrics.csv`；每个 run 的可追踪产物保存在：

```text
avn/results/runs/<run-id>/manifest.json
avn/results/runs/<run-id>/summary.json
avn/results/runs/<run-id>/diagnostics.json
```

后续结果分析除 SR、SPL、SoftSPL、DTG、NDTG、NA、SNA、SWS 和 reward 外，
还需基于相同 episode ID 计算：

```text
PSR = P(success_FeedTTA | success_Source)
CSR = P(success_FeedTTA | failure_Source)
ASR = (PSR + CSR) / 2
```

## 8. 当前结果登记

| 模型 | 场景 | Stage 1 最优配置 | Stage 2 最优配置 | SR | SPL | 状态 |
|---|---|---|---|---:|---:|---|
| SMT+Audio | single-source | — | — | — | — | 未运行 |
| ENMuS | single-source | — | — | — | — | 未运行 |
| SMT+Audio | multi-source | — | — | — | — | 待配置冻结后复验 |
| ENMuS | multi-source | — | — | — | — | 待配置冻结后复验 |

在正式日志完成并通过 manifest 校验前，不从 console 中手工摘取临时数值填入主表。
