# R2R 跨 split 稳健重评计划（v1）

## 结论与目标

现有 R2R `val_unseen` 表不能继续作为最终主结果。问题不是简单的
“unseen 更难”，而是旧实验把单个 `val_seen` scene-blocked 顺序上选出的
continual-TTA winner，直接迁移到了长度更长、每个 scene 连续暴露更多次的
`val_unseen` 流。这样同时混入了单顺序过拟合、更新剂量变化和长相关流漂移。

本轮重评只回答一个问题：**在不查看 `val_unseen` 结果来选参的前提下，哪些
低漂移、方法语义固定的配置能在随机 episode stream 上稳定工作？** 不能把
“五种方法必须全为正收益”写成选择条件；如果冻结配置仍有负收益，应如实报告。

旧 `val_unseen` 聚合结果已经被研究者查看过，因此本轮只能称为预注册的重评与
确认，不能称为从未触碰过的 pristine lockbox。结构性约束仍然是：新一轮的
`val_unseen` 三个顺序不参与候选排序，冻结文件生成后才允许启动评估。

旧 canonical order seed 0 不删除、不改写，保留为
`scene-blocked continual stress test`。新的主协议使用全局打乱的 order seeds
`1/2/3`，三个模型消费完全配对的 episode 顺序。

## 为什么旧 `val_unseen` 不能直接解释为泛化失败

R2R 离散评估中，`val_seen` 有 1,021 个 instruction episodes、56 个 scenes，
`val_unseen` 有 2,349 个 episodes、11 个 scenes。旧顺序按
`scene_id -> natural episode_id` 排列，因此平均连续 scene block 从约 18 条增加
到约 214 条，最大 block 从 33 增加到 300 条。所有五种方法均为
`episodic=false`，所以参数和优化器状态在这些 episode 之间持续累积。

旧冻结结果已经显示同一配置在长流上的参数漂移放大，例如 DUET Tent
`0.0253 -> 0.0565`、DUET FSTTA `0.0175 -> 0.0665`。FSTTA 论文报告五个
shuffled streams，FeedTTA/ATENA 也按随机顺序或多个随机 seed 报告；旧
scene-blocked 顺序应视为额外压力测试，而不是论文对齐主协议。

## 方法约束

| 方法 | 更新信号与时机 | 本轮固定项 | 只允许变化的因素 |
|---|---|---|---|
| Tent-LN | 每个高层 action 对合法动作分布做 entropy minimization | continual；`update_interval=1`；模型特定 LN scope；AdamW；不裁剪梯度；不改动作策略 | DUET last-9 LN：`1e-6/1e-5`；HAMT all-LN：`1e-6/3e-6`；GOAT all-LN：`3e-6/1e-5` |
| FSTTA | FAST 在 episode 内聚合 action 梯度；SLOW 跨 episode 更新 anchor | last-4 LN；`M=3,N=4,q=.1,rho=.95,tau=.7,[a,b]=[.9,1.1]`；FAST optimizer 每 episode reset；SLOW persistent；不裁剪梯度 | 三模型均含 `(1e-4,3e-5)` 低 slow-LR 锚；DUET/HAMT 对比 `(1e-4,1e-4)`，GOAT 对比 `(3e-4,1e-4)` |
| EAM | 冻结 Source 分支；auxiliary 分支用可靠性 gate 与 reservoir replay 更新 | continual；模型特定 gate/memory/batch/interval 固定；buffer 未达到 batch size 前不更新 | 每模型两个低 LR |
| FeedTTA-argmax† | 每个 episode 结束后按二值成功/失败更新 trajectory score gradient | target-native argmax；模型特定 scope；`p/gamma/alpha` 固定 | 低 LR；另保留一个显式 length-normalized 候选 |
| ATENA† | episode mixture entropy；高不确定时查询二值反馈，否则 self-label | continual；完整可训练 policy；lazy evaluator query；`self_loss_weight=.1` | 成对 query/self LR；GOAT 保留已知低-query 区域 |

`†` 表示使用二值 episode feedback，不能归入无监督 TTA。FeedTTA 的论文用
`tau ~ pi_theta` 描述 REINFORCE，但没有发布动作 selector；本项目遵从前述决定，
主结果使用目标模型原生 argmax，因此必须报告为 `FeedTTA-argmax`。它不是无偏
on-policy sampling 复现。

FSTTA 明确排除旧 `M=1` 和 `M=8` winner：`M=1` 使 covariance/GDA 退化，
`M=8` 在 R2R 短 episode 末丢弃大量未满窗口梯度。论文默认高 slow LR
`6e-4/1e-3` 已在现有 DUET `val_seen` 上出现大幅退化，只作为既有 fidelity
证据，不再消耗本轮正式候选预算。两个正式点优先改变跨 episode 累积的 slow
LR，而不是重复测试近似相同的 fast LR；GOAT 保留一个较强 fast 分支用于避免
低 LR 完全退化为 Source。

Tent 不再使用草案中的 `max_grad_norm=1`：历史正向点与方法对齐实验均未裁剪，
而现有梯度范数表明阈值 1 会实际改变更新。DUET 使用已有唯一双指标正向的
last-9-LN 区域；HAMT/GOAT 固定各自已有优胜 scope，只比较低 LR。

## 已发现、必须先修的实现问题

1. DUET ATENA 官方 self-prediction 表示是 global/local CLS 拼接（`2D`），旧
   本地实现却取平均（`D`）。修复后所有旧 DUET-ATENA 数值作废。
2. EAM 论文 warm-up 在 reservoir 小于 batch size 时不更新；旧实现执行
   current-only update。修复后旧 EAM 数值仅作为历史证据。
3. 不改变 FeedTTA 的 feedback 符号、ATENA 的 lazy-query 时序，也不把
   `val_unseen` 指标用于回选配置。

## 候选规模与选择规则

每个模型的候选数为：Tent 2、FSTTA 2、EAM 2、FeedTTA 3、ATENA 2，共
11 个。每个候选完整运行 shuffled `val_seen` seeds `1/2/3`，因此选择阶段为
`3 models x 11 candidates x 3 orders = 99` 个 TTA jobs；Source 为确定性 argmax，
直接复用现有正式指标，不再运行。

每个 model-method cell 的可行门槛为：

1. 三个 seed 的 median SR 和 median SPL 都严格高于 Source；
2. worst-seed SR/SPL 不低于 Source `0.2 pp`；
3. 结果完整、有限，且方法 diagnostics/feedback contract 通过。

在可行候选中依次最大化 worst-seed SR gain、median SR gain、worst-seed SPL
gain、median SPL gain，再选择平均参数漂移更小者。没有候选通过时，仍按同一
排序冻结最稳候选，但将该 cell 标记为 `robust_gate_failed`，不得宣称方法有效。

## `val_unseen` 锁盒评估

全部 15 个 model-method winner 冻结并写入带 digest 的 registry 后，才允许启动
`val_unseen`。每个 winner 完整运行 shuffled seeds `1/2/3`，共 45 个 jobs；
报告 mean、sample std、worst seed 和逐 seed 指标。任何 `val_unseen` 结果都不得
回流选参。本轮完成前，旧 `results/final/r2r/` 继续保留但标记为
scene-blocked stress evidence；完成后再原子替换主结果目录。

全局 shuffle 消除了 seed-0 的超长 scene block，但不会消除 2,349 对 1,021
episodes 带来的约 2.3 倍流长度。本轮主结果仍要求完全冻结同一配置，不临时按
split 缩放 LR；这是为了避免五种方法按 episode/action/update 数采用不同缩放
口径而引入新的算法因素。最终报告必须同时展示参数漂移。若长流仍退化，只能
登记为跨 horizon 失败；任何 horizon-normalized LR sensitivity 必须另行预注册、
两套协议并列报告，不能查看 `val_unseen` 后择优。

执行始终按 `DUET -> HAMT -> GOAT`，每个模型内按
`Tent -> FSTTA -> EAM -> FeedTTA -> ATENA` 设置 barrier；同一 phase 只并行
该 model-method 的候选/顺序任务，沿用已测显存上限，不混跑不同方法。

正式执行分成三个独立进程，不使用一步完成的 `--stage all`：

```bash
BATCH=vln-r2r-cross-split-robust-v1
python3 vln/scripts/run_r2r_cross_split_robust_eval.py \
  --batch-id "$BATCH" --plan-only
python3 vln/scripts/run_r2r_cross_split_robust_eval.py \
  --batch-id "$BATCH" --stage selection --resume --confirm-reviewed
python3 vln/scripts/run_r2r_cross_split_robust_eval.py \
  --batch-id "$BATCH" --stage freeze --resume --confirm-reviewed
python3 vln/scripts/run_r2r_cross_split_robust_eval.py \
  --batch-id "$BATCH" --stage evaluation --resume --confirm-reviewed
```

第三条命令只生成并认证 `FROZEN.json`；第四条命令只能从该文件构造 45 个
评估任务。每一步均重新认证持久 job、metrics、diagnostics 与 formal manifest，
不能通过修改中间 `SUMMARY.json` 注入未注册候选。
