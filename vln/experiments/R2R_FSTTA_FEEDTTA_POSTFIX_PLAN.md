# R2R FSTTA / FeedTTA 论文对齐修正与补搜计划

状态：已完成。78/78 个任务在 commit `b2e37dd` 上通过验证；搜索 spec
SHA256 为 `f979f7809265b9144a9ab75da1ede8871b5f93eefb1e87014a42a90011158490`。
结果与解释见
[`R2R_FSTTA_FEEDTTA_POSTFIX_V1.md`](../results/analysis/hparam_search/R2R_FSTTA_FEEDTTA_POSTFIX_V1.md)。

## 1. 原文复核结论

FeedTTA 原文（ICML 2025）用 `tau ~ pi_theta` 和 REINFORCE 公式描述轨迹，
但没有明确写实验动作是 `sample`、`argmax` 还是 `greedy`，也没有公开官方
实现。论文采用的 DUET/HAMT 原始测试入口均使用 argmax。因此此前在 VLN 中
强制 sampling 只能称为一种公式导向移植，不能称为已验证的原实验协议。本轮
离散/连续 VLN 恢复目标模型原生 argmax；AVN 的 PPO evaluator 继续保持原生
sampling。

FeedTTA 原文还规定：停止后取得一次 `+1/-1` episode feedback，冻结语言/视觉
编码器并从 cross-modal encoder 起更新。论文在 R2R/R2R-CE 表中写的是
`p=.05, alpha=+.1`；此前本项目所谓 paper anchor `alpha=-.2` 实际来自
REVERIE val-unseen。论文未报告 gamma，本轮同时保留已有好区 `.8` 和目标策略
常用 `.9`。

FSTTA 原文要求每 `M` 个动作执行 FAST、每 `N` 个 episode 执行 SLOW，最后四个
LayerNorm 可训练；`q=.1, rho=.95, tau=.7, [a,b]=[.9,1.1]`。当前 FAST/SLOW
公式主体与论文一致，但历史方差 EMA 曾在每个 episode 开始时被清零，违反原文
“maintained for all samples throughout the test stage”。本轮将其改为测试流级状态。

## 2. 同时修复的问题

1. FeedTTA 默认动作改为 target-native argmax，显式 sampling 仅保留为消融。
2. DUET/GOAT/HAMT 的 R2R feedback 直接调用对应 evaluator 对最终提交轨迹计算
   success；DUET/GOAT 的历史 stop-score endpoint 不再与反馈 endpoint 分离。
3. FeedTTA 增加三种可审计 scope：
   - `paper_full`：完整 cross-modal stack 和动作头（不含其前面的地图/位置
     embedding）；
   - `last_crossmodal`：最后一层 cross-modal block 和动作头；
   - `action_head`：仅动作头。
4. GOAT 当前动作之后才计算、且进入下一步前已 detach 的 pooler/local-history
   模块不再计入 FeedTTA scope。
5. FSTTA diagnostics 必须记录 `variance_history_lifetime=test_stream`；FeedTTA
   diagnostics 必须记录 argmax、scope、完整 feedback 数，并要求 feedback 成功数
   与 evaluator SR 推算的成功数一致。

## 3. 搜索规模

只搜索 FSTTA 和 FeedTTA，三个模型各自独立选参：

| 方法 | DUET | HAMT | GOAT | 合计 |
|---|---:|---:|---:|---:|
| FSTTA | 12 | 12 | 12 | 36 |
| FeedTTA | 14 | 14 | 14 | 42 |
| **总计** | **26** | **26** | **26** | **78** |

FSTTA 不再重复已经失败的低学习率大网格，只围绕父批 winner、DUET 的联合
SR/SPL ridge、GOAT 的次级 ridge 做局部搜索，并明确重跑原文 DUET 配置
`(lr_fast,lr_slow,M,N)=(6e-4,1e-3,3,4)`。FeedTTA 以
`last_crossmodal` 为主，联合搜索小范围 LR、`gamma={.8,.9}`、论文正 alpha 与
各模型旧负 alpha 好区，并各保留 `paper_full`、`action_head` 和 no-SGR 对照。
历史 FeedTTA winner 只提供 LR/SGR 数值先验；旧作业使用 sampling 和旧 full
scope，不能冒充本轮 argmax/新 scope 配置的精确 parent replay。每项数值先验均
绑定旧 `job.json` 的路径与 SHA256，并在启动前核对 run tag、方法、模型和原参数。

## 4. 对照、选择与执行

- 复用已固定的三个 argmax Source；不重复运行 Source，不再运行 sampled control。
- 完整运行 R2R `val_seen` 1,021 episodes，order seed 0；不使用 prefix 筛选。
- 每个 job 从 Source checkpoint 重新开始；按成功数、SPL、较低 drift、较少更新
  排序。
- 严格顺序为 DUET FSTTA、DUET FeedTTA、HAMT FSTTA、HAMT FeedTTA、GOAT
  FSTTA、GOAT FeedTTA；模型和方法之间均设 barrier。
- 并发使用此前测量和保守投影得到的上限：FSTTA `14/11/14`，FeedTTA
  `6/5/6`。其中 GOAT FSTTA/FeedTTA 是基于五任务基线的保守投影，其余四项有
  已完成观测；29,000 MiB 为计划线，30,000 MiB 为紧急停止线。
- 正式启动前校验复用 Source 的 GRID、SUMMARY、job、parameters、metrics 和
  formal manifest 的 SHA256；缺文件或 hash 不符时禁止启动，但不会重跑 Source。
- winner 只有在相对 Source 多至少 1 个 success，或 success 持平且 SPL 至少
  `+0.10 pp` 时，才晋级 order/SGR seeds 1、2 的确认实验。

这轮结果必须继续把 FeedTTA 标为使用二值 episode feedback 的方法，不能与
无监督 FSTTA 混称为 unsupervised TTA。
