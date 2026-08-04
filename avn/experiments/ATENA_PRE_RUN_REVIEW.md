# AVN ATENA 运行前复核

状态：**代码复核完成；通过 1/5/20 episode smoke 后批准完整网格**

最后复核：2026-08-04

适用模型：SMT+Audio、ENMuS

ATENA 当前实现以论文和官方 DUET 代码为复现依据，但 VLN 与 AVN 在动作粒度、轨迹长度、成功条件和网络结构上差异明显。任何 smoke test、超参数搜索或正式评估开始前，都必须重新完成本文件中的判断；不能直接把默认参数当作最终实验设定。

## 1. 当前对齐基线

- 伪专家动作和环境执行动作均为策略 argmax。
- 使用原始动作熵和 episode 平均熵，默认候选阈值为 `delta=0.1`。
- episode 结束后联合优化有符号混合熵与自预测 BCE。
- 自预测 BCE 的梯度进入预测头和策略表示。
- 默认更新预训练策略中原本可训练的全部参数。
- 每个 episode 重建 AdamW；预测头与策略使用相同的 query/self 学习率。
- 使用 AVN Habitat `success` 作为被查询的二元 oracle 标签。

上述内容定义“论文/官方对齐版 ATENA”。若改变动作选择、更新范围、损失或查询规则，必须在实验名和结果表中标为 ATENA-AVN 变体，不能覆盖对齐基线。

### 1.1 本次代码审计结论

论文正文与开源实现存在细节差异时，本项目以 `references/repos/tta/vln/NeurIPS25_att_vln` 中的 DUET/ETPNav 代码为准。当前 AVN 映射如下：

| ATENA 官方代码行为 | AVN 实现 | 结论 |
|---|---|---|
| 执行策略概率最大的动作，并把同一动作构造成 one-hot 伪专家 | `select_action()` 强制 argmax，且校验实际动作与 argmax 相同 | 对齐 |
| 对每步混合分布熵求和并在 episode 末除以动作步数 | episode 末对逐步重放的混合熵梯度乘 `1/T` 后累积 | 数学等价 |
| 成功时最小化、失败时最大化混合熵 | 根据最终二元反馈给平均混合熵梯度乘 `+1/-1` | 对齐 |
| 平均内部状态输入两层 MLP、ReLU、LayerNorm、线性二分类头 | 使用动作头之前的 AVN 融合表示，预测头结构和 BERT 初始化方式相同 | 结构对齐；状态语义为任务映射 |
| 平均原始动作熵大于 `delta` 时查询真实反馈，否则采用自预测 | 使用未归一化的原始 Shannon 熵和严格的 `>` 判定 | 对齐 |
| query/self 分支采用不同学习率，每个 episode 新建 AdamW | `LR_QUERY/LR_SELF` 分支每次重建 AdamW，不跨 episode 保存动量 | 对齐 |
| 优化完整导航策略和自预测头 | 更新原本可训练的 action-policy 参数与预测头；排除 AVN 独有且不参与动作损失的 value critic，保留基线原本冻结的 ENMuS 感知编码器 | 语义对齐 |
| 官方最多约 15 步并直接保留整图 | AVN 最长约 500 步，保存 CPU 输入后逐步确定性重放 | 目标和梯度对齐，执行方式适配 AVN |

此前实现中 `EPISODIC=True` 会在下个 episode 开始时立即抹掉刚完成的更新，现在已显式拒绝该无效配置。非 query episode 的真实 success 只在训练标签固定之后用于离线统计自预测准确率，不进入损失或反馈选择。

仍不可消除的任务差异包括：AVN 使用固定的低层四动作空间、轨迹显著更长；自预测状态是 AVN 融合表示而非 DUET 的全局/局部 VLN 表示；实验中的 oracle 是 Habitat 自动计算的成功标签而不是真实人类反馈。这些差异必须在论文方法复现说明中披露。

## 2. 运行前必须重新判断

### 动作协议

- 固定同一 checkpoint 和 episode 顺序，先比较 Source-sample 与 Source-argmax。
- ATENA 的主对照必须包含 Source-argmax，不能把动作协议带来的收益记成适应收益。
- 确认 FeedTTA 等其他方法的动作协议，并在 manifest 中逐项记录。

### 长轨迹与反馈归因

- 核对本次数据流的平均、P95 和最大 episode 步数。
- 判断最终成败是否足以给整条低层动作序列分配同方向反馈。
- 单独保留 MEO+真实反馈、entropy query、完整 SAL 三个阶段的消融计划，用于定位负迁移来源。

### 查询规则与反馈预算

- `delta` 必须包含论文/DUET 的 `0.1`。已有 AVN 诊断显示平均动作熵约为 `0.7`，因此本轮使用 `{0.1, 0.5, 0.75, 1.0}`，用于实际覆盖高、中、低查询率；不能只照搬都会接近全查询的低阈值。
- 预估并记录每个候选阈值的 query rate；查询率接近 0% 或 100% 时不得直接进入正式评估。
- 结果必须同时报告 oracle episode 数量、query rate 和自预测准确率。
- ATENA 属于消费二元 episode 反馈的方法，不能作为无监督 TTA 报告。

### 更新范围与稳定性

- 论文/官方对齐基线使用 `PARAM_SCOPE=all`。
- 若完整策略更新产生 OOM、NaN、明显灾难性漂移或不可接受时延，停止该 run；不得在同一实验 ID 下静默改成 LayerNorm 或动作头更新。
- 参数高效版本应建立独立配置和实验 ID，并作为 ATENA-AVN 消融报告。

### 显存与运行时间

- 官方代码会保留整条 episode 的计算图；AVN 最长约 500 个动作步，直接照搬会使显存随轨迹长度增长。当前实现采用 eval 模式下数学等价的逐步重放：在线阶段只保存 CPU 输入和隐藏状态，episode 末逐步重算并累积联合目标梯度，不改变 MEO/SAL 目标。
- 先运行 1、5、20 episode 的递增 smoke test；每个阶段记录峰值显存、CPU 轨迹缓存、episode 自适应时间、是否出现 NaN/Inf、重放特征误差以及参数漂移。
- `max_replay_feature_abs_error` 必须不超过 `1e-5`，否则说明 dropout 或状态变更破坏了等价重放，该 run 无效。

## 3. 允许解除门禁的最低条件

- Source-sample 与 Source-argmax 已有可核验结果。
- checkpoint、数据版本、episode 顺序和随机种子已经固定。
- 已确定本轮是 smoke、开发集搜索还是正式评估。
- 已明确 `LR_QUERY`、`LR_SELF`、`MIX_LAMBDA`、`QUERY_THRESHOLD` 和 `PARAM_SCOPE` 的候选范围。
- 已确认反馈预算和需要保存的逐 episode 诊断字段。
- 已指定出现 OOM、NaN、query collapse 或显著 Source 退化时的停止条件。

完成复核后，只在本次 run 的配置覆盖中设置：

```yaml
TTA:
  METHOD: atena
  ATENA:
    PREFLIGHT_APPROVED: True
```

不得把共享默认值永久改成 `True`。正式结果还必须满足项目统一的 run manifest 要求。

## 4. 首轮建议顺序

1. 保留现有 Source-sample，并在本批次补跑 Source-argmax。
2. 用正式网格中的论文锚点完成 1/5/20 episode 显存与数值 smoke test。
3. 一次性运行 `atena_grid.yaml` 定义的 288 个 ATENA 点；它同时覆盖学习率、`lambda`、`delta` 和自预测损失权重。
4. 通过 query rate 区分近似全真实反馈、完整 SAL 和近似全自反馈；主表候选只从 query rate 位于 `[5%, 95%]` 且不低于匹配 Source-argmax 的点中选择。
5. 若所有有效点都发生 query collapse 或均低于 Source-argmax，再根据诊断决定是否做极少量边界补点，而不是预先追加第二阶段大网格。

完整网格入口：

```bash
python3 avn/scripts/run_atena_grid.py --dry-run
```

网格是以下组合在每个模型上的完整笛卡尔积：

- `(LR_QUERY, LR_SELF)`：`(3e-8,1e-8)`、`(1e-7,1e-8)`、`(3e-7,3e-8)`、官方 ETPNav 的 `(5e-7,1e-8)`、官方 DUET-R2R 的 `(8e-7,1e-7)`、官方 DUET-REVERIE 的 `(5e-6,1e-7)`；
- `MIX_LAMBDA`：`0.25/0.5/0.75`；
- `QUERY_THRESHOLD`：`0.1/0.5/0.75/1.0`；
- `SELF_LOSS_WEIGHT`：`0.1/0.25`。

因此每个模型有 `6×3×4×2=144` 个 ATENA 点；再加两个模型各一个 Source-argmax，共 `290` 个作业。学习率采用成对搜索而不做两个学习率的无约束笛卡尔积，是为了保留开源代码中三个已经联合调好的 query/self 组合，同时避免大量不合理的 `LR_SELF > LR_QUERY` 组合。
