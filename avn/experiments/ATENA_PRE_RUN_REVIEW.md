# AVN ATENA 运行前复核

状态：**未批准运行**  
最后复核：2026-07-21  
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

- `delta` 必须在开发流上搜索，至少覆盖论文的 `{0.1, 0.2, 0.3}`。
- 预估并记录每个候选阈值的 query rate；查询率接近 0% 或 100% 时不得直接进入正式评估。
- 结果必须同时报告 oracle episode 数量、query rate 和自预测准确率。
- ATENA 属于消费二元 episode 反馈的方法，不能作为无监督 TTA 报告。

### 更新范围与稳定性

- 论文/官方对齐基线使用 `PARAM_SCOPE=all`。
- 若完整策略更新产生 OOM、NaN、明显灾难性漂移或不可接受时延，停止该 run；不得在同一实验 ID 下静默改成 LayerNorm 或动作头更新。
- 参数高效版本应建立独立配置和实验 ID，并作为 ATENA-AVN 消融报告。

### 显存与运行时间

- ATENA 会保留整条 episode 的计算图。先运行 1、5、20 episode 的递增 smoke test。
- 每个阶段记录峰值显存、episode 自适应时间、是否出现 NaN/Inf，以及参数漂移。
- 任何阶段 OOM 都应先评估等价的梯度重计算方案，而不是改变算法目标。

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

1. Source-sample 与 Source-argmax。
2. 对齐版 ATENA 的 1/5/20 episode 显存与数值 smoke test。
3. MEO+全真实反馈，用于判断混合熵在 AVN 上是否存在正信号。
4. MEO+entropy query，用于判断平均熵是否是有效查询指标。
5. 完整 SAL，用于判断自预测头的增益和错误累积。
6. 通过以上检查后，再进行学习率、`lambda`、`delta` 和参数范围搜索。
