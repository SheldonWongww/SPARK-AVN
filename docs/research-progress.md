# NavTTA 科研进展

> 最后更新：2026-07-21  
> 当前阶段：实验范围已确定，进入 Source 导航基线复现与已有 TTA 方法复现阶段。

## 1. 当前研究目标

研究聚焦于三类具身导航任务：VLN、AVN 和 ObjectNav。现阶段不扩展到具身操作、开放词汇导航或其他具身任务，也暂不开始设计自己的 TTA 方法。

当前里程碑是：在固定 benchmark 和导航模型上完成 Source 基线，并复现 Tent 与四种已有 VLN-TTA 方法，填满主对比表中除 Ours 之外的全部结果。完成充分的超参数实验、结果核验和归档之后，再进入自研方法迭代阶段。

## 2. 已确定的实验范围

| 任务 | Benchmark / 协议 | 场景与仿真环境 | 导航基线模型 |
|---|---|---|---|
| VLN | R2R、REVERIE，离散导航协议 | MP3D 离散导航图；沿用模型官方训练与评估协议 | DUET、HAMT |
| AVN | Semantic AudioGoal Navigation；单声源与多声源设置 | SoundSpaces，MP3D | SMT+Audio、ENMuS |
| ObjectNav | Habitat ObjectNav Challenge 2022 协议，`objectnav_hm3d_v1` | Habitat，HM3DSem v0.1 | PIRLNav、ZSON |

补充说明：

- PIRLNav 使用 `IL-HD -> RL fine-tuning` 路线，属于使用 ObjectNav 示范与奖励训练的策略。
- ZSON 在 ImageNav 上训练，通过多模态目标表示零样本迁移到 ObjectNav；在结果中必须保留 `zero-shot ObjectNav` 标识，不能记为 ObjectNav 监督训练模型。
- VLN 暂不增加第三个导航模型；AVN 暂不增加 SMT+Audio 和 ENMuS 之外的模型；ObjectNav 固定使用 PIRLNav 和 ZSON。

## 3. 已确定的对比方法

每个适用的“任务 × benchmark × 导航模型”组合均需完成以下实验：

1. Source：不进行测试时适应的原始导航策略。
2. Tent。
3. FSTTA。
4. EAM。
5. FeedTTA。
6. ATENA。
7. Ours：待上述对比结果完成后再设计和迭代，当前不实施。

论文和实验表格统一使用 `Tent` 命名。具体模型实际更新的归一化层、参数范围和参数数量记录在运行配置与 manifest 中，不在方法名称上区分 BN、GN 或 LN。

不同方法的监督信息必须忠实保留：使用动作预测熵的方法与使用 episode 成败反馈或人类反馈的方法不能在实验记录中混淆。每次运行都要记录该方法实际消费的测试时信号。

## 4. 当前代码与结果状态

### VLN

- DUET 源码已下载：`references/repos/navigation/vln/VLN-DUET`，当前参考 commit 为 `93e8b233164bc079a6db48b8a0a78d123ec8de41`。
- HAMT 源码已下载：`references/repos/navigation/vln/VLN-HAMT`，当前参考 commit 为 `c8b9ee12125f9fe36c51d2ab928fde38f7d846bd`。
- 当前尚无登记为正式结果的 Source 或 TTA 实验。

### AVN

- SMT+Audio 和 ENMuS 已有本地工作代码。
- SMT/Scene Memory Transformer 参考 commit 为 `cbb62352123ffbcbdeda5e9767c78eb9a8d1e6e2`。
- ENMuS 参考 commit 为 `2cf856cd4c8c604edee232c1180e299ca96a112c`。
- Source checkpoint 已在本地，但仍需补齐来源、哈希和配置 manifest。
- Tent、FSTTA、EAM、FeedTTA 已有原型；ATENA 已完成论文/官方对齐原型，但由于 AVN 长轨迹反馈归因、查询率和显存风险，当前设置了运行门禁，必须按 `avn/experiments/ATENA_PRE_RUN_REVIEW.md` 重新复核后才能启动。现有结果均尚未通过正式实验的可复现性验收。

### ObjectNav

- PIRLNav 源码已下载：`references/repos/navigation/objectnav/pirlnav`，当前参考 commit 为 `8235b0e3b589818441f6783fecd5c4fa8ad53f1b`。
- ZSON 源码已下载：`references/repos/navigation/objectnav/zson`，当前参考 commit 为 `a0415137aeb36dab962b467bdfcfb51dbaa6ed71`。
- 当前尚未登记正式数据、Source checkpoint、训练结果或 TTA 结果。

以上 `references/repos/` 中的仓库只作为上游参考。后续研究性修改应进入对应任务目录，不直接覆盖参考仓库。

## 5. 接下来的执行顺序

1. 为每个模型固定上游 commit、运行环境、数据版本、训练配置和评估协议。
2. 准备数据与 checkpoint manifest，记录来源、版本、许可证、路径和 SHA256。
3. 先完成每个导航模型的 Source 训练或官方权重复现，确保基础指标可信。
4. 为模型接入统一的 logits、feature、action、episode 和反馈接口；验证关闭 TTA 时与原始 Source 等价。
5. 依次复现 Tent、FSTTA、EAM、FeedTTA 和 ATENA。
6. 在固定开发划分上完成学习率、更新频率、参数范围等超参数搜索，并保留全部实验记录。
7. 冻结超参数后运行最终评估，生成逐 episode 结果、汇总指标、时延和显存统计。
8. 填满三个任务中除 Ours 外的主对比表，再开始自研 TTA 方法。

## 6. 正式结果的最低要求

- 同一方法必须使用相同的 Source checkpoint、episode 集合、顺序、动作选择协议和评估指标。
- 每次运行记录 Git commit、配置、数据版本、checkpoint SHA256、随机种子、硬件和实际更新参数。
- 保留逐 episode 结果和完整超参数实验，不只保留汇总数字或 TensorBoard 截图。
- Source 与所有 TTA 方法至少完成多随机种子评估，并报告均值与标准差。
- 在 Source 基线未稳定复现之前，不开始该模型的大规模 TTA sweep。
- 只有带完整 manifest 且可重复的实验才能进入正式对比表；旧结果或来源不完整的结果只能保留在 `results/legacy/`。
