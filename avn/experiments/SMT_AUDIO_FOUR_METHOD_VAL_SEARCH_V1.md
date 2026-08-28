# SMT+Audio 四方法验证集搜索计划 v1

状态：已冻结，等待学校服务器预检与运行。

## 决策

SMT+Audio 的 EAM、FeedTTA、ATENA、IDEA 直接在各自的 `single_source/val`
和 `multi_source/val` 上独立选参。所有方法统一执行任务原生的 categorical
sampling；不新增 Source-argmax，也不在本批次重复 Source。已有同流 Source-sample
结果只作为开发集 SR 下限，当前 provenance 不完整，因此本批结果只能标记为
`validation-selected`，不能直接升级为正式主表证据。

机器可执行定义是
`avn/experiments/smt_audio_four_method_val_search_v1.json`。搜索开始后禁止查看
中间结果追加配置；若没有候选满足选择约束，只报告 Pareto frontier 并停止。

## 规模与调度

| GPU | 方法 | 每个声源候选数 | 默认并发 | 24GB 硬上限 |
|---:|---|---:|---:|---:|
| 0 | EAM | 12 | 4 | 10 |
| 1 | FeedTTA | 15 | 4 | 10 |
| 2 | ATENA-AVN(sample) | 20 | 3 | 10 |
| 3 | IDEA | 12 | 2 | 10 |

每条方法 lane 先完整结束 single-source，再开始 multi-source；四条 lane 彼此并行。
总计 `2 × (12 + 15 + 20 + 12) = 118` 个 TTA 作业。默认并发保持保守值，
运行时可在 `[1,10]` 内显式设置每个方法的并发，不能超过表中上限。

2026-08-28 根据学校服务器 4 张 24GB RTX 3090 上观察到的低显存占用，用户批准将
四条 lane 的调度硬上限统一提高为 10。该数值是操作层面的准入上限，并非经过各方法
峰值显存压力测试得到的安全证明；采用高于默认值的并发时仍需监控 GPU 峰值、主机内存
和模拟器 CPU 压力。搜索网格、数据流、随机性、方法超参数和选择规则均未改变。

## 固定协议

- split：`val`，角色是 development；每个 setting 为固定 seed-0、2000 episode
  stream。
- fresh process：每个候选、每个 setting 都从对应 Source checkpoint 独立启动，
  不跨候选或 single/multi 继承适配状态。
- 动作：全部为 `sample`。EAM/IDEA 无监督；FeedTTA/ATENA 消耗 episode 结束后的
  binary success，属于 feedback-supervised。
- FeedTTA 按真正执行的 sampled action 累积策略梯度；配置中的动作协议会驱动
  实际 selector，而不只是作为诊断标签记录。
- ATENA 官方 VLN 版本使用 argmax；本实验是显式命名的
  `ATENA-AVN(sample)` 任务适配版，以真正执行的 sampled action 构造 mixture
  pseudo expert，使延迟 success/failure 与产生该结果的轨迹一致。共享实现的默认值
  仍为官方 `policy_argmax`，只有 AVN 配置显式传 `sample_from_policy` 时改变。完整
  偏差与验证门见 `ATENA_AVN_SAMPLE_REPRODUCTION_CONTRACT.md`。
- IDEA 的 source anchor 分别从 single/multi 的 source-train 中按固定 manifest
  选择 128 条轨迹，使用同样的 sampled-action 协议；两个 setting 的统计资产严禁
  混用，也禁止 target-val warm-up。

## 搜索范围

- EAM：`LR={3e-9,1e-8,3e-8,1e-7}` ×
  `UPDATE_INTERVAL={64,128,256}`。其余保持论文机制和已审计 AVN 映射。
- FeedTTA：12 个 `LR={1e-8,3e-8,1e-7}` ×
  `gamma={.90,.95,.99,1.0}` 强度点；另在 `3e-8/.95` 上预设
  `p=.1`、`alpha=-.1` 和 `p=0` 三个 SGR/无 SGR 对照，共 15 点。
- ATENA：12 个低学习率对 × entropy threshold 核心点，2 个 `lambda` 点，
  3 个 self-loss 点，及 ETPNav、DUET-R2R、DUET-REVERIE 三个官方学习率锚点，
  共 20 点。
- IDEA：`LR={3e-4,1e-3,3e-3,1e-2}` × `tau={.5,.7,1.0}`；固定
  `L=4,Kmax=32,lambda=.4,Fisher beta=.1,O=50`，共 12 点。

## 选择与冻结

single/multi、方法之间均独立选择。候选必须运行成功、manifest 有效、覆盖完整
2000 episode、指标有限，并满足 `SR >=` 同 setting 的 Source-sample SR；ATENA
还要求 query rate 落在 `[5%,95%]`。在合格候选中最大化 SPL，依次用 SR、
SoftSPL、较低参数漂移和较小 job id 破同分。`no_sgr_control` 仅是机制对照，
不作为 FeedTTA 主方法候选。

由于选参与报告使用同一个 val，最终结果必须写作 validation-selected。后续若需要
正式推断，只能冻结所选配置后在预先声明且未参与选参的 stream/seed 上确认，不能
再根据确认结果改超参数。

## 方法复现偏差账本

| 方法 | 论文/官方 | 本次 AVN port | 分类与影响 |
|---|---|---|---|
| EAM | 在线无监督、记忆重放、辅助分支 | 保留核心机制，映射到 SMT Transformer + action head | 任务映射 |
| FeedTTA | episode feedback + SGR；轨迹写作从策略采样 | 使用实际 sampled action 和延迟 success | 核心一致 |
| ATENA | 伪专家与执行动作均为 argmax | 两者均改为实际 sampled action | 明确命名的任务协议变体，不能宣称动作协议完全复现 |
| IDEA | 冻结基座、prompt 对齐、历史资产与 bridge | 保留核心机制，source/target 轨迹均采用 AVN sample | 任务映射；动作影响域统计分布 |

ATENA 的目标函数、查询门、二元反馈时序、自预测头、episode-end 更新和精确重放
保持不变。IDEA 的源统计、Fisher 权重、bridge、覆盖门和资产库机制保持不变。

## 可审计设计契约映射

本批次沿用 launcher 已执行的 `navtta.avn.smt_audio_val_search.v1` JSON，而不是另建
一份不能直接执行的通用 schema。其审计字段映射如下：

- 生命周期：`spec_id/status/supersedes` 固定本次开发决策；旧实验只保留为历史证据。
- 数据与 horizon：JSON 的 `data.streams` 分别固定 single/multi 的索引、顺序和内容
  SHA256；`episodes=2000`、环境原生每 episode 上限 500 步。
- 随机性：模型、episode 顺序和动作采样均为 seed 0 的 fresh process；FeedTTA SGR
  使用 seed 0 的独立 Torch generator，IDEA 使用独立 CPU generator，EAM replay
  使用已固定的 Python RNG；ATENA 延迟创建 self head 时保存并恢复动作 RNG 状态。
- update dose：EAM 的间隔是每个 action step，FeedTTA/ATENA 每个 episode 最多一次，
  IDEA 每个未覆盖 action step 最多优化 50 步；每项均由候选或固定字段给出。
- Source：复用的同流 sample Source 只作为开发阶段 SR floor；因 checkpoint 来源证明
  尚不完整，不允许把本批结果直接写成 formal result。
- 监督：EAM/IDEA 为无监督；FeedTTA/ATENA 使用 episode-end binary success，必须
  作为 feedback-supervised 方法报告。
- 泄漏控制：single/multi 独立选参；只允许完整 val 网格、预先声明的诊断和 Source
  门槛参与选择；禁止查看中间结果扩网格、跨 setting 继承状态或使用测试集。
- 失败与停止：算法失败保留为结果；只有基础设施错误可在同一 immutable batch、
  同一配置下显式 `--retry-failed`。任一候选缺失时不生成 winner。
