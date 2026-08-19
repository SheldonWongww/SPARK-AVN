# NavTTA 方法选择、分类体系与 CVPR 2027 实验规划

更新日期：2026-08-11
目标：在不立即承诺工程落地的前提下，为具身导航测试时适应建立一个可审稿、可扩展、可复现的 10–12 方法比较体系，并尽可能完整地规划正文与附录实验。

## 0. 执行摘要

### 0.1 最重要的决策

1. **推荐主比较池采用 11 个 TTA，而不是为了形式硬凑 12 个。** Source 不计入 TTA 数量，Ours 暂不计入已有方法数量。
2. 三类方法按“测试时主要证据来自哪个时间尺度”划分：
   - A 类：单步预测证据与可靠性控制，3 个；
   - B 类：持续流、记忆与稳定性控制，4 个；
   - C 类：episode 结果或环境反馈，4 个。
3. 推荐的 11 个方法为：
   - A：Tent、EATA、SAR；
   - B：FSTTA、EAM、CoTTA、RoTTA；
   - C：FeedTTA、ATENA、BiTTA、PDF。
4. 当前已实现的 5 个方法自然分布在三类中：
   - A：Tent；
   - B：FSTTA、EAM；
   - C：FeedTTA、ATENA。
5. 需要新增 6 个实现即可达到 11 个：EATA、SAR、CoTTA、RoTTA、BiTTA、PDF。
6. 如果最终必须做 12 个已有方法，第 12 位不要提前固定：
   - 若 Ours 是 adapter/parameter-efficient 路线，加入 ViDA；
   - 若 Ours 是 reconstruction/feature alignment 路线，加入 TTT-MAE 或 PEA；
   - 若 Ours 强调 VLN prompt/moment alignment，加入 IDEA；
   - 不建议仅为凑数加入与导航协议明显不一致的方法。
7. **C 类不能与 A/B 类在同一监督口径下争夺“最佳方法”。** C 类消费二元成功、累计 reward、主动查询或其他任务反馈，必须单列结果或至少独立分组、独立加粗。
8. 论文的中心实验结论不应是“某个方法在某个模型上最高”，而应是：
   - 无反馈 TTA 能否跨任务、跨模型稳定增益；
   - 使用流历史的稳定方法能否减少单步 TTA 的错误累积；
   - 任务反馈在相同查询、环境交互和计算预算下是否值得其额外成本。

### 0.2 推荐的三类 11 方法体系

| 类别 | 已实现 | 建议新增 | 数量 | 主要研究问题 |
|---|---|---|---:|---|
| A. 单步预测证据与可靠性 | Tent | EATA、SAR | 3 | 当前动作分布本身能否提供可靠的无标签适应信号？ |
| B. 持续流、记忆与稳定性 | FSTTA、EAM | CoTTA、RoTTA | 4 | 连续观测、历史样本、teacher/anchor 能否抑制错误累积与遗忘？ |
| C. episode 结果与环境反馈 | FeedTTA、ATENA | BiTTA、PDF | 4 | 延迟且稀疏的成败反馈能否比无反馈信号更有效地修正策略？ |

这个分类是用于组织实验的“主标签”，不是声称方法之间完全没有交叉。例如 FSTTA 的优化目标仍是熵，CoTTA 也使用当前样本的 teacher target；它们被放在 B 类，是因为论文的主要贡献和实验变量是跨步/跨流稳定机制，而不是新的单步损失。

## 1. 研究边界与统一问题定义

### 1.1 具身导航 TTA 的严格因果定义

设预训练导航策略为 `πθ(a_t | h_t, o_t, g)`，其中：

- `o_t` 是当前视觉、深度、音频等观测；
- `h_t` 是当前 episode 内的历史状态、地图或 recurrent memory；
- `g` 是语言指令、声音目标或对象目标；
- `a_t` 是动作或当前可导航候选节点。

严格在线协议应满足：

1. 动作 `a_t` 必须由更新前的模型产生；
2. 由当前观测构造的无监督更新只能影响 `t+1` 及之后的动作；
3. episode 结果反馈只能在 episode 结束后到达，只能影响之后的 episode；
4. 不允许读取最短路、真实距离、目标可见性、成功标签或 evaluator 内部状态，除非该信号明确计入反馈预算；
5. policy hidden state、地图状态与 TTA 参数状态必须分别定义 reset 规则；
6. hidden test 若不提供在线反馈，C 类不能通过读取隐藏 ground truth 模拟部署反馈。

### 1.2 为什么导航不能简单当作图像分类

分类 TTA 的常见假设在导航中会被系统性破坏：

- **闭环分布变化**：适配后的动作会改变后续观测，错误更新会主动把智能体带入更差的数据分布；
- **强时序相关**：相邻帧和同一房间内 episode 高度重复，不是独立同分布样本；
- **动态动作空间**：VLN 每一步候选 waypoint 数量和语义不同，固定类别 prototype 或历史概率向量不能直接复用；
- **真实动作先验不平衡**：`forward`、`stop`、转向或候选节点天然不均衡，分类方法的 class-balanced memory 可能人为扭曲策略；
- **多模态语义敏感增强**：水平翻转会改变左右语义，音频翻转会改变声源方位，语言 token 删除可能改变目标；
- **episode 级任务目标**：单步低熵不等于最终成功，短期自信和长期导航正确性可能负相关；
- **状态不可重放**：历史 policy input 可能依赖地图、外部 memory、模拟器状态和 recurrent hidden state，单独缓存 RGB 并不足以复现当时决策。

因此，所谓“复现某个 TTA”至少包括三个层次：

1. 忠实保留原方法的主要证据和优化机制；
2. 对动态候选、mask、多模态增强和状态重放做导航化定义；
3. 把所有偏离官方分类实现的改动记录为 port，而不是仍称完全原版。

## 2. 两层分类体系

### 2.1 第一层：用于主表排版的三类

#### A 类：单步预测证据与可靠性控制

主要利用当前 step 的动作分布、熵、sharpness 或可靠性筛选，不依赖 episode 结果，也不以长期 memory 为主要贡献。

适合回答：最小改造、最低状态成本的 TTA 在导航中是否已经足够？

#### B 类：持续流、记忆与稳定性控制

主要利用跨 step/episode 的梯度、参数轨迹、EMA teacher、source anchor、历史样本或时间感知 memory。

适合回答：导航数据的时序性究竟是负担，还是可以被 TTA 利用的信号？

#### C 类：episode 结果与环境反馈

明确消费成功/失败、累计 reward、主动人工反馈或可验证任务结果。它们不是严格无监督 TTA，而是 feedback-supervised deployment learning。

适合回答：在允许有限交互监督时，多少反馈足以超过无反馈方法，以及每次反馈值多少钱？

### 2.2 第二层：每个方法都要记录的正交标签

只用三类会隐藏重要差别。建议论文方法表为每行同时登记以下标签：

| 轴 | 推荐取值 |
|---|---|
| 测试时监督 | `unlabeled` / `pseudo-label` / `binary feedback` / `dense reward` / `human query` |
| 源准备 | `none` / `source weights only` / `source statistics` / `source calibration set` / `meta-training` |
| 更新对象 | norm affine / full policy / adapter / prompt / action head / feature state / non-parametric memory |
| 更新粒度 | action step / fixed window / episode / domain transition |
| 历史状态 | none / EMA / replay buffer / prototype / fast-slow weights / success memory |
| reset | per sample / per episode / per scene / per domain / never |
| 动作协议 | argmax / stochastic sample / multi-sample voting / planner rerank |
| 额外环境使用 | none / extra rollout / retry / reset / reference video |
| 计算 | forward 数、backward 数、峰值显存、buffer、每千 episode GPU 小时 |
| 移植性质 | native navigation / faithful port / navigation-inspired variant / task-specific reconstruction |

Ours 最终可以有一个主类别，同时拥有多个副标签。例如一个“使用 trajectory memory、偶尔查询 episode 反馈”的方法，主类别可以是 B，副标签中注明 active binary feedback；也可以在方法设计上明确拆成 `Ours-U` 和 `Ours-F` 两个协议版本。

## 3. 推荐主比较池：11 个方法逐项分析

### 3.1 A 类：单步预测证据与可靠性

#### Tent（已实现）

- 核心：最小化当前动作分布熵，通常只更新 normalization affine 参数。
- 价值：最简单、最经典、开销低，是所有后续稳定化方法必须超过的下界。
- 导航风险：模型可能对错误动作低熵；变量动作数使原始熵尺度变化；GN/LN 模型并不等同原始 BN-Tent。
- 报告要求：明确写 `Tent-BN/LN/GN` 的实际参数集合；VLN 应先 mask 非法候选，并同时报告原始熵和 `H(p)/log|A_t|`。

#### EATA（建议新增，条件推荐）

- 核心：低熵可靠样本筛选、与历史预测相似度的冗余过滤，以及 Fisher/EWC 式 anti-forgetting 正则。
- 为什么值得：能直接回答“对所有步盲目做 Tent 是否是问题”；也是 FSTTA 论文已经采用过的比较方法。
- 协议陷阱：完整 EATA 通常需要约 2,000 个干净 ID/source-like 无标签样本估计 Fisher；不做 Fisher 的版本更准确地应称 ETA。
- VLN 风险：动态候选维度使历史平均概率向量不可直接定义；应在 candidate-conditioned embedding 或固定语义 action head 上定义冗余度。
- AVN/ObjectNav：固定动作空间较容易；但连续轨迹高度相似，冗余过滤可能跳过大部分 step。
- 主表建议：同时提供 `ETA` 与 `EATA` 的协议标签，若没有合法 source calibration artifact，不能把 ETA 冒充 EATA。

#### SAR（建议新增，A 类第一优先级）

- 核心：可靠熵筛选、SAM sharpness-aware 两阶段更新、熵 EMA 崩溃检测和 source reset。
- 为什么值得：原论文专门研究 batch size 1、混合 shift、动态 wild stream 和 label imbalance，与在线导航最接近。
- 优点：适配 BN/LN/GN affine，相比依赖固定 BN 统计的方法更容易覆盖 Transformer 和 Habitat policy。
- 风险：两次 forward/backward 增加动作延迟；导航策略本来就可能在接近目标时极低熵，固定 reset 阈值可能误判崩溃。
- 移植要求：按有效动作数归一化可靠性阈值；记录 SAM 第一次/第二次 forward 是否使用完全相同的 recurrent/map state。

### 3.2 B 类：持续流、记忆与稳定性

#### FSTTA（已实现）

- 核心：在动作窗口内对梯度做分解和一致方向聚合形成 fast update；跨 episode 对参数轨迹分解形成 slow update。
- 价值：直接面向在线 VLN，建立了“短期适应 + 长期稳定”的导航 TTA 基准。
- 局限：更新信号仍来自动作熵；固定 fast/slow 周期可能与不同 episode 长度和连续控制频率不匹配。
- 实验重点：fast-only、slow-only、dynamic LR、窗口长度、episode 顺序、参数漂移和真实动作延迟。

#### EAM（已实现）

- 核心：冻结 source branch、训练 auxiliary branch；用 reservoir/replay 重算历史 step，通过双分支决策保护源知识。
- 价值：是少数直接面向 source-free online VLN 的 replay 方法，能检验 memory 是否比单步梯度更稳定。
- 风险：需要缓存完整 policy input 和外部状态；双模型显存明显高；伪标签可靠性仍受 source 错误影响。
- 实验重点：memory size、replay batch、更新间隔、gate、current-only 冷启动、source/aux 混合系数、memory 年龄分布。

#### CoTTA（建议新增，canonical CTTA）

- 核心：student、EMA teacher、source anchor，多增强 teacher soft target，以及随机 source restoration。
- 为什么值得：它是最具代表性的 continual TTA，且已经在 FSTTA 论文中被导航化比较。
- 资源代价：通常要保存三份模型；低置信样本可能需要多次增强；原版倾向更新全模型。
- 导航风险：分类增强不能照搬。视觉水平翻转、音频通道交换、方向词处理都会改变动作语义；必须设计语义保持增强。
- 公平做法：正文报告 method-native scope；另做与 FSTTA 相同参数量/相同 backward 数的 cost-matched 版本，不能只比较一个人为弱化版本。

#### RoTTA（建议新增，B 类第一优先级）

- 核心：robust BN、EMA teacher、按 timeliness 和 uncertainty 管理的 memory、周期性 teacher-student 更新。
- 为什么值得：原设定明确面向 correlated practical test streams；导航恰好是最强相关的数据流之一。
- 风险：官方 class-balanced memory 假设固定类别且真实类别应平衡；导航动作天然不平衡，强行平衡转向/STOP 可能有害。
- 移植建议：至少比较三种 memory key：固定动作、candidate-conditioned state-action embedding、无类别平衡 FIFO/time-aware memory。
- 模型限制：官方 robust BN 对 LN/GN 主体不适用；需要把“memory/teacher 核心机制”与“RBN 组件”分别做 native 和 port ablation。

### 3.3 C 类：episode 结果与环境反馈

#### FeedTTA（已实现）

- 核心：逐 step 累积 sampled-action score gradient，episode 结束后用成功/失败决定 REINFORCE 方向，并用 SGR 抑制非平稳更新。
- 价值：与导航 episode 的成败结构天然一致，也是 C 类最直接的基础方法。
- 公平要求：必须和使用相同 sampling seed/动作协议的 frozen Source 比较；不能和 argmax Source 直接计算增益。
- 关键实验：反馈噪声、反馈延迟、成功/失败不对称、折扣因子、SGR 比例、长轨迹信用分配。

#### ATENA（已实现）

- 核心：mixture entropy optimization；高不确定 episode 查询外部二元反馈，低不确定 episode 使用 learned self-prediction。
- 价值：把反馈预算和主动查询纳入 TTA，适合形成完整的 budget–performance 曲线。
- 风险：查询策略本身使用熵，self-success head 可能在新任务上失准；REVERIE 的反馈究竟是导航成功还是完整 grounding 成功必须预注册。
- 关键实验：random query、entropy query、diversity query；无 SAL、无 MEO、只成功/只失败；query calibration 和每次查询收益。

#### BiTTA（建议新增）

- 核心：二元环境反馈和高置信预测一致性形成双路径适应，研究只有“对/错”信息时的 TTA。
- 为什么值得：它把 binary feedback 从导航专用方法提升为一般 TTA 问题，可作为 FeedTTA/ATENA 的跨领域理论参照。
- 导航适配：episode success 可以作为自然 binary signal；但必须把原任务的单样本反馈定义转换为 trajectory feedback，并明确信用分配。
- 风险：如果把每个 episode 的最终成功复制给所有 step，会引入强标签噪声；需要和 trajectory-level objective、last-k credit、eligibility trace 比较。

#### PDF（建议新增，VLA 代表方法）

- 全名：Test-Time Perturbation Tuning with Delayed Feedback for Vision-Language-Action Models。
- 核心：冻结 base VLA，但在 episode 后用 reward/success 更新约 9M 参数的 perturbation head；episode 内通过不确定性分配增强预算并投票。
- 重要澄清：它不是“零参数更新”；准确描述是“不微调 base VLA，但更新附加 head”。
- 为什么值得：是直接面向 VLA 的 delayed-feedback TTA，和 FeedTTA 形成“更新主策略参数 vs 更新轻量 head”的强对照。
- 导航移植：将固定 action-token perturbation head 改成 candidate-conditioned action scorer；AVN/ObjectNav 固定离散动作更容易。
- 风险：多增强投票的计算较重；如果 benchmark 不提供实时 episode outcome，只能在明确的模拟 oracle 协议下评估。

## 4. 为什么推荐 11，而不是立刻固定 12

11 个方法已经形成 3/4/4 的完整结构，而且新增实现数量为 6，能覆盖：

- 基础熵最小化；
- 可靠性筛选和 sharpness；
- gradient/parameter temporal aggregation；
- replay memory；
- EMA teacher 和 source restoration；
- time-aware memory；
- 被动 binary feedback；
- 主动反馈；
- 通用 binary-feedback TTA；
- VLA delayed-feedback head adaptation。

这比增加一个机制高度重复、协议又不完整的方法更有价值。第 12 位应该服务于 Ours 的科学定位：

| Ours 的主机制 | 第 12 方法 | 原因 |
|---|---|---|
| 高/低秩 adapter、parameter-efficient update | ViDA | 对照 adapter 容量分工与 teacher consistency |
| masked modeling、重建或自监督预训练 | TTT-MAE | 对照真正非熵的测试时训练信号 |
| feature/moment alignment、无反传适应 | PEA | 对照只改 embedding、不改模型权重的路线 |
| prompt、source statistics、跨域资产复用 | IDEA | 对照 VLN 原生 soft prompt 和 asset memory |
| contrastive memory/prototype | AdaContrast 或 GOLD | 对照表示空间 memory 和低秩子空间 |

若 Ours 尚未定型，先实现 11 个，等方法确定后再选择第 12 个，能避免主表结构反过来绑架方法设计。

## 5. 其他候选方法：适合什么、不适合什么

### 5.1 高价值候选，但暂不进入统一主池

| 方法 | 主要信号 | 导航研究价值 | 不进 11 方法主池的原因 | 推荐位置 |
|---|---|---|---|---|
| ViDA | 高/低秩 adapter、EMA teacher、一致性 | adapter 专题很有价值 | 需要侵入式注入和专用 source checkpoint；公开源初始化流程不完整；计算重 | 第 12 可选；adapter 消融 |
| TTT-MAE | masked reconstruction | 真正非熵；RGB/depth/audio token 都可构造 | 需要兼容 decoder 和 source-stage 准备，不能直接套预训练策略 | challenge baseline；自监督附表 |
| PEA | source/target embedding 几何对齐，无反传 | 参数零更新、低延迟，对大策略有吸引力 | 需要 source statistics；动作 head 与中间表示的对应关系需重定义 | 效率/representation 附表 |
| IDEA | soft visual prompts、moment matching、asset library | 直接的在线 VLN 2026 方法 | 只对 VLN 原生；需要 source samples/statistics；AVN/ObjectNav port 不自然 | VLN 专项强基线 |
| GOLD | classifier-sensitive low-rank subspace、自训练和 prototype contrast | 低秩、高效 CTTA；适合分析 adaptation subspace | 固定 classifier/类别语义对动态 VLN 候选不成立 | 低秩/效率消融或第 12 位 |
| AdaContrast | contrastive queue、近邻精炼伪标签 | episode 相邻帧和重复场景可提供正样本 | 固定类别和 memory 成本；需要设计导航正负样本 | 表示学习附表 |
| RMT | robust mean teacher、contrastive source prototypes | 可作为 CoTTA/RoTTA 的 teacher 系补充 | 与主池机制重复，source prototype 对动态候选困难 | teacher 系消融 |

### 5.2 导航原生或相邻工作

#### DAVIS

`Anticipating the Unseen Discrepancy for Vision and Language Navigation` 使用 momentum contrast 和相似语义观察的一致性，说明非熵信号能够用于 VLN。它的重要性主要在相关工作和设计启发：原协议更接近两阶段/semi-supervised adaptation，需要训练期配合，并不是当前 NavTTA 的完全 post-hoc online 设置。可作为“训练准备允许时的 VLN 上界”，不宜与无需重训的 Tent/FSTTA 直接等价排名。

#### Global Map Consistency

通过 round-trip trajectory 的全局地图一致性自监督适配，具有很强的导航结构先验。但它需要主动收集适配轨迹、额外环境交互和较长优化，更接近 self-supervised domain adaptation。适合用来设计 map-cycle consistency 消融，不适合放入被动单遍在线 TTA 主表。

#### Search-TTA

使用目标检测正负证据和空间 Poisson point-process likelihood，原生面向 UAV visual search，并支持图像、文字、声音 query。它对目标搜索型 AVN/ObjectNav 很有启发，但其 detector-derived signal、episode 内 reset 和额外搜索协议无法自然覆盖指令跟随 VLN。建议：

- 在 AVN/ObjectNav 专项表作为 task-feedback 方法；
- 若能定义同等 detector/环境信息预算，可替代 PDF 成为 C 类第 4 项；
- 不应为了填满跨任务表而在 R2R 上构造虚假的检测反馈。

### 5.3 轻量方法为何不优先

#### T3A

维护低不确定度 support/prototype，不反向传播，适合固定类别空间。AVN/ObjectNav 若使用固定 `forward/left/right/stop` 动作可做轻量对照；VLN 动态候选没有跨 step 稳定的类别中心，因此不推荐成为三任务统一主基线。

#### DUA

只更新 BN statistics，计算极低。但 SMT+Audio、ENMuS、DUET、GOAT、ETPNav、BEVBert、PIRLNav 等模型经常以 LN/GN 为主；没有 BN 时方法退化为 N/A。它适合出现在“归一化结构兼容性表”，不适合占用 11 个主方法名额。

#### LAME

不更新参数，通过 Laplacian 图目标联合调整一批固定标签空间输出。严格 batch-1 因果流和动态候选 VLN 与其假设冲突；若使用滑动窗口，会引入决策延迟和跨 step 标签不一致。更适合分类或固定动作 AVN 的辅助实验。

### 5.4 高成本但可形成未来研究方向

#### Diffusion-TTA

用 conditional diffusion likelihood 给判别模型提供生成式反馈，是真正非熵 TTA，但需要源域训练的 diffusion model，并且多步生成远重于导航实时预算。可以作为“生成模型反馈 ceiling”，只在少量 keyframe/anchor setting 运行，不建议全矩阵复现。

#### ViTTA

视频动作识别中，同一 clip 的时间采样视图共享一个类别；导航中最优动作随时间变化。因此直接要求不同时间采样得到相同动作分布通常是错误假设。可保留其“特征统计对齐”部分用于感知 encoder，但那应明确称 ViTTA-inspired，而不是完整 ViTTA。

### 5.5 VLA/机器人方法的准确定位

| 方法 | 测试时是否更新参数 | 信号 | 对 NavTTA 的启发 | 定位 |
|---|---:|---|---|---|
| PDF | 是，更新附加 perturbation head | episode reward/success | 轻量 head、延迟反馈、增强投票 | C 类主候选 |
| EVOLVE-VLA | 是 | learned dense progress | 比 binary feedback 更细的信用分配 | feedback ceiling；非无监督 |
| VITA | 是，只更新 value adapter | meta-learned reconstruction | progress/value/STOP/waypoint reranking | 相邻工作；非 action-policy TTA |
| WAM-TTT | 是，更新 adaptive memory | human video prediction | 自监督人类视频作为部署信号 | 需专门训练，非 post-hoc baseline |
| RoboTTT | 是，fast weights | sequence-action forcing | 用权重压缩长上下文 | 需从头按其 recipe 训练 |
| Retrieve-then-Steer | 否，维护 success memory | reference video + progress critic | 成功轨迹检索和非参数适配 | 单列 non-parametric adaptation |
| TTRV | 是，VLM test-time RL | 多次采样频率和熵 reward | 无标签 test-time RL | VLM 相邻；尚非具身策略 |
| MG-Select | 否 | verifier-free best-of-N | 多动作采样选择 | test-time scaling，不是 TTA |
| RoboMonkey | 否 | sampling + verifier | 动作候选验证 | test-time scaling，不是 TTA |
| GPC | 否 | predictive world model | look-ahead 与候选动作 rerank | inference-time planning，不是 TTA |

MG-Select、RoboMonkey、GPC 可以作为论文相关工作中的边界说明，或组成单独的 inference-time compute 对照，但不能和 Tent/FSTTA 用同一“更新方法”标签。

## 6. 三任务适配性矩阵

标记含义：`H` 为可自然定义；`M` 为需要显著 port 或协议说明；`L` 为核心假设冲突。它表示研究适配性，不代表最终一定能获得正增益。

| 方法 | VLN 离散候选 | VLN 连续 | AVN | ObjectNav | 最大接口风险 |
|---|:---:|:---:|:---:|:---:|---|
| Tent | H | H | H | H | 动作数归一化与 LN/GN scope |
| EATA | M | M | H | H | 动态概率向量、Fisher calibration |
| SAR | H | H | H | H | SAM 双 forward 状态一致性、reset threshold |
| FSTTA | H | H | H | H | 窗口长度和动作频率跨任务缩放 |
| EAM | H | M | H | H | 完整 policy state replay 和双模型显存 |
| CoTTA | M | M | M | M | 语义保持多模态增强、三模型资源 |
| RoTTA | M | M | H | H | fixed-class balanced memory、RBN 依赖 |
| FeedTTA | H | H | H | H | sampled control、长轨迹信用分配 |
| ATENA | H | H | H | H | 成功定义、self-feedback calibration |
| BiTTA | M | M | H | H | 单样本 binary feedback 到 trajectory 的转换 |
| PDF | M | M | H | H | VLA action-token head 到 candidate scorer |

### 6.1 VLN 特有规则

1. 所有熵、KL 和置信度只在合法候选及 STOP 上计算；
2. EATA/RoTTA/T3A 的固定类别 memory 不能直接用 waypoint ID；
3. candidate memory 至少应包含 candidate embedding、当前 state embedding、instruction-conditioned score 和 action mask；
4. REVERIE 反馈需要区分：导航 success、object grounding success、完整 remote grounding success；
5. 离散 R2R/REVERIE 与连续 R2R-CE 不共享动作频率超参数，除非先按物理距离或决策次数归一化。

### 6.2 AVN 特有规则

1. 视觉增强不能改变声源相对方位；音频增强需保持目标身份与方位标签；
2. 单声源到多声源属于结构性条件变化，不只是噪声 severity；
3. 多声源条件应报告 target/distractor confusion，而不仅是平均 SPL；
4. 固定动作空间使 EATA/SAR/RoTTA/BiTTA/PDF 更容易成为第一批新增 pilot；
5. feedback 方法需要明确 success 是到达目标声源、选择正确声源，还是只到达任一声源。

### 6.3 ObjectNav 特有规则

1. STOP 语义、目标可见性与 success distance 必须由 evaluator 定义，不能让适配器读取；
2. 按目标类别报告 macro average，避免 chair 等高频类别主导总体结果；
3. GN policy 上原版 BN-Tent/DUA 可能 N/A，不能偷偷改名后仍声称原版；
4. 模块化 mapping/planning 系统若没有统一可微动作 head，不适合作为所有 weight-TTA 的载体。

## 7. ObjectNav 的 benchmark 决策冲突

本次设想提出“MP3D、两个导航模型”。但当前工作区已记录的正式计划是：

- Habitat ObjectNav 2022；
- `objectnav_hm3d_v1` / HM3DSem v0.1；
- PIRLNav 和 ZSON；
- 当前仍处于 isolated/reference-only 状态。

因此未来实施前必须显式做出一次 benchmark decision，不能在论文表格里把 MP3D 与 HM3D 混写。

| 方案 | 优点 | 风险 | 建议 |
|---|---|---|---|
| 保留 HM3D + PIRLNav/ZSON | 更接近现代 Habitat ObjectNav；与当前调研一致 | checkpoint/重训和旧 Habitat 栈成本高 | 作为 ObjectNav 主结果更稳妥 |
| 改成 MP3D + 两模型 | 可与 AVN 共用 MP3D 场景，形成跨任务同场景分析 | ObjectNav 协议更旧；需要重新选模型、split、checkpoint | 只有在“统一场景跨任务 TTA”成为核心卖点时采用 |
| HM3D 主表 + MP3D 附加 | 兼顾现代 benchmark 与跨任务同场景诊断 | 计算量最大 | 若资源允许，最适合“实验非常详实”的目标 |

一个有潜力的 CVPR 故事是：HM3D 验证 ObjectNav 的现代外部有效性；MP3D 只做 AVN/ObjectNav 同场景 cross-task shift 诊断。这样不会牺牲主 benchmark 的时代性，又能利用 MP3D 的跨任务对齐价值。

## 8. 全论文实验宇宙

### 8.1 建议固定的 12 个正式 setting

“setting”定义为一个数据集、导航模型和场景条件的组合，不等于一张表。

| 任务 | Benchmark/条件 | 模型 | setting 数 |
|---|---|---|---:|
| VLN | R2R | DUET、GOAT | 2 |
| VLN | REVERIE | DUET、GOAT | 2 |
| VLN | R2R-CE | ETPNav、BEVBert | 2 |
| AVN | 单声源 | SMT+Audio、ENMuS | 2 |
| AVN | 多声源 | SMT+Audio、ENMuS | 2 |
| ObjectNav | 待最终决定的 MP3D 或 HM3D 协议 | 两个异构模型 | 2 |
| 合计 |  |  | 12 |

这个矩阵的价值不只在“数量多”，而在三个正交维度：

1. 任务目标：语言路径、声源搜索、对象搜索；
2. 环境控制：离散图导航与连续物理导航；
3. 架构：图/Transformer、recurrent policy、地图/BEV 或端到端策略。

### 8.2 全矩阵与锚点矩阵

如果每个 TTA 都在 12 setting、5 seeds 上运行，11 个方法加 matched Source 已经至少是 `12 × 12 × 5 = 720` 次正式运行，尚未计算超参数搜索、消融和 robustness。建议预注册两层矩阵：

- **全方法锚点矩阵**：每个任务选择一个 canonical model，共 3–4 个 setting，运行全部 11 个方法；
- **跨模型扩展矩阵**：第二模型只运行 Source、每类最强 1–2 个方法、Ours；
- 若计算允许，再补齐全部格子；缺失格必须由预注册兼容性/资源规则决定，不能根据结果好坏选择性补齐。

但如果论文的核心卖点就是“统一 TTA 在三类导航任务和多模型上普适”，则应把完整 12-setting 矩阵作为最终目标，把锚点矩阵仅用于开发阶段。

## 9. 主结果表如何组织

### 9.1 正文推荐方案：四张任务表 + 一张协议表

#### Table 1：方法假设与资源

列建议为：

`Category | Signal | Source preparation | Feedback | Update unit | State/memory | Trainable scope | Extra rollout | Native/Port`

这张表必须放正文，因为它决定所有结果是否可公平解释。

#### Table 2：离散 VLN

R2R 和 REVERIE 放在同一张表的两个 panel：

- R2R：DUET、GOAT，各报 SR/SPL；
- REVERIE：DUET、GOAT，各报 RGS/RGSPL，SR/SPL 放附录；
- A/B 类与 C 类用粗横线分隔；只在各自监督组内加粗。

#### Table 3：连续 VLN

R2R-CE：ETPNav、BEVBert，各报 SR/SPL；次指标 OSR、NE、碰撞、动作数放附录。连续环境单独成表是合理的，因为动作频率、waypoint predictor 和物理碰撞会改变 TTA 的更新次数与延迟含义。

#### Table 4：AVN

同一张表做 single-source 与 multi-source 两个 panel：

- 行：Source、11 个 TTA、Ours；
- 列：SMT+Audio/ENMuS 的 SR、SPL；
- 主表只放 SR/SPL，SoftSPL、DTG、NDTG、Reward、NA、SNA、SWS 放附录；
- multi-source 必须显示相对 single-source 冻结超参数的迁移结果。

#### Table 5：ObjectNav

两个模型，各报 Success/SPL；SoftSPL、DTG、碰撞、步数和 goal-category macro 放附录。若同时做 HM3D 与 MP3D，则正文只保留一个主 benchmark，另一个做附录 cross-scene/task transfer。

### 9.2 每个结果格的格式

推荐：`mean ± SD (Δ matched Source)`。

- `Δ` 必须相对完全匹配的 Source，而不是引用原论文 Source；
- FeedTTA sampled policy 只能和 sampled Source 比；
- 多增强/投票方法需要 frozen-base + same-voting control；
- 不要对 SPL、RGSPL 等不同含义的指标直接求全局平均；跨 setting 用平均 rank、正增益比例、中位数百分点增益和最差 setting 增益。

### 9.3 是否把 C 类放在同一任务表

两种都可以，但推荐：

- 表内保留 C 类，便于读者看到同一任务上的绝对上限；
- 使用 `†` 标出 feedback-supervised；
- A/B 和 C 分块，只在块内加粗；
- 再单独做一张 feedback budget 表，不能只展示 100% oracle feedback 的最高数字。

## 10. 指标体系

### 10.1 任务指标

| 任务 | 主 endpoint | 次指标 |
|---|---|---|
| R2R | SPL | SR、NE、OSR、nDTW、SDTW、CLS、TL |
| REVERIE | RGSPL | RGS、SR、SPL、grounding accuracy、TL |
| R2R-CE | SPL | SR、OSR、NE、碰撞率、动作数、物理路径长度 |
| AVN | SPL 与 SR | SoftSPL、DTG、NDTG、Reward、NA、SNA、SWS |
| ObjectNav | SPL | Success、SoftSPL、DTG、碰撞率、步数、category macro |

### 10.2 通用成功转移指标

单看平均 SR/SPL 无法区分“修正失败”和“破坏成功”。建议对完全配对的 Source/TTA episode 计算：

- `PSR = P(TTA success | Source success)`：成功保持率；
- `CSR = P(TTA success | Source failure)`：失败修正率；
- `ASR = (PSR + CSR) / 2`：平衡适应成功率；
- `Harm = 1 - PSR`：破坏率；
- `Net correction = P(0→1) - P(1→0)`；
- paired SPL delta 的分布，而不仅是均值。

这组指标特别适合说明：Tent 可能提高平均置信度却破坏原本成功的 episode；feedback 方法可能增加探索并修正更多失败，但同时拉长路径。

### 10.3 TTA 过程指标

每个方法都应统一记录：

- update attempt、accepted update、skip/reset 次数；
- entropy、normalized entropy、loss、pseudo-label confidence；
- gradient norm、update norm、相对 source 参数漂移；
- EMA/source/auxiliary 分支分歧；
- buffer 占用、样本年龄和重放次数；
- query count、feedback type、feedback accuracy；
- p50/p95 inference、adaptation 和 end-to-end latency；
- 峰值 GPU/CPU memory；
- forward/backward 次数和可训练参数比例。

## 11. 公平性与协议控制

### 11.1 必须固定的 16 条规则

1. 所有方法从同一个 checkpoint、数据版本、episode manifest 和 evaluator 启动。
2. 主协议为单进程、单环境、batch size 1、严格因果在线流。
3. 每个 benchmark、split、声源条件和 stream seed 都从 Source checkpoint 新进程启动。
4. 不把 val-seen 适配状态带入 val-unseen 或 test。
5. 当前动作由 pre-update policy 产生，更新只影响未来动作。
6. episode feedback 终止后到达，只影响未来 episode。
7. recurrent/map state 每 episode 按模型原协议重置；TTA 参数是否跨 episode 保留单独声明。
8. Source 必须匹配动作选择：argmax 对 argmax，sample 对同 seed sample。
9. VLN 先 mask 无效候选再计算损失，STOP 定义保持一致。
10. 分类阈值迁移到动态动作空间时同时报告原公式与 `H/log|A|` port。
11. 固定预计算视觉特征的模型不能声称更新了视觉 encoder。
12. EATA Fisher、source statistics、MAE decoder、meta-training 等准备均计入方法假设。
13. 多模态增强必须保持指令、空间方向、目标身份和动作语义。
14. 反馈方法读取的每种环境字段都列入 feedback budget；无监督方法禁止读取。
15. 主表采用 method-native 设置，另做 parameter/backward/memory-matched 对照。
16. 数值发散是算法结果；只有可证明的基础设施错误才能重跑，且必须复用相同 manifest。

### 11.2 Source-equivalence / zero-write parity

每个 port 在正式运行前做零写入审计：

1. 完整执行 loss、forward、backward、memory、gate、query 等控制流；
2. 拦截 optimizer step 和所有 buffer/model 写入；
3. 与 matched Source 比较逐 episode action、trajectory、metric；
4. 对 parameter 和 persistent buffer 计算前后 SHA256；
5. 任何不同都必须解释为动作协议、随机数消耗或状态 mutation，而不是“数值噪声”。

### 11.3 method-native 与 controlled 两种比较

只做统一 6,144 参数比较会弱化 CoTTA/ViDA 等方法；只做官方默认又会让不同方法资源差距巨大。建议正文/附录同时给：

- **Native**：尽可能忠实原论文的参数范围、teacher、增强和 memory；
- **Controlled**：统一可训练参数量、每 step backward 数或峰值 memory；
- 论文主要结论应明确来自哪种协议，不能在两者之间选择对 Ours 最有利的数字。

## 12. 超参数选择与冻结

### 12.1 数据隔离

- 只在 val-seen、独立 development scenes 或明确的 dev split 调参；
- val-unseen、corruption、multi-source transfer 和 hidden test 不再调参；
- 若 benchmark 没有官方 dev/test 分离，按 scene ID 固定哈希切 `val-dev/val-test`，在第一次 sweep 前登记。

### 12.2 搜索预算

- 每个方法、每个 canonical setting 使用相同上限，例如 8–12 个候选；
- 官方配置必须是候选之一；
- 先做 2-episode lifecycle smoke，再做 256-episode stability screen，最后才跑完整 stream；
- 使用 3 个 development order seeds 的均值选 winner，不能挑单 seed 峰值；
- winner 必须满足数值稳定、无 late-stream collapse、无越权反馈等门槛。

### 12.3 超参数迁移实验

这是很适合论文的额外贡献：

1. 在每个任务的模型 A 上调参；
2. 零调参迁移模型 B；
3. AVN 在 single-source 调参，原样迁移 multi-source；
4. 离散 VLN 调参后迁移 R2R-CE，只允许按物理动作频率预注册缩放；
5. 报告 native-tuned、task-shared、global-shared 三种配置的性能差距。

如果 Ours 在 global-shared 设置仍稳定，而基线需要 model-specific 搜索，这会比单点最高分更有说服力。

## 13. 统计设计

### 13.1 正式 seeds

- 全矩阵至少 5 个预注册 stream-order seeds；
- 关键 anchor setting 建议 10 个 seeds；
- 模拟器、动作 sampling、增强、memory sampling 和 optimizer 随机数均从 manifest 派生；
- 所有方法使用成对 episode order，FeedTTA 的 sampling Source 另做成对控制。

### 13.2 置信区间与显著性

推荐对 paired episode delta 做 10,000 次 hierarchical moving-block bootstrap：

1. seed 为最高层；
2. scene 为第二层；
3. scene 内按连续 episode block 重采样，保留时序相关；
4. Source/TTA 始终成对采样。

主检验只预注册两类：

- Ours vs matched Source；
- Ours vs 最强同监督基线。

同一表的多 setting 使用 Holm correction。跨任务汇总使用 Friedman/平均 rank、正增益 setting 数和中位数百分点增益，不把不同指标直接平均成一个“总分”。关键 setting 若有 10 个 seeds，可增加 seed-level exact paired permutation。

### 13.3 需要报告的负面证据

- 最差 seed、最差 scene 和最差 shift；
- `Source success → TTA failure` 数量；
- OOM、NaN、reset storm、memory overflow、deadline miss；
- 未完成组合及预注册原因；
- late-stream collapse 不能用提前停止后的最佳 checkpoint 替代最终结果。

## 14. Ours 的核心消融菜单

由于 Ours 尚未最终确定，下面按可组合模块设计；将来只选择真正存在的组件，不应为了填表制造无意义消融。

### 14.1 信号来源

- entropy；
- reliable entropy；
- teacher consistency；
- masked reconstruction；
- temporal/transition prediction；
- instruction-view 或 goal-observation contrastive；
- binary success/failure；
- learned progress/value；
- self-predicted feedback；
- 以上信号的组合与梯度冲突分析。

### 14.2 稳定性组件

- source anchor；
- EMA teacher；
- fast/slow weights；
- replay buffer；
- timeliness weighting；
- reliable gate；
- stochastic restoration；
- gradient projection/clipping/reversion；
- reset detector；
- 不同组件的 2×2 因子实验，如 `feedback × anchor`、`memory × gate`，验证协同而不只做 leave-one-out。

### 14.3 更新位置

- vision encoder；
- audio encoder；
- language encoder；
- cross-modal fusion；
- recurrent/map/planner；
- action head；
- value/progress head；
- normalization affine；
- last-K blocks；
- LoRA/adapter/prompt；
- 全模型。

建议同时报告实际 trainable tensor 名单、标量数量、梯度覆盖率和每层漂移，而不只写“更新最后几层”。

### 14.4 更新时间和强度

- per-action vs per-episode；
- update interval `{1, 8, 32, 128}`；
- steps `{1, 2, 4}`；
- learning rate log scale；
- buffer/window size；
- continual vs per-scene/per-episode reset；
- short/medium/long episode；
- fixed update schedule vs uncertainty-triggered update。

### 14.5 反馈机制

- 0%、1%、5%、10%、25%、100% oracle budget；
- random、entropy、margin、diversity、expected-change query；
- success-only、failure-only、双向更新；
- immediate、1/5/20 episode delayed feedback；
- 5%、10%、20% label flip；
- missing feedback；
- human、simulator oracle、VLM/LLM oracle、自预测 head；
- 每次查询增益、AUBC 和达到某性能所需查询数。

### 14.6 预算匹配

- 参数量相同；
- backward 次数相同；
- forward/augmentation 次数相同；
- GPU memory 上限相同；
- buffer 字节数相同；
- 每 action deadline 相同；
- 总环境步数和 reset/retry 次数相同。

## 15. 具身分布偏移与鲁棒性实验

### 15.1 stream 结构

- scene-blocked：同一场景连续出现；
- random interleaved：场景随机交错；
- recurring：A→B→A，检验恢复旧域；
- gradual：shift severity 逐渐增加/减少；
- abrupt：突然切域；
- clean→shift→clean：同时测 adaptation、forgetting 和 recovery；
- curriculum vs anti-curriculum：由易到难与由难到易；
- worst-case order：按 Source 失败概率排序，但排序规则只能事后预先固定，不能为某方法定制。

### 15.2 模态 shift

#### VLN

- RGB brightness、blur、noise、JPEG；
- viewpoint feature dropout；
- instruction paraphrase、长度、指代表达；
- landmark/object synonym；
- 视觉 shift、语言 shift 和复合 shift；
- 连续环境中的 actuation slip、odometry drift、waypoint predictor noise。

#### AVN

- SNR、混响时间 T60、RIR domain、音频时间偏移；
- 声源方位扰动、目标/干扰源幅度比；
- 1/2/3+ 声源；
- intermittent sound、静音片段；
- RGB/depth shift 与 audio shift 的单独和复合组合；
- 未见声音类别、未见房间声学。

#### ObjectNav

- RGB/depth corruption；
- depth missing/noise；
- actuation noise、碰撞和定位误差；
- goal-category frequency shift；
- 小目标/遮挡/多实例；
- scene style、家具密度和楼层结构切片。

每种 corruption 建议三级 severity，只让 Source、每类最强基线和 Ours 跑完整鲁棒矩阵，避免成本失控。

### 15.3 长流与遗忘

- 流长度 100/500/1,000/2,000 episodes；
- 重复 domain cycles；
- early/mid/late 性能；
- stream AUC；
- first surpass Source；
- peak-to-final drop；
- clean recovery；
- source-domain retention；
- worst-block 和 CVaR。

## 16. 机制诊断实验

### 16.1 时间曲线

统一画：rolling SR/SPL、entropy、loss、update acceptance、gradient norm、parameter drift、query rate。至少展示：

- Tent：是否发生 entropy collapse；
- SAR：reset 是否和任务失败对应；
- FSTTA：fast/slow update 节点；
- EAM/RoTTA：memory 何时开始产生收益；
- feedback 方法：获得反馈后后续 episode 的 correction/harm 如何变化。

### 16.2 表示与参数分析

- 逐层相对参数漂移；
- source/adapted representation CKA；
- gradient cosine similarity；
- fast vs slow、source vs auxiliary、student vs teacher 距离；
- top singular values/有效 rank；
- adapter/prompt 方向可视化；
- 不同 scene/domain 的更新向量聚类；
- source-domain Fisher importance 与实际漂移相关性。

### 16.3 memory 分析

- memory 样本年龄直方图；
- scene、动作、目标类别、成功/失败组成；
- 被保留/淘汰样本的 uncertainty；
- 同一当前 state 检索到的历史样本可视化；
- 固定动作 memory 与 candidate-conditioned memory 对比；
- memory size–performance–RAM Pareto；
- replay staleness 和 pseudo-label correctness（仅事后使用 oracle 诊断）。

### 16.4 calibration 与风险

- success probability calibration；
- action confidence 与最终失败的 AUROC；
- risk–coverage curve；
- normalized entropy 与动作数关系；
- query calibration：高熵 episode 是否真的更可能失败；
- ECE/Brier score；
- selective update coverage–performance 曲线。

## 17. 效率实验

### 17.1 测量协议

- 同一 GPU、batch 1、单环境；
- CUDA synchronize 后计时；
- 预热后至少 1,000 action steps，重复 3 次；
- 分开报告 simulator、base inference、adaptation 和 episode-end update；
- 报 p50/p95 而不仅是均值；
- 保存峰值 GPU、CPU RAM、buffer、checkpoint 增量和 GPU hours/1,000 episodes。

### 17.2 推荐效率图

- `ΔSPL vs p95 action latency`；
- `ΔSPL vs peak VRAM`；
- `ΔSPL vs trainable parameter ratio`；
- `ΔSPL vs feedback queries`；
- `stream AUC vs total GPU hours`。

不要发明一个把准确率、延迟、显存随意加权的总分；Pareto frontier 更透明。

## 18. 定性与失败案例

### 18.1 避免 cherry-pick

预先规定四类 episode：

1. Source 失败 → TTA 成功；
2. Source 成功 → TTA 保持；
3. Source 成功 → TTA 破坏；
4. 所有方法失败。

每类按固定 seed 和预定义分位数选取，公开 episode ID。不能只展示最漂亮的修正案例。

### 18.2 每类任务的图内容

#### VLN

- top-down trajectory、instruction、候选节点；
- 错误转向、循环、提前 STOP、正确路线但错误终点；
- 动作概率、更新/query 标记；
- REVERIE 增加 grounding box 和 object score。

#### AVN

- target/distractor 方位；
- 声音强度或 spectrogram；
- 轨迹与声源位置；
- target confusion、静音区、反射歧义；
- TTA 更新前后的动作分布。

#### ObjectNav

- 目标实例、可见区域、STOP；
- 碰撞、卡死、超时、找到但未停止；
- goal category 和 scene clutter；
- Source/TTA path overlay。

## 19. 建议的正文图表

在最终 Ours 尚未确定时，可预留如下实验叙事：

| 编号 | 内容 | 回答的问题 |
|---|---|---|
| Fig. 1 | 三任务严格因果在线协议 + 三类证据来源 | 什么是导航 TTA，监督边界在哪里？ |
| Fig. 2 | 12 setting 相对 Source 增益热图 | 哪类方法跨任务最稳定？ |
| Fig. 3 | clean→shift→clean 的性能、更新率、漂移三联图 | 能否适应、抗忘、恢复？ |
| Fig. 4 | ΔSPL–latency/VRAM Pareto | 性能是否值得部署成本？ |
| Fig. 5 | feedback budget–performance/AUBC | 多少反馈足够？ |
| Fig. 6 | VLN/AVN/ObjectNav 轨迹案例 | 方法具体修正了什么，又破坏了什么？ |
| Table 1 | 方法协议与资源 | 比较是否公平？ |
| Table 2 | 离散 VLN | R2R/REVERIE 是否稳定？ |
| Table 3 | R2R-CE | 连续环境是否成立？ |
| Table 4 | AVN | 单/多声源是否迁移？ |
| Table 5 | ObjectNav | 是否跨目标任务泛化？ |

若正文空间不足：保留 Table 1、三张任务主表、增益热图、动力学图和 Pareto；完整 feedback curve 与轨迹大图移至附录。

## 20. CVPR 2027 正文与附录规划

CVPR 2027 的最终格式应以届时 CFP 为准。以下以近年常见的 8 页正文限制作为规划占位，不把尚未发布的规则当作事实。

### 20.1 正文建议页数

| 部分 | 目标页数 | 内容 |
|---|---:|---|
| Introduction | 0.8–1.0 | 闭环分布偏移、三任务统一问题、贡献 |
| Related Work | 0.6–0.8 | 通用 TTA、导航 TTA、feedback/VLA TTA |
| Problem/Protocol | 0.7–0.9 | 因果在线协议、三类 taxonomy、监督边界 |
| Method | 1.5–2.0 | Ours，核心模块与复杂度 |
| Experiments | 3.0–3.6 | 主表、鲁棒性、消融、效率、机制图 |
| Limitations/Conclusion | 0.3–0.5 | 反馈、模拟器、部署边界 |

实验部分不应只堆表。推荐围绕三个 claim，每个 claim 至少有一个主表和一个机制图：

1. 跨任务/跨模型稳定增益；
2. 长流稳定与较低 harm rate；
3. 在给定反馈/计算预算下更优的 Pareto。

### 20.2 附录建议目录

#### A. Reproducibility passport

- 所有 dataset/split/model/checkpoint 版本；
- top-level Git commit、环境、硬件、seed；
- checkpoint、episode stream、数据索引 SHA256；
- run manifest schema；
- hidden test submission ID、时间和 leaderboard 版本。

#### B. 方法实现审计

- 每个上游 repo commit/license；
- 原论文算法与导航 port 的逐项差异；
- trainable keys、参数数、optimizer、reset；
- pseudo-code；
- Source-equivalence 和 zero-write parity；
- 变量动作空间、mask、STOP、replay state 定义。

#### C. 超参数

- 完整搜索空间；
- 每方法候选数和总预算；
- winner 选择规则；
- 冻结配置；
- sensitivity 曲线；
- global/task/model-specific 配置迁移。

#### D. 全结果

- 全模型、全 seed、全 split；
- 所有任务次指标；
- 每 scene/goal category/sound count；
- paired Source/TTA episode transition table；
- N/A/OOM/失败组合。

#### E. 统计

- bootstrap block 长度和层级；
- 置信区间；
- Holm correction；
- exact permutation；
- pilot 方差和功效分析；
- 最小可检测效果。

#### F. 完整消融

- component、signal、scope、memory、update timing；
- 2×2 因子实验；
- parameter/backward/memory matched；
- source preparation ablation；
- method-native vs controlled。

#### G. Shift 和长流

- corruption 生成配方和 severity；
- stream order；
- abrupt/gradual/recurring/clean recovery；
- 100–2,000 episode 长度；
- order robustness；
- worst-group/CVaR。

#### H. 反馈研究

- budget 曲线；
- delay/noise/missing；
- oracle 类型；
- 查询选择；
- 环境步、retry、reset 和人类时间成本；
- hidden test 无反馈限制。

#### I. 效率

- 硬件、warm-up、同步、计时方式；
- p50/p95；
- GPU/CPU/RAM/buffer；
- forward/backward/FLOPs；
- GPU hours；
- Pareto 全图。

#### J. 定性结果与失败案例

- 全部预注册案例；
- Source→TTA 成败转移；
- 轨迹、概率、更新、query；
- 不成功案例和局限性。

#### K. 伦理与部署边界

- 模拟 oracle 与真实人类反馈差异；
- 错误反馈的安全风险；
- 在线参数更新的可恢复性；
- 数据隐私和 success memory；
- 不把 evaluator ground truth 当作真实部署可用信息。

## 21. 建议的复现顺序

### Phase 0：先把当前 5 个做成可发表证据

1. 重跑所有 provisional AVN 结果，补齐 run manifest、checkpoint/data digest；
2. 五种方法都完成 zero-write parity、parameter scope 和 diagnostics 审计；
3. 固定 Source action protocol；
4. 先得到一个任务、两个模型、三 seeds 的可信闭环，再扩任务。

### Phase 1：新增 SAR 与 RoTTA

- SAR：最接近 batch-1 dynamic stream，工程边界清晰；
- RoTTA：最能检验相关 episode stream 和 memory 的价值；
- 优先在 AVN 固定动作空间 pilot，再扩 VLN 动态候选。

### Phase 2：新增 EATA 与 CoTTA

- EATA：先决定完整 EATA 还是 ETA；准备合法 source calibration artifact；
- CoTTA：先做语义保持增强审计和资源测量，再决定 native/controlled scope；
- 用 FSTTA 论文导航化设置作为参考，但不要把未发布 wrapper 当作官方实现。

### Phase 3：新增 BiTTA 与 PDF

- 建立统一 feedback API：类型、时间、预算、噪声、delay；
- BiTTA 需要 trajectory credit port；
- PDF 需要 candidate-conditioned perturbation head 和 frozen-head matched control；
- C 类先只在 val/simulator 评估，不越权读取 hidden test outcome。

### Phase 4：决定第 12 位与 Ours

- Ours 定型后根据机制选择 ViDA、TTT-MAE、PEA、IDEA 或 GOLD；
- 第 12 位应增强论文的反事实比较，而不是增加一行近似重复方法。

## 22. 风险登记与提前止损标准

| 风险 | 观察信号 | 止损/替代 |
|---|---|---|
| 动态候选破坏 fixed-class method | memory/probability 维度不一致 | candidate-conditioned port；仍不成立则标 N/A |
| 语义增强错误 | flip 后动作/目标语义变化 | 只保留 feature noise/dropout 或等变变换 |
| 长流坍塌 | entropy→0、SR 后段下降、频繁 reset | 降强度、anchor/reset；如实报告失败 |
| feedback 越权 | 读取 evaluator hidden fields | feedback API allowlist + audit |
| replay 不可复现 | 历史 state forward 不一致 | 缓存完整 state 或放弃 exact replay claim |
| source preparation 不公平 | 某方法额外用源数据 | 主表标签 + source-free 变体 |
| 全矩阵成本失控 | GPU-hour 超预算 | anchor matrix + 预注册扩展规则 |
| ObjectNav benchmark 漂移 | MP3D/HM3D 混用 | 单独研究决策和 manifest，不混列 |
| hidden test 无反馈 | benchmark 不返回 success | C 类只报合法部署/val 协议 |
| 受限文献或资产被提交 | licensed PDF 出现在 Git | 本目录 PDF 默认 `.gitignore`，只跟踪来源和哈希 |

## 23. 可以形成的论文级结论清单

以下结论都必须由对应实验支持，不能预先写死：

1. **Generality**：Ours 在 12 setting 中有多少个正增益，最差 setting 是否仍不低于 Source；
2. **Stability**：长流末段、clean recovery、harm rate 是否优于 Tent/FSTTA/RoTTA；
3. **Efficiency**：相同 latency/memory/backward budget 下是否更优；
4. **Feedback efficiency**：达到同等 SPL 需要多少查询，AUBC 是否优于 FeedTTA/ATENA/BiTTA/PDF；
5. **Transfer**：一个模型/单声源调参后是否能零调参迁移另一模型/多声源；
6. **Modality awareness**：视觉、音频、语言、融合层 shift 下，方法是否只修复对应组件；
7. **Correction vs harm**：提升来自修正 Source 失败，还是用少量严重退化换取平均增益；
8. **Mechanism**：memory、anchor、feedback 或 representation alignment 的实际作用是否与理论叙事一致；
9. **Protocol honesty**：无反馈和反馈方法的优势边界是否被清楚呈现，而不是混为一个冠军榜。

## 24. 最终建议

主线先锁定 11 个方法：

- A：Tent、EATA、SAR；
- B：FSTTA、EAM、CoTTA、RoTTA；
- C：FeedTTA、ATENA、BiTTA、PDF。

这是目前最兼顾文献代表性、三类均衡、导航适配性和审稿可解释性的组合。ViDA、TTT-MAE、PEA、IDEA、GOLD 不应消失，而应作为根据 Ours 机制选择的第 12 位、专项强基线或附录 challenge baseline。

实验应以 12 个 task/model/condition setting 为目标，正文按离散 VLN、连续 VLN、AVN、ObjectNav 组织四张任务主表，并增加方法协议表。无反馈与 feedback-supervised 方法分组排名；完整的 robustness、长流、反馈预算、效率、机制诊断、统计和失败案例放入正文关键图及系统附录。

最关键的不是把每个格子填满，而是确保每个格子的 Source、监督、动作协议、数据流、参数范围和证据 provenance 都可审计。只要这一点成立，三任务 × 多模型 × 11 方法的实验本身就足以形成一篇非常扎实的具身导航 TTA benchmark/method 论文基础。
