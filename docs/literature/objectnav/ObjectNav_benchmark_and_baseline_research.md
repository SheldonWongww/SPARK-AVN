# ObjectNav benchmark 与导航基线调研

> 调研与链接核验日期：2026-07-20  
> 范围：先为 ObjectNav 的 Source、Tent、FSTTA、EAM、FeedTTA、ATENA 对比实验确定一个可执行起点；不扩展到开放词汇 ObjectNav 或其他具身任务。

## 1. 结论

第一阶段建议固定为：

- **Benchmark**：Habitat ObjectNav Challenge 2022 协议。
- **场景数据**：HM3DSem v0.1。
- **Episode 数据**：`objectnav_hm3d_v1`。
- **导航模型**：PIRLNav 的 `IL-HD -> RL fine-tuning` 策略。
- **动作空间**：离散动作，与 PIRLNav 和现有 VLN-TTA 的动作概率适应方式一致。
- **主指标**：Success Rate（SR）和 SPL；同时保留 SoftSPL、Distance to Goal、推理时延、峰值显存和更新次数。
- **初始范围**：只完成 Source、Tent、FSTTA、EAM、FeedTTA、ATENA；不在这一阶段加入 OVON、PoliFormer、VLFM 或模块化规划器。

这是“最适合当前 TTA 研究”的选择，不等于声称 PIRLNav 仍是 2026 年整个 ObjectNav 领域的最新 SOTA。选择它的原因是：协议成熟、模型端到端且有动作 logits、训练与评估代码均存在、与 Habitat 生态一致，并且能较直接地承载五种待比较的适应方法。

有一个必须提前接受的风险：**PIRLNav 官方发布的两个策略 checkpoint 链接目前均为 404，未找到可验证镜像**。因此不能把“有开源权重”写进项目计划；第一阶段应按“需要重训”排期。若重训成本暂时无法接受，才退回 AllenAct RoboTHOR 2021 作为工程备选，而不应悄悄改变主 benchmark。

## 2. Benchmark 选择

### 2.1 推荐协议：Habitat ObjectNav 2022 / HM3DSem v0.1

官方协议的关键设定如下：

| 项目 | 设定 |
|---|---|
| 仿真器 | Habitat Sim |
| 场景 | HM3DSem v0.1，共 120 个场景 |
| 场景划分 | train / val / test = 80 / 20 / 20 |
| 目标类别 | chair、couch、potted plant、bed、toilet、tv，共 6 类 |
| 传感器 | RGB-D、无噪声 GPS+Compass；具体模型可以只使用其子集 |
| 动作与时限 | PIRLNav 使用离散动作，最多 500 steps |
| Success | STOP 时，在任意目标实例 1.0 m 欧氏距离内，并且该目标可由 oracle 原地转身或抬头/低头看见 |
| 主指标 | SPL；同时报告 Success、SoftSPL、Distance to Goal |

采用该版本而不是立即使用 HM3DSem v0.2 的原因：

1. PIRLNav 的训练代码、示范数据和论文结果均基于 2022/v0.1 协议。
2. v0.1 是成熟的标准 ObjectNav，而非 open-vocabulary 变体。
3. 离散动作策略能提供清晰的 categorical action logits，适合熵最小化和四种 VLN-TTA 方法。
4. 先在一个确定协议上得到完整对比表，比同时兼容 2022、2023 两套协议更重要。

### 2.2 暂不选择的协议

| 协议 | 事实 | 暂不作为第一阶段主 benchmark 的原因 |
|---|---|---|
| Habitat ObjectNav 2023 / HM3DSem v0.2 | 216 场景，145/36/35 划分，6 类目标，Hello Stretch 设定 | 更近期，但与 PIRLNav 2022 的训练、动作和已报告结果不完全同协议；混用会使 Source 复现失去参照 |
| RoboTHOR ObjectNav 2021 | 数据和 AllenAct checkpoint 当前可下载 | 场景较少、协议较旧、不是 Habitat/HM3D 主线；适合作为失败备选，不适合仅因权重可用就替换主 benchmark |
| HM3D-OVON | 数据与 DAgRL checkpoint 当前可获取 | 属于 open-vocabulary ObjectNav，目标空间和研究问题已经变化，应留作后续泛化实验 |
| ProcTHOR-Objaverse / PoliFormer | 现代 Transformer 策略，代码和三个 checkpoint 链接可用 | 使用 Stretch、Objaverse 资产、不同传感器和动作协议，工程与数据负担明显更大 |
| iTHOR | 小型合成单房间场景 | 环境过小，作为主要 TTA 结论的说服力不足 |

**禁止把 HM3DSem v0.1/v1 与 v0.2/v2 的数值放在同一列直接比较。** 两者的场景数量、机器人设定和 episode 协议不同。

## 3. 候选代码库实审

下面的“训练/评估代码”结论来自实际文件检查，不是只依据 README。

| 项目 | 审计 commit | 训练代码 | 评估代码 | 当前权重状态 | TTA 适配性 | 决策 |
|---|---|---:|---:|---|---|---|
| PIRLNav | `8235b0e3b589818441f6783fecd5c4fa8ad53f1b` | IL + PPO 均有 | 有 | 策略权重失效；编码器有镜像 | categorical logits；端到端；高 | **主模型，修复后重训** |
| Habitat-Lab 官方 DD-PPO | `0fb6f43ffe806a8088a171b036336c093bcf604e` | 有 | 有 | 未发现 HM3D ObjectNav 策略权重 | categorical logits；端到端；中高 | smoke test / 故障对照，不作为最终强基线 |
| AllenAct ObjectNav | `d055fc9d4533f086e0340fe0a838ed42c28d932e` | DD-PPO + DAgger | 有 | RoboTHOR 2021 权重实测可下载 | categorical logits；端到端；高 | PIRLNav 重训无法推进时的备选 |
| HM3D-OVON | `8300fcc9fcd820637ac202cb43db080229f53410` | PPO + DAgger | 有 | DAgRL 权重可下载 | 可适配，但任务是 OVON | 后续泛化，不进第一阶段 |
| PoliFormer | `c9442d969d6fda0dac775ab4dd3f22a9fcf47d57` | on-policy PPO | 有 | 3 个权重链接均返回 200 | Transformer logits；可适配但协议不同 | 后续跨仿真器验证候选 |
| VLFM | `584ed56008754fde7997d904983607def8328322` | **没有常规策略训练流水线** | 有 | 不作为此处判断重点 | 零样本模块化系统 | 排除主模型 |

### 3.1 PIRLNav：最合适，但发布状态并非开箱即用

已确认存在：

- `pirlnav/il_trainer.py`：行为克隆/模仿学习训练与评估。
- `pirlnav/ppo_trainer.py`：PPO/RL fine-tuning 与评估。
- `pirlnav/policy/policy.py`：显式 `CategoricalNet`，可取得 action logits 和 entropy。
- `configs/tasks/objectnav_hm3d.yaml`：HM3D ObjectNav 离散协议。
- Habitat-Lab 子模块固定到 `0f454f62e41050bc90ca468c62db35d7484923ff`。
- Habitat-Sim 子模块固定到 `011191f65f37587f5a5452a93d840b5684593a00`。
- Python 源码静态编译检查通过。

数据与权重实测：

- Hugging Face 数据集 `axel81/pirlnav` 当前根目录实际包含 `objectnav_hm3d_hd` 和 `objectnav_hm3d_fe`。
- README 声称的 `objectnav_hm3d_sp` 当前未出现在该数据集根目录，不能按“已验证可用”记录。
- 官方 ObjectNav episode v1 链接返回 200，HTTP `Content-Length=138845369`。
- 官方 HM3D episode v2 链接也返回 200，但不属于当前选定协议。
- 官方 OVRL 编码器链接失效；Hugging Face 的 ZSON 镜像可访问，文件大小为 `354840983` bytes。
- `objectnav_il_hd.ckpt` 和 `objectnav_rl_ft_hd.ckpt` 两个官方 S3 URL 均返回 404；截至审计日期未找到可信镜像。

必须先修复的问题：

1. `run.py` 使用 PID、当前时间和 `os.urandom` 自动生成随机种子，无法进行严格重复实验。
2. `scripts/1-objectnav-il.sh` 虽然接收 dataset 参数，却实际硬编码到 `objectnav_hm3d_10k` 的 overfit 路径。
3. README 写的是 `scripts/2-objectnav-rl-ft.sh`，仓库实际文件名是 `scripts/1-objectnav-rl-ft.sh`。
4. 训练脚本含作者服务器绝对路径、Slurm partition 和 A40 约束，需要拆成环境无关命令与服务器提交脚本。
5. 环境是 Python 3.7、PyTorch 1.12.1、CUDA 11.3 和旧 Habitat commit；应优先原样容器化复现，不应一开始升级整个栈。
6. checkpoint 配置可能覆盖命令行评估配置，必须把最终实际生效配置写入 run manifest。

### 3.2 Tent 的归一化层问题

PIRLNav 和 Habitat 自定义 ResNet 主体使用 **GroupNorm**，而不是 BatchNorm。原始 Tent 的常见实现只收集 BatchNorm affine 参数并重置 batch statistics，因此严格照搬会出现“没有可更新参数”。

第一阶段应采用以下命名：

- `Tent-BN`：严格原版选择规则；在 PIRLNav 上记为 N/A，不伪造结果。
- `Tent-GN`：保持 entropy minimization，只更新 GroupNorm 的 weight/bias；这是实际主对比项。

论文中必须说明 `Tent-GN` 是为了适配导航模型归一化结构所做的必要移植。FSTTA、EAM、FeedTTA、ATENA 若使用相同参数子集，也应统一写成 GN scope，不能让各方法暗中更新不同规模的参数。

### 3.3 AllenAct：最可靠的可下载备选

已确认：

- `projects/objectnav_baselines/` 内有 iTHOR、RoboTHOR 的 RGB、Depth、RGB-D DD-PPO 配置，以及 RoboTHOR DAgger 配置。
- AllenAct 主框架有训练、validation 和 test evaluator。
- RoboTHOR ObjectNav 2021 checkpoint 压缩包已实际下载并成功列出内部 `.pt` 文件。
- 压缩包大小 `64212503` bytes，SHA256：`02c0524856de6a8e2cfbd4acf624018b23ce7a8def0b506f64f928420dcfa79a`。
- RoboTHOR ObjectNav episode 数据链接返回 200，大小 `2536140` bytes。

问题：README 中部分实验文件名已经过时，例如文档写 `objectnav_robothor_rgbd_resnetgru_ddppo.py`，实际文件名包含 `resnet18gru`。此外，预训练模型对应 2021 RoboTHOR 协议，不能拿它的结果替代 HM3D 主结果。

### 3.4 Habitat 官方 DD-PPO：代码完整但不是最终强基线

当前 Habitat-Lab 具有：

- `ddppo_objectnav_hm3d.yaml` 训练配置。
- PPO/DD-PPO trainer、checkpoint 保存/加载和 evaluator。
- `PointNavResNetPolicy` 对 ObjectGoalSensor 的支持与 categorical action distribution。

未发现与当前 HM3D ObjectNav 配置配套的官方完整策略 checkpoint。它适合验证场景、episode、指标和训练栈是否正确，也可作为从零训练的控制模型，但不应取代 PIRLNav 成为唯一的强基线。

### 3.5 HM3D-OVON、PoliFormer 与 VLFM

- **HM3D-OVON**：代码不是空仓库，PPO、DAgger、RNN、Transformer 和评估均存在；DAgRL checkpoint 可下载，大小 `482614361` bytes。但 README/脚本引用了若干不存在的配置文件名，且任务是开放词汇 ObjectNav。
- **PoliFormer**：训练入口、online evaluator、categorical logits 和三个 checkpoint 下载脚本均存在；三个链接分别返回 200，大小约 244 MB、674 MB、675 MB。它是有价值的现代模型，但依赖 Objaverse、ProcTHOR-Objaverse、Stretch、DINOv2、Detic 和不同动作协议，超出第一阶段。
- **VLFM**：具有评估导向的策略和 HM3D 配置，但没有常规的策略训练流水线，不符合“先训练导航基线，再复现 TTA”的要求。

模块化方法（如 PONI、Stubborn）也不适合作为第一 TTA 载体：大量性能来自检测、显式地图和规划器，并不存在一个统一、可微、直接输出动作分布的策略。它们可以作为非自适应导航参考，不适合作为 Tent/FSTTA 主模型。

## 4. 第一阶段实验协议

### 4.1 主表

先只使用一个导航模型，避免范围扩张：

| Model | Source | Tent-GN | FSTTA | EAM | FeedTTA | ATENA | Ours（后续） |
|---|---:|---:|---:|---:|---:|---:|---:|
| PIRLNav IL-HD -> RL-FT |  |  |  |  |  |  |  |

每个方法至少记录：SR、SPL、SoftSPL、Distance to Goal、每 action 推理时间、每 action 适应时间、峰值显存、更新次数、参数漂移。

### 4.2 TTA stream

第一阶段推荐 **scene-level online TTA**：

1. 按 scene 对 episode 分组。
2. 进入一个新 scene 时从同一 Source checkpoint 和 optimizer 初始状态开始。
3. 在该 scene 的 episode 之间保留适应状态。
4. 离开 scene 后重置，不在第一阶段研究跨 scene 的持续遗忘。
5. 固定 episode ID 集合；用 3 个预先记录的 scene/episode 顺序种子报告均值和标准差。

这样每个未见房屋是一个清晰的目标域，既允许积累测试时信息，也暂时隔离 continual TTA 的遗忘问题。跨场景持续适应留到主表完成之后。

### 4.3 超参数与最终评估隔离

HM3DSem v0.1 的公开 val 有 20 个未见场景。为避免在同一批场景上反复挑最好结果：

- 在第一次 TTA sweep 前，按 scene ID 的固定哈希划分 `val-dev` 与 `val-test`，建议 10/10。
- 只在 `val-dev` 做学习率、更新间隔、阈值和参数 scope 搜索。
- 冻结配置后只运行一次 `val-test` 主结果。
- 全 val 结果可以作为与旧论文 Source 数值的辅助对照，但不能冒充未参与调参的最终测试结果。
- 若 2022 EvalAI 的 `test-standard` 仍接受提交，再把它作为最终官方测试；需先确认平台状态。

### 4.4 公平性约束

- 所有方法使用完全相同的 Source checkpoint、episode 集合、顺序、动作采样规则和 STOP 判定。
- Source 本身也运行相同 3 个顺序种子。
- 每个方法记录实际更新的参数名称、参数量、学习率和 optimizer 状态。
- 不允许用 episode success、oracle distance、目标可见性等评价信息更新无监督方法。
- 使用二值 episode 成功反馈的方法必须单独标记为 `feedback-supervised`，不能与 Tent/FSTTA 的无监督设定无标注地混在一起宣称公平。
- PIRLNav 配置里的 `SUCCESS_DISTANCE=0.1` 是针对 `VIEW_POINTS` 距离的内部实现，而官方文字协议是“目标 1.0 m + oracle visibility”。复现时应验证两者在固定 episode 上产生一致 success，不能直接把配置值改成 1.0。

## 5. 实施顺序

1. 固定 PIRLNav、Habitat-Lab、Habitat-Sim commit 和容器镜像。
2. 下载并登记 HM3DSem v0.1、ObjectNav v1 episodes、HD demonstrations、OVRL encoder 的来源和 SHA256。
3. 修复随机种子、硬编码路径和训练脚本，但不升级模型架构。
4. 先运行随机策略/最短路径或官方 DD-PPO smoke test，核对 episode 数量与指标定义。
5. 训练 PIRLNav IL-HD；复现 IL Source。
6. 从 IL checkpoint 做 RL fine-tuning；复现 RL-FT Source。若 Source 未稳定，不开始 TTA。
7. 接入统一的 logits/features/action/episode hooks。
8. 依次完成 Tent-GN、FSTTA、EAM、FeedTTA、ATENA，并逐个做 Source-equivalence 和小流 smoke test。
9. 在 `val-dev` 做完整超参数 sweep，冻结配置后生成 `val-test` 主表。
10. 主对比表完成后，再开始自己的方法；此时才讨论 CTTA、遗忘、额外模型或 OVON。

## 6. 进入实现前的验收门槛

- 相同 checkpoint、seed、episode stream 连续运行两次，逐 episode action 与指标一致。
- `TTA=off` 的接入版本与原始 Source 结果一致。
- 每种 TTA 都能导出实际更新参数列表；列表为空时直接失败。
- 每个 run manifest 包含 Git commit、容器/环境、dataset version、checkpoint SHA256、seed、硬件和完整配置。
- 结果同时保存逐 episode JSONL 与汇总表，不能只保存 TensorBoard 截图。
- 正式主表只读取冻结的逐 episode 结果，不人工抄写。

## 7. 主要来源

- ObjectNav 综述：本目录的 `A_Survey_of_Object_Goal_Navigation.md`。
- Habitat Challenge 2022：<https://aihabitat.org/challenge/2022/>。
- Habitat Challenge 2023：<https://aihabitat.org/challenge/2023/>。
- Habitat 数据清单：<https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md>。
- PIRLNav：<https://github.com/Ram81/pirlnav>。
- PIRLNav demonstrations：<https://huggingface.co/datasets/axel81/pirlnav>。
- OVRL encoder 镜像：<https://huggingface.co/gunjan050/ZSON/blob/main/omnidata_DINO_02.pth>。
- Habitat-Lab：<https://github.com/facebookresearch/habitat-lab>。
- AllenAct：<https://github.com/allenai/allenact>。
- HM3D-OVON：<https://github.com/naokiyokoyama/ovon>。
- PoliFormer：<https://github.com/allenai/poliformer>。
- VLFM：<https://github.com/rai-opensource/vlfm>。
