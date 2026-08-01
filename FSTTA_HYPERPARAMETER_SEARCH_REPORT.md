# FSTTA 在 AVN 上的超参数搜索实验报告

更新日期：2026-08-01

状态：探索性超参数开发报告。SMT+Audio 的 240 组核心网格和 92 组机制探索、
ENMuS 的 48 组模型级强度网格均已完成。ENMuS 批次 48/48 保持 clean commit
并通过校验；SMT+Audio 机制探索中有 28 组因运行中途代码提交发生变化而只能
作为探索性证据。本地仍缺少逐运行 `manifest.json`，Source checkpoint 与数据
资产 provenance 也不完整，因此当前结果不能直接进入论文正式主对比表。

## 1. 结论摘要

本报告联合解析以下两批 SMT+Audio single-source、seed-0、固定 2000-episode
开发流结果：

- `fstta_grid/fstta-smt-single-v1-seed0`：240 组宽范围核心网格；
- `fstta_exploration/fstta-exploration-all-v1-seed0`：92 组定向机制探索。
- `fstta_enmus_grid/fstta-enmus-grid-avn-val-rerun-v2-seed0`：48 组
  ENMuS 模型级适应强度网格。

1. **宽网格结论保持不变：不受控的 SLOW 强度会使策略坍塌。** 240 组中只有
   2 组同时超过 Source 的 SR/SPL；`slow_lr` 从 `1e-4` 增至 `3e-3` 时，
   平均 SR 从 48.75% 降至 5.07%。这说明 VLN/DUET 的更新尺度不能直接迁移到
   SMT+Audio AVN。
2. **定向探索找到了稳定且关键导航指标为正的完整 FSTTA 候选。** 通过保留论文式
   FAST/SLOW 双分支、降低慢更新强度并把 N 扩到 32，validated job 35
   (`fast_lr=3e-7, M=16, slow_lr=1e-4, N=32`) 达到
   SR/SPL/SoftSPL=56.55/30.3007/37.0208，相对 Source 分别提高
   2.40/0.8778/0.5103 个百分点；DTG 也降低 0.274，SNA 提高 1.9232。它的
   NA 增加 1.8455、SWS 降低 0.60 个百分点，因此不能表述为所有指标均改善。
3. **最高 validated SR/SPL 来自 FAST-only 机制消融，而不是完整 FSTTA。**
   job 41 使用 `fast_lr=1e-8, M=2, mean-gradient, use_slow=False`，达到
   56.80/30.3347/37.3230。它比 job 35 只高 0.25 SR 和 0.0340 SPL，且移除了
   FSTTA 的 SLOW 机制，适合作为机制证据，不适合作为论文主表中的完整 FSTTA。
   最高 validated SoftSPL 则是完整 FSTTA job 18 的 37.4818。
4. **更小的 slow LR 和更低的慢更新频率确实控制了漂移。** 在 slow-boundary
   套件中，`slow_lr=1e-5/3e-5/1e-4` 的平均漂移分别为
   0.00120/0.00362/0.01216；N 从 8 增至 32 时平均漂移从 0.00909 降至
   0.00280，平均 SPL 从 29.1984 升至 29.5086。
5. **SLOW 分支在稳定区域有平均正效应，但不是每个配置都受益。** 相对四个
   匹配 FAST-only control，36 个 SLOW 配置平均提高 0.5764 SR、0.1936 SPL
   和 0.1326 SoftSPL；22/36 同时提高 SR/SPL，19/36 三项同时提高。
6. **论文的 concordant gradient 不是 AVN 上唯一有效的 FAST 几何。**
   24 组 FAST-only 实验中，mean-gradient 产生单点最优 job 41，而低学习率下
   last-gradient 的边际均值最高；三种模式存在明显的 LR×M 交互，不能据单点
   宣称某一种梯度聚合普遍更优。
7. **持续 AdamW 状态不是当前慢漂移的唯一原因。** slow-optimizer 套件中，
   persistent AdamW 的平均 SR/SPL 为 54.95/29.5473，高于 window-reset
   AdamW 的 54.625/29.2809。SGD 得到 55.00/29.6911，但其慢锚漂移约
   `8.6e-8`，两个 slow LR 产生完全相同指标，说明该名义学习率下 SGD 近似于
   关闭 SLOW，而不是公平证明 SGD 优于 AdamW。
8. **q 与 FAST LR scaler 的作用较弱且不单调。** q 从 0.1 到 0.99 的组均
   SR 只相差 0.2125、SPL 只相差 0.2740 个百分点；scaler on/off 的平均差也
   只有 -0.1438 SR/+0.0839 SPL，8 个配对中方向不一致。N=16 相比 N=8 的
   稳定改善更明确。
9. **92 组机制探索没有出现宽网格中的末段低性能风险。** 每组都有 40 个
   50-episode 窗口，末四窗平均 SR 均为 48.5%--63.5%，没有触发既定风险
   启发式。稳定区域内漂移与 SR 几乎不相关，但与 SPL/SoftSPL 仍呈中等负相关，
   说明漂移约束主要有助于路径效率，而不是线性决定成功率。
10. **结果仍需一次冻结配置复验。** 同一 commit、同一可见配置的重复任务在
    `fast_lr=3e-7` 下最多相差 1.10 SR 和 0.8725 SPL，表明采样策略与数值
    非确定性会被持续适应放大。最终主表前应在固定 commit 上重跑 Source 与
    job 35，而不能直接把开发网格最大值当作正式结果。
11. **ENMuS 上存在比 Source 更明确的正向区域。** 48 组中有 19 组同时超过
    Source 的 SR/SPL。按 SPL 主、SR 次且要求二者均超过 Source，job 0
    (`fast_lr=1e-8, M=16, slow_lr=1e-5, N=32`) 达到
    SR/SPL/SoftSPL=68.55/37.3752/41.1575，相对 Source 提高
    2.00/1.3272/0.9390 个百分点，同时 NA 减少 2.860。
12. **SMT+Audio 的冻结候选不能直接迁移到 ENMuS。** SMT+Audio job 35 的
    同配置在 ENMuS 网格中对应 job 40，只得到 65.10/35.5873，低于 Source 的
    66.55/36.0480。两个模型都偏好低漂移区域，但有效学习率尺度不同；主表前
    应分别冻结模型级候选，而不能声称一套数值超参数跨模型通用。

综合来看，新的证据把结论从“只有狭窄弱信号”推进为：**FSTTA 在受控慢更新
强度下可以在 SMT+Audio 和 ENMuS AVN 上同时改善成功率与路径效率；但两个
模型需要不同的强度校准，SMT+Audio 上完整 FAST/SLOW 方法又仅略低于
FAST-only 最优消融，因此主表收益、跨模型泛化和 SLOW 机制贡献仍必须通过
冻结配置复验确认。**

## 2. 数据范围与完整性

### 2.1 搜索空间

| 变量 | 取值 |
|---|---|
| FAST learning rate | `1e-8, 1e-7, 3e-7, 1e-6` |
| FAST 窗口 M（动作梯度数） | `2, 3, 4, 8, 16` |
| SLOW learning rate | `1e-4, 3e-4, 1e-3, 3e-3` |
| SLOW 窗口 N（episode 数） | `2, 4, 8` |
| 笛卡尔积 | `4 × 5 × 4 × 3 = 240` |

固定条件为 SMT+Audio、single-source、seed 0、2000 episodes、20 scenes ×
100 episodes、global shuffle、sample 动作、单进程、continual adaptation。
适配参数为最后 4 个 LayerNorm 的 affine 参数，共 8 个张量、2048 个标量。

固定的 FSTTA 参数为：`q=0.1, rho=0.95, tau=0.7, a=0.9, b=1.1`；FAST
和 SLOW 均使用 AdamW，`betas=(0.9, 0.99)`、`weight_decay=0`，最大梯度范数
为 1.0。FAST optimizer moment 在 episode 边界重置，SLOW optimizer moment
跨 N-episode 窗口保留。

### 2.2 批次完整性

| 检查项 | 结果 |
|---|---:|
| 计划/完成 | 240/240 |
| 唯一配置 | 240 |
| exitcode / runner_exitcode 为 0 | 240/240 |
| validation=ok | 240/240 |
| 9 项最终聚合指标完整 | 240/240 |
| 40 个诊断窗口完整 | 240/240 |
| manifest 路径指针 | 240 |
| 本地可解析 manifest | 0 |
| 当前可进入正式结果表 | 0 |

解析最终指标时使用每个日志末尾的 9 条 `Average episode ...`，而不是最后一条
`[TTA] episodes=2000 window=...`；后者只是最后 50 个 episode 的滑动窗口。
240 组日志未发现 NaN、Inf、Traceback、重复配置或重复最终指标向量。

### 2.3 Source 对齐情况

Source 与 240 个 FSTTA 运行在下列项目上完全一致：

- model、single-source setting、seed 0、2000 episodes、sample 动作；
- checkpoint SHA256：`8007dc0d...ef03`；
- stream order SHA256：`07f32759...380c`；
- stream content SHA256：`dd411c4a...fdf2`。

但 Source commit 为 `48ea6285...`，FSTTA commit 为 `a5ec8e08...`，并非同一
代码快照；同时 checkpoint manifest 标为 `provenance_incomplete`，数据资产
manifest 为 `partial`，Source 与 FSTTA 的 per-run manifest 均未同步到本地。
因此这里属于协议和二进制指纹高度对齐的探索性比较，不是形式化可复核结果。

### 2.4 新增机制探索批次的完整性与 commit 分裂

新增批次计划 92 组，四个套件分别为 slow-boundary 40、fast-geometry 24、
slow-optimizer 12 和 q-scaler 16。所有运行均完成 2000 episodes，并具有完整的
9 项最终聚合指标和 40 个诊断窗口；没有运行时异常、OOM、NaN 或缺失日志。

| 套件 | 计划/完成 | runner exit=0 | validation=ok | 实际 Git commit | 使用口径 |
|---|---:|---:|---:|---|---|
| slow-boundary | 40/40 | 40 | 40 | `5ed6ac6` | validated 探索结果 |
| fast-geometry | 24/24 | 24 | 24 | `5ed6ac6` | validated 探索结果 |
| slow-optimizer | 12/12 | 12 | 0 | `ab5f9d8` | commit-mismatch exploratory |
| q-scaler | 16/16 | 16 | 0 | `ab5f9d8` | commit-mismatch exploratory |
| **合计** | **92/92** | **92** | **64** | 2 个 commit | 28 组不得升级为正式结果 |

后 28 组的 `runner_exitcode=0`，`exitcode=90` 仅由 launcher 的运行后校验产生：
batch 锁定 `git_commit=5ed6ac6...`，实际 manifest 记录为 `ab5f9d8...`。两批使用
相同 checkpoint SHA256 `8007dc0d...ef03`、相同 stream-order SHA256
`07f32759...380c` 和 stream-content SHA256 `dd411c4a...fdf2`。可见代码差异
没有改变 FSTTA 更新公式或该批配置，但增加了适配器的通用 no-op hook 和 EAM
实现；这不足以越过 formal provenance 规则。因此：

- slow-optimizer 与 q-scaler 内部比较可作为共同 actual commit 下的机制线索；
- 这 28 组不能计入“同一预注册批次 92/92 validated”的表述；
- 后续任何长批次运行期间都不应 `git pull` 或切换 commit。

### 2.5 ENMuS 模型级网格的完整性

ENMuS 批次使用 single-source、canonical AVN val、seed 0、固定 2000 episodes
和 native action selection，搜索空间为：

| 变量 | 取值 |
|---|---|
| FAST learning rate | `1e-8, 3e-8, 1e-7, 3e-7` |
| FAST 窗口 M | `16, 32` |
| SLOW learning rate | `1e-5, 3e-5, 1e-4` |
| SLOW 窗口 N | `32, 64` |
| 笛卡尔积 | `4 × 2 × 3 × 2 = 48` |

其余设置保持完整论文式双分支：最后 4 个 LayerNorm、`q=0.1`、concordant
FAST、FAST LR scaler、persistent AdamW SLOW、continual adaptation。

| 检查项 | 结果 |
|---|---:|
| 计划/完成/成功 | 48/48/48 |
| validation=ok | 48/48 |
| 唯一配置 | 48 |
| 9 项最终聚合指标完整 | 48/48 |
| 逐运行 41 字段 FSTTA diagnostics | 48/48 |
| NaN/Inf/OOM/Traceback | 0 |
| manifest 路径指针 | 48 |
| 本地可解析 manifest | 0 |
| 当前可进入正式结果表 | 0 |

该批次在 clean commit `4d70c032...` 上运行；checkpoint SHA256 为
`4f37a377...ecefcd`，dataset index、stream order 和 stream content SHA256
分别为 `838532d8...081c80`、`07f32759...a380c` 和
`dd411c4a...37fdf2`，与 ENMuS Source 的 checkpoint 和 canonical single-source
流一致。但是 Source 来自 commit `48ea6285...`，48 个 manifest 仅保留了服务器
绝对路径，本机均不可解析；checkpoint provenance 仍为 incomplete，数据资产
manifest 仍为 partial/formal_use=false。因此这是内部协议一致的开发网格，而非
正式主表运行。

ENMuS 的 diagnostics 没有保存 SMT+Audio 实验中的 50-episode 窗口历史，故可
检查最终漂移、更新计数和数值错误，但不能使用同一套 late-collapse 启发式判断
末段行为。

## 3. 当前 AVN FSTTA 的实际语义

当前实现是一个以论文核心几何为基础、为 AVN 做过数值稳定化的
**FSTTA-style** 移植，不应描述为官方代码的逐行复现。

- 每个动作对预测动作分布做无监督熵最小化，不使用 success 等 episode
  feedback；动作由更新前分布采样，参数更新从下一动作开始生效。
- FAST 缓存 M 个动作梯度，利用低秩 SVD 对梯度协方差方向做逆方差加权，
  再按均值梯度范数校准并由 AdamW 更新。episode 末不足 M 个梯度会丢弃。
- 每个 episode 结束保存当前参数轨迹；累积 N 个 episode 后，SLOW 用相对旧锚
  的轨迹方向构造慢梯度，通过独立、持续的 AdamW 更新锚点，再将 FAST 参数
  snap 到新锚点。
- `q=0.1` 时最近一个 episode 对参考方向的有效权重约为 90%，所以增大 N 的
  主要作用是减少慢更新频率和改变轨迹 PCA 几何，而不是均匀平均更长历史。
- 每个 episode 重置 FAST buffer、方差历史和 FAST AdamW moment，但不重置
  已更新的模型参数；SLOW anchor 与其 AdamW moment 持续跨 episode 保留。

相对论文/公开实现的重要差异包括：AVN SMT+Audio 的 LayerNorm 维度和导航训练
方式与 DUET 不同；本实验使用 sample 而非 argmax；FAST learning rate 最高仅
`1e-6`，远低于论文的 `6e-4`；当前实现采用低秩截断、全局梯度裁剪、eval mode
以及 `weight_decay=0`。公开代码本身还存在未定义变量、慢模型未正确部署等问题，
因此本项目采用的是“论文思想对齐 + 可执行修正”的实现口径。

## 4. 与 Source 的总体比较

Source 的完整结果为：

| 方法 | SR↑ | SPL↑ | SoftSPL↑ | DTG↓ | NDTG↓ | NA↓ | SNA↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source | 54.15 | 29.4229 | 36.5105 | 5.0005 | 0.306263 | 153.1720 | 42.9696 |

240 个配置相对 Source 的计数为：

| 判据 | 配置数 | 占比 |
|---|---:|---:|
| SR 高于 Source | 22 | 9.17% |
| SPL 高于 Source | 2 | 0.83% |
| SR 与 SPL 同时更高 | 2 | 0.83% |
| SoftSPL 高于 Source | 0 | 0.00% |
| SR < 50% | 200 | 83.33% |
| SR < 25% | 147 | 61.25% |
| SR < 10% | 88 | 36.67% |

整个网格的最高 SoftSPL 仅为 36.0469，仍低于 Source 的 36.5105；整个网格
最低 NA 为 161.4405，也高于 Source 的 153.1720。换言之，即使最优候选在
SR/SPL 上有弱增益，也没有提升路径软效率或动作效率。

## 5. 最优配置与 Pareto 前沿

### 5.1 预声明规则下的推荐候选

实验 YAML 预先规定按 SPL 主排序、SR 次排序，并拒绝数值失败和末段坍塌。
在该规则下，job 170 是应冻结的开发候选：

| 配置 | fast LR | M | slow LR | N | SR | SPL | SoftSPL | DTG | NA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source | - | - | - | - | 54.15 | 29.4229 | 36.5105 | 5.0005 | 153.1720 |
| job 170 | 3e-7 | 16 | 1e-4 | 8 | **56.00** | **29.5332** | 35.9718 | 4.9295 | 164.4475 |
| Δ |  |  |  |  | +1.85 | +0.1103 | -0.5387 | -0.0710 | +11.2755 |

job 170 的最后四个 50-episode 窗口平均 SR 为 59.50%，最低为 50.00%；相对
参数漂移为 0.02209，250 次 SLOW 尝试全部成功。它是本次网格中唯一同时具有
最高 SPL、较高 SR、低漂移和良好末段行为的候选。

另一个同时超过 Source SR/SPL 的 job 49 为
`fast_lr=1e-8, M=16, slow_lr=1e-4, N=4`，SR/SPL 为 55.45/29.4339；
其 SPL 只高 0.0110 个百分点，并且被 job 170 在 SR/SPL 上同时支配。

### 5.2 SR/SPL Pareto 前沿

| job | fast LR | M | slow LR | N | SR | SPL | SoftSPL | 末四窗 SR 均值 | 参数漂移 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 170 | 3e-7 | 16 | 1e-4 | 8 | 56.00 | 29.5332 | 35.9718 | 59.50 | 0.02209 |
| 53 | 1e-8 | 16 | 3e-4 | 8 | 56.10 | 29.0397 | 35.4897 | 58.50 | 0.04537 |
| 86 | 1e-7 | 4 | 1e-4 | 8 | **56.45** | 28.8650 | 34.7425 | 55.00 | 0.02250 |

job 86 的 SR 最高，但 SPL 比 Source 低 0.5579 个百分点，因此不符合预声明的
SPL 优先选择标准。job 53 虽有较高 SR，但 SPL 仍低于 Source，且漂移约为
job 170 的两倍。

## 6. 超参数影响

### 6.1 SLOW learning rate：主导稳定性

| slow LR | 配置数 | 平均 SR | 平均 SPL | 平均 SoftSPL | 平均漂移 | 风险启发式触发数 |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-4 | 60 | 48.75 | 24.4538 | 31.0931 | 0.04472 | 21 |
| 3e-4 | 60 | 28.41 | 13.8078 | 21.3173 | 0.13995 | 54 |
| 1e-3 | 60 | 10.92 | 5.2231 | 12.7155 | 0.24412 | 60 |
| 3e-3 | 60 | 5.07 | 2.4379 | 9.1770 | 0.30025 | 60 |

最佳结果位于搜索空间最低的 `slow_lr=1e-4` 边界，而论文使用的 `1e-3` 在
AVN 上几乎全面坍塌。这表明当前 AVN 移植的慢分支更新幅度明显过大；不能把
VLN/DUET 的 slow learning rate 直接迁移到 SMT+Audio。

### 6.2 N：降低慢更新频率有效

| N | 配置数 | 平均 SR | 平均 SPL | 平均漂移 | 风险启发式触发数 |
|---:|---:|---:|---:|---:|---:|
| 2 | 80 | 17.57 | 8.5765 | 0.22056 | 75 |
| 4 | 80 | 23.34 | 11.4652 | 0.17959 | 65 |
| 8 | 80 | 28.95 | 14.4003 | 0.14663 | 55 |

N 增大时 SLOW 尝试次数由约 `2000/N` 降低，参数漂移与失败率同步下降。由于
`q=0.1` 极度偏向最近 episode，这里更可能是更新频率/轨迹几何效应，而不是
更长历史平均带来的收益。

### 6.3 M：稀疏 FAST 更新更稳定

| M | 配置数 | 平均 SR | 平均 SPL | 平均漂移 | 风险启发式触发数 |
|---:|---:|---:|---:|---:|---:|
| 2 | 48 | 16.56 | 7.9376 | 0.18939 | 43 |
| 3 | 48 | 20.09 | 9.7940 | 0.18682 | 43 |
| 4 | 48 | 21.74 | 10.6149 | 0.18165 | 40 |
| 8 | 48 | 26.45 | 13.0390 | 0.17410 | 38 |
| 16 | 48 | 31.59 | 16.0180 | 0.17934 | 31 |

更大的 M 减少 FAST optimizer step，并用更多动作梯度估计协方差，整体上更
稳定。这与 Tent 实验中“控制累计更新预算”比单纯追求更快熵下降更重要的观察
一致。由于 M 同时改变更新频率和梯度几何，当前网格不能将两种机制分离。

### 6.4 FAST learning rate：次于慢分支

| fast LR | 配置数 | 平均 SR | 平均 SPL | 平均漂移 |
|---:|---:|---:|---:|---:|
| 1e-8 | 60 | 27.20 | 13.4147 | 0.16540 |
| 1e-7 | 60 | 22.26 | 10.9980 | 0.18546 |
| 3e-7 | 60 | 21.89 | 10.7652 | 0.19011 |
| 1e-6 | 60 | 21.79 | 10.7449 | 0.18807 |

FAST learning rate 的边际相关较弱，且 job 170 使用的并非最小 fast LR。
这说明在当前双分支实现中，SLOW update 与 M/N 决定的更新预算优先于继续细化
FAST learning rate；不能从边际均值推断 `1e-8` 对所有条件都最优。

### 6.5 关键交互：slow LR × N

| slow LR | N=2 平均 SR/SPL | N=4 平均 SR/SPL | N=8 平均 SR/SPL |
|---:|---:|---:|---:|
| 1e-4 | 41.94 / 20.6271 | 50.18 / 25.0741 | **54.12 / 27.6603** |
| 3e-4 | 18.60 / 8.9455 | 27.43 / 13.2558 | 39.19 / 19.2222 |
| 1e-3 | 6.60 / 3.1269 | 10.62 / 5.0877 | 15.54 / 7.4548 |
| 3e-3 | 3.13 / 1.6066 | 5.14 / 2.4431 | 6.95 / 3.2641 |

`slow_lr=1e-4, N=8` 的 20 组中，12 组 SR 超过 Source，但只有 1 组 SPL
超过 Source。它是稳定区域，不等同于普遍有效区域。

## 7. 末段行为与内部诊断

### 7.1 “末段坍塌”启发式的口径

本文为开发筛选定义了一个风险启发式：最后四个 50-episode 窗口平均 SR
低于 25%，或比全程 SR 低超过 15 个百分点。240 组中 195 组触发；其中
190 组是末四窗均值低于 25%，79 组是相对全程下降超过 15 个百分点，两者
重合 74 组。

该指标**不是 FSTTA 论文定义的 collapse 指标**，也不能把 195 组都解释成
“后期才发生坍塌”：第一项会把全程持续低性能的配置也计入。报告中应称其为
“末段低性能/下降风险启发式”，正式持续 TTA 分析应使用逐 episode 配对轨迹、
滑窗置信区间和明确的 change-point 判据。

### 7.2 参数漂移、熵与策略坍塌

| 变量 | 与 SR 的 Pearson r | 与 SPL 的 Pearson r | 解释限制 |
|---|---:|---:|---|
| `log10(slow_lr)` | -0.867 | -0.859 | 网格伴随关系 |
| 相对参数漂移 | -0.941 | -0.934 | 漂移也可能是失败结果，而非唯一原因 |
| 平均动作熵 | +0.981 | +0.986 | 失败运行多为低熵错误策略 |
| 平均 LR scale | -0.927 | -0.919 | 与状态/梯度几何共同变化 |

最严重的失败运行并不是“不够自信”，而是持续把错误动作分布推向更低熵。
因此对 AVN 来说，单独的 entropy objective 不足以作为可靠的在线学习信号；
参数漂移阈值、Source anchor、恢复机制或导航时序一致性信号可能是必要约束。

### 7.3 SLOW 几何退化

57 个运行发生过 SLOW skip，共 3087 次：1765 次为 degenerate trajectory，
1322 次为 zero reference direction。所有运行均满足
`slow_updates + slow_skips = floor(2000/N)`；最终 FAST 参数与 SLOW anchor
的相对距离均为 0，说明 snap 语义正常执行。这些 skip 不是程序失败，而是当前
轨迹几何无法产生有效 slow direction 的信号。

## 8. 与 Tent 结果的关系

| 方法/配置 | SR | SPL | 相对 Source ΔSR/ΔSPL |
|---|---:|---:|---:|
| Source | 54.15 | 29.4229 | - |
| Tent 调参最佳 | 57.70 | 30.7582 | +3.55 / +1.3353 |
| FSTTA 核心网格 job 170 | 56.00 | 29.5332 | +1.85 / +0.1103 |
| FSTTA 完整方法 job 35 | 56.55 | 30.3007 | +2.40 / +0.8778 |
| FAST-only 消融 job 41 | 56.80 | 30.3347 | +2.65 / +0.9118 |

两项研究共同支持“AVN 的 TTA 首先是稳定性和更新预算问题”。Tent 通过极小
learning rate 或限制累计更新避免坍塌；FSTTA 的定向探索则通过更低的 slow LR
和更大的 N 控制慢锚漂移，使完整方法从核心网格的弱 SPL 增益提升到
+0.8778。新的完整 FSTTA 候选仍低于调参后的 Tent，但差距已经从 1.2250 SPL
缩小到 0.4575 SPL。FAST-only job 41 与完整 job 35 几乎持平，说明目前不能把
收益主要归因于 SLOW 几何；SLOW 的作用更接近稳定区域中的小幅平均增益。

这不是对 FSTTA 原论文结论的否定。论文使用 DUET/VLN、不同参数维度和动作
协议；当前实现也不是官方代码的逐位复现。合理结论是：**论文超参数和稳定更新
假设不能直接迁移到 SMT+Audio AVN，但经过任务级强度校准后，FSTTA 核心机制
可以在当前开发流上产生正向结果。**

## 9. 后续实验建议

### 9.1 当前应该冻结什么

按 SPL 主排序、SR 次排序，并要求保留论文 FAST/SLOW 双分支的规则，新的开发
候选应冻结为 validated slow-boundary job 35：

```text
NORM_SCOPE=last_k_ln
LAST_K_LN=4
LR=3e-7
FSTTA.M=16
FSTTA.LR_SLOW=1e-4
FSTTA.N=32
FSTTA.Q=0.1
FSTTA.FAST_GRAD_MODE=concordant
FSTTA.USE_FAST_LR_SCALER=True
FSTTA.USE_SLOW=True
FSTTA.SLOW_OPTIMIZER=AdamW
FSTTA.RESET_SLOW_OPTIMIZER_EACH_WINDOW=False
EPISODIC=False
STEPS=1
```

job 41 的 SPL 更高 0.0340 个百分点，但它设置 `USE_SLOW=False` 且使用
mean-gradient，不再是完整的论文式 FSTTA，故只保留为机制消融。job 35 下一步
应在**同一 Git commit** 下与 Source 一起复验，并生成包含配置、checkpoint、
数据版本、seed、硬件和完整 SHA256 的本地 run manifest。只有确认运行通过后，
才可作为 FSTTA 主表结果；不能直接把开发网格最大值填入正式表。

### 9.2 主表扩展顺序

1. SMT+Audio single-source：冻结配置后的确认性复验；
2. SMT+Audio multi-source：不再重新调参，直接检验迁移；
3. ENMuS single/multi：沿用同一配置，判断跨模型泛化；
4. 若统一配置严重失败，再在独立 dev stream 上声明一个很小的模型级校准实验，
   不能在正式测试流上反复选参。

### 9.3 当前停止继续扩网格

四组机制探索已经回答了低 slow LR、大 N、FAST 几何、SLOW optimizer state、
q 和 LR scaler 的主要问题。此时继续在同一开发流上扩展超参数会增加选择偏差，
优先级应改为：

1. 固定 SMT+Audio job 35，在单一 commit 上复验 SMT+Audio single-source；
2. ENMuS 模型级网格已经证明 SMT+Audio 数值配置不能直接迁移；因此分别冻结
   SMT+Audio job 35 与 ENMuS job 0，再扩展到各自的 multi-source 条件，不继续
   扩大搜索空间；
3. 将 job 41 作为 `w/o SLOW + mean gradient` 机制消融，而不是 FSTTA 主结果；
4. 等主对比完成后再测试 drift threshold、Source-anchor interpolation 或回滚；
5. 正式时延实验使用单 GPU、无并发、固定预热与重复测量。当前每卡 8 个任务的
   wall-clock 日志不能用于论文时延表。

## 10. 证据与可复现性清单

- 批次定义：`avn/experiments/fstta_core_grid.yaml`
- 启动脚本：`avn/scripts/run_fstta_grid.sh`
- 批次元数据：
  `avn/results/logs/fstta_grid/fstta-smt-single-v1-seed0/batch.env`
- 批次状态：
  `avn/results/logs/fstta_grid/fstta-smt-single-v1-seed0/SUMMARY`
- 240 组配置：
  `avn/results/logs/fstta_grid/fstta-smt-single-v1-seed0/grid.csv`
- 逐运行日志：
  `avn/results/logs/fstta_grid/fstta-smt-single-v1-seed0/jobs/<run_tag>/console.log`
- 机制探索定义：`avn/experiments/fstta_exploration.yaml`
- 机制探索启动脚本：`avn/scripts/run_fstta_explorations.py`
- 机制探索批次元数据与状态：
  `avn/results/logs/fstta_exploration/fstta-exploration-all-v1-seed0/{batch.env,SUMMARY}`
- 92 组机制探索配置：
  `avn/results/logs/fstta_exploration/fstta-exploration-all-v1-seed0/grid.csv`
- 机制探索逐运行日志：
  `avn/results/logs/fstta_exploration/fstta-exploration-all-v1-seed0/jobs/<run_tag>/console.log`
- Source 对照：
  `avn/results/logs/source_reval/source-reval-v1-seed0/metrics.csv`
- Tent 对照与前序稳定性分析：`TENT_HYPERPARAMETER_SEARCH_REPORT.md`

当前 `avn/experiments/fstta_core_grid.yaml` 仍写着 `status: planned`，与已经完成的
日志状态不一致；后续在正式登记该批次时应更新实验元数据，但不应借此把缺少
manifest 的开发日志升级为正式结果。

## 11. 四组机制探索的详细结果

### 11.1 最优结果与方法身份

| 配置 | 方法身份 | SR | SPL | SoftSPL | DTG | NDTG | NA | SNA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Source | frozen source | 54.15 | 29.4229 | 36.5105 | 5.0005 | 0.306263 | 153.1720 | 42.9696 |
| core job 170 | 完整 FSTTA | 56.00 | 29.5332 | 35.9718 | 4.9295 | 0.299450 | 164.4475 | 44.0750 |
| exploration job 18 | 完整 FSTTA；最高 SoftSPL | 55.60 | 29.9963 | **37.4818** | 4.8185 | 0.295622 | 153.6485 | 44.0019 |
| exploration job 35 | **完整 FSTTA 候选** | 56.55 | 30.3007 | 37.0208 | **4.7265** | 0.289784 | 155.0175 | **44.8928** |
| exploration job 41 | FAST-only 消融 | **56.80** | **30.3347** | 37.3230 | 4.7470 | 0.289085 | 152.4080 | 44.5110 |

job 35 为 `fast_lr=3e-7, M=16, slow_lr=1e-4, N=32, q=0.1`，使用
concordant FAST、LR scaler 和 persistent AdamW SLOW。它完成约 18,479 次
FAST update 和 62 次 SLOW update，最终相对参数漂移为 0.00686，末四个窗口
平均 SR 为 63.0%。

job 41 为 `fast_lr=1e-8, M=2, mean-gradient, use_slow=False`，完成约
151,952 次 FAST update，最终漂移仅 0.000457。它说明低 LR 下高频平均梯度
更新本身可以有效，但不能用来证明 FSTTA 的慢分支有效。

### 11.2 slow-boundary：慢更新强度与频率

仅统计 36 个启用 SLOW 的运行：

| slow LR | 运行数 | 平均 SR | 平均 SPL | 平均 SoftSPL | 平均漂移 |
|---:|---:|---:|---:|---:|---:|
| 1e-5 | 12 | 54.933 | **29.4971** | **36.6569** | **0.00120** |
| 3e-5 | 12 | 54.988 | 29.4686 | 36.5875 | 0.00362 |
| 1e-4 | 12 | **55.046** | 29.2073 | 36.0743 | 0.01216 |

| N | 运行数 | 平均 SR | 平均 SPL | 平均 SoftSPL | 平均漂移 |
|---:|---:|---:|---:|---:|---:|
| 8 | 12 | **55.088** | 29.1984 | 36.2134 | 0.00909 |
| 16 | 12 | 54.992 | 29.4660 | 36.5308 | 0.00508 |
| 32 | 12 | 54.888 | **29.5086** | **36.5746** | **0.00280** |

这两个边际表说明：降低 slow LR 和增大 N 都能减少漂移并改善路径效率，但不
保证 SR 单调提高。job 35 出现在 `slow_lr=1e-4,N=32`，是 FAST 配置与慢更新
频率交互形成的单点最优，不能推翻低 slow LR 的总体稳定性优势。

相对相同 fast LR/M 的四个 FAST-only controls，SLOW 配置的平均增益为
+0.5764 SR、+0.1936 SPL、+0.1326 SoftSPL；22/36 同时提高 SR/SPL，
19/36 三项同时提高。这支持“稳定慢分支有小幅平均价值”，但 14/36 没有同时
提高 SR/SPL，说明慢分支仍不是无条件有益。

### 11.3 fast-geometry：M 与梯度聚合

24 组 FAST-only 运行的边际统计如下：

| 切片 | 运行数 | 平均 SR | 平均 SPL | 平均 SoftSPL | 平均漂移 |
|---|---:|---:|---:|---:|---:|
| fast LR=1e-8 | 12 | **55.233** | **29.6715** | **36.5987** | **0.000188** |
| fast LR=3e-7 | 12 | 54.913 | 28.7825 | 35.5722 | 0.008923 |
| concordant | 8 | 54.850 | **29.3302** | **36.2767** | **0.003154** |
| mean | 8 | 55.088 | 29.0822 | 35.8554 | 0.005578 |
| last | 8 | **55.281** | 29.2686 | 36.1243 | 0.004935 |

边际均值与单点最优方向不一致：mean 的总体 SPL 最低，但 `1e-8,M=2` 的 mean
组合产生全套件最高 SR/SPL；在 `1e-8` 切片上，last 的平均 SR/SPL 为
55.6875/29.9189。由此只能得出“梯度聚合方式与 LR/M 强交互”，不能声称简单
均值或最后梯度已经普遍优于论文 concordant geometry。

### 11.4 slow-optimizer：状态持续、窗口重置与 SGD

以下 12 组均完成运行，但因 commit mismatch 只能作为探索性机制结果：

| SLOW optimizer | 状态语义 | 运行数 | 平均 SR | 平均 SPL | 平均 SoftSPL | 平均漂移 |
|---|---|---:|---:|---:|---:|---:|
| AdamW | persistent | 4 | **54.950** | 29.5473 | 36.3697 | 0.01126 |
| AdamW | 每窗口重置 | 4 | 54.625 | 29.2809 | 36.1387 | 0.01134 |
| SGD, momentum=0 | persistent | 4 | **55.000** | **29.6911** | **36.4771** | **8.6e-8** |

persistent AdamW 没有比 window-reset AdamW 更差，尤其 N=16 时其平均
SR/SPL=55.325/29.7847，而 reset 为 54.475/29.1760。因此此前“主要是跨窗口
AdamW moment 导致慢漂移”的假设不受支持。

SGD 的两个 slow LR (`3e-5,1e-4`) 在相同 N 下给出逐项完全相同的最终指标，
且慢锚几乎不漂移。这说明 SGD 的名义 LR 与 AdamW 有效步长不匹配，结果更接近
FAST-only control；不能把它写成“SGD optimizer 优于 AdamW”。

### 11.5 q 与 FAST LR scaler

以下 16 组同样是 commit-mismatch exploratory：

| q | 运行数 | 平均 SR | 平均 SPL | 平均 SoftSPL | 平均漂移 |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 4 | 54.863 | 29.0229 | 35.8284 | 0.01748 |
| 0.5 | 4 | 54.975 | **29.2970** | 36.0491 | 0.01725 |
| 0.9 | 4 | 55.000 | 29.1579 | 35.9392 | 0.01709 |
| 0.99 | 4 | **55.075** | 29.2815 | **36.2194** | **0.01686** |

q 没有形成单调且跨指标一致的最优点。q 越大时参考方向更接近均匀利用 N 个
episode，漂移略降，但变化量远小于 slow LR 和 N 的影响。q=0.5、N=16、
scaler off 的 job 83 取得该套件最高 SPL=30.0707，但不应据一个未 validated
单点替换论文默认 q=0.1。

| 设置 | 运行数 | 平均 SR | 平均 SPL | 平均 SoftSPL | 平均漂移 |
|---|---:|---:|---:|---:|---:|
| scaler on | 8 | 54.906 | **29.2318** | **36.0647** | **0.01708** |
| scaler off | 8 | **55.050** | 29.1479 | 35.9534 | 0.01726 |

scaler on 的平均 LR scale 为 1.083，接近上限 1.1，但配对效果方向混合：8 对中
仅 5 对 SPL 更高，SR 则多数持平或更低。因此 scaler 可按论文设定保留，但当前
数据不支持把它视为主要增益来源。相比之下，N=16 相对 N=8 平均提高
0.2313 SR、0.3369 SPL，并把漂移从 0.02191 降到 0.01244，证据更一致。

### 11.6 重复运行与结论边界

探索计划有意在不同套件中重复若干 control。相同 commit、相同 checkpoint、
相同 stream 和相同可见配置下：

- 两个 `fast_lr=1e-8` controls 的重复结果逐项一致；
- `fast_lr=3e-7,M=16,concordant,use_slow=False` 的 job 39 与 job 61 分别为
  54.55/29.1125 和 55.65/29.9850，差 1.10 SR、0.8725 SPL；
- 这种差异可能来自 GPU 数值非确定性、sample 动作的早期微小分岔及其被在线
  适应放大，现有聚合日志不能进一步做因果归因。

因此，本报告把 92 组用于机制筛选和冻结候选，不做显著性声明。论文主表仍只
使用一次预先冻结顺序，但选定配置必须在最终 commit 上重新运行；探索日志中的
最大值不直接进入主表。

## 12. ENMuS 48 组模型级强度网格

### 12.1 最优配置与 Source 对比

按“SR、SPL 均高于 Source，再以 SPL 主、SR 次排序”，最佳配置是 job 0：

```text
fast_lr=1e-8
M=16
slow_lr=1e-5
N=32
q=0.1
fast_grad_mode=concordant
use_fast_lr_scaler=True
use_slow=True
slow_optimizer=AdamW
reset_slow_optimizer_each_window=False
```

| 配置 | Reward | SR↑ | SPL↑ | SoftSPL↑ | DTG↓ | NDTG↓ | NA↓ | SNA↑ | SWS↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source | 14.658422 | 66.55 | 36.0480 | 40.2185 | 3.2115 | 0.195937 | 164.256 | 51.1794 | 2.20 |
| job 0 | 15.041523 | **68.55** | **37.3752** | **41.1575** | **3.1220** | 0.196867 | **161.396** | **52.3454** | 2.05 |
| Δ | +0.383101 | +2.00 | +1.3272 | +0.9390 | -0.0895 | +0.000930 | -2.860 | +1.1660 | -0.15 |

job 0 同时取得整个网格最高 SR 和 SPL；除 NDTG 轻微增加 0.000930、SWS 下降
0.15 个百分点外，主要成功率、路径效率、SoftSPL、DTG、NA 和 SNA 均改善。

如果采用更严格的“SR、SPL、SoftSPL、DTG、NDTG、NA、SNA 七项方向均优于
Source”约束，只有 job 5 和 job 9 满足。其中 SPL 更高的 job 5 为
`1e-8/M16/1e-4/N64`，SR/SPL/SoftSPL=67.60/37.0844/41.0624，
DTG/NDTG=3.1620/0.193793，NA/SNA=163.869/51.6325。论文主表仍应优先采用
预声明的 SPL-first job 0，并把 job 5 作为稳健性备选，而不是事后改变选择规则。

### 12.2 超过 Source 的配置数

| 判据 | 配置数 | 占比 |
|---|---:|---:|
| SR 高于 Source | 22 | 45.83% |
| SPL 高于 Source | 29 | 60.42% |
| SR 与 SPL 同时更高 | 19 | 39.58% |
| SoftSPL 高于 Source | 15 | 31.25% |
| DTG 更低 | 4 | 8.33% |
| NDTG 更低 | 2 | 4.17% |
| NA 更低 | 8 | 16.67% |
| SNA 更高 | 12 | 25.00% |
| 七项方向全部更好 | 2 | 4.17% |

全网格平均 SR/SPL/SoftSPL 为 66.4490/36.1548/40.0692。平均 SR 和 SPL
略高于 Source，但平均 SoftSPL 略低，因此不能把 job 0 的收益解释为整个网格
普遍改善；正向结果仍集中在低强度、低漂移区域。

### 12.3 四个搜索维度的边际结果

| 维度 | 取值 | 平均 SR | 平均 SPL | 平均 SoftSPL | SR/SPL 双胜 | 平均漂移 |
|---|---:|---:|---:|---:|---:|---:|
| fast LR | `1e-8` | **66.7333** | **36.3094** | **40.2110** | 6/12 | **0.001844** |
|  | `3e-8` | 66.3417 | 36.1521 | 40.0810 | 4/12 | 0.002020 |
|  | `1e-7` | 66.4250 | 36.0827 | 40.0095 | 4/12 | 0.002422 |
|  | `3e-7` | 66.2958 | 36.0751 | 39.9752 | 5/12 | 0.002441 |
| M | 16 | **66.5083** | **36.2282** | **40.1614** | 9/24 | 0.002296 |
|  | 32 | 66.3896 | 36.0814 | 39.9769 | 10/24 | **0.002067** |
| slow LR | `1e-5` | **66.5719** | **36.2783** | **40.1861** | 7/16 | **0.000470** |
|  | `3e-5` | 66.3406 | 36.1814 | 40.0958 | 6/16 | 0.001403 |
|  | `1e-4` | 66.4344 | 36.0047 | 39.9255 | 6/16 | 0.004672 |
| N | 32 | 66.3521 | 36.1153 | 40.0353 | 8/24 | 0.002844 |
|  | 64 | **66.5458** | **36.1943** | **40.1030** | 11/24 | **0.001520** |

边际方向与 SMT+Audio 的稳定性观察一致：更低的 FAST/SLOW LR 和更稀疏的
SLOW 更新总体更安全。`slow_lr=1e-5` 的平均漂移只有 `1e-4` 的约 10%，N=64
又把 N=32 的平均漂移近乎减半。M=16 的平均性能略好，但 M=32 的漂移略低，
说明 FAST 更新频率与路径性能之间仍有交互，不能只按漂移单调选择。

### 12.4 稳定性与诊断

- 48 组平均相对参数漂移为 0.0021818，范围 0.0002587--0.0072517；job 0
  为 0.0005361。
- 所有 SLOW attempt 均成功，`slow_skipped_updates=0`；未出现 NaN、Inf、OOM、
  Traceback 或异常退出。
- LR scaler 的均值几乎恒定在 1.10000002，且没有 lower-bound hit，说明它在
  本批次基本饱和上界，不能据此证明动态 scaler 带来独立收益。
- compact ENMuS 日志没有 50-episode 历史窗口，因此“没有数值异常”不等于
  “已经排除末段坍塌”；冻结复验应保存与 SMT+Audio 相同的窗口诊断。

### 12.5 跨模型结论与冻结建议

SMT+Audio 完整方法候选 job 35 的数值配置
`3e-7/M16/1e-4/N32` 在 ENMuS 中对应 job 40，其 SR/SPL/SoftSPL 只有
65.10/35.5873/39.7108，低于 ENMuS Source 的 66.55/36.0480/40.2185。
因此当前证据支持的是“两个模型都需要控制累计漂移”，而不是“同一组 FSTTA
学习率跨模型通用”。

ENMuS 下一步应冻结 job 0，在同一 Git commit 下重跑 Source 和 FSTTA，并保存
本地可解析 manifest、完整资产 provenance、硬件信息与 50-episode 窗口诊断。
若该复验成立，再把相同 ENMuS 候选迁移到 multi-source；不应继续在同一开发流
扩大网格或把本次 48 选 1 的最大值直接填入正式主表。
