# AVN 重评与固定参数搜索 v1

日期：2026-09-09。执行规格为 [avn_reval_search_v1.json](avn_reval_search_v1.json)，入口为 [run_avn_reval_search.py](../scripts/run_avn_reval_search.py)。本轮使用 `seed=0`、`audio_seed=0`，先重评 ENMuS 的 Source/Tent/FSTTA/EAM 和 SMT+Audio 的 Source，再为两个模型的 FeedTTA、ATENA 分别搜索单源与多源配置。历史记录用于确定候选参数；本轮重新产生 Source、运行 manifest 和结果。

本文件独立固定本轮执行预算。全部 2000-episode 运行仍属于验证集开发与选参；运行证据完整不等于具有独立留出集的论文最终结果。预训练权重的来源限制仍以 [imported_pretrained.yaml](../checkpoints/manifests/imported_pretrained.yaml) 为准。

## 1. 数据、音频与配对 Source

每次完整运行使用单环境、batch size 1，在 20 个场景各取 100 个 episode 后组成的固定全局打乱流上顺序执行 2000 个 episode。模型动作按策略分布采样，方法在 episode 之间持续适应；每个候选、每种单源/多源设置都从独立进程和原始 checkpoint 开始。禁止沿用上一候选的参数或优化器状态。

ENMuS 固定以下配置：

```text
EVAL.PROTOCOL_PROFILE = enmus_clavn_aligned_v1
TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_MODE = episode_seeded_v1
TASK_CONFIG.SIMULATOR.AUDIO.SCHEDULE_SEED = 0
```

`enmus_clavn_aligned_v1` 对齐 CL-AVN 的传感器、任务和音频语义，并保留其音频间隔生成公式、episode 音频参数。`episode_seeded_v1` 为每个 episode 和音源角色创建独立随机流，种子由模式、音频 seed、单源/多源设置、scene ID、episode ID、角色共同确定。目标音源与干扰音源不共享随机状态，音频时序也不依赖全局 NumPy 的消耗历史。这样，即使 TTA 改变轨迹，同一个 episode 的 Source/TTA 音频开关时序仍能配对。

这仍是单环境的 2000-episode 全局流。旧 CL-AVN 的八环境、按场景评估与历史全局 RNG 音频序列没有被逐项重放；旧表的 Source SR 36.60 也不是本轮通过门槛。新 profile、解析后的语义摘要、每个 episode 实际生成的音频开关时序哈希都会记录并与本轮 Source 比对。SMT+Audio 保留原生音频路径，使用同模型、同设置、同数据流和当前运行代码的 Source 对照。

Source 必须来自同一 batch，绑定同模型、同单源/多源设置、相同完整数据流、checkpoint、代码与环境。四个完整 Source 均通过验证后才允许进入搜索；20-episode smoke Source 只服务于对应 smoke。任何旧 Source 数字都不会自动补入本轮对照。

完整 SHA256 固定在 JSON 的 `data.streams`、`checkpoints` 与 `auxiliary_checkpoints`。包括两个数据索引、两条 episode 顺序/内容指纹、四个主 checkpoint，以及 ENMuS 的 audio encoder、visual encoder、SELD 三个辅助权重。预检读取实际文件逐项校验；该清单不代表所有场景、音频/RIR 二进制都已具备完整历史来源。

## 2. 重评矩阵：10 次完整运行

| 模型 | 方法 | 设置 | 本轮固定参数 | 历史选择依据 |
|---|---|---|---|---|
| SMT+Audio | Source | 单源、多源 | 无适应，各 1 次 | 重新建立当前配对对照 |
| ENMuS | Source | 单源、多源 | 无适应，各 1 次 | 重新建立当前配对对照 |
| ENMuS | Tent | 单源 | `last_ln`，LR `1e-8`，interval `1` | SR/Pareto 点 `j030` |
| ENMuS | Tent | 多源 | `last_ln`，LR `1e-7`，interval `1` | 历史多源最优点 |
| ENMuS | FSTTA | 单源、多源 | fast LR `1e-8`，M `16`；slow LR `1e-5`，N `32` | 历史 `j000`，两种设置各重评一次 |
| ENMuS | EAM | 单源、多源 | LR `3e-9`，interval `128` | 边界搜索 `j011`，两种设置各重评一次 |

Tent 单源采用的是历史 `j030`（SR 68.50，SPL 37.0462），没有改成仅按 SPL 选出的 `j080`（SR 67.95，SPL 37.0598），也没有采用旧主表统一配置 `ln/1e-8/u1`。这是本轮预先指定的历史配置重评；后面的 FeedTTA/ATENA 搜索统一按第 5 节排序。

其余参数也固定，避免配置默认值变化导致“同 LR、不同方法”：

| 方法 | 固定参数 |
|---|---|
| Tent | Adam，betas `(0.9, 0.999)`，weight decay `0`，max grad norm `1`，`LAST_K_LN=4`，`RESET_BN_STATS=True`，每 episode 最大更新次数 `-1` |
| FSTTA | `last_k_ln`、K `4`；Q `0.1`、rho `0.95`、tau `0.7`、A `0.9`、B `1.1`；启用 slow 和 fast LR scaler；fast grad mode `concordant`；fast/slow 均 AdamW，betas `(0.9, 0.99)`，weight decay `0`，slow momentum `0`，eigen eps `1e-6`，全局 max grad norm `1`、`RESET_BN_STATS=True`；每 episode 重置 fast optimizer，不在每个 window 重置 slow optimizer，不在每 episode 重置 variance history |
| EAM | 参数前缀 `net.smt_state_encoder.transformer`、`action_distribution`；confidence scale `0.4`，memory `32`，batch `8`；Adam，betas `(0.9, 0.999)`，weight decay `0`，max grad norm `0` |

所有方法均固定 `NUM_PROCESSES=1`、`EVAL.USE_CKPT_CONFIG=False`、`EVAL.SPLIT=val`、`EVAL.ACTION_SELECTION=sample`、`TTA.EPISODIC=False`、`TTA.STEPS=1`。上述参数以 spec 中的最终展开值为准。

历史依据：[Tent 搜索报告](../results/analysis/hparam_search/TENT_HYPERPARAMETER_SEARCH_REPORT.md)、[FSTTA 搜索报告](../results/analysis/hparam_search/FSTTA_HYPERPARAMETER_SEARCH_REPORT.md)、[FSTTA 多源运行器](../scripts/run_fstta_enmus_multi.py)、[EAM 搜索报告](../results/analysis/hparam_search/EAM_HYPERPARAMETER_SEARCH_REPORT.md)。这些旧指标不被写成本轮已完成结果。

## 3. 固定搜索：8 个单元，每单元 12 点

一个搜索单元由模型、方法、单源/多源设置共同确定。两个模型 × 两个反馈方法 × 两种设置，共 8 个单元、96 次完整运行。单源和多源独立选优，即使候选列表相同也不共享 winner。

### FeedTTA：每模型 4 个 LR × 3 个 gamma

| 模型 | LR 列表 | 每个 LR 的 gamma |
|---|---|---|
| SMT+Audio | `3e-9, 1e-8, 3e-8, 1e-7` | `0.95, 0.99, 1.0` |
| ENMuS | `1e-8, 3e-8, 1e-7, 3e-7` | `0.95, 0.99, 1.0` |

按 LR 优先、gamma 次序编号 `f00` 至 `f11`。固定 `p=0.05`、`alpha=-0.2`；Adam，betas `(0.9, 0.999)`，eps `1e-5`，weight decay `0`，max grad norm `0`，不做梯度归一化；参数前缀为 `net.smt_state_encoder` 和 `action_distribution`；`SGR_MODE=paper_main`、`SGR_SEED=0`，动作协议为 `sample_from_policy`。

[历史 FeedTTA 搜索](../results/analysis/hparam_search/FEEDTTA_HYPERPARAMETER_SEARCH_REPORT.md) 中 SMT+Audio 的有效最佳区域为 LR `3e-8`、gamma `0.95`，`>=1e-6` 的学习率表现差，因此本轮围绕低学习率区域搜索。ENMuS 旧 24 点虽完成导航，却全部因适应参数计数校验失败而无有效 winner；其中 `3e-7` 只为本轮候选设计提供探针，不能继承为已验证最佳配置。

### ATENA-AVN(sample)：四个单元使用相同 12 点

| ID | Query LR | Self LR | Mix lambda | Query threshold |
|---|---:|---:|---:|---:|
| a00 | 3e-8 | 1e-8 | 0.50 | 0.50 |
| a01 | 3e-8 | 1e-8 | 0.50 | 0.75 |
| a02 | 1e-7 | 1e-8 | 0.50 | 0.50 |
| a03 | 1e-7 | 1e-8 | 0.50 | 0.75 |
| a04 | 3e-7 | 3e-8 | 0.50 | 0.50 |
| a05 | 3e-7 | 3e-8 | 0.50 | 0.75 |
| a06 | 1e-6 | 1e-7 | 0.50 | 0.50 |
| a07 | 1e-6 | 1e-7 | 0.50 | 0.75 |
| a08 | 3e-7 | 3e-8 | 0.50 | 0.10 |
| a09 | 3e-7 | 3e-8 | 0.50 | 1.00 |
| a10 | 3e-7 | 3e-8 | 0.25 | 0.75 |
| a11 | 3e-7 | 3e-8 | 0.75 | 0.75 |

全部固定 self-loss weight `0.1`、AdamW、betas `(0.9, 0.999)`、weight decay `0.01`、max grad norm `0`；每 episode 构造优化器。`PARAM_SCOPE=all` 与 `TASK_UPDATE_SCOPE=replay_reachable_actor_navigation_policy` 共同限定实际可回放到的 actor 导航参数及 self head，排除 critic 和计算图不可达分支；日志记录实际参数名称、数量与漂移。

跨任务结果只用于确定搜索范围：PONI 的 FeedTTA `3e-7/0.99` 曾有效，但其 ATENA `query LR=3e-7/self LR=3e-8/lambda=0.5/threshold=0.01` 导致 100% 查询，不能据此认为低阈值适合 AVN；HAMT 的 `4e-7/5e-8/lambda=0.75/threshold=0.1` 有提升，而 query LR `1.6e-6` 表现差，所以本轮 `1e-6` 仅作为上界探针；连续 VLN 的记录支持再检查 lambda `0.25`。阈值 `0.5/0.75` 是预先冻结的 AVN 探索值，不宣称已跨任务验证。

证据入口：[ATENA AVN 迁移契约](ATENA_AVN_SAMPLE_REPRODUCTION_CONTRACT.md)、[PONI legacy 归档](../../objectnav/results/legacy/poni_mp3d_tta_20260908/README.md)、[R2R-CE registry](../../vln/results/final/r2r-ce/registry.json)。HAMT 依据为本地归档 `autodl-analysis/vln/results/tuning/consistency_v2_five_41c128d/r2r/consistency-v2-five-41c128d/hamt-r2r/atena/FROZEN.json`，属于选参设计证据，不是运行依赖或独立留出评估。

FeedTTA 在 episode 结束后使用真实二值成功/失败反馈；ATENA 仅在熵门控决定查询时读取真实成功，其余 episode 使用 self head。两者与无监督 Tent/FSTTA/EAM 分组报告。ATENA 的 pseudo-expert 使用实际执行的采样动作，报告名称固定为 **ATENA-AVN(sample)**，与官方 VLN argmax 协议的差异保留在迁移契约。到目标距离、最短路径和未来 episode 结果不作为适应输入。

## 4. 阶段、GPU 分配与运行数量

`--stage all` 自动按 smoke → reval → search 执行。smoke 先运行 4 个 Source，再运行 8 个 FeedTTA/ATENA 首候选，每次 20 个 episode，共 12 次，仅验证技术路径。reval 先完成 4 个完整 Source，再完成 6 个 ENMuS 无监督方法运行，共 10 次。search 共 96 次，每次 2000 个 episode。总计 118 个计划任务；完整运行 106 次。

| GPU | 模型 / 方法 | 每个搜索轮次 |
|---|---|---|
| 0 | SMT+Audio / FeedTTA | 单源 3 点 + 多源 3 点 |
| 1 | SMT+Audio / ATENA | 单源 3 点 + 多源 3 点 |
| 2 | ENMuS / FeedTTA | 单源 3 点 + 多源 3 点 |
| 3 | ENMuS / ATENA | 单源 3 点 + 多源 3 点 |

每轮每卡并发 6 个任务，总计 24 个。共有 4 个全局轮次，分别消费各单元候选列表的第 `0–2`、`3–5`、`6–8`、`9–11` 项；上一轮全部完成并通过验证后才启动下一轮。发生技术失败时停止推进并保留已完成结果；不会因得分不佳扩大预算或修改候选。

每个任务使用独立进程组，调度器处理中断时清理其任务进程组。相同 batch 具有独占锁，避免两个调度器同时写入。这里的并发安排按四张可用 GPU 编号 `0,1,2,3` 固定；运行时不要自行重映射为另一组可见设备。

## 5. 有效性、排序与负结果

只有实际完成预期 episode、指标有限、配置/资产/Source 绑定一致、manifest 验证通过，且方法诊断满足约定的任务才能参与排序。适应参数计数、真实参数变化、回放一致性、采样动作和 ENMuS 实际音频时序等检查属于技术有效性；不会用“超过历史 Source”或“查询比例必须落在某个经验区间”排除结果。

各单元按以下固定顺序选优：SPL 降序 → SR 降序 → SoftSPL 降序 → 相对参数漂移升序 → candidate ID 升序。每单元 12 个有效候选全部完成后标记 `complete=true`；部分候选的暂时 winner 明确标记为 provisional。ATENA query rate 单独报告，不进入排名。

所有相对本轮匹配 Source 的负 delta 都保留。失败、被中断或失效的 attempt 也保留状态与错误，不能被新 attempt 覆盖。CSV 中 SR/SPL/SoftSPL 和 delta 使用 `0–1` 比例；转换为百分比或百分点时乘以 100。本轮输出不会自动覆写旧主表、Excel、legacy 归档，也不会把 validation-selected 结果自动升级为论文正式结论。

## 6. 在服务器同步与运行

服务器已有仓库目录为 `/data1/wxy/code/NavTTA`。两个模型统一使用 `/data1/wxy/anaconda3/envs/enmus/bin/python3`。GitHub 仓库为 `https://github.com/SheldonWongww/SPARK-AVN`，本轮代码分支为 `navtta-eval-fixes-20260909`；其与该仓库原 `main` 的历史独立，首次同步按以下命令创建本地分支。

```bash
cd /data1/wxy/code/NavTTA
git status --short
git fetch https://github.com/SheldonWongww/SPARK-AVN.git navtta-eval-fixes-20260909
git switch -c navtta-eval-fixes-20260909 FETCH_HEAD
```

先保存服务器上自己尚未提交的代码变更，再切换分支。若该分支已存在，使用 `git switch navtta-eval-fixes-20260909`，随后再次 fetch 并 `git merge --ff-only FETCH_HEAD`。数据、场景、音频和 checkpoint 继续使用服务器现有的 AVN 路径，不通过 Git 同步。

设置解释器与本次唯一 batch ID，安装共享包，然后查看计划和预检：

```bash
export NAVTTA_AVN_PYTHON=/data1/wxy/anaconda3/envs/enmus/bin/python3
export NAVTTA_AVN_BATCH=avn-reval-search-v1-seed0-audio0-20260909
"$NAVTTA_AVN_PYTHON" -m pip install -e core

"$NAVTTA_AVN_PYTHON" avn/scripts/run_avn_reval_search.py \
  --spec avn/experiments/avn_reval_search_v1.json \
  --stage all --batch-id "$NAVTTA_AVN_BATCH" \
  --smt-python "$NAVTTA_AVN_PYTHON" --enmus-python "$NAVTTA_AVN_PYTHON" \
  --dry-run

"$NAVTTA_AVN_PYTHON" avn/scripts/run_avn_reval_search.py \
  --spec avn/experiments/avn_reval_search_v1.json \
  --stage all --batch-id "$NAVTTA_AVN_BATCH" \
  --smt-python "$NAVTTA_AVN_PYTHON" --enmus-python "$NAVTTA_AVN_PYTHON" \
  --preflight-only
```

dry-run 只生成命令计划，不需要 GPU、数据或 checkpoint。预检校验真实数据和 checkpoint 哈希、四张 GPU 可见性、所选 Python/torch 环境、Source 全状态哈希能力以及运行代码摘要；预检本身不执行导航。通过后在 `screen` 中启动完整流程：

```bash
mkdir -p "avn/results/logs/reval_search/$NAVTTA_AVN_BATCH/_launcher"
screen -dmS "$NAVTTA_AVN_BATCH" bash -c '
set -euo pipefail
cd /data1/wxy/code/NavTTA
exec "$NAVTTA_AVN_PYTHON" -u avn/scripts/run_avn_reval_search.py \
  --spec avn/experiments/avn_reval_search_v1.json \
  --stage all --batch-id "$NAVTTA_AVN_BATCH" \
  --smt-python "$NAVTTA_AVN_PYTHON" --enmus-python "$NAVTTA_AVN_PYTHON" \
  >> "avn/results/logs/reval_search/$NAVTTA_AVN_BATCH/_launcher/scheduler.log" 2>&1
'
```

查看状态与调度日志：

```bash
"$NAVTTA_AVN_PYTHON" avn/scripts/run_avn_reval_search.py \
  --batch-id "$NAVTTA_AVN_BATCH" --status
tail -n 80 "avn/results/logs/reval_search/$NAVTTA_AVN_BATCH/_launcher/scheduler.log"
```

恢复同一批次时保持 commit、spec、代码、环境、数据和 checkpoint 不变。在持久终端中执行：

```bash
"$NAVTTA_AVN_PYTHON" -u avn/scripts/run_avn_reval_search.py \
  --spec avn/experiments/avn_reval_search_v1.json \
  --stage all --batch-id "$NAVTTA_AVN_BATCH" \
  --smt-python "$NAVTTA_AVN_PYTHON" --enmus-python "$NAVTTA_AVN_PYTHON" \
  --resume --retry-failed
```

`--resume` 重新验证已有完成证据后跳过有效任务；`--retry-failed` 为失败/中断/失效任务新建 attempt，保留旧 attempt。若只需检查已完成任务并继续未启动任务，可省略 `--retry-failed`。修改任何冻结运行身份后应采用新 batch ID；不能把改代码后的结果接入旧 batch。单独运行 `--stage search` 要求同 batch 已有四个验证通过的 2000-episode Source，可先运行 `--stage reval`，随后通过 `--resume --stage search` 接续。

## 7. 日志与结果分层

```text
avn/results/analysis/reval_search/<batch>/
  batch.json                 # 运行身份、任务/attempt 状态与事件
  metrics.csv                # 全部任务、有效指标、Source delta、诊断和错误
  selected_configs.json      # 8 单元的有效候选数、完成标记与暂定/最终 winner
  summary.md                 # 重评表与搜索概览
avn/results/logs/reval_search/<batch>/
  _launcher/scheduler.log
  <stage>/<model>/<method>/<setting>/<candidate>/attempt-N/
    command.json
    console.log
    result.json              # 紧凑验证证据与配对 Source
avn/results/runs/<run_id>/
  manifest.json              # commit、配置、数据、权重、seed、硬件和 artifacts
  raw/model/tb/
    val_stats_0.json
    tta_diagnostics_0.json
    eval_protocol_0.json     # ENMuS
    audio_schedule_0.json    # ENMuS
```

`batch.json` 是恢复入口；`result.json` 与 manifest 共同验证已完成任务，Source 失效时依赖结果也会失效。运行器只生成本轮目录内的汇总。后续如需将紧凑结果纳入跟踪，应先审核 manifest 与完整单元状态；原始日志、TensorBoard、模型二进制、数据和音频/RIR 不加入 Git。

本地 dry-run 已确认 `12 smoke + 10 reval + 96 search = 118`，且四个搜索轮次均为每卡 6 个、总计 24 个任务。GPU 导航与真实资产预检需在上述服务器环境中执行，其输出才是本轮实际运行证据。
