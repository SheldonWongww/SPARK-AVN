# StreamVLN v5 搜索与固定配置复验

规格由 [streamvln_val_unseen_search_v5.json](streamvln_val_unseen_search_v5.json) 定义。只搜索 FSTTA、EAM、FeedTTA、ATENA，各 4 组，共 16 次完整 R2R-CE `val_unseen`。Tent 固定使用旧 `streamvln-vu-compact-tent-01-v4` 的配置及 `legacy_v4` readout；IDEA 不在本轮范围内。

本轮目标是分清实际参数写入、行动变化和最终导航成绩，并在固定预算内检查学习率与各方法一个关键机制的交互。搜索不保证涨点。`val_unseen` 是本轮开发集，不能将其搜索最佳值描述为未参与选参的测试成绩。

## 旧结果的用途

v4 的逐配置证据见 [legacy 审计摘要](../results/legacy/streamvln_compact_v4_audit/ANALYSIS.md)，结果、配置和来源摘要见 [summary.json](../results/legacy/streamvln_compact_v4_audit/summary.json)。原始档案位于 `vln/results/legacy/streamvln_compact_v4_audit/raw/`；来源为 `中间结果/streamvln_val_unseen_compact_v4.zip` 和 `中间结果/streamvln-vu-compact-source-v4.zip`。旧结果仅用于诊断和明确标注的开发对照，不能因复用就自动升级为正式研究结果。

已读取的 legacy 诊断支持以下调整：

- FeedTTA 前两组各完成 1,839 条、进行 1,839 次 optimizer update，但最终 `relative_param_drift=0`；这说明更新计数不能代表有效参数变化。两组 `alpha=+0.1` 实际为 gradient scaling，本轮统一使用 `alpha=-0.2` 的 stochastic gradient reversion。
- FSTTA 最低档的 459 次 slow 尝试中跳过了 458 次；中档跳过 139 次。需要增加有效更新幅度，同时分离 slow 窗口长度的影响。
- EAM 三组均完成 1,839 条，其结果文件相同；最高档辅助参数相对漂移约 `0.00271665`，但融合从未改变其 Source 分支的行动。这是门控和决策变化的问题，不能一概解释成没有梯度。
- ATENA 三组档案分别只有 284、312、294 条，不能与完整 Source 的总体成绩直接比较。query 比例分别约为 96.1%、63.5%、99.7%；旧最高档几乎没有 self 分支。

这些数值描述旧档案的运行状态，不作为正式性能表。旧 v4 直接使用 FP32 readout，也需要与原生 decoder 的数值路径区别开来；多个方法相同的小幅 SR 变化不能直接归因于适应。

## readout 与方法范围

四个新方法统一声明 `streamvln_readout_protocol=native_residual`。原生模型保持冻结，适应变量仍是其最后 RMSNorm 的 FP32 权重副本，四个行动 token 的 head 权重冻结。以同一生成状态的原生四行动 scores 为锚：

```text
adapted_scores = native_scores + (adapted_fp32_readout - frozen_fp32_readout)
```

先计算括号内差值，使未发生参数更新时残差为零。RMSNorm 权重仍直接优化，不引入 delta-gamma 参数化。必须同时保留原生行动 token 的平票顺序，避免仅因四维数组的排列改变 argmax。

这是最后 RMSNorm 上的轻量适应，不能标注为整个 StreamVLN、跨模态模块或完整导航策略的更新。FeedTTA 使用 `scope_profile=configured_prefixes`，实际前缀为 `norm`。ATENA 的 replay reachable 范围也仅包含该 readout；它的自预测辅助头单独训练。ATENA online 与 replay 都用 `norm(hidden)` 作为 features，使 self-prediction loss 能回传至适应的 norm。

适应单位是每个 `generate()` 调用的第一个行动 token；该 action chunk 的后续行动仍由冻结模型生成。因此 FSTTA 的 M、EAM 的 replay/interval、FeedTTA 的 gamma 都以生成调用为单位，不能解释成每一个 `env.step`。FeedTTA 与 ATENA 消耗二元 episode 反馈，报告时保留与 FSTTA/EAM 无监督适应的区别。

EAM 和 ATENA 的 replay 输入在入库前保存 detached hidden 和原生四行动 scores；永久冻结的 readout 参考权重保存在模型 buffer 中。历史条目使用自己的 scores 和同一冻结参考，不能复用当前时刻 scores，也不能把入库时已适应的参数当成 Source。EAM source 和 auxiliary 分支分别在同一历史锚上评估；source 应精确还原记录的 native scores。

## 四组配置

所有配置固定 seed 0、native greedy/argmax 行动和 continual episode 协议，`max_grad_norm=1.0`。每行是一个 2×2 网格，其余参数固定；完整参数以 JSON 为准。

| 方法 | 因子一 | 因子二 | 固定项及目的 |
|---|---|---|---|
| FSTTA | fast LR `3e-7 / 3e-6`，slow LR 始终为其 3 倍 | N `4 / 16` | M=3，GDA、PDA、q=0.1、rho=0.95、tau=0.7、a/b=0.9/1.1，保留原有 optimizer/reset 规则；分离更新幅度与 slow 周期 |
| EAM | LR `1e-6 / 3e-6` | confidence scale `0.4 / 0.8` | memory=32、batch=8、interval=1，Adam；固定旧 02/03 的 replay 形状，检查门控是否限制行动变化 |
| FeedTTA | LR `1e-6 / 3e-6` | gamma `0.7 / 0.95` | p=0.05、alpha=-0.2、SGR seed=0、Adam eps=1e-5、非归一化梯度；分离更新幅度与轨迹信用分配长度 |
| ATENA | query LR `3e-7 / 3e-6`，self LR 始终为其 0.1 倍 | query threshold `0.3 / 0.5` | mixture lambda=0.25、self loss weight=0.1、AdamW、weight decay=0；检查 query/self 频率与更新幅度 |

EAM 使用 `confidence_scale × ln(4)` 的熵阈值。两档约为 `0.5545 / 1.1090`；低档会排除熵约为 `ln(2)` 的两行动近似平票状态，高档允许更多此类状态参与可靠样本更新或辅助融合。高阈值也可能引入错误伪标签，需要同 LR 的低阈值对照。

ATENA 低档的 self LR 为 `3e-8`，仍可能低于部分 FP32 norm 权重的写入分辨率，作为保守下界保留。必须结合 query/self 次数及实际参数变化解释；辅助头学习或 optimizer 调用次数不等于部署策略已有效更新。高档 self LR 为 `3e-7`。若低档仍近乎不动，如实记录，不用 `val_seen` 成绩临时改变其学习率。

## 阶段、预算与选择

1. 复用旧 `val_unseen` Source 和固定 Tent。JSON 固定了各自结果路径及 SHA256；读取时核对摘要、1,839 个唯一 episode、完整结束记录及既定评测配置。文件缺失、摘要不符或只有 partial 不能冒充可复用的完整结果。
2. `search` 运行四方法各 4 组，共 16 次完整 `val_unseen`，每次 1,839 条。仅使用完成且有效的运行，以 SPL 为第一指标、SR 为第二指标，在每个方法内选出一个 winner。不能用执行中前缀的均值参与排名。
3. 固定这 4 个 winner 后，`seen` 跑 6 次完整 `val_seen`，每次 778 条：Source、固定 Tent、四方法 winner。`val_seen` 用于检查固定设置表现，不再选参或替换 winner。
4. `report` 汇总完整度、复用来源、诊断、每方法的四点比较及固定设置结果。`all` 顺序执行搜索和固定配置复验，预算为 16 次 `val_unseen` 加 6 次 `val_seen`；省略 `--stage` 时仅执行 `search`。旧 Source/Tent 的 `val_unseen` 不重新搜索。

Source 保留原生解码，并复用已指定的旧完整 `val_unseen` 结果。Tent 在两种 split 都保持旧参数和 `legacy_v4` 数值路径，不自动迁移到新 residual readout。报告必须区分这个固定 legacy Tent 与四个新方法的 readout 协议；不能把二者的差距单独解释成算法优劣。

每张 GPU 同时最多一个完整 StreamVLN 评测进程；每个进程使用单环境。4 张卡最多并行 4 个 job，不在同卡并发多个 7B 实例。每条作业从同一初始 checkpoint 开始，模型、优化器、replay 和 RNG 不在不同候选或 split 间继承。

## 执行接口

待旧实验释放所选 GPU 后，在服务器更新代码并使用已有 StreamVLN 环境。无需升级服务器 PyTorch/Transformers：

```bash
cd /data1/wxy/code/NavTTA
git pull --ff-only origin main
STREAM_PY=/data1/wxy/exp_data/NavTTA/vln/envs/streamvln/bin/python
"$STREAM_PY" -m pip install -e core
"$STREAM_PY" vln/scripts/run_streamvln_search_v5.py --stage all --gpus 0,1,2,3 --dry-run
```

推荐一次后台执行全部阶段：

```bash
mkdir -p vln/results/tuning/streamvln_val_unseen_search_v5
nohup "$STREAM_PY" -u vln/scripts/run_streamvln_search_v5.py \
  --stage all --gpus 0,1,2,3 \
  > vln/results/tuning/streamvln_val_unseen_search_v5/scheduler.log 2>&1 &
```

也可在持久终端中分别执行以下阶段；它们与 `all` 是替代关系：

```bash
"$STREAM_PY" vln/scripts/run_streamvln_search_v5.py --stage search --gpus 0,1,2,3
"$STREAM_PY" vln/scripts/run_streamvln_search_v5.py --stage seen --gpus 0,1,2,3
"$STREAM_PY" vln/scripts/run_streamvln_search_v5.py --stage report
```

运行中断后，在相同命令追加 `--retry-incomplete`，归档 partial 后从第一个 episode 全新执行受影响作业。continual TTA 不能只跳过已有 episode 后续跑，因为缺少前缀的模型、优化器、replay 和 RNG 状态。完整成功的作业仍按其配置、manifest 及结果摘要验证后跳过。重复启动有 GPU 与作业两层锁保护；旧版本任务通过 `nvidia-smi` 的进程检查识别，已有计算进程的 GPU 不会启动新评测。

日志和汇总位于 `vln/results/tuning/streamvln_val_unseen_search_v5/`：`scheduler.log` 看总体进度，`launcher_logs/<tag>.log` 看启动失败原因，`jobs/<tag>/<split>/` 保存评测结果和诊断；Source seen 在 `vln/results/source/streamvln-vs-source-frozen-v5/streamvln-r2r-ce/val_seen/`。`selection.json` 固定四个 winner，`report.json` 给出参数和诊断，`report.md` 给出完整运行的指标与 Source 差值。相同结果文件会在 JSON 中单独列组。

脚本按 archive SHA256 复用旧 Source/Tent 结果；若服务器文件只是移动了位置，用 `--source-result` 或 `--tent-result` 指定新路径，内容摘要仍须相同。新任务启动时校验实际模型 shards、视觉塔、tokenizer 和数据摘要，核验记录写入 `vln/manifests/generated/`，各 run manifest 引用它；相关运行代码须与记录的 Git commit 一致。

## 诊断与记录

零更新校验应保留 native scores、native argmax、固定参考 readout 与适应 readout 的区别，并覆盖平票。有效学习检查同时看实际 norm 参数变化、logit 残差和相对 native 的行动翻转；旧 `overridden_first_actions` 对多种方法恒为零，不能继续充当通用学习有效性指标。EAM 另记录融合相对冻结 Source 的翻转，避免混淆门控开启与行动改变。

方法诊断至少检查：FSTTA fast/slow 次数与 slow 跳过原因；EAM source/aux gate、可靠样本量、辅助漂移和融合翻转；FeedTTA 成功/失败反馈数、SGR 反转比例与有效更新；ATENA query/self 比例、两种分支的更新及 self-loss 对 norm 的连通性。发生数值失效或完整度不足的作业不能参与 winner 选择；有效但没有提升或没有行动翻转的作业仍要保留结果，不能因其“不好看”删除。

配置、checkpoint、数据版本、seed、硬件、顶层 Git commit 和 readout 协议写入每次运行 manifest；复用项保留原始路径、摘要和 provenance 状态。数据与 checkpoint 的 provenance/SHA256 归任务 `manifests/` 管理。原始日志、视频、资产和 checkpoint 不入 Git，legacy 或来源不完整的指标继续留在 `results/legacy/`，不进入正式表格。
