# R2R Cartesian v2 结果与四方法低学习率补搜设计

更新时间：2026-08-20（Asia/Shanghai）

## 1. 范围与完整性

本报告分析 batch `vln-r2r-modelwise-cartesian-v2-seed0` 在 R2R
`val_seen` 固定 canonical order、seed 0 上的结果。所有超参数选择都以 SR
为第一指标、SPL 为第二指标；不使用 `val_unseen` 或 test 选参。

- 运行 commit：`a258ac5b1d604cbdaecd64fe0ac184241e8384c1`
- 搜索 spec SHA256：
  `642558f1c72cfc5eea1e4d2d0b1eec5873696aad549eb87583e6f082f12e8417`
- ATENA：300/300 validated，0 failed，`complete=true`，`terminal=true`
- ATENA `metrics.csv`：300 行；`TOP5.csv`：15 行；三个模型 winner 齐全
- ATENA admin jobs：300 个 exit code 均为 0
- ATENA tuning jobs：DUET/HAMT/GOAT 各 100 个
- ATENA formal manifests：300 个，全部 `status=completed`、`exit_code=0`
- 300 份 manifest 的 immutable identity 均可重算；按本地对应路径映射后，
  声明的 900 个 result artifacts 均存在且 size/SHA256 匹配
- Scheduler 恢复链：`atena-c4` exit 137 → `atena-c6` exit 137 →
  `atena-goat-c5` exit 0；前两次仅硬切换 scheduler 并接管存活 worker，未产生
  失败或重复 job
- 本地 ATENA 源证据：3,321 个文件、47,344,531 bytes
- 路径与文件内容集合 SHA256：
  `b710c170357c9fa613650e5bedf7304beb07cbabd5027b013fc04123382e9ed9`
- 本地压缩归档：`ATENA_EVIDENCE_BUNDLE_20260820.tar.gz`，6,058,240 bytes
- 压缩归档 SHA256：
  `13b5c7a18324398be4c4b30c651f4ac383521fd5ac8b079b0ba2eab7205cb538`

集合 digest 的算法是：按 repository-relative POSIX path 排序；对每个 regular
file 依次向 SHA256 输入 UTF-8 path、一个 NUL byte、该文件内容 SHA256 的原始
32 bytes；归档文件本身不参与集合 digest。

压缩包是 **ATENA 运行证据归档**：包含 ATENA admin/tuning、300 份 ATENA
formal manifest、三段 ATENA launcher 生命周期和 batch 顶层三个 compact
aggregate。它不包含 matched Source tuning/formal manifest，也不包含其他四种
方法的 raw evidence，因此不是完整五方法或正式 ATENA-vs-Source bundle。

原始 admin、tuning、launcher 与 formal manifest 均保存在 Git 忽略目录；
本报告和补搜协议只记录紧凑结论，不把 raw logs 加入 Git。

## 2. 五种方法相对标准 Source 的结果

标准 argmax Source 的 `SR/SPL` 分别为：DUET `78.84/72.88`、HAMT
`75.61/72.18`、GOAT `84.82/80.05`。下表为每个方法的 SR-first winner，
括号内是相对 Source 的绝对百分点变化 `ΔSR/ΔSPL`。

| 方法 | DUET | HAMT | GOAT |
|---|---|---|---|
| Tent | `78.84/73.05` (`+0.00/+0.17`) | `76.59/73.17` (`+0.98/+0.99`) | `84.92/80.32` (`+0.10/+0.27`) |
| FSTTA | `79.24/72.33` (`+0.40/-0.55`) | `76.30/72.79` (`+0.69/+0.61`) | `84.92/80.21` (`+0.10/+0.16`) |
| EAM | `80.31/74.75` (`+1.47/+1.87`) | `76.69/73.40` (`+1.08/+1.22`) | `85.31/80.55` (`+0.49/+0.50`) |
| FeedTTA† | `78.75/69.34` (`-0.09/-3.54`) | `70.81/66.45` (`-4.80/-5.73`) | `84.82/79.07` (`+0.00/-0.98`) |
| ATENA† | `80.41/76.29` (`+1.57/+3.41`) | `77.47/74.16` (`+1.86/+1.98`) | `84.82/80.13` (`+0.00/+0.08`) |

`†` 表示方法消费 episode 成功/失败二值反馈。ATENA/FeedTTA 不能与前三种
无监督 TTA 合并成一个“最高性能”排名。

主要结论：

1. EAM 是本轮最稳定的无监督方法：三个模型都同时提高 SR 和 SPL。
2. ATENA 在 DUET 和 HAMT 上形成大面积正收益区域；这两个模型上 100 个点中
   分别有 60/59 个点提高 SR，58/58 个点同时提高 SR 和 SPL。
3. GOAT 上 ATENA 的 100 个点没有一个严格提高 SR；winner 只做到 SR 持平、
   SPL `+0.08`。该模型只能做低学习率边界补搜，不能预设一定会优于 Source。
4. FSTTA 在 DUET 的 SR-first winner 牺牲了 SPL；补搜必须同时保留
   `M=8,N=4` 的 joint SR/SPL ridge，而不是只追最高 SR。
5. 固定 action seed 0 下，FeedTTA 的 sampled policy 显示出显著协议差距。
   在相同 action RNG 和其他协议一致时，低学习率极限应趋近 sampled
   no-update control，而不是 argmax Source；该差距仍需多 action seed 确认，
   因此低 LR 只作为受控诊断线。

## 3. ATENA 详细结果

| 模型 | winner 参数 `(lq, ls, λ, δ)` | SR/SPL | ΔSR/ΔSPL | 查询反馈 | self labels |
|---|---|---:|---:|---:|---:|
| DUET | `(1.6e-6, 2e-7, 0.5, 0.0)` | `80.41/76.29` | `+1.57/+3.41` | `1021/1021` (100%) | 0 |
| HAMT | `(3.2e-6, 4e-7, 0.25, 0.1)` | `77.47/74.16` | `+1.86/+1.98` | `796/1021` (77.96%) | 225 |
| GOAT | `(4e-7, 5e-8, 0.0, 0.2)` | `84.82/80.13` | `+0.00/+0.08` | `204/1021` (19.98%) | 817 |

三个 winner 合计查询 2,021/3,063 episodes（65.98%）。整个 300 点搜索查询
185,075/306,300 episodes（60.42%）。DUET winner 通过查询全部 episode 得到
最佳性能，所以补搜除性能 winner 外，还需报告近似性能下的最低 query-rate
Pareto 点。

响应面显示：

- DUET 与 HAMT 已形成明确正收益区域，本轮不再投入 ATENA 补搜预算。
- GOAT 的学习率 pair 和 `λ` 同时位于下边界；下一轮保持 8:1 学习率比例，
  将 query/self LR 向下扩展，且只在该模型启用 ATENA。

## 4. 四方法低学习率补搜

完整候选点、固定参数、停止条件和确认规则见：

[`r2r_five_method_local_refinement_v1.json`](../../../experiments/r2r_five_method_local_refinement_v1.json)

| 方法 | DUET | HAMT | GOAT | 合计 | 主要扩展方向 |
|---|---:|---:|---:|---:|---|
| Tent | 10 | 10 | 10 | 30 | 固定 `update_interval=1`，扫描 `1e-9` 至 paper anchor `1.5625e-5` |
| FSTTA | 72 | 71 | 71 | 214 | fast/slow LR 同时下探；每模型保留 default schedule 与模型 schedule |
| FeedTTA† | 55 | 56 | 56 | 167 | `1e-8` 至 `3e-6`，每模型 3 个 gamma、3 个 SGR profile |
| ATENA† | 0 | 0 | 55 | 55 | 只补 GOAT 的低 LR、低 mixture winner 区域 |
| **搜索合计** | **137** | **137** | **192** | **466** | — |

另运行每模型一次标准 argmax Source 和一次 FeedTTA sampled no-update control，
共 6 个 controls；确认实验前总量为 472 jobs。EAM 已有三个模型均同时提高
SR/SPL，下一轮不再补搜。Tent/FSTTA/FeedTTA 均保留父批次锚点，ATENA 仅保留
GOAT 锚点；所有方法按模型严格串行。

FeedTTA 的 sampled-control 分解为：

- DUET：control `77.38/67.43` → winner `78.75/69.34`，适配本身
  `+1.37/+1.91`，但仍低于 argmax Source `-0.09/-3.54`。
- HAMT：control `69.25/64.68` → winner `70.81/66.45`，适配本身
  `+1.56/+1.77`，但 argmax Source 差距仍很大。
- GOAT：control `83.55/78.67` → winner `84.82/79.07`，SR 已追平
  argmax Source，SPL 尚差 `0.98`。

因此不运行原先 108 点公共低-LR Cartesian；改用 167 个 model-wise 低 LR
候选，其中每个模型仍明显少于首轮的 250 点。
promotion 以新批次内重跑的 parent-winner anchor 为基准：只有当新点的精确
成功 episode 数至少增加 1，或成功数持平且连续指标 SPL 至少提高预注册的
0.1 pp 容差，才进入多 seed 确认；与历史 parent metric 的比较只用于报告
batch drift。R2R 的单 episode SR 量子为
`100/1021=0.097943 pp`；不能对两位小数 SR 直接使用 `>=0.1` 阈值。

## 5. 严格模型串行与并发

下一轮采用模型主序：

```text
DUET：Source → Tent → FSTTA → FeedTTA control → FeedTTA
HAMT：上一模型全部归零后执行同一顺序
GOAT：Source → Tent → FSTTA → FeedTTA control → FeedTTA → ATENA
```

只设置 `max_per_model` 不能形成模型 barrier；旧 Cartesian runner 会跳过已满
模型并发射其他模型。新的 `run_r2r_local_refinement.py` 已按 16 个 filtered
phase 实现严格 model/method barrier，并由定向测试验证 472-job 展开、phase
顺序、失败阻断和 resume 校验。只有活动 method/model 不可避免的最后一个
partial wave 可以低于并发上限，不能为了填槽提前启动下一方法或模型。Source
和 FeedTTA sampled control 每次固定单任务运行，不参与并发校准。

下表是校准 baseline 和条件测试上限；除明确标为纯模型实测的两项外，均由混合
composition 或更低纯模型并发推算，必须逐级实测后才能成为生产 cap。`*` 项
距离 29,000MiB 计划线较近，只有前一级纯模型稳态边际足够时才允许测试。

| 方法 | DUET | HAMT | GOAT |
|---|---:|---:|---:|
| Tent | `5→10` | `5→10` | `5→10` |
| FSTTA | `5→14` | `5→12` | `5→14` |
| FeedTTA | `5` | `5→6*` | `5→6` |
| ATENA | — | — | `5`（纯模型实测 peak 26,495MiB；不测试 6） |

任何生产计划的稳态 GPU used 不超过 29,000MiB，从而保留约 3,760MiB
headroom；若观测值达到 30,000MiB，立即停止升级并回退，30,000MiB 不是可计划
目标。每次只增加一个 worker，等待所有 GPU 子进程完成模型加载后再判断。
`--max-gpu-memory-mib` 应按 method/model 设在 23,500–25,000MiB，29,000/
30,000 绝不能直接填入该参数。发射前门槛只是二级保护，不能替代硬并发上限，
因为冷启动期间新进程可能暂时只显示约 238MiB。

## 6. 解释边界

- 所有差值来自单一固定顺序 seed 0 的调参证据；SR 单 episode 量子为
  `0.097943 pp`，尚未估计跨 seed 方差。
- 单 episode 增益或持平结果必须经过 seeds 1、2 的 matched-control 确认。
  FeedTTA 使用配对 `(order, action, sgr)=(1,1,1),(2,2,2)`，不是 2×2
  笛卡尔积；每个 replica 还需同 order/action seed 的 sampled no-update
  control（control 不含 SGR seed）。
- `val_unseen` 只用于冻结参数后的最终确认，不能回流选参。
- 当前 Cartesian 布局尚无兼容的 post-hoc late-collapse 工具；完整 manifest
  证明运行与产物有效，但不能替代后段稳定性审计。
- FeedTTA/ATENA 必须继续标为 binary-feedback-supervised TTA，并报告反馈预算。
