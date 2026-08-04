# AVN 主对比表结果登记册

更新日期：2026-08-04

用途：集中登记未来论文 AVN 主对比表采用的冻结结果。表格以导航模型为一级
分组，列按单声源和多声源展开。当前填入统一 Source 重评估结果、冻结配置的
Tent 候选结果，以及已经确定的 FSTTA/EAM 主表结果；`—`
表示该方法尚未完成最终配置复验或尚不具备可写入主表的结果。
ENMuS FSTTA single-source 使用搜索 job 0 的已有运行，multi-source 使用同一
超参数配置的迁移运行；两者均为当前最终选定数值。由于本地 run manifest
与 checkpoint 训练 provenance 尚未闭环，继续标记为 `[P]`。
SMT+Audio FSTTA multi-source 使用其 single-source 冻结配置的直接迁移结果；
EAM 在两个模型上也均将 single-source 搜索后冻结的配置直接迁移到
multi-source，四项结果均已登记。

## 1. 指标与登记规则

- `Reward`、`SR`、`SPL`、`SoftSPL`、`SNA`、`SWS` 越高越好。
- `DTG`、`NDTG`、`NA` 越低越好。
- `SR`、`SPL`、`SoftSPL`、`SNA`、`SWS` 在表中按百分制记录；其余指标保留
  原始量纲。
- AV-Nav 和 SAVi 当前只登记 Source。SMT+Audio 和 ENMuS 登记 Source 及
  Tent、FSTTA、EAM、FeedTTA、ATENA。
- Tent、FSTTA、EAM 属于无监督 TTA；`FeedTTA†` 和 `ATENA†` 使用二值 episode
  feedback，论文中必须与无监督 TTA 明确区分。
- 主对比采用一个预先冻结的 seed-0 episode 顺序；顺序鲁棒性和长流实验另表
  报告，不在本表内取多顺序平均。

## 2. 论文主对比表草案

<table>
  <thead>
    <tr>
      <th rowspan="2">导航模型</th>
      <th rowspan="2">方法</th>
      <th colspan="9">单声源（2000 episodes）</th>
      <th colspan="9">多声源（2000 episodes）</th>
    </tr>
    <tr>
      <th>Reward↑</th><th>DTG↓</th><th>NDTG↓</th><th>SR↑</th><th>SPL↑</th><th>SoftSPL↑</th><th>NA↓</th><th>SNA↑</th><th>SWS↑</th>
      <th>Reward↑</th><th>DTG↓</th><th>NDTG↓</th><th>SR↑</th><th>SPL↑</th><th>SoftSPL↑</th><th>NA↓</th><th>SNA↑</th><th>SWS↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AV-Nav</td><td>Source [P]</td>
      <td>10.227417</td><td>5.2580</td><td>0.319561</td><td>50.55</td><td>26.9668</td><td>33.3274</td><td>222.7060</td><td>36.5014</td><td>2.65</td>
      <td>5.051718</td><td>7.6190</td><td>0.571998</td><td>19.40</td><td>9.3109</td><td>24.7731</td><td>129.3270</td><td>11.8698</td><td>2.05</td>
    </tr>
    <tr>
      <td>SAVi</td><td>Source [P]</td>
      <td>10.154002</td><td>5.1055</td><td>0.321816</td><td>45.35</td><td>27.1933</td><td>41.8248</td><td>120.2985</td><td>34.6512</td><td>2.20</td>
      <td>5.345304</td><td>7.5840</td><td>0.557804</td><td>22.65</td><td>11.6291</td><td>27.6400</td><td>101.4685</td><td>16.0290</td><td>1.80</td>
    </tr>
    <tr>
      <td rowspan="6">SMT+Audio</td><td>Source [P]</td>
      <td>11.480265</td><td>5.0005</td><td>0.306263</td><td>54.15</td><td>29.4229</td><td>36.5105</td><td>153.1720</td><td>42.9696</td><td>3.20</td>
      <td>5.299168</td><td>7.7375</td><td>0.562006</td><td>25.90</td><td>13.4196</td><td>24.8589</td><td>164.7320</td><td>19.1793</td><td>1.45</td>
    </tr>
    <tr>
      <td>Tent [P]</td>
      <td>11.726524</td><td>4.8190</td><td>0.293822</td><td>55.80</td><td>29.8232</td><td>36.8647</td><td>153.6960</td><td>43.9578</td><td>2.75</td>
      <td>5.338717</td><td>7.7215</td><td>0.559223</td><td>26.10</td><td>13.0639</td><td>24.1813</td><td>173.3770</td><td>19.0262</td><td>1.35</td>
    </tr>
    <tr>
      <td>FSTTA [P]</td>
      <td>11.890811</td><td>4.7265</td><td>0.289784</td><td>56.55</td><td>30.3007</td><td>37.0208</td><td>155.0175</td><td>44.8928</td><td>2.60</td>
      <td>5.380221</td><td>7.6480</td><td>0.556494</td><td>26.25</td><td>13.5170</td><td>25.6032</td><td>172.5765</td><td>19.2579</td><td>1.95</td>
    </tr>
    <tr>
      <td>EAM [P]</td>
      <td>11.834155</td><td>4.8325</td><td>0.295396</td><td>56.20</td><td>30.5805</td><td>37.4332</td><td>150.0830</td><td>45.2148</td><td>2.30</td>
      <td>5.716312</td><td>7.5095</td><td>0.538814</td><td>27.50</td><td>14.1282</td><td>25.3413</td><td>166.3175</td><td>20.2154</td><td>1.30</td>
    </tr>
    <tr><td>FeedTTA†</td><td colspan="9">—</td><td colspan="9">—</td></tr>
    <tr><td>ATENA†</td><td colspan="9">—</td><td colspan="9">—</td></tr>
    <tr>
      <td rowspan="6">ENMuS</td><td>Source [P]</td>
      <td>14.658422</td><td>3.2115</td><td>0.195937</td><td>66.55</td><td>36.0480</td><td>40.2185</td><td>164.2560</td><td>51.1794</td><td>2.20</td>
      <td>6.925159</td><td>7.3140</td><td>0.572916</td><td>34.75</td><td>17.1118</td><td>24.9445</td><td>159.4825</td><td>25.8528</td><td>1.00</td>
    </tr>
    <tr>
      <td>Tent [P]</td>
      <td>14.664061</td><td>3.2565</td><td>0.200093</td><td>67.35</td><td>36.6857</td><td>40.3085</td><td>166.1920</td><td>51.2352</td><td>2.50</td>
      <td>7.261280</td><td>7.0725</td><td>0.553091</td><td>35.55</td><td>17.1682</td><td>25.4856</td><td>159.0205</td><td>26.0775</td><td>1.10</td>
    </tr>
    <tr>
      <td>FSTTA [P]</td>
      <td>15.041523</td><td>3.1220</td><td>0.196867</td><td>68.55</td><td>37.3752</td><td>41.1575</td><td>161.3960</td><td>52.3454</td><td>2.05</td>
      <td>7.178199</td><td>7.1615</td><td>0.561001</td><td>35.85</td><td>17.5484</td><td>25.3055</td><td>159.4285</td><td>26.4713</td><td>1.25</td>
    </tr>
    <tr>
      <td>EAM [P]</td>
      <td>14.859267</td><td>3.2590</td><td>0.193446</td><td>68.15</td><td>36.9083</td><td>40.6694</td><td>162.9215</td><td>51.8355</td><td>2.20</td>
      <td>7.193380</td><td>7.1725</td><td>0.550503</td><td>35.40</td><td>17.1222</td><td>25.1913</td><td>156.8105</td><td>26.4333</td><td>0.90</td>
    </tr>
    <tr><td>FeedTTA†</td><td colspan="9">—</td><td colspan="9">—</td></tr>
    <tr><td>ATENA†</td><td colspan="9">—</td><td colspan="9">—</td></tr>
  </tbody>
</table>

`[P]`：证据链尚未闭环，暂不具备正式论文结果资格。该标记不代表超参数
尚未选定；当前已登记的 FSTTA 和 EAM 数值均为最终选定结果。

`†`：该方法消费二值 episode feedback，不属于严格无监督 TTA。

## 3. 当前 Source 批次

| 项目 | 当前记录 |
|---|---|
| Batch ID | `source-reval-v1-seed0` |
| Git commit | `48ea6285fed8de7e6d67fe0eba84652333d452b7` |
| Seed | `0` |
| 每组 episode 数 | `2000` |
| 单声源顺序 SHA256 | `07f327590ccee2999b3f6bcb2fc412f39d9802cf932b14933fd0bdd9e5ca380c` |
| 多声源顺序 SHA256 | `cc2f1ce8319fae6a1313750c2b1235ac39985e8d2fe6270a70fda7b12d2a6525` |
| 执行完整性 | 8/8 完成，exitcode 均为 0，9 项聚合指标均可解析 |
| 当前 provenance | `legacy_checkpoint_provenance_incomplete` |
| 本地指标来源 | [`logs/source_reval/source-reval-v1-seed0/metrics.csv`](logs/source_reval/source-reval-v1-seed0/metrics.csv) |

当前 `metrics.csv` 指向的逐运行 manifest 未随日志下载到本地，checkpoint 的训练
provenance、数据版本和硬件信息也未形成符合项目规范的正式 run manifest。因此
本页先保存数值和主表结构，但这些 Source 行仍属于候选值；完成 provenance 补录
或按最终代码重新评估前，不得去掉 `[P]`，也不得直接复制到论文正式主表。

## 4. 当前 Tent 配置与批次

| 项目 | 当前记录 |
|---|---|
| 冻结配置 | `NORM_SCOPE=ln`、`LR=1e-8`、`UPDATE_INTERVAL=1`、`EPISODIC=False`、`STEPS=1` |
| 优化器 | Adam，betas=(0.9, 0.999)，weight decay=0，max grad norm=1.0 |
| Seed / episode 顺序 | `0` / 与对应 Source 重评估一致 |
| 每组 episode 数 | `2000` |
| SMT+Audio 单声源来源 | [`tent-core-v1-seed0/j090`](logs/tent_core_grid/tent-core-v1-seed0/jobs/tentcore-tent-core-v1-seed0-j090-ln-lr1em8-u1/) |
| ENMuS 单声源来源 | [`tent-core-enmus-v1-seed0/j090`](logs/tent_core_grid/tent-core-enmus-v1-seed0/jobs/tentcore-tent-core-enmus-v1-seed0-j090-ln-lr1em8-u1/) |
| 多声源来源 | [`tent-main-multi-v1-seed0/metrics.csv`](logs/tent_main_multi/tent-main-multi-v1-seed0/metrics.csv) |

single-source 数值复用冻结配置在核心超参数网格中的既有运行；multi-source 数值
来自配置冻结后执行的主表批次。当前下载到本地的 multi-source `metrics.csv` 虽然
记录了逐运行 manifest 路径，但对应 manifest 未一并下载；single-source 目录也只
含日志和参数文件。因此四项 Tent 结果暂标为 `[P]`，补齐 commit、checkpoint digest、
数据版本、硬件及完整配置的 run manifest 后才能转为正式结果。

## 5. 当前 FSTTA 候选与复验进度

当前冻结候选来自 SMT+Audio 单声源机制探索的 validated job 35：

| 项目 | 当前记录 |
|---|---|
| 参数范围 | 最后 4 个 LayerNorm affine 参数（`NORM_SCOPE=last_k_ln`、`LAST_K_LN=4`） |
| FAST | `LR=3e-7`、`M=16`、`FAST_GRAD_MODE=concordant`、启用 LR scaler |
| SLOW | `LR_SLOW=1e-4`、`N=32`、`Q=0.1`、启用 SLOW、persistent AdamW |
| 论文式固定项 | `RHO=0.95`、`TAU=0.7`、`A=0.9`、`B=1.1` |
| 持续适应 | `EPISODIC=False`、`STEPS=1` |
| 优化器 | FAST/SLOW AdamW，betas=(0.9, 0.99)，weight decay=0，max grad norm=1.0 |
| 开发结果来源 | [`fstta-exploration job 35`](logs/fstta_exploration/fstta-exploration-all-v1-seed0/jobs/fsttaexp-fstta-exploration-all-v1-seed0-sb-j035-flr3em7-m16-slr1em4-n32-q0p1-gconcordant-sc1-us1-soaw-wr0/) |

该 single-source 数值用于记录 SMT+Audio 配置冻结依据，仍属于开发批次
结果，故暂标 `[P]`。该配置冻结后只在 SMT+Audio 的其他声源设置上直接
迁移，不再重新选参；SMT+Audio single-source 也应在最终固定 commit 上独立
复验后才能移除 `[P]`。ENMuS 对该数值配置的迁移失败，因此单独使用
ENMuS single-source 开发网格冻结模型级配置，再原样迁移到 ENMuS
multi-source。

SMT+Audio multi-source 已使用上述 job 35 配置直接迁移完成：

| 项目 | 当前记录 |
|---|---|
| Batch / job | `fstta-main-v1-seed0 / j00-smt_audio-multi_source` |
| Git commit / worktree | `816f253e2bad892da44634a92563c2881cd4cf9e` / clean |
| Seed / episodes | 0 / 2000 |
| checkpoint SHA256 | `c5c039a35da13a58a8f771738208603c93d3727dbebbea0a6dbfcccd16bdddd8` |
| stream-order SHA256 | `cc2f1ce8319fae6a1313750c2b1235ac39985e8d2fe6270a70fda7b12d2a6525` |
| stream-content SHA256 | `deab5e0c91abeb999563927b6c80c05bc6dfcbdd94b455b815bd303386918f2d` |
| 完整性 | 2000 episodes，exitcode=0，`validation=ok` |
| 结果 | SR/SPL/SoftSPL=`26.25/13.5170/25.6032` |
| 本地证据 | [`j00 console.log`](logs/fstta_main/fstta-main-v1-seed0/jobs/j00-smt_audio-multi_source/console.log) |

该 checkpoint 与 episode stream digest 均和 matched Source 一致。最终 console 汇总为
`success=0.262500`（525/2000）；批次 `metrics.csv` 将该字段误记为 `0.26270`，
因此本表以原始 console 为准。相对 Source，SR/SPL 提高 0.35/0.0974
个百分点；提升较小，但这是冻结配置未重新调参的直接
迁移结果，因此按用户确定的规则作为 SMT+Audio multi-source FSTTA 最终数值。
本地仍缺逐运行 manifest 和 checkpoint 训练 provenance，故保留 `[P]`。

ENMuS 使用其模型级 single-source 网格按本报告 SPL-first 规则选出的 job 0：

| 项目 | 当前记录 |
|---|---|
| 参数范围 | 最后 4 个 LayerNorm affine 参数（8 个张量） |
| FAST | `LR=1e-8`、`M=16`、`FAST_GRAD_MODE=concordant`、启用 LR scaler |
| SLOW | `LR_SLOW=1e-5`、`N=32`、`Q=0.1`、persistent AdamW |
| single-source 结果 | job 0 已有运行；SR/SPL=`68.55/37.3752` |
| single-source 来源 | [`fstta_enmus_grid job 0`](logs/fstta_enmus_grid/fstta-enmus-grid-avn-val-rerun-v2-seed0/jobs/fsttaexp-fstta-enmus-grid-avn-val-rerun-v2-seed0-ei-j000-flr1em8-m16-slr1em5-n32-q0p1-gconcordant-sc1-us1-soaw-wr0/) |
| multi-source batch | `fstta-enmus-multi-v1-seed0`，1/1 完成且 `validation=ok` |
| Git commit | `f27e257cf9d5dc323919b1632fe51c1b369d9ee3`，clean worktree |
| Seed / episodes | 0 / 2000 |
| stream-order SHA256 | `cc2f1ce8319fae6a1313750c2b1235ac39985e8d2fe6270a70fda7b12d2a6525` |
| 本地证据 | [`fstta_enmus_multi`](logs/fstta_enmus_multi/fstta-enmus-multi-v1-seed0/) |

该配置就是 ENMuS single-source 网格的 job 0。其已有运行作为单声源
FSTTA 最终结果；随后不根据多声源结果重新调参，直接将 job 0 配置迁移到
multi-source。单声源 SR/SPL 相对 matched Source 提高 2.00/1.3272 个百分点；
多声源提高 1.10/0.4366 个百分点。两组数值均已填入主表。当前下载
内容只有 run manifest 的服务器路径指针，且 Source checkpoint provenance
不完整，所以仍保留 `[P]`。

## 6. 当前 EAM 模型级最终配置

两个模型均采用 `eam-boundary-joint-v1-seed0` single-source 边界网格中各自
SR/SPL 同时最高的配置。按当前研究决定，这两个已有运行直接作为 EAM
single-source 最终数值，不再另选统一 LR：

| 模型 | job | LR | UPDATE_INTERVAL | Reward | DTG | NDTG | SR | SPL | SoftSPL | NA | SNA | SWS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SMT+Audio | 5 | `1e-8` | 128 | 11.834155 | 4.8325 | 0.295396 | 56.20 | 30.5805 | 37.4332 | 150.0830 | 45.2148 | 2.30 |
| ENMuS | 11 | `3e-9` | 128 | 14.859267 | 3.2590 | 0.193446 | 68.15 | 36.9083 | 40.6694 | 162.9215 | 51.8355 | 2.20 |

共同固定项为 `scope=full_transformer_plus_head`、`CONFIDENCE_SCALE=0.4`、
`MEMORY_SIZE=32`、`BATCH_SIZE=8`、`EPISODIC=False`、`STEPS=1`，优化器为
Adam，`betas=(0.9, 0.999)`、零 weight decay、无梯度裁剪。两项运行均位于
commit `f27e257cf9d5dc323919b1632fe51c1b369d9ee3` 的 clean worktree，使用同一
canonical single-source val、seed 0、2000-episode stream：

| 项目 | SMT+Audio | ENMuS |
|---|---|---|
| checkpoint SHA256 | `8007dc0de8b0e994244d4f2fdb4a642bcc6213b4e9694568c93b10141f53ef03` | `4f37a377cc7fcb888c545850c91883560a908ba5366072df787e4c8238ecefcd` |
| stream-order SHA256 | `07f327590ccee2999b3f6bcb2fc412f39d9802cf932b14933fd0bdd9e5ca380c` | 同左 |
| stream-content SHA256 | `dd411c4aafaf626b2848d20b92d1832ea46a5380c57107043fc639996837fdf2` | 同左 |
| 本地证据 | [`SMT job 5`](logs/eam_boundary_grid/eam-boundary-joint-v1-seed0/smt_audio/jobs/eamboundary-eam-boundary-joint-v1-seed0-j005-smt_audio-lr1em8-u128/) | [`ENMuS job 11`](logs/eam_boundary_grid/eam-boundary-joint-v1-seed0/enmus/jobs/eamboundary-eam-boundary-joint-v1-seed0-j011-enmus-lr3em9-u128/) |

相对 matched Source，SMT+Audio 的 single-source SR/SPL 提高
2.05/1.1576 个百分点，ENMuS 提高 1.60/0.8603 个百分点。

`eam-main-multi-v1-seed0` 已在 commit
`99f46dfa3010ca41d0a2898db1e88e1934ae5bb9` 的 clean worktree 上完成，
2/2 jobs 均为 `exitcode=0` 且 `validation=ok`。两个模型使用各自
single-source 冻结配置，没有在 multi-source 上重新选参：

| 模型 | LR | UPDATE_INTERVAL | Reward | DTG | NDTG | SR | SPL | SoftSPL | NA | SNA | SWS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SMT+Audio | `1e-8` | 128 | 5.716312 | 7.5095 | 0.538814 | 27.50 | 14.1282 | 25.3413 | 166.3175 | 20.2154 | 1.30 |
| ENMuS | `3e-9` | 128 | 7.193380 | 7.1725 | 0.550503 | 35.40 | 17.1222 | 25.1913 | 156.8105 | 26.4333 | 0.90 |

multi-source 相对 matched Source，SMT+Audio 的 SR/SPL 提高
1.60/0.7086 个百分点，ENMuS 提高 0.65/0.0104 个百分点。后者
SPL 几乎持平，且低于 Tent/FSTTA，说明 EAM 的跨声源迁移增益具有
模型依赖性。批次使用 multi-source checkpoint SHA256
`c5c039a35da13a58a8f771738208603c93d3727dbebbea0a6dbfcccd16bdddd8` /
`3b1ccc9421b8fd6b9bad8a165528fa2323161d8b3c74642be13b2a59a5ae0464`，
dataset index SHA256 为
`45d8dbdea540e78b01b252a3958afc4657745374d45185100d731df6a6cb849d`，
stream order/content SHA256 分别为
`cc2f1ce8319fae6a1313750c2b1235ac39985e8d2fe6270a70fda7b12d2a6525` /
`deab5e0c91abeb999563927b6c80c05bc6dfcbdd94b455b815bd303386918f2d`，
均与 matched Source 一致。

四个 EAM 数值已经冻结，但因本地缺少可解析 run manifest 及完整
checkpoint 训练 provenance，暂继续标为 `[P]`。multi-source 证据位于
[`eam_main`](logs/eam_main/eam-main-multi-v1-seed0/)。

## 7. 后续填表规则

1. 每种 TTA 方法先冻结配置，再分别运行 SMT+Audio/ENMuS 的单声源和多声源。
2. 只有与 Source 使用相同 checkpoint、episode 内容和顺序的结果才能横向比较。
3. 写入结果时同时登记对应 run manifest；缺少 commit、配置、checkpoint digest、
   数据版本、seed 或硬件信息的结果保留为 provisional。
4. SMT+Audio FSTTA multi-source 明确采用 single-source job 35 冻结配置的
   直接迁移运行，不根据多声源结果重新调参。
5. ENMuS FSTTA 例外地明确采用 single-source 搜索 job 0 的已有运行作为最终
   结果；必须保留“48 选 1”的选择来源说明，不把它写成独立复验。
6. EAM single-source 分别采用 SMT+Audio job 5 和 ENMuS job 11；后续
   multi-source 必须直接迁移对应模型的冻结配置，不在多声源上重新选参。
7. `Main Comparison` 仅存最终候选；完整超参数网格、机制实验和消融结果保留在
   各方法的独立报告与实验数据表中。
