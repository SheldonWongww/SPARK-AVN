# AVN 主对比表结果登记册

更新日期：2026-07-29

用途：集中登记未来论文 AVN 主对比表采用的冻结结果。表格以导航模型为一级
分组，列按单声源和多声源展开。当前填入统一 Source 重评估结果、冻结配置的
Tent 候选结果，以及用于冻结 FSTTA 配置的 SMT+Audio 单声源开发结果；`—`
表示该方法尚未完成最终配置复验或尚不具备可写入主表的结果。

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
      <td colspan="9">—</td>
    </tr>
    <tr><td>EAM</td><td colspan="9">—</td><td colspan="9">—</td></tr>
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
    <tr><td>FSTTA</td><td colspan="9">—</td><td colspan="9">—</td></tr>
    <tr><td>EAM</td><td colspan="9">—</td><td colspan="9">—</td></tr>
    <tr><td>FeedTTA†</td><td colspan="9">—</td><td colspan="9">—</td></tr>
    <tr><td>ATENA†</td><td colspan="9">—</td><td colspan="9">—</td></tr>
  </tbody>
</table>

`[P]`：当前为 provisional 候选值；尚不能作为正式论文结果。

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

## 5. 当前 FSTTA 候选与最终复验计划

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

该 single-source 数值用于记录配置冻结依据，仍属于开发批次结果，故暂标 `[P]`。
冻结后不再针对其他模型或声源重新选参；使用相同配置运行 SMT+Audio
multi-source、ENMuS single-source 和 ENMuS multi-source。三项完成并通过 run
manifest 校验后再补入本表；SMT+Audio single-source 也应在最终固定 commit 上
独立复验后才能移除 `[P]`。

## 6. 后续填表规则

1. 每种 TTA 方法先冻结配置，再分别运行 SMT+Audio/ENMuS 的单声源和多声源。
2. 只有与 Source 使用相同 checkpoint、episode 内容和顺序的结果才能横向比较。
3. 写入结果时同时登记对应 run manifest；缺少 commit、配置、checkpoint digest、
   数据版本、seed 或硬件信息的结果保留为 provisional。
4. 方法开发网格的最大值不能直接写入本表；必须使用冻结配置完成一次独立复验。
5. `Main Comparison` 仅存最终候选；完整超参数网格、机制实验和消融结果保留在
   各方法的独立报告与实验数据表中。
