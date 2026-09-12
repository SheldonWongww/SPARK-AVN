# StreamVLN v6：seen 重搜前的归档审计

本目录仅保存历史设计证据。机器可读结果见 [summary.json](summary.json)，包含
原始结果、配置、诊断及 Source launcher 的 SHA256。本地归档缺少实际 run
manifest 和 Source result.json，不能升级为正式结果。

## 完整性

`中间结果/streamvln_val_unseen_search_v6/jobs/` 下 12 份 unseen 结果均为完整
1,839 条加终端 aggregate；逐条 scene/episode 顺序、唯一性和 SR/SPL 均值经核验。
seen 的 FSTTA、EAM、Tent 均有 778 条和 aggregate。FeedTTA 的 622 条、
ATENA 的 625 条均为 canonical 前缀，但没有终端 aggregate，日志停在运行中。
现有证据不能区分导出时仍在运行和之后外部中断，不宣称它们算法崩溃。

| seen 记录 | episode 数 | SR | SPL | 证据资格 |
|---|---:|---:|---:|---|
| Source | 778（日志声明） | 68.5090 | 62.5255 | 仅终端日志 aggregate |
| Tent | 778 | 68.5090 | 62.6574 | 完整流，缺原始 manifest |
| FSTTA | 778 | 67.7378 | 61.8658 | 完整流，缺原始 manifest |
| EAM | 778 | 67.3522 | 61.6537 | 完整流，缺原始 manifest |
| FeedTTA† | 622 | — | — | 不完整，不排名 |
| ATENA† | 625 | — | — | 不完整，不排名 |

FSTTA seen 完成 4,933 次 FAST、48 次 SLOW 更新，漂移约 `7.73e-6`。
EAM seen 完成 15,668 次更新，auxiliary 漂移约 `.00457`，两分支门控率均约
99.6%。FeedTTA/ATENA 前缀分别有 622/625 次更新，ATENA 查询率 34.08%。
这些诊断用于设计范围，不把不同长度流的 SR/SPL 与完整 Source 比较。

unseen 中 EAM c=.4 的两组即使发生参数变化，也没有执行动作变化，最终结果完全
相同；c=.8 的 LR=3e-6 优于 1e-6。FeedTTA gamma=.7 在两个 LR 下均优于 .95；
ATENA 最佳为 query LR=3e-7、threshold=.3。
FP32 readout 的翻转数与原生 BF16 行动翻转数含义不同，不能将其中任何一个
单独解释成“适应没有作用”。

## 跨 benchmark 参考边界

`summary.json` 固定六份已有 VLN/AVN 分析文件的摘要。R2R 的 M=1、较稀疏
SLOW、FeedTTA argmax/正 alpha，以及 R2R-CE 的 EAM 小 LR/memory 和 ATENA
query/self LR 分离，是搜索机制的依据；不将其他模型的最佳 LR 当作 StreamVLN
必须使用的默认值。

另外保存 SMT+Audio ATENA 三份 compact result 的摘要和关键字段：
single-source 的 s16（query/self=`1e-7/1e-8`、lambda=.5、threshold=1、
self weight=.25）SR/SPL=`56.65/30.5499`，query rate=.002；c00 为
`56.00/30.3520`、query rate=.9695。两种路由都可能有效。
本地 multi-source 的 20 个候选全部低于 matched Source SR=25.90，最高 SPL
c03 的 SR/SPL 也只有 `19.50/10.7773`。这不支持直接复制高阈值到 StreamVLN。
上述 AVN 记录也缺原始 manifest，且采用 sampled action，全部为历史设计证据。

四方法的新三点方案和服务器命令见
[STREAMVLN_VAL_SEEN_SEARCH_V1.md](../../../experiments/STREAMVLN_VAL_SEEN_SEARCH_V1.md)。
这轮为 seen 调参；不能把在已参与开发的 split 上选出的最好结果当作独立确认。
