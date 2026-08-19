# 具身导航 TTA / VLA 文献包（CVPR 2027 规划）

更新日期：2026-08-11。

本目录收集用于 NavTTA 方法选择与 CVPR 2027 实验规划的论文。当前包含 39 个可解析 PDF，约 181 MB，覆盖：

- 项目已实现的 Tent、FSTTA、EAM、FeedTTA、ATENA；
- 拟进入 10–12 方法主对比池的 EATA、SAR、CoTTA、RoTTA、BiTTA、PDF 等；
- reconstruction、contrastive、feature alignment、memory、prompt、VLA feedback 等候选路线；
- MG-Select、RoboMonkey、GPC 等应与参数 TTA 区分的 test-time scaling/planning 对照；
- 三份 CVPR 2026 supplementary material。

文件说明：

- `sources.tsv`：标题、年份、venue、在线来源、分组和协议备注。
- `SHA256SUMS`：全部本地 PDF 的 SHA256。
- `TTA_METHOD_SELECTION_AND_CVPR2027_EXPERIMENT_PLAN.md`：方法适用性、三类 taxonomy、11/12 方法组合、主实验和附录蓝图。

校验方式：

```bash
shasum -a 256 -c SHA256SUMS
```

## 来源例外

`03_EAM.pdf` 是有效的 13 页 IEEE TMM 正式 PDF，嵌入元数据包含论文标题、作者、Article ID `10856447` 和 DOI `10.1109/TMM.2025.3535356`。该文件页脚标明仅限 East China Normal University 授权使用，并非公开 OA 版本；本次工作也无法恢复该本地文件的准确 acquisition URL。IEEE 公共 PDF 端点在当前执行环境返回 HTTP 418。因此 `sources.tsv` 将其标为 `local_copy`，不伪造开放下载来源，也不得向公开仓库再分发。

其余文件均由 `sources.tsv` 中记录的 arXiv、PMLR、CVF、NeurIPS 或 ICLR 公开地址下载。论文版权仍归原作者和出版方所有，本目录仅用于本地科研阅读。目录内 `.gitignore` 默认忽略全部 PDF，以免把本地阅读副本和受限 IEEE 文件意外提交到 Git；分析文档、来源表和哈希仍可正常跟踪。
