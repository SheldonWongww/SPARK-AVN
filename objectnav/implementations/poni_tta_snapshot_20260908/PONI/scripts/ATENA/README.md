# PONI + ATENA

ATENA 保存每个 episode 的 RGB-D 输入、逐像素语义 argmax 和 RedNet 最深融合特征
的 global-average pooling。episode 结束后用平均策略熵决定是否查询真实 Habitat
success；非查询 episode 使用自预测成功标签。随后逐步 replay 整条轨迹并执行一次
更新。

这是 active binary-feedback TTA。默认 `PARAM_SCOPE=all`，与 NavTTA official
mapping 一致，会训练完整 RedNet（81.9M参数）以及一个2048维成功自预测 MLP。
可显式设置 `ATENA_PARAM_SCOPE=bn` 做参数高效消融。

默认参数：query/self LR `1e-6/1e-7`、mixture lambda `0.5`、query threshold
`0.1`、self-loss weight `0.1`、AdamW、weight decay `0.01`。

```bash
PARTS="0" TEST_EPISODE_COUNT=1 bash scripts/ATENA/run_atena_mp3d.sh 0
python scripts/ATENA/monitor_atena.py
```

ATENA 在 episode 结束时重放所有 RedNet 步，最长500步，且 CPU trajectory snapshot
可能占用大量内存；正式实验前必须先做单 episode 预检。当前实现只完成代码接入，
未自动启动任何实验。

`run_atena_optimized_mp3d.sh` 将 query threshold 降至0.01，只适配BN参数，并将
query/self LR降至 `1e-8/1e-9`，梯度裁剪为1.0。
