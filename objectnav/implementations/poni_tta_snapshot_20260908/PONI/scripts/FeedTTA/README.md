# PONI + FeedTTA

FeedTTA 每帧把 RedNet 的逐像素 argmax 类别视为已执行语义决策，在线累计其
score gradient。episode 结束后读取 Habitat `success`：成功强化整条语义决策
轨迹，失败使用反向梯度，并应用 SGR（stochastic gradient reversion）。

这是使用二元 episode feedback 的 TTA，不应与无监督 Tent/FSTTA/EAM 放在同一
监督类别。默认冻结 RGB/Depth encoder，只训练
`deconv4, agant0, final_conv, final_deconv_custom`。

默认参数：`lr=5e-6`、`P=0.05`、`alpha=-0.2`、`gamma=0.99`、Adam、每 episode
一次 optimizer step。默认不做轨迹长度归一化，与 NavTTA Eq.(3) 实现一致。

```bash
PARTS="0" TEST_EPISODE_COUNT=1 bash scripts/FeedTTA/run_feedtta_mp3d.sh 0
python scripts/FeedTTA/monitor_feedtta.py
```

评测入口通过运行时 VectorEnv proxy 延迟传递 terminal success；PONI 原 evaluator
未被修改。当前实现只完成代码接入，未自动启动任何实验。

