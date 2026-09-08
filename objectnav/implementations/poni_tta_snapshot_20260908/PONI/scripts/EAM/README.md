# PONI + EAM

EAM 保留一个完全冻结的 Source RedNet，并复制一个 Auxiliary RedNet。每个语义
推理步骤先写入 reservoir，再从更新后的 reservoir 做 replay；Source 可靠像素的
联合决策被用作 Auxiliary 伪标签。当前帧使用更新前的 source/aux mixture，更新
只影响后续帧。

PONI 映射：冻结 RGB/Depth encoder，默认只训练 RedNet 后端
`deconv4, agant0, final_conv, final_deconv_custom`。决策单位为29类逐像素语义
分类，不使用 episode 成功反馈，属于无监督 TTA。

默认参数与 NavTTA EAM 一致：`lr=1e-5`、confidence scale `0.4`、memory `32`
步、batch `8` 步、update interval `1`、Adam。模块前缀是针对 RedNet 的任务映射。

```bash
PARTS="0" TEST_EPISODE_COUNT=1 bash scripts/EAM/run_eam_mp3d.sh 0
python scripts/EAM/monitor_eam.py
```

EAM 同时保留两套81.9M参数 RedNet，并在 replay 时额外执行最多8次前向，计算与
显存成本显著高于 Tent/FSTTA。正式实验前应先做单 episode 预检。当前实现只完成
代码接入，未自动启动任何实验。

`run_eam_optimized_mp3d.sh` 使用 `lr=1e-8`，每64个导航步采样一次EAM更新，
且每次只保留最多2048个有效深度内的低熵前景像素。
