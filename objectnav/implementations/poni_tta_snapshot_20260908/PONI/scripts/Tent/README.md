# PONI + Tent

本目录在不修改 PONI、NavTTA 原有源码的前提下，把 NavTTA 的
`TentAdapter` 接入 PONI 的 RedNet 语义分割模块。Tent 对每帧的逐像素类别
logits 最小化预测熵，仅更新 RedNet 的 BatchNorm 仿射参数；当前帧使用更新前
预测，更新作用于后续帧。独立的 GlobalAgent 包装器会把 episode 边界传给
NavTTA，默认采用 continual adaptation；因此后续也可正确测试 episodic reset
和每个 episode 的更新预算。

## 默认实验

```bash
cd /home/king/workspace/PONI
bash scripts/Tent/run_tent_mp3d.sh 0
```

参数 `0` 是 GPU 编号。默认评测 MP3D 的 11 个 `val_part`，每张 GPU 同时只运行
一个分片。每次启动都会在
`experiments/Tent/mp3d_poni_seed_123/runs/<时间戳_pid>/` 创建独立目录，不会
读取或覆盖上一次运行。可先做单 episode 冒烟测试：

```bash
PARTS="0" TEST_EPISODE_COUNT=1 \
SAVE_ROOT=/tmp/poni_tent_smoke \
bash scripts/Tent/run_tent_mp3d.sh 0
```

这里的 `SAVE_ROOT` 是实验根目录，实际文件写入
`/tmp/poni_tent_smoke/runs/<run_id>/`。默认自动生成唯一 `RUN_ID`；只有显式设置
相同 `RUN_ID` 才会续跑同一次实验，例如：

```bash
RUN_ID=full_seed100 SAVE_ROOT=/tmp/poni_tent \
bash scripts/Tent/run_tent_mp3d.sh 0
```

相对 `SAVE_ROOT` 统一按 `PONI_ROOT` 解析，因此启动器进入 `hlab/` 后不会改变
日志落盘位置；也可以直接使用绝对路径。

## 实时监控

另开一个终端，监控默认输出目录：

```bash
python scripts/Tent/monitor_tent.py
```

无参数时会在整个 `experiments/Tent/` 下自动发现最新 run，包括自定义的
`full_eval/runs/<run_id>`。

如果运行时设置了 `SAVE_ROOT`，监控端必须指向同一个目录：

```bash
python scripts/Tent/monitor_tent.py \
  --save-root /tmp/poni_tent_smoke
```

监控器接受实验根目录，会自动选择 `runs/` 下最新一次运行；监控保持开启时也会
自动切换到新启动的 run。界面每 5 秒刷新一次，显示总/分片 episode 进度、GPU 利用率与显存、评测
进程、运行时间、速度和预计剩余时间。只打印一次快照可使用 `--once`；调整刷新
周期可使用 `--interval 2`。按 `Ctrl-C` 只退出监控，不影响实验进程。

每个 run 都会写入自己的 `run_manifest.json`，并在正常完成、失败或中断时记录最终状态。
ETA 按本次运行新完成的 episode 速度估算，至少完成一个新 episode 后才会显示。

## 默认超参数

除归一化层范围外，下列值与 NavTTA `TentAdapter` / `build_adapter` 默认值一致。
NavTTA 的默认 `last_k_ln` 面向 Transformer；RedNet 没有 LayerNorm，因此这里
使用全部 BatchNorm（`bn`）。

| 参数 | 默认值 | 环境变量 |
| --- | ---: | --- |
| learning rate | `1e-6` | `TTA_LR` |
| optimizer | `Adam` | `TTA_OPTIMIZER` |
| steps | `1` | `TTA_STEPS` |
| episodic | `false` | `TTA_EPISODIC` |
| reset BN stats | `true` | `TTA_RESET_BN_STATS` |
| norm scope | `bn` | `TTA_NORM_SCOPE` |
| update interval | `1` | `TTA_UPDATE_INTERVAL` |
| max updates / episode | `-1`（无限制） | `TTA_MAX_UPDATES_PER_EPISODE` |
| gradient clipping | `1.0` | `TTA_MAX_GRAD_NORM` |
| weight decay | `0.0` | `TTA_WEIGHT_DECAY` |

后续调参可直接覆盖环境变量，例如：

```bash
PARTS="0 1" TTA_LR=1e-5 TTA_OPTIMIZER=AdamW \
SAVE_ROOT=/tmp/poni_tent_lr1e-5 \
bash scripts/Tent/run_tent_mp3d.sh 0
```

每个分片除了 PONI 原有的 `stats.json` 和日志，还会生成
`tent_diagnostics_*.json`，记录实际配置、被更新的参数及更新次数/损失/漂移等
NavTTA 诊断信息。实验依赖相邻的 `/home/king/workspace/NavTTA/core`；若仓库位置
不同，可设置 `PONI_ROOT` 和 `NAVTTA_ROOT`。

当前实现不会保存适配后的 RedNet checkpoint。每个分片进程结束时只保存指标和
Tent 诊断信息；continual 参数不会跨分片继承。

## 保守优化配置

默认 continual 实验在前三个分片出现过低熵、BN 漂移和导航指标下降，因此保留
默认脚本用于复现，另提供 `run_tent_optimized_mp3d.sh`：

```bash
PARTS="2" TEST_EPISODE_COUNT=-1 \
bash scripts/Tent/run_tent_optimized_mp3d.sh 0
```

优化配置仍采用 continual TTA，参数跨 episode 保留；使用 `lr=3e-7`、保留训练期
BN statistics、每 4 步更新、每 episode 最多 32 次更新。结果默认写到
`experiments/Tent/optimized/`。如需单独测试 episodic ablation，可显式设置
`TTA_EPISODIC=true`。
完成后与同一 Source 分片比较：

```bash
python scripts/compare_tta_to_source.py \
  --experiment-root experiments/Tent/optimized --part 2
```

## 结构性优化

`run_tent_structural_mp3d.sh` 只适配 `deconv3/deconv4/agant1/agant0/final_conv`
中的后端 BN，并只对深度1～5m且预测为非背景的像素计算熵；少于128个前景像素
时跳过该帧更新。默认采用 episodic K4/U32。
