# PONI + FSTTA

本目录在不修改 PONI/NavTTA 原源码的前提下，将 NavTTA `FSTTAAdapter` 接入
PONI RedNet。逐帧语义 logits 用于熵损失，只有 BatchNorm 仿射参数可训练。

默认协议：

- FAST：每 `M=3` 帧聚合一次 concordant gradient，并用自适应倍率调整 fast LR。
- SLOW：每 `N=4` 个完整 episode 更新一次持久 slow anchor，然后将 fast 参数
  对齐到该 anchor。
- fast 优化器为 AdamW，每个 episode 清空 fast moments；slow AdamW moments
  跨 episode/window 保留。
- 当前帧使用更新前 logits，参数更新影响后续帧。

## 运行

```bash
cd /home/king/workspace/PONI
bash scripts/FSTTA/run_fstta_mp3d.sh 0
```

每次启动都会写入独立目录：
`experiments/FSTTA/mp3d_poni_seed_123/runs/<时间戳_pid>/`。`SAVE_ROOT` 表示
实验根目录而非单次运行目录；默认生成唯一 `RUN_ID`，显式复用相同 `RUN_ID`
时才会续跑。
相对 `SAVE_ROOT` 统一按 `PONI_ROOT` 解析，不受启动器后续 `cd hlab` 影响。

单 episode 冒烟测试不会触发默认的 slow update；建议至少运行 4 个 episode：

```bash
PARTS="0" TEST_EPISODE_COUNT=4 \
SAVE_ROOT=/tmp/poni_fstta_smoke \
bash scripts/FSTTA/run_fstta_mp3d.sh 0
```

监控：

```bash
python scripts/FSTTA/monitor_fstta.py
```

无参数时会搜索整个 `experiments/FSTTA/` 并选择最新 run。也可以限定实验根目录：

```bash
python scripts/FSTTA/monitor_fstta.py \
  --save-root /tmp/poni_fstta_smoke
```

监控器会自动选择实验根目录中最新的 run，并在新 run 启动后自动切换。

## 默认超参数

默认值来自 NavTTA `build_adapter` 的 FSTTA 分支；RedNet 没有 LayerNorm，因此
`NORM_SCOPE` 映射为 `bn`。

| 参数 | 默认值 | 环境变量 |
| --- | ---: | --- |
| fast LR | `1e-6` | `FSTTA_LR_FAST` |
| slow LR | `1e-4` | `FSTTA_LR_SLOW` |
| fast window | `3` | `FSTTA_M` |
| slow window | `4` episodes | `FSTTA_N` |
| q / rho / tau | `0.1 / 0.95 / 0.7` | `FSTTA_Q/RHO/TAU` |
| LR scale bounds | `0.9 / 1.1` | `FSTTA_A/B` |
| optimizer | `AdamW` | `FSTTA_OPTIMIZER` |
| beta1 / beta2 | `0.9 / 0.99` | `FSTTA_BETA1/BETA2` |
| use slow | `true` | `FSTTA_USE_SLOW` |
| fast gradient | `concordant` | `FSTTA_FAST_GRAD_MODE` |
| norm scope | `bn` | `FSTTA_NORM_SCOPE` |

每个分片结束后会生成 `fstta_diagnostics_*.json`，包含 fast/slow 更新次数、
LR scale、梯度方差、slow anchor 漂移等信息。当前不会保存适配 checkpoint，且
FSTTA 状态不会跨独立 `val_part` 进程继承。

## 与 Tent 并行

两个方法是独立实验，可以并行且不会共享参数；默认输出目录也不同。推荐在两张
GPU 上分别运行。单 GPU 同时运行会让两套 RedNet 前向/反向、Habitat 和规划器
竞争 GPU/CPU，可能显存不足，通常不会获得接近 2 倍的吞吐提升。

## 保守优化配置

```bash
PARTS="2" TEST_EPISODE_COUNT=-1 \
bash scripts/FSTTA/run_fstta_optimized_mp3d.sh 0
```

优化配置保留训练期 BN statistics，将 fast/slow LR 调整为 `3e-7/1e-5`，并将
fast/slow window 调整为 `M=8/N=8`。结果默认写到
`experiments/FSTTA/optimized/`。

```bash
python scripts/compare_tta_to_source.py \
  --experiment-root experiments/FSTTA/optimized --part 2
```

## 结构性优化

`run_fstta_structural_mp3d.sh` 只适配最后两个 decoder stage 及 skip adapter 的
BN，并用有效深度前景像素熵。默认使用 fast/slow `3e-7/1e-5`、`M=8/N=8`。
