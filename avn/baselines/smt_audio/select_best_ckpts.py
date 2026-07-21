#!/usr/bin/env python3
"""
为 av_nav / savi / smt_audio 三个方法的单/多声源训练，按照 run.py 里 --eval-best
使用的 find_best_ckpt_idx 逻辑（读 model_dir/tb，在 val/*spl 里取最大值对应的 ckpt
编号）选出最佳 checkpoint，复制并重命名为 {single,multi}_best_val.pth，存到
    data/pretrained_weights/semantic_audionav/<method>/
下，作为后续 TTA 实验的源模型权重。

在服务器上、sound-spaces 工作目录根目录运行：
    python select_best_ckpts.py            # 实际复制
    python select_best_ckpts.py --dry-run  # 只打印选择结果，不复制

前提：这些方法都已跑过「逐 ckpt 的 val 验证」（eval.yaml, SPLIT=val），
tb 里有 val/spl 记录。否则会提示找不到。
"""
import os
import shutil
import argparse

# 纯读 tb，不需要 GPU；放在 import tensorflow 之前
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tensorflow as tf


# ---- 训练输出目录（--model-dir）。如与你实际命名不同，改这里即可 ----
# (method, scene, model_dir, 输出子目录名)
JOBS = [
    ("av_nav",    "single", "data/models/av_nav_tta_single_source",   "av_nav"),
    ("av_nav",    "multi",  "data/models/av_nav_tta_multi_source",    "av_nav"),
    ("savi",      "single", "data/models/savi_tta_single_source",     "savi"),
    ("savi",      "multi",  "data/models/savi_tta_multi_source",      "savi"),
    ("smt_audio", "single", "data/models/smt_audio_tta_single_source", "smt_audio"),
    ("smt_audio", "multi",  "data/models/smt_audio_tta_multi_source",  "smt_audio"),
]

PRETRAIN_ROOT = "data/pretrained_weights/semantic_audionav"


def find_best_ckpt_idx(event_dir_path, min_step=-1, max_step=10000):
    """复刻 ss_baselines/.../run.py 的 find_best_ckpt_idx：
    在 tb 事件文件里找 tag 以 'val' 开头且含 'spl' 的标量，取最大值对应的 step
    （= checkpoint 编号）。"""
    if not os.path.isdir(event_dir_path):
        print(f"  [warn] tb 目录不存在: {event_dir_path}")
        return -1, -1.0

    max_value = 0.0
    max_index = -1
    for event in sorted(os.listdir(event_dir_path)):
        if "events" not in event:
            continue
        path = os.path.join(event_dir_path, event)
        try:
            for e in tf.compat.v1.train.summary_iterator(path):
                if len(e.summary.value) == 0:
                    continue
                tag = e.summary.value[0].tag
                # 训练事件文件（Environment/* 等）第一个非 val 标量即跳过该文件
                if not tag.startswith("val"):
                    break
                if "spl" not in tag:
                    continue
                if not (min_step <= e.step <= max_step):
                    continue
                if e.summary.value[0].simple_value > max_value:
                    max_value = e.summary.value[0].simple_value
                    max_index = e.step
        except Exception as ex:
            print(f"  [warn] 读取 {path} 出错: {ex}")
    return max_index, max_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="只打印选择结果，不复制")
    args = parser.parse_args()

    summary = []
    for method, scene, model_dir, out_subdir in JOBS:
        tb_dir = os.path.join(model_dir, "tb")
        data_dir = os.path.join(model_dir, "data")
        print(f"\n=== {method} / {scene} ===")
        print(f"  model_dir: {model_dir}")

        idx, val = find_best_ckpt_idx(tb_dir)
        if idx < 0:
            print(f"  [skip] 未在 {tb_dir} 找到 val/*spl（是否已跑过逐 ckpt 的 val 验证？）")
            summary.append((method, scene, "SKIP", None, None))
            continue

        src = os.path.join(data_dir, f"ckpt.{idx}.pth")
        if not os.path.isfile(src):
            print(f"  [skip] 最佳 idx={idx} 但 ckpt 文件不存在: {src}")
            summary.append((method, scene, "MISSING", idx, None))
            continue

        out_dir = os.path.join(PRETRAIN_ROOT, out_subdir)
        dst = os.path.join(out_dir, f"{scene}_best_val.pth")
        print(f"  best: ckpt.{idx}.pth   (val spl/softspl 最大值 = {val:.4f})")
        print(f"  copy: {src}  ->  {dst}")
        if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)
            shutil.copyfile(src, dst)
            print("  done.")
        summary.append((method, scene, f"ckpt.{idx}.pth", idx, val))

    print("\n================ 汇总 ================")
    for method, scene, status, idx, val in summary:
        v = f"{val:.4f}" if isinstance(val, float) else "-"
        print(f"  {method:10s} {scene:6s} -> {status:14s} (val={v})")
    print("dry-run，未实际复制。" if args.dry_run else "全部复制完成。")


if __name__ == "__main__":
    main()
