#!/usr/bin/env python3
"""Live terminal monitor for PONI + Tent experiments."""

import argparse
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


TQDM_PATTERN = re.compile(r"\b(\d+)\s*/\s*(\d+)\s*\[")
SPLIT_PATTERN = re.compile(r"\bEVAL\.SPLIT\s+val_part_(\d+)\b")


def format_duration(seconds):
    if seconds is None or seconds < 0:
        return "计算中"
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    value = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    return "{}天 {}".format(days, value) if days else value


def format_bytes_kib(value):
    mib = float(value) / 1024.0
    if mib >= 1024:
        return "{:.1f} GiB".format(mib / 1024.0)
    return "{:.0f} MiB".format(mib)


def safe_json(path, default=None):
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def count_available_episodes(poni_root, part):
    content_dir = (
        poni_root
        / "data/datasets/objectnav/mp3d/v1/val_parts"
        / "val_part_{}".format(part)
        / "content"
    )
    total = 0
    for path in content_dir.glob("*.json.gz"):
        try:
            with gzip.open(path, "rt") as handle:
                total += len(json.load(handle).get("episodes", []))
        except (OSError, ValueError):
            continue
    return total


def completed_from_stats(save_root, part):
    stats_path = (
        save_root / "tb_seed_100_val_part_{}".format(part) / "stats.json"
    )
    payload = safe_json(stats_path, {})
    return len(payload) if isinstance(payload, dict) else 0


def progress_from_console(path, start_time):
    try:
        stat = path.stat()
        if stat.st_mtime + 1 < start_time:
            return 0, None, stat.st_mtime
        with path.open("rb") as handle:
            if stat.st_size > 4 * 1024 * 1024:
                handle.seek(stat.st_size - 4 * 1024 * 1024)
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return 0, None, None
    matches = TQDM_PATTERN.findall(text.replace("\r", "\n"))
    if not matches:
        return 0, None, stat.st_mtime
    current, total = matches[-1]
    return int(current), int(total), stat.st_mtime


def evaluator_processes(save_root, eval_script, run_id=None):
    try:
        result = subprocess.run(
            [
                "ps",
                "-eo",
                "pid=,etimes=,pcpu=,pmem=,rss=,stat=,args=",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []

    processes = []
    root_text = str(save_root)
    run_marker = "runs/{}".format(run_id) if run_id else None
    for line in result.stdout.splitlines():
        matches_run = root_text in line or (
            run_marker is not None and run_marker in line
        )
        if eval_script not in line or not matches_run:
            continue
        fields = line.strip().split(None, 6)
        if len(fields) != 7:
            continue
        match = SPLIT_PATTERN.search(fields[6])
        processes.append(
            {
                "pid": int(fields[0]),
                "elapsed": int(fields[1]),
                "cpu": float(fields[2]),
                "memory_percent": float(fields[3]),
                "rss_kib": int(fields[4]),
                "state": fields[5],
                "part": int(match.group(1)) if match else None,
            }
        )
    return processes


def gpu_process_memory():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}
    output = {}
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            output[int(fields[0])] = fields[1] + " MiB"
        except ValueError:
            continue
    return output


def gpu_status(gpu_id):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except FileNotFoundError:
        return None, "未安装 nvidia-smi"
    except subprocess.TimeoutExpired:
        return None, "nvidia-smi 查询超时"
    if result.returncode != 0:
        message = result.stderr.strip().splitlines()
        return None, message[-1] if message else "nvidia-smi 查询失败"
    rows = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) == 7:
            rows.append(fields)
    selected = next((row for row in rows if row[0] == str(gpu_id)), None)
    if selected is None:
        return None, "没有找到 GPU {}".format(gpu_id)
    return {
        "index": selected[0],
        "name": selected[1],
        "util": selected[2],
        "memory_used": selected[3],
        "memory_total": selected[4],
        "temperature": selected[5],
        "power": selected[6],
    }, None


def progress_bar(fraction, width=32):
    fraction = min(1.0, max(0.0, fraction))
    filled = int(round(width * fraction))
    return "[{}{}]".format("#" * filled, "-" * (width - filled))


class Monitor:
    def __init__(self, args):
        self.args = args
        self.requested_root = Path(args.save_root).resolve()
        self.save_root = self.requested_root
        self.manifest_path = self.save_root / "run_manifest.json"
        self.expected_cache = {}

    def refresh_run_root(self):
        """Select the newest isolated run when given an experiment root."""

        runs_dir = self.requested_root / "runs"
        candidates = list(runs_dir.glob("*/run_manifest.json"))
        candidates.extend(
            self.requested_root.glob("*/runs/*/run_manifest.json")
        )
        if candidates:
            def sort_key(path):
                payload = safe_json(path, {})
                try:
                    started = float(payload.get("start_time_epoch", 0.0))
                except (TypeError, ValueError):
                    started = 0.0
                try:
                    modified = path.stat().st_mtime
                except OSError:
                    modified = 0.0
                return started, modified

            selected = max(candidates, key=sort_key).parent
        else:
            selected = self.requested_root
        if selected != self.save_root:
            self.save_root = selected
            self.manifest_path = self.save_root / "run_manifest.json"
            self.expected_cache.clear()

    def manifest(self):
        manifest = safe_json(self.manifest_path)
        if manifest is not None:
            return manifest
        parts = [int(value) for value in self.args.parts.split()]
        return {
            "status": "unknown",
            "started_at": None,
            "start_time_epoch": 0.0,
            "launcher_pid": None,
            "poni_root": str(Path(self.args.poni_root).resolve()),
            "save_root": str(self.save_root),
            "gpu_id": str(self.args.gpu_id),
            "parts": parts,
            "test_episode_count": self.args.test_episode_count,
            "baseline_completed_episodes": {},
        }

    def runtime_output_root(self, manifest):
        """Resolve logs from runs started with the legacy relative-path bug."""

        explicit = manifest.get("runtime_output_root")
        if explicit:
            return Path(explicit).resolve()
        poni_root = Path(manifest.get("poni_root", self.args.poni_root)).resolve()
        try:
            relative = self.save_root.relative_to(poni_root)
        except ValueError:
            return self.save_root
        legacy_root = poni_root / "hlab" / relative
        if legacy_root.is_dir() and (
            any(legacy_root.glob("console_seed_*.log"))
            or any(legacy_root.glob("tb_seed_*/stats.json"))
        ):
            return legacy_root
        return self.save_root

    @staticmethod
    def merge_runtime_status(manifest, runtime_root, manifest_root):
        if runtime_root == manifest_root:
            return manifest
        runtime_manifest = safe_json(runtime_root / "run_manifest.json")
        if not isinstance(runtime_manifest, dict):
            return manifest
        output = dict(manifest)
        for key in (
            "status",
            "ended_at",
            "end_time_epoch",
            "elapsed_seconds",
        ):
            if key in runtime_manifest:
                output[key] = runtime_manifest[key]
        return output

    def expected(self, manifest, part):
        if part not in self.expected_cache:
            available = count_available_episodes(
                Path(manifest["poni_root"]), part
            )
            requested = int(manifest.get("test_episode_count", -1))
            self.expected_cache[part] = (
                available if requested < 0 else min(requested, available)
            )
        return self.expected_cache[part]

    def snapshot(self):
        self.refresh_run_root()
        manifest = self.manifest()
        data_root = self.runtime_output_root(manifest)
        manifest = self.merge_runtime_status(manifest, data_root, self.save_root)
        start_time = float(manifest.get("start_time_epoch") or 0.0)
        processes = evaluator_processes(
            data_root,
            self.args.eval_script,
            run_id=manifest.get("run_id"),
        )
        running_parts = {
            process["part"] for process in processes if process["part"] is not None
        }
        parts = []
        for part in manifest.get("parts", []):
            part = int(part)
            expected = self.expected(manifest, part)
            stats_count = completed_from_stats(data_root, part)
            console_count, console_total, console_mtime = progress_from_console(
                data_root
                / "console_seed_100_val_part_{}.log".format(part),
                start_time,
            )
            current = min(expected, max(stats_count, console_count))
            if expected > 0 and current >= expected:
                status = "完成"
            elif part in running_parts:
                status = "运行中"
            elif (
                manifest.get("status") == "running"
                and console_mtime is not None
                and time.time() - console_mtime < 120.0
            ):
                status = "运行中(日志)"
            elif console_total is not None:
                status = "已停止"
            else:
                status = "等待"
            parts.append(
                {
                    "part": part,
                    "current": current,
                    "expected": expected,
                    "status": status,
                }
            )

        total = sum(part["expected"] for part in parts)
        current = sum(part["current"] for part in parts)
        baseline_map = manifest.get("baseline_completed_episodes", {})
        baseline = sum(
            min(part["expected"], int(baseline_map.get(str(part["part"]), 0)))
            for part in parts
        )
        now = time.time()
        end_time = manifest.get("end_time_epoch")
        elapsed = max(0.0, float(end_time or now) - start_time) if start_time else 0.0
        new_progress = max(0, current - baseline)
        rate = new_progress / elapsed if elapsed > 0 and new_progress > 0 else 0.0
        remaining = max(0, total - current)
        eta = remaining / rate if rate > 0 else None
        if current >= total and total > 0:
            eta = 0.0

        gpu, gpu_error = gpu_status(manifest.get("gpu_id", self.args.gpu_id))
        gpu_memory = gpu_process_memory() if gpu is not None else {}
        for process in processes:
            process["gpu_memory"] = gpu_memory.get(process["pid"], "-")
        return {
            "manifest": manifest,
            "data_root": data_root,
            "parts": parts,
            "processes": processes,
            "total": total,
            "current": current,
            "elapsed": elapsed,
            "eta": eta,
            "rate": rate,
            "gpu": gpu,
            "gpu_error": gpu_error,
        }

    def render(self, snapshot):
        manifest = snapshot["manifest"]
        total = snapshot["total"]
        current = snapshot["current"]
        fraction = current / total if total else 0.0
        processes = snapshot["processes"]
        completed_parts = sum(
            part["status"] == "完成" for part in snapshot["parts"]
        )
        running_parts = sum(
            part["status"].startswith("运行中") for part in snapshot["parts"]
        )

        lines = [
            "PONI + {} 运行监控  {}".format(
                self.args.method_label,
                datetime.now().astimezone().isoformat(timespec="seconds")
            ),
            "实验根目录: {}".format(self.requested_root),
            "本次运行: {}".format(
                manifest.get("run_id") or self.save_root.name
            ),
            "Manifest 目录: {}".format(self.save_root),
            "输出目录: {}".format(snapshot["data_root"]),
            "状态: {}  启动时间: {}  Launcher PID: {}".format(
                manifest.get("status", "unknown"),
                manifest.get("started_at") or "未知",
                manifest.get("launcher_pid") or "未知",
            ),
            "",
            "总进度 {} {:6.2f}%  {}/{} episodes".format(
                progress_bar(fraction), fraction * 100.0, current, total
            ),
            "分片: 完成 {}/{}，运行中 {}  |  已运行 {}  |  预计剩余 {}".format(
                completed_parts,
                len(snapshot["parts"]),
                running_parts,
                format_duration(snapshot["elapsed"]),
                format_duration(snapshot["eta"]),
            ),
            "速度: {:.3f} episodes/s ({:.2f} episodes/min)".format(
                snapshot["rate"], snapshot["rate"] * 60.0
            ),
            "",
        ]

        gpu = snapshot["gpu"]
        if gpu is None:
            lines.append("GPU: 不可用 ({})".format(snapshot["gpu_error"]))
        else:
            lines.append(
                "GPU {index} {name}: 利用率 {util}%  显存 {memory_used}/{memory_total} MiB  "
                "温度 {temperature}C  功耗 {power}W".format(**gpu)
            )

        lines.extend(["", "评测进程:"])
        if not processes:
            lines.append(
                "  当前没有匹配的 {} 进程".format(self.args.eval_script)
            )
        else:
            lines.append("  PID      分片     运行时间    CPU%   RSS       GPU显存   状态")
            for process in processes:
                lines.append(
                    "  {pid:<8} {part:<8} {elapsed:<11} {cpu:>5.1f}   "
                    "{rss:<9} {gpu:<9} {state}".format(
                        pid=process["pid"],
                        part=(
                            "part_{}".format(process["part"])
                            if process["part"] is not None
                            else "?"
                        ),
                        elapsed=format_duration(process["elapsed"]),
                        cpu=process["cpu"],
                        rss=format_bytes_kib(process["rss_kib"]),
                        gpu=process["gpu_memory"],
                        state=process["state"],
                    )
                )

        lines.extend(["", "分片进度:", "  分片       状态      episodes     进度"])
        for part in snapshot["parts"]:
            fraction = (
                part["current"] / part["expected"] if part["expected"] else 0.0
            )
            lines.append(
                "  val_part_{:<3} {:<9} {:>5}/{:<5} {:6.2f}%".format(
                    part["part"],
                    part["status"],
                    part["current"],
                    part["expected"],
                    fraction * 100.0,
                )
            )
        return "\n".join(lines)


def build_parser(
    default_method="Tent",
    default_eval_script="eval_poni_tent.py",
    default_save_root=None,
):
    script_dir = Path(__file__).resolve().parent
    poni_root = script_dir.parents[1]
    if default_save_root is None:
        default_save_root = poni_root / "experiments/Tent"
    parser = argparse.ArgumentParser(
        description="Monitor PONI + {} progress, GPU, runtime, and ETA.".format(
            default_method
        )
    )
    parser.add_argument(
        "--save-root",
        default=str(default_save_root),
    )
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-clear", action="store_true")
    parser.add_argument("--poni-root", default=str(poni_root))
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--parts", default="0 1 2 3 4 5 6 7 8 9 10")
    parser.add_argument("--test-episode-count", type=int, default=-1)
    parser.set_defaults(
        method_label=default_method,
        eval_script=default_eval_script,
    )
    return parser


def main(
    default_method="Tent",
    default_eval_script="eval_poni_tent.py",
    default_save_root=None,
):
    args = build_parser(
        default_method=default_method,
        default_eval_script=default_eval_script,
        default_save_root=default_save_root,
    ).parse_args()
    monitor = Monitor(args)
    try:
        while True:
            output = monitor.render(monitor.snapshot())
            if not args.no_clear and not args.once and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(output, flush=True)
            if args.once:
                break
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("\n监控已停止；实验进程不受影响。")


if __name__ == "__main__":
    main()
