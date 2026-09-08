#!/usr/bin/env python3
"""Monitor one-process global-continual PONI TTA runs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import time


PROGRESS = re.compile(r"\b(\d+)\s*/\s*(\d+)\s*\[")


def load(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def latest(root):
    candidates = list((root / "runs").glob("*/run_manifest.json"))
    if not candidates:
        raise SystemExit("No global-continual run found under {}".format(root))
    return max(
        candidates,
        key=lambda p: float((load(p, {}) or {}).get("start_time_epoch", 0)),
    ).parent


def duration(seconds):
    if seconds is None: return "计算中"
    seconds = int(max(0, seconds)); days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600); minutes, seconds = divmod(seconds, 60)
    value = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    return "{}天 {}".format(days, value) if days else value


def console_progress(path, started):
    try:
        stat = path.stat(); data = path.read_bytes()[-4 * 1024 * 1024:]
    except OSError:
        return 0, None, None
    matches = PROGRESS.findall(data.decode(errors="replace").replace("\r", "\n"))
    if not matches: return 0, None, stat.st_mtime
    current, total = matches[-1]
    return int(current), int(total), stat.st_mtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = Path(args.save_root).resolve()
    try:
        while True:
            run = latest(root); manifest = load(run / "run_manifest.json", {})
            total = int(manifest["stream"]["total_episodes"])
            stats = load(run / "tb/stats.json", {})
            if stats and len(stats) >= total:
                current = len(stats)
            else:
                current, _, mtime = console_progress(
                    run / "console.log", manifest["start_time_epoch"]
                )
            now = time.time(); elapsed = float(
                manifest.get("elapsed_seconds", now - manifest["start_time_epoch"])
            )
            rate = current / elapsed if current and elapsed else 0
            eta = (total - current) / rate if rate else None
            if current >= total: eta = 0
            print("PONI global continual monitor  {}".format(
                datetime.now().astimezone().isoformat(timespec="seconds")
            ))
            print("Method: {}  Run: {}".format(manifest["method"], manifest["run_id"]))
            print("Status: {}  Protocol: {}".format(
                manifest["status"], manifest["protocol"]
            ))
            print("Progress: {}/{} ({:.2f}%)".format(
                current, total, 100.0 * current / total
            ))
            print("Elapsed: {}  ETA: {}  Speed: {:.2f} episodes/min".format(
                duration(elapsed), duration(eta), rate * 60
            ))
            print("Scene order: " + " -> ".join(
                item["part"] for item in manifest["stream"]["scene_order"]
            ))
            if args.once: break
            time.sleep(max(1, args.interval))
            print("\033[2J\033[H", end="")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

