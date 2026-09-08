#!/usr/bin/env python3
"""Manifest writer for a single global-continual stream."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import time


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def parse_tta(values):
    output = {}
    for value in values:
        key, raw = value.split("=", 1)
        output[key] = raw
    return output


def start(args):
    stream = json.loads(Path(args.stream_manifest).read_text())
    payload = {
        "schema_version": 1,
        "protocol": "global_continual",
        "method": args.method,
        "status": "running",
        "run_id": args.run_id,
        "launcher_pid": int(args.launcher_pid),
        "started_at": now(),
        "start_time_epoch": time.time(),
        "poni_root": str(Path(args.poni_root).resolve()),
        "navtta_root": str(Path(args.navtta_root).resolve()),
        "save_root": str(Path(args.save_root).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "gpu_id": args.gpu_id,
        "stream": stream,
        "tta": parse_tta(args.tta),
        "resume_supported": False,
    }
    write(Path(args.path), payload)


def finish(args):
    path = Path(args.path)
    payload = json.loads(path.read_text())
    ended = time.time()
    payload.update({
        "status": args.status,
        "ended_at": now(),
        "end_time_epoch": ended,
        "elapsed_seconds": ended - payload["start_time_epoch"],
    })
    write(path, payload)


def parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("start")
    for name in (
        "path", "method", "run_id", "launcher_pid", "poni_root",
        "navtta_root", "save_root", "model_path", "gpu_id",
        "stream_manifest",
    ):
        s.add_argument("--" + name.replace("_", "-"), required=True)
    s.add_argument("--tta", action="append", default=[])
    s.set_defaults(handler=start)
    f = sub.add_parser("finish")
    f.add_argument("--path", required=True)
    f.add_argument(
        "--status", choices=("complete", "failed", "interrupted"), required=True
    )
    f.set_defaults(handler=finish)
    return p


def main():
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

