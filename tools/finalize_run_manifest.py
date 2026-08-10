#!/usr/bin/env python3
"""Atomically mark a NavTTA run manifest completed or failed."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="compact result artifact as NAME=PATH",
    )
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["exit_code"] = args.exit_code
    manifest["status"] = "completed" if args.exit_code == 0 else "failed"
    if args.artifact:
        artifacts = []
        names = set()
        for item in args.artifact:
            if "=" not in item:
                parser.error("--artifact must use NAME=PATH")
            name, path = item.split("=", 1)
            path = os.path.abspath(path)
            if not name or name in names or not os.path.isfile(path):
                parser.error("invalid result artifact: {}".format(item))
            names.add(name)
            artifacts.append(
                {
                    "name": name,
                    "path": path,
                    "size": os.path.getsize(path),
                    "sha256": sha256(path),
                }
            )
        manifest["result_artifacts"] = artifacts

    temporary = manifest_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, manifest_path)


if __name__ == "__main__":
    main()
