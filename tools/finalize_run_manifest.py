#!/usr/bin/env python3
"""Atomically mark a NavTTA run manifest completed or failed."""

import argparse
import json
import os
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["exit_code"] = args.exit_code
    manifest["status"] = "completed" if args.exit_code == 0 else "failed"

    temporary = manifest_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, manifest_path)


if __name__ == "__main__":
    main()
