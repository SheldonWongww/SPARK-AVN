#!/usr/bin/env python3
"""Clone pinned, read-only upstream references from references/catalog.json."""

import argparse
import json
import os
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", action="append", help="clone only matching repository names")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(repo_root, "references", "catalog.json"), encoding="utf-8") as handle:
        entries = json.load(handle)["repositories"]
    selected = set(args.name or [])

    for entry in entries:
        if selected and entry["name"] not in selected:
            continue
        destination = os.path.join(repo_root, entry["path"])
        if os.path.isdir(os.path.join(destination, ".git")):
            head = subprocess.check_output(
                ["git", "-C", destination, "rev-parse", "HEAD"], text=True
            ).strip()
            state = "pinned" if head == entry["commit"] else "different-head"
            print("{}: exists ({})".format(entry["name"], state))
            continue
        commands = [
            ["git", "clone", entry["url"], destination],
            ["git", "-C", destination, "checkout", "--detach", entry["commit"]],
        ]
        for command in commands:
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
