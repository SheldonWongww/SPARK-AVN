#!/usr/bin/env python3
"""Download a large range-capable file in resumable parallel segments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import re
import threading
import time

import requests


CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
PRINT_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--proxy", action="append", default=[])
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def report(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def segment_bounds(total_size: int, count: int, index: int) -> tuple[int, int]:
    start = total_size * index // count
    end = total_size * (index + 1) // count - 1
    return start, end


def download_segment(args: argparse.Namespace, index: int) -> Path:
    start, end = segment_bounds(args.size, args.segments, index)
    expected_size = end - start + 1
    part = args.output.with_name(f"{args.output.name}.seg{index:02d}.part")
    current_size = part.stat().st_size if part.exists() else 0
    if current_size > expected_size:
        raise RuntimeError(f"segment {index} is too large: {current_size} > {expected_size}")

    attempt = 0
    while current_size < expected_size:
        attempt += 1
        if attempt > args.max_attempts:
            raise RuntimeError(f"segment {index} exceeded maximum attempts")

        absolute_start = start + current_size
        proxy = args.proxy[(index + attempt - 1) % len(args.proxy)] if args.proxy else None
        proxies = {"http": proxy, "https": proxy} if proxy else None
        report(
            f"segment {index}: attempt {attempt}, "
            f"{current_size}/{expected_size} bytes, proxy={proxy or 'direct'}"
        )

        try:
            with requests.get(
                args.url,
                headers={"Range": f"bytes={absolute_start}-{end}"},
                proxies=proxies,
                stream=True,
                timeout=(30, args.timeout),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"server ignored Range: HTTP {response.status_code}")

                match = CONTENT_RANGE_RE.fullmatch(response.headers.get("Content-Range", ""))
                if match is None:
                    raise RuntimeError(
                        "unexpected Content-Range: "
                        + response.headers.get("Content-Range", "<missing>")
                    )
                returned_start, returned_end, returned_total = map(int, match.groups())
                if returned_start != absolute_start or returned_end != end:
                    raise RuntimeError(
                        f"unexpected returned range: {returned_start}-{returned_end}"
                    )
                if returned_total != args.size:
                    raise RuntimeError(f"source size mismatch: {returned_total} != {args.size}")

                with part.open("ab") as output:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        current_size += len(chunk)

        except (OSError, RuntimeError, requests.RequestException) as error:
            current_size = part.stat().st_size if part.exists() else 0
            delay = min(60, max(5, attempt * 2))
            report(f"segment {index}: interrupted at {current_size}: {error}; retry in {delay}s")
            time.sleep(delay)

    if current_size != expected_size:
        raise RuntimeError(f"segment {index} size mismatch: {current_size} != {expected_size}")
    report(f"segment {index}: complete")
    return part


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    if args.segments < 1:
        raise SystemExit("--segments must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        size = args.output.stat().st_size
        if size != args.size:
            raise SystemExit(f"existing output has wrong size: {size} != {args.size}")
        print(f"already complete: {args.output}")
        print(f"sha256={sha256_file(args.output)}")
        return 0

    with ThreadPoolExecutor(max_workers=args.segments) as executor:
        parts = list(executor.map(lambda index: download_segment(args, index), range(args.segments)))

    output_part = args.output.with_name(args.output.name + ".part")
    with output_part.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    output.write(chunk)

    if output_part.stat().st_size != args.size:
        raise SystemExit(
            f"assembled size mismatch: {output_part.stat().st_size} != {args.size}"
        )

    output_part.replace(args.output)
    for part in parts:
        part.unlink()

    print(f"complete: {args.output}")
    print(f"sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
