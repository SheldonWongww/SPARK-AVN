#!/usr/bin/env python3
"""Resume a single ZIP member from a range-capable remote archive.

The caller supplies the member's compressed data offset and sizes, obtained
from the ZIP central directory.  Only that byte range is downloaded.  The
raw DEFLATE stream is then expanded and checked against its ZIP CRC32.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys
import time
import zlib

import requests


CONTENT_RANGE_RE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--data-offset", required=True, type=int)
    parser.add_argument("--compressed-size", required=True, type=int)
    parser.add_argument("--uncompressed-size", required=True, type=int)
    parser.add_argument("--crc32", required=True, type=lambda value: int(value, 16))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def verify_output(path: Path, expected_size: int, expected_crc: int) -> str:
    size = 0
    crc = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            size += len(chunk)
            crc = zlib.crc32(chunk, crc)
            digest.update(chunk)

    crc &= 0xFFFFFFFF
    if size != expected_size:
        raise RuntimeError(f"size mismatch for {path}: {size} != {expected_size}")
    if crc != expected_crc:
        raise RuntimeError(f"CRC32 mismatch for {path}: {crc:08x} != {expected_crc:08x}")
    return digest.hexdigest()


def download_range(args: argparse.Namespace, compressed_path: Path) -> None:
    expected_end = args.data_offset + args.compressed_size - 1
    current_size = compressed_path.stat().st_size if compressed_path.exists() else 0
    if current_size > args.compressed_size:
        raise RuntimeError(
            f"partial download is too large: {current_size} > {args.compressed_size}"
        )

    session = requests.Session()
    attempt = 0
    started = time.monotonic()
    last_reported = current_size

    while current_size < args.compressed_size:
        attempt += 1
        if attempt > args.max_attempts:
            raise RuntimeError("maximum download attempts exceeded")

        absolute_start = args.data_offset + current_size
        headers = {"Range": f"bytes={absolute_start}-{expected_end}"}
        print(
            f"attempt {attempt}: resume {current_size}/{args.compressed_size} "
            f"({100.0 * current_size / args.compressed_size:.2f}%)",
            flush=True,
        )

        try:
            with session.get(
                args.url,
                headers=headers,
                stream=True,
                timeout=(30, args.timeout),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"server ignored Range request: HTTP {response.status_code}")

                match = CONTENT_RANGE_RE.fullmatch(response.headers.get("Content-Range", ""))
                if match is None or int(match.group(1)) != absolute_start:
                    raise RuntimeError(
                        "unexpected Content-Range: "
                        + response.headers.get("Content-Range", "<missing>")
                    )

                with compressed_path.open("ab") as output:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        current_size += len(chunk)
                        if current_size - last_reported >= 256 * 1024 * 1024:
                            elapsed = max(time.monotonic() - started, 1e-6)
                            speed = (current_size - last_reported) / elapsed / (1024 * 1024)
                            print(
                                f"downloaded {current_size}/{args.compressed_size} "
                                f"({100.0 * current_size / args.compressed_size:.2f}%), "
                                f"recent average {speed:.2f} MiB/s",
                                flush=True,
                            )
                            last_reported = current_size
                            started = time.monotonic()
        except (requests.RequestException, OSError, RuntimeError) as error:
            current_size = compressed_path.stat().st_size if compressed_path.exists() else 0
            delay = min(60, 5 * attempt)
            print(f"download interrupted: {error}; retrying in {delay}s", flush=True)
            time.sleep(delay)

    if current_size != args.compressed_size:
        raise RuntimeError(f"compressed size mismatch: {current_size} != {args.compressed_size}")


def decompress(args: argparse.Namespace, compressed_path: Path) -> str:
    output_part = args.output.with_name(args.output.name + ".part")
    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    crc = 0
    size = 0
    digest = hashlib.sha256()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with compressed_path.open("rb") as source, output_part.open("wb") as output:
        while chunk := source.read(8 * 1024 * 1024):
            data = inflater.decompress(chunk)
            if data:
                output.write(data)
                size += len(data)
                crc = zlib.crc32(data, crc)
                digest.update(data)

        data = inflater.flush()
        if data:
            output.write(data)
            size += len(data)
            crc = zlib.crc32(data, crc)
            digest.update(data)

    crc &= 0xFFFFFFFF
    if not inflater.eof:
        raise RuntimeError("compressed member ended before the DEFLATE end marker")
    if size != args.uncompressed_size:
        raise RuntimeError(f"uncompressed size mismatch: {size} != {args.uncompressed_size}")
    if crc != args.crc32:
        raise RuntimeError(f"CRC32 mismatch: {crc:08x} != {args.crc32:08x}")

    output_part.replace(args.output)
    compressed_path.unlink()
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    compressed_path = args.output.with_name(args.output.name + ".deflate.part")

    if args.output.exists():
        digest = verify_output(args.output, args.uncompressed_size, args.crc32)
        print(f"already complete: {args.output}")
        print(f"sha256={digest}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    download_range(args, compressed_path)
    digest = decompress(args, compressed_path)
    print(f"complete: {args.output}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
