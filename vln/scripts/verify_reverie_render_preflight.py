#!/usr/bin/env python3
"""Create fail-closed evidence for one real headless MatterSim panorama.

This verifier never downloads or repairs Matterport assets.  It proves that a
separately supplied rendering-enabled MatterSim build can read the exact raw
RGB cubemap and produce the canonical 36-view contact sheet used by the
REVERIE FeedTTA-LLM provider.
"""

from __future__ import absolute_import

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
VLN_ROOT = REPO_ROOT / "vln"
PROVIDER_PATH = VLN_ROOT / "navtta_vln" / "reverie_llm_feedback.py"
PROVIDER_SPEC = importlib.util.spec_from_file_location(
    "_navtta_reverie_render_preflight", str(PROVIDER_PATH)
)
PROVIDER = importlib.util.module_from_spec(PROVIDER_SPEC)
PROVIDER_SPEC.loader.exec_module(PROVIDER)

SCHEMA = "navtta.reverie_mattersim_render_preflight.v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class PreflightError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_evidence(path, role):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise PreflightError("missing required {}: {}".format(role, path))
    return {
        "role": role,
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def verify_file_evidence_unchanged(records):
    for record in records:
        current = file_evidence(record["path"], record["role"])
        if (
            current["size_bytes"] != record["size_bytes"]
            or current["sha256"] != record["sha256"]
        ):
            raise PreflightError(
                "required input changed during render: {}".format(
                    record["path"]
                )
            )


def _cache_bool(cache_text, name):
    match = re.search(
        r"(?m)^{}:BOOL=(ON|OFF)$".format(re.escape(name)), cache_text
    )
    if match is None:
        raise PreflightError(
            "MatterSim CMake cache does not declare {}".format(name)
        )
    return match.group(1) == "ON"


def verify_headless_build(build_dir):
    build_dir = Path(build_dir).expanduser().resolve()
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        raise PreflightError(
            "missing MatterSim CMakeCache.txt; a pinned rendering build is required"
        )
    cache_text = cache_path.read_text(encoding="utf-8", errors="strict")
    egl = _cache_bool(cache_text, "EGL_RENDERING")
    osmesa = _cache_bool(cache_text, "OSMESA_RENDERING")
    enabled = [name for name, value in (("egl", egl), ("osmesa", osmesa)) if value]
    if len(enabled) != 1:
        raise PreflightError(
            "MatterSim is not a uniquely configured headless renderer: "
            "EGL_RENDERING={}, OSMESA_RENDERING={}; the standard NavTTA "
            "navigation build intentionally has both OFF, so provide a "
            "separate pinned EGL or OSMesa build"
            .format("ON" if egl else "OFF", "ON" if osmesa else "OFF")
        )
    modules = sorted(build_dir.glob("MatterSim*.so"))
    if len(modules) != 1:
        raise PreflightError(
            "expected exactly one MatterSim extension in {}; found {}"
            .format(build_dir, len(modules))
        )
    return {
        "build_dir": str(build_dir),
        "backend": enabled[0],
        "cmake_cache": file_evidence(cache_path, "mattersim_cmake_cache"),
        "module_candidate": file_evidence(modules[0], "mattersim_extension"),
    }


def verify_import_locations(mattersim_build):
    core_root = (REPO_ROOT / "core").resolve()
    try:
        import navtta_core
    except Exception as error:
        raise PreflightError("cannot import current repository navtta_core") from error
    core_module = Path(navtta_core.__file__).resolve()
    try:
        core_module.relative_to(core_root)
    except ValueError as error:
        raise PreflightError(
            "navtta_core import is outside current repository core: {}"
            .format(core_module)
        ) from error

    build_dir = Path(mattersim_build).resolve()
    sys.path.insert(0, str(build_dir))
    try:
        mattersim = importlib.import_module("MatterSim")
    except Exception as error:
        raise PreflightError("render-capable MatterSim import failed") from error
    module_path = Path(mattersim.__file__).resolve()
    try:
        module_path.relative_to(build_dir)
    except ValueError as error:
        raise PreflightError(
            "MatterSim import escaped requested render build: {}".format(module_path)
        ) from error
    return {
        "navtta_core": file_evidence(core_module, "navtta_core_module"),
        "mattersim": file_evidence(module_path, "loaded_mattersim_extension"),
    }


def verify_source_assets(connectivity_dir, scan_data_dir, scan, viewpoint):
    for value, label in ((scan, "scan"), (viewpoint, "viewpoint")):
        if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
            raise PreflightError("invalid {} identifier".format(label))
    connectivity_dir = Path(connectivity_dir).expanduser().resolve()
    scan_data_dir = Path(scan_data_dir).expanduser().resolve()
    if not connectivity_dir.is_dir():
        raise PreflightError(
            "missing connectivity directory: {}".format(connectivity_dir)
        )
    if not scan_data_dir.is_dir():
        raise PreflightError(
            "missing Matterport raw RGB root: {}".format(scan_data_dir)
        )
    scans_path = connectivity_dir / "scans.txt"
    rgb_path = (
        scan_data_dir / scan / "matterport_skybox_images"
        / (viewpoint + "_skybox_small.jpg")
    )
    scans_evidence = file_evidence(scans_path, "connectivity_scan_list")
    scan_ids = {
        line.strip() for line in scans_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not scan_ids or any(SAFE_ID.fullmatch(item) is None for item in scan_ids):
        raise PreflightError("connectivity scans.txt contains invalid scan IDs")
    if scan not in scan_ids:
        raise PreflightError("scan is absent from connectivity scans.txt")
    connectivity_evidence = []
    for scan_id in sorted(scan_ids):
        item = file_evidence(
            connectivity_dir / (scan_id + "_connectivity.json"),
            "connectivity_graph",
        )
        item["scan_id"] = scan_id
        connectivity_evidence.append(item)
    connectivity_path = connectivity_dir / (scan + "_connectivity.json")
    raw_rgb_evidence = file_evidence(rgb_path, "raw_rgb_cubemap")
    required = [scans_evidence] + connectivity_evidence + [raw_rgb_evidence]
    try:
        connectivity = json.loads(connectivity_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError("scan connectivity JSON is unreadable") from error
    matches = [
        item for item in connectivity
        if isinstance(item, dict) and str(item.get("image_id")) == viewpoint
    ] if isinstance(connectivity, list) else []
    if len(matches) != 1 or matches[0].get("included") is not True:
        raise PreflightError(
            "viewpoint is missing, duplicated, or excluded in connectivity"
        )
    try:
        from PIL import Image

        with Image.open(str(rgb_path)) as image:
            image.load()
            raw_image = {
                "format": image.format,
                "mode": image.mode,
                "width": int(image.width),
                "height": int(image.height),
            }
    except Exception as error:
        raise PreflightError("raw RGB cubemap is not a readable image") from error
    if (
        raw_image["format"] != "JPEG"
        or raw_image["mode"] != "RGB"
        or raw_image["width"] <= 0
        or raw_image["height"] <= 0
        or raw_image["width"] % 6 != 0
    ):
        raise PreflightError("raw RGB cubemap layout is not canonical")
    return required, raw_image


def repository_identity(require_clean):
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain",
         "--untracked-files=no"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    ).stdout
    clean = not bool(status.strip())
    if require_clean and not clean:
        raise PreflightError("formal render preflight requires a clean tracked tree")
    return {"root": str(REPO_ROOT), "git_commit": commit, "tracked_tree_clean": clean}


def _atomic_write(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def run_preflight(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        output_dir.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise PreflightError(
            "render evidence must be outside the Git repository"
        )
    if output_dir.exists():
        raise PreflightError(
            "render evidence directory already exists: {}".format(output_dir)
        )

    build = verify_headless_build(args.mattersim_build)
    imports = verify_import_locations(args.mattersim_build)
    required_files, raw_image = verify_source_assets(
        args.connectivity_dir, args.scan_data_dir, args.scan, args.viewpoint
    )
    renderer = PROVIDER.MatterSimPanoramaRenderer(
        args.connectivity_dir, args.scan_data_dir
    )
    try:
        png, panorama = renderer.render(args.scan, args.viewpoint)
        PROVIDER.Qwen2VLFeedbackProvider._validate_panorama(png, panorama)
    except Exception as error:
        raise PreflightError(
            "real 36-view MatterSim headless render failed"
        ) from error

    try:
        from PIL import Image, ImageStat

        image = Image.open(BytesIO(png))
        image.load()
        extrema = [list(pair) for pair in image.getextrema()]
        standard_deviation = [float(value) for value in ImageStat.Stat(image).stddev]
    except Exception as error:
        raise PreflightError("rendered panorama cannot be inspected") from error
    if not any(high > low for low, high in extrema):
        raise PreflightError("rendered panorama is a constant image")
    verify_file_evidence_unchanged(required_files)
    verify_file_evidence_unchanged([
        build["cmake_cache"], build["module_candidate"],
        imports["navtta_core"], imports["mattersim"],
    ])

    repository = repository_identity(args.require_clean_git)
    output_dir.mkdir(parents=True, exist_ok=False)
    panorama_path = output_dir / "endpoint_panorama.png"
    evidence_path = output_dir / "render_evidence.json"
    _atomic_write(panorama_path, png)
    document = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "verifier": file_evidence(Path(__file__), "render_preflight_verifier"),
        "scan_id": args.scan,
        "viewpoint_id": args.viewpoint,
        "headless_build": build,
        "imports": imports,
        "required_input_files": required_files,
        "raw_rgb": {
            "file": next(
                item for item in required_files
                if item["role"] == "raw_rgb_cubemap"
            ),
            "image": raw_image,
        },
        "render": {
            "metadata": panorama,
            "output_path": str(panorama_path),
            "output_size_bytes": len(png),
            "output_sha256": hashlib.sha256(png).hexdigest(),
            "channel_extrema": extrema,
            "channel_standard_deviation": standard_deviation,
        },
        "status": "passed",
    }
    encoded = (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    _atomic_write(evidence_path, encoded)
    return {
        "schema": "navtta.reverie_mattersim_render_preflight_pointer.v1",
        "status": "passed",
        "evidence_path": str(evidence_path),
        "evidence_sha256": hashlib.sha256(encoded).hexdigest(),
        "panorama_path": str(panorama_path),
        "panorama_sha256": hashlib.sha256(png).hexdigest(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mattersim-build", required=True)
    parser.add_argument("--connectivity-dir", required=True)
    parser.add_argument("--scan-data-dir", required=True)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--viewpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        output = run_preflight(parse_args(argv))
    except (PreflightError, OSError, subprocess.SubprocessError) as error:
        print("error: {}".format(error), file=sys.stderr)
        return 1
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
