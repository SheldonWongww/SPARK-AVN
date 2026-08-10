#!/usr/bin/env python3
"""Create a compact, deterministic export of a completed VLN TTA search.

The search scheduler intentionally keeps raw model outputs outside Git.  This
utility turns one completed five-method campaign into a portable evidence
bundle containing only text, JSON, and CSV.  It never follows checkpoint or
prediction paths recorded by run manifests.
"""

import argparse
import csv
from collections import deque
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_tta_hparam_search as search  # noqa: E402
from tools.run_manifest_identity import (  # noqa: E402
    immutable_identity_sha256,
)


DEFAULT_SEARCH_ROOT = REPO_ROOT / "vln/results/logs/hparam_search"
DEFAULT_RUNS_ROOT = REPO_ROOT / "vln/results/runs"
DEFAULT_EXPORT_ROOT = REPO_ROOT / "vln/results/logs/hparam_exports"
DEFAULT_ASSET_MANIFEST = REPO_ROOT / "vln/manifests/assets/eval_assets.json"
DEFAULT_ENVIRONMENT_MANIFEST = (
    REPO_ROOT / "vln/manifests/environments/eval_environments.json"
)
DEFAULT_ORDER_ROOT = REPO_ROOT / "vln/manifests/episode_order"
METHODS = tuple(search.METHODS)
SETTINGS = tuple(search.load_spec()["settings"])
STRICT_STAGES = {"smoke", "controls", "final", "final_controls", "orders"}
STAGE_FILES = (
    "stage_manifest.json",
    "SUMMARY.json",
    "metrics.csv",
    "promotion_preview.json",
    "progress.json",
    "resource.csv",
    "RESOURCE_PROFILE.json",
    "grid.csv",
)
OPTIONAL_METHOD_FILES = (
    "FINAL_SELECTION.json",
    "FROZEN_HPARAMETERS.json",
    "ORDER_ROBUSTNESS.json",
)
PROHIBITED_PARTS = {
    "checkpoints", "checkpoint", "predictions", "prediction", "preds",
    "tensorboard", "videos", "video",
}
PROHIBITED_SUFFIXES = {
    ".pth", ".pt", ".ckpt", ".safetensors", ".bin", ".onnx",
    ".pkl", ".pickle", ".h5", ".hdf5", ".npy", ".npz",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
}
CONSOLE_SIGNAL = re.compile(
    r"(?:Env name:|Average episode|Traceback|\b(?:error|exception|failed)\b|"
    r"RuntimeError|CUDA|out of memory|scheduler_(?:start|finish)|"
    r"TTA diagnostics|StreamVLN)",
    re.IGNORECASE,
)
SAFE_BATCH_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
MAX_CONSOLE_LINE_BYTES = 16 * 1024
MAX_CONSOLE_SIGNAL_LINES = 200
SETTING_CHECKPOINT_ASSET = {
    "duet-r2r": "duet_r2r_checkpoint",
    "duet-reverie": "duet_reverie_checkpoint",
    "hamt-r2r": "hamt_r2r_e2e_checkpoint",
    "hamt-reverie": "hamt_reverie_checkpoint",
    "goat-r2r": "goat_r2r_checkpoint",
    "goat-reverie": "goat_reverie_checkpoint",
    "etpnav-r2r-ce": "etpnav_checkpoint",
    "bevbert-r2r-ce": "bevbert_ce_checkpoint",
}
SETTING_AUXILIARY_ASSETS = {
    "duet-r2r": {
        "pano_features": "duet_r2r_pano_features",
        "submission_viewpoint_candidates": "goat_candidate_relative_angles",
        "mattersim_python": "@matterport3d_simulator_build",
    },
    "duet-reverie": {
        "pano_features": "duet_r2r_pano_features",
        "object_features": "duet_reverie_object_features",
        "object_boxes": "reverie_bboxes",
        "submission_viewpoint_candidates": "goat_candidate_relative_angles",
        "mattersim_python": "@matterport3d_simulator_build",
    },
    "hamt-r2r": {
        "pano_features": "hamt_r2r_e2e_features",
        "mattersim_python": "@matterport3d_simulator_build",
    },
    "hamt-reverie": {
        "pano_features": "hamt_r2r_e2e_features",
        "object_features": "hamt_reverie_object_features",
        "object_boxes": "reverie_bboxes",
        "mattersim_python": "@matterport3d_simulator_build",
    },
    "goat-r2r": {
        "pano_features": "goat_r2r_clip_features",
        "backdoor": "goat_r2r_updated_backdoor",
        "frontdoor": "goat_r2r_updated_frontdoor",
        "submission_viewpoint_candidates": "goat_candidate_relative_angles",
        "mattersim_python": "@matterport3d_simulator_build",
    },
    "goat-reverie": {
        "pano_features": "goat_r2r_clip_features",
        "backdoor": "goat_reverie_updated_backdoor",
        "frontdoor": "goat_reverie_updated_frontdoor",
        "submission_viewpoint_candidates": "goat_candidate_relative_angles",
        "object_features": "goat_reverie_object_features",
        "object_boxes": "goat_reverie_bboxes",
        "mattersim_python": "@matterport3d_simulator_build",
    },
    "etpnav-r2r-ce": {
        "waypoint_predictor": "etpnav_waypoint_predictor",
        "depth_encoder": "ce_ddppo_depth_encoder",
        "clip_vit_b32": "openai_clip_vit_b32",
    },
    "bevbert-r2r-ce": {
        "waypoint_predictor": "etpnav_waypoint_predictor",
        "depth_encoder": "ce_ddppo_depth_encoder",
        "clip_vit_b16": "openai_clip_vit_b16",
    },
}
SETTING_ORDER_DIRECTORY = {
    "duet-r2r": "r2r_duet_hamt",
    "duet-reverie": "reverie_duet_hamt",
    "hamt-r2r": "r2r_duet_hamt",
    "hamt-reverie": "reverie_duet_hamt",
    "goat-r2r": "r2r_goat",
    "goat-reverie": "reverie_goat",
    "etpnav-r2r-ce": "r2r_ce_v1_3_unified",
    "bevbert-r2r-ce": "r2r_ce_v1_3_unified",
}
CONTINUOUS_PER_EPISODE_ARTIFACT = {
    "etpnav-r2r-ce": (
        "metrics/source_val_seen/"
        "stats_ep_ckpt_59_val_seen_r0_w1.json"
    ),
    "bevbert-r2r-ce": (
        "metrics/source_val_seen/"
        "stats_ep_ckpt_47_val_seen_r0_w1.json"
    ),
}
CONTINUOUS_PER_EPISODE_METRICS = (
    "steps_taken",
    "distance_to_goal",
    "success",
    "oracle_success",
    "path_length",
    "collisions",
    "spl",
    "ndtw",
    "sdtw",
    "ghost_cnt",
)
PER_EPISODE_STAGES = {"final_controls", "final", "orders"}


class ExportError(RuntimeError):
    """The campaign is incomplete, inconsistent, or unsafe to export."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError("cannot read JSON {}: {}".format(path, error))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def validate_batch_id(value):
    if (not isinstance(value, str) or value in (".", "..")
            or SAFE_BATCH_ID.fullmatch(value) is None):
        raise ExportError(
            "batch_id must be one safe path component: {!r}".format(value)
        )
    return value


def path_is_within(path, root):
    try:
        Path(path).relative_to(root)
        return True
    except ValueError:
        return False


def reject_symlink(path, label, stop=None):
    path = Path(path).absolute()
    stop = Path(stop).absolute() if stop is not None else None
    current = path
    while True:
        if current.is_symlink():
            raise ExportError("{} uses a symlink: {}".format(label, current))
        if stop is None or current == stop:
            break
        if current.parent == current:
            break
        current = current.parent


def require_regular_file(path, label, stop=None):
    path = Path(path)
    reject_symlink(path, label, stop=stop)
    if not path.is_file():
        raise ExportError("{} is not a regular file: {}".format(label, path))
    return path


def safe_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def ensure_safe_export_path(path, root):
    path = Path(path)
    relative = path.relative_to(root)
    lowered = {part.lower() for part in relative.parts}
    if lowered.intersection(PROHIBITED_PARTS):
        raise ExportError("prohibited export path: {}".format(relative))
    if path.suffix.lower() in PROHIBITED_SUFFIXES:
        raise ExportError("prohibited export suffix: {}".format(relative))
    if path.name.startswith("events.out.tfevents"):
        raise ExportError("TensorBoard event escaped export filter")


def iter_bounded_console_records(source, max_line_bytes=MAX_CONSOLE_LINE_BYTES,
                                 chunk_bytes=64 * 1024):
    """Yield CR/LF-delimited UTF-8 records without retaining huge lines."""
    prefix = bytearray()
    truncated = False
    skip_lf = False

    def append(segment):
        nonlocal truncated
        remaining = max_line_bytes - len(prefix)
        if remaining > 0:
            prefix.extend(segment[:remaining])
        if len(segment) > remaining:
            truncated = True

    def emit():
        nonlocal prefix, truncated
        text = bytes(prefix).decode("utf-8", errors="replace")
        if truncated:
            text += " …[line truncated]"
        prefix = bytearray()
        truncated = False
        return text

    with Path(source).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            start = 0
            if skip_lf:
                if chunk.startswith(b"\n"):
                    start = 1
                skip_lf = False
            while start < len(chunk):
                cr = chunk.find(b"\r", start)
                lf = chunk.find(b"\n", start)
                positions = [value for value in (cr, lf) if value >= 0]
                if not positions:
                    append(chunk[start:])
                    break
                end = min(positions)
                append(chunk[start:end])
                delimiter = chunk[end]
                if delimiter == 13 and end + 1 == len(chunk):
                    skip_lf = True
                    start = end + 1
                elif delimiter == 13 and chunk[end + 1:end + 2] == b"\n":
                    start = end + 2
                else:
                    start = end + 1
                yield emit()
    if prefix or truncated:
        yield emit()


def compact_console(source, destination, source_label, first_lines=12,
                    last_lines=80, signal_limit=MAX_CONSOLE_SIGNAL_LINES):
    """Write a line-numbered head/signal/tail representation of a log."""
    source = Path(source)
    head = []
    tail = deque(maxlen=last_lines)
    signal_head = []
    signal_tail = deque(maxlen=max(0, signal_limit // 2))
    signal_count = 0
    count = 0
    signal_head_limit = signal_limit - signal_tail.maxlen
    for count, line in enumerate(iter_bounded_console_records(source), 1):
        item = (count, line)
        if count <= first_lines:
            head.append(item)
        tail.append(item)
        if CONSOLE_SIGNAL.search(line):
            signal_count += 1
            if len(signal_head) < signal_head_limit:
                signal_head.append(item)
            else:
                signal_tail.append(item)
    signals = signal_head + list(signal_tail)
    selected = {}
    for number, line in head + signals + list(tail):
        selected[number] = line
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# compact NavTTA console export\n")
        stream.write("# source: {}\n".format(source_label))
        stream.write("# source_sha256: {}\n".format(sha256(source)))
        stream.write("# source_bytes: {}\n".format(source.stat().st_size))
        stream.write("# source_lines: {}\n".format(count))
        stream.write("# source_signal_lines: {}\n".format(signal_count))
        stream.write("# retained_lines: {}\n".format(len(selected)))
        for number in sorted(selected):
            stream.write("[L{:08d}] {}\n".format(number, selected[number]))


def load_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _extract_selected_run_tag(document, setting):
    if not isinstance(document, dict):
        return None
    settings = document.get("settings")
    if not isinstance(settings, dict) or setting not in settings:
        return None
    value = settings[setting]
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    for key in ("selected_run_tag", "winner_run_tag", "run_tag"):
        if isinstance(value.get(key), str):
            return value[key]
    winner = value.get("winner") or value.get("selected")
    if isinstance(winner, str):
        return winner
    if isinstance(winner, dict) and isinstance(winner.get("run_tag"), str):
        return winner["run_tag"]
    return None


def _extract_frozen_parameters(document, setting):
    if not isinstance(document, dict):
        return None
    settings = document.get("settings")
    if not isinstance(settings, dict) or setting not in settings:
        return None
    value = settings[setting]
    if not isinstance(value, dict):
        return None
    if isinstance(value.get("frozen_parameters"), dict):
        return value["frozen_parameters"]
    if isinstance(value.get("parameters"), dict):
        return value["parameters"]
    # Current FROZEN_HPARAMETERS.json stores the parameter dictionary directly.
    if not any(key in value for key in (
            "selected_run_tag", "winner_run_tag", "run_tag", "metrics")):
        return value
    return None


class BatchExporter:
    def __init__(self, batch_id, method_roots, runs_root, output_dir,
                 spec_path=search.SPEC_PATH, export_root=DEFAULT_EXPORT_ROOT,
                 allow_legacy_prefix_controls=False,
                 asset_manifest_path=DEFAULT_ASSET_MANIFEST,
                 environment_manifest_path=DEFAULT_ENVIRONMENT_MANIFEST,
                 order_root=DEFAULT_ORDER_ROOT):
        self.batch_id = validate_batch_id(batch_id)
        self.method_roots = {key: Path(value) for key, value in method_roots.items()}
        self.runs_root = Path(runs_root)
        self.output_dir = Path(output_dir)
        self.export_root = Path(export_root)
        self.allow_legacy_prefix_controls = bool(allow_legacy_prefix_controls)
        self.asset_manifest_path = require_regular_file(
            asset_manifest_path, "canonical VLN asset manifest"
        )
        self.environment_manifest_path = require_regular_file(
            environment_manifest_path, "canonical VLN environment manifest"
        )
        self.order_root = Path(order_root)
        reject_symlink(self.order_root, "canonical episode-order root")
        self.spec_path = Path(spec_path)
        require_regular_file(self.spec_path, "search specification")
        self.spec = search.load_spec(self.spec_path)
        self.settings = tuple(self.spec["settings"])
        self.source_inventory = []
        self.formal_by_run_tag = self._index_formal_manifests()
        self.stage_rows = []
        self.selected_rows = []
        self.frozen_rows = []
        self.rejection_rows = []
        self.seen_run_tags = set()
        self.file_hash_cache = {}
        self.canonical_assets = self._load_canonical_assets()
        self.canonical_environment = read_json(self.environment_manifest_path)
        self.canonical_orders = self._load_canonical_orders()
        self.formal_identity_by_setting = {}
        self.per_episode_source_by_run_tag = {}
        self.per_episode_exports = []

    def _load_canonical_assets(self):
        document = read_json(self.asset_manifest_path)
        if document.get("schema_version") != 1 or not isinstance(
                document.get("assets"), list):
            raise ExportError("canonical VLN asset manifest schema is invalid")
        output = {}
        for item in document["assets"]:
            asset_id = item.get("id") if isinstance(item, dict) else None
            if (not isinstance(asset_id, str) or not asset_id
                    or asset_id in output):
                raise ExportError("canonical VLN asset IDs are invalid")
            digest = item.get("sha256")
            if not isinstance(digest, str) or HEX_SHA256.fullmatch(digest) is None:
                raise ExportError("canonical VLN asset SHA256 is invalid")
            output[asset_id] = item
        required_ids = set(SETTING_CHECKPOINT_ASSET.values())
        for mapping in SETTING_AUXILIARY_ASSETS.values():
            required_ids.update(value for value in mapping.values()
                                if not value.startswith("@"))
        missing = sorted(required_ids.difference(output))
        if missing:
            raise ExportError("canonical VLN assets are missing: {}".format(
                ", ".join(missing)
            ))
        return output

    def _load_canonical_orders(self):
        output = {}
        for setting, directory in SETTING_ORDER_DIRECTORY.items():
            path = self.order_root / directory / "val_seen.json"
            require_regular_file(path, "canonical {} episode order".format(setting),
                                 stop=self.order_root)
            document = read_json(path)
            if (document.get("schema") != "navtta.episode_order.v1"
                    or document.get("split") != "val_seen"):
                raise ExportError("canonical episode order is invalid for {}".format(
                    setting
                ))
            output[setting] = {
                "path": path.absolute(),
                "sha256": sha256(path),
                "document": document,
            }
        return output

    def _validated_output_paths(self):
        export_root = self.export_root.absolute()
        output = self.output_dir.absolute()
        reject_symlink(export_root, "export root")
        if export_root.exists() and not export_root.is_dir():
            raise ExportError("export root is not a directory: {}".format(export_root))
        if output == export_root or not path_is_within(output, export_root):
            raise ExportError("output must be a child of export root {}: {}".format(
                export_root, output
            ))
        if output.is_symlink():
            raise ExportError("output directory uses a symlink: {}".format(output))
        resolved_root = export_root.resolve(strict=False)
        resolved_output = output.resolve(strict=False)
        if (resolved_output == resolved_root
                or not path_is_within(resolved_output, resolved_root)):
            raise ExportError(
                "resolved output escapes export root {}: {}".format(
                    resolved_root, resolved_output
                )
            )
        if output.exists() or output.is_symlink():
            reject_symlink(output, "output directory", stop=export_root)
            if not output.is_dir():
                raise ExportError("output exists and is not a directory: {}".format(
                    output
                ))
        else:
            existing = output.parent
            while not existing.exists() and existing != export_root:
                existing = existing.parent
            reject_symlink(existing, "output parent", stop=export_root)
        return export_root, output

    def _verified_file_sha256(self, path, label):
        path = require_regular_file(path, label)
        stat = path.stat()
        key = (str(path.absolute()), stat.st_size, stat.st_mtime_ns)
        if key not in self.file_hash_cache:
            self.file_hash_cache[key] = sha256(path)
        return self.file_hash_cache[key]

    def _expected_auxiliary_sha256(self, setting):
        output = {}
        for name, asset_id in SETTING_AUXILIARY_ASSETS[setting].items():
            if asset_id == "@matterport3d_simulator_build":
                try:
                    digest = self.canonical_environment["native_dependencies"][
                        "matterport3d_simulator_build"
                    ]["sha256"]
                except (KeyError, TypeError):
                    raise ExportError(
                        "canonical environment lacks MatterSim build SHA256"
                    )
            else:
                digest = self.canonical_assets[asset_id]["sha256"]
            if not isinstance(digest, str) or HEX_SHA256.fullmatch(digest) is None:
                raise ExportError("canonical auxiliary SHA256 is invalid")
            output[name] = digest
        return output

    def _index_formal_manifests(self):
        values = {}
        if not self.runs_root.is_dir():
            return values
        reject_symlink(self.runs_root, "formal runs root")
        for path in sorted(self.runs_root.glob("*/manifest.json")):
            require_regular_file(path, "formal run manifest", stop=self.runs_root)
            document = read_json(path)
            run_tag = document.get("run_tag")
            if not isinstance(run_tag, str) or not run_tag.startswith(self.batch_id):
                continue
            values.setdefault(run_tag, []).append((path, document))
        return values

    def _record_copy(self, source, destination, source_label):
        self.source_inventory.append({
            "source": source_label,
            "source_bytes": Path(source).stat().st_size,
            "source_sha256": sha256(source),
            "exported": destination.relative_to(self.working_dir).as_posix(),
            "exported_bytes": destination.stat().st_size,
            "exported_sha256": sha256(destination),
        })

    def _validate_file_metadata(self, metadata, hash_field, label,
                                manifest_directory=None):
        if not isinstance(metadata, dict):
            raise ExportError("{} metadata is not an object".format(label))
        path_value = metadata.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ExportError("{} path is missing".format(label))
        path = Path(path_value)
        if not path.is_absolute():
            if manifest_directory is None:
                raise ExportError("{} path is unexpectedly relative".format(label))
            path = Path(manifest_directory) / path
        path = require_regular_file(path, label)
        size = metadata.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ExportError("{} size metadata is invalid".format(label))
        if path.stat().st_size != size:
            raise ExportError("{} size mismatch".format(label))
        expected_hash = metadata.get(hash_field)
        if not isinstance(expected_hash, str) or HEX_SHA256.fullmatch(
                expected_hash) is None:
            raise ExportError("{} SHA256 metadata is invalid".format(label))
        if self._verified_file_sha256(path, label) != expected_hash:
            raise ExportError("{} SHA256 mismatch".format(label))
        return path

    def _validate_job_config(self, job, job_dir, config_document):
        identity_fields = {
            "method": job.get("config_method"),
            "search_method": job.get("search_method"),
            "stage": job.get("stage"),
            "episodes": job.get("episodes"),
            "order_seed": job.get("order_seed"),
            "parameters": job.get("parameters"),
        }
        if config_document.get("schema") != "navtta.vln_tta_job.v1":
            raise ExportError("unsupported job parameter schema")
        for key, expected in identity_fields.items():
            if canonical(config_document.get(key)) != canonical(expected):
                raise ExportError("job/parameters identity mismatch: {}".format(key))
        expected_path = (job_dir / "parameters.json").absolute()
        declared_value = Path(job.get("config_path", ""))
        if not declared_value.is_absolute():
            raise ExportError("job config_path must be absolute")
        declared_path = declared_value.absolute()
        reject_symlink(declared_path, "declared job config")
        if declared_path != expected_path:
            raise ExportError("job config_path does not name its parameters.json")
        declared_job_value = Path(job.get("job_dir", ""))
        if not declared_job_value.is_absolute():
            raise ExportError("job job_dir must be absolute")
        declared_job_dir = declared_job_value.absolute()
        reject_symlink(declared_job_dir, "declared job directory")
        if declared_job_dir != job_dir.absolute():
            raise ExportError("job job_dir identity mismatch")

    def _validate_metrics_identity(self, job, result, csv_row, expected_episodes):
        identity_fields = (
            "ordinal", "base_run_tag", "run_tag", "attempt", "setting",
            "model", "family", "benchmark", "search_method", "config_method",
            "stage", "episodes", "order_seed", "parameters",
            "parent_run_tags", "config_path", "job_dir", "result_root", "command",
        )
        for key in identity_fields:
            if canonical(result.get(key)) != canonical(job.get(key)):
                raise ExportError("job/metrics identity mismatch for {}: {}".format(
                    job.get("run_tag"), key
                ))
        if int(result.get("expected_episodes", -1)) != expected_episodes:
            raise ExportError("metrics expected_episodes mismatch for {}".format(
                job["run_tag"]
            ))
        if result.get("requires_posthoc_late_collapse_check") is not True:
            raise ExportError("metrics omit the late-collapse review marker")
        for key in ("run_tag", "setting", "search_method", "config_method", "stage"):
            if csv_row.get(key, "") != str(job.get(key, "")):
                raise ExportError("metrics.csv identity mismatch for {}: {}".format(
                    job["run_tag"], key
                ))
        expected_order = "" if job.get("order_seed") is None else str(
            job["order_seed"]
        )
        if csv_row.get("order_seed", "") != expected_order:
            raise ExportError("metrics.csv order_seed mismatch for {}".format(
                job["run_tag"]
            ))
        metrics = result.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ExportError("metrics payload is empty for {}".format(job["run_tag"]))
        for name, value in metrics.items():
            try:
                csv_value = float(csv_row[name])
                metric_value = float(value)
            except (KeyError, TypeError, ValueError):
                raise ExportError("metrics.csv lacks numeric {} for {}".format(
                    name, job["run_tag"]
                ))
            if csv_value != metric_value:
                raise ExportError("metrics.csv value mismatch for {} {}".format(
                    job["run_tag"], name
                ))

    def _validate_diagnostics(self, job, result, expected_episodes):
        result_root = Path(job.get("result_root", ""))
        if not result_root.is_absolute() or not result_root.is_dir():
            raise ExportError("job result_root is not an absolute directory")
        reject_symlink(result_root, "job result root")
        expected_path = (result_root / "tta_diagnostics.json").absolute()
        recorded_path = result.get("diagnostics_path")
        source_job = job.get("config_method") == "source"
        continuous_source = source_job and job.get("family") == "continuous"
        sampled_discrete_source = (
            source_job
            and job.get("family") == "discrete"
            and job.get("parameters", {}).get("action_selection") == "sample"
        )
        diagnostics_required = (
            not source_job or continuous_source or sampled_discrete_source
        )
        if recorded_path in (None, ""):
            if result.get("diagnostics_sha256") not in (None, ""):
                raise ExportError("metrics record a diagnostics hash without a path")
            if expected_path.exists():
                raise ExportError("metrics omit existing TTA diagnostics")
            if result.get("adapter_diagnostics") is not None:
                raise ExportError("metrics contain adapter diagnostics without evidence")
            if diagnostics_required:
                raise ExportError("job omits required TTA diagnostics")
            return None
        if not isinstance(recorded_path, str) or not recorded_path:
            raise ExportError("TTA metrics omit diagnostics_path")
        path = Path(recorded_path).absolute()
        if path != expected_path:
            raise ExportError("diagnostics must be result_root/tta_diagnostics.json")
        require_regular_file(path, "TTA diagnostics", stop=result_root)
        recorded_hash = result.get("diagnostics_sha256")
        if not isinstance(recorded_hash, str) or HEX_SHA256.fullmatch(
                recorded_hash) is None:
            raise ExportError("diagnostics_sha256 is invalid")
        if self._verified_file_sha256(path, "TTA diagnostics") != recorded_hash:
            raise ExportError("TTA diagnostics SHA256 mismatch")
        document = read_json(path)
        expected_schema = (
            "navtta.vln_ce_tta.v1"
            if job.get("family") == "continuous"
            else "navtta.vln_discrete_tta.v1"
        )
        if document.get("schema") != expected_schema:
            raise ExportError("TTA diagnostics schema mismatch")
        if document.get("method") != job.get("config_method"):
            raise ExportError("TTA diagnostics method mismatch")
        if document.get("stream") != "val_seen":
            raise ExportError("TTA diagnostics stream is not val_seen")
        if int(document.get("episode_count", -1)) != expected_episodes:
            raise ExportError("TTA diagnostics episode_count mismatch")
        adapter = document.get("adapter")
        if source_job and not sampled_discrete_source:
            if adapter is not None or result.get("adapter_diagnostics") is not None:
                raise ExportError("Source job contains unexpected adapter evidence")
            return path
        if not isinstance(adapter, dict):
            raise ExportError("TTA diagnostics adapter is missing")
        if int(adapter.get("episodes", -1)) != expected_episodes:
            raise ExportError("TTA adapter episode count mismatch")
        if sampled_discrete_source:
            updates = adapter.get("updates")
            if (not isinstance(updates, int) or isinstance(updates, bool)
                    or updates != 0):
                raise ExportError("sampled Source adapter performed updates")
            drift = adapter.get("relative_param_drift")
            if (not isinstance(drift, (int, float)) or isinstance(drift, bool)
                    or drift != 0.0):
                raise ExportError("sampled Source adapter changed parameters")
            if adapter.get("control") != "matched_source_sampling_no_update":
                raise ExportError("sampled Source adapter control marker mismatch")
            action_steps = adapter.get("action_steps")
            if (not isinstance(action_steps, int) or isinstance(action_steps, bool)
                    or action_steps < 0):
                raise ExportError("sampled Source adapter action_steps is invalid")
        if canonical(adapter) != canonical(result.get("adapter_diagnostics")):
            raise ExportError("metrics and TTA adapter diagnostics disagree")
        return path

    def _canonical_episode_ids(self, setting, expected_count):
        order = self.canonical_orders[setting]["document"]
        records = order.get("episodes")
        if (not isinstance(records, list) or len(records) != expected_count
                or int(order.get("episode_count", -1)) != expected_count):
            raise ExportError(
                "canonical episode-order count mismatch for {}".format(setting)
            )
        identifiers = []
        for ordinal, record in enumerate(records):
            identifier = record.get("episode_id") if isinstance(record, dict) else None
            if not isinstance(identifier, str) or not identifier:
                raise ExportError(
                    "canonical episode-order ID {} is invalid for {}".format(
                        ordinal, setting
                    )
                )
            identifiers.append(identifier)
        if len(set(identifiers)) != len(identifiers):
            raise ExportError(
                "canonical episode-order IDs are not unique for {}".format(setting)
            )
        return identifiers

    def _validate_continuous_per_episode_stats(
            self, job, artifact_records, expected_count):
        expected_name = CONTINUOUS_PER_EPISODE_ARTIFACT[job["setting"]]
        stats_names = sorted(
            name for name in artifact_records
            if Path(name).name.startswith("stats_ep_")
        )
        if stats_names != [expected_name]:
            raise ExportError(
                "formal continuous result must contain exactly {}".format(
                    expected_name
                )
            )
        artifact = artifact_records[expected_name]
        result_root = Path(job["result_root"]).absolute()
        expected_path = (result_root / expected_name).absolute()
        if artifact["path"] != expected_path:
            raise ExportError(
                "continuous per-episode stats path mismatch for {}".format(
                    job["run_tag"]
                )
            )
        raw = read_json(expected_path)
        expected_ids = self._canonical_episode_ids(job["setting"], expected_count)
        if not isinstance(raw, dict):
            raise ExportError("continuous per-episode stats are not an object")
        actual_ids = list(raw)
        if actual_ids != expected_ids:
            raise ExportError(
                "continuous per-episode stats ID count/order mismatch for {}"
                .format(job["run_tag"])
            )
        expected_metrics = set(CONTINUOUS_PER_EPISODE_METRICS)
        for identifier in expected_ids:
            metrics = raw[identifier]
            if not isinstance(metrics, dict) or set(metrics) != expected_metrics:
                raise ExportError(
                    "continuous episode {} metric schema mismatch".format(
                        identifier
                    )
                )
            for name in CONTINUOUS_PER_EPISODE_METRICS:
                value = metrics[name]
                if (isinstance(value, bool)
                        or not isinstance(value, (int, float))):
                    raise ExportError(
                        "continuous episode {} metric {} is not numeric".format(
                            identifier, name
                        )
                    )
                try:
                    finite = math.isfinite(value)
                except (OverflowError, TypeError, ValueError):
                    finite = False
                if not finite:
                    raise ExportError(
                        "continuous episode {} metric {} is not finite".format(
                            identifier, name
                        )
                    )
        return {
            "name": expected_name,
            "path": expected_path,
            "size": artifact["size"],
            "sha256": artifact["sha256"],
            "episode_ids": expected_ids,
            "metrics": raw,
            "order_sha256": self.canonical_orders[job["setting"]][
                "document"
            ]["order_sha256"],
            "episode_order_manifest_sha256": self.canonical_orders[
                job["setting"]
            ]["sha256"],
        }

    def _export_continuous_per_episode_stats(self, job, source, destination):
        if sha256(source["path"]) != source["sha256"]:
            raise ExportError(
                "continuous per-episode stats changed after formal validation"
            )
        document = {
            "schema": "navtta.vln_per_episode_metrics.v1",
            "setting": job["setting"],
            "split": "val_seen",
            "run_tag": job["run_tag"],
            "episode_count": len(source["episode_ids"]),
            "order_sha256": source["order_sha256"],
            "episode_order_manifest_sha256": source[
                "episode_order_manifest_sha256"
            ],
            "source_artifact": {
                "name": source["name"],
                "path": str(source["path"]),
                "size": source["size"],
                "sha256": source["sha256"],
            },
            "episodes": [{
                "ordinal": ordinal,
                "episode_id": identifier,
                "metrics": source["metrics"][identifier],
            } for ordinal, identifier in enumerate(source["episode_ids"])],
        }
        ensure_safe_export_path(destination, self.working_dir)
        write_json(destination, document)
        self._record_copy(source["path"], destination, str(source["path"]))
        metadata = {
            "setting": job["setting"],
            "run_tag": job["run_tag"],
            "stage": job["stage"],
            "split": "val_seen",
            "exported_path": destination.relative_to(
                self.working_dir
            ).as_posix(),
            "exported_size": destination.stat().st_size,
            "exported_sha256": sha256(destination),
            "source_name": source["name"],
            "source_path": str(source["path"]),
            "source_size": source["size"],
            "source_sha256": source["sha256"],
            "episode_count": len(source["episode_ids"]),
            "order_sha256": source["order_sha256"],
            "episode_order_manifest_sha256": source[
                "episode_order_manifest_sha256"
            ],
        }
        self.per_episode_exports.append(metadata)
        return metadata

    def _validate_formal_manifest(self, manifest_path, document, job,
                                  stage_manifest, diagnostics_path,
                                  expected_exit_code=0):
        required = {
            "run_id", "task", "benchmark", "model", "method", "run_tag",
            "source_setting", "seed", "git_commit", "config",
            "config_overrides", "checkpoint", "auxiliary_checkpoints",
            "dataset", "pinned_manifests", "hardware", "started_at",
            "completed_at", "status", "exit_code",
            "immutable_identity_sha256",
        }
        if expected_exit_code == 0:
            required.add("result_artifacts")
        missing = sorted(required.difference(document))
        if missing:
            raise ExportError("formal manifest misses fields: {}".format(
                ", ".join(missing)
            ))
        expected = {
            "task": "vln",
            "model": job.get("model"),
            "method": job.get("config_method"),
            "run_tag": job.get("run_tag"),
            "seed": 0,
            "git_commit": stage_manifest.get("git_commit"),
            "status": "completed" if expected_exit_code == 0 else "failed",
            "exit_code": expected_exit_code,
        }
        for key, value in expected.items():
            if document.get(key) != value:
                raise ExportError("formal manifest {} mismatch for {}".format(
                    key, job["run_tag"]
                ))
        identity = document.get("immutable_identity_sha256")
        if not isinstance(identity, str) or HEX_SHA256.fullmatch(identity) is None:
            raise ExportError("formal immutable identity SHA256 is invalid")
        if immutable_identity_sha256(document) != identity:
            raise ExportError("formal immutable identity SHA256 mismatch")

        source_parts = str(document.get("source_setting", "")).split(":")
        if (len(source_parts) != 4 or source_parts[0] != job.get("setting")
                or source_parts[1] != "val_seen"
                or source_parts[3] != job.get("config_method")):
            raise ExportError("formal source_setting mismatch")
        expected_run_id = "{}-{}-val_seen-{}".format(
            job["run_tag"], job["setting"], source_parts[2]
        )
        if document.get("run_id") != expected_run_id:
            raise ExportError("formal run_id mismatch")
        if Path(manifest_path).parent.name != expected_run_id:
            raise ExportError("formal manifest directory/run_id mismatch")
        config = document.get("config")
        # Every search job, including matched Source controls, is launched
        # with --tta-config.  run_source_eval.sh therefore pins that exact job
        # configuration rather than its ordinary Source runner anchor.
        expected_config = job.get("config_path")
        if config != expected_config:
            raise ExportError("formal config reference mismatch")
        overrides = document.get("config_overrides")
        if (not isinstance(overrides, list) or not overrides
                or not all(isinstance(item, str) for item in overrides)):
            raise ExportError("formal config_overrides are invalid")

        manifest_dir = Path(manifest_path).parent
        checkpoint_path = self._validate_file_metadata(
            document["checkpoint"], "sha256", "formal checkpoint", manifest_dir
        )
        expected_checkpoint = self.canonical_assets[
            SETTING_CHECKPOINT_ASSET[job["setting"]]
        ]["sha256"]
        if document["checkpoint"]["sha256"] != expected_checkpoint:
            raise ExportError("formal checkpoint is not the canonical setting asset")
        auxiliary = document.get("auxiliary_checkpoints")
        if not isinstance(auxiliary, list) or not auxiliary:
            raise ExportError("formal auxiliary checkpoints are missing")
        names = set()
        auxiliary_identity = {}
        for index, item in enumerate(auxiliary):
            name = item.get("name") if isinstance(item, dict) else None
            if not isinstance(name, str) or not name or name in names:
                raise ExportError("formal auxiliary checkpoint names are invalid")
            names.add(name)
            self._validate_file_metadata(
                item, "sha256", "formal auxiliary {}".format(name), manifest_dir
            )
            auxiliary_identity[name] = item["sha256"]
        expected_auxiliary = self._expected_auxiliary_sha256(job["setting"])
        if auxiliary_identity != expected_auxiliary:
            raise ExportError("formal auxiliary assets are not canonical for {}".format(
                job["setting"]
            ))
        dataset_path = self._validate_file_metadata(
            document["dataset"], "index_sha256", "formal dataset", manifest_dir
        )
        del dataset_path
        dataset = document["dataset"]
        for key in ("version", "stream_order_sha256", "stream_content_sha256"):
            if not isinstance(dataset.get(key), str) or not dataset[key]:
                raise ExportError("formal dataset {} is invalid".format(key))
        if dataset.get("version") != document.get("benchmark"):
            raise ExportError("formal dataset version/benchmark mismatch")

        pinned = document.get("pinned_manifests")
        if not isinstance(pinned, dict) or set(pinned) != {
                "assets", "environment", "episode_order"}:
            raise ExportError("formal pinned manifests are incomplete")
        pinned_paths = {}
        for name, metadata in sorted(pinned.items()):
            pinned_paths[name] = self._validate_file_metadata(
                metadata, "sha256", "formal pinned {}".format(name), manifest_dir
            )
        canonical_pinned = {
            "assets": (
                self.asset_manifest_path.absolute(),
                sha256(self.asset_manifest_path),
            ),
            "environment": (
                self.environment_manifest_path.absolute(),
                sha256(self.environment_manifest_path),
            ),
            "episode_order": (
                self.canonical_orders[job["setting"]]["path"],
                self.canonical_orders[job["setting"]]["sha256"],
            ),
        }
        for name, (expected_path, expected_hash) in canonical_pinned.items():
            if (pinned_paths[name].absolute() != expected_path
                    or pinned[name].get("sha256") != expected_hash):
                raise ExportError("formal pinned {} is not canonical".format(name))
        assets_document = read_json(pinned_paths["assets"])
        if (assets_document.get("schema_version") != 1
                or not isinstance(assets_document.get("assets"), list)):
            raise ExportError("formal asset manifest schema is invalid")
        environment_document = read_json(pinned_paths["environment"])
        if (environment_document.get("schema_version") != 1
                or not isinstance(environment_document.get("environments"), dict)):
            raise ExportError("formal environment manifest schema is invalid")
        order = read_json(pinned_paths["episode_order"])
        canonical_order = self.canonical_orders[job["setting"]]["document"]
        if canonical(order) != canonical(canonical_order):
            raise ExportError("formal episode-order content is not canonical")
        expected_count = int(self.spec["setting_episode_counts"][job["setting"]])
        if (order.get("schema") != "navtta.episode_order.v1"
                or order.get("split") != "val_seen"
                or int(order.get("episode_count", -1)) != expected_count
                or order.get("benchmark") != document.get("benchmark")):
            raise ExportError("formal episode-order manifest identity mismatch")
        if (order.get("order_sha256") != dataset["stream_order_sha256"]
                or order.get("dataset", {}).get("sha256")
                != dataset["stream_content_sha256"]):
            raise ExportError("formal dataset/order hashes disagree")
        order_dataset = Path(order.get("dataset", {}).get("path", ""))
        if not order_dataset.is_absolute():
            order_dataset = REPO_ROOT / order_dataset
        manifest_dataset = Path(dataset["path"])
        if order_dataset.absolute() != manifest_dataset.absolute():
            raise ExportError("formal dataset/order paths disagree")
        if (dataset.get("index_sha256") != canonical_order["dataset"]["sha256"]
                or dataset.get("stream_content_sha256")
                != canonical_order["dataset"]["sha256"]
                or dataset.get("stream_order_sha256")
                != canonical_order["order_sha256"]
                or document.get("benchmark") != canonical_order["benchmark"]):
            raise ExportError("formal dataset/order identity is not canonical")

        hardware = document.get("hardware")
        hardware_fields = {
            "hostname", "platform", "python", "cuda_visible_devices", "torch",
            "torch_cuda", "cuda_available", "cudnn", "gpu_name", "gpu_capability",
        }
        if (not isinstance(hardware, dict)
                or not hardware_fields.issubset(hardware)
                or hardware.get("cuda_available") is not True
                or not isinstance(hardware.get("gpu_name"), str)
                or not hardware["gpu_name"]
                or not isinstance(hardware.get("gpu_capability"), list)):
            raise ExportError("formal hardware metadata is incomplete")
        try:
            started = datetime.fromisoformat(document["started_at"])
            completed = datetime.fromisoformat(document["completed_at"])
        except (TypeError, ValueError):
            raise ExportError("formal timestamps are invalid")
        if completed < started:
            raise ExportError("formal completion precedes start")

        artifacts = document.get("result_artifacts")
        if expected_exit_code == 0 and (
                not isinstance(artifacts, list) or not artifacts):
            raise ExportError("formal result_artifacts are missing")
        if artifacts is None:
            artifacts = []
        if not isinstance(artifacts, list):
            raise ExportError("formal result_artifacts are not a list")
        artifact_names = set()
        artifact_records = {}
        result_root = Path(job["result_root"]).absolute()
        diagnostics_artifact = False
        for item in artifacts:
            name = item.get("name") if isinstance(item, dict) else None
            if not isinstance(name, str) or not name or name in artifact_names:
                raise ExportError("formal result artifact names are invalid")
            artifact_names.add(name)
            path = self._validate_file_metadata(
                item, "sha256", "formal result artifact {}".format(name),
                manifest_dir,
            ).absolute()
            if not path_is_within(path, result_root):
                raise ExportError("formal result artifact escapes result_root")
            reject_symlink(path, "formal result artifact", stop=result_root)
            if name != path.relative_to(result_root).as_posix():
                raise ExportError("formal result artifact name/path mismatch")
            artifact_records[name] = {
                "path": path,
                "size": item["size"],
                "sha256": item["sha256"],
            }
            if path == diagnostics_path:
                diagnostics_artifact = True
        if diagnostics_path is not None and not diagnostics_artifact:
            raise ExportError("formal artifacts omit tta_diagnostics.json")
        if (expected_exit_code == 0 and job.get("family") == "continuous"
                and job.get("stage") in PER_EPISODE_STAGES):
            self.per_episode_source_by_run_tag[job["run_tag"]] = (
                self._validate_continuous_per_episode_stats(
                    job, artifact_records,
                    int(self.spec["setting_episode_counts"][job["setting"]]),
                )
            )
        identity = {
            "benchmark": document["benchmark"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": document["checkpoint"]["sha256"],
            "auxiliary_sha256": auxiliary_identity,
            "dataset_version": dataset["version"],
            "dataset_index_sha256": dataset["index_sha256"],
            "stream_order_sha256": dataset["stream_order_sha256"],
            "stream_content_sha256": dataset["stream_content_sha256"],
            "episode_order_manifest_sha256": pinned["episode_order"]["sha256"],
        }
        prior = self.formal_identity_by_setting.get(job["setting"])
        if prior is not None and canonical(prior) != canonical(identity):
            raise ExportError(
                "full-stream formal asset identity differs across methods for {}"
                .format(job["setting"])
            )
        self.formal_identity_by_setting[job["setting"]] = identity
        return identity

    def _copy_json(self, source, destination, source_label):
        require_regular_file(source, source_label)
        ensure_safe_export_path(destination, self.working_dir)
        write_json(destination, read_json(source))
        self._record_copy(source, destination, source_label)

    def _copy_text(self, source, destination, source_label):
        require_regular_file(source, source_label)
        ensure_safe_export_path(destination, self.working_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Path(source).open("r", encoding="utf-8", errors="replace") as input_stream:
            text = input_stream.read().replace("\r\n", "\n").replace("\r", "\n")
        with destination.open("w", encoding="utf-8", newline="\n") as output_stream:
            output_stream.write(text)
        self._record_copy(source, destination, source_label)

    def _copy_console(self, source, destination, source_label):
        require_regular_file(source, source_label)
        ensure_safe_export_path(destination, self.working_dir)
        compact_console(source, destination, source_label)
        self._record_copy(source, destination, source_label)

    def _formal_manifest_for(self, job, successful, full_stream):
        matches = self.formal_by_run_tag.get(job["run_tag"], [])
        if len(matches) > 1:
            raise ExportError("multiple formal manifests for {}".format(
                job["run_tag"]
            ))
        if not full_stream:
            if matches:
                raise ExportError(
                    "non-formal prefix job unexpectedly has a run manifest: {}"
                    .format(job["run_tag"])
                )
            return None
        if not matches:
            if successful:
                raise ExportError("missing formal run manifest for {}".format(
                    job["run_tag"]
                ))
            return None
        path, document = matches[0]
        if document.get("run_tag") != job["run_tag"]:
            raise ExportError("formal run-tag mismatch for {}".format(job["run_tag"]))
        if document.get("method") != job.get("config_method"):
            raise ExportError("formal method mismatch for {}".format(job["run_tag"]))
        if document.get("model") != job.get("model"):
            raise ExportError("formal model mismatch for {}".format(job["run_tag"]))
        expected_status = "completed" if successful else None
        if expected_status and document.get("status") != expected_status:
            raise ExportError("successful job has non-completed manifest: {}".format(
                job["run_tag"]
            ))
        if successful and int(document.get("exit_code", -1)) != 0:
            raise ExportError("successful job has nonzero formal exit code: {}".format(
                job["run_tag"]
            ))
        return path, document

    def _validate_scheduler_reparse(self, job, result):
        try:
            reparsed = search.parse_metrics(job, self.spec)
        except Exception as error:
            raise ExportError(
                "scheduler parse_metrics rejected {}: {}".format(
                    job.get("run_tag"), error
                )
            )
        fields = (
            "metrics", "expected_episodes", "diagnostics_path",
            "diagnostics_sha256", "adapter_diagnostics",
            "requires_posthoc_late_collapse_check",
        )
        for key in fields:
            try:
                matches = canonical(reparsed.get(key)) == canonical(result.get(key))
            except (TypeError, ValueError) as error:
                raise ExportError(
                    "invalid reparsed {} for {}: {}".format(
                        key, job.get("run_tag"), error
                    )
                )
            if not matches:
                raise ExportError(
                    "scheduler parse_metrics disagrees for {}: {}".format(
                        job.get("run_tag"), key
                    )
                )
        return reparsed

    def _copy_attempts(self, job_dir, export_job_dir, source_prefix):
        attempts_root = job_dir / "attempts"
        if not attempts_root.is_dir():
            return
        reject_symlink(attempts_root, "job attempts directory", stop=job_dir)
        for attempt in sorted(path for path in attempts_root.iterdir()
                              if path.is_dir()):
            reject_symlink(attempt, "job attempt directory", stop=job_dir)
            target = export_job_dir / "attempts" / safe_component(attempt.name)
            for name in ("console.log", "exitcode", "worker_state.json"):
                source = attempt / name
                if not source.is_file():
                    continue
                label = "{}/attempts/{}/{}".format(
                    source_prefix, attempt.name, name
                )
                destination = target / name
                if name == "console.log":
                    self._copy_console(source, destination, label)
                elif name.endswith(".json"):
                    self._copy_json(source, destination, label)
                else:
                    self._copy_text(source, destination, label)
            archived_manifest = attempt / "formal_run_manifest" / "manifest.json"
            if archived_manifest.is_file():
                self._copy_json(
                    archived_manifest, target / "formal_run_manifest.json",
                    "{}/attempts/{}/formal_run_manifest/manifest.json".format(
                        source_prefix, attempt.name
                    ),
                )

    def _copy_job(self, method, stage, job_dir, error_tags, csv_row,
                  stage_manifest):
        job_path = job_dir / "job.json"
        job = read_json(job_path)
        run_tag = job.get("run_tag")
        if not isinstance(run_tag, str):
            raise ExportError("job has no run_tag: {}".format(job_path))
        if run_tag in self.seen_run_tags:
            raise ExportError("duplicate run_tag across campaign: {}".format(run_tag))
        self.seen_run_tags.add(run_tag)
        export_job = (
            self.working_dir / "methods" / method / "stages" / stage / "jobs"
            / safe_component(job_dir.name)
        )
        source_prefix = "hparam_search/{}/{}/stages/{}/jobs/{}".format(
            method, self.batch_id, stage, job_dir.name
        )
        self._copy_json(job_path, export_job / "job.json",
                        source_prefix + "/job.json")
        parameters = job_dir / "parameters.json"
        if not parameters.is_file():
            raise ExportError("missing parameters for {}".format(run_tag))
        parameter_document = read_json(parameters)
        self._validate_job_config(job, job_dir, parameter_document)
        self._copy_json(parameters, export_job / "parameters.json",
                        source_prefix + "/parameters.json")
        console = job_dir / "console.log"
        if console.is_file():
            self._copy_console(console, export_job / "console.log",
                               source_prefix + "/console.log")
        for name in ("exitcode", "worker_state.json"):
            source = job_dir / name
            if not source.is_file():
                continue
            if name.endswith(".json"):
                self._copy_json(source, export_job / name,
                                source_prefix + "/" + name)
            else:
                self._copy_text(source, export_job / name,
                                source_prefix + "/" + name)

        metrics_path = job_dir / "metrics.json"
        successful = metrics_path.is_file() and run_tag not in error_tags
        exit_path = job_dir / "exitcode"
        if not exit_path.is_file():
            raise ExportError("terminal job lacks exitcode: {}".format(run_tag))
        try:
            exit_code = int(exit_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as error:
            raise ExportError("invalid exitcode for {}: {}".format(run_tag, error))
        if successful and exit_code != 0:
            raise ExportError("validated job has nonzero exitcode: {}".format(run_tag))
        if not successful and exit_code == 0 and run_tag not in error_tags:
            raise ExportError("zero-exit job lacks validated metrics: {}".format(run_tag))
        worker_state_path = job_dir / "worker_state.json"
        if not worker_state_path.is_file():
            raise ExportError("terminal job lacks worker_state.json: {}".format(run_tag))
        worker_state = read_json(worker_state_path)
        if (worker_state.get("status") != "finished"
                or int(worker_state.get("exit_code", -1)) != exit_code):
            raise ExportError("worker state/exitcode mismatch for {}".format(run_tag))
        result = None
        diagnostics_path = None
        if successful:
            result = read_json(metrics_path)
            self._validate_scheduler_reparse(job, result)
            expected_episodes = (
                int(job["episodes"]) if int(job["episodes"]) > 0
                else int(self.spec["setting_episode_counts"][job["setting"]])
            )
            self._validate_metrics_identity(
                job, result, csv_row, expected_episodes
            )
            diagnostics_path = self._validate_diagnostics(
                job, result, expected_episodes
            )
            self._copy_json(metrics_path, export_job / "metrics.json",
                            source_prefix + "/metrics.json")
            if diagnostics_path is not None:
                self._copy_json(
                    diagnostics_path, export_job / "tta_diagnostics.json",
                    "result/{}/tta_diagnostics.json".format(run_tag),
                )

        full_stream = int(job.get("episodes", 0)) == -1
        formal = self._formal_manifest_for(job, successful, full_stream)
        if formal is not None:
            formal_path, formal_document = formal
            self._validate_formal_manifest(
                formal_path, formal_document, job, stage_manifest,
                diagnostics_path, expected_exit_code=exit_code,
            )
            per_episode_source = self.per_episode_source_by_run_tag.get(run_tag)
            if per_episode_source is not None:
                per_episode_metadata = self._export_continuous_per_episode_stats(
                    job, per_episode_source,
                    export_job / "per_episode_metrics.json",
                )
                if result is not None:
                    result["compact_per_episode_metrics"] = per_episode_metadata
            self._copy_json(
                formal_path, export_job / "formal_run_manifest.json",
                "runs/{}/manifest.json".format(formal_path.parent.name),
            )
        self._copy_attempts(job_dir, export_job, source_prefix)
        return job, result

    def _validate_stage(self, method, stage, stage_dir):
        reject_symlink(
            stage_dir, "{} {} stage".format(method, stage),
            stop=self.method_roots[method],
        )
        manifest_path = stage_dir / "stage_manifest.json"
        summary_path = stage_dir / "SUMMARY.json"
        if not manifest_path.is_file() or not summary_path.is_file():
            raise ExportError("{} {} lacks manifest or SUMMARY".format(method, stage))
        require_regular_file(manifest_path, "stage manifest", stop=stage_dir)
        require_regular_file(summary_path, "stage SUMMARY", stop=stage_dir)
        manifest = read_json(manifest_path)
        summary = read_json(summary_path)
        if manifest.get("schema") != "navtta.vln_tta_search_stage.v1":
            raise ExportError("unsupported search stage manifest schema")
        if summary.get("schema") != "navtta.vln_tta_stage_summary.v1":
            raise ExportError("unsupported search stage SUMMARY schema")
        for key, expected in (
            ("batch_id", self.batch_id), ("method", method), ("stage", stage),
        ):
            if manifest.get(key) != expected:
                raise ExportError("{} {} manifest {} mismatch".format(
                    method, stage, key
                ))
        expected_stage_episodes = search.stage_episode_count(stage, self.spec)
        if int(manifest.get("episodes", -999)) != expected_stage_episodes:
            raise ExportError("{} {} manifest episode count mismatch".format(
                method, stage
            ))
        if set(manifest.get("settings", [])) != set(self.settings):
            raise ExportError("{} {} does not cover all eight settings".format(
                method, stage
            ))
        job_dirs = sorted(path.parent for path in (stage_dir / "jobs").glob(
            "*/job.json"
        ))
        for job_dir in job_dirs:
            reject_symlink(job_dir, "job directory", stop=stage_dir)
        if int(manifest.get("job_count", -1)) != len(job_dirs):
            raise ExportError("{} {} job count mismatch".format(method, stage))
        planned = int(summary.get("planned", -1))
        validated = int(summary.get("validated", -1))
        errors = summary.get("errors", [])
        if (planned != len(job_dirs) or validated + len(errors) != planned
                or not summary.get("terminal")):
            raise ExportError("{} {} is not terminal and internally consistent".format(
                method, stage
            ))
        if stage in STRICT_STAGES and (errors or not summary.get("complete")):
            raise ExportError("strict stage {} {} is incomplete".format(method, stage))
        if stage in ("stage1", "stage2", "stage3"):
            if not summary.get("promotion_ready"):
                raise ExportError("screening stage {} {} was not promoted".format(
                    method, stage
                ))
            if not (stage_dir / "promotion_preview.json").is_file():
                raise ExportError("screening stage lacks promotion preview")
        metrics_path = stage_dir / "metrics.csv"
        if not metrics_path.is_file():
            raise ExportError("{} {} lacks metrics.csv".format(method, stage))
        require_regular_file(metrics_path, "stage metrics.csv", stop=stage_dir)
        metric_rows = load_csv(metrics_path)
        if len(metric_rows) != validated:
            raise ExportError("{} {} metrics row count mismatch".format(method, stage))
        if set(row.get("setting") for row in metric_rows) != set(self.settings):
            raise ExportError("{} {} metrics omit a setting".format(method, stage))
        if stage == "final":
            expected = int(self.spec["protocol"]["full_val_seen_finalists_per_setting"])
            for setting in self.settings:
                count = sum(row.get("setting") == setting for row in metric_rows)
                if count != expected:
                    raise ExportError("final stage {} {} has {} candidates, expected {}".format(
                        method, setting, count, expected
                    ))
        error_tags = {item.get("run_tag") for item in errors}
        csv_by_tag = {}
        for row in metric_rows:
            run_tag = row.get("run_tag")
            if not run_tag or run_tag in csv_by_tag:
                raise ExportError("{} {} has duplicate/empty metrics.csv run tags".format(
                    method, stage
                ))
            csv_by_tag[run_tag] = row
        results = []
        jobs = []
        ordinals = set()
        base_run_tags = set()
        config_paths = set()
        config_identities = set()
        for job_dir in job_dirs:
            raw_job_path = require_regular_file(
                job_dir / "job.json", "job manifest", stop=job_dir
            )
            raw_job = read_json(raw_job_path)
            expected_job_identity = {
                "setting": raw_job.get("setting"),
                "model": search.SETTING_MODEL.get(raw_job.get("setting")),
                "search_method": method,
                "stage": stage,
                "episodes": expected_stage_episodes,
            }
            if raw_job.get("setting") not in self.settings:
                raise ExportError("{} {} job has an unknown setting".format(
                    method, stage
                ))
            for key, expected in expected_job_identity.items():
                if raw_job.get(key) != expected:
                    raise ExportError("{} {} job {} mismatch".format(
                        method, stage, key
                    ))
            expected_family = (
                "continuous" if raw_job["setting"] in search.CONTINUOUS_SETTINGS
                else "discrete"
            )
            if raw_job.get("family") != expected_family:
                raise ExportError("{} {} job family mismatch".format(method, stage))
            expected_benchmark = (
                "reverie" if raw_job["setting"].endswith("reverie")
                else "r2r-ce" if raw_job["setting"].endswith("r2r-ce") else "r2r"
            )
            if raw_job.get("benchmark") != expected_benchmark:
                raise ExportError("{} {} job benchmark mismatch".format(method, stage))
            expected_config_method = (
                "source" if stage in ("controls", "final_controls") else method
            )
            if raw_job.get("config_method") != expected_config_method:
                raise ExportError("{} {} job config_method mismatch".format(
                    method, stage
                ))
            ordinal = raw_job.get("ordinal")
            if (isinstance(ordinal, bool) or not isinstance(ordinal, int)
                    or ordinal in ordinals):
                raise ExportError("{} {} has duplicate/invalid ordinals".format(
                    method, stage
                ))
            ordinals.add(ordinal)
            base_tag = raw_job.get("base_run_tag")
            if not isinstance(base_tag, str) or not base_tag or base_tag in base_run_tags:
                raise ExportError("{} {} has duplicate/invalid base run tags".format(
                    method, stage
                ))
            base_run_tags.add(base_tag)
            config_path = raw_job.get("config_path")
            if (not isinstance(config_path, str) or not config_path
                    or config_path in config_paths):
                raise ExportError("{} {} has duplicate/invalid config paths".format(
                    method, stage
                ))
            config_paths.add(config_path)
            config_identity = canonical({
                "setting": raw_job.get("setting"),
                "config_method": raw_job.get("config_method"),
                "parameters": raw_job.get("parameters"),
                "order_seed": raw_job.get("order_seed"),
            })
            if config_identity in config_identities:
                raise ExportError("{} {} has duplicate configurations".format(
                    method, stage
                ))
            config_identities.add(config_identity)
            csv_row = csv_by_tag.get(raw_job.get("run_tag"))
            if raw_job.get("run_tag") not in error_tags and csv_row is None:
                raise ExportError("validated job is absent from metrics.csv")
            job, result = self._copy_job(
                method, stage, job_dir, error_tags, csv_row or {}, manifest
            )
            jobs.append(job)
            if result is not None:
                results.append(result)
        if ordinals != set(range(len(job_dirs))):
            raise ExportError("{} {} ordinals are not contiguous".format(method, stage))
        if set(csv_by_tag) != {item["run_tag"] for item in results}:
            raise ExportError("{} {} metrics.csv/result tag mismatch".format(
                method, stage
            ))
        if len(results) != validated:
            raise ExportError("{} {} validated job evidence mismatch".format(
                method, stage
            ))
        for error in errors:
            self.rejection_rows.append({
                "method": method,
                "setting": "",
                "stage": stage,
                "run_tag": error.get("run_tag", ""),
                "reason": error.get("error", "exit_code_{}".format(
                    error.get("exit_code", "unknown")
                )),
                "kind": "invalid_or_failed_job",
            })
        self.stage_rows.append({
            "method": method,
            "stage": stage,
            "planned": planned,
            "validated": validated,
            "failed": len(errors),
            "complete": bool(summary.get("complete")),
            "promotion_ready": summary.get("promotion_ready"),
            "episodes": manifest.get("episodes"),
            "git_commit": manifest.get("git_commit"),
            "spec_sha256": manifest.get("spec_sha256"),
        })
        return {
            "manifest": manifest,
            "summary": summary,
            "jobs": jobs,
            "results": results,
        }

    def _copy_stage_files(self, method, stage, stage_dir):
        target = self.working_dir / "methods" / method / "stages" / stage
        prefix = "hparam_search/{}/{}/stages/{}".format(
            method, self.batch_id, stage
        )
        for name in STAGE_FILES:
            source = stage_dir / name
            if not source.is_file():
                continue
            destination = target / name
            if name.endswith(".json"):
                self._copy_json(source, destination, prefix + "/" + name)
            else:
                self._copy_text(source, destination, prefix + "/" + name)
        scheduler_log = stage_dir / "scheduler.log"
        if scheduler_log.is_file():
            self._copy_console(
                scheduler_log, target / "scheduler.log",
                prefix + "/scheduler.log",
            )

    def _optional_documents(self, method, method_root):
        found = []
        candidates = [method_root / name for name in OPTIONAL_METHOD_FILES]
        for stage in ("final", "final_controls", "orders"):
            candidates.extend(
                method_root / "stages" / stage / name
                for name in OPTIONAL_METHOD_FILES
            )
        seen = set()
        for source in candidates:
            if not source.is_file() or source.resolve() in seen:
                continue
            seen.add(source.resolve())
            relative = source.relative_to(method_root)
            destination = (
                self.working_dir / "methods" / method / "official"
                / Path(*[safe_component(part) for part in relative.parts])
            )
            self._copy_json(
                source, destination,
                "hparam_search/{}/{}/{}".format(
                    method, self.batch_id, relative.as_posix()
                ),
            )
            found.append({
                "name": source.name,
                "source_relative": relative.as_posix(),
                "document": read_json(source),
            })
        return found

    def _validate_official_documents(self, method, method_root, documents,
                                     git_commit, spec_sha256):
        by_relative = {item["source_relative"]: item["document"]
                       for item in documents}
        selection = by_relative.get("FINAL_SELECTION.json")
        frozen = by_relative.get("FROZEN_HPARAMETERS.json")
        if not self.allow_legacy_prefix_controls and (
                selection is None or frozen is None):
            raise ExportError(
                "current-protocol export requires root FINAL_SELECTION.json and "
                "FROZEN_HPARAMETERS.json"
            )
        if selection is not None:
            expected = {
                "schema": "navtta.vln_tta_final_selection.v1",
                "method": method,
                "split": "val_seen",
                "finalist_stage": "final",
                "matched_source_stage": "final_controls",
                "git_commit": git_commit,
                "spec_sha256": spec_sha256,
            }
            for key, value in expected.items():
                if selection.get(key) != value:
                    raise ExportError("FINAL_SELECTION {} mismatch".format(key))
            if set(selection.get("settings", {})) != set(self.settings):
                raise ExportError("FINAL_SELECTION setting coverage mismatch")
            for setting, value in selection["settings"].items():
                if (not isinstance(value, dict)
                        or not isinstance(value.get("winner_run_tag"), str)
                        or not isinstance(value.get("source_run_tag"), str)
                        or not isinstance(value.get("frozen_parameters"), dict)
                        or not isinstance(value.get("winner_metrics"), dict)
                        or not isinstance(value.get("source_metrics"), dict)
                        or not isinstance(value.get("selection"), dict)):
                    raise ExportError(
                        "FINAL_SELECTION entry is incomplete for {}".format(setting)
                    )
        if frozen is not None:
            expected = {
                "schema": "navtta.vln_tta_frozen_hparams.v1",
                "method": method,
                "split": "val_seen",
                "generated_from": "FINAL_SELECTION.json",
                "git_commit": git_commit,
                "spec_sha256": spec_sha256,
            }
            for key, value in expected.items():
                if frozen.get(key) != value:
                    raise ExportError("FROZEN_HPARAMETERS {} mismatch".format(key))
            if set(frozen.get("settings", {})) != set(self.settings):
                raise ExportError("FROZEN_HPARAMETERS setting coverage mismatch")

    def _promotion_rejections(self, method, method_root):
        for stage in ("stage1", "stage2", "stage3"):
            path = method_root / "stages" / stage / "promotion_preview.json"
            if not path.is_file():
                continue
            document = read_json(path)
            for setting_record in document.get("settings", []):
                setting = setting_record.get("setting", "")
                for item in setting_record.get("ranked", []):
                    if item.get("eligible", True):
                        continue
                    for reason in item.get("reasons", ["ineligible"]):
                        self.rejection_rows.append({
                            "method": method,
                            "setting": setting,
                            "stage": stage,
                            "run_tag": item.get("run_tag", ""),
                            "reason": reason,
                            "kind": "selection_guard",
                        })

    def _select_settings(self, method, stages, optional_documents):
        source_stage = "final_controls" if "final_controls" in stages else "controls"
        source_results = stages[source_stage]["results"]
        final_results = stages["final"]["results"]
        source_by_setting = {}
        for result in source_results:
            source_by_setting.setdefault(result["setting"], []).append(result)
        final_by_setting = {}
        for result in final_results:
            final_by_setting.setdefault(result["setting"], []).append(result)
        official_selections = [
            item["document"] for item in optional_documents
            if item["name"] == "FINAL_SELECTION.json"
        ]
        official_frozen = [
            item["document"] for item in optional_documents
            if item["name"] == "FROZEN_HPARAMETERS.json"
        ]
        output = {}
        for setting in self.settings:
            sources = source_by_setting.get(setting, [])
            if len(sources) != 1:
                raise ExportError("{} {} has {} matched Source controls".format(
                    method, setting, len(sources)
                ))
            finalists = final_by_setting.get(setting, [])
            winner_list, ranking = search.rank_and_promote(
                finalists, sources[0], method, setting, 1, self.spec,
                force_anchor=False,
            )
            winner = winner_list[0]
            for document in official_selections:
                official_tag = _extract_selected_run_tag(document, setting)
                if (document.get("schema") == "navtta.vln_tta_final_selection.v1"
                        and official_tag is None):
                    raise ExportError("official final selection lacks winner run tag")
                if official_tag is not None and official_tag != winner["run_tag"]:
                    raise ExportError(
                        "official and derived final selections disagree for {} {}: "
                        "{} != {}".format(
                            method, setting, official_tag, winner["run_tag"]
                        )
                    )
                if document.get("schema") == "navtta.vln_tta_final_selection.v1":
                    entry = document["settings"][setting]
                    if entry.get("source_run_tag") != sources[0]["run_tag"]:
                        raise ExportError(
                            "official matched Source run tag disagrees for {} {}"
                            .format(method, setting)
                        )
                    comparisons = (
                        ("frozen_parameters", winner["parameters"]),
                        ("winner_metrics", winner["metrics"]),
                        ("source_metrics", sources[0]["metrics"]),
                        ("selection", ranking),
                    )
                    for key, expected in comparisons:
                        if canonical(entry.get(key)) != canonical(expected):
                            raise ExportError(
                                "official final selection {} disagrees for {} {}"
                                .format(key, method, setting)
                            )
            for document in official_frozen:
                parameters = _extract_frozen_parameters(document, setting)
                if (document.get("schema") == "navtta.vln_tta_frozen_hparams.v1"
                        and parameters is None):
                    raise ExportError("official frozen parameters are missing")
                if parameters is not None and canonical(parameters) != canonical(
                        winner["parameters"]):
                    raise ExportError(
                        "official frozen parameters disagree for {} {}".format(
                            method, setting
                        )
                    )
            source = sources[0]
            full_expected = int(self.spec["setting_episode_counts"][setting])
            matched_full_stream = int(source.get("expected_episodes", -1)) == full_expected
            selection = search._selection_spec(setting, self.spec)
            primary = selection["primary"].upper()
            metrics = winner["metrics"]
            source_metrics = source["metrics"]
            diagnostics = winner.get("adapter_diagnostics") or {}
            winner_per_episode = winner.get("compact_per_episode_metrics")
            source_per_episode = source.get("compact_per_episode_metrics")
            asset_identity = self.formal_identity_by_setting.get(setting)
            if asset_identity is None and not self.allow_legacy_prefix_controls:
                raise ExportError("missing canonical formal asset identity for {}".format(
                    setting
                ))
            row = {
                "method": method,
                "supervision": self.spec["methods"][method]["supervision"],
                "setting": setting,
                "model": search.SETTING_MODEL[setting],
                "benchmark": winner.get("benchmark", ""),
                "selected_run_tag": winner["run_tag"],
                "matched_source_run_tag": source["run_tag"],
                "matched_source_stage": source_stage,
                "matched_source_full_stream": matched_full_stream,
                "primary_metric": primary,
                "primary_value": metrics[primary],
                "matched_source_primary": source_metrics.get(primary),
                "primary_delta": (
                    metrics[primary] - source_metrics[primary]
                    if primary in source_metrics else None
                ),
                "sr": metrics.get("SR"),
                "matched_source_sr": source_metrics.get("SR"),
                "sr_delta": (
                    metrics["SR"] - source_metrics["SR"]
                    if "SR" in metrics and "SR" in source_metrics else None
                ),
                "relative_param_drift": diagnostics.get("relative_param_drift"),
                "updates": diagnostics.get("updates"),
                "feedback_observed_episodes": diagnostics.get(
                    "feedback_observed_episodes"
                ),
                "feedback_observation_rate": diagnostics.get(
                    "feedback_observation_rate"
                ),
                "queries": diagnostics.get("queries"),
                "query_rate": diagnostics.get("query_rate"),
                "parameters_json": canonical(winner["parameters"]),
                "checkpoint_sha256": (
                    asset_identity.get("checkpoint_sha256")
                    if asset_identity else None
                ),
                "auxiliary_sha256_json": (
                    canonical(asset_identity.get("auxiliary_sha256"))
                    if asset_identity else None
                ),
                "dataset_version": (
                    asset_identity.get("dataset_version") if asset_identity else None
                ),
                "dataset_index_sha256": (
                    asset_identity.get("dataset_index_sha256")
                    if asset_identity else None
                ),
                "stream_order_sha256": (
                    asset_identity.get("stream_order_sha256")
                    if asset_identity else None
                ),
                "stream_content_sha256": (
                    asset_identity.get("stream_content_sha256")
                    if asset_identity else None
                ),
                "episode_order_manifest_sha256": (
                    asset_identity.get("episode_order_manifest_sha256")
                    if asset_identity else None
                ),
                "per_episode_metrics_path": (
                    winner_per_episode.get("exported_path")
                    if winner_per_episode else None
                ),
                "per_episode_metrics_sha256": (
                    winner_per_episode.get("exported_sha256")
                    if winner_per_episode else None
                ),
                "per_episode_metrics_source_sha256": (
                    winner_per_episode.get("source_sha256")
                    if winner_per_episode else None
                ),
                "matched_source_per_episode_metrics_path": (
                    source_per_episode.get("exported_path")
                    if source_per_episode else None
                ),
                "matched_source_per_episode_metrics_sha256": (
                    source_per_episode.get("exported_sha256")
                    if source_per_episode else None
                ),
                "matched_source_per_episode_metrics_source_sha256": (
                    source_per_episode.get("source_sha256")
                    if source_per_episode else None
                ),
            }
            all_metrics = sorted(set(metrics).union(source_metrics))
            for metric in all_metrics:
                row["tta_{}".format(metric.lower())] = metrics.get(metric)
                row["source_{}".format(metric.lower())] = source_metrics.get(metric)
            self.selected_rows.append(row)
            frozen_row = {
                "method": method,
                "setting": setting,
                "selected_run_tag": winner["run_tag"],
                "parameters_json": canonical(winner["parameters"]),
            }
            for key, value in sorted(winner["parameters"].items()):
                frozen_row["param_{}".format(key)] = (
                    canonical(value) if isinstance(value, (dict, list)) else value
                )
            self.frozen_rows.append(frozen_row)
            for item in ranking["ranked"]:
                if item["eligible"]:
                    continue
                for reason in item["reasons"]:
                    self.rejection_rows.append({
                        "method": method,
                        "setting": setting,
                        "stage": "final",
                        "run_tag": item["run_tag"],
                        "reason": reason,
                        "kind": "selection_guard",
                    })
            output[setting] = {
                "winner": {
                    "run_tag": winner["run_tag"],
                    "parameters": winner["parameters"],
                    "metrics": metrics,
                    "adapter_diagnostics": diagnostics,
                    "per_episode_metrics": winner_per_episode,
                },
                "matched_source": {
                    "run_tag": source["run_tag"],
                    "stage": source_stage,
                    "full_stream": matched_full_stream,
                    "expected_episodes": source.get("expected_episodes"),
                    "metrics": source_metrics,
                    "per_episode_metrics": source_per_episode,
                },
                "selection": ranking,
                "asset_identity": asset_identity,
            }
        return output

    def _export_method(self, method):
        method_root = self.method_roots[method]
        if not method_root.is_dir():
            raise ExportError("missing method batch root: {}".format(method_root))
        reject_symlink(method_root, "{} batch root".format(method))
        stages_root = method_root / "stages"
        workflow = tuple(search.workflow_stages(method, include_orders=False))
        # Batches created immediately before the full-val Source-control stage
        # was introduced remain exportable, but every selected row is marked
        # ``matched_source_full_stream=false``.  FINAL_SELECTION is likewise
        # optional because it can be derived deterministically from finalists.
        required = (
            workflow if not self.allow_legacy_prefix_controls
            else tuple(stage for stage in workflow if stage != "final_controls")
        )
        missing = [stage for stage in required
                   if not (stages_root / stage).is_dir()]
        if missing:
            raise ExportError("{} missing stages: {}".format(
                method, ", ".join(missing)
            ))
        discovered = sorted(
            path.name for path in stages_root.iterdir()
            if path.is_dir() and (path / "stage_manifest.json").is_file()
        )
        ordered = []
        for stage in workflow:
            if stage == "final_controls" and stage not in discovered:
                continue
            ordered.append(stage)
        for stage in discovered:
            if stage not in ordered:
                ordered.append(stage)
        stages = {}
        for stage in ordered:
            stage_dir = stages_root / stage
            stages[stage] = self._validate_stage(method, stage, stage_dir)
            self._copy_stage_files(method, stage, stage_dir)
        self._promotion_rejections(method, method_root)
        optional_documents = self._optional_documents(method, method_root)
        commits = sorted({
            value["manifest"].get("git_commit") for value in stages.values()
        })
        spec_hashes = sorted({
            value["manifest"].get("spec_sha256") for value in stages.values()
        })
        if len(commits) != 1 or len(spec_hashes) != 1:
            raise ExportError("{} mixes commits or search specifications".format(method))
        self._validate_official_documents(
            method, method_root, optional_documents, commits[0], spec_hashes[0]
        )
        settings = self._select_settings(method, stages, optional_documents)
        return {
            "git_commit": commits[0],
            "spec_sha256": spec_hashes[0],
            "supervision": self.spec["methods"][method]["supervision"],
            "stages": [{
                "stage": stage,
                "planned": stages[stage]["summary"]["planned"],
                "validated": stages[stage]["summary"]["validated"],
                "errors": stages[stage]["summary"]["errors"],
                "complete": stages[stage]["summary"]["complete"],
            } for stage in ordered],
            "optional_official_artifacts": [{
                "name": item["name"],
                "source_relative": item["source_relative"],
            } for item in optional_documents],
            "settings": settings,
        }

    def _write_report_tables(self, report):
        write_json(self.working_dir / "report.json", report)
        selected_fields = [
            "method", "supervision", "setting", "model", "benchmark",
            "selected_run_tag", "matched_source_run_tag", "matched_source_stage",
            "matched_source_full_stream", "primary_metric", "primary_value",
            "matched_source_primary", "primary_delta", "sr", "matched_source_sr",
            "sr_delta", "relative_param_drift", "updates",
            "feedback_observed_episodes", "feedback_observation_rate", "queries",
            "query_rate", "parameters_json",
            "checkpoint_sha256", "auxiliary_sha256_json", "dataset_version",
            "dataset_index_sha256", "stream_order_sha256",
            "stream_content_sha256", "episode_order_manifest_sha256",
            "per_episode_metrics_path", "per_episode_metrics_sha256",
            "per_episode_metrics_source_sha256",
            "matched_source_per_episode_metrics_path",
            "matched_source_per_episode_metrics_sha256",
            "matched_source_per_episode_metrics_source_sha256",
        ]
        extra_metric_fields = sorted(set().union(*(
            set(row).difference(selected_fields) for row in self.selected_rows
        ))) if self.selected_rows else []
        write_csv(
            self.working_dir / "selected_results.csv",
            sorted(self.selected_rows, key=lambda row: (
                METHODS.index(row["method"]), self.settings.index(row["setting"])
            )),
            selected_fields + extra_metric_fields,
        )
        frozen_fixed = ["method", "setting", "selected_run_tag", "parameters_json"]
        frozen_extra = sorted(set().union(*(
            set(row).difference(frozen_fixed) for row in self.frozen_rows
        ))) if self.frozen_rows else []
        write_csv(
            self.working_dir / "frozen_parameters.csv",
            sorted(self.frozen_rows, key=lambda row: (
                METHODS.index(row["method"]), self.settings.index(row["setting"])
            )),
            frozen_fixed + frozen_extra,
        )
        write_csv(
            self.working_dir / "stage_summary.csv",
            sorted(self.stage_rows, key=lambda row: (
                METHODS.index(row["method"]), row["stage"]
            )),
            ["method", "stage", "planned", "validated", "failed", "complete",
             "promotion_ready", "episodes", "git_commit", "spec_sha256"],
        )
        unique_rejections = {
            (row["method"], row["setting"], row["stage"], row["run_tag"],
             row["reason"], row["kind"]): row
            for row in self.rejection_rows
        }
        write_csv(
            self.working_dir / "rejections.csv",
            [unique_rejections[key] for key in sorted(unique_rejections)],
            ["method", "setting", "stage", "run_tag", "reason", "kind"],
        )

    def _write_inventory(self):
        self.source_inventory.sort(key=lambda row: row["exported"])
        write_csv(
            self.working_dir / "source_inventory.csv", self.source_inventory,
            ["exported", "exported_bytes", "exported_sha256", "source",
             "source_bytes", "source_sha256"],
        )
        files = sorted(
            path for path in self.working_dir.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        sums = self.working_dir / "SHA256SUMS"
        with sums.open("w", encoding="utf-8", newline="\n") as stream:
            for path in files:
                ensure_safe_export_path(path, self.working_dir)
                stream.write("{}  {}\n".format(
                    sha256(path), path.relative_to(self.working_dir).as_posix()
                ))

    def export(self, overwrite=False):
        if set(self.method_roots) != set(METHODS):
            raise ExportError("method roots must contain exactly {}".format(
                ", ".join(METHODS)
            ))
        if tuple(self.settings) != SETTINGS:
            raise ExportError("search specification does not contain canonical settings")
        export_root, output_dir = self._validated_output_paths()
        export_root.mkdir(parents=True, exist_ok=True)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.export_root = export_root
        self.output_dir = output_dir
        if self.output_dir.exists() and not overwrite:
            raise ExportError("output exists; pass --overwrite: {}".format(
                self.output_dir
            ))
        temporary = Path(tempfile.mkdtemp(
            prefix=".{}-".format(safe_component(self.output_dir.name)),
            dir=str(self.output_dir.parent),
        ))
        self.working_dir = temporary
        try:
            methods = {}
            for method in METHODS:
                methods[method] = self._export_method(method)
            commits = sorted({value["git_commit"] for value in methods.values()})
            spec_hashes = sorted({value["spec_sha256"] for value in methods.values()})
            if len(commits) != 1 or len(spec_hashes) != 1:
                raise ExportError("campaign mixes commits or search specifications")
            actual_spec_sha = sha256(self.spec_path)
            if spec_hashes != [actual_spec_sha]:
                raise ExportError("campaign spec SHA256 does not match supplied spec")
            full_matched_sources = all(
                setting["matched_source"]["full_stream"]
                for method in methods.values()
                for setting in method["settings"].values()
            )
            qualifications = []
            if not full_matched_sources:
                qualifications.append(
                    "full-val final_controls are absent; gains use the canonical "
                    "screening prefix Source and are not formal matched comparisons"
                )
            report = {
                "schema": "navtta.vln_tta_hparam_compact_export.v1",
                "batch_id": self.batch_id,
                "git_commit": commits[0],
                "search_spec": {
                    "schema": self.spec["schema"],
                    "experiment_id": self.spec.get("experiment_id"),
                    "sha256": actual_spec_sha,
                    "split": self.spec["split"],
                    "screening_episodes": self.spec["screening_episodes"],
                    "full_val_seen_finalists_per_setting": self.spec["protocol"][
                        "full_val_seen_finalists_per_setting"
                    ],
                },
                "validation": {
                    "methods": list(METHODS),
                    "settings": list(self.settings),
                    "method_setting_cells": len(METHODS) * len(self.settings),
                    "all_required_stages_terminal": True,
                    "full_val_seen_matched_source_controls": full_matched_sources,
                    "formal_matched_comparison_ready": full_matched_sources,
                    "binary_artifacts_copied": False,
                    "trajectory_or_prediction_artifacts_copied": False,
                    "continuous_per_episode_metric_exports": len(
                        self.per_episode_exports
                    ),
                    "selection_policy": "scheduler rank_and_promote, force_anchor=false",
                    "qualifications": qualifications,
                    "canonical_identity_sources": {
                        "asset_manifest_sha256": sha256(
                            self.asset_manifest_path
                        ),
                        "environment_manifest_sha256": sha256(
                            self.environment_manifest_path
                        ),
                        "episode_order_manifest_sha256": {
                            setting: value["sha256"]
                            for setting, value in sorted(
                                self.canonical_orders.items()
                            )
                        },
                    },
                },
                "methods": methods,
                "per_episode_metrics": sorted(
                    self.per_episode_exports,
                    key=lambda item: item["exported_path"],
                ),
            }
            self._write_report_tables(report)
            self._write_inventory()
            for path in self.working_dir.rglob("*"):
                if path.is_file():
                    ensure_safe_export_path(path, self.working_dir)
            if self.output_dir.exists():
                # Revalidate immediately before the only destructive action to
                # narrow the opportunity for an output-path symlink swap.
                self._validated_output_paths()
                shutil.rmtree(str(self.output_dir))
            os.replace(str(self.working_dir), str(self.output_dir))
            return report
        except Exception:
            shutil.rmtree(str(temporary), ignore_errors=True)
            raise


def standard_method_roots(search_root, batch_id):
    batch_id = validate_batch_id(batch_id)
    return {
        method: Path(search_root) / method / batch_id for method in METHODS
    }


def roots_from_common_batch_root(batch_root):
    root = Path(batch_root)
    reject_symlink(root, "common batch root")
    roots = {method: root / method for method in METHODS}
    missing = [method for method, path in roots.items()
               if not (path / "stages").is_dir()]
    if missing:
        raise ExportError(
            "--batch-root must contain METHOD/stages for all methods; missing {}"
            .format(", ".join(missing))
        )
    manifests = []
    for method, path in roots.items():
        smoke = path / "stages" / "smoke" / "stage_manifest.json"
        if not smoke.is_file():
            raise ExportError("cannot infer batch ID from {}".format(smoke))
        require_regular_file(smoke, "smoke stage manifest", stop=root)
        manifests.append(read_json(smoke).get("batch_id"))
    if len(set(manifests)) != 1 or not manifests[0]:
        raise ExportError("common batch root mixes batch IDs")
    return validate_batch_id(manifests[0]), roots


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--batch-id")
    inputs.add_argument(
        "--batch-root",
        help="portable root containing METHOD/stages for all five methods",
    )
    parser.add_argument("--search-root", default=str(DEFAULT_SEARCH_ROOT))
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--spec", default=str(search.SPEC_PATH))
    parser.add_argument("--export-root", default=str(DEFAULT_EXPORT_ROOT))
    parser.add_argument("--output-dir")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-legacy-prefix-controls",
        action="store_true",
        help=(
            "allow a pre-final_controls batch; exported gains are explicitly "
            "marked non-formal"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.batch_root:
        batch_id, method_roots = roots_from_common_batch_root(args.batch_root)
    else:
        batch_id = args.batch_id
        method_roots = standard_method_roots(args.search_root, batch_id)
    export_root = Path(args.export_root)
    output = Path(args.output_dir) if args.output_dir else export_root / batch_id
    try:
        BatchExporter(
            batch_id=batch_id,
            method_roots=method_roots,
            runs_root=args.runs_root,
            output_dir=output,
            spec_path=args.spec,
            export_root=export_root,
            allow_legacy_prefix_controls=args.allow_legacy_prefix_controls,
        ).export(overwrite=args.overwrite)
    except ExportError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
