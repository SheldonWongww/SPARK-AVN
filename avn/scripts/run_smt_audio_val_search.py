#!/usr/bin/env python3
"""Run the frozen SMT+Audio four-method AVN validation search.

Four independent lanes are pinned to four physical GPUs.  Within every lane,
single-source is a strict barrier before multi-source.  Every candidate starts
in a fresh process, and resume reuses only fully revalidated results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, TextIO, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = (
    REPO_ROOT / "avn/experiments/smt_audio_four_method_val_search_v1.json"
)
LOG_ROOT = REPO_ROOT / "avn/results/logs/smt_audio_val_search"
RUN_ROOT = REPO_ROOT / "avn/results/runs"
RUNNER = REPO_ROOT / "avn/scripts/eval_smt_audio.sh"
FINGERPRINT_TOOL = REPO_ROOT / "avn/scripts/fingerprint_episode_stream.py"
MANIFEST_VALIDATOR = REPO_ROOT / "tools/validate_run_manifest.py"
IDEA_COLLECTOR = REPO_ROOT / "avn/scripts/collect_smt_audio_idea_source_stats.sh"
MODEL_CONFIG = (
    "ss_baselines/savi/config/tta_avn/{}/smt_audio_tta_test.yaml"
)
METHODS = ("eam", "feedtta", "atena", "idea")
SOURCE_SETTINGS = ("single_source", "multi_source")
SEED = 0
CANONICAL_EPISODES = 2000
METRICS = (
    "reward",
    "distance_to_goal",
    "normalized_distance_to_goal",
    "success",
    "spl",
    "softspl",
    "na",
    "sna",
    "sws",
)
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

DATASETS = {
    setting: REPO_ROOT / (
        "avn/data/datasets/tta_test/{}/mp3d/v1/val/val.json.gz".format(setting)
    )
    for setting in SOURCE_SETTINGS
}
SOURCE_DATASETS = {
    setting: REPO_ROOT / (
        "avn/data/datasets/train/{}/mp3d/v1/train/train.json.gz".format(setting)
    )
    for setting in SOURCE_SETTINGS
}
CHECKPOINTS = {
    "single_source": REPO_ROOT / "avn/checkpoints/source/smt_audio/single_best_val.pth",
    "multi_source": REPO_ROOT / "avn/checkpoints/source/smt_audio/multi_best_val.pth",
}
IDEA_MANIFESTS = {
    setting: REPO_ROOT / (
        "avn/manifests/idea_source/smt_audio_{}_sample_seed0.json".format(setting)
    )
    for setting in SOURCE_SETTINGS
}
IDEA_STATS = {
    setting: REPO_ROOT / (
        "avn/results/idea_source_statistics/"
        "smt_audio_{}_sample_seed0.json".format(setting)
    )
    for setting in SOURCE_SETTINGS
}
RUNTIME_FILES = (
    Path(__file__).resolve(),
    RUNNER,
    FINGERPRINT_TOOL,
    MANIFEST_VALIDATOR,
    IDEA_COLLECTOR,
    REPO_ROOT / "avn/scripts/build_idea_source_manifest.py",
    REPO_ROOT / "avn/navtta_avn/idea_source.py",
    REPO_ROOT / "core/navtta_core/tta/tta_core.py",
    REPO_ROOT / "core/navtta_core/tta/idea.py",
    REPO_ROOT / "core/navtta_core/tta/fusion.py",
    REPO_ROOT / "avn/baselines/smt_audio/ss_baselines/savi/ppo/ppo_trainer.py",
    REPO_ROOT / "avn/baselines/smt_audio/ss_baselines/savi/config/default.py",
    REPO_ROOT / "avn/baselines/smt_audio/ss_baselines/savi/config/tta_avn/single_source/smt_audio_tta_test.yaml",
    REPO_ROOT / "avn/baselines/smt_audio/ss_baselines/savi/config/tta_avn/multi_source/smt_audio_tta_test.yaml",
    REPO_ROOT / "avn/baselines/smt_audio/ss_baselines/savi/config/tta_avn/single_source/smt_audio_idea_source_stats.yaml",
    REPO_ROOT / "avn/baselines/smt_audio/ss_baselines/savi/config/tta_avn/multi_source/smt_audio_idea_source_stats.yaml",
    *IDEA_MANIFESTS.values(),
)


class UserError(RuntimeError):
    """An expected campaign error that should be shown without a traceback."""


@dataclass(frozen=True)
class Job:
    job_id: int
    method_job_id: int
    setting_job_id: int
    method: str
    source_setting: str
    point_id: str
    point: Mapping[str, object]
    gpu_slot: int
    gpu: str
    run_tag: str

    @property
    def key(self) -> str:
        return "{}:{}:{}".format(self.method, self.source_setting, self.point_id)


@dataclass
class Worker:
    kind: str
    process: subprocess.Popen
    output: TextIO
    attempt_dir: Path
    gpu: str
    job: Optional[Job] = None
    source_setting: str = ""
    previous_manifests: Tuple[Path, ...] = ()


@dataclass(frozen=True)
class Provenance:
    git_commit: str
    tracked_status: str
    spec_sha256: str
    runtime_sha256: str
    dataset_sha256: Mapping[str, str]
    stream_order_sha256: Mapping[str, str]
    stream_content_sha256: Mapping[str, str]
    checkpoint_sha256: Mapping[str, str]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise UserError(message)


def utc_now(compact: bool = False) -> str:
    fmt = "%Y%m%dT%H%M%SZ" if compact else "%Y-%m-%dT%H:%M:%SZ"
    return datetime.now(timezone.utc).strftime(fmt)


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    temporary.write_text(content, encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=str):
        relative = path.relative_to(REPO_ROOT.resolve()).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def run_capture(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        cwd=str(REPO_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise UserError(
            "command failed ({}): {}".format(
                completed.returncode, detail or " ".join(command)
            )
        )
    return completed.stdout.strip()


def positive_int(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return int(value)


def parse_gpus(value: str) -> Tuple[str, ...]:
    values = tuple(value.split(","))
    if len(values) != 4 or any(
        re.fullmatch(r"0|[1-9][0-9]*", item) is None for item in values
    ):
        raise argparse.ArgumentTypeError(
            "expected exactly four comma-separated GPU ids"
        )
    if len(set(values)) != 4:
        raise argparse.ArgumentTypeError("GPU ids must be distinct")
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen 118-job SMT+Audio EAM/FeedTTA/ATENA/IDEA val search."
        )
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3"))
    parser.add_argument("--eam-concurrency", type=positive_int, default=4)
    parser.add_argument("--feedtta-concurrency", type=positive_int, default=4)
    parser.add_argument("--atena-concurrency", type=positive_int, default=3)
    parser.add_argument("--idea-concurrency", type=positive_int, default=2)
    parser.add_argument("--episodes", type=positive_int)
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--smoke-setting",
        choices=SOURCE_SETTINGS,
        default="single_source",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--prepare-idea-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--poll-seconds", type=positive_int, default=2)
    args = parser.parse_args(argv)

    args.spec = args.spec.resolve()
    args.episodes = args.episodes or (20 if args.smoke else CANONICAL_EPISODES)
    if args.episodes > CANONICAL_EPISODES:
        parser.error("--episodes cannot exceed 2000")
    if not args.smoke and args.episodes != CANONICAL_EPISODES:
        parser.error("full search requires exactly 2000 episodes")
    if args.smoke and args.episodes >= CANONICAL_EPISODES:
        parser.error("--smoke requires fewer than 2000 episodes")
    if args.allow_dirty and not args.smoke:
        parser.error("--allow-dirty is permitted only with --smoke")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    if (args.resume or args.status) and not args.batch_id:
        parser.error("--resume/--status requires an explicit --batch-id")
    if not args.batch_id:
        suffix = "smoke-" if args.smoke else ""
        args.batch_id = "smt-four-val-v1-{}{}".format(suffix, utc_now(True))
    if len(args.batch_id) > 80 or SAFE_ID.fullmatch(args.batch_id) is None:
        parser.error("--batch-id must contain at most 80 safe characters")
    return args


def load_spec(path: Path) -> Mapping[str, object]:
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("cannot read experiment spec: {}".format(error)) from error
    require(isinstance(spec, dict), "experiment spec must be an object")
    require(
        spec.get("schema") == "navtta.avn.smt_audio_val_search.v1",
        "unexpected experiment schema",
    )
    require(spec.get("status") == "active", "experiment spec is not active")
    scope = spec.get("scope")
    require(isinstance(scope, dict), "spec scope is missing")
    require(scope.get("task") == "avn", "spec task must be avn")
    require(scope.get("model") == "smt_audio", "spec model must be smt_audio")
    require(tuple(scope.get("methods", ())) == METHODS, "spec method order mismatch")
    require(
        tuple(scope.get("source_settings_in_order", ())) == SOURCE_SETTINGS,
        "spec source-setting order mismatch",
    )
    require(
        spec.get("protocol", {}).get("action_selection") == "sample",
        "AVN campaign must use sampled actions",
    )
    require(
        spec["protocol"].get("method_rng", {}).get("atena_self_head")
        == "forked_torch_rng_state_restored_before_next_action",
        "ATENA self-head RNG isolation contract mismatch",
    )
    require(spec.get("data", {}).get("seed") == SEED, "spec seed must be 0")
    require(
        spec.get("data", {}).get("episodes") == CANONICAL_EPISODES,
        "spec must contain the 2000-episode horizon",
    )
    require(
        spec.get("source_control", {}).get("rerun_in_campaign") is False,
        "this campaign must not schedule extra Source jobs",
    )
    scheduler = spec.get("scheduler", {})
    require(
        scheduler.get("gpu_method_mapping") == {
            "eam": 0, "feedtta": 1, "atena": 2, "idea": 3
        },
        "spec GPU-method mapping mismatch",
    )
    require(
        scheduler.get("hard_caps") == {
            "eam": 10, "feedtta": 10, "atena": 10, "idea": 10
        },
        "spec concurrency caps mismatch",
    )
    feedtta = spec["protocol"]["methods"]["feedtta"]
    require(
        feedtta.get("action_port")
        == "sample_from_policy_and_optimize_executed_action",
        "FeedTTA AVN action-port contract mismatch",
    )
    atena = spec["protocol"]["methods"]["atena"]
    require(
        atena.get("action_port")
        == "sample_from_policy_and_use_executed_action_as_pseudo_expert",
        "ATENA AVN action-port contract mismatch",
    )
    return spec


def method_points(spec: Mapping[str, object], method: str) -> List[Dict[str, object]]:
    search = spec["search"][method]
    if method in ("feedtta", "atena"):
        points = [dict(item) for item in search["candidates"]]
    elif method == "eam":
        points = []
        for index, (lr, interval) in enumerate(itertools.product(
            search["grid"]["lr"], search["grid"]["update_interval"]
        )):
            points.append({
                "id": "e{:02d}".format(index),
                "lr": str(lr),
                "update_interval": int(interval),
                "variant": "boundary_grid",
            })
    elif method == "idea":
        points = []
        for index, (lr, tau) in enumerate(itertools.product(
            search["grid"]["lr"], search["grid"]["tau"]
        )):
            points.append({
                "id": "d{:02d}".format(index),
                "lr": str(lr),
                "tau": str(tau),
                "variant": "prompt_lr_tau_grid",
            })
    else:  # pragma: no cover - guarded by the frozen method list
        raise UserError("unsupported method: {}".format(method))

    expected = int(search["candidate_count_per_source_setting"])
    require(len(points) == expected, "{} candidate-count mismatch".format(method))
    ids = [str(point.get("id", "")) for point in points]
    require(all(ids) and len(set(ids)) == len(ids), "{} point ids are invalid".format(method))
    canonical = [json.dumps(point, sort_keys=True) for point in points]
    require(len(set(canonical)) == len(points), "{} contains duplicate points".format(method))
    return points


def build_jobs(
    spec: Mapping[str, object], args: argparse.Namespace
) -> List[Job]:
    mapping = spec["scheduler"]["gpu_method_mapping"]
    digest = hashlib.sha256(args.batch_id.encode("utf-8")).hexdigest()[:8]
    jobs: List[Job] = []
    global_job_id = 0
    method_job_ids = {method: 0 for method in METHODS}
    for source_setting in SOURCE_SETTINGS:
        for method in METHODS:
            points = method_points(spec, method)
            for setting_job_id, point in enumerate(points):
                selected = (
                    not args.smoke
                    or (
                        source_setting == args.smoke_setting
                        and setting_job_id == 0
                    )
                )
                if selected:
                    point_id = str(point["id"])
                    run_tag = "smtv1-{}-{}-{}-{}".format(
                        digest,
                        method,
                        "single" if source_setting == "single_source" else "multi",
                        point_id,
                    )
                    slot = int(mapping[method])
                    jobs.append(Job(
                        job_id=global_job_id,
                        method_job_id=method_job_ids[method],
                        setting_job_id=setting_job_id,
                        method=method,
                        source_setting=source_setting,
                        point_id=point_id,
                        point=point,
                        gpu_slot=slot,
                        gpu=args.gpus[slot],
                        run_tag=run_tag,
                    ))
                global_job_id += 1
                method_job_ids[method] += 1
    expected = len(METHODS) if args.smoke else 118
    require(len(jobs) == expected, "expanded plan must contain {} jobs".format(expected))
    require(len({job.key for job in jobs}) == len(jobs), "duplicate job identity")
    require(len({job.run_tag for job in jobs}) == len(jobs), "duplicate run tag")
    return jobs


def concurrency(args: argparse.Namespace) -> Mapping[str, int]:
    return {
        "eam": args.eam_concurrency,
        "feedtta": args.feedtta_concurrency,
        "atena": args.atena_concurrency,
        "idea": args.idea_concurrency,
    }


def validate_runtime_limits(spec: Mapping[str, object], args: argparse.Namespace) -> None:
    caps = spec["scheduler"]["hard_caps"]
    for method, value in concurrency(args).items():
        require(
            value <= int(caps[method]),
            "{} concurrency {} exceeds frozen cap {}".format(
                method, value, caps[method]
            ),
        )


def fingerprint_stream(dataset: Path, episodes: int) -> Tuple[str, str]:
    output = run_capture((
        sys.executable,
        str(FINGERPRINT_TOOL),
        "--dataset",
        str(dataset),
        "--seed",
        str(SEED),
        "--episode-count",
        str(episodes),
    ))
    values = output.split()
    require(
        len(values) == 2 and all(SHA256.fullmatch(value) for value in values),
        "episode fingerprint tool returned invalid output",
    )
    return values[0], values[1]


def validate_preflight(
    spec: Mapping[str, object], args: argparse.Namespace
) -> Provenance:
    required = (
        args.spec,
        *RUNTIME_FILES,
        *DATASETS.values(),
        *SOURCE_DATASETS.values(),
        *CHECKPOINTS.values(),
        *IDEA_MANIFESTS.values(),
        *(
            REPO_ROOT / "avn/baselines/smt_audio" / MODEL_CONFIG.format(setting)
            for setting in SOURCE_SETTINGS
        ),
    )
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, "missing required files:\n  " + "\n  ".join(missing))
    for executable in ("git", "bash"):
        require(shutil.which(executable) is not None, "{} is unavailable".format(executable))

    commit = run_capture(("git", "rev-parse", "HEAD"))
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid Git commit")
    tracked_status = run_capture((
        "git", "status", "--porcelain", "--untracked-files=no"
    ))
    if tracked_status and not args.allow_dirty:
        raise UserError("tracked worktree changes detected; commit before launch")

    for path in (args.spec, *RUNTIME_FILES):
        relative = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        tracked = subprocess.run(
            ("git", "cat-file", "-e", "HEAD:{}".format(relative)),
            cwd=str(REPO_ROOT),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tracked.returncode != 0 and not args.allow_dirty:
            raise UserError("runtime dependency is not tracked by HEAD: {}".format(relative))

    dataset_hashes: Dict[str, str] = {}
    order_hashes: Dict[str, str] = {}
    content_hashes: Dict[str, str] = {}
    checkpoint_hashes: Dict[str, str] = {}
    streams = spec["data"]["streams"]
    expected_checkpoints = spec["source_control"]["checkpoint_sha256"]
    idea_source = spec["protocol"]["methods"]["idea"]["source_statistics"]
    for setting in SOURCE_SETTINGS:
        dataset_hashes[setting] = sha256_file(DATASETS[setting])
        require(
            dataset_hashes[setting] == streams[setting]["dataset_index_sha256"],
            "{} val dataset index SHA256 mismatch".format(setting),
        )
        order_hashes[setting], content_hashes[setting] = fingerprint_stream(
            DATASETS[setting], args.episodes
        )
        if args.episodes == CANONICAL_EPISODES:
            require(
                order_hashes[setting] == streams[setting]["stream_order_sha256"],
                "{} stream-order SHA256 mismatch".format(setting),
            )
            require(
                content_hashes[setting] == streams[setting]["stream_content_sha256"],
                "{} stream-content SHA256 mismatch".format(setting),
            )
        checkpoint_hashes[setting] = sha256_file(CHECKPOINTS[setting])
        require(
            checkpoint_hashes[setting] == expected_checkpoints[setting],
            "{} checkpoint SHA256 mismatch".format(setting),
        )
        expected_manifest = idea_source[setting]
        require(
            Path(expected_manifest["manifest"]) == IDEA_MANIFESTS[setting].relative_to(REPO_ROOT),
            "{} IDEA manifest path differs from the frozen spec".format(setting),
        )
        require(
            sha256_file(IDEA_MANIFESTS[setting]) == expected_manifest["manifest_sha256"],
            "{} IDEA source manifest SHA256 mismatch".format(setting),
        )
        manifest, _ = validate_idea_manifest(setting)
        require(
            manifest["dataset"]["bundle_sha256"] == expected_manifest["dataset_bundle_sha256"],
            "{} IDEA source dataset-bundle SHA256 mismatch".format(setting),
        )
        require(
            manifest["episode_order_sha256"] == expected_manifest["episode_order_sha256"],
            "{} IDEA source episode-order SHA256 mismatch".format(setting),
        )

    return Provenance(
        git_commit=commit,
        tracked_status=tracked_status,
        spec_sha256=sha256_file(args.spec),
        runtime_sha256=sha256_files(RUNTIME_FILES),
        dataset_sha256=dataset_hashes,
        stream_order_sha256=order_hashes,
        stream_content_sha256=content_hashes,
        checkpoint_sha256=checkpoint_hashes,
    )


def require_repository_unchanged(provenance: Provenance) -> None:
    require(
        run_capture(("git", "rev-parse", "HEAD")) == provenance.git_commit,
        "repository HEAD changed while campaign was running",
    )
    current = run_capture(("git", "status", "--porcelain", "--untracked-files=no"))
    require(
        current == provenance.tracked_status,
        "tracked worktree changed while campaign was running",
    )


def source_asset_paths(setting: str) -> Tuple[Path, Path]:
    return IDEA_MANIFESTS[setting], IDEA_STATS[setting]


def validate_idea_manifest(setting: str) -> Tuple[Mapping[str, object], str]:
    manifest_path, _ = source_asset_paths(setting)
    require(manifest_path.is_file(), "IDEA source manifest is missing: {}".format(manifest_path))
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from avn.navtta_avn.idea_source import (
            load_source_manifest,
            verify_manifest_assets,
        )
        manifest_sha = sha256_file(manifest_path)
        manifest, _ = load_source_manifest(
            manifest_path,
            manifest_sha,
            model="smt_audio",
            source_setting=setting,
        )
        verify_manifest_assets(
            manifest, SOURCE_DATASETS[setting], CHECKPOINTS[setting]
        )
    except (OSError, ValueError) as error:
        raise UserError(
            "invalid {} IDEA source manifest: {}".format(setting, error)
        ) from error
    finally:
        if sys.path and sys.path[0] == str(REPO_ROOT):
            del sys.path[0]
    return manifest, manifest_sha


def validate_idea_asset(setting: str) -> Mapping[str, str]:
    manifest_path, stats_path = source_asset_paths(setting)
    manifest, manifest_sha = validate_idea_manifest(setting)
    require(stats_path.is_file(), "IDEA source statistics are missing: {}".format(stats_path))

    try:
        artifact = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("invalid IDEA source statistics: {}".format(error)) from error
    require(artifact.get("schema") == "navtta.idea.source_statistics", "IDEA statistics schema mismatch")
    require(int(artifact.get("version", -1)) == 1, "IDEA statistics version mismatch")
    require(int(artifact.get("feature_dim", 0)) > 0, "IDEA feature dimension is invalid")
    require(int(artifact.get("num_layers", 0)) > 0, "IDEA layer count is invalid")
    provenance = artifact.get("provenance")
    require(isinstance(provenance, dict), "IDEA statistics provenance is missing")
    expected = {
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "dataset_sha256": manifest["dataset"]["bundle_sha256"],
        "dataset_version": "v1",
        "split": "train",
        "episode_selection_manifest_sha256": manifest_sha,
        "selection_protocol_sha256": manifest["protocol_sha256"],
        "action_selection": "sample",
        "model": "smt_audio",
        "source_setting": setting,
        "trajectory_count": 128,
    }
    for key, value in expected.items():
        require(
            provenance.get(key) == value,
            "IDEA source statistics provenance mismatch for {}".format(key),
        )
    require(
        provenance.get("selection_protocol") == manifest["selection"],
        "IDEA source selection protocol mismatch",
    )
    for key in ("model_state_sha256", "trajectory_ids_sha256"):
        require(
            SHA256.fullmatch(str(provenance.get(key, ""))) is not None,
            "IDEA statistics {} is invalid".format(key),
        )
    layers = artifact.get("layers")
    require(
        isinstance(layers, list) and len(layers) == int(artifact["num_layers"]),
        "IDEA statistics layers are incomplete",
    )
    width = int(artifact["feature_dim"])
    for index, layer in enumerate(layers):
        require(isinstance(layer, dict), "IDEA layer {} is invalid".format(index))
        require(int(layer.get("count", 0)) > 1, "IDEA layer {} count is invalid".format(index))
        for field in ("sum", "sumsq"):
            values = layer.get(field)
            require(
                isinstance(values, list) and len(values) == width,
                "IDEA layer {} {} width mismatch".format(index, field),
            )
            require(
                all(math.isfinite(float(value)) for value in values),
                "IDEA layer {} {} contains non-finite values".format(index, field),
            )
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "statistics_path": str(stats_path.resolve()),
        "statistics_sha256": sha256_file(stats_path),
        "model_state_sha256": str(provenance["model_state_sha256"]),
    }


def idea_asset_if_ready(setting: str) -> Optional[Mapping[str, str]]:
    manifest_path, stats_path = source_asset_paths(setting)
    if not manifest_path.exists() and not stats_path.exists():
        return None
    if manifest_path.exists() and not stats_path.exists():
        return None
    return validate_idea_asset(setting)


def expected_overrides(
    job: Job,
    episodes: int,
    idea_asset: Optional[Mapping[str, str]] = None,
) -> List[str]:
    values: List[Tuple[str, str]] = [
        ("TEST_EPISODE_COUNT", str(episodes)),
        ("NUM_PROCESSES", "1"),
        ("EVAL.SPLIT", "val"),
        ("EVAL.USE_CKPT_CONFIG", "False"),
        ("EVAL.ACTION_SELECTION", "sample"),
        ("TTA.EPISODIC", "False"),
        ("TTA.STEPS", "1"),
    ]
    point = job.point
    if job.method == "eam":
        values.extend([
            ("TTA.EAM.LR", str(point["lr"])),
            ("TTA.EAM.CONFIDENCE_SCALE", "0.4"),
            ("TTA.EAM.MEMORY_SIZE", "32"),
            ("TTA.EAM.BATCH_SIZE", "8"),
            ("TTA.EAM.UPDATE_INTERVAL", str(point["update_interval"])),
            ("TTA.EAM.PARAM_SCOPE", "module_prefixes"),
            ("TTA.EAM.TRAINABLE_PREFIXES", '["net.smt_state_encoder.transformer","action_distribution"]'),
            ("TTA.EAM.OPTIMIZER", "Adam"),
            ("TTA.EAM.BETA1", "0.9"),
            ("TTA.EAM.BETA2", "0.999"),
            ("TTA.EAM.WEIGHT_DECAY", "0.0"),
            ("TTA.EAM.MAX_GRAD_NORM", "0.0"),
        ])
    elif job.method == "feedtta":
        values.extend([
            ("TTA.FEEDTTA.LR", str(point["lr"])),
            ("TTA.FEEDTTA.GAMMA", str(point["gamma"])),
            ("TTA.FEEDTTA.P", str(point["p"])),
            ("TTA.FEEDTTA.ALPHA", str(point["alpha"])),
            ("TTA.FEEDTTA.SGR_SEED", "0"),
            ("TTA.FEEDTTA.SGR_MODE", "paper_main"),
            ("TTA.FEEDTTA.NORMALIZE_GRADIENT", "False"),
            ("TTA.FEEDTTA.ACTION_SELECTION_PROTOCOL", "sample_from_policy"),
            ("TTA.FEEDTTA.PARAM_SCOPE", "module_prefixes"),
            ("TTA.FEEDTTA.TRAINABLE_PREFIXES", '["net.smt_state_encoder","action_distribution"]'),
            ("TTA.FEEDTTA.OPTIMIZER", "Adam"),
            ("TTA.FEEDTTA.BETA1", "0.9"),
            ("TTA.FEEDTTA.BETA2", "0.999"),
            ("TTA.FEEDTTA.WEIGHT_DECAY", "0.0"),
            ("TTA.FEEDTTA.EPS", "1e-5"),
            ("TTA.FEEDTTA.MAX_GRAD_NORM", "0.0"),
        ])
    elif job.method == "atena":
        values.extend([
            ("TTA.ATENA.PREFLIGHT_APPROVED", "True"),
            ("TTA.ATENA.LR_QUERY", str(point["lr_query"])),
            ("TTA.ATENA.LR_SELF", str(point["lr_self"])),
            ("TTA.ATENA.MIX_LAMBDA", str(point["mix_lambda"])),
            ("TTA.ATENA.QUERY_THRESHOLD", str(point["query_threshold"])),
            ("TTA.ATENA.SELF_LOSS_WEIGHT", str(point["self_loss_weight"])),
            ("TTA.ATENA.PARAM_SCOPE", "all"),
            ("TTA.ATENA.ACTION_SELECTION_PROTOCOL", "sample_from_policy"),
            ("TTA.ATENA.TASK_UPDATE_SCOPE", "replay_reachable_actor_navigation_policy"),
            ("TTA.ATENA.OPTIMIZER", "AdamW"),
            ("TTA.ATENA.BETA1", "0.9"),
            ("TTA.ATENA.BETA2", "0.999"),
            ("TTA.ATENA.WEIGHT_DECAY", "0.01"),
            ("TTA.ATENA.MAX_GRAD_NORM", "0.0"),
        ])
    elif job.method == "idea":
        require(idea_asset is not None, "IDEA source asset is not ready")
        values.extend([
            ("TTA.IDEA.PROMPT_LENGTH", "4"),
            ("TTA.IDEA.K_MAX", "32"),
            ("TTA.IDEA.LAMBDA", "0.4"),
            ("TTA.IDEA.TAU", str(point["tau"])),
            ("TTA.IDEA.FISHER_BETA", "0.1"),
            ("TTA.IDEA.OPT_STEPS", "50"),
            ("TTA.IDEA.LR", str(point["lr"])),
            ("TTA.IDEA.OPTIMIZER", "AdamW"),
            ("TTA.IDEA.WEIGHT_DECAY", "0.0"),
            ("TTA.IDEA.USE_FISHER", "True"),
            ("TTA.IDEA.RIDGE", "1e-4"),
            ("TTA.IDEA.MAX_GRAD_NORM", "0.0"),
            ("TTA.IDEA.PROMPT_LAYERS", "0"),
            ("TTA.IDEA.SOURCE_TRAJECTORIES", "128"),
            ("TTA.IDEA.SOURCE_COLLECTION", "False"),
            ("TTA.IDEA.SOURCE_STATS_PATH", idea_asset["statistics_path"]),
            ("TTA.IDEA.SOURCE_STATS_SHA256", idea_asset["statistics_sha256"]),
            ("TTA.IDEA.SOURCE_EPISODE_MANIFEST", idea_asset["manifest_path"]),
            ("TTA.IDEA.SOURCE_EPISODE_MANIFEST_SHA256", idea_asset["manifest_sha256"]),
        ])
    else:  # pragma: no cover
        raise UserError("unsupported method: {}".format(job.method))
    flattened: List[str] = []
    for key, value in values:
        flattened.extend((key, value))
    return flattened


def job_command(
    job: Job,
    episodes: int,
    idea_asset: Optional[Mapping[str, str]] = None,
) -> List[str]:
    return [
        "bash",
        str(RUNNER),
        job.source_setting,
        job.method,
        str(SEED),
        *expected_overrides(job, episodes, idea_asset),
    ]


def job_environment(job: Job, provenance: Provenance) -> Dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": job.gpu,
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NAVTTA_RUN_TAG": job.run_tag,
        "NAVTTA_STREAM_ORDER_SHA256": provenance.stream_order_sha256[job.source_setting],
        "NAVTTA_STREAM_CONTENT_SHA256": provenance.stream_content_sha256[job.source_setting],
    })
    return environment


def plan_payload(jobs: Sequence[Job]) -> List[Mapping[str, object]]:
    return [
        {
            "job_id": job.job_id,
            "method_job_id": job.method_job_id,
            "setting_job_id": job.setting_job_id,
            "key": job.key,
            "run_tag": job.run_tag,
            "method": job.method,
            "source_setting": job.source_setting,
            "gpu_slot": job.gpu_slot,
            "gpu": job.gpu,
            "point_id": job.point_id,
            "point": dict(job.point),
        }
        for job in jobs
    ]


def plan_csv(jobs: Sequence[Job]) -> str:
    output = io.StringIO(newline="")
    fields = (
        "job_id", "method_job_id", "setting_job_id", "run_tag", "gpu",
        "method", "source_setting", "point_id", "point_json",
    )
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for job in jobs:
        writer.writerow({
            "job_id": job.job_id,
            "method_job_id": job.method_job_id,
            "setting_job_id": job.setting_job_id,
            "run_tag": job.run_tag,
            "gpu": job.gpu,
            "method": job.method,
            "source_setting": job.source_setting,
            "point_id": job.point_id,
            "point_json": json.dumps(job.point, sort_keys=True, separators=(",", ":")),
        })
    return output.getvalue()


def print_plan(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    print("SMT+Audio four-method AVN val search")
    print("  batch: {}".format(args.batch_id))
    print("  action protocol: sample (actual executed action)")
    print("  episodes/job: {}".format(args.episodes))
    print("  jobs: {}".format(len(jobs)))
    limits = concurrency(args)
    for index, method in enumerate(METHODS):
        count = sum(job.method == method for job in jobs)
        print(
            "  GPU {}: {:8s} jobs={} concurrency={}".format(
                args.gpus[index], method, count, limits[method]
            )
        )
    for setting in SOURCE_SETTINGS:
        counts = {
            method: sum(
                job.method == method and job.source_setting == setting
                for job in jobs
            )
            for method in METHODS
        }
        print("  {}: {}".format(setting, json.dumps(counts, sort_keys=True)))


def batch_identity(
    args: argparse.Namespace,
    provenance: Provenance,
    jobs: Sequence[Job],
) -> Mapping[str, object]:
    return {
        "schema": "navtta.avn.smt_audio_val_search.batch.v1",
        "batch_id": args.batch_id,
        "created_at": utc_now(),
        "git_commit": provenance.git_commit,
        "tracked_status": provenance.tracked_status,
        "spec_path": str(args.spec),
        "spec_sha256": provenance.spec_sha256,
        "runtime_sha256": provenance.runtime_sha256,
        "gpus": list(args.gpus),
        "concurrency": dict(concurrency(args)),
        "episodes": args.episodes,
        "seed": SEED,
        "smoke": args.smoke,
        "smoke_setting": args.smoke_setting if args.smoke else None,
        "job_count": len(jobs),
        "dataset_sha256": dict(provenance.dataset_sha256),
        "stream_order_sha256": dict(provenance.stream_order_sha256),
        "stream_content_sha256": dict(provenance.stream_content_sha256),
        "checkpoint_sha256": dict(provenance.checkpoint_sha256),
    }


def initialize_batch(
    args: argparse.Namespace,
    provenance: Provenance,
    jobs: Sequence[Job],
) -> Path:
    batch_dir = LOG_ROOT / args.batch_id
    identity = batch_identity(args, provenance, jobs)
    identity_path = batch_dir / "batch.json"
    if batch_dir.exists() and not args.resume:
        raise UserError("batch already exists; use --resume: {}".format(batch_dir))
    if args.resume:
        require(identity_path.is_file(), "resume batch is missing batch.json")
        try:
            previous = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise UserError("cannot read resume identity: {}".format(error)) from error
        for key in (
            "batch_id", "git_commit", "spec_sha256", "runtime_sha256",
            "gpus", "concurrency", "episodes", "seed", "smoke",
            "smoke_setting", "job_count", "dataset_sha256",
            "stream_order_sha256", "stream_content_sha256",
            "checkpoint_sha256",
        ):
            require(previous.get(key) == identity.get(key), "resume identity mismatch: {}".format(key))
    else:
        batch_dir.mkdir(parents=True)
        atomic_write(identity_path, json_text(identity))
        atomic_write(batch_dir / "plan.json", json_text(plan_payload(jobs)))
        atomic_write(batch_dir / "plan.csv", plan_csv(jobs))
        atomic_write(batch_dir / "spec.json", args.spec.read_text(encoding="utf-8"))
    return batch_dir


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(path: Path, resume: bool) -> Path:
    try:
        path.mkdir()
    except FileExistsError:
        owner_path = path / "owner.json"
        owner = {}
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        same_host = owner.get("host") == socket.gethostname()
        old_pid = owner.get("pid")
        if same_host and isinstance(old_pid, int) and pid_alive(old_pid):
            raise UserError("scheduler lock is owned by live pid {}".format(old_pid))
        require(resume, "stale scheduler lock exists; inspect it and resume the same batch")
        require(same_host, "cannot reclaim a lock created on another host")
        try:
            owner_path.unlink()
            path.rmdir()
            path.mkdir()
        except OSError as error:
            raise UserError("cannot reclaim stale scheduler lock: {}".format(error)) from error
    atomic_write(path / "owner.json", json_text({
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "started_at": utc_now(),
    }))
    return path


def release_lock(path: Path) -> None:
    try:
        (path / "owner.json").unlink()
        path.rmdir()
    except FileNotFoundError:
        pass


def next_attempt_dir(root: Path) -> Path:
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    numbers = [
        int(path.name)
        for path in attempts.iterdir()
        if path.is_dir() and re.fullmatch(r"[0-9]{3}", path.name)
    ]
    result = attempts / "{:03d}".format(max(numbers, default=0) + 1)
    result.mkdir()
    return result


def manifests_for_tag(run_tag: str) -> Tuple[Path, ...]:
    if not RUN_ROOT.is_dir():
        return ()
    return tuple(sorted(RUN_ROOT.glob("*-{}-*/manifest.json".format(run_tag))))


def job_root(batch_dir: Path, job: Job) -> Path:
    return batch_dir / job.method / job.source_setting / "jobs" / job.run_tag


def launch_job(
    batch_dir: Path,
    job: Job,
    args: argparse.Namespace,
    provenance: Provenance,
    idea_asset: Optional[Mapping[str, str]],
) -> Worker:
    root = job_root(batch_dir, job)
    attempt = next_attempt_dir(root)
    command = job_command(job, args.episodes, idea_asset)
    parameters = {
        "job": plan_payload((job,))[0],
        "command": command,
        "started_at": utc_now(),
        "git_commit": provenance.git_commit,
        "spec_sha256": provenance.spec_sha256,
        "checkpoint_sha256": provenance.checkpoint_sha256[job.source_setting],
        "dataset_sha256": provenance.dataset_sha256[job.source_setting],
        "stream_order_sha256": provenance.stream_order_sha256[job.source_setting],
        "stream_content_sha256": provenance.stream_content_sha256[job.source_setting],
        "idea_asset": dict(idea_asset or {}),
    }
    atomic_write(attempt / "parameters.json", json_text(parameters))
    output = (attempt / "console.log").open("w", encoding="utf-8", buffering=1)
    previous = manifests_for_tag(job.run_tag)
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        env=job_environment(job, provenance),
        stdout=output,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    atomic_write(attempt / "pid", "{}\n".format(process.pid))
    atomic_write(root / "latest_attempt", "{}\n".format(attempt.name))
    return Worker(
        kind="job",
        process=process,
        output=output,
        attempt_dir=attempt,
        gpu=job.gpu,
        job=job,
        previous_manifests=previous,
    )


def launch_idea_collection(
    batch_dir: Path, setting: str, gpu: str
) -> Worker:
    root = batch_dir / "idea_source" / setting
    attempt = next_attempt_dir(root)
    command = ["bash", str(IDEA_COLLECTOR), setting, str(SEED)]
    atomic_write(attempt / "parameters.json", json_text({
        "kind": "idea_source_statistics",
        "source_setting": setting,
        "gpu": gpu,
        "command": command,
        "started_at": utc_now(),
    }))
    output = (attempt / "console.log").open("w", encoding="utf-8", buffering=1)
    environment = os.environ.copy()
    environment.update({
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpu,
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        env=environment,
        stdout=output,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    atomic_write(attempt / "pid", "{}\n".format(process.pid))
    atomic_write(root / "latest_attempt", "{}\n".format(attempt.name))
    return Worker(
        kind="idea_source",
        process=process,
        output=output,
        attempt_dir=attempt,
        gpu=gpu,
        source_setting=setting,
    )


def nested_value(document: object, keys: Sequence[str]) -> object:
    value = document
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def override_map(manifest: Mapping[str, object]) -> Mapping[str, str]:
    values = manifest.get("config_overrides")
    require(isinstance(values, list), "manifest config overrides are missing")
    require(len(values) % 2 == 0, "manifest config overrides have odd length")
    output: Dict[str, str] = {}
    for index in range(0, len(values), 2):
        key = str(values[index])
        require(key not in output, "duplicate config override: {}".format(key))
        output[key] = str(values[index + 1])
    return output


def finite_diagnostic(diagnostics: Mapping[str, object], key: str) -> float:
    try:
        value = float(diagnostics[key])
    except (KeyError, TypeError, ValueError) as error:
        raise UserError("missing or nonnumeric diagnostic: {}".format(key)) from error
    require(math.isfinite(value), "non-finite diagnostic: {}".format(key))
    return value


def read_metrics(run_dir: Path, episodes: int) -> Tuple[Mapping[str, float], Path]:
    path = run_dir / "raw/model/tb/val_stats_{}.json".format(SEED)
    try:
        stats = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("cannot read episode metrics: {}".format(error)) from error
    require(isinstance(stats, dict), "episode metrics are not an object")
    require(len(stats) == episodes, "episode metric count mismatch")
    aggregate: Dict[str, float] = {}
    for metric in METRICS:
        values = []
        for episode_id, record in stats.items():
            require(isinstance(record, dict), "invalid episode {}".format(episode_id))
            try:
                value = float(record[metric])
            except (KeyError, TypeError, ValueError) as error:
                raise UserError("episode {} has invalid {}".format(episode_id, metric)) from error
            require(math.isfinite(value), "non-finite episode metric {}".format(metric))
            values.append(value)
        aggregate[metric] = sum(values) / len(values)
    return aggregate, path


def read_diagnostics(run_dir: Path) -> Tuple[Mapping[str, object], Path]:
    path = run_dir / "raw/model/tb/tta_diagnostics_{}.json".format(SEED)
    try:
        diagnostics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("cannot read TTA diagnostics: {}".format(error)) from error
    require(isinstance(diagnostics, dict), "TTA diagnostics are not an object")
    return diagnostics, path


def validate_method_diagnostics(
    job: Job,
    diagnostics: Mapping[str, object],
    episodes: int,
    idea_asset: Optional[Mapping[str, str]],
) -> None:
    require(diagnostics.get("episodes") == episodes, "diagnostic episode count mismatch")
    require(int(diagnostics.get("action_steps", 0)) > 0, "no action steps recorded")
    require(diagnostics.get("task_action_selection") == "sample", "task action protocol mismatch")
    point = job.point
    if job.method == "eam":
        require(diagnostics.get("optimizer") == "Adam", "EAM optimizer mismatch")
        require(diagnostics.get("param_scope") == "module_prefixes", "EAM scope mismatch")
        require(
            diagnostics.get("trainable_prefixes") == [
                "net.smt_state_encoder.transformer", "action_distribution"
            ],
            "EAM trainable-prefix mismatch",
        )
        require(int(diagnostics.get("adapted_parameter_count", 0)) == 1057284, "EAM parameter count mismatch")
        require(math.isclose(finite_diagnostic(diagnostics, "current_lr"), float(point["lr"]), rel_tol=1e-12), "EAM LR mismatch")
        require(int(diagnostics.get("update_interval", -1)) == int(point["update_interval"]), "EAM interval mismatch")
        require(diagnostics.get("replay_unit") == "action_step", "EAM replay-unit mismatch")
        require(diagnostics.get("update_timing") == "after_preupdate_action_selection", "EAM timing mismatch")
    elif job.method == "feedtta":
        require(diagnostics.get("optimizer") == "Adam", "FeedTTA optimizer mismatch")
        require(diagnostics.get("action_selection_protocol") == "sample_from_policy", "FeedTTA action protocol mismatch")
        require(diagnostics.get("action_selection") == "sample", "FeedTTA action selector mismatch")
        require(diagnostics.get("policy_gradient_action") == "task_runner_executed_action", "FeedTTA gradient-action source mismatch")
        require(diagnostics.get("feedback_type") == "binary_episode_success", "FeedTTA feedback mismatch")
        require(diagnostics.get("update_timing") == "once_after_episode_feedback", "FeedTTA timing mismatch")
        require(diagnostics.get("sgr_mode") == "paper_main", "FeedTTA SGR mode mismatch")
        for key, point_key in (
            ("current_lr", "lr"), ("gamma", "gamma"),
            ("reversal_probability", "p"), ("reversal_scale", "alpha"),
        ):
            require(math.isclose(finite_diagnostic(diagnostics, key), float(point[point_key]), rel_tol=1e-12, abs_tol=1e-15), "FeedTTA {} mismatch".format(key))
        require(int(diagnostics.get("feedback_episodes", -1)) == episodes, "FeedTTA feedback count mismatch")
    elif job.method == "atena":
        require(diagnostics.get("optimizer") == "AdamW", "ATENA optimizer mismatch")
        require(diagnostics.get("action_selection") == "sample", "ATENA action mismatch")
        require(diagnostics.get("action_selection_protocol") == "sample_from_policy", "ATENA action protocol mismatch")
        require(diagnostics.get("pseudo_expert_action") == "executed_task_native_sample", "ATENA pseudo-expert mismatch")
        require(diagnostics.get("action_rng_isolated_from_self_head_initialization") is True, "ATENA self-head perturbed the action RNG")
        require(int(diagnostics.get("sampled_action_steps", -1)) == int(diagnostics["action_steps"]), "ATENA sampled-action count mismatch")
        require(diagnostics.get("retains_episode_graph") is False, "ATENA graph retention enabled")
        require(diagnostics.get("gradient_reconstruction") == "exact_step_replay_in_eval_mode", "ATENA replay mismatch")
        require(finite_diagnostic(diagnostics, "max_replay_feature_abs_error") <= 1e-5, "ATENA replay tolerance exceeded")
        for key, point_key in (
            ("lr_query", "lr_query"), ("lr_self", "lr_self"),
            ("mix_lambda", "mix_lambda"),
            ("query_threshold", "query_threshold"),
            ("self_loss_weight", "self_loss_weight"),
        ):
            require(math.isclose(finite_diagnostic(diagnostics, key), float(point[point_key]), rel_tol=1e-12), "ATENA {} mismatch".format(key))
        query_rate = finite_diagnostic(diagnostics, "query_rate")
        require(0.0 <= query_rate <= 1.0, "ATENA query rate is invalid")
        match_rate = finite_diagnostic(diagnostics, "sample_argmax_match_rate")
        require(0.0 <= match_rate <= 1.0, "ATENA sample/argmax match rate is invalid")
    elif job.method == "idea":
        require(idea_asset is not None, "IDEA asset is absent during validation")
        require(diagnostics.get("method") == "idea", "IDEA diagnostic method mismatch")
        require(diagnostics.get("trains_base_policy") is False, "IDEA trained base policy")
        require(diagnostics.get("base_parameter_grads_none") is True, "IDEA left base gradients")
        require(diagnostics.get("source_statistics_mode") == "offline_artifact", "IDEA source mode mismatch")
        require(diagnostics.get("source_statistics_sha256") == idea_asset["statistics_sha256"], "IDEA source-statistics digest mismatch")
        require(int(diagnostics.get("source_statistics_trajectory_count", -1)) == 128, "IDEA source trajectory count mismatch")
        require(int(diagnostics.get("prompt_length", -1)) == 4, "IDEA prompt length mismatch")
        require(int(diagnostics.get("library_capacity", -1)) == 32, "IDEA capacity mismatch")
        require(int(diagnostics.get("opt_steps", -1)) == 50, "IDEA optimization-step mismatch")
        require(math.isclose(finite_diagnostic(diagnostics, "current_lr"), float(point["lr"]), rel_tol=1e-12), "IDEA LR mismatch")
        require(math.isclose(finite_diagnostic(diagnostics, "tau"), float(point["tau"]), rel_tol=1e-12), "IDEA tau mismatch")
        require(int(diagnostics.get("covered_steps", 0)) + int(diagnostics.get("new_domain_steps", 0)) == int(diagnostics["action_steps"]), "IDEA coverage accounting mismatch")

    for name in ("mean_entropy", "last_entropy"):
        finite_diagnostic(diagnostics, name)


def validate_job(
    manifest_path: Path,
    job: Job,
    args: argparse.Namespace,
    provenance: Provenance,
    idea_asset: Optional[Mapping[str, str]],
) -> Mapping[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(MANIFEST_VALIDATOR),
            "--manifest", str(manifest_path),
            "--run-tag", job.run_tag,
            "--model", "smt_audio",
            "--method", job.method,
            "--source-setting", job.source_setting,
            "--seed", str(SEED),
            "--git-commit", provenance.git_commit,
            "--checkpoint-sha256", provenance.checkpoint_sha256[job.source_setting],
            "--stream-order-sha256", provenance.stream_order_sha256[job.source_setting],
            "--stream-content-sha256", provenance.stream_content_sha256[job.source_setting],
            "--require-immutable-identity",
            "--require-result-artifacts",
        ],
        cwd=str(REPO_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(completed.returncode == 0, completed.stderr.strip() or "manifest validation failed")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("invalid run manifest: {}".format(error)) from error
    require(manifest.get("config") == MODEL_CONFIG.format(job.source_setting), "config path mismatch")
    require(nested_value(manifest, ("hardware", "cuda_visible_devices")) == job.gpu, "manifest GPU mismatch")
    require(nested_value(manifest, ("dataset", "index_sha256")) == provenance.dataset_sha256[job.source_setting], "manifest dataset mismatch")
    if job.method == "idea":
        require(idea_asset is not None, "IDEA asset is missing from validation")
        require(
            nested_value(manifest, ("pinned_manifests", "assets", "sha256"))
            == idea_asset["manifest_sha256"],
            "IDEA source manifest was not pinned by the run manifest",
        )
    actual = override_map(manifest)
    expected_list = expected_overrides(job, args.episodes, idea_asset)
    expected = {
        expected_list[index]: expected_list[index + 1]
        for index in range(0, len(expected_list), 2)
    }
    require(actual == expected, "manifest config overrides differ from frozen job")

    metrics, stats_path = read_metrics(manifest_path.parent, args.episodes)
    diagnostics, diagnostics_path = read_diagnostics(manifest_path.parent)
    validate_method_diagnostics(job, diagnostics, args.episodes, idea_asset)
    result = {
        "job": plan_payload((job,))[0],
        "validated_at": utc_now(),
        "git_commit": provenance.git_commit,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "stats": str(stats_path.resolve()),
        "stats_sha256": sha256_file(stats_path),
        "diagnostics": str(diagnostics_path.resolve()),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "metrics": dict(metrics),
        "diagnostic_summary": {
            key: diagnostics[key]
            for key in (
                "updates", "relative_param_drift", "query_rate",
                "sample_argmax_match_rate", "coverage_rate", "library_size",
                "source_statistics_sha256",
            )
            if key in diagnostics
        },
    }
    return result


def discover_manifest(worker: Worker) -> Path:
    require(worker.job is not None, "worker has no TTA job")
    current = manifests_for_tag(worker.job.run_tag)
    created = [path for path in current if path not in set(worker.previous_manifests)]
    require(len(created) == 1, "expected exactly one new run manifest")
    return created[0]


def finalize_job_worker(
    worker: Worker,
    status: int,
    args: argparse.Namespace,
    provenance: Provenance,
    assets: Mapping[str, Mapping[str, str]],
) -> bool:
    require(worker.job is not None, "cannot finalize a non-job worker")
    worker.output.close()
    root = worker.attempt_dir.parents[1]
    validation = "failed"
    composite = status
    if status == 0:
        try:
            manifest = discover_manifest(worker)
            result = validate_job(
                manifest,
                worker.job,
                args,
                provenance,
                assets.get(worker.job.source_setting) if worker.job.method == "idea" else None,
            )
        except UserError as error:
            composite = 90
            with (worker.attempt_dir / "console.log").open("a", encoding="utf-8") as handle:
                handle.write("\n[launcher validation]\n{}\n".format(error))
        else:
            validation = "ok"
            atomic_write(worker.attempt_dir / "result.json", json_text(result))
            atomic_write(root / "result.json", json_text(result))
            atomic_write(root / "manifest.path", str(manifest.resolve()) + "\n")
    atomic_write(worker.attempt_dir / "outcome.json", json_text({
        "runner_exitcode": status,
        "composite_exitcode": composite,
        "validation": validation,
        "finished_at": utc_now(),
    }))
    atomic_write(root / "validation", validation + "\n")
    atomic_write(root / "exitcode", "{}\n".format(composite))
    return composite == 0 and validation == "ok"


def reusable_job(
    batch_dir: Path,
    job: Job,
    args: argparse.Namespace,
    provenance: Provenance,
    assets: Mapping[str, Mapping[str, str]],
) -> Tuple[bool, bool]:
    root = job_root(batch_dir, job)
    if not args.resume or not root.exists():
        return False, False
    try:
        exitcode = int((root / "exitcode").read_text(encoding="utf-8").strip())
        validation = (root / "validation").read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return False, False
    if exitcode != 0 or validation != "ok":
        return False, not args.retry_failed
    try:
        manifest_path = Path((root / "manifest.path").read_text(encoding="utf-8").strip())
        result = validate_job(
            manifest_path,
            job,
            args,
            provenance,
            assets.get(job.source_setting) if job.method == "idea" else None,
        )
    except (OSError, UserError) as error:
        raise UserError(
            "validated resume result is no longer reusable for {}: {}".format(job.key, error)
        ) from error
    atomic_write(root / "result.json", json_text(result))
    return True, False


def mark_blocked(batch_dir: Path, job: Job, reason: str) -> None:
    root = job_root(batch_dir, job)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write(root / "validation", "blocked\n")
    atomic_write(root / "exitcode", "91\n")
    atomic_write(root / "blocked_reason", reason + "\n")


def read_job_result(batch_dir: Path, job: Job) -> Mapping[str, object]:
    path = job_root(batch_dir, job) / "result.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def job_marker_state(batch_dir: Path, job: Job) -> str:
    root = job_root(batch_dir, job)
    try:
        validation = (root / "validation").read_text(encoding="utf-8").strip()
        exitcode = int((root / "exitcode").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return "pending"
    if exitcode == 0 and validation == "ok":
        return "validated"
    if validation == "blocked":
        return "blocked"
    return "failed"


def write_metrics_csv(batch_dir: Path, jobs: Sequence[Job]) -> None:
    fields = (
        "job_id", "method_job_id", "setting_job_id", "run_tag", "gpu",
        "method", "source_setting", "point_id", "variant", "status",
        "validation", "manifest", "point_json",
    ) + METRICS + (
        "relative_param_drift", "query_rate", "sample_argmax_match_rate",
        "coverage_rate", "library_size", "source_statistics_sha256",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for job in jobs:
        root = job_root(batch_dir, job)
        result = read_job_result(batch_dir, job)
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        diagnostics = result.get("diagnostic_summary", {}) if isinstance(result, dict) else {}
        try:
            status = (root / "exitcode").read_text(encoding="utf-8").strip()
        except OSError:
            status = ""
        try:
            validation = (root / "validation").read_text(encoding="utf-8").strip()
        except OSError:
            validation = ""
        row = {
            "job_id": job.job_id,
            "method_job_id": job.method_job_id,
            "setting_job_id": job.setting_job_id,
            "run_tag": job.run_tag,
            "gpu": job.gpu,
            "method": job.method,
            "source_setting": job.source_setting,
            "point_id": job.point_id,
            "variant": job.point.get("variant", ""),
            "status": status,
            "validation": validation,
            "manifest": result.get("manifest", ""),
            "point_json": json.dumps(job.point, sort_keys=True, separators=(",", ":")),
        }
        for metric in METRICS:
            row[metric] = metrics.get(metric, "") if isinstance(metrics, dict) else ""
        for name in fields[-6:]:
            row[name] = diagnostics.get(name, "") if isinstance(diagnostics, dict) else ""
        writer.writerow(row)
    atomic_write(batch_dir / "metrics.csv", output.getvalue())


def selection_report(
    spec: Mapping[str, object], batch_dir: Path, jobs: Sequence[Job]
) -> Mapping[str, object]:
    report: Dict[str, object] = {
        "generated_at": utc_now(),
        "complete_grid_required": True,
        "selection_rule": spec["selection"],
        "source": spec["source_control"],
        "selections": {},
    }
    complete = all(job_marker_state(batch_dir, job) == "validated" for job in jobs)
    report["complete_grid"] = complete
    if not complete:
        report["selection_status"] = "withheld_until_every_job_validates"
        return report
    selections: Dict[str, object] = {}
    source_metrics = spec["source_control"]["metrics"]
    for setting in SOURCE_SETTINGS:
        for method in METHODS:
            ranked = []
            for job in jobs:
                if job.source_setting != setting or job.method != method:
                    continue
                if method == "feedtta" and job.point.get("variant") == "no_sgr_control":
                    continue
                result = read_job_result(batch_dir, job)
                metrics = result["metrics"]
                diagnostics = result.get("diagnostic_summary", {})
                if float(metrics["success"]) < float(source_metrics[setting]["success"]):
                    continue
                if method == "atena" and not (
                    0.05 <= float(diagnostics.get("query_rate", -1.0)) <= 0.95
                ):
                    continue
                drift = float(diagnostics.get("relative_param_drift", 0.0))
                rank = (
                    -float(metrics["spl"]),
                    -float(metrics["success"]),
                    -float(metrics["softspl"]),
                    drift,
                    job.job_id,
                )
                ranked.append((rank, job, result))
            key = "{}:{}".format(method, setting)
            if not ranked:
                selections[key] = {
                    "status": "no_eligible_candidate",
                    "reason": "No predeclared candidate passed the frozen constraints.",
                }
                continue
            ranked.sort(key=lambda item: item[0])
            _, winner, result = ranked[0]
            selections[key] = {
                "status": "validation_selected",
                "job_id": winner.job_id,
                "run_tag": winner.run_tag,
                "point": dict(winner.point),
                "metrics": result["metrics"],
                "diagnostic_summary": result.get("diagnostic_summary", {}),
            }
    report["selection_status"] = "complete"
    report["selections"] = selections
    return report


def snapshot_state(
    batch_dir: Path,
    jobs: Sequence[Job],
    active: Iterable[Worker],
    started_at: str,
) -> Mapping[str, object]:
    active_workers = list(active)
    counts = {state: 0 for state in ("validated", "failed", "blocked", "pending")}
    by_lane: Dict[str, object] = {}
    for job in jobs:
        counts[job_marker_state(batch_dir, job)] += 1
    for method in METHODS:
        lane = {}
        for setting in SOURCE_SETTINGS:
            selected = [
                job for job in jobs
                if job.method == method and job.source_setting == setting
            ]
            lane[setting] = {
                state: sum(job_marker_state(batch_dir, job) == state for job in selected)
                for state in counts
            }
            lane[setting]["expected"] = len(selected)
        by_lane[method] = lane
    return {
        "updated_at": utc_now(),
        "started_at": started_at,
        "expected": len(jobs),
        "counts": counts,
        "active": [
            {
                "kind": worker.kind,
                "pid": worker.process.pid,
                "gpu": worker.gpu,
                "job": worker.job.key if worker.job is not None else None,
                "source_setting": worker.source_setting,
            }
            for worker in active_workers
            if worker.process.poll() is None
        ],
        "lanes": by_lane,
    }


def write_summary(
    spec: Mapping[str, object],
    batch_dir: Path,
    jobs: Sequence[Job],
    started_at: str,
    smoke: bool,
) -> Mapping[str, object]:
    write_metrics_csv(batch_dir, jobs)
    state = snapshot_state(batch_dir, jobs, (), started_at)
    complete = state["counts"]["validated"] == len(jobs)
    summary = {
        "batch_id": batch_dir.name,
        "finished_at": utc_now(),
        "expected": len(jobs),
        "validated": state["counts"]["validated"],
        "failed": state["counts"]["failed"],
        "blocked": state["counts"]["blocked"],
        "missing": state["counts"]["pending"],
        "complete": complete,
        "result_role": "smoke" if smoke else "validation_selected",
    }
    atomic_write(batch_dir / "SUMMARY.json", json_text(summary))
    selection = (
        {
            "generated_at": utc_now(),
            "selection_status": "withheld_for_smoke_run",
            "selections": {},
        }
        if smoke
        else selection_report(spec, batch_dir, jobs)
    )
    atomic_write(batch_dir / "SELECTION.json", json_text(selection))
    return summary


def terminate_workers(workers: Iterable[Worker]) -> None:
    active = [worker for worker in workers if worker.process.poll() is None]
    for worker in active:
        try:
            os.killpg(worker.process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline and any(
        worker.process.poll() is None for worker in active
    ):
        time.sleep(0.5)
    for worker in active:
        if worker.process.poll() is None:
            try:
                os.killpg(worker.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        if not worker.output.closed:
            worker.output.close()


def phase_complete(
    jobs: Sequence[Job], terminal: Set[str], method: str, setting: str
) -> bool:
    """Return whether a lane may advance past a setting.

    A setting omitted by a smoke plan is vacuously complete.  Full plans have
    jobs in both phases, so their single-before-multi barrier is unchanged.
    """
    phase_jobs = [
        job for job in jobs
        if job.method == method and job.source_setting == setting
    ]
    return not phase_jobs or all(job.key in terminal for job in phase_jobs)


def prepare_idea_assets_only(
    batch_dir: Path, args: argparse.Namespace
) -> int:
    gpu = args.gpus[3]
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    global_lock = acquire_lock(LOG_ROOT / ".global_scheduler.lock", args.resume)
    try:
        batch_lock = acquire_lock(batch_dir / ".scheduler.lock", args.resume)
    except BaseException:
        release_lock(global_lock)
        raise
    complete = False
    try:
        for setting in SOURCE_SETTINGS:
            asset = idea_asset_if_ready(setting)
            if asset is not None:
                print("IDEA {} source asset already valid: {}".format(setting, asset["statistics_sha256"]))
                continue
            worker = launch_idea_collection(batch_dir, setting, gpu)
            status = worker.process.wait()
            worker.output.close()
            outcome = {
                "runner_exitcode": status,
                "validation": "failed",
                "finished_at": utc_now(),
            }
            if status != 0:
                atomic_write(worker.attempt_dir / "outcome.json", json_text(outcome))
                raise UserError("IDEA {} source collection failed".format(setting))
            asset = validate_idea_asset(setting)
            outcome["validation"] = "ok"
            atomic_write(worker.attempt_dir / "asset.json", json_text(asset))
            atomic_write(worker.attempt_dir.parents[1] / "asset.json", json_text(asset))
            atomic_write(worker.attempt_dir / "outcome.json", json_text(outcome))
            print("IDEA {} source asset prepared: {}".format(setting, asset["statistics_sha256"]))
        complete = True
        return 0
    finally:
        if complete:
            release_lock(batch_lock)
            release_lock(global_lock)


def run_scheduler(
    spec: Mapping[str, object],
    args: argparse.Namespace,
    jobs: Sequence[Job],
    provenance: Provenance,
    batch_dir: Path,
) -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    global_lock = acquire_lock(LOG_ROOT / ".global_scheduler.lock", args.resume)
    try:
        lock = acquire_lock(batch_dir / ".scheduler.lock", args.resume)
    except BaseException:
        release_lock(global_lock)
        raise
    started_at = utc_now()
    scheduler_log = (batch_dir / "scheduler.log").open(
        "a", encoding="utf-8", buffering=1
    )

    def log(message: str) -> None:
        line = "{} {}".format(utc_now(), message)
        print(line, flush=True)
        scheduler_log.write(line + "\n")

    assets: Dict[str, Mapping[str, str]] = {}
    asset_failures: Dict[str, str] = {}
    pending: Dict[str, Job] = {}
    terminal = set()
    retained_failures = set()

    # Non-IDEA jobs can be classified immediately.  IDEA jobs wait until their
    # source artifact has been validated because that digest is part of their
    # immutable config overrides.
    for setting in SOURCE_SETTINGS:
        try:
            asset = idea_asset_if_ready(setting)
        except UserError as error:
            asset_failures[setting] = str(error)
        else:
            if asset is not None:
                assets[setting] = asset
                atomic_write(
                    batch_dir / "idea_source" / setting / "asset.json",
                    json_text(asset),
                )

    for job in jobs:
        if job.method == "idea" and job.source_setting not in assets:
            pending[job.key] = job
            continue
        reusable, retained_failure = reusable_job(
            batch_dir, job, args, provenance, assets
        )
        if reusable:
            terminal.add(job.key)
            log("reuse validated {}".format(job.key))
        elif retained_failure:
            terminal.add(job.key)
            retained_failures.add(job.key)
            log("retain failed {} (use --retry-failed to retry)".format(job.key))
        else:
            pending[job.key] = job

    active: Dict[int, Worker] = {}
    active_by_method = {method: 0 for method in METHODS}
    collecting_settings = set()
    phases = {method: 0 for method in METHODS}
    interrupted = [None]
    old_handlers = {}

    def request_stop(signum: int, _frame: object) -> None:
        interrupted[0] = signum

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)

    clean_finish = False
    try:
        log("scheduler started batch={} commit={}".format(args.batch_id, provenance.git_commit))
        log("concurrency={}".format(json.dumps(concurrency(args), sort_keys=True)))
        next_repo_check = time.monotonic()
        while len(terminal) < len(jobs) or active:
            if interrupted[0] is not None:
                raise KeyboardInterrupt
            if time.monotonic() >= next_repo_check:
                require_repository_unchanged(provenance)
                next_repo_check = time.monotonic() + 15.0

            # Advance each method independently, but enforce the setting
            # barrier inside that method.
            for method in METHODS:
                while phases[method] < len(SOURCE_SETTINGS):
                    setting = SOURCE_SETTINGS[phases[method]]
                    phase_jobs = [
                        job for job in jobs
                        if job.method == method and job.source_setting == setting
                    ]
                    # Smoke plans intentionally contain only one setting.  An
                    # empty earlier phase is already satisfied; otherwise a
                    # multi-source-only smoke would wait forever at the absent
                    # single-source barrier.
                    if phase_complete(jobs, terminal, method, setting):
                        phases[method] += 1
                        if phase_jobs:
                            log("barrier passed method={} setting={}".format(method, setting))
                        continue
                    break

            # IDEA source-statistics collection is a prerequisite in its own
            # lane and never overlaps IDEA target jobs on the same GPU.
            idea_phase = phases["idea"]
            if idea_phase < len(SOURCE_SETTINGS):
                setting = SOURCE_SETTINGS[idea_phase]
                if setting in asset_failures:
                    reason = asset_failures[setting]
                    for job in jobs:
                        if job.method == "idea" and job.source_setting == setting and job.key not in terminal:
                            mark_blocked(batch_dir, job, reason)
                            terminal.add(job.key)
                            pending.pop(job.key, None)
                    log("IDEA {} blocked: {}".format(setting, reason))
                elif setting not in assets and setting not in collecting_settings:
                    worker = launch_idea_collection(batch_dir, setting, args.gpus[3])
                    active[worker.process.pid] = worker
                    collecting_settings.add(setting)
                    log("launched IDEA source collection setting={} gpu={} pid={}".format(setting, worker.gpu, worker.process.pid))

            # Fill every lane up to its independent cap.
            for method in METHODS:
                if phases[method] >= len(SOURCE_SETTINGS):
                    continue
                setting = SOURCE_SETTINGS[phases[method]]
                if method == "idea" and setting not in assets:
                    continue
                capacity = concurrency(args)[method] - active_by_method[method]
                if capacity <= 0:
                    continue
                candidates = [
                    job for job in jobs
                    if job.method == method
                    and job.source_setting == setting
                    and job.key in pending
                ]
                for job in candidates[:capacity]:
                    require_repository_unchanged(provenance)
                    worker = launch_job(
                        batch_dir,
                        job,
                        args,
                        provenance,
                        assets.get(setting) if method == "idea" else None,
                    )
                    active[worker.process.pid] = worker
                    active_by_method[method] += 1
                    del pending[job.key]
                    log("launched {} gpu={} pid={}".format(job.key, job.gpu, worker.process.pid))

            finished = []
            for pid, worker in list(active.items()):
                status = worker.process.poll()
                if status is None:
                    continue
                finished.append(pid)
                if worker.kind == "idea_source":
                    worker.output.close()
                    collecting_settings.discard(worker.source_setting)
                    if status == 0:
                        try:
                            asset = validate_idea_asset(worker.source_setting)
                        except UserError as error:
                            asset_failures[worker.source_setting] = str(error)
                            atomic_write(worker.attempt_dir / "outcome.json", json_text({
                                "runner_exitcode": status,
                                "validation": "failed",
                                "error": str(error),
                                "finished_at": utc_now(),
                            }))
                            log("IDEA source validation failed setting={}: {}".format(worker.source_setting, error))
                        else:
                            assets[worker.source_setting] = asset
                            atomic_write(worker.attempt_dir / "asset.json", json_text(asset))
                            atomic_write(worker.attempt_dir.parents[1] / "asset.json", json_text(asset))
                            atomic_write(worker.attempt_dir / "outcome.json", json_text({
                                "runner_exitcode": status,
                                "validation": "ok",
                                "finished_at": utc_now(),
                            }))
                            log("IDEA source ready setting={} sha256={}".format(worker.source_setting, asset["statistics_sha256"]))
                            for job in jobs:
                                if (
                                    job.method != "idea"
                                    or job.source_setting != worker.source_setting
                                    or job.key not in pending
                                ):
                                    continue
                                reusable, retained_failure = reusable_job(
                                    batch_dir, job, args, provenance, assets
                                )
                                if reusable:
                                    terminal.add(job.key)
                                    del pending[job.key]
                                    log("reuse validated {}".format(job.key))
                                elif retained_failure:
                                    terminal.add(job.key)
                                    retained_failures.add(job.key)
                                    del pending[job.key]
                                    log(
                                        "retain failed {} (use --retry-failed to retry)"
                                        .format(job.key)
                                    )
                    else:
                        reason = "collector exit code {}".format(status)
                        asset_failures[worker.source_setting] = reason
                        atomic_write(worker.attempt_dir / "outcome.json", json_text({
                            "runner_exitcode": status,
                            "validation": "failed",
                            "finished_at": utc_now(),
                        }))
                        log("IDEA source failed setting={} status={}".format(worker.source_setting, status))
                else:
                    job = worker.job
                    require(job is not None, "TTA worker missing job")
                    ok = finalize_job_worker(worker, status, args, provenance, assets)
                    active_by_method[job.method] -= 1
                    terminal.add(job.key)
                    log("finished {} runner_status={} validated={}".format(job.key, status, int(ok)))
            for pid in finished:
                del active[pid]

            state = snapshot_state(batch_dir, jobs, active.values(), started_at)
            atomic_write(batch_dir / "STATE.json", json_text(state))
            write_metrics_csv(batch_dir, jobs)
            if (len(terminal) < len(jobs) or active) and not finished:
                time.sleep(args.poll_seconds)

        summary = write_summary(spec, batch_dir, jobs, started_at, args.smoke)
        log("summary={}".format(json.dumps(summary, sort_keys=True)))
        clean_finish = True
        return 0 if summary["complete"] else 1
    except KeyboardInterrupt:
        terminate_workers(active.values())
        code = 128 + interrupted[0] if interrupted[0] is not None else 130
        log("scheduler interrupted exitcode={}".format(code))
        return code
    except UserError as error:
        terminate_workers(active.values())
        log("scheduler integrity failure={}".format(error))
        return 2
    except Exception as error:  # pragma: no cover - last-resort evidence path
        terminate_workers(active.values())
        log("scheduler unexpected failure={!r}".format(error))
        return 1
    finally:
        for worker in active.values():
            if not worker.output.closed:
                worker.output.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        scheduler_log.close()
        if clean_finish:
            release_lock(lock)
            release_lock(global_lock)


def print_status(batch_id: str) -> int:
    batch_dir = LOG_ROOT / batch_id
    require(batch_dir.is_dir(), "batch does not exist: {}".format(batch_dir))
    state_path = batch_dir / "STATE.json"
    if state_path.is_file():
        print(state_path.read_text(encoding="utf-8"), end="")
    else:
        print(json_text({"batch_id": batch_id, "status": "created_without_state"}), end="")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.status:
        return print_status(args.batch_id)
    spec = load_spec(args.spec)
    validate_runtime_limits(spec, args)
    jobs = build_jobs(spec, args)
    print_plan(args, jobs)
    if args.dry_run:
        print("Dry run complete; no files or processes were created.")
        return 0

    provenance = validate_preflight(spec, args)
    if args.preflight_only:
        print(
            "Preflight passed: commit={} spec_sha256={} runtime_sha256={}".format(
                provenance.git_commit,
                provenance.spec_sha256,
                provenance.runtime_sha256,
            )
        )
        return 0
    batch_dir = initialize_batch(args, provenance, jobs)
    if args.prepare_idea_only:
        return prepare_idea_assets_only(batch_dir, args)
    return run_scheduler(spec, args, jobs, provenance, batch_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
