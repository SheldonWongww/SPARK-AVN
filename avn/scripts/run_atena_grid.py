#!/usr/bin/env python3
"""Run the official-code-aligned ATENA grid on canonical single-source AVN.

The batch contains a matched Source-argmax control for each model plus the
complete ATENA Cartesian product.  Jobs are pinned to one Git commit, the same
2,000 val episodes/order used by Source/Tent, and immutable checkpoint/data
digests.  Concurrency is user-controlled per physical GPU.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, TextIO, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = REPO_ROOT / "avn" / "results" / "logs" / "atena_grid"
RUN_ROOT = REPO_ROOT / "avn" / "results" / "runs"
DATASET = (
    REPO_ROOT
    / "avn"
    / "data"
    / "datasets"
    / "tta_test"
    / "single_source"
    / "mp3d"
    / "v1"
    / "val"
    / "val.json.gz"
)
FINGERPRINT_TOOL = REPO_ROOT / "avn" / "scripts" / "fingerprint_episode_stream.py"
MANIFEST_VALIDATOR = REPO_ROOT / "tools" / "validate_run_manifest.py"
EXPERIMENT_SPEC = REPO_ROOT / "avn" / "experiments" / "atena_grid.yaml"

MODELS = ("smt_audio", "enmus")
SEED = 0
CANONICAL_EPISODES = 2000
CANONICAL_DATASET_INDEX_SHA256 = (
    "838532d8e10064dd2bccbdbb7e75b8ca7cb5c4e7a3db579c3b40cfab18081c80"
)
CANONICAL_STREAM_ORDER_SHA256 = (
    "07f327590ccee2999b3f6bcb2fc412f39d9802cf932b14933fd0bdd9e5ca380c"
)
CANONICAL_STREAM_CONTENT_SHA256 = (
    "dd411c4aafaf626b2848d20b92d1832ea46a5380c57107043fc639996837fdf2"
)

MODEL_RUNNERS = {
    "smt_audio": REPO_ROOT / "avn" / "scripts" / "eval_smt_audio.sh",
    "enmus": REPO_ROOT / "avn" / "scripts" / "eval_enmus.sh",
}
MODEL_CONFIGS = {
    "smt_audio": (
        "ss_baselines/savi/config/tta_avn/single_source/"
        "smt_audio_tta_test.yaml"
    ),
    "enmus": "sen_baselines/enmus/config/single_source/enmus_tta_test.yaml",
}
MODEL_CONFIG_PATHS = {
    "smt_audio": (
        REPO_ROOT / "avn" / "baselines" / "smt_audio" / MODEL_CONFIGS["smt_audio"]
    ),
    "enmus": REPO_ROOT / "avn" / "baselines" / "enmus" / MODEL_CONFIGS["enmus"],
}
MODEL_CHECKPOINTS = {
    "smt_audio": (
        REPO_ROOT
        / "avn"
        / "checkpoints"
        / "source"
        / "smt_audio"
        / "single_best_val.pth"
    ),
    "enmus": (
        REPO_ROOT
        / "avn"
        / "checkpoints"
        / "source"
        / "enmus"
        / "single_source_best_val.pth"
    ),
}
MODEL_CHECKPOINT_SHA256 = {
    "smt_audio": (
        "8007dc0de8b0e994244d4f2fdb4a642bcc6213b4e9694568c93b10141f53ef03"
    ),
    "enmus": (
        "4f37a377cc7fcb888c545850c91883560a908ba5366072df787e4c8238ecefcd"
    ),
}
ENMUS_AUXILIARY = {
    "audio_encoder": (
        REPO_ROOT
        / "avn/baselines/enmus/data/pretrained_weights/semantic_audionav/enmus/"
        "audio_encoder_best_val.pth"
    ),
    "visual_encoder": (
        REPO_ROOT
        / "avn/baselines/enmus/data/pretrained_weights/semantic_audionav/enmus/"
        "visual_encoder_best_val.pth"
    ),
    "seld_encoder": (
        REPO_ROOT
        / "avn/baselines/enmus/data/pretrained_weights/semantic_audionav/enmus/"
        "seld_crnn_best_val.h5"
    ),
}
ENMUS_AUXILIARY_SHA256 = {
    "audio_encoder": (
        "0939d0546f47b291eac35f6570c3fdb97c46dd45e0730d5e559f5696d4233b47"
    ),
    "visual_encoder": (
        "daa326623772a5fb2db3abd74413ea0bbc822f2dc66029db5a25828df95a69b5"
    ),
    "seld_encoder": (
        "7d36aa62eb6fa8043bebb8ebb8bb11a972f5fec9e039ae7260400dbdc38646db"
    ),
}

# Three AVN-informed low-rate pairs plus all three released ATENA anchors:
# ETPNav (5e-7/1e-8), DUET-R2R (8e-7/1e-7), and DUET-REVERIE
# (5e-6/1e-7).  Keeping the released pairs intact has priority over imposing a
# fixed ratio between query and self learning rates.
LR_PAIRS = (
    ("3e-8", "1e-8"),
    ("1e-7", "1e-8"),
    ("3e-7", "3e-8"),
    ("5e-7", "1e-8"),
    ("8e-7", "1e-7"),
    ("5e-6", "1e-7"),
)
MIX_LAMBDAS = ("0.25", "0.5", "0.75")
# 0.1 is the official DUET setting.  The higher values are required because
# observed AVN action entropy is around 0.7, so 0.1 alone would query nearly all
# episodes and would not test SAL's self-feedback branch.
QUERY_THRESHOLDS = ("0.1", "0.5", "0.75", "1.0")
SELF_LOSS_WEIGHTS = ("0.1", "0.25")
OPTIMIZER = "AdamW"
WEIGHT_DECAY = "0.01"
PARAM_SCOPE = "all"
GRID_JOBS_PER_MODEL = (
    len(LR_PAIRS)
    * len(MIX_LAMBDAS)
    * len(QUERY_THRESHOLDS)
    * len(SELF_LOSS_WEIGHTS)
)
FULL_JOB_COUNT = len(MODELS) * (1 + GRID_JOBS_PER_MODEL)

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


class UserError(RuntimeError):
    """Expected experiment error printed without a traceback."""


@dataclass(frozen=True)
class Point:
    lr_query: str
    lr_self: str
    mix_lambda: str
    query_threshold: str
    self_loss_weight: str


@dataclass(frozen=True)
class Job:
    job_id: int
    model_job_id: int
    model: str
    kind: str
    point: Optional[Point]
    gpu_index: int
    gpu: str
    run_tag: str


@dataclass
class Worker:
    job: Job
    process: subprocess.Popen
    output: TextIO
    attempt_dir: Path
    previous_manifests: Tuple[Path, ...]


@dataclass(frozen=True)
class Provenance:
    git_commit: str
    dataset_index_sha256: str
    stream_order_sha256: str
    stream_content_sha256: str
    checkpoint_sha256: Mapping[str, str]
    auxiliary_sha256: Mapping[str, str]
    model_config_sha256: Mapping[str, str]
    launcher_sha256: str
    experiment_spec_sha256: str


def utc_now(compact: bool = False) -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ" if compact else "%Y-%m-%dT%H:%M:%SZ")


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    temporary.write_text(content, encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise UserError(message)


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
            "Run 288 ATENA search jobs plus two matched Source-argmax controls "
            "on four GPUs."
        )
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpus,
        default=parse_gpus("0,1,2,3"),
        metavar="A,B,C,D",
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=positive_int,
        default=1,
        help=(
            "concurrent ATENA/source processes per GPU; begin with 1 for the "
            "memory smoke test, then choose the measured safe value"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=positive_int,
        default=CANONICAL_EPISODES,
        help="episodes per job; values below 2000 require --smoke",
    )
    parser.add_argument(
        "--batch-id",
        default="",
        help="stable batch id (default: atena-grid-v1-seed0-UTC timestamp)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one official-anchor ATENA job plus Source-argmax per model",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes > CANONICAL_EPISODES:
        parser.error("--episodes must not exceed 2000")
    if args.episodes < CANONICAL_EPISODES and not args.smoke:
        parser.error("--episodes below 2000 require --smoke")
    if args.smoke and args.episodes == CANONICAL_EPISODES:
        parser.error("--smoke must use fewer than 2000 episodes")
    if not args.batch_id:
        if args.resume:
            parser.error("--resume requires an explicit --batch-id")
        args.batch_id = "atena-grid-v1-seed0-{}".format(utc_now(compact=True))
    if len(args.batch_id) > 64 or SAFE_ID.fullmatch(args.batch_id) is None:
        parser.error("--batch-id must be at most 64 safe characters")
    return args


def all_points() -> Tuple[Point, ...]:
    points = tuple(
        Point(lr_query, lr_self, mix_lambda, threshold, self_weight)
        for lr_query, lr_self in LR_PAIRS
        for mix_lambda in MIX_LAMBDAS
        for threshold in QUERY_THRESHOLDS
        for self_weight in SELF_LOSS_WEIGHTS
    )
    require(
        len(points) == GRID_JOBS_PER_MODEL == 144,
        "ATENA Cartesian product must contain 144 points/model",
    )
    require(len(set(points)) == len(points), "duplicate ATENA grid point")
    return points


def numeric_slug(value: str) -> str:
    return value.replace("-", "m").replace(".", "p").replace("+", "")


def point_slug(point: Point) -> str:
    return "qlr{}-slr{}-lam{}-d{}-w{}".format(
        numeric_slug(point.lr_query),
        numeric_slug(point.lr_self),
        numeric_slug(point.mix_lambda),
        numeric_slug(point.query_threshold),
        numeric_slug(point.self_loss_weight),
    )


def build_jobs(args: argparse.Namespace) -> List[Job]:
    jobs: List[Job] = []
    next_id = 0
    points = all_points()
    # Released DUET-R2R values, contained unchanged in the formal grid.
    anchor = Point("8e-7", "1e-7", "0.75", "0.1", "0.1")
    selected_points = (anchor,) if args.smoke else points

    # Put both matched controls first so their reference metrics are available
    # near the beginning of a long batch.
    for model_index, model in enumerate(MODELS):
        source_tag = "atena-{}-{}-source-argmax".format(
            args.batch_id, "sa" if model == "smt_audio" else "em"
        )
        jobs.append(
            Job(
                job_id=next_id,
                model_job_id=-1,
                model=model,
                kind="source_argmax",
                point=None,
                gpu_index=model_index,
                gpu=args.gpus[model_index],
                run_tag=source_tag,
            )
        )
        next_id += 1

    for model_index, model in enumerate(MODELS):
        for model_job_id, point in enumerate(selected_points):
            # The +2 offset puts the two smoke anchors on GPUs 2/3 while the
            # two Source controls use GPUs 0/1.  In a full grid it remains
            # exactly balanced at 36 ATENA jobs per physical GPU/model.
            gpu_index = (model_job_id + model_index + 2) % len(args.gpus)
            tag = "atena-{}-{}-j{:03d}-{}".format(
                args.batch_id,
                "sa" if model == "smt_audio" else "em",
                model_job_id,
                point_slug(point),
            )
            jobs.append(
                Job(
                    job_id=next_id,
                    model_job_id=model_job_id,
                    model=model,
                    kind="atena",
                    point=point,
                    gpu_index=gpu_index,
                    gpu=args.gpus[gpu_index],
                    run_tag=tag,
                )
            )
            next_id += 1
    expected = 4 if args.smoke else FULL_JOB_COUNT
    require(len(jobs) == expected, "ATENA job-count invariant failed")
    require(len({job.run_tag for job in jobs}) == len(jobs), "duplicate run tag")
    if not args.smoke:
        for model in MODELS:
            model_grid = [
                job for job in jobs if job.model == model and job.kind == "atena"
            ]
            require(len(model_grid) == 144, "model grid count mismatch")
            counts = [
                sum(job.gpu_index == index for job in model_grid)
                for index in range(4)
            ]
            require(counts == [36, 36, 36, 36], "grid is not GPU-balanced")
    return jobs


def point_dict(point: Optional[Point]) -> Dict[str, object]:
    if point is None:
        return {}
    return {
        "lr_query": point.lr_query,
        "lr_self": point.lr_self,
        "mix_lambda": point.mix_lambda,
        "query_threshold": point.query_threshold,
        "self_loss_weight": point.self_loss_weight,
    }


def expected_overrides(job: Job, episodes: int) -> List[str]:
    overrides = [
        "TEST_EPISODE_COUNT", str(episodes),
        "NUM_PROCESSES", "1",
        "EVAL.SPLIT", "val",
        "EVAL.USE_CKPT_CONFIG", "False",
        "EVAL.ACTION_SELECTION", "argmax",
    ]
    if job.kind == "atena":
        require(job.point is not None, "ATENA job has no point")
        overrides.extend([
            "TTA.EPISODIC", "False",
            "TTA.ATENA.PREFLIGHT_APPROVED", "True",
            "TTA.ATENA.LR_QUERY", job.point.lr_query,
            "TTA.ATENA.LR_SELF", job.point.lr_self,
            "TTA.ATENA.MIX_LAMBDA", job.point.mix_lambda,
            "TTA.ATENA.QUERY_THRESHOLD", job.point.query_threshold,
            "TTA.ATENA.SELF_LOSS_WEIGHT", job.point.self_loss_weight,
            "TTA.ATENA.PARAM_SCOPE", PARAM_SCOPE,
            "TTA.ATENA.ACTION_SELECTION_PROTOCOL", "policy_argmax",
            "TTA.ATENA.OPTIMIZER", OPTIMIZER,
            "TTA.ATENA.BETA1", "0.9",
            "TTA.ATENA.BETA2", "0.999",
            "TTA.ATENA.WEIGHT_DECAY", WEIGHT_DECAY,
            "TTA.ATENA.MAX_GRAD_NORM", "0.0",
        ])
    return overrides


def job_command(job: Job, episodes: int) -> List[str]:
    method = "source" if job.kind == "source_argmax" else "atena"
    return [
        "bash",
        str(MODEL_RUNNERS[job.model]),
        "single_source",
        method,
        str(SEED),
        *expected_overrides(job, episodes),
    ]


def fingerprint_stream(episodes: int) -> Tuple[str, str]:
    output = run_capture((
        sys.executable,
        str(FINGERPRINT_TOOL),
        "--dataset", str(DATASET),
        "--seed", str(SEED),
        "--episode-count", str(episodes),
    ))
    fields = output.split()
    require(len(fields) == 2, "invalid stream fingerprint output")
    require(
        all(re.fullmatch(r"[0-9a-f]{64}", item) for item in fields),
        "invalid stream fingerprint digest",
    )
    return fields[0], fields[1]


def preflight(args: argparse.Namespace) -> Provenance:
    required = [
        Path(__file__), EXPERIMENT_SPEC, FINGERPRINT_TOOL, MANIFEST_VALIDATOR,
        DATASET, *MODEL_RUNNERS.values(), *MODEL_CONFIG_PATHS.values(),
        *MODEL_CHECKPOINTS.values(), *ENMUS_AUXILIARY.values(),
    ]
    for path in required:
        require(path.is_file(), "missing required file: {}".format(path))
    commit = run_capture(("git", "rev-parse", "HEAD"))
    require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid Git commit")
    tracked_status = run_capture((
        "git", "status", "--porcelain", "--untracked-files=no"
    ))
    require(
        not tracked_status,
        "tracked worktree changes detected; commit them before ATENA runs",
    )
    dataset_hash = sha256_file(DATASET)
    require(
        dataset_hash == CANONICAL_DATASET_INDEX_SHA256,
        "dataset index does not match canonical Source/Tent val",
    )
    order_hash, content_hash = fingerprint_stream(args.episodes)
    if args.episodes == CANONICAL_EPISODES:
        require(
            order_hash == CANONICAL_STREAM_ORDER_SHA256,
            "episode order does not match canonical Source/Tent val",
        )
        require(
            content_hash == CANONICAL_STREAM_CONTENT_SHA256,
            "episode content does not match canonical Source/Tent val",
        )
    checkpoint_hashes = {}
    for model, path in MODEL_CHECKPOINTS.items():
        digest = sha256_file(path)
        require(
            digest == MODEL_CHECKPOINT_SHA256[model],
            "{} source checkpoint SHA256 mismatch".format(model),
        )
        checkpoint_hashes[model] = digest
    auxiliary_hashes = {}
    for name, path in ENMUS_AUXILIARY.items():
        digest = sha256_file(path)
        require(
            digest == ENMUS_AUXILIARY_SHA256[name],
            "ENMuS {} checkpoint SHA256 mismatch".format(name),
        )
        auxiliary_hashes[name] = digest
    return Provenance(
        git_commit=commit,
        dataset_index_sha256=dataset_hash,
        stream_order_sha256=order_hash,
        stream_content_sha256=content_hash,
        checkpoint_sha256=checkpoint_hashes,
        auxiliary_sha256=auxiliary_hashes,
        model_config_sha256={
            model: sha256_file(path) for model, path in MODEL_CONFIG_PATHS.items()
        },
        launcher_sha256=sha256_file(Path(__file__)),
        experiment_spec_sha256=sha256_file(EXPERIMENT_SPEC),
    )


def print_plan(args: argparse.Namespace, jobs: Sequence[Job]) -> None:
    print("ATENA AVN canonical single-source grid")
    print("  jobs:                 {}".format(len(jobs)))
    print("  ATENA/model:          {}".format(1 if args.smoke else 144))
    print("  Source-argmax/model:  1")
    print("  GPUs:                 {}".format(",".join(args.gpus)))
    print("  jobs/GPU:             {}".format(args.jobs_per_gpu))
    print("  max active jobs:      {}".format(args.jobs_per_gpu * 4))
    print("  episodes/job:         {}".format(args.episodes))
    print("  batch:                {}".format(args.batch_id))
    print("  logs:                 {}".format(LOG_ROOT / args.batch_id))
    print("  LR pairs:             {}".format(LR_PAIRS))
    print("  lambdas:              {}".format(MIX_LAMBDAS))
    print("  thresholds:           {}".format(QUERY_THRESHOLDS))
    print("  self-loss weights:    {}".format(SELF_LOSS_WEIGHTS))
    for job in jobs[: min(4, len(jobs))]:
        print("  sample command:       {}".format(" ".join(job_command(job, args.episodes))))


def write_grid(path: Path, jobs: Sequence[Job]) -> None:
    fields = [
        "job_id", "model_job_id", "model", "kind", "gpu", "run_tag",
        "lr_query", "lr_self", "mix_lambda", "query_threshold",
        "self_loss_weight",
    ]
    temporary = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            row = {
                "job_id": job.job_id,
                "model_job_id": job.model_job_id,
                "model": job.model,
                "kind": job.kind,
                "gpu": job.gpu,
                "run_tag": job.run_tag,
                **point_dict(job.point),
            }
            writer.writerow(row)
    os.replace(str(temporary), str(path))


def job_root(batch_dir: Path, job: Job) -> Path:
    return batch_dir / "jobs" / job.run_tag


def next_attempt_dir(root: Path) -> Path:
    attempts = root / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name)
        for path in attempts.iterdir()
        if path.is_dir() and re.fullmatch(r"[0-9]{3}", path.name)
    ]
    result = attempts / "{:03d}".format(max(existing, default=0) + 1)
    result.mkdir()
    return result


def manifests_for_tag(run_tag: str) -> Tuple[Path, ...]:
    if not RUN_ROOT.is_dir():
        return ()
    return tuple(sorted(RUN_ROOT.glob("*{}*/manifest.json".format(run_tag))))


def launch_job(
    job: Job,
    args: argparse.Namespace,
    batch_dir: Path,
    provenance: Provenance,
) -> Worker:
    root = job_root(batch_dir, job)
    root.mkdir(parents=True, exist_ok=True)
    attempt = next_attempt_dir(root)
    command = job_command(job, args.episodes)
    parameters = {
        "job_id": job.job_id,
        "model_job_id": job.model_job_id,
        "model": job.model,
        "kind": job.kind,
        "gpu": job.gpu,
        "run_tag": job.run_tag,
        "episodes": args.episodes,
        "git_commit": provenance.git_commit,
        "point": point_dict(job.point),
        "command": command,
        "started_at": utc_now(),
    }
    atomic_write(attempt / "parameters.json", json_text(parameters))
    output = (attempt / "console.log").open("w", encoding="utf-8", buffering=1)
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": job.gpu,
        "NAVTTA_RUN_TAG": job.run_tag,
        "NAVTTA_STREAM_ORDER_SHA256": provenance.stream_order_sha256,
        "NAVTTA_STREAM_CONTENT_SHA256": provenance.stream_content_sha256,
        "NAVTTA_EVAL_SPLIT": "val",
        "PYTHONUNBUFFERED": "1",
    })
    previous = manifests_for_tag(job.run_tag)
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        env=environment,
        stdout=output,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return Worker(job, process, output, attempt, previous)


def nested_value(document: object, path: Sequence[str]) -> object:
    value = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def override_map(manifest: Mapping[str, object]) -> Dict[str, str]:
    values = manifest.get("config_overrides")
    require(isinstance(values, list), "manifest config overrides are missing")
    require(len(values) % 2 == 0, "manifest config overrides have odd length")
    output: Dict[str, str] = {}
    for index in range(0, len(values), 2):
        key = str(values[index])
        require(key not in output, "duplicate config override: {}".format(key))
        output[key] = str(values[index + 1])
    return output


def validate_manifest(
    manifest_path: Path,
    job: Job,
    args: argparse.Namespace,
    provenance: Provenance,
) -> Mapping[str, object]:
    method = "source" if job.kind == "source_argmax" else "atena"
    completed = subprocess.run(
        [
            sys.executable, str(MANIFEST_VALIDATOR),
            "--manifest", str(manifest_path),
            "--run-tag", job.run_tag,
            "--model", job.model,
            "--method", method,
            "--source-setting", "single_source",
            "--seed", str(SEED),
            "--git-commit", provenance.git_commit,
            "--checkpoint-sha256", provenance.checkpoint_sha256[job.model],
            "--stream-order-sha256", provenance.stream_order_sha256,
            "--stream-content-sha256", provenance.stream_content_sha256,
        ],
        cwd=str(REPO_ROOT),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(
        completed.returncode == 0,
        completed.stderr.strip() or "run manifest validation failed",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("cannot read run manifest: {}".format(error)) from error
    require(manifest.get("config") == MODEL_CONFIGS[job.model], "config mismatch")
    require(
        nested_value(manifest, ("dataset", "index_sha256"))
        == provenance.dataset_index_sha256,
        "dataset digest mismatch",
    )
    require(
        nested_value(manifest, ("hardware", "cuda_visible_devices")) == job.gpu,
        "GPU identity mismatch",
    )
    actual_overrides = override_map(manifest)
    expected = expected_overrides(job, args.episodes)
    for index in range(0, len(expected), 2):
        require(
            actual_overrides.get(expected[index]) == expected[index + 1],
            "config override mismatch: {}".format(expected[index]),
        )
    return manifest


def read_stats(run_dir: Path, episodes: int) -> Tuple[Dict[str, float], Path]:
    stats_path = run_dir / "raw/model/tb/val_stats_{}.json".format(SEED)
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("cannot read episode statistics: {}".format(error)) from error
    require(isinstance(stats, dict), "episode statistics are not a dictionary")
    require(len(stats) == episodes, "episode statistics count mismatch")
    aggregate = {}
    for metric in METRICS:
        values = []
        for episode_key, record in stats.items():
            require(isinstance(record, dict), "invalid episode {}".format(episode_key))
            try:
                value = float(record[metric])
            except (KeyError, TypeError, ValueError) as error:
                raise UserError(
                    "episode {} missing/nonnumeric {}".format(episode_key, metric)
                ) from error
            require(math.isfinite(value), "non-finite episode metric")
            values.append(value)
        aggregate[metric] = sum(values) / len(values)
    return aggregate, stats_path


def finite_diagnostic(diagnostics: Mapping[str, object], key: str) -> float:
    try:
        value = float(diagnostics[key])
    except (KeyError, TypeError, ValueError) as error:
        raise UserError("missing/nonnumeric diagnostic: {}".format(key)) from error
    require(math.isfinite(value), "non-finite diagnostic: {}".format(key))
    return value


def validate_atena_diagnostics(
    run_dir: Path,
    job: Job,
    episodes: int,
    metrics: Mapping[str, float],
) -> Tuple[Mapping[str, object], Path]:
    require(job.point is not None, "ATENA point is missing")
    path = run_dir / "raw/model/tb/tta_diagnostics_{}.json".format(SEED)
    try:
        diagnostics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("cannot read ATENA diagnostics: {}".format(error)) from error
    require(isinstance(diagnostics, dict), "ATENA diagnostics are invalid")
    require(diagnostics.get("episodes") == episodes, "episode count mismatch")
    require(diagnostics.get("updates") == episodes, "episode update count mismatch")
    require(diagnostics.get("action_steps", 0) > 0, "no action steps recorded")
    require(diagnostics.get("optimizer") == OPTIMIZER, "optimizer mismatch")
    require(diagnostics.get("param_scope") == PARAM_SCOPE, "scope mismatch")
    require(diagnostics.get("action_selection") == "argmax", "action protocol mismatch")
    require(diagnostics.get("retains_episode_graph") is False, "graph retention enabled")
    require(
        diagnostics.get("gradient_reconstruction")
        == "exact_step_replay_in_eval_mode",
        "gradient replay mode mismatch",
    )
    names = diagnostics.get("adapted_parameter_names")
    require(isinstance(names, list) and names, "adapted parameter names missing")
    require(len(names) == len(set(names)), "duplicate adapted parameter names")
    require(
        not any(str(name).startswith("critic.") for name in names),
        "value-only critic was selected",
    )
    require(
        int(diagnostics.get("adapted_parameter_count", 0)) > 0,
        "adapted parameter count is invalid",
    )
    require(
        int(diagnostics.get("self_prediction_parameter_count", 0)) > 0,
        "self-prediction head parameter count is invalid",
    )
    for key, expected in (
        ("lr_query", job.point.lr_query),
        ("lr_self", job.point.lr_self),
        ("mix_lambda", job.point.mix_lambda),
        ("query_threshold", job.point.query_threshold),
        ("self_loss_weight", job.point.self_loss_weight),
    ):
        require(
            math.isclose(
                finite_diagnostic(diagnostics, key),
                float(expected),
                rel_tol=1e-12,
                abs_tol=0.0,
            ),
            "ATENA hyperparameter mismatch: {}".format(key),
        )
    queries = int(diagnostics.get("queries", -1))
    self_labels = int(diagnostics.get("self_label_episodes", -1))
    require(queries >= 0 and self_labels >= 0, "feedback counts are invalid")
    require(queries + self_labels == episodes, "feedback episode count mismatch")
    query_rate = finite_diagnostic(diagnostics, "query_rate")
    require(
        math.isclose(query_rate, queries / episodes, abs_tol=1e-12),
        "query rate mismatch",
    )
    actual_successes = int(diagnostics.get("actual_successful_episodes", -1))
    require(
        actual_successes == int(round(float(metrics["success"]) * episodes)),
        "diagnostic/metric success count mismatch",
    )
    require(
        int(diagnostics.get("max_trajectory_steps", 0)) > 0,
        "trajectory length diagnostic is invalid",
    )
    require(
        int(diagnostics.get("max_trajectory_storage_bytes", 0)) > 0,
        "trajectory storage diagnostic is invalid",
    )
    require(
        finite_diagnostic(diagnostics, "max_replay_feature_abs_error") <= 1e-5,
        "episode replay exceeded deterministic tolerance",
    )
    for key in (
        "mean_entropy", "last_entropy", "last_grad_norm",
        "relative_param_drift", "mean_episode_entropy",
        "mean_query_episode_entropy", "mean_self_episode_entropy",
        "query_prediction_accuracy", "self_prediction_accuracy",
        "all_prediction_accuracy_offline", "mean_self_prediction_loss",
        "mean_episode_adaptation_seconds", "max_episode_adaptation_seconds",
    ):
        finite_diagnostic(diagnostics, key)
    require(
        int(diagnostics.get("cuda_peak_memory_allocated_bytes", -1)) >= 0,
        "CUDA peak-memory diagnostic is invalid",
    )
    return diagnostics, path


def validate_completed_job(
    worker: Worker,
    args: argparse.Namespace,
    provenance: Provenance,
) -> Dict[str, object]:
    current = manifests_for_tag(worker.job.run_tag)
    previous = set(worker.previous_manifests)
    created = [path for path in current if path not in previous]
    require(len(created) == 1, "expected exactly one new run manifest")
    manifest_path = created[0]
    manifest = validate_manifest(manifest_path, worker.job, args, provenance)
    run_dir = manifest_path.parent
    metrics, stats_path = read_stats(run_dir, args.episodes)
    diagnostics: Mapping[str, object] = {}
    diagnostics_path: Optional[Path] = None
    if worker.job.kind == "atena":
        diagnostics, diagnostics_path = validate_atena_diagnostics(
            run_dir, worker.job, args.episodes, metrics
        )
    result: Dict[str, object] = {
        "job_id": worker.job.job_id,
        "model_job_id": worker.job.model_job_id,
        "model": worker.job.model,
        "kind": worker.job.kind,
        "method": "source" if worker.job.kind == "source_argmax" else "atena",
        "run_tag": worker.job.run_tag,
        "gpu": worker.job.gpu,
        "episodes": args.episodes,
        "git_commit": provenance.git_commit,
        "manifest": str(manifest_path.resolve()),
        "stats": str(stats_path.resolve()),
        "point": point_dict(worker.job.point),
        "metrics": metrics,
        "validated_at": utc_now(),
    }
    if diagnostics_path is not None:
        result["diagnostics"] = str(diagnostics_path.resolve())
        result["query_rate"] = diagnostics["query_rate"]
        result["query_prediction_accuracy"] = diagnostics[
            "query_prediction_accuracy"
        ]
        result["self_prediction_accuracy"] = diagnostics[
            "self_prediction_accuracy"
        ]
        result["relative_param_drift"] = diagnostics["relative_param_drift"]
        result["adapted_parameter_count"] = diagnostics[
            "adapted_parameter_count"
        ]
    # Ensure the validated run is exactly the process-created run.
    require(manifest.get("status") == "completed", "run did not complete")
    return result


def finalize_worker(
    worker: Worker,
    args: argparse.Namespace,
    provenance: Provenance,
    batch_dir: Path,
) -> bool:
    worker.output.close()
    runner_status = int(worker.process.returncode)
    atomic_write(worker.attempt_dir / "runner_exitcode", "{}\n".format(runner_status))
    root = job_root(batch_dir, worker.job)
    if runner_status != 0:
        atomic_write(worker.attempt_dir / "validation", "runner_failed\n")
        atomic_write(root / "validation", "runner_failed\n")
        return False
    try:
        result = validate_completed_job(worker, args, provenance)
    except UserError as error:
        atomic_write(worker.attempt_dir / "validation", "failed\n")
        atomic_write(worker.attempt_dir / "validation_error", str(error) + "\n")
        atomic_write(root / "validation", "failed\n")
        atomic_write(root / "validation_error", str(error) + "\n")
        return False
    atomic_write(worker.attempt_dir / "validation", "ok\n")
    atomic_write(worker.attempt_dir / "result.json", json_text(result))
    atomic_write(root / "validation", "ok\n")
    atomic_write(root / "result.json", json_text(result))
    manifest_path = Path(str(result["manifest"]))
    shutil.copy2(manifest_path, root / "manifest.json")
    if worker.job.kind == "atena":
        shutil.copy2(Path(str(result["diagnostics"])), root / "diagnostics.json")
    return True


def validated_result(batch_dir: Path, job: Job) -> Optional[Mapping[str, object]]:
    root = job_root(batch_dir, job)
    try:
        if (root / "validation").read_text(encoding="utf-8").strip() != "ok":
            return None
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        result.get("run_tag") != job.run_tag
        or result.get("model") != job.model
        or result.get("kind") != job.kind
    ):
        return None
    return result


def write_metrics_csv(
    batch_dir: Path, jobs: Sequence[Job]
) -> Tuple[int, int]:
    fields = [
        "job_id", "model_job_id", "model", "kind", "method", "run_tag",
        "gpu", "lr_query", "lr_self", "mix_lambda", "query_threshold",
        "self_loss_weight", *METRICS, "query_rate",
        "query_prediction_accuracy", "self_prediction_accuracy",
        "relative_param_drift", "adapted_parameter_count", "manifest",
    ]
    rows = []
    for job in jobs:
        result = validated_result(batch_dir, job)
        if result is None:
            continue
        metrics = result.get("metrics", {})
        row = {
            "job_id": job.job_id,
            "model_job_id": job.model_job_id,
            "model": job.model,
            "kind": job.kind,
            "method": result.get("method", ""),
            "run_tag": job.run_tag,
            "gpu": job.gpu,
            **point_dict(job.point),
            **{metric: metrics.get(metric, "") for metric in METRICS},
            "query_rate": result.get("query_rate", ""),
            "query_prediction_accuracy": result.get(
                "query_prediction_accuracy", ""
            ),
            "self_prediction_accuracy": result.get(
                "self_prediction_accuracy", ""
            ),
            "relative_param_drift": result.get("relative_param_drift", ""),
            "adapted_parameter_count": result.get("adapted_parameter_count", ""),
            "manifest": result.get("manifest", ""),
        }
        rows.append(row)
    temporary = batch_dir / "metrics.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(batch_dir / "metrics.csv"))
    return len(rows), len(jobs) - len(rows)


def acquire_lock(batch_id: str) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / ".{}.lock".format(batch_id)
    try:
        path.mkdir()
    except FileExistsError as error:
        raise UserError("ATENA batch is already locked: {}".format(path)) from error
    atomic_write(path / "owner.json", json_text({
        "pid": os.getpid(), "host": os.uname().nodename, "started_at": utc_now()
    }))
    return path


def release_lock(path: Path) -> None:
    try:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    except OSError:
        pass


def prepare_batch(
    args: argparse.Namespace,
    jobs: Sequence[Job],
    provenance: Provenance,
) -> Path:
    batch_dir = LOG_ROOT / args.batch_id
    metadata = {
        "experiment": "atena_avn_full_cartesian_v1",
        "result_role": "hyperparameter_search",
        "batch_id": args.batch_id,
        "created_at": utc_now(),
        "git_commit": provenance.git_commit,
        "smoke": args.smoke,
        "episodes": args.episodes,
        "expected_jobs": len(jobs),
        "models": list(MODELS),
        "source_setting": "single_source",
        "split": "val",
        "seed": SEED,
        "dataset_index_sha256": provenance.dataset_index_sha256,
        "stream_order_sha256": provenance.stream_order_sha256,
        "stream_content_sha256": provenance.stream_content_sha256,
        "checkpoint_sha256": dict(provenance.checkpoint_sha256),
        "auxiliary_sha256": dict(provenance.auxiliary_sha256),
        "model_config_sha256": dict(provenance.model_config_sha256),
        "launcher_sha256": provenance.launcher_sha256,
        "experiment_spec_sha256": provenance.experiment_spec_sha256,
        "grid": {
            "lr_pairs": [list(pair) for pair in LR_PAIRS],
            "mix_lambdas": list(MIX_LAMBDAS),
            "query_thresholds": list(QUERY_THRESHOLDS),
            "self_loss_weights": list(SELF_LOSS_WEIGHTS),
        },
    }
    if args.resume:
        require(batch_dir.is_dir(), "resume batch does not exist")
        try:
            old = json.loads((batch_dir / "batch.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise UserError("cannot read resume metadata: {}".format(error)) from error
        for key in (
            "git_commit", "smoke", "episodes", "expected_jobs",
            "dataset_index_sha256", "stream_order_sha256",
            "stream_content_sha256", "checkpoint_sha256", "grid",
        ):
            require(old.get(key) == metadata.get(key), "resume metadata mismatch: {}".format(key))
    else:
        require(not batch_dir.exists(), "batch already exists; use --resume")
        batch_dir.mkdir(parents=True)
        (batch_dir / "jobs").mkdir()
        atomic_write(batch_dir / "batch.json", json_text(metadata))
        write_grid(batch_dir / "grid.csv", jobs)
    return batch_dir


def terminate_workers(workers: Sequence[Worker]) -> None:
    for worker in workers:
        if worker.process.poll() is None:
            worker.process.terminate()
    deadline = time.time() + 10
    while time.time() < deadline and any(
        worker.process.poll() is None for worker in workers
    ):
        time.sleep(0.2)
    for worker in workers:
        if worker.process.poll() is None:
            worker.process.kill()
        worker.output.close()


def run_scheduler(
    args: argparse.Namespace,
    jobs: Sequence[Job],
    provenance: Provenance,
    batch_dir: Path,
) -> int:
    pending = [job for job in jobs if validated_result(batch_dir, job) is None]
    skipped = len(jobs) - len(pending)
    active: List[Worker] = []
    gpu_counts = {gpu: 0 for gpu in args.gpus}
    failed = 0
    scheduler_path = batch_dir / "scheduler.log"
    log = scheduler_path.open("a", encoding="utf-8", buffering=1)

    def message(value: str) -> None:
        rendered = "[{}] {}".format(utc_now(), value)
        print(rendered, flush=True)
        log.write(rendered + "\n")

    interrupted = False

    def handle_signal(signum, _frame):
        nonlocal interrupted
        interrupted = True
        message("received signal {}; stopping active jobs".format(signum))

    old_int = signal.signal(signal.SIGINT, handle_signal)
    old_term = signal.signal(signal.SIGTERM, handle_signal)
    try:
        message(
            "start expected={} pending={} skipped={} jobs_per_gpu={}".format(
                len(jobs), len(pending), skipped, args.jobs_per_gpu
            )
        )
        while pending or active:
            if interrupted:
                terminate_workers(active)
                break
            launched = True
            while launched:
                launched = False
                for index, job in enumerate(pending):
                    if gpu_counts[job.gpu] >= args.jobs_per_gpu:
                        continue
                    worker = launch_job(job, args, batch_dir, provenance)
                    active.append(worker)
                    gpu_counts[job.gpu] += 1
                    pending.pop(index)
                    message(
                        "launch job={} model={} kind={} gpu={} pid={} remaining={}".format(
                            job.job_id, job.model, job.kind, job.gpu,
                            worker.process.pid, len(pending)
                        )
                    )
                    launched = True
                    break
            finished = [worker for worker in active if worker.process.poll() is not None]
            for worker in finished:
                active.remove(worker)
                gpu_counts[worker.job.gpu] -= 1
                valid = finalize_worker(
                    worker, args, provenance, batch_dir
                )
                failed += int(not valid)
                message(
                    "finish job={} model={} kind={} runner={} validated={}".format(
                        worker.job.job_id, worker.job.model, worker.job.kind,
                        worker.process.returncode, valid
                    )
                )
                write_metrics_csv(batch_dir, jobs)
            if (pending or active) and not finished:
                time.sleep(2)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        log.close()
    metrics_count, missing = write_metrics_csv(batch_dir, jobs)
    summary = {
        "batch_id": args.batch_id,
        "git_commit": provenance.git_commit,
        "smoke": args.smoke,
        "episodes": args.episodes,
        "expected": len(jobs),
        "validated": metrics_count,
        "failed": failed,
        "missing": missing,
        "complete": not interrupted and missing == 0 and failed == 0,
        "completed_at": utc_now(),
    }
    atomic_write(batch_dir / "SUMMARY.json", json_text(summary))
    return 130 if interrupted else (0 if summary["complete"] else 1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    jobs = build_jobs(args)
    print_plan(args, jobs)
    if args.dry_run:
        return 0
    provenance = preflight(args)
    lock = acquire_lock(args.batch_id)
    try:
        batch_dir = prepare_batch(args, jobs, provenance)
        return run_scheduler(args, jobs, provenance, batch_dir)
    finally:
        release_lock(lock)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
