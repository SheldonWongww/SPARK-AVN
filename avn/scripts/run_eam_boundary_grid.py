#!/usr/bin/env python3
"""Run the joint SMT+Audio/ENMuS EAM weak-update boundary grid.

The full grid is deliberately fixed to the canonical single-source AVN val
stream: seed 0, 2,000 episodes, sampled actions, and the Source/Tent episode
order.  A single scheduler owns the GPU counters, so ``--jobs-per-gpu`` is a
combined limit across both models rather than a per-child-scheduler limit.
"""

import argparse
import csv
import hashlib
import io
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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_BASE = REPO_ROOT / "avn" / "results" / "logs" / "eam_boundary_grid"
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

MODELS = ("smt_audio", "enmus")
LRS = ("3e-9", "1e-8", "3e-8")
UPDATE_INTERVALS = (32, 64, 128)
SEED = 0
CANONICAL_EPISODES = 2000
PREFIXES = (
    "net.smt_state_encoder.transformer",
    "action_distribution",
)

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
        / "avn"
        / "baselines"
        / "enmus"
        / "data"
        / "pretrained_weights"
        / "semantic_audionav"
        / "enmus"
        / "audio_encoder_best_val.pth"
    ),
    "visual_encoder": (
        REPO_ROOT
        / "avn"
        / "baselines"
        / "enmus"
        / "data"
        / "pretrained_weights"
        / "semantic_audionav"
        / "enmus"
        / "visual_encoder_best_val.pth"
    ),
    "seld_encoder": (
        REPO_ROOT
        / "avn"
        / "baselines"
        / "enmus"
        / "data"
        / "pretrained_weights"
        / "semantic_audionav"
        / "enmus"
        / "seld_crnn_best_val.h5"
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

# These counts lock the declared full-Transformer-plus-head scope.  ENMuS uses
# its one-layer MSMT custom decoder, so its scope is larger than SMT+Audio's
# standard one-layer Transformer even though the module prefixes are identical.
# Independent checkpoint audit: the ENMuS Transformer has 84 state entries and
# 3,500,549 stored scalars; excluding 15 BatchNorm running/count buffers (2,565
# scalars) leaves 69 parameter tensors/3,497,984 scalars.  Its 2-tensor,
# 1,028-scalar action head gives the validated total of 71/3,499,012 below.
EXPECTED_TENSORS = {"smt_audio": 36, "enmus": 71}
EXPECTED_PARAMETERS = {"smt_audio": 1057284, "enmus": 3499012}

CONFIDENCE_SCALE = "0.4"
MEMORY_SIZE = 32
BATCH_SIZE = 8
PARAM_SCOPE = "module_prefixes"
EPISODIC = "False"
STEPS = 1
OPTIMIZER = "Adam"
BETA1 = "0.9"
BETA2 = "0.999"
WEIGHT_DECAY = "0.0"
MAX_GRAD_NORM = "0.0"

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


class UserError(RuntimeError):
    """A launcher error that should be shown without a traceback."""


@dataclass(frozen=True)
class Job:
    job_id: int
    model_job_id: int
    model: str
    lr: str
    update_interval: int
    gpu_index: int
    gpu: str
    run_tag: str


@dataclass(frozen=True)
class Provenance:
    git_commit: str
    tracked_status: str
    tracked_worktree_dirty: bool
    dataset_index_sha256: str
    stream_order_sha256: str
    stream_content_sha256: str
    checkpoint_sha256: Mapping[str, str]
    auxiliary_sha256: Mapping[str, str]


@dataclass
class Worker:
    job: Job
    process: subprocess.Popen
    output: TextIO
    job_dir: Path


class Logger:
    def __init__(self, path: Path) -> None:
        self._handle = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        print(message, flush=True)
        self._handle.write(message + "\n")

    def close(self) -> None:
        self._handle.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_write(path: Path, value: str) -> None:
    temporary = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
    temporary.write_text(value, encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_capture(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(REPO_ROOT),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        raise UserError("cannot run {}: {}".format(command[0], error)) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise UserError(
            "command failed ({}): {}".format(
                completed.returncode, detail or " ".join(command)
            )
        )
    return completed.stdout.strip()


def parse_gpu_csv(value: str) -> Tuple[str, ...]:
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+){3}", value):
        raise argparse.ArgumentTypeError(
            "expected exactly four comma-separated numeric GPU ids"
        )
    values = tuple(value.split(","))
    if len(set(values)) != 4:
        raise argparse.ArgumentTypeError("GPU ids must be distinct")
    return values


def positive_integer(value: str) -> int:
    if not re.fullmatch(r"[1-9][0-9]*", value):
        raise argparse.ArgumentTypeError("expected a positive integer")
    return int(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 18 joint EAM boundary jobs: SMT+Audio and ENMuS each use "
            "3 LRs x 3 update intervals on the canonical single-source val stream."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Full run (fixed seed 0 and 2,000 episodes):\n"
            "  python3 avn/scripts/run_eam_boundary_grid.py --gpus 0,1,2,3 "
            "--jobs-per-gpu 2 --batch-id eam-boundary-joint-v1-seed0\n\n"
            "Two-job, two-episode pathway smoke (still pass four GPU ids):\n"
            "  python3 avn/scripts/run_eam_boundary_grid.py --gpus 0,1,2,3 "
            "--jobs-per-gpu 1 --smoke --episodes 2 --allow-dirty "
            "--batch-id eam-boundary-smoke"
        ),
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpu_csv,
        default=parse_gpu_csv("0,1,2,3"),
        metavar="A,B,C,D",
        help="exactly four distinct physical GPU ids (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=positive_integer,
        default=1,
        help="combined concurrent jobs per GPU across both models (default: 1)",
    )
    parser.add_argument(
        "--episodes",
        type=positive_integer,
        default=CANONICAL_EPISODES,
        help="episodes per job; values below 2000 require --smoke",
    )
    parser.add_argument(
        "--batch-id",
        default="",
        help="stable batch id (default: UTC timestamp)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the immutable named batch and revalidate completed jobs",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit tracked worktree changes (never recommended for formal runs)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="select one configuration per model instead of all 18 jobs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print and validate the plan without filesystem or process changes",
    )
    args = parser.parse_args(argv)

    if args.jobs_per_gpu > len(MODELS) * len(LRS) * len(UPDATE_INTERVALS):
        parser.error("--jobs-per-gpu cannot exceed the 18-job full grid")
    if args.episodes > CANONICAL_EPISODES:
        parser.error("--episodes cannot exceed the 2,000-episode canonical stream")
    if not args.smoke and args.episodes != CANONICAL_EPISODES:
        parser.error("full boundary runs require exactly 2,000 episodes")
    if args.allow_dirty and not args.smoke:
        parser.error("--allow-dirty is permitted only with --smoke")
    if args.resume and not args.batch_id:
        parser.error("--resume requires an explicit --batch-id")
    if not args.batch_id:
        args.batch_id = "eam-boundary-joint-v1-seed0-{}".format(timestamp_slug())
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.batch_id):
        parser.error(
            "--batch-id may contain only letters, numbers, dot, underscore, and hyphen"
        )
    if len(args.batch_id) > 96:
        parser.error("--batch-id cannot exceed 96 characters")
    return args


def value_slug(value: str) -> str:
    return value.replace(".", "p").replace("+", "p").replace("-", "m")


def make_run_tag(
    batch_id: str, job_id: int, model: str, lr: str, interval: int
) -> str:
    return "eamboundary-{}-j{:03d}-{}-lr{}-u{}".format(
        batch_id, job_id, model, value_slug(lr), interval
    )


def build_plan(args: argparse.Namespace) -> List[Job]:
    plan: List[Job] = []
    global_job_id = 0
    for model_index, model in enumerate(MODELS):
        for lr_index, lr in enumerate(LRS):
            for interval_index, interval in enumerate(UPDATE_INTERVALS):
                model_job_id = lr_index * len(UPDATE_INTERVALS) + interval_index
                selected = not args.smoke or model_job_id == 0
                if selected:
                    # Each model touches every GPU.  The one-slot ENMuS offset
                    # yields a balanced combined 5/5/4/4 full-grid assignment.
                    gpu_index = (model_job_id + model_index) % len(args.gpus)
                    plan.append(
                        Job(
                            job_id=global_job_id,
                            model_job_id=model_job_id,
                            model=model,
                            lr=lr,
                            update_interval=interval,
                            gpu_index=gpu_index,
                            gpu=args.gpus[gpu_index],
                            run_tag=make_run_tag(
                                args.batch_id,
                                global_job_id,
                                model,
                                lr,
                                interval,
                            ),
                        )
                    )
                global_job_id += 1

    expected = 2 if args.smoke else 18
    if len(plan) != expected or len({job.run_tag for job in plan}) != expected:
        raise UserError("internal EAM boundary plan size or tag collision")
    for model in MODELS:
        model_jobs = [job for job in plan if job.model == model]
        expected_model_jobs = 1 if args.smoke else 9
        combinations = {(job.lr, job.update_interval) for job in model_jobs}
        if len(model_jobs) != expected_model_jobs or len(combinations) != expected_model_jobs:
            raise UserError("invalid {} Cartesian grid".format(model))
    if not args.smoke:
        gpu_counts = [sum(job.gpu_index == index for job in plan) for index in range(4)]
        if sorted(gpu_counts) != [4, 4, 5, 5]:
            raise UserError("joint EAM jobs are not balanced across four GPUs")
        if any(
            len({job.gpu_index for job in plan if job.model == model}) != 4
            for model in MODELS
        ):
            raise UserError("each model must be represented on every GPU")
    return plan


def plan_csv(plan: Iterable[Job], episodes: int) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "job_id",
            "model_job_id",
            "run_tag",
            "gpu",
            "model",
            "source_setting",
            "method",
            "scope",
            "expected_tensors",
            "expected_parameters",
            "lr",
            "update_interval",
            "seed",
            "episodes",
        )
    )
    for job in plan:
        writer.writerow(
            (
                job.job_id,
                job.model_job_id,
                job.run_tag,
                job.gpu,
                job.model,
                "single_source",
                "eam",
                "full_transformer_plus_head",
                EXPECTED_TENSORS[job.model],
                EXPECTED_PARAMETERS[job.model],
                job.lr,
                job.update_interval,
                SEED,
                episodes,
            )
        )
    return output.getvalue()


def print_plan(args: argparse.Namespace, plan: Sequence[Job]) -> None:
    print("AVN joint EAM weak-update boundary grid")
    print("  models:              smt_audio,enmus")
    print("  source/split:        single_source/val")
    print("  LRs:                 {}".format(",".join(LRS)))
    print(
        "  update intervals:    {}".format(
            ",".join(str(value) for value in UPDATE_INTERVALS)
        )
    )
    print("  seed/episodes:       {}/{}".format(SEED, args.episodes))
    print("  GPUs:                {}".format(",".join(args.gpus)))
    print("  combined jobs/GPU:   {}".format(args.jobs_per_gpu))
    print("  selected jobs:       {}".format(len(plan)))
    print("  batch:               {}".format(args.batch_id))
    print("  log root:            {}".format(LOG_BASE / args.batch_id))
    print(plan_csv(plan, args.episodes), end="")


def fingerprint_stream(episodes: int) -> Tuple[str, str]:
    output = run_capture(
        (
            sys.executable,
            str(FINGERPRINT_TOOL),
            "--dataset",
            str(DATASET),
            "--seed",
            str(SEED),
            "--episode-count",
            str(episodes),
        )
    )
    values = output.split()
    if len(values) != 2 or any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None for value in values
    ):
        raise UserError("episode-stream fingerprint tool returned invalid output")
    return values[0], values[1]


def validate_preflight(args: argparse.Namespace) -> Provenance:
    required = (
        Path(__file__).resolve(),
        DATASET,
        FINGERPRINT_TOOL,
        MANIFEST_VALIDATOR,
        *(MODEL_RUNNERS[model] for model in MODELS),
        *(MODEL_CHECKPOINTS[model] for model in MODELS),
        *ENMUS_AUXILIARY.values(),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise UserError("missing required files:\n  " + "\n  ".join(missing))
    for command in ("git", "bash"):
        if shutil.which(command) is None:
            raise UserError("{} is unavailable".format(command))

    provenance_sources = (
        Path(__file__).resolve(),
        FINGERPRINT_TOOL.resolve(),
        MANIFEST_VALIDATOR.resolve(),
        *(MODEL_RUNNERS[model].resolve() for model in MODELS),
    )
    for path in provenance_sources:
        try:
            relative = path.relative_to(REPO_ROOT.resolve())
        except ValueError as error:
            raise UserError("provenance source is outside the repository: {}".format(path)) from error
        tracked = subprocess.run(
            ("git", "cat-file", "-e", "HEAD:{}".format(relative.as_posix())),
            cwd=str(REPO_ROOT),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tracked.returncode != 0:
            raise UserError(
                "launcher dependency is not tracked by HEAD: {}".format(relative)
            )

    commit = run_capture(("git", "rev-parse", "HEAD"))
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise UserError("invalid Git commit: {}".format(commit))
    tracked_status = run_capture(
        ("git", "status", "--porcelain", "--untracked-files=no")
    )
    if tracked_status and not args.allow_dirty:
        raise UserError(
            "tracked worktree changes detected; commit them or use --allow-dirty"
        )

    dataset_hash = sha256_file(DATASET)
    if dataset_hash != CANONICAL_DATASET_INDEX_SHA256:
        raise UserError("dataset index does not match canonical Source/Tent val")
    order_hash, content_hash = fingerprint_stream(args.episodes)
    if args.episodes == CANONICAL_EPISODES:
        if order_hash != CANONICAL_STREAM_ORDER_SHA256:
            raise UserError("episode order does not match canonical Source/Tent val")
        if content_hash != CANONICAL_STREAM_CONTENT_SHA256:
            raise UserError("episode content does not match canonical Source/Tent val")

    checkpoint_hashes: Dict[str, str] = {}
    for model in MODELS:
        digest = sha256_file(MODEL_CHECKPOINTS[model])
        if digest != MODEL_CHECKPOINT_SHA256[model]:
            raise UserError("{} Source checkpoint SHA256 mismatch".format(model))
        checkpoint_hashes[model] = digest

    auxiliary_hashes: Dict[str, str] = {}
    for name, path in ENMUS_AUXILIARY.items():
        digest = sha256_file(path)
        if digest != ENMUS_AUXILIARY_SHA256[name]:
            raise UserError("ENMuS {} SHA256 mismatch".format(name))
        auxiliary_hashes[name] = digest

    return Provenance(
        git_commit=commit,
        tracked_status=tracked_status,
        tracked_worktree_dirty=bool(tracked_status),
        dataset_index_sha256=dataset_hash,
        stream_order_sha256=order_hash,
        stream_content_sha256=content_hash,
        checkpoint_sha256=checkpoint_hashes,
        auxiliary_sha256=auxiliary_hashes,
    )


def require_repository_unchanged(provenance: Provenance) -> None:
    if run_capture(("git", "rev-parse", "HEAD")) != provenance.git_commit:
        raise UserError("repository HEAD changed while the grid was running")
    status = run_capture(("git", "status", "--porcelain", "--untracked-files=no"))
    if status != provenance.tracked_status:
        raise UserError("tracked worktree changed while the grid was running")


def batch_spec(
    args: argparse.Namespace, provenance: Provenance, plan: Sequence[Job]
) -> Mapping[str, object]:
    return {
        "experiment": "eam_weak_update_boundary_joint_v1",
        "result_role": "hyperparameter_search",
        "protocol": "source_tent_aligned_single_source_val_seed0",
        "models": list(MODELS),
        "source_setting": "single_source",
        "split": "val",
        "method": "eam",
        "action_selection": {
            "smt_audio": "sample",
            "enmus": "native_sample_deterministic_false",
        },
        "seed": SEED,
        "episodes": args.episodes,
        "gpus": list(args.gpus),
        "jobs_per_gpu_combined": args.jobs_per_gpu,
        "smoke": args.smoke,
        "expected_jobs": len(plan),
        "lrs": list(LRS),
        "update_intervals": list(UPDATE_INTERVALS),
        "scope": "full_transformer_plus_head",
        "trainable_prefixes": list(PREFIXES),
        "expected_tensors": EXPECTED_TENSORS,
        "expected_parameters": EXPECTED_PARAMETERS,
        "confidence_scale": CONFIDENCE_SCALE,
        "memory_size": MEMORY_SIZE,
        "batch_size": BATCH_SIZE,
        "param_scope": PARAM_SCOPE,
        "episodic": EPISODIC,
        "steps": STEPS,
        "optimizer": OPTIMIZER,
        "beta1": BETA1,
        "beta2": BETA2,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "git_commit": provenance.git_commit,
        "tracked_worktree_dirty": provenance.tracked_worktree_dirty,
        "allow_dirty": args.allow_dirty,
        "dataset_index_sha256": provenance.dataset_index_sha256,
        "stream_order_sha256": provenance.stream_order_sha256,
        "stream_content_sha256": provenance.stream_content_sha256,
        "checkpoint_sha256": dict(provenance.checkpoint_sha256),
        "enmus_auxiliary_sha256": dict(provenance.auxiliary_sha256),
    }


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def initialize_batch(
    args: argparse.Namespace, provenance: Provenance, plan: Sequence[Job]
) -> Path:
    batch_dir = LOG_BASE / args.batch_id
    expected_spec = json_text(batch_spec(args, provenance, plan))
    expected_plan = plan_csv(plan, args.episodes)
    if args.resume:
        if not batch_dir.is_dir():
            raise UserError("resume batch does not exist: {}".format(batch_dir))
        spec_path = batch_dir / "batch.json"
        plan_path = batch_dir / "grid.csv"
        if not spec_path.is_file() or spec_path.read_text(encoding="utf-8") != expected_spec:
            raise UserError("resume arguments or immutable inputs do not match batch.json")
        if not plan_path.is_file() or plan_path.read_text(encoding="utf-8") != expected_plan:
            raise UserError("resume plan does not match grid.csv")
    else:
        if batch_dir.exists():
            raise UserError("batch already exists: {}".format(batch_dir))
        LOG_BASE.mkdir(parents=True, exist_ok=True)
        batch_dir.mkdir()
        atomic_write(batch_dir / "batch.json", expected_spec)
        atomic_write(batch_dir / "grid.csv", expected_plan)

    for model in MODELS:
        (batch_dir / model / "jobs").mkdir(parents=True, exist_ok=True)
    return batch_dir


def acquire_lock(lock: Path, description: str) -> Path:
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise UserError(
            "{} is already running or has a stale lock: {}".format(description, lock)
        ) from error
    try:
        atomic_write(
            lock / "owner",
            "pid={}\nhost={}\nstarted_at={}\n".format(
                os.getpid(), os.uname().nodename, utc_now()
            ),
        )
    except BaseException:
        lock.rmdir()
        raise
    return lock


def release_lock(lock: Optional[Path]) -> None:
    if lock is None:
        return
    try:
        (lock / "owner").unlink()
        lock.rmdir()
    except FileNotFoundError:
        pass


def expected_override_map(job: Job, episodes: int) -> Mapping[str, str]:
    values = {
        "NUM_PROCESSES": "1",
        "EVAL.USE_CKPT_CONFIG": "False",
        "TTA.EPISODIC": EPISODIC,
        "TTA.STEPS": str(STEPS),
        "TTA.EAM.LR": job.lr,
        "TTA.EAM.CONFIDENCE_SCALE": CONFIDENCE_SCALE,
        "TTA.EAM.MEMORY_SIZE": str(MEMORY_SIZE),
        "TTA.EAM.BATCH_SIZE": str(BATCH_SIZE),
        "TTA.EAM.UPDATE_INTERVAL": str(job.update_interval),
        "TTA.EAM.PARAM_SCOPE": PARAM_SCOPE,
        "TTA.EAM.TRAINABLE_PREFIXES": json.dumps(
            list(PREFIXES), separators=(",", ":")
        ),
        "TTA.EAM.OPTIMIZER": OPTIMIZER,
        "TTA.EAM.BETA1": BETA1,
        "TTA.EAM.BETA2": BETA2,
        "TTA.EAM.WEIGHT_DECAY": WEIGHT_DECAY,
        "TTA.EAM.MAX_GRAD_NORM": MAX_GRAD_NORM,
        "TEST_EPISODE_COUNT": str(episodes),
    }
    if job.model == "smt_audio":
        values["EVAL.ACTION_SELECTION"] = "sample"
    else:
        values["EVAL.SPLIT"] = "val"
    return values


def runner_command(job: Job, episodes: int) -> List[str]:
    command = [
        "bash",
        str(MODEL_RUNNERS[job.model]),
        "single_source",
        "eam",
        str(SEED),
    ]
    for key, value in expected_override_map(job, episodes).items():
        command.extend((key, value))
    return command


def job_environment(job: Job, provenance: Provenance) -> Dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": job.gpu,
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NAVTTA_RUN_TAG": job.run_tag,
            "NAVTTA_STREAM_ORDER_SHA256": provenance.stream_order_sha256,
            "NAVTTA_STREAM_CONTENT_SHA256": provenance.stream_content_sha256,
        }
    )
    if job.model == "enmus":
        environment["NAVTTA_EVAL_SPLIT"] = "val"
    return environment


def parameters_text(job: Job, args: argparse.Namespace, provenance: Provenance) -> str:
    values = (
        ("job_id", job.job_id),
        ("model_job_id", job.model_job_id),
        ("run_tag", job.run_tag),
        ("gpu", job.gpu),
        ("model", job.model),
        ("source_setting", "single_source"),
        ("split", "val"),
        ("method", "eam"),
        (
            "action_selection",
            "sample" if job.model == "smt_audio" else "native_sample_deterministic_false",
        ),
        ("scope", "full_transformer_plus_head"),
        ("trainable_prefixes", json.dumps(list(PREFIXES), separators=(",", ":"))),
        ("expected_tensors", EXPECTED_TENSORS[job.model]),
        ("expected_parameters", EXPECTED_PARAMETERS[job.model]),
        ("lr", job.lr),
        ("update_interval", job.update_interval),
        ("seed", SEED),
        ("episodes", args.episodes),
        ("git_commit", provenance.git_commit),
        ("checkpoint_sha256", provenance.checkpoint_sha256[job.model]),
        ("dataset_index_sha256", provenance.dataset_index_sha256),
        ("stream_order_sha256", provenance.stream_order_sha256),
        ("stream_content_sha256", provenance.stream_content_sha256),
    )
    return "".join("{}={}\n".format(key, value) for key, value in values)


def archive_previous(job_dir: Path) -> None:
    names = (
        "parameters.env",
        "console.log",
        "manifest.path",
        "runner_exitcode",
        "validation",
        "exitcode",
        "metrics.json",
    )
    if not any((job_dir / name).exists() for name in names):
        return
    suffix = timestamp_slug()
    for name in names:
        path = job_dir / name
        if path.exists():
            path.rename(job_dir / "{}.previous.{}".format(name, suffix))


def nested_value(document: object, keys: Sequence[str]) -> object:
    value = document
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise UserError(message)


def validate_manifest_details(
    manifest: Mapping[str, object],
    job: Job,
    episodes: int,
    provenance: Provenance,
) -> None:
    require(manifest.get("config") == MODEL_CONFIGS[job.model], "config path mismatch")
    require(
        nested_value(manifest, ("dataset", "index_sha256"))
        == provenance.dataset_index_sha256,
        "manifest dataset index mismatch",
    )
    require(
        nested_value(manifest, ("hardware", "cuda_visible_devices")) == job.gpu,
        "manifest GPU identity mismatch",
    )
    checkpoint_path = nested_value(manifest, ("checkpoint", "path"))
    require(
        isinstance(checkpoint_path, str)
        and Path(checkpoint_path).resolve() == MODEL_CHECKPOINTS[job.model].resolve(),
        "manifest checkpoint path mismatch",
    )

    overrides = manifest.get("config_overrides")
    require(isinstance(overrides, list), "manifest config overrides are missing")
    require(len(overrides) % 2 == 0, "manifest config overrides have odd length")
    override_map: Dict[str, str] = {}
    for index in range(0, len(overrides), 2):
        key = str(overrides[index])
        require(key not in override_map, "duplicate config override: {}".format(key))
        override_map[key] = str(overrides[index + 1])
    for key, expected in expected_override_map(job, episodes).items():
        require(
            override_map.get(key) == expected,
            "config override mismatch for {}".format(key),
        )
    if job.model == "enmus":
        require(
            "EVAL.ACTION_SELECTION" not in override_map,
            "ENMuS must use its native sampled-action path",
        )

    auxiliary = manifest.get("auxiliary_checkpoints")
    if job.model == "smt_audio":
        require(auxiliary == [], "unexpected SMT+Audio auxiliary checkpoints")
    else:
        require(isinstance(auxiliary, list), "ENMuS auxiliary checkpoints missing")
        actual = {}
        for item in auxiliary:
            require(isinstance(item, dict), "invalid ENMuS auxiliary checkpoint")
            actual[item.get("name")] = item.get("sha256")
        require(actual == dict(provenance.auxiliary_sha256), "ENMuS auxiliary SHA256 mismatch")


def validate_stats_and_diagnostics(
    manifest_path: Path, job: Job, episodes: int
) -> Dict[str, float]:
    run_dir = manifest_path.resolve().parent
    stats_path = run_dir / "raw" / "model" / "tb" / "val_stats_{}.json".format(SEED)
    diagnostics_path = (
        run_dir / "raw" / "model" / "tb" / "tta_diagnostics_{}.json".format(SEED)
    )
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("cannot read EAM result artifacts: {}".format(error)) from error

    require(isinstance(stats, dict), "episode statistics are not a dictionary")
    require(len(stats) == episodes, "episode statistics count mismatch")
    for episode_key, episode_stats in stats.items():
        require(
            isinstance(episode_stats, dict),
            "episode statistics are invalid for {}".format(episode_key),
        )
        for metric in METRICS:
            require(metric in episode_stats, "episode {} missing {}".format(episode_key, metric))
            try:
                value = float(episode_stats[metric])
            except (TypeError, ValueError) as error:
                raise UserError(
                    "episode {} has nonnumeric {}".format(episode_key, metric)
                ) from error
            require(math.isfinite(value), "episode {} has non-finite {}".format(episode_key, metric))

    require(isinstance(diagnostics, dict), "TTA diagnostics are not a dictionary")
    require(diagnostics.get("episodes") == episodes, "diagnostic episode count mismatch")
    require(diagnostics.get("optimizer") == OPTIMIZER, "EAM optimizer mismatch")
    require(diagnostics.get("param_scope") == PARAM_SCOPE, "scope mode mismatch")
    require(diagnostics.get("trainable_prefixes") == list(PREFIXES), "prefix list mismatch")
    require(
        diagnostics.get("adapted_parameter_count") == EXPECTED_PARAMETERS[job.model],
        "adapted parameter count mismatch for {}".format(job.model),
    )
    names = diagnostics.get("adapted_parameter_names")
    require(isinstance(names, list), "adapted parameter names are missing")
    require(len(names) == EXPECTED_TENSORS[job.model], "adapted tensor count mismatch")
    require(len(set(names)) == len(names), "duplicate adapted parameter names")
    require(
        all(
            any(name == prefix or name.startswith(prefix + ".") for prefix in PREFIXES)
            for name in names
        ),
        "adapted parameter outside declared prefixes",
    )
    try:
        current_lr = float(diagnostics["current_lr"])
        confidence_scale = float(diagnostics["confidence_scale"])
    except (KeyError, TypeError, ValueError) as error:
        raise UserError("missing or invalid EAM scalar diagnostic") from error
    require(math.isclose(current_lr, float(job.lr), rel_tol=1e-12), "EAM LR mismatch")
    require(
        math.isclose(confidence_scale, float(CONFIDENCE_SCALE), rel_tol=1e-12),
        "EAM confidence scale mismatch",
    )
    require(diagnostics.get("memory_size_steps") == MEMORY_SIZE, "memory size mismatch")
    require(diagnostics.get("batch_size_steps") == BATCH_SIZE, "batch size mismatch")
    require(
        diagnostics.get("update_interval") == job.update_interval,
        "update interval mismatch",
    )
    require(diagnostics.get("update_interval_unit") == "action_step", "interval unit mismatch")
    require(diagnostics.get("replay_unit") == "action_step", "replay unit mismatch")
    require(
        diagnostics.get("replay_sampling") == "updated_reservoir_including_current",
        "replay sampling semantics mismatch",
    )
    require(
        diagnostics.get("short_buffer_behavior") == "current_only_update",
        "short-buffer semantics mismatch",
    )
    require(
        diagnostics.get("update_timing") == "after_preupdate_action_selection",
        "EAM update timing mismatch",
    )

    action_steps = int(diagnostics.get("action_steps", 0))
    attempts = int(diagnostics.get("update_attempts", -1))
    updates = int(diagnostics.get("updates", -1))
    seen_steps = int(diagnostics.get("seen_steps", -1))
    require(action_steps > 0, "no EAM action steps recorded")
    require(seen_steps == action_steps, "seen/action-step mismatch")
    require(attempts == action_steps // job.update_interval, "update-attempt count mismatch")
    require(0 <= updates <= attempts, "actual EAM update count is invalid")
    require(
        diagnostics.get("current_only_batches") == min(action_steps, BATCH_SIZE - 1),
        "current-only batch count mismatch",
    )
    require(
        diagnostics.get("replay_size") == min(action_steps, MEMORY_SIZE),
        "final replay size mismatch",
    )
    expected_replayed = sum(
        1 if step < BATCH_SIZE else BATCH_SIZE
        for step in range(job.update_interval, action_steps + 1, job.update_interval)
    )
    require(
        diagnostics.get("replayed_steps") == expected_replayed,
        "replayed-step count mismatch",
    )
    for name in (
        "mean_entropy",
        "last_entropy",
        "last_grad_norm",
        "relative_param_drift",
        "mean_train_loss",
        "last_train_loss",
        "mean_max_action_probability",
    ):
        try:
            value = float(diagnostics[name])
        except (KeyError, TypeError, ValueError) as error:
            raise UserError("missing or nonnumeric diagnostic: {}".format(name)) from error
        require(math.isfinite(value), "non-finite diagnostic: {}".format(name))

    aggregated = {}
    for metric in METRICS:
        values = [float(record[metric]) for record in stats.values()]
        value = sum(values) / len(values)
        require(math.isfinite(value), "non-finite aggregate metric: {}".format(metric))
        aggregated[metric] = value
    return aggregated


def validate_job_artifacts(
    manifest_path: Path,
    job: Job,
    args: argparse.Namespace,
    provenance: Provenance,
) -> Dict[str, float]:
    if not manifest_path.is_file():
        raise UserError("run manifest is missing: {}".format(manifest_path))
    completed = subprocess.run(
        (
            sys.executable,
            str(MANIFEST_VALIDATOR),
            "--manifest",
            str(manifest_path),
            "--run-tag",
            job.run_tag,
            "--model",
            job.model,
            "--method",
            "eam",
            "--source-setting",
            "single_source",
            "--seed",
            str(SEED),
            "--git-commit",
            provenance.git_commit,
            "--checkpoint-sha256",
            provenance.checkpoint_sha256[job.model],
            "--stream-order-sha256",
            provenance.stream_order_sha256,
            "--stream-content-sha256",
            provenance.stream_content_sha256,
        ),
        cwd=str(REPO_ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise UserError(completed.stderr.strip() or "run manifest validation failed")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("invalid run manifest: {}".format(error)) from error
    require(isinstance(manifest, dict), "run manifest is not a dictionary")
    validate_manifest_details(manifest, job, args.episodes, provenance)
    return validate_stats_and_diagnostics(manifest_path, job, args.episodes)


def write_job_metrics(
    manifest_path: Path,
    job: Job,
    job_dir: Path,
    episodes: int,
    metrics: Mapping[str, float],
) -> None:
    payload = {
        "job_id": job.job_id,
        "model_job_id": job.model_job_id,
        "run_tag": job.run_tag,
        "model": job.model,
        "source_setting": "single_source",
        "eval_split": "val",
        "result_role": "hyperparameter_search",
        "method": "eam",
        "seed": SEED,
        "episodes": episodes,
        "manifest": str(manifest_path),
        "configuration": {
            "lr": job.lr,
            "update_interval": job.update_interval,
            "scope": "full_transformer_plus_head",
        },
        "metrics": dict(metrics),
    }
    try:
        atomic_write(job_dir / "metrics.json", json_text(payload))
    except OSError as error:
        raise UserError("cannot write compact job metrics: {}".format(error)) from error


def manifest_from_console(path: Path) -> Optional[Path]:
    candidate: Optional[Path] = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                for line in raw_line.replace("\r", "\n").splitlines():
                    value = line.strip()
                    if value.endswith("/manifest.json"):
                        current = Path(value)
                        if current.is_file():
                            candidate = current
    except OSError:
        return None
    return candidate


def completed_manifest_path(job_dir: Path) -> Optional[Path]:
    path = job_dir / "manifest.path"
    if not path.is_file():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return Path(value) if value else None


def reusable_job(
    job: Job,
    job_dir: Path,
    args: argparse.Namespace,
    provenance: Provenance,
) -> bool:
    if not args.resume:
        return False
    try:
        exitcode = (job_dir / "exitcode").read_text(encoding="utf-8").strip()
        validation = (job_dir / "validation").read_text(encoding="utf-8").strip()
    except OSError:
        return False
    manifest_path = completed_manifest_path(job_dir)
    if exitcode != "0" or validation != "ok" or manifest_path is None:
        return False
    try:
        metrics = validate_job_artifacts(manifest_path, job, args, provenance)
        write_job_metrics(manifest_path, job, job_dir, args.episodes, metrics)
    except UserError:
        return False
    return True


def launch_job(
    job: Job,
    batch_dir: Path,
    args: argparse.Namespace,
    provenance: Provenance,
    logger: Logger,
) -> Worker:
    job_dir = batch_dir / job.model / "jobs" / job.run_tag
    job_dir.mkdir(parents=True, exist_ok=True)
    archive_previous(job_dir)
    atomic_write(job_dir / "parameters.env", parameters_text(job, args, provenance))
    console_path = job_dir / "console.log"
    output = console_path.open("w", encoding="utf-8", buffering=1)
    command = runner_command(job, args.episodes)
    output.write("===== attempt {} =====\n".format(utc_now()))
    output.write("command={}\n".format(" ".join(command)))
    output.flush()
    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=job_environment(job, provenance),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        output.close()
        raise
    logger.log(
        "{} launched job={} model={} tag={} gpu={} active_pid={}".format(
            utc_now(), job.job_id, job.model, job.run_tag, job.gpu, process.pid
        )
    )
    return Worker(job=job, process=process, output=output, job_dir=job_dir)


def finalize_worker(
    worker: Worker,
    status: int,
    args: argparse.Namespace,
    provenance: Provenance,
) -> int:
    worker.output.close()
    atomic_write(worker.job_dir / "runner_exitcode", "{}\n".format(status))
    composite = status
    validation = "failed"
    manifest_path = manifest_from_console(worker.job_dir / "console.log")
    if manifest_path is not None:
        atomic_write(worker.job_dir / "manifest.path", str(manifest_path) + "\n")
    if status == 0 and manifest_path is not None:
        try:
            metrics = validate_job_artifacts(
                manifest_path, worker.job, args, provenance
            )
            write_job_metrics(
                manifest_path,
                worker.job,
                worker.job_dir,
                args.episodes,
                metrics,
            )
        except UserError as error:
            with (worker.job_dir / "console.log").open("a", encoding="utf-8") as handle:
                handle.write("\n[launcher validation]\n{}\n".format(error))
            composite = 90
        else:
            validation = "ok"
            with (worker.job_dir / "console.log").open("a", encoding="utf-8") as handle:
                handle.write("\n[launcher validation]\n{}\n".format(manifest_path))
    elif status == 0:
        composite = 90
        with (worker.job_dir / "console.log").open("a", encoding="utf-8") as handle:
            handle.write("\n[launcher validation]\nmanifest path not found\n")
    atomic_write(worker.job_dir / "validation", validation + "\n")
    atomic_write(worker.job_dir / "exitcode", "{}\n".format(composite))
    return composite


def terminate_workers(workers: Iterable[Worker], logger: Logger) -> None:
    active = [worker for worker in workers if worker.process.poll() is None]
    if not active:
        return
    logger.log("{} terminating {} active worker(s)".format(utc_now(), len(active)))
    for worker in active:
        try:
            os.killpg(worker.process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if all(worker.process.poll() is not None for worker in active):
            break
        time.sleep(0.5)
    for worker in active:
        if worker.process.poll() is None:
            try:
                os.killpg(worker.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            worker.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        if not worker.output.closed:
            worker.output.close()


def write_batch_metrics(
    batch_dir: Path,
    plan: Sequence[Job],
    args: argparse.Namespace,
    provenance: Provenance,
) -> None:
    fields = (
        "job_id",
        "model_job_id",
        "run_tag",
        "gpu",
        "status",
        "validation",
        "model",
        "source_setting",
        "eval_split",
        "result_role",
        "method",
        "seed",
        "episodes",
        "git_commit",
        "checkpoint_sha256",
        "dataset_index_sha256",
        "stream_order_sha256",
        "stream_content_sha256",
        "lr",
        "update_interval",
        "scope",
        "manifest",
    ) + METRICS
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for job in plan:
        job_dir = batch_dir / job.model / "jobs" / job.run_tag
        try:
            status = (job_dir / "exitcode").read_text(encoding="utf-8").strip()
        except OSError:
            status = "missing"
        try:
            validation = (job_dir / "validation").read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            validation = "missing"
        payload = {}
        metrics_path = job_dir / "metrics.json"
        if metrics_path.is_file():
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
        row = {
            "job_id": job.job_id,
            "model_job_id": job.model_job_id,
            "run_tag": job.run_tag,
            "gpu": job.gpu,
            "status": status,
            "validation": validation,
            "model": job.model,
            "source_setting": "single_source",
            "eval_split": "val",
            "result_role": "hyperparameter_search",
            "method": "eam",
            "seed": SEED,
            "episodes": args.episodes,
            "git_commit": provenance.git_commit,
            "checkpoint_sha256": provenance.checkpoint_sha256[job.model],
            "dataset_index_sha256": provenance.dataset_index_sha256,
            "stream_order_sha256": provenance.stream_order_sha256,
            "stream_content_sha256": provenance.stream_content_sha256,
            "lr": job.lr,
            "update_interval": job.update_interval,
            "scope": "full_transformer_plus_head",
            "manifest": payload.get("manifest", ""),
        }
        recorded_metrics = payload.get("metrics", {})
        if not isinstance(recorded_metrics, dict):
            recorded_metrics = {}
        for metric in METRICS:
            row[metric] = recorded_metrics.get(metric, "")
        writer.writerow(row)
    atomic_write(batch_dir / "metrics.csv", output.getvalue())


def summarize(
    batch_dir: Path,
    plan: Sequence[Job],
    args: argparse.Namespace,
    provenance: Provenance,
    launched: int,
    skipped: int,
) -> Tuple[str, bool]:
    write_batch_metrics(batch_dir, plan, args, provenance)
    per_model = {}
    total_successful = 0
    total_failed = 0
    total_missing = 0
    total_validated = 0
    total_metrics = 0
    for model in MODELS:
        successful = failed = missing = validated = metrics_files = 0
        for job in (item for item in plan if item.model == model):
            job_dir = batch_dir / model / "jobs" / job.run_tag
            try:
                status = (job_dir / "exitcode").read_text(encoding="utf-8").strip()
            except OSError:
                missing += 1
                continue
            if status == "0":
                successful += 1
            else:
                failed += 1
            try:
                if (job_dir / "validation").read_text(encoding="utf-8").strip() == "ok":
                    validated += 1
            except OSError:
                pass
            if (job_dir / "metrics.json").is_file():
                metrics_files += 1
        expected = sum(item.model == model for item in plan)
        per_model[model] = {
            "expected": expected,
            "successful": successful,
            "failed": failed,
            "missing": missing,
            "validated": validated,
            "metrics": metrics_files,
        }
        total_successful += successful
        total_failed += failed
        total_missing += missing
        total_validated += validated
        total_metrics += metrics_files

    complete = (
        total_successful == len(plan)
        and total_failed == 0
        and total_missing == 0
        and total_validated == len(plan)
        and total_metrics == len(plan)
    )
    summary = {
        "batch_id": args.batch_id,
        "git_commit": provenance.git_commit,
        "result_role": "hyperparameter_search",
        "protocol": "source_tent_aligned_single_source_val_seed0",
        "completed_at": utc_now(),
        "tracked_worktree_dirty": provenance.tracked_worktree_dirty,
        "smoke": args.smoke,
        "expected": len(plan),
        "launched_this_invocation": launched,
        "skipped_validated": skipped,
        "successful": total_successful,
        "failed": total_failed,
        "missing": total_missing,
        "validated": total_validated,
        "metrics": total_metrics,
        "complete": complete,
        "per_model": per_model,
        "dataset_index_sha256": provenance.dataset_index_sha256,
        "stream_order_sha256": provenance.stream_order_sha256,
        "stream_content_sha256": provenance.stream_content_sha256,
    }
    return json_text(summary), complete


def run_grid(
    args: argparse.Namespace,
    plan: Sequence[Job],
    provenance: Provenance,
    batch_dir: Path,
) -> int:
    batch_lock = acquire_lock(batch_dir / ".scheduler.lock", "batch")
    logger = Logger(batch_dir / "scheduler.log")
    pending: List[Job] = []
    skipped = 0
    for job in plan:
        job_dir = batch_dir / job.model / "jobs" / job.run_tag
        if reusable_job(job, job_dir, args, provenance):
            skipped += 1
            logger.log("{} skip validated tag={}".format(utc_now(), job.run_tag))
        else:
            pending.append(job)

    workers: Dict[int, Worker] = {}
    active_per_gpu = [0 for _ in args.gpus]
    launched = 0
    failures = 0
    stop_signal: List[Optional[int]] = [None]
    old_handlers = {}

    def request_stop(received: int, _frame: object) -> None:
        stop_signal[0] = received

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)

    try:
        logger.log("scheduler_started_at={}".format(utc_now()))
        logger.log("git_commit={}".format(provenance.git_commit))
        logger.log("combined_jobs_per_gpu={}".format(args.jobs_per_gpu))
        logger.log("stream_order_sha256={}".format(provenance.stream_order_sha256))
        logger.log("stream_content_sha256={}".format(provenance.stream_content_sha256))
        next_repository_check = time.monotonic()

        while pending or workers:
            if stop_signal[0] is not None:
                raise KeyboardInterrupt
            if time.monotonic() >= next_repository_check:
                require_repository_unchanged(provenance)
                next_repository_check = time.monotonic() + 10.0

            launched_one = True
            while launched_one:
                launched_one = False
                for index, job in enumerate(pending):
                    if active_per_gpu[job.gpu_index] >= args.jobs_per_gpu:
                        continue
                    require_repository_unchanged(provenance)
                    worker = launch_job(job, batch_dir, args, provenance, logger)
                    workers[worker.process.pid] = worker
                    active_per_gpu[job.gpu_index] += 1
                    launched += 1
                    del pending[index]
                    launched_one = True
                    break

            finished = []
            for pid, worker in workers.items():
                status = worker.process.poll()
                if status is None:
                    continue
                composite = finalize_worker(worker, status, args, provenance)
                active_per_gpu[worker.job.gpu_index] -= 1
                if composite != 0:
                    failures += 1
                logger.log(
                    "{} finished tag={} runner_status={} status={} active_total={}".format(
                        utc_now(),
                        worker.job.run_tag,
                        status,
                        composite,
                        sum(active_per_gpu),
                    )
                )
                finished.append(pid)
            for pid in finished:
                del workers[pid]

            if (pending or workers) and not finished:
                time.sleep(1.0)

        summary, complete = summarize(
            batch_dir, plan, args, provenance, launched, skipped
        )
        atomic_write(batch_dir / "SUMMARY.json", summary)
        for line in summary.rstrip().splitlines():
            logger.log(line)
        return 0 if complete and failures == 0 else 1
    except KeyboardInterrupt:
        terminate_workers(workers.values(), logger)
        signum = stop_signal[0]
        status = 128 + signum if signum is not None else 130
        logger.log("scheduler_interrupted exitcode={}".format(status))
        return status
    except UserError as error:
        terminate_workers(workers.values(), logger)
        logger.log("scheduler_integrity_failure={}".format(error))
        return 2
    except Exception as error:
        terminate_workers(workers.values(), logger)
        logger.log("scheduler_unexpected_failure={!r}".format(error))
        return 1
    finally:
        for worker in workers.values():
            if not worker.output.closed:
                worker.output.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        logger.close()
        release_lock(batch_lock)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args)
    print_plan(args, plan)
    if args.dry_run:
        print("Dry run complete: {} jobs, nothing launched.".format(len(plan)))
        return 0

    LOG_BASE.mkdir(parents=True, exist_ok=True)
    global_lock = acquire_lock(
        LOG_BASE / ".global_scheduler.lock", "joint EAM boundary scheduler"
    )
    preflight_path = LOG_BASE / "{}.preflight.log".format(args.batch_id)
    try:
        atomic_write(
            preflight_path,
            "preflight_started_at={}\nbatch_id={}\n".format(utc_now(), args.batch_id),
        )
        provenance = validate_preflight(args)
        with preflight_path.open("a", encoding="utf-8") as handle:
            handle.write(
                "preflight_passed_at={}\ngit_commit={}\n"
                "tracked_worktree_dirty={}\ndataset_index_sha256={}\n"
                "stream_order_sha256={}\nstream_content_sha256={}\n".format(
                    utc_now(),
                    provenance.git_commit,
                    int(provenance.tracked_worktree_dirty),
                    provenance.dataset_index_sha256,
                    provenance.stream_order_sha256,
                    provenance.stream_content_sha256,
                )
            )
            for model in MODELS:
                handle.write(
                    "checkpoint_{}_sha256={}\n".format(
                        model, provenance.checkpoint_sha256[model]
                    )
                )
            for name in sorted(provenance.auxiliary_sha256):
                handle.write(
                    "enmus_{}_sha256={}\n".format(
                        name, provenance.auxiliary_sha256[name]
                    )
                )
        batch_dir = initialize_batch(args, provenance, plan)
        return run_grid(args, plan, provenance, batch_dir)
    except UserError as error:
        with preflight_path.open("a", encoding="utf-8") as handle:
            handle.write("error={}\n".format(error))
        raise
    finally:
        release_lock(global_lock)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
