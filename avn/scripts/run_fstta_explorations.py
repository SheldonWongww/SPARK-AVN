#!/usr/bin/env python3
"""Run focused FSTTA mechanism explorations on SMT+Audio single-source AVN.

The launcher intentionally uses only the Python standard library.  A dry run
constructs and checks the complete plan before any dataset or checkpoint is
required; real launches additionally freeze all provenance in the batch
directory and validate every completed run before it can be resumed as done.
"""

from __future__ import annotations

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
from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Deque, Dict, List, Mapping, Optional, Sequence, TextIO, Tuple


SEED = 0
MAX_EPISODES = 2000
MODEL = "smt_audio"
SOURCE_SETTING = "single_source"
EVAL_SPLIT = "val"
EXPLICIT_EVAL_SPLIT = False
EXPERIMENT_TITLE = "AVN FSTTA focused exploration"
BATCH_ID_PREFIX = "fstta-exploration"
FIXED_SUITE: Optional[str] = None
ALLOW_DIRTY_OPTION = True
REQUIRED_GPU_COUNT: Optional[int] = None
SUITE_ORDER = (
    "slow_boundary",
    "fast_geometry",
    "slow_optimizer",
    "q_scaler",
)
EXPECTED_SUITE_COUNTS = {
    "slow_boundary": 40,
    "fast_geometry": 24,
    "slow_optimizer": 12,
    "q_scaler": 16,
}
EXPECTED_ALL_COUNT = 92
SUITE_SLUGS = {
    "slow_boundary": "sb",
    "fast_geometry": "fg",
    "slow_optimizer": "so",
    "q_scaler": "qs",
}

NORM_SCOPE = "last_k_ln"
LAST_K_LN = 4
EPISODIC = False
STEPS = 1
RESET_BN_STATS = True
RHO = "0.95"
TAU = "0.7"
A = "0.9"
B = "1.1"
FAST_OPTIMIZER = "AdamW"
BETA1 = "0.9"
BETA2 = "0.99"
WEIGHT_DECAY = "0.0"
MAX_GRAD_NORM = "1.0"
RESET_FAST_OPTIMIZER_EACH_EPISODE = True
EIGEN_EPS = "1e-6"
ACTION_SELECTION = "sample"
RUNNER_ENV: Mapping[str, str] = {}

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "avn" / "scripts" / "eval_smt_audio.sh"
MANIFEST_VALIDATOR = REPO_ROOT / "tools" / "validate_run_manifest.py"
FINGERPRINT_TOOL = REPO_ROOT / "avn" / "scripts" / "fingerprint_episode_stream.py"
CHECKPOINT = (
    REPO_ROOT / "avn" / "checkpoints" / "source" / "smt_audio"
    / "single_best_val.pth"
)
DATASET = (
    REPO_ROOT / "avn" / "data" / "datasets" / "tta_test"
    / "single_source" / "mp3d" / "v1" / "val" / "val.json.gz"
)
LOG_BASE = REPO_ROOT / "avn" / "results" / "logs" / "fstta_exploration"
AUXILIARY_CHECKPOINTS: Mapping[str, Path] = {}
PROVENANCE_SOURCE_FILES: Tuple[Path, ...] = (
    Path(__file__).resolve(),
    RUNNER.resolve(),
    FINGERPRINT_TOOL.resolve(),
    MANIFEST_VALIDATOR.resolve(),
)
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


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def attempt_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def config_bool(value: bool) -> str:
    return "True" if value else "False"


def slug(value: object) -> str:
    return str(value).replace(".", "p").replace("+", "p").replace("-", "m")


@dataclass(frozen=True)
class JobConfig:
    suite: str
    fast_lr: str = "3e-7"
    fast_window: int = 16
    slow_lr: str = "1e-4"
    slow_window: int = 8
    q: str = "0.1"
    use_slow: bool = True
    fast_grad_mode: str = "concordant"
    use_fast_lr_scaler: bool = True
    slow_optimizer: str = "AdamW"
    slow_momentum: str = "0.0"
    reset_slow_optimizer_each_window: bool = False


@dataclass(frozen=True)
class PlannedJob:
    job_id: int
    suite_job_id: int
    run_tag: str
    gpu: str
    config: JobConfig


@dataclass(frozen=True)
class Provenance:
    git_commit: str
    worktree_dirty: bool
    checkpoint_sha256: str
    dataset_index_sha256: str
    stream_order_sha256: str
    stream_content_sha256: str
    auxiliary_checkpoint_sha256: Tuple[Tuple[str, str], ...]


@dataclass
class RunningJob:
    job: PlannedJob
    process: subprocess.Popen
    console: TextIO
    job_dir: Path


def base_job(suite: str, **changes: object) -> JobConfig:
    return replace(JobConfig(suite=suite), **changes)


def build_slow_boundary() -> List[JobConfig]:
    jobs: List[JobConfig] = []
    for fast_lr in ("1e-8", "3e-7"):
        for fast_window in (8, 16):
            for slow_lr in ("1e-5", "3e-5", "1e-4"):
                for slow_window in (8, 16, 32):
                    jobs.append(
                        base_job(
                            "slow_boundary",
                            fast_lr=fast_lr,
                            fast_window=fast_window,
                            slow_lr=slow_lr,
                            slow_window=slow_window,
                        )
                    )
    # One paired FAST-only control for each (fast LR, M) setting.  LR_SLOW and
    # N remain explicit but are inactive while USE_SLOW=False.
    for fast_lr in ("1e-8", "3e-7"):
        for fast_window in (8, 16):
            jobs.append(
                base_job(
                    "slow_boundary",
                    fast_lr=fast_lr,
                    fast_window=fast_window,
                    use_slow=False,
                )
            )
    return jobs


def build_fast_geometry() -> List[JobConfig]:
    return [
        base_job(
            "fast_geometry",
            fast_lr=fast_lr,
            fast_window=fast_window,
            fast_grad_mode=mode,
            use_slow=False,
        )
        for fast_lr in ("1e-8", "3e-7")
        for fast_window in (2, 4, 8, 16)
        for mode in ("concordant", "mean", "last")
    ]


def build_slow_optimizer() -> List[JobConfig]:
    variants = (
        ("AdamW", "0.0", False),
        ("AdamW", "0.0", True),
        ("SGD", "0.0", False),
    )
    return [
        base_job(
            "slow_optimizer",
            slow_lr=slow_lr,
            slow_window=slow_window,
            slow_optimizer=optimizer,
            slow_momentum=momentum,
            reset_slow_optimizer_each_window=reset_each_window,
        )
        for slow_lr in ("3e-5", "1e-4")
        for slow_window in (8, 16)
        for optimizer, momentum, reset_each_window in variants
    ]


def build_q_scaler() -> List[JobConfig]:
    return [
        base_job(
            "q_scaler",
            q=q,
            slow_window=slow_window,
            use_fast_lr_scaler=use_scaler,
        )
        for q in ("0.1", "0.5", "0.9", "0.99")
        for slow_window in (8, 16)
        for use_scaler in (True, False)
    ]


SUITE_BUILDERS = {
    "slow_boundary": build_slow_boundary,
    "fast_geometry": build_fast_geometry,
    "slow_optimizer": build_slow_optimizer,
    "q_scaler": build_q_scaler,
}


def job_tag(batch_id: str, job_id: int, config: JobConfig) -> str:
    suite_slug = SUITE_SLUGS.get(config.suite, slug(config.suite))
    optimizer_slug = "aw" if config.slow_optimizer == "AdamW" else "sgd"
    return (
        "fsttaexp-{batch}-{suite}-j{job:03d}-flr{flr}-m{m}-slr{slr}"
        "-n{n}-q{q}-g{mode}-sc{sc}-us{use_slow}-so{optimizer}-wr{reset}"
    ).format(
        batch=batch_id,
        suite=suite_slug,
        job=job_id,
        flr=slug(config.fast_lr),
        m=config.fast_window,
        slr=slug(config.slow_lr),
        n=config.slow_window,
        q=slug(config.q),
        mode=config.fast_grad_mode,
        sc=int(config.use_fast_lr_scaler),
        use_slow=int(config.use_slow),
        optimizer=optimizer_slug,
        reset=int(config.reset_slow_optimizer_each_window),
    )


def build_plan(
    selected_suite: str, batch_id: str, gpus: Sequence[str]
) -> List[PlannedJob]:
    catalog = {name: SUITE_BUILDERS[name]() for name in SUITE_ORDER}
    for name, expected in EXPECTED_SUITE_COUNTS.items():
        actual = len(catalog[name])
        if actual != expected:
            raise RuntimeError(
                "internal {} suite-size error: {} (expected {})".format(
                    name, actual, expected
                )
            )
    if sum(len(catalog[name]) for name in SUITE_ORDER) != EXPECTED_ALL_COUNT:
        raise RuntimeError("internal all-suite size is not 92")

    selected_names = SUITE_ORDER if selected_suite == "all" else (selected_suite,)
    plan: List[PlannedJob] = []
    suite_indices: Counter = Counter()
    for name in selected_names:
        for config in catalog[name]:
            job_id = len(plan)
            suite_job_id = suite_indices[name]
            suite_indices[name] += 1
            plan.append(
                PlannedJob(
                    job_id=job_id,
                    suite_job_id=suite_job_id,
                    run_tag=job_tag(batch_id, job_id, config),
                    gpu=gpus[job_id % len(gpus)],
                    config=config,
                )
            )

    expected = (
        EXPECTED_ALL_COUNT
        if selected_suite == "all"
        else EXPECTED_SUITE_COUNTS[selected_suite]
    )
    if len(plan) != expected:
        raise RuntimeError(
            "internal selected-plan size error: {} (expected {})".format(
                len(plan), expected
            )
        )
    tags = [job.run_tag for job in plan]
    if len(tags) != len(set(tags)):
        raise RuntimeError("internal plan contains duplicate run tags")
    if any(not re.fullmatch(r"[A-Za-z0-9._-]+", tag) for tag in tags):
        raise RuntimeError("internal plan contains an unsafe run tag")
    return plan


PLAN_FIELDS = (
    "job_id",
    "suite_job_id",
    "suite",
    "run_tag",
    "gpu",
    "model",
    "source_setting",
    "method",
    "action_selection",
    "fast_lr",
    "M",
    "slow_lr",
    "N",
    "q",
    "use_slow",
    "fast_grad_mode",
    "use_fast_lr_scaler",
    "slow_optimizer",
    "slow_momentum",
    "reset_slow_optimizer_each_window",
    "seed",
    "episodes",
)


def plan_text(plan: Sequence[PlannedJob], episodes: int) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=PLAN_FIELDS, lineterminator="\n")
    writer.writeheader()
    for job in plan:
        config = job.config
        writer.writerow(
            {
                "job_id": job.job_id,
                "suite_job_id": job.suite_job_id,
                "suite": config.suite,
                "run_tag": job.run_tag,
                "gpu": job.gpu,
                "model": MODEL,
                "source_setting": SOURCE_SETTING,
                "method": "fstta",
                "action_selection": ACTION_SELECTION or "native",
                "fast_lr": config.fast_lr,
                "M": config.fast_window,
                "slow_lr": config.slow_lr,
                "N": config.slow_window,
                "q": config.q,
                "use_slow": config_bool(config.use_slow),
                "fast_grad_mode": config.fast_grad_mode,
                "use_fast_lr_scaler": config_bool(config.use_fast_lr_scaler),
                "slow_optimizer": config.slow_optimizer,
                "slow_momentum": config.slow_momentum,
                "reset_slow_optimizer_each_window": config_bool(
                    config.reset_slow_optimizer_each_window
                ),
                "seed": SEED,
                "episodes": episodes,
            }
        )
    return output.getvalue()


def parse_gpus(value: str) -> Tuple[str, ...]:
    gpus = tuple(item.strip() for item in value.split(","))
    if not gpus or any(not re.fullmatch(r"[0-9]+", item) for item in gpus):
        raise argparse.ArgumentTypeError(
            "gpus must be a comma-separated list of nonnegative integers"
        )
    if len(gpus) != len(set(gpus)):
        raise argparse.ArgumentTypeError("GPU ids must be distinct")
    return gpus


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one configured FSTTA exploration suite on the fixed "
            "{} {} seed-0 stream.".format(MODEL, SOURCE_SETTING)
        )
    )
    if FIXED_SUITE is None:
        parser.add_argument("suite", choices=(*SUITE_ORDER, "all"))
    else:
        parser.set_defaults(suite=FIXED_SUITE)
    gpu_help = "comma-separated physical GPU ids (default: 0,1,2,3)"
    if REQUIRED_GPU_COUNT is not None:
        gpu_help += "; exactly {} required".format(REQUIRED_GPU_COUNT)
    parser.add_argument(
        "--gpus",
        default=parse_gpus("0,1,2,3"),
        type=parse_gpus,
        metavar="LIST",
        help=gpu_help,
    )
    parser.add_argument(
        "--jobs-per-gpu",
        default=5,
        type=positive_int,
        metavar="N",
        help="concurrent jobs on each GPU (default: 5; maximum: 16)",
    )
    parser.add_argument(
        "--batch-id",
        help="stable batch id (default: suite and current UTC timestamp)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an immutable named batch and skip validated jobs",
    )
    if ALLOW_DIRTY_OPTION:
        parser.add_argument(
            "--allow-dirty",
            action="store_true",
            help="permit tracked worktree changes",
        )
    else:
        parser.set_defaults(allow_dirty=False)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print and validate the plan without requiring data assets",
    )
    parser.add_argument(
        "--episodes",
        default=MAX_EPISODES,
        type=positive_int,
        metavar="N",
        help="episodes per job (default: 2000; maximum: 2000)",
    )
    args = parser.parse_args(argv)
    if args.jobs_per_gpu > 16:
        parser.error("--jobs-per-gpu must not exceed 16")
    if REQUIRED_GPU_COUNT is not None and len(args.gpus) != REQUIRED_GPU_COUNT:
        parser.error("--gpus must contain exactly {} GPU ids".format(
            REQUIRED_GPU_COUNT
        ))
    if args.episodes > MAX_EPISODES:
        parser.error("the canonical stream contains only 2000 episodes")
    if args.resume and not args.batch_id:
        parser.error("--resume requires --batch-id")
    if not args.batch_id:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.batch_id = "{}-{}-seed0-{}".format(
            BATCH_ID_PREFIX, args.suite, timestamp
        )
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.batch_id):
        parser.error(
            "batch-id may contain only letters, numbers, dot, underscore, and hyphen"
        )
    if len(args.batch_id) > 96:
        parser.error("batch-id must not exceed 96 characters")
    return args


def print_plan_summary(args: argparse.Namespace, plan: Sequence[PlannedJob]) -> None:
    print(EXPERIMENT_TITLE)
    print("  repository:       {}".format(REPO_ROOT))
    print("  model/source:     {}/{}".format(MODEL, SOURCE_SETTING))
    print("  evaluation split: {}".format(EVAL_SPLIT))
    print("  suite:            {}".format(args.suite))
    print("  jobs:             {}".format(len(plan)))
    print("  GPUs:             {}".format(",".join(args.gpus)))
    print("  jobs/GPU:         {} concurrent".format(args.jobs_per_gpu))
    print("  seed:             {}".format(SEED))
    print("  episodes/job:     {}".format(args.episodes))
    print("  action selection: {}".format(ACTION_SELECTION or "native"))
    print("  batch:            {}".format(args.batch_id))
    print("  logs:             {}".format(LOG_BASE / args.batch_id))


def run_capture(command: Sequence[str], cwd: Optional[Path] = None) -> str:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "command failed ({}): {}".format(
                completed.returncode, detail or " ".join(command)
            )
        )
    return completed.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_clean_repository(provenance: Provenance) -> None:
    if provenance.worktree_dirty:
        return
    commit = run_capture(
        ("git", "-C", str(REPO_ROOT), "rev-parse", "HEAD")
    ).strip()
    if commit != provenance.git_commit:
        raise RuntimeError("repository HEAD changed while the batch was running")
    status = run_capture(
        (
            "git",
            "-C",
            str(REPO_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
    )
    if status.strip():
        raise RuntimeError(
            "tracked worktree changed while the batch was running"
        )


def load_provenance(allow_dirty: bool, episodes: int) -> Provenance:
    for executable in ("bash", "git"):
        if shutil.which(executable) is None:
            raise RuntimeError("{} is unavailable".format(executable))
    for path, description in (
        (RUNNER, "evaluation runner"),
        (MANIFEST_VALIDATOR, "run-manifest validator"),
        (FINGERPRINT_TOOL, "episode-stream fingerprint tool"),
        (CHECKPOINT, "source checkpoint"),
        (DATASET, "configured TTA dataset"),
    ):
        if not path.is_file():
            raise RuntimeError("missing {}: {}".format(description, path))
    for name, path in AUXILIARY_CHECKPOINTS.items():
        if not path.is_file():
            raise RuntimeError(
                "missing auxiliary checkpoint '{}': {}".format(name, path)
            )

    commit = run_capture(("git", "-C", str(REPO_ROOT), "rev-parse", "HEAD")).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("invalid Git commit: {}".format(commit))
    status = run_capture(
        (
            "git",
            "-C",
            str(REPO_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
    )
    untracked_sources = []
    for source_path in PROVENANCE_SOURCE_FILES:
        try:
            relative = source_path.resolve().relative_to(REPO_ROOT.resolve())
        except ValueError as error:
            raise RuntimeError(
                "provenance source is outside the repository: {}".format(source_path)
            ) from error
        tracked = subprocess.run(
            (
                "git",
                "-C",
                str(REPO_ROOT),
                "ls-files",
                "--error-unmatch",
                str(relative),
            ),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tracked.returncode != 0:
            untracked_sources.append(str(relative))
    worktree_dirty = bool(status.strip() or untracked_sources)
    if worktree_dirty and not allow_dirty:
        details = ""
        if untracked_sources:
            details = " (untracked experiment sources: {})".format(
                ", ".join(untracked_sources)
            )
        raise RuntimeError(
            "experiment sources or tracked worktree differ from HEAD{}; "
            "commit them first or use --allow-dirty".format(details)
        )

    fingerprint_output = run_capture(
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
    fingerprints = fingerprint_output.split()
    if len(fingerprints) != 2 or any(
        not re.fullmatch(r"[0-9a-f]{64}", item) for item in fingerprints
    ):
        raise RuntimeError(
            "invalid episode-stream fingerprints: {}".format(
                fingerprint_output.strip()
            )
        )
    checkpoint_sha256 = sha256_file(CHECKPOINT)
    if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256):
        raise RuntimeError("invalid checkpoint SHA256")
    dataset_index_sha256 = sha256_file(DATASET)
    auxiliary_hashes = tuple(
        (name, sha256_file(path))
        for name, path in sorted(AUXILIARY_CHECKPOINTS.items())
    )
    return Provenance(
        git_commit=commit,
        worktree_dirty=worktree_dirty,
        checkpoint_sha256=checkpoint_sha256,
        dataset_index_sha256=dataset_index_sha256,
        stream_order_sha256=fingerprints[0],
        stream_content_sha256=fingerprints[1],
        auxiliary_checkpoint_sha256=auxiliary_hashes,
    )


def batch_spec_text(
    args: argparse.Namespace,
    plan: Sequence[PlannedJob],
    provenance: Provenance,
) -> str:
    suite_counts = Counter(job.config.suite for job in plan)
    lines = (
        ("batch_id", args.batch_id),
        ("suite", args.suite),
        ("expected_jobs", len(plan)),
        ("git_commit", provenance.git_commit),
        ("worktree_dirty", config_bool(provenance.worktree_dirty)),
        ("model", MODEL),
        ("source_setting", SOURCE_SETTING),
        ("eval_split", EVAL_SPLIT),
        ("method", "fstta"),
        ("seed", SEED),
        ("episodes", args.episodes),
        ("gpus", ",".join(args.gpus)),
        ("action_selection", ACTION_SELECTION or "native"),
        ("num_processes", 1),
        ("eval_use_ckpt_config", config_bool(False)),
        ("norm_scope", NORM_SCOPE),
        ("last_k_ln", LAST_K_LN),
        ("episodic", config_bool(EPISODIC)),
        ("steps", STEPS),
        ("reset_bn_stats", config_bool(RESET_BN_STATS)),
        ("rho", RHO),
        ("tau", TAU),
        ("a", A),
        ("b", B),
        ("fast_optimizer", FAST_OPTIMIZER),
        ("beta1", BETA1),
        ("beta2", BETA2),
        ("weight_decay", WEIGHT_DECAY),
        ("max_grad_norm", MAX_GRAD_NORM),
        (
            "reset_fast_optimizer_each_episode",
            config_bool(RESET_FAST_OPTIMIZER_EACH_EPISODE),
        ),
        ("eigen_eps", EIGEN_EPS),
        (
            "suite_counts",
            ",".join(
                "{}:{}".format(name, suite_counts[name])
                for name in SUITE_ORDER
                if suite_counts[name]
            ),
        ),
        ("checkpoint_sha256", provenance.checkpoint_sha256),
        ("dataset_index_sha256", provenance.dataset_index_sha256),
        ("stream_order_sha256", provenance.stream_order_sha256),
        ("stream_content_sha256", provenance.stream_content_sha256),
    )
    auxiliary_lines = tuple(
        ("auxiliary_{}_sha256".format(name), digest)
        for name, digest in provenance.auxiliary_checkpoint_sha256
    )
    return "".join(
        "{}={}\n".format(key, value) for key, value in lines + auxiliary_lines
    )


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def initialize_batch(
    args: argparse.Namespace,
    plan_csv: str,
    spec: str,
) -> Tuple[Path, Path]:
    log_root = LOG_BASE / args.batch_id
    jobs_root = log_root / "jobs"
    if args.resume:
        if not log_root.is_dir():
            raise RuntimeError("resume batch does not exist: {}".format(log_root))
        batch_path = log_root / "batch.env"
        plan_path = log_root / "grid.csv"
        if not batch_path.is_file() or not plan_path.is_file():
            raise RuntimeError("resume batch is missing batch.env or grid.csv")
        if batch_path.read_text(encoding="utf-8") != spec:
            raise RuntimeError(
                "resume arguments or immutable inputs do not match batch.env"
            )
        if plan_path.read_text(encoding="utf-8") != plan_csv:
            raise RuntimeError("resume arguments do not match grid.csv")
        jobs_root.mkdir(exist_ok=True)
        return log_root, jobs_root

    if log_root.exists():
        raise RuntimeError("batch already exists: {}".format(log_root))
    LOG_BASE.mkdir(parents=True, exist_ok=True)
    log_root.mkdir()
    jobs_root.mkdir()
    atomic_write(log_root / "batch.env", spec)
    atomic_write(log_root / "grid.csv", plan_csv)
    return log_root, jobs_root


class BatchLock:
    def __init__(self, log_root: Path):
        self.path = log_root / ".scheduler.lock"
        try:
            self.path.mkdir()
        except FileExistsError as error:
            raise RuntimeError(
                "batch is already running or has a stale lock: {}".format(self.path)
            ) from error
        self.owner = self.path / "owner"
        atomic_write(
            self.owner,
            "pid={}\nhost={}\nstarted_at={}\n".format(
                os.getpid(), os.uname().nodename, utc_now()
            ),
        )
        self.retained = False

    def retain(self, reason: str, exit_code: int) -> None:
        self.retained = True
        with self.owner.open("a", encoding="utf-8") as handle:
            handle.write(
                "status={}\nscheduler_exit_code={}\n".format(reason, exit_code)
            )

    def release(self) -> None:
        if self.retained:
            return
        try:
            self.owner.unlink()
            self.path.rmdir()
        except FileNotFoundError:
            pass


class SchedulerLogger:
    def __init__(self, path: Path):
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def command_for_job(job: PlannedJob, episodes: int) -> List[str]:
    config = job.config
    pairs = [
        ("NUM_PROCESSES", "1"),
        ("EVAL.USE_CKPT_CONFIG", config_bool(False)),
        ("TTA.LR", config.fast_lr),
        ("TTA.NORM_SCOPE", NORM_SCOPE),
        ("TTA.LAST_K_LN", str(LAST_K_LN)),
        ("TTA.EPISODIC", config_bool(EPISODIC)),
        ("TTA.STEPS", str(STEPS)),
        ("TTA.RESET_BN_STATS", config_bool(RESET_BN_STATS)),
        ("TTA.MAX_GRAD_NORM", MAX_GRAD_NORM),
        ("TTA.FSTTA.M", str(config.fast_window)),
        ("TTA.FSTTA.N", str(config.slow_window)),
        ("TTA.FSTTA.Q", config.q),
        ("TTA.FSTTA.LR_SLOW", config.slow_lr),
        ("TTA.FSTTA.RHO", RHO),
        ("TTA.FSTTA.TAU", TAU),
        ("TTA.FSTTA.A", A),
        ("TTA.FSTTA.B", B),
        ("TTA.FSTTA.USE_SLOW", config_bool(config.use_slow)),
        ("TTA.FSTTA.FAST_GRAD_MODE", config.fast_grad_mode),
        (
            "TTA.FSTTA.USE_FAST_LR_SCALER",
            config_bool(config.use_fast_lr_scaler),
        ),
        ("TTA.FSTTA.OPTIMIZER", FAST_OPTIMIZER),
        ("TTA.FSTTA.BETA1", BETA1),
        ("TTA.FSTTA.BETA2", BETA2),
        ("TTA.FSTTA.WEIGHT_DECAY", WEIGHT_DECAY),
        ("TTA.FSTTA.SLOW_OPTIMIZER", config.slow_optimizer),
        ("TTA.FSTTA.SLOW_MOMENTUM", config.slow_momentum),
        (
            "TTA.FSTTA.RESET_OPTIMIZER_EACH_EPISODE",
            config_bool(RESET_FAST_OPTIMIZER_EACH_EPISODE),
        ),
        (
            "TTA.FSTTA.RESET_SLOW_OPTIMIZER_EACH_WINDOW",
            config_bool(config.reset_slow_optimizer_each_window),
        ),
        ("TTA.FSTTA.EIGEN_EPS", EIGEN_EPS),
        ("TEST_EPISODE_COUNT", str(episodes)),
    ]
    if ACTION_SELECTION is not None:
        pairs.insert(2, ("EVAL.ACTION_SELECTION", ACTION_SELECTION))
    if EXPLICIT_EVAL_SPLIT:
        pairs.insert(2, ("EVAL.SPLIT", EVAL_SPLIT))
    command = [
        "bash",
        str(RUNNER),
        SOURCE_SETTING,
        "fstta",
        str(SEED),
    ]
    for key, value in pairs:
        command.extend((key, value))
    return command


def parameters_text(
    job: PlannedJob, episodes: int, provenance: Provenance
) -> str:
    config = job.config
    values = (
        ("job_id", job.job_id),
        ("suite_job_id", job.suite_job_id),
        ("suite", config.suite),
        ("run_tag", job.run_tag),
        ("gpu", job.gpu),
        ("model", MODEL),
        ("source_setting", SOURCE_SETTING),
        ("eval_split", EVAL_SPLIT),
        ("method", "fstta"),
        ("action_selection", ACTION_SELECTION or "native"),
        ("fast_lr", config.fast_lr),
        ("M", config.fast_window),
        ("slow_lr", config.slow_lr),
        ("N", config.slow_window),
        ("q", config.q),
        ("use_slow", config_bool(config.use_slow)),
        ("fast_grad_mode", config.fast_grad_mode),
        ("use_fast_lr_scaler", config_bool(config.use_fast_lr_scaler)),
        ("slow_optimizer", config.slow_optimizer),
        ("slow_momentum", config.slow_momentum),
        (
            "reset_slow_optimizer_each_window",
            config_bool(config.reset_slow_optimizer_each_window),
        ),
        ("seed", SEED),
        ("episodes", episodes),
        ("git_commit", provenance.git_commit),
        ("worktree_dirty", config_bool(provenance.worktree_dirty)),
        ("checkpoint_sha256", provenance.checkpoint_sha256),
        ("dataset_index_sha256", provenance.dataset_index_sha256),
        ("stream_order_sha256", provenance.stream_order_sha256),
        ("stream_content_sha256", provenance.stream_content_sha256),
    )
    auxiliary_values = tuple(
        ("auxiliary_{}_sha256".format(name), digest)
        for name, digest in provenance.auxiliary_checkpoint_sha256
    )
    return "".join(
        "{}={}\n".format(key, value)
        for key, value in values + auxiliary_values
    )


def read_status(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def find_manifest(console_path: Path) -> Optional[Path]:
    try:
        content = console_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(content.replace("\r", "\n").splitlines()):
        candidate = line.strip()
        if candidate.endswith("/manifest.json"):
            path = Path(candidate)
            if path.is_file():
                return path
    return None


def require_equal(
    diagnostics: Mapping[str, object], key: str, expected: object
) -> None:
    if key not in diagnostics or diagnostics[key] != expected:
        raise ValueError(
            "diagnostic {} expected {!r}, got {!r}".format(
                key, expected, diagnostics.get(key)
            )
        )


def require_float(
    diagnostics: Mapping[str, object], key: str, expected: float
) -> None:
    try:
        actual = float(diagnostics[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("diagnostic {} is missing or nonnumeric".format(key)) from error
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "diagnostic {} expected {}, got {}".format(key, expected, actual)
        )


def validate_diagnostics(
    manifest_path: Path, job: PlannedJob, episodes: int
) -> None:
    run_dir = manifest_path.resolve().parent
    stats_path = (
        run_dir
        / "raw"
        / "model"
        / "tb"
        / "{}_stats_{}.json".format(EVAL_SPLIT, SEED)
    )
    diagnostics_path = (
        run_dir / "raw" / "model" / "tb" / "tta_diagnostics_{}.json".format(SEED)
    )
    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    with diagnostics_path.open("r", encoding="utf-8") as handle:
        diagnostics = json.load(handle)
    if not isinstance(stats, dict) or len(stats) != episodes:
        raise ValueError(
            "episode statistics count is {}, expected {}".format(
                len(stats) if isinstance(stats, dict) else "non-dict", episodes
            )
        )
    if not isinstance(diagnostics, dict):
        raise ValueError("FSTTA diagnostics are not a JSON object")

    config = job.config
    require_equal(diagnostics, "episodes", episodes)
    require_float(diagnostics, "fast_lr", float(config.fast_lr))
    require_equal(diagnostics, "fast_window", config.fast_window)
    require_equal(diagnostics, "use_slow", config.use_slow)
    require_equal(diagnostics, "slow_window", config.slow_window)
    require_float(diagnostics, "slow_lr", float(config.slow_lr))
    require_float(diagnostics, "q", float(config.q))
    require_equal(diagnostics, "fast_optimizer", FAST_OPTIMIZER)
    require_equal(diagnostics, "slow_optimizer", config.slow_optimizer)
    require_equal(diagnostics, "fast_grad_mode", config.fast_grad_mode)
    require_equal(
        diagnostics, "use_fast_lr_scaler", config.use_fast_lr_scaler
    )
    require_float(diagnostics, "slow_momentum", float(config.slow_momentum))
    require_equal(
        diagnostics,
        "reset_slow_optimizer_each_window",
        config.reset_slow_optimizer_each_window,
    )

    slow_updates = int(diagnostics.get("slow_updates", -1))
    slow_skips = int(diagnostics.get("slow_skipped_updates", -1))
    slow_attempts = int(diagnostics.get("slow_attempts", -1))
    slow_pending = int(diagnostics.get("slow_pending_episodes", -1))
    if config.use_slow:
        expected_attempts = episodes // config.slow_window
        expected_pending = episodes % config.slow_window
        if slow_attempts != expected_attempts:
            raise ValueError("FSTTA slow-attempt count mismatch")
        if slow_pending != expected_pending:
            raise ValueError("FSTTA pending slow-window count mismatch")
        if slow_updates + slow_skips != expected_attempts:
            raise ValueError("FSTTA completed slow-attempt count mismatch")
    elif any(value != 0 for value in (slow_updates, slow_skips, slow_attempts, slow_pending)):
        raise ValueError("FAST-only control unexpectedly performed slow adaptation")

    expected_resets = (
        slow_attempts
        if config.use_slow and config.reset_slow_optimizer_each_window
        else 0
    )
    require_equal(diagnostics, "slow_optimizer_resets", expected_resets)
    names = diagnostics.get("adapted_parameter_names")
    expected_tensors = 2 * LAST_K_LN
    if (
        not isinstance(names, list)
        or len(names) != expected_tensors
        or len(set(names)) != expected_tensors
        or any(not isinstance(name, str) or not name for name in names)
    ):
        raise ValueError(
            "adapted_parameter_names do not describe {} unique tensors".format(
                expected_tensors
            )
        )


def validate_artifacts(
    manifest_path: Path,
    job: PlannedJob,
    episodes: int,
    provenance: Provenance,
) -> Tuple[bool, str]:
    if not manifest_path.is_file():
        return False, "manifest is missing: {}".format(manifest_path)
    command = (
        sys.executable,
        str(MANIFEST_VALIDATOR),
        "--manifest",
        str(manifest_path),
        "--run-tag",
        job.run_tag,
        "--model",
        MODEL,
        "--method",
        "fstta",
        "--source-setting",
        SOURCE_SETTING,
        "--seed",
        str(SEED),
        "--git-commit",
        provenance.git_commit,
        "--checkpoint-sha256",
        provenance.checkpoint_sha256,
        "--stream-order-sha256",
        provenance.stream_order_sha256,
        "--stream-content-sha256",
        provenance.stream_content_sha256,
    )
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, "manifest validation could not run: {}".format(error)
    validator_output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        return False, validator_output or "run-manifest validation failed"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, "run manifest could not be read: {}".format(error)
    expected_overrides = command_for_job(job, episodes)[5:]
    if manifest.get("config_overrides") != expected_overrides:
        return False, "run-manifest config_overrides do not match the planned job"
    if (
        manifest.get("dataset", {}).get("index_sha256")
        != provenance.dataset_index_sha256
    ):
        return False, "run-manifest dataset index SHA256 mismatch"
    actual_auxiliary = {
        item.get("name"): item.get("sha256")
        for item in manifest.get("auxiliary_checkpoints", [])
    }
    expected_auxiliary = dict(provenance.auxiliary_checkpoint_sha256)
    if actual_auxiliary != expected_auxiliary:
        return False, "run-manifest auxiliary checkpoint SHA256 mismatch"
    if sha256_file(DATASET) != provenance.dataset_index_sha256:
        return False, "dataset index changed while the batch was running"
    try:
        current_fingerprints = run_capture(
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
        ).split()
    except RuntimeError as error:
        return False, "cannot re-fingerprint completed episode stream: {}".format(
            error
        )
    expected_fingerprints = (
        provenance.stream_order_sha256,
        provenance.stream_content_sha256,
    )
    if tuple(current_fingerprints) != expected_fingerprints:
        return False, "episode stream changed while the batch was running"
    try:
        validate_diagnostics(manifest_path, job, episodes)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        return False, "artifact diagnostics validation failed: {}".format(error)
    return True, validator_output


def collect_job_evidence(
    manifest_path: Path,
    job: PlannedJob,
    job_dir: Path,
    episodes: int,
) -> None:
    """Copy compact diagnostics and aggregate raw per-episode metrics."""
    run_dir = manifest_path.resolve().parent
    stats_path = (
        run_dir
        / "raw"
        / "model"
        / "tb"
        / "{}_stats_{}.json".format(EVAL_SPLIT, SEED)
    )
    diagnostics_path = (
        run_dir
        / "raw"
        / "model"
        / "tb"
        / "tta_diagnostics_{}.json".format(SEED)
    )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(stats, dict) or len(stats) != episodes:
        raise ValueError("cannot aggregate incomplete per-episode statistics")

    metrics = {}
    for metric in METRICS:
        try:
            values = [float(record[metric]) for record in stats.values()]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid per-episode metric '{}'".format(metric)) from error
        value = sum(values) / len(values)
        if not math.isfinite(value):
            raise ValueError("non-finite aggregate metric '{}'".format(metric))
        metrics[metric] = value

    config = job.config
    payload = {
        "job_id": job.job_id,
        "run_tag": job.run_tag,
        "model": MODEL,
        "source_setting": SOURCE_SETTING,
        "eval_split": EVAL_SPLIT,
        "result_role": (
            "development_calibration"
            if EVAL_SPLIT == "train"
            else "evaluation"
        ),
        "method": "fstta",
        "seed": SEED,
        "episodes": episodes,
        "manifest": str(manifest_path),
        "configuration": {
            "fast_lr": config.fast_lr,
            "M": config.fast_window,
            "slow_lr": config.slow_lr,
            "N": config.slow_window,
            "q": config.q,
            "use_slow": config.use_slow,
            "fast_grad_mode": config.fast_grad_mode,
            "use_fast_lr_scaler": config.use_fast_lr_scaler,
            "slow_optimizer": config.slow_optimizer,
            "reset_slow_optimizer_each_window": (
                config.reset_slow_optimizer_each_window
            ),
        },
        "metrics": metrics,
    }
    atomic_write(
        job_dir / "metrics.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    temporary = job_dir / "diagnostics.json.tmp"
    shutil.copy2(str(diagnostics_path), str(temporary))
    os.replace(str(temporary), str(job_dir / "diagnostics.json"))


ATTEMPT_ARTIFACTS = {
    "exitcode": "exitcode.previous.{stamp}",
    "runner_exitcode": "runner_exitcode.previous.{stamp}",
    "validation": "validation.previous.{stamp}",
    "console.log": "console.previous.{stamp}.log",
    "manifest.path": "manifest.previous.{stamp}.path",
    "parameters.env": "parameters.previous.{stamp}.env",
    "metrics.json": "metrics.previous.{stamp}.json",
    "diagnostics.json": "diagnostics.previous.{stamp}.json",
}


def preserve_previous_attempt(job_dir: Path) -> None:
    if not any((job_dir / name).exists() for name in ATTEMPT_ARTIFACTS):
        return
    stamp = attempt_stamp()
    for source_name, destination_template in ATTEMPT_ARTIFACTS.items():
        source = job_dir / source_name
        if not source.exists():
            continue
        destination = job_dir / destination_template.format(stamp=stamp)
        counter = 1
        while destination.exists():
            destination = job_dir / (
                destination_template.format(stamp=stamp) + ".{}".format(counter)
            )
            counter += 1
        source.rename(destination)


class Scheduler:
    def __init__(
        self,
        args: argparse.Namespace,
        plan: Sequence[PlannedJob],
        jobs_root: Path,
        provenance: Provenance,
        logger: SchedulerLogger,
    ):
        self.args = args
        self.plan = plan
        self.jobs_root = jobs_root
        self.provenance = provenance
        self.logger = logger
        self.active: Dict[int, RunningJob] = {}
        self.active_by_gpu: Counter = Counter()
        self.stop_signal: Optional[int] = None
        self.launched = 0
        self.skipped = 0
        self.reap_failures = 0

    def request_stop(self, signum: int) -> None:
        self.stop_signal = signum

    def check_stop(self) -> None:
        if self.stop_signal is not None:
            raise InterruptedError(
                "scheduler received signal {}".format(self.stop_signal)
            )

    def completed_and_valid(self, job: PlannedJob, job_dir: Path) -> bool:
        if not self.args.resume:
            return False
        require_clean_repository(self.provenance)
        if read_status(job_dir / "exitcode") != "0":
            return False
        if read_status(job_dir / "validation") != "ok":
            return False
        manifest_text = read_status(job_dir / "manifest.path")
        if not manifest_text:
            return False
        manifest_path = Path(manifest_text)
        valid, _ = validate_artifacts(
            manifest_path, job, self.args.episodes, self.provenance
        )
        if not valid:
            return False
        try:
            collect_job_evidence(
                manifest_path, job, job_dir, self.args.episodes
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return False
        return True

    def prepare_queue(self) -> Deque[PlannedJob]:
        pending: Deque[PlannedJob] = deque()
        for job in self.plan:
            self.check_stop()
            job_dir = self.jobs_root / job.run_tag
            job_dir.mkdir(exist_ok=True)
            if self.completed_and_valid(job, job_dir):
                self.skipped += 1
                self.logger.log("{} skip validated tag={}".format(utc_now(), job.run_tag))
                continue
            preserve_previous_attempt(job_dir)
            pending.append(job)
        return pending

    def launch(self, job: PlannedJob) -> None:
        self.check_stop()
        require_clean_repository(self.provenance)
        job_dir = self.jobs_root / job.run_tag
        atomic_write(
            job_dir / "parameters.env",
            parameters_text(job, self.args.episodes, self.provenance),
        )
        console_path = job_dir / "console.log"
        console = console_path.open("w", encoding="utf-8", buffering=1)
        console.write("\n===== attempt {} =====\n".format(utc_now()))
        console.flush()

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
                "NAVTTA_STREAM_ORDER_SHA256": self.provenance.stream_order_sha256,
                "NAVTTA_STREAM_CONTENT_SHA256": self.provenance.stream_content_sha256,
            }
        )
        environment.update(RUNNER_ENV)
        try:
            process = subprocess.Popen(
                command_for_job(job, self.args.episodes),
                cwd=str(REPO_ROOT),
                env=environment,
                stdout=console,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            console.close()
            raise
        self.active[process.pid] = RunningJob(job, process, console, job_dir)
        self.active_by_gpu[job.gpu] += 1
        self.launched += 1
        self.logger.log(
            "{} launched job={} tag={} gpu={} active_on_gpu={} total_active={}".format(
                utc_now(),
                job.job_id,
                job.run_tag,
                job.gpu,
                self.active_by_gpu[job.gpu],
                len(self.active),
            )
        )

    def finalize(self, running: RunningJob, runner_status: int) -> int:
        running.console.close()
        manifest_path = find_manifest(running.job_dir / "console.log")
        if manifest_path is not None:
            atomic_write(running.job_dir / "manifest.path", str(manifest_path) + "\n")
        validation = "failed"
        composite_status = runner_status
        validation_detail = ""
        if runner_status == 0 and manifest_path is not None:
            try:
                require_clean_repository(self.provenance)
                valid, validation_detail = validate_artifacts(
                    manifest_path,
                    running.job,
                    self.args.episodes,
                    self.provenance,
                )
            except RuntimeError as error:
                valid = False
                validation_detail = "repository validation failed: {}".format(
                    error
                )
            if valid:
                try:
                    collect_job_evidence(
                        manifest_path,
                        running.job,
                        running.job_dir,
                        self.args.episodes,
                    )
                    validation = "ok"
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
                    validation_detail = "evidence collection failed: {}".format(error)
                    composite_status = 90
            else:
                composite_status = 90
        elif runner_status == 0:
            composite_status = 90
            validation_detail = "manifest path was not found in runner output"

        if validation_detail:
            with (running.job_dir / "console.log").open("a", encoding="utf-8") as handle:
                handle.write("\n[launcher validation]\n{}\n".format(validation_detail))
        if runner_status == 0 and validation != "ok":
            with (running.job_dir / "console.log").open("a", encoding="utf-8") as handle:
                handle.write("artifact validation failed\n")

        atomic_write(running.job_dir / "runner_exitcode", "{}\n".format(runner_status))
        atomic_write(running.job_dir / "validation", validation + "\n")
        atomic_write(running.job_dir / "exitcode", "{}\n".format(composite_status))
        return composite_status

    def reap(self) -> bool:
        reaped = False
        for pid, running in list(self.active.items()):
            runner_status = running.process.poll()
            if runner_status is None:
                continue
            reaped = True
            composite_status = self.finalize(running, runner_status)
            del self.active[pid]
            self.active_by_gpu[running.job.gpu] -= 1
            if composite_status != 0:
                self.reap_failures += 1
            self.logger.log(
                "{} finished tag={} runner_status={} status={} active={}".format(
                    utc_now(),
                    running.job.run_tag,
                    runner_status,
                    composite_status,
                    len(self.active),
                )
            )
        return reaped

    def run(self) -> None:
        pending = self.prepare_queue()
        while pending or self.active:
            self.check_stop()
            launched_this_pass = False
            for _ in range(len(pending)):
                job = pending.popleft()
                if self.active_by_gpu[job.gpu] < self.args.jobs_per_gpu:
                    self.launch(job)
                    launched_this_pass = True
                else:
                    pending.append(job)
            reaped = self.reap()
            if (pending or self.active) and not launched_this_pass and not reaped:
                time.sleep(1.0)

    def terminate_all(self) -> None:
        if not self.active:
            return
        self.logger.log(
            "{} terminating {} active worker(s)".format(utc_now(), len(self.active))
        )
        for running in self.active.values():
            if running.process.poll() is None:
                try:
                    os.killpg(running.process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if all(item.process.poll() is not None for item in self.active.values()):
                break
            time.sleep(0.2)
        for running in self.active.values():
            if running.process.poll() is None:
                try:
                    os.killpg(running.process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            try:
                running.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            running.console.close()
        self.active.clear()
        self.active_by_gpu.clear()


def write_batch_metrics(
    log_root: Path,
    plan: Sequence[PlannedJob],
    episodes: int,
    provenance: Provenance,
) -> None:
    fields = (
        "job_id",
        "run_tag",
        "gpu",
        "status",
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
        "fast_lr",
        "M",
        "slow_lr",
        "N",
        "q",
        "use_slow",
        "fast_grad_mode",
        "use_fast_lr_scaler",
        "manifest",
    ) + METRICS
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for job in plan:
        job_dir = log_root / "jobs" / job.run_tag
        status = read_status(job_dir / "exitcode") or "missing"
        payload = {}
        metrics_path = job_dir / "metrics.json"
        if metrics_path.is_file():
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
        config = job.config
        row = {
            "job_id": job.job_id,
            "run_tag": job.run_tag,
            "gpu": job.gpu,
            "status": status,
            "model": MODEL,
            "source_setting": SOURCE_SETTING,
            "eval_split": EVAL_SPLIT,
            "result_role": (
                "development_calibration"
                if EVAL_SPLIT == "train"
                else "evaluation"
            ),
            "method": "fstta",
            "seed": SEED,
            "episodes": episodes,
            "git_commit": provenance.git_commit,
            "checkpoint_sha256": provenance.checkpoint_sha256,
            "dataset_index_sha256": provenance.dataset_index_sha256,
            "stream_order_sha256": provenance.stream_order_sha256,
            "stream_content_sha256": provenance.stream_content_sha256,
            "fast_lr": config.fast_lr,
            "M": config.fast_window,
            "slow_lr": config.slow_lr,
            "N": config.slow_window,
            "q": config.q,
            "use_slow": config_bool(config.use_slow),
            "fast_grad_mode": config.fast_grad_mode,
            "use_fast_lr_scaler": config_bool(config.use_fast_lr_scaler),
            "manifest": payload.get("manifest", ""),
        }
        recorded_metrics = payload.get("metrics", {})
        for metric in METRICS:
            row[metric] = recorded_metrics.get(metric, "")
        writer.writerow(row)
    atomic_write(log_root / "metrics.csv", output.getvalue())


def summarize(
    log_root: Path,
    plan: Sequence[PlannedJob],
    scheduler: Scheduler,
    provenance: Provenance,
) -> Tuple[str, bool]:
    write_batch_metrics(
        log_root, plan, scheduler.args.episodes, provenance
    )
    successful = 0
    failed = 0
    missing = 0
    manifest_pointers = 0
    validated = 0
    for job in plan:
        job_dir = log_root / "jobs" / job.run_tag
        status = read_status(job_dir / "exitcode")
        if not status:
            missing += 1
        elif status == "0":
            successful += 1
        else:
            failed += 1
        if (job_dir / "manifest.path").is_file():
            manifest_pointers += 1
        if read_status(job_dir / "validation") == "ok":
            validated += 1
    complete = (
        successful == len(plan)
        and failed == 0
        and missing == 0
        and validated == len(plan)
    )
    values = (
        ("batch_id", log_root.name),
        ("git_commit", provenance.git_commit),
        ("worktree_dirty", config_bool(provenance.worktree_dirty)),
        ("completed_at", utc_now()),
        ("expected", len(plan)),
        ("launched_this_invocation", scheduler.launched),
        ("skipped_completed", scheduler.skipped),
        ("successful", successful),
        ("failed", failed),
        ("missing", missing),
        ("manifest_pointers", manifest_pointers),
        ("validated", validated),
        ("reap_failures", scheduler.reap_failures),
        ("checkpoint_sha256", provenance.checkpoint_sha256),
        ("dataset_index_sha256", provenance.dataset_index_sha256),
        ("stream_order_sha256", provenance.stream_order_sha256),
        ("stream_content_sha256", provenance.stream_content_sha256),
    )
    auxiliary_values = tuple(
        ("auxiliary_{}_sha256".format(name), digest)
        for name, digest in provenance.auxiliary_checkpoint_sha256
    )
    return (
        "".join(
            "{}={}\n".format(key, value)
            for key, value in values + auxiliary_values
        ),
        complete,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    plan = build_plan(args.suite, args.batch_id, args.gpus)
    csv_text = plan_text(plan, args.episodes)
    print_plan_summary(args, plan)
    if args.dry_run:
        sys.stdout.write(csv_text)
        print(
            "\nDry run complete: {} jobs, unique tags verified, nothing launched."
            .format(len(plan))
        )
        return 0

    provenance = load_provenance(args.allow_dirty, args.episodes)
    spec = batch_spec_text(args, plan, provenance)
    log_root, jobs_root = initialize_batch(args, csv_text, spec)
    lock = BatchLock(log_root)
    logger = SchedulerLogger(log_root / "scheduler.log")
    scheduler = Scheduler(args, plan, jobs_root, provenance, logger)
    old_handlers = {}
    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, lambda received, _frame, s=scheduler: s.request_stop(received))

    exit_code = 1
    try:
        logger.log("scheduler_started_at={}".format(utc_now()))
        logger.log("git_commit={}".format(provenance.git_commit))
        logger.log("worktree_dirty={}".format(config_bool(provenance.worktree_dirty)))
        logger.log("checkpoint_sha256={}".format(provenance.checkpoint_sha256))
        logger.log("dataset_index_sha256={}".format(provenance.dataset_index_sha256))
        logger.log("stream_order_sha256={}".format(provenance.stream_order_sha256))
        logger.log("stream_content_sha256={}".format(provenance.stream_content_sha256))
        for name, digest in provenance.auxiliary_checkpoint_sha256:
            logger.log("auxiliary_{}_sha256={}".format(name, digest))
        logger.log("resume={}".format(config_bool(args.resume)))
        logger.log("jobs_per_gpu={}".format(args.jobs_per_gpu))
        scheduler.run()
        summary, complete = summarize(log_root, plan, scheduler, provenance)
        atomic_write(log_root / "SUMMARY", summary)
        for line in summary.rstrip().splitlines():
            logger.log(line)
        if complete:
            logger.log("All {} FSTTA exploration jobs completed successfully.".format(len(plan)))
            exit_code = 0
        else:
            logger.log(
                "Batch incomplete; rerun with --batch-id {} --resume.".format(
                    args.batch_id
                )
            )
            exit_code = 1
    except (InterruptedError, KeyboardInterrupt) as error:
        scheduler.terminate_all()
        signum = scheduler.stop_signal
        exit_code = 128 + signum if signum is not None else 130
        reason = "signal_{}".format(signum) if signum is not None else "keyboard_interrupt"
        logger.log("scheduler interrupted: {}".format(error))
        lock.retain(reason, exit_code)
    except Exception as error:
        scheduler.terminate_all()
        logger.log("scheduler failed: {}".format(error))
        lock.retain("unexpected_scheduler_exit", 1)
        exit_code = 1
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        lock.release()
        logger.close()
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
