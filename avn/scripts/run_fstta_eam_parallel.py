#!/usr/bin/env python3
"""Run the ENMuS FSTTA and SMT+Audio EAM grids concurrently.

The two child schedulers keep independent batch directories, locks, manifests,
and resume state.  Static per-GPU quotas cap their combined concurrency, while
both grids still use all four physical GPUs.  A child failure is recorded but
does not terminate the sibling grid.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, TextIO, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
FSTTA_RUNNER = REPO_ROOT / "avn" / "scripts" / "run_fstta_enmus_grid.py"
EAM_RUNNER = REPO_ROOT / "avn" / "scripts" / "run_eam_intensity_grid.sh"
LOG_BASE = REPO_ROOT / "avn" / "results" / "logs" / "parallel_fstta_eam"
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class UserError(Exception):
    """An expected launcher error that should not print a traceback."""


@dataclass
class ChildProcess:
    name: str
    command: List[str]
    process: subprocess.Popen
    output: TextIO


class Logger:
    def __init__(self, path: Path):
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def utc_now(compact: bool = False) -> str:
    now = datetime.now(timezone.utc)
    if compact:
        return now.strftime("%Y%m%dT%H%M%SZ")
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def shell_join(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


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
                completed.returncode, detail or shell_join(command)
            )
        )
    return completed.stdout.strip()


def parse_gpus(value: str) -> Tuple[str, ...]:
    gpus = tuple(item.strip() for item in value.split(","))
    if len(gpus) != 4 or any(not re.fullmatch(r"[0-9]+", item) for item in gpus):
        raise argparse.ArgumentTypeError(
            "must contain exactly four comma-separated numeric GPU ids"
        )
    if len(set(gpus)) != 4:
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


def nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 48-job ENMuS FSTTA grid and 72-job SMT+Audio EAM grid "
            "concurrently with independent schedulers."
        )
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpus,
        default=parse_gpus("0,1,2,3"),
        metavar="LIST",
        help="exactly four physical GPU ids (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--fstta-jobs-per-gpu",
        type=positive_int,
        default=3,
        metavar="N",
        help="ENMuS FSTTA concurrent jobs per GPU (default: 3)",
    )
    parser.add_argument(
        "--eam-jobs-per-gpu",
        type=positive_int,
        default=2,
        metavar="N",
        help="SMT+Audio EAM concurrent jobs per GPU (default: 2)",
    )
    parser.add_argument(
        "--episodes",
        type=positive_int,
        default=2000,
        metavar="N",
        help="episodes per child job (default: 2000; maximum: 2000)",
    )
    parser.add_argument(
        "--group-id",
        help="stable joint-run id used to derive both child batch ids",
    )
    parser.add_argument(
        "--startup-delay",
        type=nonnegative_int,
        default=10,
        metavar="SECONDS",
        help="delay between the two child schedulers (default: 10)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume existing child batches; launch a missing child fresh",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run both child dry-runs without creating batch directories",
    )
    args = parser.parse_args(argv)
    if args.fstta_jobs_per_gpu > 16:
        parser.error("--fstta-jobs-per-gpu must not exceed 16")
    if args.eam_jobs_per_gpu > 8:
        parser.error("--eam-jobs-per-gpu must not exceed 8")
    if args.fstta_jobs_per_gpu + args.eam_jobs_per_gpu > 5:
        parser.error(
            "combined FSTTA+EAM concurrency must not exceed 5 jobs per GPU"
        )
    if args.episodes > 2000:
        parser.error("--episodes must not exceed 2000")
    if args.startup_delay > 300:
        parser.error("--startup-delay must not exceed 300 seconds")
    if args.group_id is None:
        if args.resume:
            parser.error("--resume requires an explicit --group-id")
        args.group_id = "avn-grids-{}".format(utc_now(compact=True))
    if not SAFE_ID.fullmatch(args.group_id) or len(args.group_id) > 56:
        parser.error(
            "--group-id must be at most 56 safe characters: letters, digits, ._-"
        )
    return args


def child_batch_ids(group_id: str) -> Tuple[str, str]:
    fstta = "fstta-enmus-grid-{}".format(group_id)
    eam = "eam-intensity-grid-{}".format(group_id)
    if len(fstta) > 96 or len(eam) > 96:
        raise UserError("derived child batch id is too long")
    return fstta, eam


def base_commands(
    args: argparse.Namespace,
    fstta_batch_id: str,
    eam_batch_id: str,
) -> Dict[str, List[str]]:
    gpu_csv = ",".join(args.gpus)
    return {
        "fstta": [
            sys.executable,
            str(FSTTA_RUNNER),
            "--gpus",
            gpu_csv,
            "--jobs-per-gpu",
            str(args.fstta_jobs_per_gpu),
            "--episodes",
            str(args.episodes),
            "--batch-id",
            fstta_batch_id,
            "--release-lock-on-interrupt",
        ],
        "eam": [
            "bash",
            str(EAM_RUNNER),
            "--gpus",
            gpu_csv,
            "--jobs-per-gpu",
            str(args.eam_jobs_per_gpu),
            "--episodes",
            str(args.episodes),
            "--batch-id",
            eam_batch_id,
        ],
    }


def print_plan(
    args: argparse.Namespace,
    commands: Dict[str, List[str]],
    fstta_batch_id: str,
    eam_batch_id: str,
) -> None:
    total = args.fstta_jobs_per_gpu + args.eam_jobs_per_gpu
    print("AVN parallel FSTTA + EAM grids")
    print("  repository:             {}".format(REPO_ROOT))
    print("  group id:               {}".format(args.group_id))
    print("  GPUs:                   {}".format(",".join(args.gpus)))
    print("  FSTTA slots/GPU:        {}".format(args.fstta_jobs_per_gpu))
    print("  EAM slots/GPU:          {}".format(args.eam_jobs_per_gpu))
    print("  combined slots/GPU:     {}".format(total))
    print("  combined active jobs:   {}".format(total * len(args.gpus)))
    print("  episodes/job:           {}".format(args.episodes))
    print("  FSTTA batch:            {}".format(fstta_batch_id))
    print("  EAM batch:              {}".format(eam_batch_id))
    print("  joint logs:             {}".format(LOG_BASE / args.group_id))
    print("  startup delay:          {}s".format(args.startup_delay))
    print("  FSTTA command:          {}".format(shell_join(commands["fstta"])))
    print("  EAM command:            {}".format(shell_join(commands["eam"])))


def dry_run(commands: Dict[str, List[str]]) -> int:
    sys.stdout.flush()
    statuses = []
    for name in ("fstta", "eam"):
        print("\n===== {} child dry-run =====".format(name.upper()))
        completed = subprocess.run(
            commands[name] + ["--dry-run"],
            cwd=str(REPO_ROOT),
            check=False,
        )
        statuses.append(completed.returncode)
    return 0 if all(status == 0 for status in statuses) else 1


def validate_repository() -> str:
    for path in (Path(__file__).resolve(), FSTTA_RUNNER, EAM_RUNNER):
        if not path.is_file():
            raise UserError("missing required launcher: {}".format(path))
    commit = run_capture(("git", "rev-parse", "HEAD"))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise UserError("invalid Git commit: {}".format(commit))
    status = run_capture(
        ("git", "status", "--porcelain", "--untracked-files=no")
    )
    if status:
        raise UserError(
            "tracked worktree changes detected; commit or restore them first"
        )
    for path in (Path(__file__).resolve(), FSTTA_RUNNER, EAM_RUNNER):
        relative = path.resolve().relative_to(REPO_ROOT.resolve())
        tracked = subprocess.run(
            ("git", "ls-files", "--error-unmatch", str(relative)),
            cwd=str(REPO_ROOT),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tracked.returncode != 0:
            raise UserError("launcher is not tracked by HEAD: {}".format(relative))
    active_patterns = (
        "avn/scripts/run_fstta_enmus_grid.py",
        "avn/scripts/run_eam_intensity_grid.sh",
    )
    for pattern in active_patterns:
        completed = subprocess.run(
            ("pgrep", "-af", pattern),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode not in (0, 1):
            raise UserError("cannot inspect active schedulers with pgrep")
        matches = [line for line in completed.stdout.splitlines() if line.strip()]
        if matches:
            raise UserError(
                "an existing child scheduler is already active:\n  "
                + "\n  ".join(matches)
            )
    return commit


def require_repository_unchanged(commit: str) -> None:
    current = run_capture(("git", "rev-parse", "HEAD"))
    if current != commit:
        raise UserError("repository HEAD changed while joint grids were running")
    status = run_capture(
        ("git", "status", "--porcelain", "--untracked-files=no")
    )
    if status:
        raise UserError("tracked worktree changed while joint grids were running")


def plan_text(
    args: argparse.Namespace,
    commit: str,
    commands: Dict[str, List[str]],
    fstta_batch_id: str,
    eam_batch_id: str,
) -> str:
    values = (
        ("group_id", args.group_id),
        ("git_commit", commit),
        ("gpus", ",".join(args.gpus)),
        ("episodes", args.episodes),
        ("fstta_jobs_per_gpu", args.fstta_jobs_per_gpu),
        ("eam_jobs_per_gpu", args.eam_jobs_per_gpu),
        (
            "combined_jobs_per_gpu",
            args.fstta_jobs_per_gpu + args.eam_jobs_per_gpu,
        ),
        ("fstta_batch_id", fstta_batch_id),
        ("eam_batch_id", eam_batch_id),
        ("fstta_command", shell_join(commands["fstta"])),
        ("eam_command", shell_join(commands["eam"])),
    )
    return "".join("{}={}\n".format(key, value) for key, value in values)


def initialize_group(
    args: argparse.Namespace,
    plan: str,
    fstta_batch_id: str,
    eam_batch_id: str,
) -> Path:
    group_dir = LOG_BASE / args.group_id
    fstta_dir = (
        REPO_ROOT
        / "avn"
        / "results"
        / "logs"
        / "fstta_enmus_grid"
        / fstta_batch_id
    )
    eam_dir = (
        REPO_ROOT
        / "avn"
        / "results"
        / "logs"
        / "eam_intensity_grid"
        / eam_batch_id
    )
    if args.resume:
        if not group_dir.is_dir():
            raise UserError("joint group does not exist: {}".format(group_dir))
        plan_path = group_dir / "plan.env"
        if not plan_path.is_file() or plan_path.read_text(encoding="utf-8") != plan:
            raise UserError("resume arguments or Git commit do not match plan.env")
        return group_dir

    if group_dir.exists():
        raise UserError("joint group already exists: {}".format(group_dir))
    existing = [str(path) for path in (fstta_dir, eam_dir) if path.exists()]
    if existing:
        raise UserError("child batch already exists:\n  " + "\n  ".join(existing))
    LOG_BASE.mkdir(parents=True, exist_ok=True)
    group_dir.mkdir()
    atomic_write(group_dir / "plan.env", plan)
    return group_dir


def acquire_lock_directory(lock: Path, description: str) -> Path:
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise UserError(
            "{} is already running or has a stale lock: {}".format(
                description, lock
            )
        ) from error
    try:
        atomic_write(
            lock / "owner",
            "pid={}\nhost={}\nstarted_at={}\n".format(
                os.getpid(), os.uname().nodename, utc_now()
            ),
        )
    except BaseException:
        try:
            lock.rmdir()
        except OSError:
            pass
        raise
    return lock


def acquire_lock(group_dir: Path) -> Path:
    return acquire_lock_directory(
        group_dir / ".orchestrator.lock", "joint group"
    )


def acquire_global_lock() -> Path:
    LOG_BASE.mkdir(parents=True, exist_ok=True)
    return acquire_lock_directory(
        LOG_BASE / ".global_orchestrator.lock", "parallel-grid orchestrator"
    )


def release_lock(lock: Path) -> None:
    try:
        (lock / "owner").unlink()
        lock.rmdir()
    except FileNotFoundError:
        pass


def child_log_path(group_dir: Path, name: str) -> Path:
    return group_dir / "{}_scheduler.stdout.log".format(name)


def launch_child(
    name: str,
    command: List[str],
    group_dir: Path,
    logger: Logger,
) -> ChildProcess:
    path = child_log_path(group_dir, name)
    output = path.open("a", encoding="utf-8", buffering=1)
    output.write("\n===== joint launch {} =====\n".format(utc_now()))
    output.write("command={}\n".format(shell_join(command)))
    output.flush()
    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        output.close()
        raise
    logger.log(
        "{} launched child={} pid={} command={}".format(
            utc_now(), name, process.pid, shell_join(command)
        )
    )
    return ChildProcess(name, command, process, output)


def terminate_children(children: Dict[str, ChildProcess], logger: Logger) -> None:
    active = [child for child in children.values() if child.process.poll() is None]
    if not active:
        return
    logger.log("{} terminating {} active child scheduler(s)".format(
        utc_now(), len(active)
    ))
    for child in active:
        try:
            os.killpg(child.process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if all(child.process.poll() is not None for child in active):
            break
        time.sleep(0.5)
    for child in active:
        if child.process.poll() is None:
            try:
                os.killpg(child.process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            child.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def child_command_for_invocation(
    name: str,
    base: List[str],
    args: argparse.Namespace,
    fstta_batch_id: str,
    eam_batch_id: str,
) -> List[str]:
    command = list(base)
    if not args.resume:
        return command
    batch_dir = (
        REPO_ROOT
        / "avn"
        / "results"
        / "logs"
        / ("fstta_enmus_grid" if name == "fstta" else "eam_intensity_grid")
        / (fstta_batch_id if name == "fstta" else eam_batch_id)
    )
    if batch_dir.is_dir():
        command.append("--resume")
    return command


def run_joint(
    args: argparse.Namespace,
    commit: str,
    group_dir: Path,
    commands: Dict[str, List[str]],
    fstta_batch_id: str,
    eam_batch_id: str,
) -> int:
    lock = acquire_lock(group_dir)
    try:
        logger = Logger(group_dir / "orchestrator.log")
    except BaseException:
        release_lock(lock)
        raise
    children: Dict[str, ChildProcess] = {}
    statuses: Dict[str, int] = {}
    stop_signal: List[Optional[int]] = [None]
    old_handlers = {}

    def request_stop(received: int, _frame: object) -> None:
        stop_signal[0] = received

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)

    try:
        logger.log("orchestrator_started_at={}".format(utc_now()))
        logger.log("group_id={}".format(args.group_id))
        logger.log("failure_policy=child_failure_does_not_terminate_sibling")

        for index, name in enumerate(("fstta", "eam")):
            if stop_signal[0] is not None:
                raise KeyboardInterrupt
            require_repository_unchanged(commit)
            command = child_command_for_invocation(
                name,
                commands[name],
                args,
                fstta_batch_id,
                eam_batch_id,
            )
            try:
                children[name] = launch_child(name, command, group_dir, logger)
            except OSError as error:
                statuses[name] = 127
                logger.log("{} child={} launch_failed={}".format(
                    utc_now(), name, error
                ))
            if index == 0 and args.startup_delay:
                deadline = time.monotonic() + args.startup_delay
                while time.monotonic() < deadline:
                    if stop_signal[0] is not None:
                        raise KeyboardInterrupt
                    time.sleep(min(0.5, deadline - time.monotonic()))

        next_repository_check = time.monotonic()
        while len(statuses) < 2:
            if stop_signal[0] is not None:
                raise KeyboardInterrupt
            if time.monotonic() >= next_repository_check:
                require_repository_unchanged(commit)
                next_repository_check = time.monotonic() + 10.0
            for name, child in children.items():
                if name in statuses:
                    continue
                status = child.process.poll()
                if status is None:
                    continue
                statuses[name] = status
                child.output.close()
                logger.log("{} child={} completed exitcode={}".format(
                    utc_now(), name, status
                ))
            if len(statuses) < 2:
                time.sleep(1.0)

        successful = all(statuses.get(name) == 0 for name in ("fstta", "eam"))
        summary = (
            "group_id={}\n"
            "completed_at={}\n"
            "fstta_exitcode={}\n"
            "eam_exitcode={}\n"
            "successful={}\n"
        ).format(
            args.group_id,
            utc_now(),
            statuses.get("fstta", 127),
            statuses.get("eam", 127),
            "True" if successful else "False",
        )
        atomic_write(group_dir / "SUMMARY", summary)
        for line in summary.rstrip().splitlines():
            logger.log(line)
        return 0 if successful else 1
    except KeyboardInterrupt:
        terminate_children(children, logger)
        signum = stop_signal[0]
        status = 128 + signum if signum is not None else 130
        logger.log("orchestrator_interrupted exitcode={}".format(status))
        return status
    except UserError as error:
        terminate_children(children, logger)
        logger.log("orchestrator_integrity_failure={}".format(error))
        return 2
    except Exception as error:
        terminate_children(children, logger)
        logger.log("orchestrator_unexpected_failure={}".format(error))
        return 1
    finally:
        for child in children.values():
            if not child.output.closed:
                child.output.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        logger.close()
        release_lock(lock)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    fstta_batch_id, eam_batch_id = child_batch_ids(args.group_id)
    commands = base_commands(args, fstta_batch_id, eam_batch_id)
    print_plan(args, commands, fstta_batch_id, eam_batch_id)
    if args.dry_run:
        return dry_run(commands)

    global_lock = acquire_global_lock()
    try:
        commit = validate_repository()
        plan = plan_text(
            args, commit, commands, fstta_batch_id, eam_batch_id
        )
        group_dir = initialize_group(
            args, plan, fstta_batch_id, eam_batch_id
        )
        return run_joint(
            args, commit, group_dir, commands, fstta_batch_id, eam_batch_id
        )
    finally:
        release_lock(global_lock)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
