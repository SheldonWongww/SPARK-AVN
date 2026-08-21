#!/usr/bin/env python3
"""Atomically launch the paired REVERIE and R2R-CE VLN campaigns."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

import run_reverie_frozen_transfer as reverie  # noqa: E402
import run_tta_hparam_search as r2r_ce  # noqa: E402
from joint_campaign_contract import (  # noqa: E402
    SCHEMA,
    JointLaunchError,
    campaign_lifetime_lock,
    process_identity,
    process_identity_alive,
)


DEFAULT_REVERIE_SPEC = (
    REPO_ROOT / "vln/experiments/reverie_r2r_frozen_transfer_v1.json"
)
DEFAULT_CE_SPEC = (
    REPO_ROOT / "vln/experiments/r2r_ce_small_hparam_search_v1.json"
)
JOINT_ROOT = REPO_ROOT / "vln/results/logs/joint_campaigns"


class UserError(RuntimeError):
    pass


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _git(*arguments):
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT)] + list(arguments), text=True
    ).strip()


def _validate_batch_id(value, label):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value or ""):
        raise UserError("invalid {}".format(label))


def _process_command_lines():
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "cmdline").read_bytes()
            except OSError:
                continue
            if raw:
                yield int(entry.name), raw.replace(b"\0", b" ").decode(
                    "utf-8", errors="replace"
                )
        return
    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,command="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return
    for line in output.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2 and fields[0].isdigit():
            yield int(fields[0]), fields[1]


def _live_batch_processes(*batch_ids):
    """Find schedulers or descendants whose argv is bound to these batches."""
    matches = []
    own_pid = os.getpid()
    markers = (
        "run_reverie_frozen_transfer.py",
        "run_tta_hparam_search.py",
        "run_source_eval.sh",
        "/worker.sh",
    )
    for pid, command in _process_command_lines():
        if pid == own_pid:
            continue
        if (
            any(marker in command for marker in markers)
            and any(batch_id in command for batch_id in batch_ids)
        ):
            matches.append({"pid": pid, "command": command})
    return matches


def _launch_identity(args, reverie_spec_path, ce_spec_path):
    return {
        "schema": SCHEMA,
        "joint_id": args.joint_id,
        "git_commit": _git("rev-parse", "HEAD"),
        "gpu": args.gpu,
        "concurrency_profile": "shared_gpu_with_r2r_ce",
        "reverie_worker_cap": 1,
        "r2r_ce_worker_cap": 1,
        "batch_ids": {
            "reverie": args.reverie_batch_id,
            "r2r_ce": args.r2r_ce_batch_id,
        },
        "specs": {
            "reverie": {
                "path": str(reverie_spec_path),
                "sha256": _sha256(reverie_spec_path),
            },
            "r2r_ce": {
                "path": str(ce_spec_path),
                "sha256": _sha256(ce_spec_path),
            },
        },
    }


def _validate_resume_identity(existing, expected):
    for key, value in expected.items():
        if existing.get(key) != value:
            raise UserError(
                "joint resume identity mismatch for {}".format(key)
            )
    live = [
        role for role, binding in existing.get("processes", {}).items()
        if process_identity_alive(binding)
    ]
    if live:
        raise UserError(
            "joint resume refused while prior schedulers are alive: {}".format(
                ", ".join(sorted(live))
            )
        )


def _write_launch_state(manifest_path, attempt_path, document):
    _atomic_json(manifest_path, document)
    _atomic_json(attempt_path, document)


def _terminate(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _validated_ack(document, role):
    binding = document["acks"][role]
    path = Path(binding["path"])
    try:
        ack = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    process = document["processes"][role]
    expected = {
        "schema": "navtta.vln_joint_campaign_ready.v1",
        "active_attempt": document["active_attempt"],
        "role": role,
        "batch_id": document["batch_ids"][role],
        "spec_sha256": document["specs"][role]["sha256"],
    }
    if any(ack.get(key) != value for key, value in expected.items()):
        raise UserError("invalid {} readiness ACK".format(role))
    ack_process = ack.get("process")
    if not isinstance(ack_process, dict):
        raise UserError("{} readiness ACK lacks process identity".format(role))
    for key in ("pid", "start_token", "cmdline_sha256"):
        if ack_process.get(key) != process.get(key):
            raise UserError("{} readiness ACK process mismatch".format(role))
    if not process_identity_alive(process):
        raise UserError("{} exited after readiness ACK".format(role))
    return ack


def _preflight(args, require_registry=True):
    for value, label in (
        (args.joint_id, "joint ID"),
        (args.reverie_batch_id, "REVERIE batch ID"),
        (args.r2r_ce_batch_id, "R2R-CE batch ID"),
    ):
        _validate_batch_id(value, label)
    if args.gpu < 0:
        raise UserError("GPU index must be nonnegative")
    reverie_spec_path = args.reverie_spec.resolve()
    ce_spec_path = args.r2r_ce_spec.resolve()
    rev_spec = reverie.load_spec(reverie_spec_path)
    ce_spec = r2r_ce.load_spec(ce_spec_path)
    if require_registry:
        reverie._validate_source_registry(
            rev_spec, require_metrics_artifacts=True
        )
        reverie.validate_r2r_registry(rev_spec)
    if ce_spec.get("budget") != {
        "screening_tta_jobs": 40,
        "full_val_seen_tta_jobs": 10,
        "source_execution_jobs": 0,
        "total_executed_jobs": 50,
    }:
        raise UserError("R2R-CE campaign is not the reviewed 40+10 plan")
    if (
        rev_spec["execution"]["default_concurrency_profile"]
        != "shared_gpu_with_r2r_ce"
        or rev_spec["execution"]["concurrency_profiles"]
        ["shared_gpu_with_r2r_ce"]["max_workers_by_method"]
        != {method: 1 for method in reverie.METHODS}
        or ce_spec["scheduler_defaults"]["max_workers"] != 1
        or ce_spec["scheduler_defaults"]["max_continuous_workers"] != 1
    ):
        raise UserError("joint worker caps must remain REVERIE=1 and R2R-CE=1")
    return reverie_spec_path, rev_spec, ce_spec_path, ce_spec


def _commands(args, manifest_path, reverie_spec_path, ce_spec_path):
    common = ["--gpu", str(args.gpu), "--confirm-reviewed",
              "--joint-launch-manifest", str(manifest_path)]
    rev = [
        sys.executable,
        str(SCRIPT_DIR / "run_reverie_frozen_transfer.py"),
        "--spec", str(reverie_spec_path),
        "--batch-id", args.reverie_batch_id,
        "--concurrency-profile", "shared_gpu_with_r2r_ce",
    ] + common
    ce = [
        sys.executable,
        str(SCRIPT_DIR / "run_tta_hparam_search.py"),
        "all",
        "--spec", str(ce_spec_path),
        "--batch-id", args.r2r_ce_batch_id,
    ] + common
    if args.resume:
        rev.append("--resume")
        ce.append("--resume")
    if args.retry_failed:
        rev.append("--retry-failed")
        ce.append("--retry-failed")
    return rev, ce


def _launch_locked(args):
    reverie_spec_path, _, ce_spec_path, _ = _preflight(args)
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise UserError("tracked worktree must be clean before joint launch")
    joint_dir = JOINT_ROOT / args.joint_id
    manifest_path = joint_dir / "launch.json"
    identity = _launch_identity(args, reverie_spec_path, ce_spec_path)
    existing = None
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise UserError("invalid existing joint manifest: {}".format(error))
    if joint_dir.exists() and any(joint_dir.iterdir()) and not args.resume:
        raise UserError("joint batch already exists; use --resume")
    if args.resume:
        if existing is None:
            raise UserError("joint resume requires an existing launch manifest")
        _validate_resume_identity(existing, identity)
    live_batch = _live_batch_processes(
        args.reverie_batch_id, args.r2r_ce_batch_id
    )
    if live_batch:
        raise UserError(
            "joint launch refused while prior batch processes are alive: {}"
            .format(
                "; ".join(
                    "pid={} {}".format(item["pid"], item["command"])
                    for item in live_batch
                )
            )
        )

    attempt = int(existing.get("active_attempt", -1)) + 1 if existing else 0
    attempt_path = joint_dir / "attempts" / "attempt-{:02d}.json".format(attempt)
    if attempt_path.exists():
        raise UserError("joint launch attempt record already exists")
    rev_command, ce_command = _commands(
        args, manifest_path, reverie_spec_path, ce_spec_path
    )
    document = {
        **identity,
        "status": "launching",
        "active_attempt": attempt,
        "commands": {"reverie": rev_command, "r2r_ce": ce_command},
        "pids": {},
        "processes": {},
        "acks": {
            role: {
                "path": str(
                    joint_dir / "acks" / "attempt-{:02d}-{}.json".format(
                        attempt, role
                    )
                )
            }
            for role in ("reverie", "r2r_ce")
        },
        "started_at_unix": time.time(),
    }
    _write_launch_state(manifest_path, attempt_path, document)
    joint_dir.mkdir(parents=True, exist_ok=True)
    rev_log = (joint_dir / "reverie_scheduler.log").open("ab", buffering=0)
    ce_log = (joint_dir / "r2r_ce_scheduler.log").open("ab", buffering=0)
    rev_process = None
    ce_process = None
    try:
        rev_process = subprocess.Popen(
            rev_command,
            cwd=str(REPO_ROOT),
            stdout=rev_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        ce_process = subprocess.Popen(
            ce_command,
            cwd=str(REPO_ROOT),
            stdout=ce_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception as error:
        if rev_process is not None:
            _terminate(rev_process)
        if ce_process is not None:
            _terminate(ce_process)
        document["status"] = "startup_failed"
        document["startup_error"] = "child spawn failed: {}".format(error)
        document["startup_exit_codes"] = {
            "reverie": rev_process.poll() if rev_process is not None else None,
            "r2r_ce": ce_process.poll() if ce_process is not None else None,
        }
        document["finished_at_unix"] = time.time()
        _write_launch_state(manifest_path, attempt_path, document)
        raise
    finally:
        rev_log.close()
        ce_log.close()
    document["pids"] = {
        "reverie": rev_process.pid,
        "r2r_ce": ce_process.pid,
    }
    document["processes"] = {
        "reverie": process_identity(rev_process.pid),
        "r2r_ce": process_identity(ce_process.pid),
    }
    if any(value is None for value in document["processes"].values()):
        _terminate(rev_process)
        _terminate(ce_process)
        document["status"] = "startup_failed"
        document["startup_error"] = "could not bind scheduler process identities"
        document["startup_exit_codes"] = {
            "reverie": rev_process.poll(),
            "r2r_ce": ce_process.poll(),
        }
        document["finished_at_unix"] = time.time()
        _write_launch_state(manifest_path, attempt_path, document)
        raise UserError("could not bind scheduler process identities")
    document["status"] = "children_spawned"
    _write_launch_state(manifest_path, attempt_path, document)

    deadline = time.monotonic() + args.startup_timeout
    acknowledgements = {}
    failure = None
    while time.monotonic() < deadline:
        for role, process in (
            ("reverie", rev_process), ("r2r_ce", ce_process)
        ):
            if process.poll() is not None and role not in acknowledgements:
                failure = "{} exited before readiness ACK with status {}".format(
                    role, process.returncode
                )
                break
            if role not in acknowledgements:
                try:
                    ack = _validated_ack(document, role)
                except UserError as error:
                    failure = str(error)
                    break
                if ack is not None:
                    acknowledgements[role] = ack
        if failure or len(acknowledgements) == 2:
            break
        time.sleep(0.1)
    if len(acknowledgements) != 2:
        failure = failure or "timed out waiting for both readiness ACKs"
        _terminate(rev_process)
        _terminate(ce_process)
        document["status"] = "startup_failed"
        document["startup_error"] = failure
        document["startup_exit_codes"] = {
            "reverie": rev_process.poll(),
            "r2r_ce": ce_process.poll(),
        }
        document["finished_at_unix"] = time.time()
        _write_launch_state(manifest_path, attempt_path, document)
        raise UserError(failure)

    document["status"] = "both_ready"
    document["ready_at_unix"] = time.time()
    document["ready_acks"] = acknowledgements
    _write_launch_state(manifest_path, attempt_path, document)
    print(json.dumps({
        "joint_manifest": str(manifest_path),
        "joint_attempt": attempt,
        "reverie_pid": rev_process.pid,
        "r2r_ce_pid": ce_process.pid,
    }, sort_keys=True))


def _launch(args):
    lock_path = JOINT_ROOT / ".locks" / "{}.lock".format(args.joint_id)
    try:
        with campaign_lifetime_lock(
            lock_path, role="joint_launcher", batch_id=args.joint_id
        ):
            return _launch_locked(args)
    except JointLaunchError as error:
        raise UserError(str(error))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-id", default="vln-reverie-r2rce-joint-v1-seed0")
    parser.add_argument(
        "--reverie-batch-id", default="vln-reverie-r2r-frozen-transfer-v1-seed0"
    )
    parser.add_argument(
        "--r2r-ce-batch-id", default="vln-r2r-ce-small-hparam-search-v1-seed0"
    )
    parser.add_argument("--reverie-spec", type=Path, default=DEFAULT_REVERIE_SPEC)
    parser.add_argument("--r2r-ce-spec", type=Path, default=DEFAULT_CE_SPEC)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    joint_dir = JOINT_ROOT / args.joint_id
    manifest_path = joint_dir / "launch.json"
    if args.status:
        if not manifest_path.is_file():
            print("joint campaign has not started")
            return 0
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        document["alive"] = {
            role: process_identity_alive(binding)
            for role, binding in document.get("processes", {}).items()
        }
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0
    reverie_spec_path, _, ce_spec_path, _ = _preflight(
        args, require_registry=not args.plan_only
    )
    if args.plan_only:
        rev, ce = _commands(
            args, manifest_path, reverie_spec_path, ce_spec_path
        )
        print(json.dumps({
            "source_jobs": 0,
            "reverie_tta_jobs": 15,
            "r2r_ce_screening_jobs": 40,
            "r2r_ce_full_jobs": 10,
            "parallel_required": True,
            "reverie_command": rev,
            "r2r_ce_command": ce,
        }, indent=2, sort_keys=True))
        return 0
    _launch(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (UserError, reverie.UserError, r2r_ce.UserError) as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
