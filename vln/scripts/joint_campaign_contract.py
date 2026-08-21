"""Fail-closed launch authorization shared by the two VLN campaigns."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


SCHEMA = "navtta.vln_joint_reverie_r2r_ce_launch.v1"


class JointLaunchError(RuntimeError):
    pass


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def process_identity(pid=None):
    """Return a PID identity that cannot silently survive PID reuse.

    Formal execution happens on Linux, where ``/proc`` gives both the process
    start tick and the exact argv bytes.  The ``ps`` fallback keeps local macOS
    planning/tests usable without weakening the Linux launch record.
    """
    try:
        pid = os.getpid() if pid is None else int(pid)
    except (TypeError, ValueError):
        return None
    proc = Path("/proc") / str(pid)
    try:
        raw_stat = (proc / "stat").read_text(encoding="utf-8")
        tail = raw_stat[raw_stat.rfind(")") + 2 :].split()
        start_token = "proc:{}".format(tail[19])
        raw_cmdline = (proc / "cmdline").read_bytes()
        if not raw_cmdline:
            raise OSError("empty /proc cmdline")
        argv = [
            value.decode("utf-8", errors="surrogateescape")
            for value in raw_cmdline.rstrip(b"\0").split(b"\0")
        ]
        source = "procfs"
    except (OSError, IndexError, ValueError):
        try:
            output = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "lstart=", "-o", "command="],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
        if not output:
            return None
        # ``lstart`` is the first five whitespace-separated fields.
        fields = output.split(None, 5)
        if len(fields) < 6:
            return None
        start_token = "ps:" + " ".join(fields[:5])
        argv = [fields[5]]
        raw_cmdline = fields[5].encode("utf-8", errors="surrogateescape")
        source = "ps"
    return {
        "pid": pid,
        "start_token": start_token,
        "cmdline_sha256": hashlib.sha256(raw_cmdline).hexdigest(),
        "argv": argv,
        "identity_source": source,
    }


def process_identity_alive(binding):
    if not isinstance(binding, dict):
        return False
    current = process_identity(binding.get("pid"))
    if current is None:
        return False
    return all(
        current.get(key) == binding.get(key)
        for key in ("pid", "start_token", "cmdline_sha256")
    )


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


@contextmanager
def campaign_lifetime_lock(path, *, role, batch_id):
    """Prevent two schedulers from mutating one campaign at the same time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise JointLaunchError(
                "{} scheduler lock is already held for {}".format(role, batch_id)
            )
        stream.seek(0)
        stream.truncate()
        json.dump(
            {
                "role": role,
                "batch_id": batch_id,
                "process": process_identity(),
            },
            stream,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        try:
            yield path
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate_joint_launch(
    manifest_path,
    *,
    repo_root,
    role,
    batch_id,
    gpu,
    spec_path,
    spec_sha256,
    timeout_seconds=60.0,
):
    """Wait for both detached schedulers, then authenticate this child PID."""
    if not manifest_path:
        raise JointLaunchError(
            "formal launch is restricted to the joint REVERIE/R2R-CE entrypoint"
        )
    repo_root = Path(repo_root).resolve()
    allowed_root = (repo_root / "vln/results/logs/joint_campaigns").resolve()
    path = Path(manifest_path).resolve()
    try:
        path.relative_to(allowed_root)
    except ValueError:
        raise JointLaunchError("joint launch manifest is outside its canonical root")

    deadline = time.monotonic() + float(timeout_seconds)
    document = None
    while time.monotonic() < deadline:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            document = None
        if isinstance(document, dict) and document.get("status") in (
            "children_spawned", "both_ready"
        ):
            break
        time.sleep(0.1)
    else:
        raise JointLaunchError("joint launcher did not authorize both campaigns")

    expected = {
        "schema": SCHEMA,
        "gpu": int(gpu),
        "concurrency_profile": "shared_gpu_with_r2r_ce",
        "reverie_worker_cap": 1,
        "r2r_ce_worker_cap": 1,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise JointLaunchError("joint launch {} mismatch".format(key))
    batches = document.get("batch_ids", {})
    specs = document.get("specs", {})
    if batches.get(role) != batch_id:
        raise JointLaunchError("joint launch batch identity mismatch")
    binding = specs.get(role, {})
    if (
        Path(str(binding.get("path", ""))).resolve() != Path(spec_path).resolve()
        or binding.get("sha256") != spec_sha256
    ):
        raise JointLaunchError("joint launch specification identity mismatch")
    processes = document.get("processes", {})
    own_process = processes.get(role)
    if not process_identity_alive(own_process) or own_process.get("pid") != os.getpid():
        raise JointLaunchError("current scheduler was not spawned by joint launcher")
    peer = "r2r_ce" if role == "reverie" else "reverie"
    if not process_identity_alive(processes.get(peer)):
        raise JointLaunchError("peer campaign was not alive at joint launch gate")
    return path, document


def write_ready_ack(manifest_path, launch_document, *, role, batch_id, spec_sha256):
    """Publish readiness only after the child owns its campaign lock/preflight."""
    manifest_path = Path(manifest_path).resolve()
    current = json.loads(manifest_path.read_text(encoding="utf-8"))
    attempt = launch_document.get("active_attempt")
    if current.get("active_attempt") != attempt:
        raise JointLaunchError("joint launch attempt changed before readiness ACK")
    process = current.get("processes", {}).get(role)
    if not process_identity_alive(process) or process.get("pid") != os.getpid():
        raise JointLaunchError("readiness ACK process identity mismatch")
    ack_binding = current.get("acks", {}).get(role, {})
    ack_path = Path(str(ack_binding.get("path", ""))).resolve()
    allowed_root = manifest_path.parent.resolve()
    try:
        ack_path.relative_to(allowed_root)
    except ValueError:
        raise JointLaunchError("readiness ACK path escapes joint campaign root")
    document = {
        "schema": "navtta.vln_joint_campaign_ready.v1",
        "active_attempt": attempt,
        "role": role,
        "batch_id": batch_id,
        "spec_sha256": spec_sha256,
        "process": process_identity(),
        "ready_at_unix": time.time(),
    }
    _atomic_json(ack_path, document)
    return ack_path


def wait_for_joint_release(
    manifest_path, launch_document, *, role, timeout_seconds=300.0
):
    """Hold the child behind a second barrier until both ACKs are accepted."""
    path = Path(manifest_path).resolve()
    attempt = launch_document.get("active_attempt")
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            document = None
        if isinstance(document, dict):
            if document.get("active_attempt") != attempt:
                raise JointLaunchError("joint launch attempt changed before release")
            if document.get("status") == "both_ready":
                process = document.get("processes", {}).get(role)
                if not process_identity_alive(process) or process.get("pid") != os.getpid():
                    raise JointLaunchError("joint release process identity mismatch")
                return document
            if document.get("status") == "startup_failed":
                raise JointLaunchError("peer campaign failed during joint startup")
        time.sleep(0.1)
    raise JointLaunchError("joint launcher did not release both ready campaigns")
