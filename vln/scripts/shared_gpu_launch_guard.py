"""Cross-scheduler GPU launch serialization for shared NavTTA campaigns.

The lock covers the resource snapshot, process creation, and a short settling
window so a second scheduler cannot inspect stale ``nvidia-smi`` state while
the first worker is still importing CUDA.  It is intentionally per GPU and
does not serialize already-running experiments.
"""

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import time

from joint_campaign_contract import process_identity_alive


DEFAULT_LOCK_ROOT = Path(
    os.environ.get(
        "NAVTTA_SHARED_GPU_LOCK_ROOT",
        "/root/autodl-tmp/tmp/navtta-shared-gpu-launch-locks",
    )
)


class ReservationLedgerError(RuntimeError):
    pass


def _nonnegative_number(value, *, integer=False):
    if isinstance(value, bool):
        return False
    if integer and not isinstance(value, int):
        return False
    if not integer and not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and float(value) >= 0.0


def _valid_process_binding(value):
    return (
        isinstance(value, dict)
        and type(value.get("pid")) is int
        and value["pid"] > 0
        and isinstance(value.get("start_token"), str)
        and bool(value["start_token"])
        and re.fullmatch(
            r"[0-9a-f]{64}", str(value.get("cmdline_sha256", ""))
        ) is not None
    )


class ReservationLedger:
    """Mutable reservation state; callers may use it only while holding flock."""

    def __init__(self, gpu, root):
        self.gpu = int(gpu)
        self.root = Path(root)
        self.path = self.root / "gpu-{}.reservations.json".format(self.gpu)
        self.document = self._load()
        self._prune()

    def _load(self):
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {
                "schema": "navtta.shared_gpu_reservations.v1",
                "gpu": self.gpu,
                "baseline_gpu_memory_mib": None,
                "baseline_cgroup_memory_gib": None,
                "reservations": {},
            }
        except (OSError, json.JSONDecodeError) as error:
            raise ReservationLedgerError(
                "cannot authenticate shared-GPU reservation ledger {}: {}"
                .format(self.path, error)
            )
        if not isinstance(value, dict):
            raise ReservationLedgerError("shared-GPU reservation ledger is not an object")
        if value.get("schema") != "navtta.shared_gpu_reservations.v1":
            raise ReservationLedgerError("shared-GPU reservation schema mismatch")
        if value.get("gpu") != self.gpu:
            raise ReservationLedgerError("shared-GPU reservation GPU mismatch")
        reservations = value.get("reservations")
        if not isinstance(reservations, dict):
            raise ReservationLedgerError("shared-GPU reservations must be an object")
        baseline_gpu = value.get("baseline_gpu_memory_mib")
        baseline_memory = value.get("baseline_cgroup_memory_gib")
        if reservations:
            if not _nonnegative_number(baseline_gpu, integer=True):
                raise ReservationLedgerError("invalid reservation GPU baseline")
            if not _nonnegative_number(baseline_memory):
                raise ReservationLedgerError("invalid reservation RAM baseline")
        elif baseline_gpu is not None or baseline_memory is not None:
            raise ReservationLedgerError("empty reservation ledger has a baseline")
        for token, record in reservations.items():
            if not isinstance(token, str) or not token or not isinstance(record, dict):
                raise ReservationLedgerError("malformed shared-GPU reservation")
            if not _nonnegative_number(record.get("gpu_memory_mib"), integer=True):
                raise ReservationLedgerError("invalid reserved GPU memory")
            if not _nonnegative_number(record.get("cgroup_memory_gib")):
                raise ReservationLedgerError("invalid reserved cgroup memory")
            owners = record.get("owners")
            if (
                not isinstance(owners, list)
                or not owners
                or not all(_valid_process_binding(owner) for owner in owners)
            ):
                raise ReservationLedgerError("invalid reservation process owners")
            if not isinstance(record.get("metadata", {}), dict):
                raise ReservationLedgerError("invalid reservation metadata")
            if not _nonnegative_number(record.get("created_at_unix")):
                raise ReservationLedgerError("invalid reservation timestamp")
        return value

    def _write(self):
        temporary = self.path.with_name(
            "{}.{}.tmp".format(self.path.name, os.getpid())
        )
        temporary.write_text(
            json.dumps(
                self.document, indent=2, sort_keys=True, allow_nan=False
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(self.path))

    def _prune(self):
        reservations = self.document["reservations"]
        stale = []
        for token, record in reservations.items():
            owners = record.get("owners", []) if isinstance(record, dict) else []
            if not any(process_identity_alive(owner) for owner in owners):
                stale.append(token)
        for token in stale:
            reservations.pop(token, None)
        if not reservations:
            self.document["baseline_gpu_memory_mib"] = None
            self.document["baseline_cgroup_memory_gib"] = None
        if stale:
            self._write()

    def snapshot(self, observed_gpu_memory_mib, observed_cgroup_memory_gib):
        reservations = self.document["reservations"]
        reserved_gpu = sum(
            int(item["gpu_memory_mib"]) for item in reservations.values()
        )
        reserved_memory = sum(
            float(item["cgroup_memory_gib"]) for item in reservations.values()
        )
        baseline_gpu = self.document.get("baseline_gpu_memory_mib")
        baseline_memory = self.document.get("baseline_cgroup_memory_gib")
        reserved_floor_gpu = (
            int(baseline_gpu) + reserved_gpu
            if baseline_gpu is not None else 0
        )
        reserved_floor_memory = (
            float(baseline_memory) + reserved_memory
            if baseline_memory is not None else 0.0
        )
        return {
            "active_reservations": len(reservations),
            "reserved_gpu_memory_mib": reserved_gpu,
            "reserved_cgroup_memory_gib": reserved_memory,
            "effective_gpu_memory_mib": max(
                int(observed_gpu_memory_mib), reserved_floor_gpu
            ),
            "effective_cgroup_memory_gib": max(
                float(observed_cgroup_memory_gib), reserved_floor_memory
            ),
        }

    def reserve(
        self,
        token,
        *,
        gpu_memory_mib,
        cgroup_memory_gib,
        observed_gpu_memory_mib,
        observed_cgroup_memory_gib,
        owner,
        metadata=None,
    ):
        token = str(token)
        if token in self.document["reservations"]:
            raise RuntimeError("duplicate shared-GPU reservation: {}".format(token))
        if not _valid_process_binding(owner):
            raise ReservationLedgerError("invalid reservation owner identity")
        if not _nonnegative_number(gpu_memory_mib, integer=True):
            raise ReservationLedgerError("invalid GPU reservation amount")
        if not _nonnegative_number(cgroup_memory_gib):
            raise ReservationLedgerError("invalid RAM reservation amount")
        if not self.document["reservations"]:
            self.document["baseline_gpu_memory_mib"] = int(
                observed_gpu_memory_mib
            )
            self.document["baseline_cgroup_memory_gib"] = float(
                observed_cgroup_memory_gib
            )
        self.document["reservations"][token] = {
            "gpu_memory_mib": int(gpu_memory_mib),
            "cgroup_memory_gib": float(cgroup_memory_gib),
            "owners": [owner],
            "metadata": dict(metadata or {}),
            "created_at_unix": time.time(),
        }
        self._write()

    def add_owner(self, token, owner):
        try:
            record = self.document["reservations"][str(token)]
        except KeyError:
            raise RuntimeError("unknown shared-GPU reservation: {}".format(token))
        if not _valid_process_binding(owner):
            raise ReservationLedgerError("invalid reservation owner identity")
        record.setdefault("owners", []).append(owner)
        self._write()

    def release(self, token):
        removed = self.document["reservations"].pop(str(token), None)
        if not self.document["reservations"]:
            self.document["baseline_gpu_memory_mib"] = None
            self.document["baseline_cgroup_memory_gib"] = None
        if removed is not None:
            self._write()
        return removed is not None


@contextmanager
def shared_gpu_launch_guard(gpu, settle_seconds=0.0, lock_root=None):
    """Hold the per-GPU launch lock and optionally wait for CUDA allocation."""
    root = Path(lock_root) if lock_root is not None else DEFAULT_LOCK_ROOT
    root.mkdir(parents=True, exist_ok=True)
    path = root / "gpu-{}.lock".format(int(gpu))
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield ReservationLedger(gpu, root)
            delay = float(settle_seconds)
            if delay > 0:
                time.sleep(delay)
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def release_shared_gpu_reservation(gpu, token, lock_root=None):
    """Release one reservation under the same per-GPU serialization lock."""
    with shared_gpu_launch_guard(gpu, lock_root=lock_root) as ledger:
        return ledger.release(token)
