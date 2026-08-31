#!/usr/bin/env python3
"""Build authenticated native-v1.2 R2R-CE matched Source controls.

This is a narrow prerequisite workflow for ETPNav and BEVBert.  It deliberately
does not edit the tracked v1.3 Source ledgers or the targeted-gap campaign
specification.  Successful runs emit two content-addressed *candidate* ledgers
for later review and promotion into ``vln/manifests``.
"""

from __future__ import print_function

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
from datetime import datetime, timezone


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
CREATE_MANIFEST = REPO_ROOT / "tools/create_run_manifest.py"
FINALIZE_MANIFEST = REPO_ROOT / "tools/finalize_run_manifest.py"
VALIDATE_MANIFEST = REPO_ROOT / "tools/validate_run_manifest.py"
PAIRING_CHECKER = REPO_ROOT / "vln/scripts/verify_bevbert_source_pairing.py"
ASSET_MANIFEST = REPO_ROOT / "vln/manifests/assets/eval_assets.json"
TRACKED_ENVIRONMENT_MANIFEST = (
    REPO_ROOT / "vln/manifests/environments/eval_environments.json"
)
ORDER_ROOT = REPO_ROOT / "vln/manifests/episode_order/r2r_ce_v1_2"

SCHEMA = "navtta.vln_r2r_ce_v1_2_source_control_workflow.v1"
LEDGER_SCHEMA = "navtta.vln_r2r_ce_v1_2_source_controls_candidate.v1"
BENCHMARK = "r2r_ce_v1_2_etpnav_bevbert"
PROTOCOL = "v1.2-native"
SPLITS = ("val_unseen", "val_seen")
EXPECTED_EPISODES = {"val_seen": 778, "val_unseen": 1839}
SETTINGS = ("etpnav-r2r-ce", "bevbert-r2r-ce")
MODEL = {
    "etpnav-r2r-ce": "etpnav",
    "bevbert-r2r-ce": "bevbert",
}
CHECKPOINT_ASSET = {
    "etpnav-r2r-ce": "etpnav_checkpoint",
    "bevbert-r2r-ce": "bevbert_ce_checkpoint",
}
CLIP_FILENAME = {
    "etpnav-r2r-ce": "ViT-B-32.pt",
    "bevbert-r2r-ce": "ViT-B-16.pt",
}
CONFIG = {
    "etpnav-r2r-ce": "vln/baselines/etpnav/run_r2r/iter_train.yaml",
    "bevbert-r2r-ce": (
        "vln/baselines/bevbert/bevbert_ce/run_r2r/iter_train.yaml"
    ),
}
DATASET_ASSET = {
    "val_seen": "etpnav_val_seen_bertidx",
    "val_unseen": "etpnav_val_unseen_bertidx",
}
GROUND_TRUTH_ASSET = {
    "val_seen": "etpnav_val_seen_gt",
    "val_unseen": "etpnav_val_unseen_gt",
}
COMMON_AUXILIARY_ASSETS = (
    ("waypoint_predictor", "etpnav_waypoint_predictor"),
    ("depth_encoder", "ce_ddppo_depth_encoder"),
)
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class WorkflowError(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(document):
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path, label="JSON"):
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WorkflowError("cannot read {} {}: {}".format(label, path, error))
    if not isinstance(value, dict):
        raise WorkflowError("{} must contain a JSON object: {}".format(label, path))
    return value


def atomic_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            document,
            stream,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        stream.write("\n")
    os.replace(str(temporary), str(path))


def relative_to_repo(path):
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(Path(path).resolve())


def git_output(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT)] + list(args),
            universal_newlines=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise WorkflowError("git {} failed: {}".format(" ".join(args), error))


def git_commit():
    return git_output("rev-parse", "HEAD")


def tracked_worktree_dirty():
    return bool(git_output("status", "--porcelain", "--untracked-files=no"))


def require_tracked(path, label):
    relative = relative_to_repo(path)
    completed = subprocess.run(
        [
            "git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch",
            "--", relative,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if completed.returncode != 0:
        raise WorkflowError(
            "formal execution requires tracked {}: {}".format(label, relative)
        )


def asset_index():
    manifest = read_json(ASSET_MANIFEST, "VLN asset manifest")
    if manifest.get("schema_version") != 1:
        raise WorkflowError("unsupported VLN asset manifest schema")
    assets = manifest.get("assets")
    if not isinstance(assets, list):
        raise WorkflowError("VLN asset manifest assets must be a list")
    result = {}
    for item in assets:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise WorkflowError("malformed VLN asset manifest record")
        if item["id"] in result:
            raise WorkflowError("duplicate asset id: {}".format(item["id"]))
        result[item["id"]] = item
    return result


def resolve_asset_path(record, vln_root):
    path_text = record.get("path")
    if not isinstance(path_text, str) or not path_text:
        raise WorkflowError("asset path is missing")
    path = Path(path_text)
    if not path.is_absolute():
        return REPO_ROOT / path
    name = path.name
    if name in ("ViT-B-16.pt", "ViT-B-32.pt"):
        return Path(vln_root) / "cache/clip" / name
    return path


def checked_asset(assets, asset_id, vln_root, cache=None):
    if asset_id not in assets:
        raise WorkflowError("asset manifest is missing {}".format(asset_id))
    record = assets[asset_id]
    expected_digest = record.get("sha256")
    expected_size = record.get("size")
    if not isinstance(expected_size, int) or expected_size <= 0:
        raise WorkflowError("{} has invalid size metadata".format(asset_id))
    if not isinstance(expected_digest, str) or not SHA256_RE.fullmatch(expected_digest):
        raise WorkflowError("{} has invalid SHA256 metadata".format(asset_id))
    path = resolve_asset_path(record, vln_root)
    if not path.is_file():
        raise WorkflowError("missing asset {}: {}".format(asset_id, path))
    if path.stat().st_size != expected_size:
        raise WorkflowError(
            "{} size mismatch: expected {}, got {}".format(
                asset_id, expected_size, path.stat().st_size
            )
        )
    key = str(path.resolve())
    actual_digest = cache.get(key) if cache is not None else None
    if actual_digest is None:
        actual_digest = sha256_file(path)
        if cache is not None:
            cache[key] = actual_digest
    if actual_digest != expected_digest:
        raise WorkflowError(
            "{} SHA256 mismatch: expected {}, got {}".format(
                asset_id, expected_digest, actual_digest
            )
        )
    return {
        "id": asset_id,
        "path": str(path.resolve()),
        "size": expected_size,
        "sha256": actual_digest,
    }


def order_binding(split, require_dataset=False):
    path = ORDER_ROOT / (split + ".json")
    document = read_json(path, "{} order manifest".format(split))
    expected = {
        "schema": "navtta.episode_order.v1",
        "benchmark": BENCHMARK,
        "split": split,
        "episode_count": EXPECTED_EPISODES[split],
    }
    for key, value in expected.items():
        if document.get(key) != value:
            raise WorkflowError(
                "{} order manifest {} mismatch: expected {!r}, got {!r}".format(
                    split, key, value, document.get(key)
                )
            )
    order_digest = document.get("order_sha256")
    dataset = document.get("dataset")
    if not isinstance(order_digest, str) or not SHA256_RE.fullmatch(order_digest):
        raise WorkflowError("{} order digest is invalid".format(split))
    if not isinstance(dataset, dict):
        raise WorkflowError("{} dataset binding is missing".format(split))
    dataset_digest = dataset.get("sha256")
    if not isinstance(dataset_digest, str) or not SHA256_RE.fullmatch(dataset_digest):
        raise WorkflowError("{} dataset digest is invalid".format(split))
    episodes = document.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != EXPECTED_EPISODES[split]:
        raise WorkflowError("{} order manifest episode list is incomplete".format(split))
    episode_ids = [str(item.get("episode_id")) for item in episodes]
    if len(set(episode_ids)) != len(episode_ids):
        raise WorkflowError("{} episode IDs are not unique".format(split))
    dataset_path = REPO_ROOT / dataset.get("path", "")
    if require_dataset:
        if not dataset_path.is_file():
            raise WorkflowError("missing {} dataset: {}".format(split, dataset_path))
        if sha256_file(dataset_path) != dataset_digest:
            raise WorkflowError("{} dataset SHA256 mismatch".format(split))
    return {
        "path": path.resolve(),
        "sha256": sha256_file(path),
        "order_sha256": order_digest,
        "dataset_path": dataset_path.resolve(),
        "dataset_sha256": dataset_digest,
        "episode_ids": episode_ids,
        "episode_count": EXPECTED_EPISODES[split],
    }


def batch_root(batch_id):
    return (
        REPO_ROOT / "vln/results/source" / batch_id
        / "_native_v1_2_source_controls"
    )


def plan_path(batch_id):
    return batch_root(batch_id) / "PLAN.json"


def build_plan(batch_id, gpus, created_at=None):
    if len(gpus) != 2 or len(set(gpus)) != 2:
        raise WorkflowError("--gpus requires exactly two distinct GPU IDs")
    commit = git_commit()
    orders = {}
    for split in SPLITS:
        binding = order_binding(split, require_dataset=False)
        orders[split] = {
            "path": relative_to_repo(binding["path"]),
            "sha256": binding["sha256"],
            "order_sha256": binding["order_sha256"],
            "dataset_path": relative_to_repo(binding["dataset_path"]),
            "dataset_sha256": binding["dataset_sha256"],
            "episodes": binding["episode_count"],
        }
    assets = asset_index()
    jobs = []
    for setting_index, setting in enumerate(SETTINGS):
        config_path = REPO_ROOT / CONFIG[setting]
        if not config_path.is_file():
            raise WorkflowError("missing tracked base config: {}".format(config_path))
        for split_index, split in enumerate(SPLITS):
            jobs.append({
                "ordinal": setting_index * len(SPLITS) + split_index,
                "key": "{}:source:{}".format(setting, split),
                "setting": setting,
                "model": MODEL[setting],
                "split": split,
                "gpu": gpus[setting_index],
                "seed": 0,
                "order_seed": 0,
                "data_version": PROTOCOL,
                "benchmark": BENCHMARK,
                "expected_episodes": EXPECTED_EPISODES[split],
                "checkpoint_asset": CHECKPOINT_ASSET[setting],
                "checkpoint_sha256": assets[CHECKPOINT_ASSET[setting]]["sha256"],
                "config_path": relative_to_repo(config_path),
                "config_sha256": sha256_file(config_path),
                "dataset_sha256": orders[split]["dataset_sha256"],
                "episode_order_sha256": orders[split]["order_sha256"],
                "run_tag_template": (
                    batch_id + "-" + setting + "-" + split + "-attempt-{attempt:03d}"
                ),
                "command_template": [
                    relative_to_repo(RUNNER), setting, split, str(gpus[setting_index]),
                    "--run-tag", "{run_tag}", "--ce-data-version", PROTOCOL,
                ],
            })
    document = {
        "schema": SCHEMA,
        "schema_version": 1,
        "batch_id": batch_id,
        "created_at": created_at or utc_now(),
        "git_commit": commit,
        "protocol": {
            "benchmark": BENCHMARK,
            "data_version": PROTOCOL,
            "method": "source",
            "model_seed": 0,
            "episode_order_seed": 0,
            "action_selection": "target_native_argmax",
            "split_order_within_model": list(SPLITS),
            "fresh_process_per_split": True,
            "cross_split_state_reuse": False,
        },
        "gpus": list(gpus),
        "runner": {
            "path": relative_to_repo(RUNNER),
            "sha256": sha256_file(RUNNER),
        },
        "workflow": {
            "path": relative_to_repo(SCRIPT_PATH),
            "sha256": sha256_file(SCRIPT_PATH),
        },
        "asset_manifest": {
            "path": relative_to_repo(ASSET_MANIFEST),
            "sha256": sha256_file(ASSET_MANIFEST),
        },
        "tracked_environment_manifest": {
            "path": relative_to_repo(TRACKED_ENVIRONMENT_MANIFEST),
            "sha256": sha256_file(TRACKED_ENVIRONMENT_MANIFEST),
        },
        "orders": orders,
        "jobs": jobs,
    }
    identity_payload = dict(document)
    document["plan_identity_sha256"] = sha256_bytes(canonical_bytes(identity_payload))
    return document


def validate_plan(document, batch_id, gpus):
    if document.get("schema") != SCHEMA or document.get("schema_version") != 1:
        raise WorkflowError("unsupported Source-control plan schema")
    recorded_identity = document.get("plan_identity_sha256")
    payload = dict(document)
    payload.pop("plan_identity_sha256", None)
    if recorded_identity != sha256_bytes(canonical_bytes(payload)):
        raise WorkflowError("PLAN.json identity mismatch")
    expected = build_plan(batch_id, gpus, created_at=document.get("created_at"))
    if canonical_bytes(expected) != canonical_bytes(document):
        raise WorkflowError(
            "existing plan differs from the current commit, runner, assets, or arguments"
        )
    return document


def ensure_plan(batch_id, gpus):
    path = plan_path(batch_id)
    if path.is_file():
        return validate_plan(read_json(path, "Source-control plan"), batch_id, gpus)
    document = build_plan(batch_id, gpus)
    atomic_json(path, document)
    return document


def job_root(root, job):
    return root / "jobs" / job["setting"] / job["split"]


def attempt_paths(root, job, attempt):
    directory = job_root(root, job) / "attempts" / "attempt-{:03d}".format(attempt)
    run_tag = job["run_tag_template"].format(attempt=attempt)
    run_id = "{}-{}-{}".format(run_tag, job["setting"], PROTOCOL)
    return {
        "directory": directory,
        "attempt": directory / "attempt.json",
        "invocation": directory / "invocation.json",
        "validation": directory / "validation.json",
        "launcher_log": directory / "launcher.log",
        "run_tag": run_tag,
        "run_id": run_id,
        "result_root": (
            REPO_ROOT / "vln/results/source" / run_tag / job["setting"] / job["split"]
        ),
        "manifest": REPO_ROOT / "vln/results/runs" / run_id / "manifest.json",
    }


def attempt_numbers(root, job):
    parent = job_root(root, job) / "attempts"
    values = []
    if parent.is_dir():
        for path in parent.iterdir():
            matched = re.fullmatch(r"attempt-([0-9]{3})", path.name)
            if matched and path.is_dir():
                values.append(int(matched.group(1)))
    return sorted(values)


def latest_attempt(root, job):
    values = attempt_numbers(root, job)
    if not values:
        return None, None
    paths = attempt_paths(root, job, values[-1])
    if not paths["attempt"].is_file():
        return values[-1], None
    return values[-1], read_json(paths["attempt"], "attempt state")


def process_identity(pid):
    stat_path = Path("/proc") / str(pid) / "stat"
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    try:
        stat = stat_path.read_text(encoding="utf-8").split()
        cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
        return {"pid": pid, "start_ticks": stat[21], "cmdline": cmdline}
    except (OSError, IndexError, UnicodeError):
        return None


def matching_runner_pids(run_tag):
    """Find same-host runner processes by the unique run tag (Linux only)."""
    matches = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return matches
    for child in proc_root.iterdir():
        if not child.name.isdigit() or int(child.name) == os.getpid():
            continue
        try:
            command = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            continue
        if str(RUNNER) in command and run_tag in command:
            matches.append(int(child.name))
    return sorted(matches)


def recorded_process_alive(state):
    process = state.get("process")
    if not isinstance(process, dict):
        return False
    if process.get("hostname") != socket.gethostname():
        return False
    current = process_identity(process.get("pid"))
    if current is None:
        return False
    return (
        current.get("start_ticks") == process.get("start_ticks")
        and state.get("run_tag", "") in current.get("cmdline", "")
        and str(RUNNER) in current.get("cmdline", "")
    )


def running_process_status(state):
    """Return a conservative lifecycle label for a recorded running process."""
    process = state.get("process")
    if not isinstance(process, dict):
        return "unverifiable_running"
    if process.get("hostname") != socket.gethostname():
        return "foreign_host_running"
    if not isinstance(process.get("pid"), int) or process.get("start_ticks") is None:
        return "unverifiable_running"
    return "running" if recorded_process_alive(state) else "stale_running"


def manifest_identity(document):
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        from run_manifest_identity import immutable_identity_sha256
    finally:
        sys.path.pop(0)
    return immutable_identity_sha256(document)


def artifact_matches(record):
    if not isinstance(record, dict):
        return False
    path = Path(record.get("path", ""))
    if not path.is_file() or path.stat().st_size != record.get("size"):
        return False
    return sha256_file(path) == record.get("sha256")


def ready_state_valid(state, job):
    if not isinstance(state, dict) or state.get("status") != "ready":
        return False
    manifest_path = Path(state.get("formal_manifest_path", ""))
    if not manifest_path.is_file():
        return False
    if sha256_file(manifest_path) != state.get("formal_manifest_sha256"):
        return False
    manifest = read_json(manifest_path, "formal manifest")
    if (
        manifest.get("status") != "completed"
        or manifest.get("exit_code") != 0
        or manifest.get("task") != "vln"
        or manifest.get("benchmark") != BENCHMARK
        or manifest.get("model") != job["model"]
        or manifest.get("method") != "source"
        or manifest.get("seed") != 0
        or manifest.get("checkpoint", {}).get("sha256")
        != job["checkpoint_sha256"]
        or manifest.get("dataset", {}).get("stream_content_sha256")
        != job["dataset_sha256"]
        or manifest.get("dataset", {}).get("stream_order_sha256")
        != job["episode_order_sha256"]
        or manifest.get("dataset", {}).get("version") != PROTOCOL
    ):
        return False
    config_records = [
        item for item in manifest.get("auxiliary_checkpoints", ())
        if isinstance(item, dict) and item.get("name") == "evaluation_config"
    ]
    if (
        len(config_records) != 1
        or config_records[0].get("sha256") != job.get("config_sha256")
    ):
        return False
    identity = manifest.get("immutable_identity_sha256")
    if not isinstance(identity, str) or manifest_identity(manifest) != identity:
        return False
    artifacts = manifest.get("result_artifacts")
    return isinstance(artifacts, list) and len(artifacts) >= 3 and all(
        artifact_matches(item) for item in artifacts
    )


def job_status(root, job):
    number, state = latest_attempt(root, job)
    if number is None:
        return "pending", None, None
    if state is None:
        paths = attempt_paths(root, job, number)
        status = (
            "unrecorded_live_process"
            if matching_runner_pids(paths["run_tag"])
            else "incomplete_attempt"
        )
        return status, number, None
    status = state.get("status", "unknown")
    if status == "running":
        status = running_process_status(state)
        if status == "stale_running" and matching_runner_pids(state.get("run_tag", "")):
            status = "unrecorded_live_process"
    elif status == "preparing" and matching_runner_pids(state.get("run_tag", "")):
        status = "unrecorded_live_process"
    elif status == "ready" and not ready_state_valid(state, job):
        status = "invalid_ready_evidence"
    return status, number, state


def runner_command(job, run_tag):
    return [
        str(RUNNER), job["setting"], job["split"], str(job["gpu"]),
        "--run-tag", run_tag, "--ce-data-version", PROTOCOL,
    ]


def source_invocation(job, paths, plan):
    command = runner_command(job, paths["run_tag"])
    return {
        "schema": "navtta.vln_source_invocation.v1",
        "batch_id": plan["batch_id"],
        "plan_identity_sha256": plan["plan_identity_sha256"],
        "git_commit": plan["git_commit"],
        "attempt": int(paths["directory"].name.split("-")[-1]),
        "run_id": paths["run_id"],
        "run_tag": paths["run_tag"],
        "setting": job["setting"],
        "model": job["model"],
        "benchmark": BENCHMARK,
        "split": job["split"],
        "data_version": PROTOCOL,
        "seed": 0,
        "order_seed": 0,
        "action_selection": "target_native_argmax",
        "command": command,
        "runner_sha256": plan["runner"]["sha256"],
        "result_root": str(paths["result_root"]),
    }


def environment_attestation(vln_root, root):
    python = Path(vln_root) / "envs/vlnce017/bin/python"
    if not python.is_file():
        raise WorkflowError("missing vlnce017 Python: {}".format(python))
    probe = r'''
from __future__ import print_function
import json, platform, sys
try:
    import importlib_metadata
except ImportError:
    import importlib.metadata as importlib_metadata
names = [
    "torch", "torchvision", "numpy", "transformers", "h5py",
    "protobuf", "habitat-sim", "gym", "tensorboard", "tensorboardX",
    "navtta-core", "msgpack-numpy", "fastdtw", "pytorch-transformers",
    "clip", "torch-scatter",
]
versions = {}
for name in names:
    try:
        versions[name] = importlib_metadata.version(name)
    except Exception as error:
        versions[name] = {"error": str(error)}
print(json.dumps({
    "python": sys.version.split()[0],
    "executable": sys.executable,
    "platform": platform.platform(),
    "packages": versions,
}, sort_keys=True))
'''
    completed = subprocess.run(
        [str(python), "-c", probe],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    if completed.returncode != 0:
        raise WorkflowError(
            "vlnce017 environment probe failed: {}".format(completed.stderr.strip())
        )
    try:
        observed = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise WorkflowError("cannot parse vlnce017 environment probe: {}".format(error))
    pip_check = subprocess.run(
        [str(python), "-m", "pip", "check"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        check=False,
    )
    if pip_check.returncode != 0:
        raise WorkflowError("vlnce017 pip check failed: {}".format(pip_check.stdout))
    tracked = read_json(TRACKED_ENVIRONMENT_MANIFEST, "environment manifest")
    expected = tracked.get("environments", {}).get("vlnce017", {})
    expected_packages = expected.get("verify_packages", {})
    mismatches = {}
    for name, expected_version in expected_packages.items():
        actual = observed["packages"].get(name)
        if actual != expected_version:
            mismatches[name] = {"expected": expected_version, "actual": actual}
    expected_python = str(expected.get("python", ""))
    actual_python = str(observed.get("python", ""))
    if expected_python and actual_python.split(".")[:2] != expected_python.split(".")[:2]:
        mismatches["python_major_minor"] = {
            "expected": expected_python,
            "actual": actual_python,
        }
    if mismatches:
        raise WorkflowError(
            "vlnce017 differs from the tracked environment contract: {}".format(
                json.dumps(mismatches, sort_keys=True)
            )
        )
    document = {
        "schema": "navtta.runtime_environment_attestation.v1",
        "environment": "vlnce017",
        "observed": observed,
        "pip_check": {"exit_code": 0, "output": pip_check.stdout.strip()},
        "tracked_manifest": {
            "path": relative_to_repo(TRACKED_ENVIRONMENT_MANIFEST),
            "size": TRACKED_ENVIRONMENT_MANIFEST.stat().st_size,
            "sha256": sha256_file(TRACKED_ENVIRONMENT_MANIFEST),
        },
        "verified_package_versions": expected_packages,
        "python_compatibility": "exact_major_minor",
    }
    path = root / "environment-attestation.json"
    if path.exists():
        current = read_json(path, "environment attestation")
        if canonical_bytes(current) != canonical_bytes(document):
            raise WorkflowError("environment attestation changed within the batch")
    else:
        atomic_json(path, document)
    return path, document


def preflight(plan, vln_root, root):
    if REPO_ROOT.resolve() != Path("/data1/wxy/code/NavTTA").resolve():
        raise WorkflowError(
            "run_source_eval.sh is server-bound; execute this workflow from "
            "/data1/wxy/code/NavTTA"
        )
    if tracked_worktree_dirty():
        raise WorkflowError("formal Source execution requires a clean tracked worktree")
    for path, label in (
        (SCRIPT_PATH, "Source-control workflow"),
        (RUNNER, "Source runner"),
        (CREATE_MANIFEST, "manifest creator"),
        (FINALIZE_MANIFEST, "manifest finalizer"),
        (VALIDATE_MANIFEST, "manifest validator"),
        (PAIRING_CHECKER, "native-v1.2 pairing verifier"),
        (ASSET_MANIFEST, "asset manifest"),
        (TRACKED_ENVIRONMENT_MANIFEST, "environment manifest"),
    ):
        require_tracked(path, label)
    for split in SPLITS:
        require_tracked(ORDER_ROOT / (split + ".json"), split + " order manifest")
    for setting in SETTINGS:
        require_tracked(REPO_ROOT / CONFIG[setting], setting + " base config")
    if git_commit() != plan["git_commit"]:
        raise WorkflowError("Git commit changed after planning")
    validate_plan(plan, plan["batch_id"], plan["gpus"])

    python = Path(vln_root) / "envs/vlnce017/bin/python"
    if not python.is_file():
        raise WorkflowError("missing vlnce017 Python: {}".format(python))
    pairing = subprocess.run(
        [
            str(python), str(PAIRING_CHECKER), "--repo-root", str(REPO_ROOT),
            "--clip-path", str(Path(vln_root) / "cache/clip/ViT-B-16.pt"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        check=False,
    )
    if pairing.returncode != 0:
        raise WorkflowError("native-v1.2 pairing check failed:\n{}".format(pairing.stdout))

    assets = asset_index()
    digest_cache = {}
    verified = []
    direct_ids = [
        "mp3d_habitat_archive",
        "etpnav_checkpoint", "bevbert_ce_checkpoint",
        "etpnav_waypoint_predictor", "ce_ddppo_depth_encoder",
        "openai_clip_vit_b32", "openai_clip_vit_b16",
        "etpnav_val_seen_bertidx", "etpnav_val_unseen_bertidx",
        "etpnav_val_seen_gt", "etpnav_val_unseen_gt",
    ]
    for asset_id in direct_ids:
        print("hashing/validating asset {}".format(asset_id), flush=True)
        verified.append(checked_asset(assets, asset_id, vln_root, digest_cache))
    for split in SPLITS:
        order_binding(split, require_dataset=True)

    environment_path, _ = environment_attestation(vln_root, root)
    dry_run_dir = root / "preflight/dry-run"
    dry_run_dir.mkdir(parents=True, exist_ok=True)
    dry_runs = []
    for job in plan["jobs"]:
        tag = "{}-plancheck-{}-{}".format(
            plan["batch_id"], job["setting"], job["split"]
        )
        command = runner_command(job, tag) + ["--dry-run"]
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            check=False,
        )
        path = dry_run_dir / (job["setting"] + "-" + job["split"] + ".log")
        path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise WorkflowError("Source runner dry-run failed: {}".format(path))
        if "R2R_VLNCE_v1-2" not in completed.stdout:
            raise WorkflowError("dry-run does not bind native v1.2: {}".format(path))
        dry_runs.append({
            "job": job["key"],
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    report = {
        "schema": "navtta.vln_r2r_ce_v1_2_source_preflight.v1",
        "batch_id": plan["batch_id"],
        "git_commit": plan["git_commit"],
        "plan_identity_sha256": plan["plan_identity_sha256"],
        "verified_at": utc_now(),
        "pairing_check": pairing.stdout.strip(),
        "verified_assets": verified,
        "environment_attestation": {
            "path": str(environment_path),
            "sha256": sha256_file(environment_path),
        },
        "dry_runs": dry_runs,
    }
    atomic_json(root / "preflight/report.json", report)
    return environment_path


def create_formal_manifest(job, paths, plan, vln_root, environment_path):
    assets = asset_index()
    checkpoint = resolve_asset_path(assets[CHECKPOINT_ASSET[job["setting"]]], vln_root)
    dataset = resolve_asset_path(assets[DATASET_ASSET[job["split"]]], vln_root)
    ground_truth = resolve_asset_path(
        assets[GROUND_TRUTH_ASSET[job["split"]]], vln_root
    )
    order = order_binding(job["split"], require_dataset=False)
    auxiliary = []
    for name, asset_id in COMMON_AUXILIARY_ASSETS:
        auxiliary.append(
            "{}={}".format(name, resolve_asset_path(assets[asset_id], vln_root))
        )
    auxiliary.extend([
        "evaluation_config={}".format(REPO_ROOT / job["config_path"]),
        "clip_encoder={}".format(
            Path(vln_root) / "cache/clip" / CLIP_FILENAME[job["setting"]]
        ),
        "evaluator_ground_truth={}".format(ground_truth),
    ])
    command = runner_command(job, paths["run_tag"])
    args = [
        str(Path(vln_root) / "envs/vlnce017/bin/python"),
        str(CREATE_MANIFEST),
        "--output", str(paths["manifest"]),
        "--run-id", paths["run_id"],
        "--task", "vln",
        "--benchmark", BENCHMARK,
        "--model", job["model"],
        "--method", "source",
        "--run-tag", paths["run_tag"],
        "--source-setting", "{}:{}:{}".format(
            job["setting"], job["split"], PROTOCOL
        ),
        "--seed", "0",
        "--config", job["config_path"],
        "--checkpoint", str(checkpoint),
        "--dataset", str(dataset),
        "--dataset-version", PROTOCOL,
        "--asset-manifest", str(ASSET_MANIFEST),
        "--environment-manifest", str(environment_path),
        "--episode-order-manifest", str(order["path"]),
        "--stream-order-sha256", order["order_sha256"],
        "--stream-content-sha256", order["dataset_sha256"],
    ]
    for item in auxiliary:
        args.extend(["--aux-checkpoint", item])
    args.append("--extra")
    args.extend(command)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    completed = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise WorkflowError("manifest creation failed: {}".format(completed.stdout))
    manifest = read_json(paths["manifest"], "pre-run formal manifest")
    checks = {
        "run_id": paths["run_id"], "task": "vln", "benchmark": BENCHMARK,
        "model": job["model"], "method": "source", "seed": 0,
        "git_commit": plan["git_commit"], "status": "running",
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise WorkflowError("pre-run formal manifest {} mismatch".format(key))
    if manifest.get("dataset", {}).get("version") != PROTOCOL:
        raise WorkflowError("pre-run formal manifest dataset version mismatch")
    config_records = [
        item for item in manifest.get("auxiliary_checkpoints", ())
        if isinstance(item, dict) and item.get("name") == "evaluation_config"
    ]
    if (
        len(config_records) != 1
        or config_records[0].get("sha256") != job["config_sha256"]
    ):
        raise WorkflowError("pre-run formal manifest config binding mismatch")


def finalize_formal_manifest(paths, exit_code, artifacts=()):
    args = [
        sys.executable, str(FINALIZE_MANIFEST),
        "--manifest", str(paths["manifest"]), "--exit-code", str(exit_code),
    ]
    for name, path in artifacts:
        args.extend(["--artifact", "{}={}".format(name, path)])
    completed = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        check=False,
    )
    if completed.returncode != 0:
        raise WorkflowError("manifest finalization failed: {}".format(completed.stdout))


def locate_result_artifacts(result_root, split):
    aggregate = list(Path(result_root).glob(
        "metrics/source_{}/stats_ckpt_*_{}.json".format(split, split)
    ))
    per_episode = list(Path(result_root).glob(
        "metrics/source_{}/stats_ep_ckpt_*_{}_r0_w1.json".format(split, split)
    ))
    if len(aggregate) != 1 or len(per_episode) != 1:
        raise WorkflowError(
            "expected exactly one aggregate and per-episode artifact; got {} and {}"
            .format(len(aggregate), len(per_episode))
        )
    return aggregate[0], per_episode[0]


def finite_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_result(job, paths):
    aggregate_path, per_episode_path = locate_result_artifacts(
        paths["result_root"], job["split"]
    )
    aggregate = read_json(aggregate_path, "aggregate metrics")
    per_episode = read_json(per_episode_path, "per-episode metrics")
    order = order_binding(job["split"], require_dataset=False)
    observed_ids = list(per_episode)
    if observed_ids != order["episode_ids"]:
        raise WorkflowError(
            "per-episode result IDs/order differ from the canonical manifest"
        )
    if len(per_episode) != job["expected_episodes"]:
        raise WorkflowError("per-episode result count is incomplete")
    if not aggregate:
        raise WorkflowError("aggregate metrics are empty")
    for name in ("success", "spl"):
        if name not in aggregate or not finite_number(aggregate[name]):
            raise WorkflowError("aggregate metric {} is missing/non-finite".format(name))
        if not 0.0 <= float(aggregate[name]) <= 1.0:
            raise WorkflowError("aggregate metric {} is outside [0, 1]".format(name))
    recomputed = {}
    maximum_delta = 0.0
    for name, value in aggregate.items():
        if not finite_number(value):
            raise WorkflowError("aggregate metric {} is non-finite".format(name))
        episode_values = []
        for episode_id, metrics in per_episode.items():
            if not isinstance(metrics, dict) or not finite_number(metrics.get(name)):
                raise WorkflowError(
                    "episode {} metric {} is missing/non-finite".format(
                        episode_id, name
                    )
                )
            episode_values.append(float(metrics[name]))
        mean = math.fsum(episode_values) / len(episode_values)
        recomputed[name] = mean
        maximum_delta = max(maximum_delta, abs(mean - float(value)))
    if maximum_delta > 1e-9:
        raise WorkflowError(
            "aggregate/per-episode metric mismatch (max delta {})".format(
                maximum_delta
            )
        )
    for episode_id, metrics in per_episode.items():
        success = metrics.get("success")
        spl = metrics.get("spl")
        if float(success) not in (0.0, 1.0):
            raise WorkflowError("episode {} success is not binary".format(episode_id))
        if not 0.0 <= float(spl) <= 1.0:
            raise WorkflowError("episode {} SPL is outside [0, 1]".format(episode_id))

    console = paths["result_root"] / "console.log"
    if not console.is_file():
        raise WorkflowError("Source console log is missing")
    text = console.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    if re.search(r"R2R_VLNCE_v1[-_]2", text) is None:
        raise WorkflowError("Source console does not prove native v1.2 input")
    counts = re.findall(r"Episodes evaluated:\s*([0-9.]+)", text)
    if not counts or int(float(counts[-1])) != job["expected_episodes"]:
        raise WorkflowError("Source console episode count is missing or incomplete")

    report = {
        "schema": "navtta.vln_r2r_ce_source_result_validation.v1",
        "run_id": paths["run_id"],
        "run_tag": paths["run_tag"],
        "setting": job["setting"],
        "model": job["model"],
        "benchmark": BENCHMARK,
        "data_version": PROTOCOL,
        "split": job["split"],
        "seed": 0,
        "order_seed": 0,
        "action_selection": "target_native_argmax",
        "episode_count": len(per_episode),
        "episode_order_sha256": order["order_sha256"],
        "dataset_sha256": order["dataset_sha256"],
        "ordered_episode_ids_match": True,
        "aggregate_recomputed_from_per_episode": True,
        "maximum_aggregate_delta": maximum_delta,
        "metrics": {
            "SR": float(aggregate["success"]) * 100.0,
            "SPL": float(aggregate["spl"]) * 100.0,
        },
        "aggregate_artifact": {
            "path": str(aggregate_path.resolve()),
            "size": aggregate_path.stat().st_size,
            "sha256": sha256_file(aggregate_path),
        },
        "per_episode_artifact": {
            "path": str(per_episode_path.resolve()),
            "size": per_episode_path.stat().st_size,
            "sha256": sha256_file(per_episode_path),
        },
        "console_evidence": {
            "path": str(console.resolve()),
            "size": console.stat().st_size,
            "sha256": sha256_file(console),
            "protocol_marker": "R2R_VLNCE_v1-2",
            "reported_episode_count": job["expected_episodes"],
        },
    }
    atomic_json(paths["validation"], report)
    return report, aggregate_path, per_episode_path


def authenticate_formal_manifest(job, paths, plan):
    order = order_binding(job["split"], require_dataset=False)
    completed = subprocess.run(
        [
            sys.executable, str(VALIDATE_MANIFEST),
            "--manifest", str(paths["manifest"]),
            "--task", "vln", "--benchmark", BENCHMARK,
            "--run-tag", paths["run_tag"], "--model", job["model"],
            "--method", "source",
            "--source-setting", "{}:{}:{}".format(
                job["setting"], job["split"], PROTOCOL
            ),
            "--seed", "0", "--git-commit", plan["git_commit"],
            "--checkpoint-sha256", job["checkpoint_sha256"],
            "--stream-order-sha256", order["order_sha256"],
            "--stream-content-sha256", order["dataset_sha256"],
            "--require-immutable-identity", "--require-result-artifacts",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        check=False,
    )
    if completed.returncode != 0:
        raise WorkflowError("formal manifest authentication failed: {}".format(
            completed.stdout
        ))
    manifest = read_json(paths["manifest"], "formal manifest")
    if manifest.get("dataset", {}).get("version") != PROTOCOL:
        raise WorkflowError("formal manifest dataset version is not v1.2-native")
    config_records = [
        item for item in manifest.get("auxiliary_checkpoints", ())
        if isinstance(item, dict) and item.get("name") == "evaluation_config"
    ]
    if (
        len(config_records) != 1
        or config_records[0].get("sha256") != job["config_sha256"]
    ):
        raise WorkflowError("formal manifest config digest mismatch")
    return manifest


ACTIVE_PROCESSES = {}
ACTIVE_PROCESSES_LOCK = threading.Lock()
STOP_EVENT = threading.Event()


def stop_active_processes(signum=None, frame=None):
    del frame
    STOP_EVENT.set()
    with ACTIVE_PROCESSES_LOCK:
        processes = list(ACTIVE_PROCESSES.values())
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            pass
    if signum is not None:
        print("received signal {}; terminating active Source jobs".format(signum), flush=True)


def mark_stale_attempt(paths, state):
    state = dict(state)
    state["status"] = "interrupted"
    state["completed_at"] = utc_now()
    state["exit_code"] = 130
    state["recovery_note"] = "previous recorded process is no longer live"
    if paths["manifest"].is_file():
        manifest = read_json(paths["manifest"], "stale formal manifest")
        if manifest.get("status") == "running":
            finalize_formal_manifest(paths, 130)
    atomic_json(paths["attempt"], state)


def execute_job(job, plan, root, vln_root, environment_path):
    status, prior_number, prior_state = job_status(root, job)
    if status == "ready":
        print("[reuse-ready] {}".format(job["key"]), flush=True)
        return True
    if status in (
        "running", "foreign_host_running", "unverifiable_running",
        "unrecorded_live_process",
    ):
        raise WorkflowError(
            "job has a live or unverifiable owner ({}): {}".format(
                status, job["key"]
            )
        )
    if status in ("stale_running", "preparing", "incomplete_attempt"):
        old_paths = attempt_paths(root, job, prior_number)
        stale_state = prior_state or {
            "schema": "navtta.vln_source_control_attempt.v1",
            "batch_id": plan["batch_id"],
            "job_key": job["key"],
            "attempt": prior_number,
            "run_id": old_paths["run_id"],
            "run_tag": old_paths["run_tag"],
            "result_root": str(old_paths["result_root"]),
            "formal_manifest_path": str(old_paths["manifest"]),
        }
        mark_stale_attempt(old_paths, stale_state)
        status = "interrupted"
    # Source controls are provenance gates rather than search candidates.
    # A failed/invalid/interrupted attempt is never silently reclassified as
    # infrastructure and retried under the same batch.  Preserve that batch
    # intact and use a fresh batch ID after diagnosing the failure.
    if status != "pending":
        raise WorkflowError(
            "job has terminal persisted status {} and is not retryable in "
            "the same batch; preserve it and use a fresh batch id: {}".format(
                status, job["key"],
            )
        )
    attempt = 1
    paths = attempt_paths(root, job, attempt)
    if paths["directory"].exists():
        raise WorkflowError("refusing to overwrite attempt: {}".format(paths["directory"]))
    if paths["result_root"].exists():
        raise WorkflowError("refusing to overwrite result root: {}".format(paths["result_root"]))
    if paths["manifest"].exists():
        raise WorkflowError("refusing to overwrite formal manifest: {}".format(paths["manifest"]))
    paths["directory"].mkdir(parents=True)
    invocation = source_invocation(job, paths, plan)
    atomic_json(paths["invocation"], invocation)
    state = {
        "schema": "navtta.vln_source_control_attempt.v1",
        "batch_id": plan["batch_id"],
        "job_key": job["key"],
        "attempt": attempt,
        "status": "preparing",
        "started_at": utc_now(),
        "run_id": paths["run_id"],
        "run_tag": paths["run_tag"],
        "result_root": str(paths["result_root"]),
        "formal_manifest_path": str(paths["manifest"]),
        "command": invocation["command"],
    }
    atomic_json(paths["attempt"], state)
    try:
        if tracked_worktree_dirty() or git_commit() != plan["git_commit"]:
            raise WorkflowError("tracked worktree/commit changed before job start")
        create_formal_manifest(job, paths, plan, vln_root, environment_path)
        command = runner_command(job, paths["run_tag"])
        with paths["launcher_log"].open("w", encoding="utf-8", newline="\n") as log:
            log.write("command: {}\n".format(" ".join(command)))
            log.flush()
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            identity = process_identity(process.pid) or {
                "pid": process.pid, "start_ticks": None, "cmdline": ""
            }
            identity["hostname"] = socket.gethostname()
            state["status"] = "running"
            state["process"] = identity
            atomic_json(paths["attempt"], state)
            with ACTIVE_PROCESSES_LOCK:
                ACTIVE_PROCESSES[job["key"]] = process
            exit_code = process.wait()
            with ACTIVE_PROCESSES_LOCK:
                ACTIVE_PROCESSES.pop(job["key"], None)
            log.write("exit_code: {}\n".format(exit_code))
        if exit_code != 0:
            finalize_formal_manifest(paths, exit_code)
            state.update({
                "status": "failed", "exit_code": exit_code,
                "completed_at": utc_now(),
            })
            atomic_json(paths["attempt"], state)
            return False
        report, aggregate_path, per_episode_path = validate_result(job, paths)
        finalize_formal_manifest(paths, 0, artifacts=(
            ("source_invocation", paths["invocation"]),
            ("aggregate_metrics", aggregate_path),
            ("per_episode_metrics", per_episode_path),
            ("result_validation", paths["validation"]),
        ))
        manifest = authenticate_formal_manifest(job, paths, plan)
        state.update({
            "status": "ready", "exit_code": 0, "completed_at": utc_now(),
            "metrics": report["metrics"],
            "formal_manifest_sha256": sha256_file(paths["manifest"]),
            "immutable_identity_sha256": manifest["immutable_identity_sha256"],
            "validation_path": str(paths["validation"]),
            "validation_sha256": sha256_file(paths["validation"]),
        })
        atomic_json(paths["attempt"], state)
        print("[ready] {} SR={:.6f} SPL={:.6f}".format(
            job["key"], report["metrics"]["SR"], report["metrics"]["SPL"]
        ), flush=True)
        return True
    except Exception as error:
        with ACTIVE_PROCESSES_LOCK:
            ACTIVE_PROCESSES.pop(job["key"], None)
        if paths["manifest"].is_file():
            try:
                manifest = read_json(paths["manifest"], "formal manifest")
                if manifest.get("status") == "running":
                    finalize_formal_manifest(paths, 65)
            except Exception as finalize_error:
                state["manifest_finalize_error"] = str(finalize_error)
        state.update({
            "status": "invalid", "exit_code": 65, "completed_at": utc_now(),
            "error": str(error),
        })
        atomic_json(paths["attempt"], state)
        print("[invalid] {}: {}".format(job["key"], error), file=sys.stderr, flush=True)
        return False


def content_addressed_write(directory, stem, document):
    data = canonical_bytes(document)
    digest = sha256_bytes(data)
    path = Path(directory) / "{}.{}.json".format(stem, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise WorkflowError("content-address collision: {}".format(path))
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        os.replace(str(temporary), str(path))
    return path, digest


def ledger_record(job, state):
    manifest_path = Path(state["formal_manifest_path"])
    manifest = read_json(manifest_path, "formal Source manifest")
    validation_path = Path(state["validation_path"])
    validation = read_json(validation_path, "Source validation")
    aggregate = validation["aggregate_artifact"]
    per_episode = validation["per_episode_artifact"]
    return {
        "model": job["model"],
        "run_id": manifest["run_id"],
        "run_tag": manifest["run_tag"],
        "git_commit": manifest["git_commit"],
        "parameters": {"action_selection": "argmax", "action_seed": 0},
        "metrics": validation["metrics"],
        "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        "dataset_sha256": manifest["dataset"]["stream_content_sha256"],
        "episode_order_sha256": manifest["dataset"]["stream_order_sha256"],
        "episode_count": validation["episode_count"],
        "evidence_status": "ready",
        "metrics_artifact_path": relative_to_repo(validation_path),
        "metrics_artifact_sha256": sha256_file(validation_path),
        "aggregate_artifact_path": relative_to_repo(aggregate["path"]),
        "aggregate_artifact_sha256": aggregate["sha256"],
        "per_episode_artifact_path": relative_to_repo(per_episode["path"]),
        "per_episode_artifact_sha256": per_episode["sha256"],
        "formal_manifest_path": relative_to_repo(manifest_path),
        "formal_manifest_sha256": sha256_file(manifest_path),
        "immutable_identity_sha256": manifest["immutable_identity_sha256"],
    }


def emit_candidate_ledgers(plan, root):
    candidates = []
    completed_times = []
    by_split = {split: {} for split in SPLITS}
    for job in plan["jobs"]:
        status, _, state = job_status(root, job)
        if status != "ready":
            raise WorkflowError(
                "cannot emit Source ledgers; {} status is {}".format(
                    job["key"], status
                )
            )
        by_split[job["split"]][job["setting"]] = ledger_record(job, state)
        completed_times.append(state["completed_at"])
    for split in SPLITS:
        order = plan["orders"][split]
        document = {
            "schema": LEDGER_SCHEMA,
            "schema_version": 1,
            "candidate_status": "review_required_not_yet_promoted",
            "benchmark": BENCHMARK,
            "data_version": PROTOCOL,
            "split": split,
            "episode_count": order["episodes"],
            "canonical_order_seed": 0,
            "source_protocol": "target_native_argmax",
            "git_commit": plan["git_commit"],
            "batch_id": plan["batch_id"],
            "plan_identity_sha256": plan["plan_identity_sha256"],
            "completed_at": max(completed_times),
            "episode_order": {
                "path": order["path"],
                "sha256": order["sha256"],
                "order_sha256": order["order_sha256"],
            },
            "records": by_split[split],
            "promotion_policy": {
                "tracked_review_required": True,
                "existing_v1_3_ledgers_overwritten": False,
                "suggested_tracked_path": (
                    "vln/manifests/r2r_ce_v1_2_{}_source_controls.json".format(split)
                ),
            },
        }
        path, digest = content_addressed_write(
            root / "candidate-ledgers",
            "r2r-ce-v1.2-native-source-controls-{}".format(split),
            document,
        )
        candidates.append({"split": split, "path": str(path), "sha256": digest})
    pointer = root / "CANDIDATE_LEDGERS.txt"
    lines = ["{}\t{}\t{}".format(item["split"], item["sha256"], item["path"])
             for item in candidates]
    pointer.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return candidates


def print_status(plan, root, as_json=False):
    jobs = []
    counts = {}
    for job in plan["jobs"]:
        status, attempt, state = job_status(root, job)
        counts[status] = counts.get(status, 0) + 1
        jobs.append({
            "key": job["key"], "gpu": job["gpu"], "status": status,
            "latest_attempt": attempt,
            "run_tag": state.get("run_tag") if isinstance(state, dict) else None,
            "error": state.get("error") if isinstance(state, dict) else None,
        })
    candidates = []
    candidate_dir = root / "candidate-ledgers"
    if candidate_dir.is_dir():
        candidates = [str(path) for path in sorted(candidate_dir.glob("*.json"))]
    document = {
        "batch_id": plan["batch_id"],
        "git_commit": plan["git_commit"],
        "plan_identity_sha256": plan["plan_identity_sha256"],
        "counts": counts,
        "jobs": jobs,
        "candidate_ledgers": candidates,
    }
    if as_json:
        print(json.dumps(document, indent=2, sort_keys=True))
    else:
        print("batch={} commit={}".format(plan["batch_id"], plan["git_commit"]))
        for item in jobs:
            print("{status:22s} gpu={gpu} attempt={attempt} {key}".format(
                status=item["status"], gpu=item["gpu"],
                attempt=item["latest_attempt"] or "-", key=item["key"]
            ))
        print("counts={}".format(json.dumps(counts, sort_keys=True)))
        for path in candidates:
            print("candidate-ledger={}".format(path))
    return document


def parse_gpus(value):
    values = value.split(",")
    if len(values) != 2 or any(not item.isdigit() for item in values):
        raise argparse.ArgumentTypeError("expected two comma-separated GPU IDs")
    parsed = [int(item) for item in values]
    if len(set(parsed)) != 2:
        raise argparse.ArgumentTypeError("GPU IDs must be distinct")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # ``required=`` for subparsers was added in Python 3.7.  Keep this
    # orchestration entry point usable from the project's Python 3.6 runtime.
    subparsers = parser.add_subparsers(dest="action")
    for action in ("plan", "run", "status", "finalize"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--batch-id", required=True)
        sub.add_argument(
            "--gpus", type=parse_gpus,
            default=[0, 1] if action in ("plan", "run") else None,
            help=(
                "two physical GPU IDs (default: 0,1 for plan/run; status and "
                "finalize infer the persisted plan)"
            ),
        )
        if action == "run":
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--plan-only", action="store_true")
            sub.add_argument(
                "--vln-root", type=Path,
                default=Path("/data1/wxy/exp_data/NavTTA/vln"),
            )
        if action == "status":
            sub.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.action is None:
        parser.error("one action is required: plan, run, status, or finalize")
    if not SAFE_ID.fullmatch(args.batch_id):
        parser.error("--batch-id must use only letters, numbers, dot, underscore, dash")
    return args


def acquire_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / ".workflow.lock").open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise WorkflowError("another scheduler owns this batch")
    return stream


def main(argv=None):
    args = parse_args(argv)
    root = batch_root(args.batch_id)
    if args.action == "status":
        if not plan_path(args.batch_id).is_file():
            raise WorkflowError("batch plan does not exist: {}".format(plan_path(args.batch_id)))
        stored = read_json(plan_path(args.batch_id), "Source-control plan")
        gpus = args.gpus if args.gpus is not None else stored.get("gpus")
        plan = validate_plan(
            stored, args.batch_id, gpus,
        )
        print_status(plan, root, as_json=args.json)
        return 0

    lock = acquire_lock(root)
    try:
        if args.action == "finalize":
            if not plan_path(args.batch_id).is_file():
                raise WorkflowError(
                    "batch plan does not exist: {}".format(plan_path(args.batch_id))
                )
            stored = read_json(plan_path(args.batch_id), "Source-control plan")
            gpus = args.gpus if args.gpus is not None else stored.get("gpus")
            plan = validate_plan(stored, args.batch_id, gpus)
        else:
            plan = ensure_plan(args.batch_id, args.gpus)
        if args.action == "plan" or (args.action == "run" and args.plan_only):
            print("plan={}".format(plan_path(args.batch_id)))
            print("plan_identity_sha256={}".format(plan["plan_identity_sha256"]))
            for job in plan["jobs"]:
                print("gpu={} {} {} {}".format(
                    job["gpu"], job["setting"], job["split"],
                    " ".join(job["command_template"]),
                ))
            return 0
        if args.action == "finalize":
            candidates = emit_candidate_ledgers(plan, root)
            for item in candidates:
                print("candidate {} {} {}".format(
                    item["split"], item["sha256"], item["path"]
                ))
            return 0

        existing_attempts = sum(
            len(attempt_numbers(root, job)) for job in plan["jobs"]
        )
        if existing_attempts and not args.resume:
            raise WorkflowError(
                "batch has existing attempts; inspect status and pass --resume"
            )
        environment_path = preflight(plan, args.vln_root, root)
        signal.signal(signal.SIGINT, stop_active_processes)
        signal.signal(signal.SIGTERM, stop_active_processes)
        jobs_by_setting = {
            setting: [job for job in plan["jobs"] if job["setting"] == setting]
            for setting in SETTINGS
        }

        def run_model_queue(setting):
            success = True
            for job in jobs_by_setting[setting]:
                if STOP_EVENT.is_set():
                    return False
                if not execute_job(
                    job, plan, root, args.vln_root, environment_path
                ):
                    success = False
                    break
            return success

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(run_model_queue, setting): setting
                for setting in SETTINGS
            }
            results = {}
            for future in concurrent.futures.as_completed(futures):
                setting = futures[future]
                try:
                    results[setting] = future.result()
                except Exception as error:
                    print("[queue-failed] {}: {}".format(setting, error), file=sys.stderr)
                    results[setting] = False
                    STOP_EVENT.set()
                    stop_active_processes()
        print_status(plan, root)
        if STOP_EVENT.is_set() or not all(results.values()):
            return 1
        candidates = emit_candidate_ledgers(plan, root)
        for item in candidates:
            print("candidate {} {} {}".format(
                item["split"], item["sha256"], item["path"]
            ))
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except WorkflowError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
