#!/usr/bin/env python3
"""Run the frozen AVN FSTTA configuration on the three remaining main-table jobs."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_BASE = REPO_ROOT / "avn" / "results" / "logs" / "fstta_main"
RUNS_ROOT = REPO_ROOT / "avn" / "results" / "runs"

FROZEN_CONFIG = {
    "norm_scope": "last_k_ln",
    "last_k_ln": 4,
    "fast_lr": "3e-7",
    "fast_window": 16,
    "slow_lr": "1e-4",
    "slow_window": 32,
    "q": "0.1",
    "rho": "0.95",
    "tau": "0.7",
    "a": "0.9",
    "b": "1.1",
    "fast_grad_mode": "concordant",
    "use_fast_lr_scaler": True,
    "use_slow": True,
    "fast_optimizer": "AdamW",
    "slow_optimizer": "AdamW",
    "slow_momentum": "0.0",
    "beta1": "0.9",
    "beta2": "0.99",
    "weight_decay": "0.0",
    "max_grad_norm": "1.0",
    "reset_fast_optimizer_each_episode": True,
    "reset_slow_optimizer_each_window": False,
    "eigen_eps": "1e-6",
    "episodic": False,
    "steps": 1,
    "reset_bn_stats": True,
}

JOB_SPECS = (
    ("smt_audio", "multi_source"),
    ("enmus", "single_source"),
    ("enmus", "multi_source"),
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

NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class UserError(Exception):
    """A launcher error that should be shown without a traceback."""


class SchedulerLogger:
    def __init__(self, path):
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message):
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


def utc_timestamp(compact=False):
    now = datetime.now(timezone.utc)
    if compact:
        return now.strftime("%Y%m%dT%H%M%SZ")
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def config_bool(value):
    return "True" if value else "False"


def shell_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def run_checked(command, cwd=None):
    try:
        return subprocess.check_output(
            [str(item) for item in command],
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        output = getattr(error, "output", "") or ""
        detail = output.strip() or str(error)
        raise UserError("command failed: {}\n{}".format(shell_join(command), detail))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runner_for_model(model):
    if model == "smt_audio":
        return REPO_ROOT / "avn" / "scripts" / "eval_smt_audio.sh"
    if model == "enmus":
        return REPO_ROOT / "avn" / "scripts" / "eval_enmus.sh"
    raise UserError("unknown model: {}".format(model))


def checkpoint_for_job(model, source_setting):
    names = {
        ("smt_audio", "multi_source"): "multi_best_val.pth",
        ("enmus", "single_source"): "single_source_best_val.pth",
        ("enmus", "multi_source"): "multi_source_best_val.pth",
    }
    try:
        name = names[(model, source_setting)]
    except KeyError:
        raise UserError(
            "unsupported final FSTTA job: {}/{}".format(model, source_setting)
        )
    return REPO_ROOT / "avn" / "checkpoints" / "source" / model / name


def dataset_for_setting(source_setting):
    return (
        REPO_ROOT
        / "avn"
        / "data"
        / "datasets"
        / "tta_test"
        / source_setting
        / "mp3d"
        / "v1"
        / "val"
        / "val.json.gz"
    )


def enmus_auxiliary_checkpoints():
    root = (
        REPO_ROOT
        / "avn"
        / "baselines"
        / "enmus"
        / "data"
        / "pretrained_weights"
        / "semantic_audionav"
        / "enmus"
    )
    return {
        "audio_encoder": root / "audio_encoder_best_val.pth",
        "visual_encoder": root / "visual_encoder_best_val.pth",
        "seld_encoder": root / "seld_crnn_best_val.h5",
    }


def run_tag(batch_id, job_id, model, source_setting):
    return "{}-j{:02d}-{}-{}".format(
        batch_id, job_id, model, source_setting
    )


def config_overrides(model, episodes):
    config = FROZEN_CONFIG
    pairs = [
        ("NUM_PROCESSES", "1"),
        ("EVAL.USE_CKPT_CONFIG", "False"),
    ]
    # SMT+Audio exposes an explicit policy selection switch. ENMuS already
    # follows its native stochastic evaluation path and has no equivalent key.
    if model == "smt_audio":
        pairs.append(("EVAL.ACTION_SELECTION", "sample"))
    pairs.extend(
        (
            ("TTA.LR", config["fast_lr"]),
            ("TTA.NORM_SCOPE", config["norm_scope"]),
            ("TTA.LAST_K_LN", str(config["last_k_ln"])),
            ("TTA.EPISODIC", config_bool(config["episodic"])),
            ("TTA.STEPS", str(config["steps"])),
            ("TTA.RESET_BN_STATS", config_bool(config["reset_bn_stats"])),
            ("TTA.MAX_GRAD_NORM", config["max_grad_norm"]),
            ("TTA.FSTTA.M", str(config["fast_window"])),
            ("TTA.FSTTA.N", str(config["slow_window"])),
            ("TTA.FSTTA.Q", config["q"]),
            ("TTA.FSTTA.LR_SLOW", config["slow_lr"]),
            ("TTA.FSTTA.RHO", config["rho"]),
            ("TTA.FSTTA.TAU", config["tau"]),
            ("TTA.FSTTA.A", config["a"]),
            ("TTA.FSTTA.B", config["b"]),
            ("TTA.FSTTA.USE_SLOW", config_bool(config["use_slow"])),
            ("TTA.FSTTA.FAST_GRAD_MODE", config["fast_grad_mode"]),
            (
                "TTA.FSTTA.USE_FAST_LR_SCALER",
                config_bool(config["use_fast_lr_scaler"]),
            ),
            ("TTA.FSTTA.OPTIMIZER", config["fast_optimizer"]),
            ("TTA.FSTTA.BETA1", config["beta1"]),
            ("TTA.FSTTA.BETA2", config["beta2"]),
            ("TTA.FSTTA.WEIGHT_DECAY", config["weight_decay"]),
            ("TTA.FSTTA.SLOW_OPTIMIZER", config["slow_optimizer"]),
            ("TTA.FSTTA.SLOW_MOMENTUM", config["slow_momentum"]),
            (
                "TTA.FSTTA.RESET_OPTIMIZER_EACH_EPISODE",
                config_bool(config["reset_fast_optimizer_each_episode"]),
            ),
            (
                "TTA.FSTTA.RESET_SLOW_OPTIMIZER_EACH_WINDOW",
                config_bool(config["reset_slow_optimizer_each_window"]),
            ),
            ("TTA.FSTTA.EIGEN_EPS", config["eigen_eps"]),
            ("TEST_EPISODE_COUNT", str(episodes)),
        )
    )
    flattened = []
    for key, value in pairs:
        flattened.extend((key, value))
    return flattened


def command_for_job(job, seed, episodes):
    return [
        "bash",
        str(job["runner"]),
        job["source_setting"],
        "fstta",
        str(seed),
    ] + config_overrides(job["model"], episodes)


def parse_gpus(value):
    gpus = value.split(",")
    if len(gpus) != 3 or any(not re.match(r"^\d+$", gpu) for gpu in gpus):
        raise UserError("--gpus must contain exactly three numeric GPU ids")
    if len(set(gpus)) != len(gpus):
        raise UserError("--gpus GPU ids must be distinct")
    return gpus


def build_jobs(batch_id, gpus):
    jobs = []
    for job_id, ((model, source_setting), gpu) in enumerate(zip(JOB_SPECS, gpus)):
        jobs.append(
            {
                "job_id": job_id,
                "model": model,
                "source_setting": source_setting,
                "gpu": gpu,
                "runner": runner_for_model(model),
                "checkpoint": checkpoint_for_job(model, source_setting),
                "dataset": dataset_for_setting(source_setting),
                "run_tag": run_tag(
                    batch_id, job_id, model, source_setting
                ),
            }
        )
    return jobs


def print_plan(batch_id, jobs, seed, episodes):
    print("AVN FSTTA frozen main-table evaluation")
    print("  repository:       {}".format(REPO_ROOT))
    print("  batch id:         {}".format(batch_id))
    print("  seed/order:       {}".format(seed))
    print("  episodes/job:     {}".format(episodes))
    print("  FSTTA:            fast_lr=3e-7 M=16 slow_lr=1e-4 N=32")
    print("  method identity:  concordant FAST + LR scaler + persistent AdamW SLOW")
    for job in jobs:
        command = command_for_job(job, seed, episodes)
        print(
            "  job={:02d} gpu={} model={} source_setting={}".format(
                job["job_id"],
                job["gpu"],
                job["model"],
                job["source_setting"],
            )
        )
        print(
            "    CUDA_VISIBLE_DEVICES={} NAVTTA_RUN_TAG={} {}".format(
                shlex.quote(job["gpu"]),
                shlex.quote(job["run_tag"]),
                shell_join(command),
            )
        )


def collect_provenance(jobs, seed, episodes):
    git_commit = run_checked(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]
    )
    stream_fingerprints = {}
    for source_setting in ("single_source", "multi_source"):
        dataset = dataset_for_setting(source_setting)
        output = run_checked(
            [
                sys.executable,
                str(REPO_ROOT / "avn" / "scripts" / "fingerprint_episode_stream.py"),
                "--dataset",
                str(dataset),
                "--seed",
                str(seed),
                "--episode-count",
                str(episodes),
            ]
        )
        parts = output.split()
        if len(parts) != 2 or any(
            not re.match(r"^[0-9a-f]{64}$", item) for item in parts
        ):
            raise UserError(
                "invalid {} episode-stream fingerprints: {}".format(
                    source_setting, output
                )
            )
        stream_fingerprints[source_setting] = {
            "order": parts[0],
            "content": parts[1],
            "dataset_index": sha256_file(dataset),
        }

    checkpoint_hashes = {}
    for job in jobs:
        key = (job["model"], job["source_setting"])
        checkpoint_hashes[key] = sha256_file(job["checkpoint"])

    auxiliary_hashes = {
        name: sha256_file(path)
        for name, path in enmus_auxiliary_checkpoints().items()
    }
    return {
        "git_commit": git_commit,
        "streams": stream_fingerprints,
        "checkpoints": checkpoint_hashes,
        "auxiliary": auxiliary_hashes,
    }


def validate_inputs(jobs, allow_dirty):
    required = [
        REPO_ROOT / "tools" / "create_run_manifest.py",
        REPO_ROOT / "tools" / "finalize_run_manifest.py",
        REPO_ROOT / "tools" / "validate_run_manifest.py",
        REPO_ROOT / "avn" / "scripts" / "fingerprint_episode_stream.py",
    ]
    for job in jobs:
        required.extend((job["runner"], job["checkpoint"], job["dataset"]))
    required.extend(enmus_auxiliary_checkpoints().values())
    missing = sorted({str(path) for path in required if not path.is_file()})
    if missing:
        raise UserError("missing required files:\n  " + "\n  ".join(missing))

    dirty_output = run_checked(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ]
    )
    worktree_dirty = bool(dirty_output)
    if worktree_dirty and not allow_dirty:
        raise UserError(
            "tracked worktree changes detected; commit them first or use --allow-dirty"
        )
    return worktree_dirty


def batch_spec_text(batch_id, gpus, seed, episodes, provenance):
    config = FROZEN_CONFIG
    values = (
        ("batch_id", batch_id),
        ("git_commit", provenance["git_commit"]),
        ("tracked_worktree_dirty", config_bool(provenance["worktree_dirty"])),
        ("seed", seed),
        ("episodes", episodes),
        ("gpus", ",".join(gpus)),
        ("method", "fstta"),
        ("norm_scope", config["norm_scope"]),
        ("last_k_ln", config["last_k_ln"]),
        ("fast_lr", config["fast_lr"]),
        ("M", config["fast_window"]),
        ("slow_lr", config["slow_lr"]),
        ("N", config["slow_window"]),
        ("q", config["q"]),
        ("rho", config["rho"]),
        ("tau", config["tau"]),
        ("a", config["a"]),
        ("b", config["b"]),
        ("fast_grad_mode", config["fast_grad_mode"]),
        ("use_fast_lr_scaler", config_bool(config["use_fast_lr_scaler"])),
        ("use_slow", config_bool(config["use_slow"])),
        ("fast_optimizer", config["fast_optimizer"]),
        ("slow_optimizer", config["slow_optimizer"]),
        (
            "reset_slow_optimizer_each_window",
            config_bool(config["reset_slow_optimizer_each_window"]),
        ),
        (
            "single_stream_order_sha256",
            provenance["streams"]["single_source"]["order"],
        ),
        (
            "single_stream_content_sha256",
            provenance["streams"]["single_source"]["content"],
        ),
        (
            "multi_stream_order_sha256",
            provenance["streams"]["multi_source"]["order"],
        ),
        (
            "multi_stream_content_sha256",
            provenance["streams"]["multi_source"]["content"],
        ),
        ("enmus_audio_sha256", provenance["auxiliary"]["audio_encoder"]),
        ("enmus_visual_sha256", provenance["auxiliary"]["visual_encoder"]),
        ("enmus_seld_sha256", provenance["auxiliary"]["seld_encoder"]),
    )
    return "".join("{}={}\n".format(key, value) for key, value in values)


def plan_csv_text(jobs, seed, episodes, provenance):
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "job_id",
            "run_tag",
            "gpu",
            "model",
            "source_setting",
            "method",
            "seed",
            "episodes",
            "checkpoint_sha256",
            "dataset_index_sha256",
            "stream_order_sha256",
            "stream_content_sha256",
        )
    )
    for job in jobs:
        stream = provenance["streams"][job["source_setting"]]
        writer.writerow(
            (
                job["job_id"],
                job["run_tag"],
                job["gpu"],
                job["model"],
                job["source_setting"],
                "fstta",
                seed,
                episodes,
                provenance["checkpoints"][(job["model"], job["source_setting"])],
                stream["dataset_index"],
                stream["order"],
                stream["content"],
            )
        )
    return output.getvalue()


def prepare_batch(log_root, batch_spec, plan, resume):
    if not resume:
        if log_root.exists():
            raise UserError("batch already exists: {}".format(log_root))
        (log_root / "jobs").mkdir(parents=True)
        (log_root / "batch.env").write_text(batch_spec, encoding="utf-8")
        (log_root / "plan.csv").write_text(plan, encoding="utf-8")
        return

    if not log_root.is_dir():
        raise UserError("resume batch does not exist: {}".format(log_root))
    for filename, expected in (("batch.env", batch_spec), ("plan.csv", plan)):
        path = log_root / filename
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise UserError(
                "resume arguments or immutable inputs do not match {}".format(path)
            )


def parameters_text(job, seed, episodes, provenance):
    config = FROZEN_CONFIG
    stream = provenance["streams"][job["source_setting"]]
    values = (
        ("job_id", job["job_id"]),
        ("run_tag", job["run_tag"]),
        ("gpu", job["gpu"]),
        ("model", job["model"]),
        ("source_setting", job["source_setting"]),
        ("method", "fstta"),
        ("seed", seed),
        ("episodes", episodes),
        ("norm_scope", config["norm_scope"]),
        ("last_k_ln", config["last_k_ln"]),
        ("fast_lr", config["fast_lr"]),
        ("M", config["fast_window"]),
        ("slow_lr", config["slow_lr"]),
        ("N", config["slow_window"]),
        ("q", config["q"]),
        ("rho", config["rho"]),
        ("tau", config["tau"]),
        ("a", config["a"]),
        ("b", config["b"]),
        ("fast_grad_mode", config["fast_grad_mode"]),
        ("use_fast_lr_scaler", config_bool(config["use_fast_lr_scaler"])),
        ("use_slow", config_bool(config["use_slow"])),
        ("fast_optimizer", config["fast_optimizer"]),
        ("slow_optimizer", config["slow_optimizer"]),
        ("beta1", config["beta1"]),
        ("beta2", config["beta2"]),
        ("weight_decay", config["weight_decay"]),
        ("max_grad_norm", config["max_grad_norm"]),
        (
            "reset_fast_optimizer_each_episode",
            config_bool(config["reset_fast_optimizer_each_episode"]),
        ),
        (
            "reset_slow_optimizer_each_window",
            config_bool(config["reset_slow_optimizer_each_window"]),
        ),
        ("git_commit", provenance["git_commit"]),
        ("tracked_worktree_dirty", config_bool(provenance["worktree_dirty"])),
        (
            "checkpoint_sha256",
            provenance["checkpoints"][(job["model"], job["source_setting"])],
        ),
        ("dataset_index_sha256", stream["dataset_index"]),
        ("stream_order_sha256", stream["order"]),
        ("stream_content_sha256", stream["content"]),
    )
    return "".join("{}={}\n".format(key, value) for key, value in values)


def archive_previous_attempt(job_dir):
    suffix = utc_timestamp(compact=True)
    for name in ("console.log", "exitcode", "validation", "validation_error.txt"):
        path = job_dir / name
        if path.exists():
            path.replace(job_dir / "{}.previous.{}".format(name, suffix))


def terminate_running_jobs(running):
    for job, process, console in running:
        if process.poll() is None:
            process.terminate()
    for job, process, console in running:
        if process.poll() is None:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if not console.closed:
            console.close()
        exitcode = job["job_dir"] / "exitcode"
        if not exitcode.exists():
            exitcode.write_text(str(process.returncode) + "\n", encoding="utf-8")


def launch_jobs(jobs, seed, episodes, provenance, log_root, resume, logger):
    running = []
    try:
        for job in jobs:
            job_dir = log_root / "jobs" / "j{:02d}-{}-{}".format(
                job["job_id"], job["model"], job["source_setting"]
            )
            job["job_dir"] = job_dir
            job_dir.mkdir(parents=True, exist_ok=True)
            exitcode = job_dir / "exitcode"
            if (
                resume
                and exitcode.is_file()
                and exitcode.read_text().strip() == "0"
            ):
                completed_ok, validation_error = completed_job_is_valid(
                    job, job_dir, seed, episodes, provenance
                )
                if completed_ok:
                    logger.log(
                        "{} skip validated job={:02d} model={} source_setting={}".format(
                            utc_timestamp(),
                            job["job_id"],
                            job["model"],
                            job["source_setting"],
                        )
                    )
                    continue
                logger.log(
                    "{} rerun invalid completed job={:02d}: {}".format(
                        utc_timestamp(), job["job_id"], validation_error
                    )
                )

            archive_previous_attempt(job_dir)
            command = command_for_job(job, seed, episodes)
            (job_dir / "parameters.env").write_text(
                parameters_text(job, seed, episodes, provenance), encoding="utf-8"
            )
            (job_dir / "command.txt").write_text(
                shell_join(command) + "\n", encoding="utf-8"
            )
            console = (job_dir / "console.log").open("w", encoding="utf-8")
            console.write("===== attempt {} =====\n".format(utc_timestamp()))
            console.flush()
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": job["gpu"],
                    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                    "PYTHONUNBUFFERED": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NAVTTA_RUN_TAG": job["run_tag"],
                    "NAVTTA_STREAM_ORDER_SHA256": provenance["streams"][
                        job["source_setting"]
                    ]["order"],
                    "NAVTTA_STREAM_CONTENT_SHA256": provenance["streams"][
                        job["source_setting"]
                    ]["content"],
                }
            )
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                )
            except BaseException:
                console.close()
                raise
            running.append((job, process, console))
            logger.log(
                "{} launched job={:02d} model={} source_setting={} gpu={} pid={} tag={}".format(
                    utc_timestamp(),
                    job["job_id"],
                    job["model"],
                    job["source_setting"],
                    job["gpu"],
                    process.pid,
                    job["run_tag"],
                )
            )
    except BaseException:
        terminate_running_jobs(running)
        raise
    return running


def wait_for_jobs(running, logger):
    failures = 0
    try:
        for job, process, console in running:
            status = process.wait()
            console.close()
            tmp = job["job_dir"] / "exitcode.tmp"
            tmp.write_text(str(status) + "\n", encoding="utf-8")
            os.replace(str(tmp), str(job["job_dir"] / "exitcode"))
            if status != 0:
                failures += 1
            logger.log(
                "{} completed job={:02d} model={} source_setting={} exitcode={}".format(
                    utc_timestamp(),
                    job["job_id"],
                    job["model"],
                    job["source_setting"],
                    status,
                )
            )
    except BaseException:
        terminate_running_jobs(running)
        raise
    return failures


def manifest_from_console(console_path):
    text = console_path.read_text(encoding="utf-8", errors="replace").replace(
        "\r", "\n"
    )
    matches = re.findall(r"(?m)^(/[^\n]*?/manifest\.json)\s*$", text)
    return Path(matches[-1]) if matches else None


def metrics_from_console(console_path):
    text = console_path.read_text(encoding="utf-8", errors="replace").replace(
        "\r", "\n"
    )
    result = {}
    for metric in METRICS:
        pattern = r"Average episode {}:\s*({})".format(
            re.escape(metric), NUMBER_PATTERN
        )
        matches = re.findall(pattern, text)
        if not matches:
            return None
        try:
            numeric = float(matches[-1])
        except ValueError:
            return None
        if not math.isfinite(numeric):
            return None
        result[metric] = matches[-1]
    return result


def path_is_within(path, parent):
    try:
        return os.path.commonpath((str(path.resolve()), str(parent.resolve()))) == str(
            parent.resolve()
        )
    except ValueError:
        return False


def validate_manifest(job, manifest, seed, episodes, provenance):
    stream = provenance["streams"][job["source_setting"]]
    command = [
        sys.executable,
        str(REPO_ROOT / "tools" / "validate_run_manifest.py"),
        "--manifest",
        str(manifest),
        "--run-tag",
        job["run_tag"],
        "--model",
        job["model"],
        "--method",
        "fstta",
        "--source-setting",
        job["source_setting"],
        "--seed",
        str(seed),
        "--git-commit",
        provenance["git_commit"],
        "--checkpoint-sha256",
        provenance["checkpoints"][(job["model"], job["source_setting"])],
        "--stream-order-sha256",
        stream["order"],
        "--stream-content-sha256",
        stream["content"],
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    if completed.returncode != 0:
        return completed.stdout.strip() or "validate_run_manifest.py failed"

    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return "cannot read manifest: {}".format(error)
    expected_overrides = config_overrides(job["model"], episodes)
    if document.get("config_overrides") != expected_overrides:
        return "manifest config_overrides do not match the frozen FSTTA command"
    if document.get("dataset", {}).get("index_sha256") != stream["dataset_index"]:
        return "manifest dataset index SHA256 mismatch"
    visible_gpu = document.get("hardware", {}).get("cuda_visible_devices")
    if visible_gpu != job["gpu"]:
        return "manifest GPU mismatch: expected={} actual={}".format(
            job["gpu"], visible_gpu
        )
    if job["model"] == "enmus":
        actual_auxiliary = {
            item.get("name"): item.get("sha256")
            for item in document.get("auxiliary_checkpoints", [])
        }
        if actual_auxiliary != provenance["auxiliary"]:
            return "manifest ENMuS auxiliary checkpoint SHA256 mismatch"
    return ""


def diagnostics_validation_error(manifest, seed, episodes):
    config = FROZEN_CONFIG
    run_dir = manifest.resolve().parent
    stats_path = run_dir / "raw" / "model" / "tb" / "val_stats_{}.json".format(
        seed
    )
    diagnostics_path = (
        run_dir / "raw" / "model" / "tb" / "tta_diagnostics_{}.json".format(seed)
    )
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return "cannot read FSTTA diagnostics artifacts: {}".format(error)
    if not isinstance(stats, dict) or len(stats) != episodes:
        return "episode statistics count is {}, expected {}".format(
            len(stats) if isinstance(stats, dict) else "non-dict", episodes
        )
    if not isinstance(diagnostics, dict):
        return "FSTTA diagnostics are not a JSON object"

    exact_expected = {
        "episodes": episodes,
        "fast_window": config["fast_window"],
        "use_slow": config["use_slow"],
        "slow_window": config["slow_window"],
        "fast_optimizer": config["fast_optimizer"],
        "slow_optimizer": config["slow_optimizer"],
        "fast_grad_mode": config["fast_grad_mode"],
        "use_fast_lr_scaler": config["use_fast_lr_scaler"],
        "reset_slow_optimizer_each_window": config[
            "reset_slow_optimizer_each_window"
        ],
        "slow_optimizer_resets": 0,
    }
    for key, expected in exact_expected.items():
        actual = diagnostics.get(key)
        if actual != expected:
            return "diagnostic {} expected={!r} actual={!r}".format(
                key, expected, actual
            )

    float_expected = {
        "fast_lr": float(config["fast_lr"]),
        "slow_lr": float(config["slow_lr"]),
        "q": float(config["q"]),
        "slow_momentum": float(config["slow_momentum"]),
    }
    for key, expected in float_expected.items():
        try:
            actual = float(diagnostics[key])
        except (KeyError, TypeError, ValueError):
            return "diagnostic {} is missing or nonnumeric".format(key)
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            return "diagnostic {} expected={} actual={}".format(
                key, expected, actual
            )

    try:
        slow_updates = int(diagnostics["slow_updates"])
        slow_skips = int(diagnostics["slow_skipped_updates"])
        slow_attempts = int(diagnostics["slow_attempts"])
        slow_pending = int(diagnostics["slow_pending_episodes"])
    except (KeyError, TypeError, ValueError):
        return "FSTTA slow-update diagnostics are missing or nonnumeric"
    expected_attempts = episodes // config["slow_window"]
    expected_pending = episodes % config["slow_window"]
    if slow_attempts != expected_attempts:
        return "FSTTA slow attempts expected={} actual={}".format(
            expected_attempts, slow_attempts
        )
    if slow_pending != expected_pending:
        return "FSTTA pending episodes expected={} actual={}".format(
            expected_pending, slow_pending
        )
    if slow_updates + slow_skips != expected_attempts:
        return "FSTTA slow updates plus skips do not equal slow attempts"

    names = diagnostics.get("adapted_parameter_names")
    expected_tensors = 2 * config["last_k_ln"]
    if (
        not isinstance(names, list)
        or len(names) != expected_tensors
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        return "adapted_parameter_names do not describe {} unique tensors".format(
            expected_tensors
        )
    return ""


def completed_job_is_valid(job, job_dir, seed, episodes, provenance):
    console_path = job_dir / "console.log"
    if not console_path.is_file():
        return False, "console.log is missing"
    manifest = manifest_from_console(console_path)
    if (
        manifest is None
        or not manifest.is_file()
        or not path_is_within(manifest, RUNS_ROOT)
    ):
        return False, "run manifest is missing or outside avn/results/runs"
    error = validate_manifest(job, manifest, seed, episodes, provenance)
    if error:
        return False, error
    if metrics_from_console(console_path) is None:
        return False, "final aggregate metrics are missing"
    error = diagnostics_validation_error(manifest, seed, episodes)
    if error:
        return False, error
    return True, ""


def write_run_summary(run_dir, job, seed, episodes, metrics, worktree_dirty):
    diagnostics_source = (
        run_dir / "raw" / "model" / "tb" / "tta_diagnostics_{}.json".format(seed)
    )
    if not diagnostics_source.is_file():
        return "diagnostics_missing"
    shutil.copy2(str(diagnostics_source), str(run_dir / "diagnostics.json"))
    summary = {
        "model": job["model"],
        "method": "fstta",
        "source_setting": job["source_setting"],
        "seed": seed,
        "episode_count": episodes,
        "canonical_main_table_configuration": episodes == 2000
        and not worktree_dirty,
        "tracked_worktree_dirty": worktree_dirty,
        "tta": FROZEN_CONFIG,
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    temporary = run_dir / "summary.json.tmp"
    temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(run_dir / "summary.json"))
    return ""


def validate_and_summarize(jobs, seed, episodes, provenance, log_root):
    rows = []
    succeeded = 0
    for job in jobs:
        job_dir = job["job_dir"]
        status = "missing"
        manifest = None
        metrics = None
        error = ""
        exitcode_path = job_dir / "exitcode"
        console_path = job_dir / "console.log"
        if exitcode_path.is_file():
            status = exitcode_path.read_text(encoding="utf-8").strip()
        if status == "0" and console_path.is_file():
            manifest = manifest_from_console(console_path)
            metrics = metrics_from_console(console_path)
            if (
                manifest is None
                or not manifest.is_file()
                or not path_is_within(manifest, RUNS_ROOT)
            ):
                status = "manifest_missing"
            else:
                error = validate_manifest(
                    job, manifest, seed, episodes, provenance
                )
                if error:
                    status = "manifest_invalid"
                elif metrics is None:
                    status = "metrics_missing"
                else:
                    error = diagnostics_validation_error(
                        manifest, seed, episodes
                    )
                    if error:
                        status = "diagnostics_invalid"
                    else:
                        error = write_run_summary(
                            manifest.parent,
                            job,
                            seed,
                            episodes,
                            metrics,
                            provenance["worktree_dirty"],
                        )
                        if error:
                            status = error

        if status == "0":
            succeeded += 1
            (job_dir / "validation").write_text("ok\n", encoding="utf-8")
            manifest_record = str(manifest.relative_to(REPO_ROOT))
        else:
            (job_dir / "validation").write_text(status + "\n", encoding="utf-8")
            if error:
                (job_dir / "validation_error.txt").write_text(
                    error + "\n", encoding="utf-8"
                )
            manifest_record = ""
            metrics = {key: "" for key in METRICS}

        rows.append(
            [
                job["model"],
                job["source_setting"],
                "fstta",
                seed,
                episodes,
                status,
                manifest_record,
            ]
            + [metrics[key] for key in METRICS]
        )

    metrics_path = log_root / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            (
                "model",
                "source_setting",
                "method",
                "seed",
                "episodes",
                "exitcode",
                "manifest",
            )
            + METRICS
        )
        writer.writerows(rows)
    return succeeded, len(jobs) - succeeded


def parse_args(argv=None):
    description = """\
Run exactly three jobs with the frozen FSTTA job-35 configuration:
  GPU 0: SMT+Audio multi_source
  GPU 1: ENMuS     single_source
  GPU 2: ENMuS     multi_source

Recommended detached launch:
  screen -dmS fstta_main python3 avn/scripts/run_fstta_main.py \\
    --gpus 0,1,2 --seed 0 --batch-id fstta-main-v1-seed0

Follow progress:
  tail -f avn/results/logs/fstta_main/fstta-main-v1-seed0/scheduler.log
"""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--batch-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.episodes < 1 or args.episodes > 2000:
        parser.error("--episodes must be in [1, 2000]")
    if args.resume and not args.batch_id:
        parser.error("--resume requires an explicit --batch-id")
    if args.batch_id is None:
        args.batch_id = "fstta-main-seed{}-{}".format(
            args.seed, utc_timestamp(compact=True)
        )
    if not BATCH_ID_PATTERN.match(args.batch_id):
        parser.error("--batch-id may contain only letters, digits, '.', '_' and '-'")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        gpus = parse_gpus(args.gpus)
        jobs = build_jobs(args.batch_id, gpus)
        print_plan(args.batch_id, jobs, args.seed, args.episodes)
        if args.dry_run:
            print("Dry run only; no files created and no jobs launched.")
            return 0

        worktree_dirty = validate_inputs(jobs, args.allow_dirty)
        provenance = collect_provenance(jobs, args.seed, args.episodes)
        provenance["worktree_dirty"] = worktree_dirty
        log_root = LOG_BASE / args.batch_id
        batch_spec = batch_spec_text(
            args.batch_id, gpus, args.seed, args.episodes, provenance
        )
        plan = plan_csv_text(jobs, args.seed, args.episodes, provenance)
        prepare_batch(log_root, batch_spec, plan, args.resume)

        lock_dir = log_root / ".scheduler.lock"
        try:
            lock_dir.mkdir()
        except FileExistsError:
            raise UserError(
                "batch is already running or has a stale lock: {}".format(lock_dir)
            )
        (lock_dir / "owner").write_text(
            "pid={}\nhost={}\nstarted_at={}\n".format(
                os.getpid(), os.uname().nodename, utc_timestamp()
            ),
            encoding="utf-8",
        )

        logger = SchedulerLogger(log_root / "scheduler.log")
        try:
            logger.log("scheduler_started_at={}".format(utc_timestamp()))
            logger.log("git_commit={}".format(provenance["git_commit"]))
            logger.log(
                "tracked_worktree_dirty={}".format(config_bool(worktree_dirty))
            )
            logger.log("resume={}".format(config_bool(args.resume)))
            if worktree_dirty:
                logger.log(
                    "warning=dirty-worktree run is exploratory and cannot enter the formal main table"
                )
            if args.episodes != 2000:
                logger.log(
                    "warning=smoke run only; not valid for the AVN main table"
                )
            running = launch_jobs(
                jobs,
                args.seed,
                args.episodes,
                provenance,
                log_root,
                args.resume,
                logger,
            )
            wait_failures = wait_for_jobs(running, logger)
            succeeded, failed = validate_and_summarize(
                jobs, args.seed, args.episodes, provenance, log_root
            )
            summary = (
                "batch_id={}\n"
                "git_commit={}\n"
                "tracked_worktree_dirty={}\n"
                "seed={}\n"
                "episodes_per_job={}\n"
                "planned={}\n"
                "succeeded={}\n"
                "failed={}\n"
                "completed_at={}\n"
            ).format(
                args.batch_id,
                provenance["git_commit"],
                config_bool(worktree_dirty),
                args.seed,
                args.episodes,
                len(jobs),
                succeeded,
                failed,
                utc_timestamp(),
            )
            (log_root / "SUMMARY").write_text(summary, encoding="utf-8")
            logger.log(
                "FSTTA main batch finished: succeeded={} failed={}".format(
                    succeeded, failed
                )
            )
            logger.log("Metrics: {}".format(log_root / "metrics.csv"))
            logger.log("Logs:    {}".format(log_root))
            if wait_failures or failed:
                logger.log(
                    "Batch incomplete; inspect logs and resume with --batch-id {} --resume.".format(
                        args.batch_id
                    )
                )
                return 1
            logger.log("All three frozen-configuration FSTTA jobs completed successfully.")
            return 0
        finally:
            logger.close()
            owner = lock_dir / "owner"
            if owner.exists():
                owner.unlink()
            try:
                lock_dir.rmdir()
            except OSError:
                pass
    except UserError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    signal.signal(signal.SIGHUP, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    raise SystemExit(main())
