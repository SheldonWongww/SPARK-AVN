#!/usr/bin/env python3
"""Run the 13 positive targeted-gap winners on val_seen and REVERIE test.

This launcher is intentionally independent from the still-running search
scheduler.  It embeds the val_unseen-selected configurations and only calls
the existing ``run_source_eval.sh`` entry point.  A row is one serial
pipeline: val_seen first, followed by test for REVERIE.  At most two row
pipelines may share one physical GPU.

Exact evaluator feedback is unavailable on the hidden REVERIE test split.
Consequently the GOAT-REVERIE FeedTTA row runs val_seen by default and records
its test stage as skipped.  ``--feedtta-test-mode llm`` enables the separately
reported FeedTTA-LLM transfer when its pinned local provider is configured.
"""

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = Path(__file__).resolve()
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
LOG_ROOT = REPO_ROOT / "vln/results/logs/frozen_positive_rows"
RESULT_ROOT = REPO_ROOT / "vln/results/tuning/frozen_positive_rows"
DEFAULT_BATCH_ID = "vln-positive-gap-fixed-v1-seed0"
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")

EPISODES = {
    ("hamt-reverie", "val_seen"): 1423,
    ("duet-reverie", "val_seen"): 1423,
    ("goat-reverie", "val_seen"): 1423,
    ("hamt-reverie", "test"): 6292,
    ("duet-reverie", "test"): 6292,
    ("goat-reverie", "test"): 6292,
    ("hamt-r2r", "val_seen"): 1021,
    ("goat-r2r", "val_seen"): 1021,
}

# These are the positive-primary-metric winners selected from the 2026-09-04
# snapshot of vln-targeted-gap-campaign-v2-seed0.  The development result
# digests make the freeze auditable without depending on mutable search logs.
FROZEN_ROWS: Tuple[Mapping[str, object], ...] = (
    {
        "queue": "q00", "cell_id": "hamt-reverie-eam",
        "setting": "hamt-reverie", "benchmark": "reverie", "method": "eam",
        "candidate_id": "lr_6em7__m32_b8_c0p6_i1",
        "development_result_sha256": "f579975c4b96d8f65c1a7b1f790caa7e126042b4a6dce29459c7c201df422293",
        "val_unseen_metrics": {"ORACLE_SR": 36.75, "SR": 33.23, "SPL": 30.67, "RGS": 19.57, "RGSPL": 17.97},
        "parameters": {"action_selection": "argmax", "batch_size": 8, "confidence_scale": 0.6, "lr": 6e-7, "memory_size": 32, "optimizer": "Adam", "update_interval": 1},
    },
    {
        "queue": "q01", "cell_id": "duet-reverie-eam",
        "setting": "duet-reverie", "benchmark": "reverie", "method": "eam",
        "candidate_id": "lr_2em7__m32_b8_c0p4_i1",
        "development_result_sha256": "239d0db8fba93ff40b4621446c9a6e64293e0e861aab0d38e97aba9c04e975f9",
        "val_unseen_metrics": {"ORACLE_SR": 50.55, "SR": 46.66, "SPL": 34.2, "RGS": 31.58, "RGSPL": 23.15},
        "parameters": {"action_selection": "argmax", "batch_size": 8, "confidence_scale": 0.4, "lr": 2e-7, "memory_size": 32, "optimizer": "Adam", "update_interval": 1},
    },
    {
        "queue": "q02", "cell_id": "goat-reverie-eam",
        "setting": "goat-reverie", "benchmark": "reverie", "method": "eam",
        "candidate_id": "lr_2em7__m32_b8_c0p6_i1",
        "development_result_sha256": "27959519d21d1af71e7e2ab7039d1d71c4403e022b32fa6b32fd5f12283e8216",
        "val_unseen_metrics": {"ORACLE_SR": 57.2, "SR": 53.39, "SPL": 37.99, "RGS": 38.94, "RGSPL": 27.37},
        "parameters": {"action_selection": "argmax", "batch_size": 8, "confidence_scale": 0.6, "lr": 2e-7, "memory_size": 32, "optimizer": "Adam", "update_interval": 1},
    },
    {
        "queue": "q03", "cell_id": "goat-reverie-feedtta",
        "setting": "goat-reverie", "benchmark": "reverie", "method": "feedtta",
        "candidate_id": "lr_4em6__g0p7_p0p05_am0p2",
        "development_result_sha256": "1a4c4960354f3775113e66a2c3aa4a1aee84fe657be2a5f3ad6b188ea4bffdc3",
        "val_unseen_metrics": {"ORACLE_SR": 66.23, "SR": 62.79, "SPL": 43.28, "RGS": 44.87, "RGSPL": 30.75},
        "parameters": {"action_selection": "argmax", "alpha": -0.2, "gamma": 0.7, "lr": 4e-6, "normalize_gradient": False, "optimizer_eps": 1e-5, "p": 0.05, "scope_profile": "paper_full", "sgr_mode": "paper_main", "sgr_seed": 0},
    },
    {
        "queue": "q05", "cell_id": "hamt-r2r-tent",
        "setting": "hamt-r2r", "benchmark": "r2r", "method": "tent",
        "candidate_id": "lr_3em7__last9_ln_i1",
        "development_result_sha256": "425dfef67eba6b1a12541572ce8345d170a43906fc1899dffcd186b66eaf4bd4",
        "val_unseen_metrics": {"LENGTHS": 11.46, "NAV_ERROR": 3.62, "SR": 66.33, "SPL": 61.57},
        "parameters": {"action_selection": "argmax", "last_k_ln": 9, "lr": 3e-7, "max_grad_norm": 0, "norm_scope": "last_k_ln", "optimizer": "Adam", "update_interval": 1},
    },
    {
        "queue": "q06", "cell_id": "hamt-r2r-fstta",
        "setting": "hamt-r2r", "benchmark": "r2r", "method": "fstta",
        "candidate_id": "lf_3em6_ls_5em6__m3_n4",
        "development_result_sha256": "6916b3aed929f01d0094ef8c67ecfb576fafe080f30c3746ecf1cb4dc8277dd4",
        "val_unseen_metrics": {"LENGTHS": 11.46, "NAV_ERROR": 3.61, "SR": 66.41, "SPL": 61.63},
        "parameters": {"a": 0.9, "action_selection": "argmax", "b": 1.1, "beta1": 0.9, "beta2": 0.99, "fast_grad_mode": "concordant", "last_k_ln": 4, "lr_fast": 3e-6, "lr_slow": 5e-6, "m": 3, "max_grad_norm": 0, "n": 4, "norm_scope": "last_k_ln", "q": 0.1, "reset_var_hist_each_episode": False, "rho": 0.95, "slow_optimizer": "AdamW", "tau": 0.7},
    },
    {
        "queue": "q07", "cell_id": "hamt-r2r-feedtta",
        "setting": "hamt-r2r", "benchmark": "r2r", "method": "feedtta",
        "candidate_id": "lr_6em7__g0p7_p0p05_a0p1",
        "development_result_sha256": "3395879dbaddbe69fef2364f55739793dae29608037e0fb2bd9c151f1e484220",
        "val_unseen_metrics": {"LENGTHS": 11.43, "NAV_ERROR": 3.54, "SR": 66.45, "SPL": 62.0},
        "parameters": {"action_selection": "argmax", "alpha": 0.1, "gamma": 0.7, "lr": 6e-7, "normalize_gradient": False, "optimizer_eps": 1e-5, "p": 0.05, "scope_profile": "paper_full", "sgr_mode": "paper_main", "sgr_seed": 0},
    },
    {
        "queue": "q08", "cell_id": "hamt-r2r-atena",
        "setting": "hamt-r2r", "benchmark": "r2r", "method": "atena",
        "candidate_id": "lq_4em7_ls_5em8__delta_0p15__lambda_0p1",
        "development_result_sha256": "a89792c39f1d38537d53b11e22a90d1f695fab4debb62904afd7b61c598533f5",
        "val_unseen_metrics": {"LENGTHS": 11.57, "NAV_ERROR": 3.57, "SR": 66.54, "SPL": 61.88},
        "parameters": {"action_selection": "argmax", "beta1": 0.9, "beta2": 0.999, "episodic": False, "lr_query": 4e-7, "lr_self": 5e-8, "max_grad_norm": 0, "mix_lambda": 0.1, "optimizer": "AdamW", "query_threshold": 0.15, "self_loss_weight": 0.1, "update_scope": "replay_reachable_high_level_navigation", "weight_decay": 0.01},
    },
    {
        "queue": "q10", "cell_id": "goat-r2r-tent",
        "setting": "goat-r2r", "benchmark": "r2r", "method": "tent",
        "candidate_id": "lr_5em6__all_ln_i1",
        "development_result_sha256": "e5c99c3641dd7940e5877470a5bd125ed2120d9a71cff3bdaf97c6ccc142232c",
        "val_unseen_metrics": {"LENGTHS": 12.56, "NAV_ERROR": 2.33, "SR": 78.29, "SPL": 68.38},
        "parameters": {"action_selection": "argmax", "lr": 5e-6, "max_grad_norm": 0, "norm_scope": "ln", "optimizer": "Adam", "update_interval": 1},
    },
    {
        "queue": "q11", "cell_id": "goat-r2r-eam",
        "setting": "goat-r2r", "benchmark": "r2r", "method": "eam",
        "candidate_id": "lr_6em7__m32_b8_c0p4_i1",
        "development_result_sha256": "1b661ba1a1febb4e8096aa1a36a439fcf996234549dbac70f4af56db47d638bb",
        "val_unseen_metrics": {"LENGTHS": 13.03, "NAV_ERROR": 2.3, "SR": 78.46, "SPL": 68.29},
        "parameters": {"action_selection": "argmax", "batch_size": 8, "confidence_scale": 0.4, "lr": 6e-7, "memory_size": 32, "optimizer": "Adam", "update_interval": 1},
    },
    {
        "queue": "q12", "cell_id": "goat-r2r-feedtta",
        "setting": "goat-r2r", "benchmark": "r2r", "method": "feedtta",
        "candidate_id": "lr_1em6__g0p95_p0p05_a0p1",
        "development_result_sha256": "a7950054c1a7c8895aee89fadc66b9cad3ebc685c8f28488310c78215f9c5fe7",
        "val_unseen_metrics": {"LENGTHS": 13.14, "NAV_ERROR": 2.26, "SR": 78.93, "SPL": 68.74},
        "parameters": {"action_selection": "argmax", "alpha": 0.1, "gamma": 0.95, "lr": 1e-6, "normalize_gradient": False, "optimizer_eps": 1e-5, "p": 0.05, "scope_profile": "paper_full", "sgr_mode": "paper_main", "sgr_seed": 0},
    },
    {
        "queue": "q14", "cell_id": "etpnav-r2r-ce-eam",
        "setting": "etpnav-r2r-ce", "benchmark": "r2r-ce", "method": "eam",
        "candidate_id": "lr_6em7__m32_b8_c0p4_i1", "ce_data_version": "v1.2-native",
        "development_result_sha256": "8886d8b378ad0f2fc0f6909586f4f4efa5a0e169780574f4f9ed52408d9d48c8",
        "val_unseen_metrics": {"PATH_LENGTH": 11.421930533281287, "DISTANCE_TO_GOAL": 4.776344528497724, "OSR": 64.76345840130506, "SR": 57.368134855899946, "SPL": 49.54666189613636},
        "parameters": {"action_selection": "argmax", "batch_size": 8, "confidence_scale": 0.4, "lr": 6e-7, "memory_size": 32, "optimizer": "Adam", "update_interval": 1},
    },
    {
        "queue": "q15", "cell_id": "bevbert-r2r-ce-eam",
        "setting": "bevbert-r2r-ce", "benchmark": "r2r-ce", "method": "eam",
        "candidate_id": "lr_5em8__m64_b16_c0p5_i4", "ce_data_version": "v1.2-native",
        "development_result_sha256": "5e3c11a26c2f8fc7c76a8cd234c85de137c644a451ea3287108f705d30a2d979",
        "val_unseen_metrics": {"PATH_LENGTH": 12.892707245546946, "DISTANCE_TO_GOAL": 4.597308518947418, "OSR": 67.64545948885264, "SR": 60.087003806416526, "SPL": 50.480234079813044},
        "parameters": {"action_selection": "argmax", "batch_size": 16, "confidence_scale": 0.5, "lr": 5e-8, "memory_size": 64, "optimizer": "Adam", "update_interval": 4},
    },
)

LLM_CONTRACT = {
    "feedback_provider": "qwen2_vl_2b_v1",
    "llm_feedback_model_id": "Qwen/Qwen2-VL-2B-Instruct",
    "llm_feedback_revision": "895c3a49bc3fa70a340399125c650a463535e71c",
    "llm_feedback_weights_sha256": "4faeb74ee719f9c35d7fa254d7cc2d7131e4b4ada7d0340615c66b22d5e16fc5",
    "llm_feedback_bundle_sha256": "cf7dd27d27987b7ae71458529e6a72e3dcc3db6e91fb7298577e03f88e238a7e",
    "llm_feedback_prompt_sha256": "9d608193a2cd444688c6f507973ab8da68940f4b11439d329195cfd2804e0ed9",
    "llm_feedback_timeout_seconds": 120.0,
    "llm_feedback_abort_on_failure": True,
}

_ACTIVE: Dict[int, subprocess.Popen] = {}
_ACTIVE_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()
_STOP = threading.Event()


class UserError(RuntimeError):
    pass


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest_value(value: object) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    with _PRINT_LOCK:
        print("[{}] {}".format(utc_now(), message), flush=True)


def parse_gpus(value: str) -> Tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("GPU ids must be a nonempty distinct list")
    if any(re.fullmatch(r"0|[1-9][0-9]*", item) is None for item in values):
        raise argparse.ArgumentTypeError("GPU ids must be nonnegative integers")
    return values


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=parse_gpus, default=parse_gpus("0,1,2,3"))
    parser.add_argument("--per-gpu-concurrency", type=int, default=2,
                        choices=(1, 2))
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--skip-reverie-test", action="store_true")
    parser.add_argument(
        "--feedtta-test-mode", choices=("skip", "llm"), default="skip",
        help=("hidden test has no exact binary feedback; use 'llm' only with "
              "the pinned local Qwen2-VL provider"),
    )
    args = parser.parse_args(argv)
    if SAFE_ID.fullmatch(args.batch_id) is None or len(args.batch_id) > 80:
        parser.error("--batch-id must be at most 80 safe characters")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    return args


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def splits_for(row: Mapping[str, object], args: argparse.Namespace) -> Tuple[str, ...]:
    values = ["val_seen"]
    if row["benchmark"] == "reverie" and not args.skip_reverie_test:
        if row["method"] != "feedtta" or args.feedtta_test_mode == "llm":
            values.append("test")
    return tuple(values)


def preflight(args: argparse.Namespace) -> None:
    if not RUNNER.is_file():
        raise UserError("missing existing evaluator: {}".format(RUNNER))
    if len(FROZEN_ROWS) != 13:
        raise UserError("frozen row count changed")
    cells = [str(item["cell_id"]) for item in FROZEN_ROWS]
    if len(set(cells)) != len(cells):
        raise UserError("duplicate frozen cell")
    for row in FROZEN_ROWS:
        if not re.fullmatch(r"[0-9a-f]{64}", str(row["development_result_sha256"])):
            raise UserError("invalid development digest for {}".format(row["cell_id"]))
        if row["benchmark"] == "r2r-ce" and row.get("ce_data_version") != "v1.2-native":
            raise UserError("R2R-CE row lost its native-v1.2 binding")
    if args.feedtta_test_mode == "llm" and not args.skip_reverie_test:
        token_value = os.environ.get("NAVTTA_LLM_FEEDBACK_TOKEN_FILE", "")
        render_value = os.environ.get(
            "NAVTTA_REVERIE_RENDER_MATTERSIM_BUILD", ""
        )
        if not token_value or not Path(token_value).is_file():
            raise UserError("FeedTTA-LLM requires NAVTTA_LLM_FEEDBACK_TOKEN_FILE")
        if not render_value or not Path(render_value).is_dir():
            raise UserError("FeedTTA-LLM requires NAVTTA_REVERIE_RENDER_MATTERSIM_BUILD")


def attempt_root(batch_id: str, row: Mapping[str, object], split: str) -> Path:
    return LOG_ROOT / batch_id / str(row["cell_id"]) / split


def latest_attempt(root: Path) -> Optional[Tuple[int, Path]]:
    values = []
    for path in root.glob("attempt-*"):
        match = re.fullmatch(r"attempt-([0-9]{3})", path.name)
        if match:
            values.append((int(match.group(1)), path))
    return max(values) if values else None


def stage_state(root: Path) -> str:
    latest = latest_attempt(root)
    if latest is None:
        return "pending"
    path = latest[1]
    exit_path = path / "exitcode"
    if not exit_path.is_file():
        return "running_or_interrupted"
    try:
        return "completed" if int(exit_path.read_text().strip()) == 0 else "failed"
    except ValueError:
        return "invalid"


def status(args: argparse.Namespace) -> int:
    counts: Dict[str, int] = {}
    for row in FROZEN_ROWS:
        for split in splits_for(row, args):
            value = stage_state(attempt_root(args.batch_id, row, split))
            counts[value] = counts.get(value, 0) + 1
            print("{} {:28s} {:8s} {}".format(
                row["queue"], row["cell_id"], split, value))
        if (row["cell_id"] == "goat-reverie-feedtta"
                and not args.skip_reverie_test
                and args.feedtta_test_mode == "skip"):
            print("{} {:28s} {:8s} skipped_no_hidden_labels".format(
                row["queue"], row["cell_id"], "test"))
            counts["skipped_no_hidden_labels"] = counts.get(
                "skipped_no_hidden_labels", 0) + 1
    print("summary={}".format(json.dumps(counts, sort_keys=True)))
    return 0


def llm_parameters(result_root: Path) -> Dict[str, object]:
    parameters = dict(LLM_CONTRACT)
    parameters.update({
        "llm_feedback_url": os.environ.get(
            "NAVTTA_LLM_FEEDBACK_URL", "http://127.0.0.1:8765"),
        "llm_feedback_token_file": str(Path(
            os.environ["NAVTTA_LLM_FEEDBACK_TOKEN_FILE"]).resolve()),
        "llm_feedback_cache_dir": str((result_root / "provider_cache").resolve()),
        "llm_feedback_transcript_path": str(
            (result_root / "llm_feedback_transcript.ndjson").resolve()),
    })
    return parameters


def job_config(batch_id: str, row: Mapping[str, object], split: str,
               result_root: Path, args: argparse.Namespace) -> Mapping[str, object]:
    parameters = dict(row["parameters"])
    if (str(row["setting"]), split) in EPISODES:
        parameters["diagnostics_expected_episodes"] = EPISODES[
            (str(row["setting"]), split)
        ]
    provider = "none"
    reported = str(row["method"])
    if row["method"] in ("feedtta", "atena"):
        provider = "task_evaluator"
    if row["method"] == "feedtta" and split == "test":
        if args.feedtta_test_mode != "llm":
            raise UserError("exact FeedTTA cannot run on hidden REVERIE test")
        parameters.update(llm_parameters(result_root))
        provider = "qwen2_vl_2b_v1"
        reported = "FeedTTA-LLM"
    return {
        "schema": "navtta.vln_tta_job.v1",
        "namespace": "tuning",
        "batch_id": batch_id,
        "stage": "frozen_positive_{}".format(split),
        "setting": row["setting"],
        "method": row["method"],
        "search_method": row["method"],
        "episodes": -1,
        "parameters": parameters,
        "selection_provenance": {
            "cell_id": row["cell_id"],
            "candidate_id": row["candidate_id"],
            "development_split": "val_unseen",
            "development_result_sha256": row["development_result_sha256"],
            "model_seed": 0,
            "episode_order_seed": 0,
            "fresh_source_restart": True,
            "reported_method_label": reported,
            "feedback_provider": provider,
        },
    }


def run_tag(row: Mapping[str, object], split: str, attempt: int) -> str:
    split_tag = "vs" if split == "val_seen" else "test"
    return "fixed-pos-v1-{}-{}-{}-a{:03d}".format(
        row["queue"], row["cell_id"], split_tag, attempt)


def command_for(row: Mapping[str, object], split: str, gpu: str,
                config_path: Path, result_root: Path, tag: str) -> List[str]:
    command = [
        "bash", str(RUNNER), str(row["setting"]), split, gpu,
        "--run-tag", tag, "--tta-config", str(config_path),
        "--result-root", str(result_root),
    ]
    if row.get("ce_data_version"):
        command.extend(["--ce-data-version", str(row["ce_data_version"])])
    return command


def prepare_attempt(batch_id: str, row: Mapping[str, object], split: str,
                    gpu: str, args: argparse.Namespace) -> Tuple[Path, List[str]]:
    root = attempt_root(batch_id, row, split)
    latest = latest_attempt(root)
    if latest is None:
        attempt = 1
    else:
        state = stage_state(root)
        if state == "completed" and args.resume:
            return latest[1], []
        if not args.resume:
            raise UserError("existing stage requires --resume: {}".format(root))
        if state != "completed" and not args.retry_failed:
            raise UserError(
                "{} is {}; pass --resume --retry-failed to retry".format(root, state)
            )
        if state == "completed":
            return latest[1], []
        attempt = latest[0] + 1
    attempt_dir = root / "attempt-{:03d}".format(attempt)
    tag = run_tag(row, split, attempt)
    result_root = (
        RESULT_ROOT / batch_id / str(row["cell_id"]) / tag / split
    ).resolve()
    config = job_config(batch_id, row, split, result_root, args)
    config_path = attempt_dir / "config.json"
    command = command_for(row, split, gpu, config_path.resolve(), result_root, tag)
    if not args.dry_run:
        attempt_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(config_path, config)
        atomic_json(attempt_dir / "job.json", {
            "created_at": utc_now(), "gpu": gpu, "row": row,
            "split": split, "run_tag": tag, "result_root": str(result_root),
            "command": command,
        })
    return attempt_dir, command


def run_stage(batch_id: str, row: Mapping[str, object], split: str,
              gpu: str, args: argparse.Namespace) -> bool:
    attempt_dir, command = prepare_attempt(batch_id, row, split, gpu, args)
    if not command:
        log("skip completed {} {}".format(row["cell_id"], split))
        return True
    if args.dry_run:
        log("dry-run gpu={} {} {} :: {}".format(
            gpu, row["cell_id"], split, " ".join(command)))
        return True
    log("launch gpu={} {} {}".format(gpu, row["cell_id"], split))
    with (attempt_dir / "console.log").open("w", encoding="utf-8", newline="\n") as output:
        process = subprocess.Popen(
            command, cwd=str(REPO_ROOT), stdout=output,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        (attempt_dir / "pid").write_text("{}\n".format(process.pid), encoding="utf-8")
        with _ACTIVE_LOCK:
            _ACTIVE[process.pid] = process
        status_code = process.wait()
        with _ACTIVE_LOCK:
            _ACTIVE.pop(process.pid, None)
    (attempt_dir / "exitcode").write_text(
        "{}\n".format(status_code), encoding="utf-8")
    atomic_json(attempt_dir / "outcome.json", {
        "completed_at": utc_now(), "exitcode": status_code,
        "cell_id": row["cell_id"], "split": split, "gpu": gpu,
    })
    log("{} gpu={} {} {} exit={}".format(
        "completed" if status_code == 0 else "failed",
        gpu, row["cell_id"], split, status_code))
    return status_code == 0


def run_pipeline(row: Mapping[str, object], gpu: str,
                 semaphore: threading.Semaphore,
                 args: argparse.Namespace) -> Mapping[str, object]:
    with semaphore:
        outcomes = {}
        for split in splits_for(row, args):
            if _STOP.is_set():
                outcomes[split] = "interrupted"
                break
            try:
                ok = run_stage(args.batch_id, row, split, gpu, args)
            except Exception as error:
                log("error gpu={} {} {}: {}".format(
                    gpu, row["cell_id"], split, error))
                outcomes[split] = "error: {}".format(error)
                break
            outcomes[split] = "completed" if ok else "failed"
            if not ok:
                break
        if (row["cell_id"] == "goat-reverie-feedtta"
                and not args.skip_reverie_test
                and args.feedtta_test_mode == "skip"):
            outcomes["test"] = "skipped_no_hidden_labels"
        return {"cell_id": row["cell_id"], "gpu": gpu, "outcomes": outcomes}


def stop_processes(signum: int, _frame: object) -> None:
    _STOP.set()
    log("received signal {}; terminating child process groups".format(signum))
    with _ACTIVE_LOCK:
        processes = list(_ACTIVE.values())
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def write_batch(args: argparse.Namespace, assignments: Sequence[Mapping[str, object]]) -> None:
    root = LOG_ROOT / args.batch_id
    payload = {
        "schema": "navtta.vln_frozen_positive_rows.batch.v1",
        "batch_id": args.batch_id,
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "script": str(SCRIPT_PATH),
        "script_sha256": hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
        "frozen_rows_sha256": digest_value(FROZEN_ROWS),
        "gpus": list(args.gpus),
        "per_gpu_concurrency": args.per_gpu_concurrency,
        "feedtta_test_mode": args.feedtta_test_mode,
        "assignments": list(assignments),
        "frozen_rows": list(FROZEN_ROWS),
    }
    path = root / "BATCH.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        keys = ("frozen_rows_sha256", "gpus", "per_gpu_concurrency",
                "feedtta_test_mode")
        if any(existing.get(key) != payload.get(key) for key in keys):
            raise UserError("existing batch identity differs: {}".format(path))
        if not args.resume and not args.dry_run:
            raise UserError("batch already exists; pass --resume")
    elif not args.dry_run:
        atomic_json(path, payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    preflight(args)
    if args.status:
        return status(args)
    assignments = [
        {"cell_id": row["cell_id"], "gpu": args.gpus[index % len(args.gpus)]}
        for index, row in enumerate(FROZEN_ROWS)
    ]
    write_batch(args, assignments)
    print("frozen positive rows: {}".format(len(FROZEN_ROWS)))
    print("gpus: {} (max {} concurrent row pipelines/GPU)".format(
        ",".join(args.gpus), args.per_gpu_concurrency))
    print("val_seen jobs: 13")
    test_count = sum("test" in splits_for(row, args) for row in FROZEN_ROWS)
    print("REVERIE test jobs: {}{}".format(
        test_count,
        " (GOAT FeedTTA test skipped: hidden labels unavailable)"
        if args.feedtta_test_mode == "skip" and not args.skip_reverie_test else "",
    ))
    semaphores = {
        gpu: threading.Semaphore(args.per_gpu_concurrency) for gpu in args.gpus
    }
    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_handlers[signum] = signal.signal(signum, stop_processes)
    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(args.gpus) * args.per_gpu_concurrency
        ) as executor:
            futures = []
            for index, row in enumerate(FROZEN_ROWS):
                gpu = args.gpus[index % len(args.gpus)]
                futures.append(executor.submit(
                    run_pipeline, row, gpu, semaphores[gpu], args))
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    if not args.dry_run:
        atomic_json(LOG_ROOT / args.batch_id / "SUMMARY.json", {
            "schema": "navtta.vln_frozen_positive_rows.summary.v1",
            "batch_id": args.batch_id,
            "completed_at": utc_now(),
            "results": sorted(results, key=lambda item: str(item["cell_id"])),
        })
    failures = [
        item for item in results
        if any(value.startswith(("failed", "error", "interrupted"))
               for value in item["outcomes"].values())
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserError as error:
        print("error: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
