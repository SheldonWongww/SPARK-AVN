#!/usr/bin/env python3
"""Shared, provenance-checked scheduler for the two AVN FeedTTA grids."""

import argparse
import ast
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
LOG_ROOT = REPO_ROOT / "avn" / "results" / "logs"
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
SEED = 0
CANONICAL_EPISODES = 2000
PREFIXES = ("net.smt_state_encoder", "action_distribution")

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
    "smt_audio": REPO_ROOT / "avn" / "baselines" / "smt_audio" / MODEL_CONFIGS["smt_audio"],
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

# Full state-fusion encoder plus action head.  Relative to the independently
# audited transformer-plus-head scope, this adds the four fusion-MLP tensors
# (139,776 scalars) and two pose-encoder tensors (96 scalars).
EXPECTED_TENSORS = {"smt_audio": 42, "enmus": 77}
EXPECTED_PARAMETERS = {"smt_audio": 1197156, "enmus": 3638884}
SOURCE_SUCCESS = {"smt_audio": 0.5415, "enmus": 0.6655}

PARAM_SCOPE = "module_prefixes"
EPISODIC = "False"
STEPS = 1
NORMALIZE_GRADIENT = "False"
SGR_SEED = 0
OPTIMIZER = "Adam"
BETA1 = "0.9"
BETA2 = "0.999"
WEIGHT_DECAY = "0.0"
OPTIMIZER_EPS = "1e-5"
MAX_GRAD_NORM = "0.0"

PROVISIONAL_PARAMETER_COUNT_VALIDATION = (
    "provisional_parameter_count_mismatch"
)
PARAMETER_COUNT_MISMATCH_MESSAGE = "adapted parameter count mismatch for enmus"

# Stage 1 and Stage 2 may use different launcher commits when the only changes
# are orchestration/provenance handling.  These paths define the actual
# FeedTTA runtime behavior that must remain byte-for-byte unchanged between the
# two stages.
FEEDTTA_STRICT_RUNTIME_PATHS = (
    "avn/baselines/smt_audio/ss_baselines/savi/ppo/ppo_trainer.py",
    "avn/baselines/smt_audio/ss_baselines/savi/config/default.py",
    "avn/baselines/smt_audio/ss_baselines/savi/config/tta_avn/single_source/"
    "smt_audio_tta_test.yaml",
    "avn/baselines/enmus/sen_baselines/enmus/config/single_source/"
    "enmus_tta_test.yaml",
    "avn/scripts/eval_smt_audio.sh",
    "avn/scripts/eval_enmus.sh",
)

FEEDTTA_SHARED_RUNTIME_PATHS = (
    "core/navtta_core/tta/tta_core.py",
    "avn/baselines/enmus/sen_baselines/enmus/ddppo/ddppo_enmus_trainer.py",
    "avn/baselines/enmus/sen_baselines/enmus/config/default.py",
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


class UserError(RuntimeError):
    """A launcher error that should be shown without a traceback."""


@dataclass(frozen=True)
class SearchPoint:
    variant: str
    lr: str
    gamma: str
    p: str
    alpha: str
    smoke_anchor: bool = False


@dataclass(frozen=True)
class ExperimentSpec:
    stage: str
    experiment: str
    log_dir_name: str
    launcher_path: Path
    experiment_spec_path: Path
    points_by_model: Mapping[str, Sequence[SearchPoint]]
    grid_metadata: Mapping[str, object]
    prerequisite_batch_id: str = ""
    reviewed_stage1_winners: Optional[Mapping[str, Mapping[str, object]]] = None
    provisional_parameter_count_models: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Job:
    job_id: int
    model_job_id: int
    model: str
    point: SearchPoint
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
    model_config_sha256: Mapping[str, str]
    experiment_spec_sha256: str
    launcher_sha256: str
    common_launcher_sha256: str
    prerequisite_metrics_sha256: str
    prerequisite_summary_sha256: str
    prerequisite_grid_sha256: str
    prerequisite_provisional_evidence_sha256: str
    prerequisite_git_commit: str
    prerequisite_runtime_compatible: bool


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


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_evidence_files(paths: Sequence[Path]) -> str:
    """Hash both evidence paths and contents in a stable order."""
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=str):
        try:
            relative = path.relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            relative = str(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_marker(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise UserError("cannot read {}: {}".format(label, error)) from error


def parse_console_aggregate_metrics(path: Path) -> Dict[str, float]:
    """Read the final nine Habitat aggregate metrics from a console log."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise UserError("cannot read Stage-1 console metrics: {}".format(error)) from error
    output: Dict[str, float] = {}
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    for metric in METRICS:
        matches = re.findall(
            r"Average episode {}:\s*({})".format(re.escape(metric), number),
            content,
        )
        require(matches, "Stage-1 console is missing aggregate {}".format(metric))
        value = float(matches[-1])
        require(math.isfinite(value), "Stage-1 console has non-finite {}".format(metric))
        output[metric] = value
    return output


def require_stage1_runtime_compatible(
    stage1_commit: str, current_commit: str
) -> bool:
    """Allow ATENA/source-control edits while rejecting FeedTTA changes.

    ATENA and FeedTTA share ``tta_core.py`` and the AVN evaluation trainers.
    A whole-file hash would therefore reject an ATENA-only implementation
    change even when FeedTTA's adapter and execution path are untouched.  The
    strict files are still compared byte-for-byte; shared Python files are
    compared as normalized ASTs with only the reviewed ATENA/source-argmax
    additions removed.
    """
    if stage1_commit == current_commit:
        return True
    completed = subprocess.run(
        (
            "git",
            "diff",
            "--quiet",
            stage1_commit,
            current_commit,
            "--",
            *FEEDTTA_STRICT_RUNTIME_PATHS,
        ),
        cwd=str(REPO_ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode == 1:
        raise UserError(
            "FeedTTA runtime behavior changed after Stage 1; rerun Stage 1 before Stage 2"
        )
    if completed.returncode != 0:
        raise UserError(
            completed.stderr.strip() or "cannot compare Stage-1 runtime compatibility"
        )

    for path in FEEDTTA_SHARED_RUNTIME_PATHS:
        stage1_source = run_capture(("git", "show", "{}:{}".format(stage1_commit, path)))
        current_source = run_capture(("git", "show", "{}:{}".format(current_commit, path)))
        if _feedtta_runtime_ast(path, stage1_source) != _feedtta_runtime_ast(
            path, current_source
        ):
            raise UserError(
                "FeedTTA shared runtime behavior changed after Stage 1 in {}; "
                "rerun Stage 1 before Stage 2".format(path)
            )
    return True


class _FeedTTASharedRuntimeNormalizer(ast.NodeTransformer):
    """Remove only reviewed non-FeedTTA edits from shared runtime ASTs."""

    def __init__(self, path: str) -> None:
        self.path = path

    def visit_Module(self, node):
        if self.path.endswith("tta_core.py"):
            node.body = [
                item
                for item in node.body
                if not (
                    isinstance(item, ast.ClassDef) and item.name == "ATENAAdapter"
                )
                and not (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "_tree_tensor_bytes"
                )
            ]
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if self.path.endswith("config/default.py"):
            targets = [ast.unparse(target) for target in node.targets]
            if any(target.endswith(".EVAL.ACTION_SELECTION") for target in targets):
                return None
        if self.path.endswith("ddppo_enmus_trainer.py") and any(
            isinstance(target, ast.Name) and target.id == "action_selection"
            for target in node.targets
        ):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if self.path.endswith("ddppo_enmus_trainer.py"):
            if ast.unparse(node.test) == (
                "action_selection not in ('sample', 'argmax')"
            ):
                return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if self.path.endswith("ddppo_enmus_trainer.py"):
            try:
                rendered = ast.unparse(node)
            except Exception:
                rendered = ""
            if "[EVAL] action_selection=%s" in rendered:
                return None
        return self.generic_visit(node)

    def visit_keyword(self, node):
        node = self.generic_visit(node)
        if (
            self.path.endswith("ddppo_enmus_trainer.py")
            and node.arg == "deterministic"
            and any(
                isinstance(item, ast.Name) and item.id == "action_selection"
                for item in ast.walk(node.value)
            )
        ):
            node.value = ast.Constant(value=False)
        return node


def _feedtta_runtime_ast(path: str, source: str) -> str:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        raise UserError("cannot parse shared FeedTTA runtime {}: {}".format(path, error))
    normalized = _FeedTTASharedRuntimeNormalizer(path).visit(tree)
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def load_provisional_enmus_stage1_rows(
    prerequisite_dir: Path, grid_rows: Sequence[Mapping[str, str]]
) -> Tuple[List[Dict[str, str]], str]:
    """Recover runner-complete ENMuS rows rejected only by count validation."""
    rows: List[Dict[str, str]] = []
    evidence: List[Path] = []
    enmus_rows = [
        row
        for row in grid_rows
        if row.get("model") == "enmus" and row.get("stage") == "stage1"
    ]
    require(len(enmus_rows) == 24, "Stage-1 ENMuS grid row count mismatch")
    for grid_row in enmus_rows:
        run_tag = grid_row.get("run_tag", "")
        require(bool(run_tag), "Stage-1 ENMuS grid row has no run_tag")
        job_dir = prerequisite_dir / "enmus" / "jobs" / run_tag
        marker_paths = {
            "runner_exitcode": job_dir / "runner_exitcode",
            "exitcode": job_dir / "exitcode",
            "validation": job_dir / "validation",
            "parameters": job_dir / "parameters.env",
            "console": job_dir / "console.log",
        }
        require(
            read_marker(marker_paths["runner_exitcode"], "Stage-1 runner exitcode") == "0",
            "Stage-1 ENMuS runner did not finish successfully: {}".format(run_tag),
        )
        require(
            read_marker(marker_paths["exitcode"], "Stage-1 composite exitcode") == "90",
            "Stage-1 ENMuS failure was not the expected post-run validation code: {}".format(
                run_tag
            ),
        )
        require(
            read_marker(marker_paths["validation"], "Stage-1 validation") == "failed",
            "Stage-1 ENMuS validation marker is unexpected: {}".format(run_tag),
        )
        console_text = read_marker(marker_paths["console"], "Stage-1 console")
        require(
            PARAMETER_COUNT_MISMATCH_MESSAGE in console_text,
            "Stage-1 ENMuS failure is not the reviewed parameter-count mismatch: {}".format(
                run_tag
            ),
        )
        metrics = parse_console_aggregate_metrics(marker_paths["console"])
        row = dict(grid_row)
        row.update({name: repr(value) for name, value in metrics.items()})
        row["status"] = "0"
        row["validation"] = PROVISIONAL_PARAMETER_COUNT_VALIDATION
        row["relative_param_drift"] = ""
        rows.append(row)
        evidence.extend(marker_paths.values())
    return rows, sha256_evidence_files(evidence)


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
    parts = value.split(",")
    if len(parts) != 4 or any(
        re.fullmatch(r"0|[1-9][0-9]*", part) is None for part in parts
    ):
        raise argparse.ArgumentTypeError(
            "expected exactly four comma-separated GPU ids without leading zeros"
        )
    if len(set(parts)) != 4:
        raise argparse.ArgumentTypeError("GPU ids must be distinct")
    return tuple(parts)


def positive_integer(value: str) -> int:
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return int(value)


def add_common_arguments(
    parser: argparse.ArgumentParser, default_batch_prefix: str
) -> None:
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
        help="stable batch id (default: {}-UTC timestamp)".format(
            default_batch_prefix
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the immutable named batch and revalidate completed jobs",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="permit tracked/uncommitted launcher code only for a smoke run",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="select the declared paper-anchor configuration for each model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print and validate the plan without filesystem or process changes",
    )
    parser.set_defaults(_default_batch_prefix=default_batch_prefix)


def finalize_common_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    if args.episodes > CANONICAL_EPISODES:
        parser.error("--episodes cannot exceed the 2,000-episode canonical stream")
    if not args.smoke and args.episodes != CANONICAL_EPISODES:
        parser.error("full FeedTTA runs require exactly 2,000 episodes")
    if args.allow_dirty and not args.smoke:
        parser.error("--allow-dirty is permitted only with --smoke")
    if args.resume and not args.batch_id:
        parser.error("--resume requires an explicit --batch-id")
    if not args.batch_id:
        args.batch_id = "{}-{}".format(
            args._default_batch_prefix, timestamp_slug()
        )
    if re.fullmatch(r"[A-Za-z0-9._-]+", args.batch_id) is None:
        parser.error(
            "--batch-id may contain only letters, numbers, dot, underscore, and hyphen"
        )
    if len(args.batch_id) > 64:
        parser.error("--batch-id cannot exceed 64 characters")
    return args


def value_slug(value: str) -> str:
    return value.replace(".", "p").replace("+", "p").replace("-", "m")


def make_run_tag(
    spec: ExperimentSpec,
    batch_id: str,
    job_id: int,
    model: str,
    point: SearchPoint,
) -> str:
    stage_slug = spec.stage.replace("stage", "s")
    model_slug = "smt" if model == "smt_audio" else "enm"
    return (
        "ft-{}-{}-j{:03d}-{}-{}-lr{}-g{}-p{}-a{}".format(
            stage_slug,
            batch_id,
            job_id,
            model_slug,
            point.variant,
            value_slug(point.lr),
            value_slug(point.gamma),
            value_slug(point.p),
            value_slug(point.alpha),
        )
    )


def build_plan(spec: ExperimentSpec, args: argparse.Namespace) -> List[Job]:
    if tuple(spec.points_by_model.keys()) != MODELS:
        raise UserError("FeedTTA specification must contain both models in order")
    plan: List[Job] = []
    global_job_id = 0
    for model_index, model in enumerate(MODELS):
        points = tuple(spec.points_by_model[model])
        if not points:
            raise UserError("empty FeedTTA grid for {}".format(model))
        identities = {
            (point.variant, point.lr, point.gamma, point.p, point.alpha)
            for point in points
        }
        if len(identities) != len(points):
            raise UserError("duplicate FeedTTA grid point for {}".format(model))
        anchors = [index for index, point in enumerate(points) if point.smoke_anchor]
        if len(anchors) != 1:
            raise UserError("{} must declare exactly one smoke anchor".format(model))
        for model_job_id, point in enumerate(points):
            selected = not args.smoke or point.smoke_anchor
            if selected:
                gpu_index = (model_job_id + model_index) % len(args.gpus)
                plan.append(
                    Job(
                        job_id=global_job_id,
                        model_job_id=model_job_id,
                        model=model,
                        point=point,
                        gpu_index=gpu_index,
                        gpu=args.gpus[gpu_index],
                        run_tag=make_run_tag(
                            spec,
                            args.batch_id,
                            global_job_id,
                            model,
                            point,
                        ),
                    )
                )
            global_job_id += 1

    expected = 2 if args.smoke else sum(
        len(spec.points_by_model[model]) for model in MODELS
    )
    if len(plan) != expected or len({job.run_tag for job in plan}) != expected:
        raise UserError("internal FeedTTA plan size or tag collision")
    if not args.smoke:
        counts = [sum(job.gpu_index == index for job in plan) for index in range(4)]
        if max(counts) - min(counts) > 1:
            raise UserError("FeedTTA jobs are not balanced across four GPUs")
        for model in MODELS:
            model_jobs = [job for job in plan if job.model == model]
            if len(model_jobs) >= 4 and len({job.gpu_index for job in model_jobs}) != 4:
                raise UserError("{} is not represented on every GPU".format(model))
    return plan


def plan_csv(plan: Iterable[Job], episodes: int, stage: str) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "job_id",
            "model_job_id",
            "run_tag",
            "gpu",
            "stage",
            "model",
            "source_setting",
            "method",
            "variant",
            "scope",
            "expected_tensors",
            "expected_parameters",
            "lr",
            "gamma",
            "p",
            "alpha",
            "sgr_seed",
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
                stage,
                job.model,
                "single_source",
                "feedtta",
                job.point.variant,
                "full_state_fusion_plus_action_head",
                EXPECTED_TENSORS[job.model],
                EXPECTED_PARAMETERS[job.model],
                job.point.lr,
                job.point.gamma,
                job.point.p,
                job.point.alpha,
                SGR_SEED,
                SEED,
                episodes,
            )
        )
    return output.getvalue()


def print_plan(
    spec: ExperimentSpec, args: argparse.Namespace, plan: Sequence[Job]
) -> None:
    print("AVN FeedTTA {} grid".format(spec.stage))
    print("  models:              smt_audio,enmus")
    print("  source/split:        single_source/val")
    print("  seed/episodes:       {}/{}".format(SEED, args.episodes))
    print("  GPUs:                {}".format(",".join(args.gpus)))
    print("  combined jobs/GPU:   {}".format(args.jobs_per_gpu))
    print("  selected jobs:       {}".format(len(plan)))
    print("  batch:               {}".format(args.batch_id))
    print("  log root:            {}".format(
        LOG_ROOT / spec.log_dir_name / args.batch_id
    ))
    print(plan_csv(plan, args.episodes, spec.stage), end="")


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


def _require_tracked(path: Path, allow_untracked_smoke: bool) -> None:
    try:
        relative = path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise UserError("provenance source is outside repository: {}".format(path)) from error
    completed = subprocess.run(
        ("git", "cat-file", "-e", "HEAD:{}".format(relative.as_posix())),
        cwd=str(REPO_ROOT),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0 and not allow_untracked_smoke:
        raise UserError("launcher dependency is not tracked by HEAD: {}".format(relative))


def validate_preflight(
    spec: ExperimentSpec, args: argparse.Namespace
) -> Provenance:
    common_launcher = Path(__file__).resolve()
    required = (
        spec.launcher_path.resolve(),
        common_launcher,
        spec.experiment_spec_path.resolve(),
        DATASET,
        FINGERPRINT_TOOL,
        MANIFEST_VALIDATOR,
        *(MODEL_RUNNERS[model] for model in MODELS),
        *(MODEL_CONFIG_PATHS[model] for model in MODELS),
        *(MODEL_CHECKPOINTS[model] for model in MODELS),
        *ENMUS_AUXILIARY.values(),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise UserError("missing required files:\n  " + "\n  ".join(missing))
    for command in ("git", "bash"):
        if shutil.which(command) is None:
            raise UserError("{} is unavailable".format(command))

    allow_untracked_smoke = bool(args.smoke and args.allow_dirty)
    for path in (
        spec.launcher_path,
        common_launcher,
        spec.experiment_spec_path,
        FINGERPRINT_TOOL,
        MANIFEST_VALIDATOR,
        *(MODEL_RUNNERS[model] for model in MODELS),
        *(MODEL_CONFIG_PATHS[model] for model in MODELS),
    ):
        _require_tracked(path, allow_untracked_smoke)

    commit = run_capture(("git", "rev-parse", "HEAD"))
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise UserError("invalid Git commit: {}".format(commit))
    tracked_status = run_capture(
        ("git", "status", "--porcelain", "--untracked-files=no")
    )
    if tracked_status and not args.allow_dirty:
        raise UserError(
            "tracked worktree changes detected; commit them before a full run"
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

    model_config_hashes = {
        model: sha256_file(MODEL_CONFIG_PATHS[model]) for model in MODELS
    }

    prerequisite_metrics_sha256 = ""
    prerequisite_summary_sha256 = ""
    prerequisite_grid_sha256 = ""
    prerequisite_provisional_evidence_sha256 = ""
    prerequisite_git_commit = ""
    prerequisite_runtime_compatible = False
    if spec.prerequisite_batch_id:
        prerequisite_dir = (
            LOG_ROOT / "feedtta_stage1" / spec.prerequisite_batch_id
        )
        summary_path = prerequisite_dir / "SUMMARY.json"
        metrics_path = prerequisite_dir / "metrics.csv"
        grid_path = prerequisite_dir / "grid.csv"
        batch_path = prerequisite_dir / "batch.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            stage1_batch = json.loads(batch_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise UserError(
                "cannot read prerequisite Stage-1 metadata: {}".format(error)
            ) from error
        require(isinstance(summary, dict), "invalid prerequisite Stage-1 summary")
        require(isinstance(stage1_batch, dict), "invalid prerequisite Stage-1 batch")
        require(summary.get("stage") == "stage1", "prerequisite is not a Stage-1 batch")
        require(summary.get("smoke") is False, "Stage 2 cannot use a smoke prerequisite")
        require(summary.get("expected") == 48, "prerequisite Stage-1 job count mismatch")
        require(summary.get("missing") == 0, "prerequisite Stage-1 has missing jobs")
        prerequisite_git_commit = str(summary.get("git_commit", ""))
        require(
            re.fullmatch(r"[0-9a-f]{40}", prerequisite_git_commit) is not None,
            "prerequisite Stage-1 Git commit is invalid",
        )
        prerequisite_runtime_compatible = require_stage1_runtime_compatible(
            prerequisite_git_commit, commit
        )
        require(
            summary.get("dataset_index_sha256") == dataset_hash,
            "Stage-1 dataset digest differs from Stage 2",
        )
        require(
            summary.get("stream_order_sha256") == order_hash,
            "Stage-1 episode order differs from Stage 2",
        )
        require(
            summary.get("stream_content_sha256") == content_hash,
            "Stage-1 episode content differs from Stage 2",
        )
        require(
            stage1_batch.get("checkpoint_sha256") == checkpoint_hashes,
            "Stage-1 checkpoint digests differ from Stage 2",
        )
        require(
            stage1_batch.get("model_config_sha256") == model_config_hashes,
            "Stage-1 model config digests differ from Stage 2",
        )
        if not metrics_path.is_file():
            raise UserError("prerequisite Stage-1 metrics.csv is missing")
        try:
            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                stage1_rows = list(csv.DictReader(handle))
            with grid_path.open("r", encoding="utf-8", newline="") as handle:
                stage1_grid_rows = list(csv.DictReader(handle))
        except OSError as error:
            raise UserError("cannot read prerequisite Stage-1 CSV evidence") from error
        require(len(stage1_grid_rows) == 48, "prerequisite Stage-1 grid row count mismatch")

        reviewed = spec.reviewed_stage1_winners
        if reviewed is None:
            require(summary.get("complete") is True, "prerequisite Stage-1 batch is incomplete")
            require(summary.get("validated") == 48, "prerequisite Stage-1 validation count mismatch")
            require(len(stage1_rows) == 48, "prerequisite Stage-1 metrics row count mismatch")
            require(
                all(
                    row.get("stage") == "stage1"
                    and row.get("method") == "feedtta"
                    and row.get("validation") == "ok"
                    and row.get("status") == "0"
                    for row in stage1_rows
                ),
                "prerequisite Stage-1 metrics contain nonvalidated rows",
            )
        else:
            require(set(reviewed) == set(MODELS), "reviewed Stage-1 winners are incomplete")
            per_model = summary.get("per_model")
            require(isinstance(per_model, dict), "Stage-1 per-model summary is missing")
            require(
                per_model.get("smt_audio") == {
                    "expected": 24,
                    "successful": 24,
                    "failed": 0,
                    "missing": 0,
                    "validated": 24,
                    "metrics": 24,
                },
                "Stage-1 SMT+Audio evidence no longer matches the reviewed batch",
            )
            require(
                per_model.get("enmus") == {
                    "expected": 24,
                    "successful": 0,
                    "failed": 24,
                    "missing": 0,
                    "validated": 0,
                    "metrics": 0,
                },
                "Stage-1 ENMuS evidence no longer matches the reviewed batch",
            )
            require(len(stage1_rows) == 48, "Stage-1 metric index row count mismatch")
            validated_rows = [
                row
                for row in stage1_rows
                if row.get("model") == "smt_audio"
                and row.get("validation") == "ok"
                and row.get("status") == "0"
            ]
            require(
                len(validated_rows) == 24,
                "validated Stage-1 metric row count mismatch",
            )
            require(
                all(
                    row.get("model") == "smt_audio"
                    and row.get("stage") == "stage1"
                    and row.get("method") == "feedtta"
                    and row.get("validation") == "ok"
                    and row.get("status") == "0"
                    for row in validated_rows
                ),
                "validated Stage-1 metrics contain unexpected rows",
            )
            provisional_rows, prerequisite_provisional_evidence_sha256 = (
                load_provisional_enmus_stage1_rows(
                    prerequisite_dir, stage1_grid_rows
                )
            )
            stage1_rows = validated_rows + provisional_rows

        for model in MODELS:
            selected_pairs = {
                (point.lr, point.gamma) for point in spec.points_by_model[model]
            }
            require(
                len(selected_pairs) == 1,
                "Stage 2 must hold one LR/gamma pair per model",
            )
            selected_lr, selected_gamma = next(iter(selected_pairs))
            model_rows = [row for row in stage1_rows if row.get("model") == model]
            require(len(model_rows) == 24, "Stage-1 model row count mismatch")
            ranked = []
            for row in model_rows:
                try:
                    success = float(row["success"])
                    spl = float(row["spl"])
                except (KeyError, TypeError, ValueError) as error:
                    raise UserError("Stage-1 selection metrics are incomplete") from error
                require(
                    all(math.isfinite(value) for value in (success, spl)),
                    "Stage-1 selection metrics are non-finite",
                )
                if success + 1e-12 >= SOURCE_SUCCESS[model]:
                    drift_text = row.get("relative_param_drift", "")
                    try:
                        drift = float(drift_text)
                    except (TypeError, ValueError):
                        drift = math.inf
                    require(
                        math.isfinite(drift) or math.isinf(drift),
                        "Stage-1 parameter drift is invalid",
                    )
                    ranked.append((spl, success, -drift, row))
            require(
                bool(ranked),
                "no {} Stage-1 point satisfies SR >= Source; review the plan".format(model),
            )
            winner = max(ranked, key=lambda item: item[:3])[3]
            if reviewed is not None:
                expected = reviewed[model]
                require(
                    str(expected.get("lr")) == selected_lr
                    and str(expected.get("gamma")) == selected_gamma,
                    "CLI selection does not match the reviewed Stage-1 choice for {}".format(
                        model
                    ),
                )
                require(
                    str(winner.get("job_id")) == str(expected.get("job_id")),
                    "recomputed Stage-1 winner differs from the reviewed job for {}".format(
                        model
                    ),
                )
                for metric in ("success", "spl"):
                    require(
                        math.isclose(
                            float(winner[metric]),
                            float(expected[metric]),
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        ),
                        "reviewed Stage-1 {} differs for {}".format(metric, model),
                    )
            require(
                winner.get("lr") == selected_lr
                and winner.get("gamma") == selected_gamma,
                "selected {}/{} is not the documented Stage-1 winner for {} "
                "(expected {}/{})".format(
                    selected_lr,
                    selected_gamma,
                    model,
                    winner.get("lr"),
                    winner.get("gamma"),
                ),
            )
            matches = [
                row
                for row in stage1_rows
                if row.get("model") == model
                and row.get("lr") == selected_lr
                and row.get("gamma") == selected_gamma
            ]
            require(
                len(matches) == 1,
                "selected {}/{} was not a unique reviewed Stage-1 point".format(
                    selected_lr, selected_gamma
                ),
            )
        prerequisite_metrics_sha256 = sha256_file(metrics_path)
        prerequisite_summary_sha256 = sha256_file(summary_path)
        prerequisite_grid_sha256 = sha256_file(grid_path)

    return Provenance(
        git_commit=commit,
        tracked_status=tracked_status,
        tracked_worktree_dirty=bool(tracked_status),
        dataset_index_sha256=dataset_hash,
        stream_order_sha256=order_hash,
        stream_content_sha256=content_hash,
        checkpoint_sha256=checkpoint_hashes,
        auxiliary_sha256=auxiliary_hashes,
        model_config_sha256=model_config_hashes,
        experiment_spec_sha256=sha256_file(spec.experiment_spec_path),
        launcher_sha256=sha256_file(spec.launcher_path),
        common_launcher_sha256=sha256_file(common_launcher),
        prerequisite_metrics_sha256=prerequisite_metrics_sha256,
        prerequisite_summary_sha256=prerequisite_summary_sha256,
        prerequisite_grid_sha256=prerequisite_grid_sha256,
        prerequisite_provisional_evidence_sha256=(
            prerequisite_provisional_evidence_sha256
        ),
        prerequisite_git_commit=prerequisite_git_commit,
        prerequisite_runtime_compatible=prerequisite_runtime_compatible,
    )


def require_repository_unchanged(provenance: Provenance) -> None:
    if run_capture(("git", "rev-parse", "HEAD")) != provenance.git_commit:
        raise UserError("repository HEAD changed while the grid was running")
    status = run_capture(("git", "status", "--porcelain", "--untracked-files=no"))
    if status != provenance.tracked_status:
        raise UserError("tracked worktree changed while the grid was running")


def batch_spec(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    provenance: Provenance,
    plan: Sequence[Job],
) -> Mapping[str, object]:
    return {
        "experiment": spec.experiment,
        "stage": spec.stage,
        "result_role": "hyperparameter_search",
        "protocol": "source_tent_aligned_single_source_val_seed0",
        "models": list(MODELS),
        "source_setting": "single_source",
        "split": "val",
        "method": "feedtta",
        "feedback_supervision": "binary_episode_success_oracle",
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
        "grid": dict(spec.grid_metadata),
        "scope": "full_state_fusion_plus_action_head",
        "trainable_prefixes": list(PREFIXES),
        "expected_tensors": EXPECTED_TENSORS,
        "expected_parameters": EXPECTED_PARAMETERS,
        "source_success_constraints": SOURCE_SUCCESS,
        "param_scope": PARAM_SCOPE,
        "episodic": EPISODIC,
        "steps": STEPS,
        "normalize_gradient": NORMALIZE_GRADIENT,
        "sgr_seed": SGR_SEED,
        "sgr_rule": "main_text_eq5_unselected_coordinates_scaled",
        "optimizer": OPTIMIZER,
        "beta1": BETA1,
        "beta2": BETA2,
        "weight_decay": WEIGHT_DECAY,
        "optimizer_eps": OPTIMIZER_EPS,
        "max_grad_norm": MAX_GRAD_NORM,
        "git_commit": provenance.git_commit,
        "tracked_worktree_dirty": provenance.tracked_worktree_dirty,
        "allow_dirty": args.allow_dirty,
        "dataset_index_sha256": provenance.dataset_index_sha256,
        "stream_order_sha256": provenance.stream_order_sha256,
        "stream_content_sha256": provenance.stream_content_sha256,
        "checkpoint_sha256": dict(provenance.checkpoint_sha256),
        "model_config_sha256": dict(provenance.model_config_sha256),
        "enmus_auxiliary_sha256": dict(provenance.auxiliary_sha256),
        "experiment_spec": str(spec.experiment_spec_path.relative_to(REPO_ROOT)),
        "experiment_spec_sha256": provenance.experiment_spec_sha256,
        "launcher_sha256": provenance.launcher_sha256,
        "common_launcher_sha256": provenance.common_launcher_sha256,
        "prerequisite_stage1_batch_id": spec.prerequisite_batch_id,
        "prerequisite_stage1_metrics_sha256": provenance.prerequisite_metrics_sha256,
        "prerequisite_stage1_summary_sha256": provenance.prerequisite_summary_sha256,
        "prerequisite_stage1_grid_sha256": provenance.prerequisite_grid_sha256,
        "prerequisite_stage1_provisional_evidence_sha256": (
            provenance.prerequisite_provisional_evidence_sha256
        ),
        "prerequisite_stage1_git_commit": provenance.prerequisite_git_commit,
        "prerequisite_stage1_runtime_compatible": (
            provenance.prerequisite_runtime_compatible
        ),
        "provisional_parameter_count_models": list(
            spec.provisional_parameter_count_models
        ),
    }


def initialize_batch(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    provenance: Provenance,
    plan: Sequence[Job],
) -> Path:
    batch_dir = LOG_ROOT / spec.log_dir_name / args.batch_id
    expected_spec = json_text(batch_spec(spec, args, provenance, plan))
    expected_plan = plan_csv(plan, args.episodes, spec.stage)
    if args.resume:
        if not batch_dir.is_dir():
            raise UserError("resume batch does not exist: {}".format(batch_dir))
        batch_path = batch_dir / "batch.json"
        plan_path = batch_dir / "grid.csv"
        if not batch_path.is_file() or batch_path.read_text(encoding="utf-8") != expected_spec:
            raise UserError("resume arguments or immutable inputs do not match batch.json")
        if not plan_path.is_file() or plan_path.read_text(encoding="utf-8") != expected_plan:
            raise UserError("resume plan does not match grid.csv")
    else:
        if batch_dir.exists():
            raise UserError("batch already exists: {}".format(batch_dir))
        batch_dir.parent.mkdir(parents=True, exist_ok=True)
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
        "TTA.FEEDTTA.LR": job.point.lr,
        "TTA.FEEDTTA.P": job.point.p,
        "TTA.FEEDTTA.ALPHA": job.point.alpha,
        "TTA.FEEDTTA.SGR_SEED": str(SGR_SEED),
        "TTA.FEEDTTA.GAMMA": job.point.gamma,
        "TTA.FEEDTTA.NORMALIZE_GRADIENT": NORMALIZE_GRADIENT,
        "TTA.FEEDTTA.PARAM_SCOPE": PARAM_SCOPE,
        "TTA.FEEDTTA.TRAINABLE_PREFIXES": json.dumps(
            list(PREFIXES), separators=(",", ":")
        ),
        "TTA.FEEDTTA.OPTIMIZER": OPTIMIZER,
        "TTA.FEEDTTA.BETA1": BETA1,
        "TTA.FEEDTTA.BETA2": BETA2,
        "TTA.FEEDTTA.WEIGHT_DECAY": WEIGHT_DECAY,
        "TTA.FEEDTTA.EPS": OPTIMIZER_EPS,
        "TTA.FEEDTTA.MAX_GRAD_NORM": MAX_GRAD_NORM,
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
        "feedtta",
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


def parameters_text(
    spec: ExperimentSpec,
    job: Job,
    args: argparse.Namespace,
    provenance: Provenance,
) -> str:
    values = (
        ("job_id", job.job_id),
        ("model_job_id", job.model_job_id),
        ("run_tag", job.run_tag),
        ("gpu", job.gpu),
        ("stage", spec.stage),
        ("model", job.model),
        ("source_setting", "single_source"),
        ("split", "val"),
        ("method", "feedtta"),
        ("variant", job.point.variant),
        (
            "action_selection",
            "sample" if job.model == "smt_audio" else "native_sample_deterministic_false",
        ),
        ("scope", "full_state_fusion_plus_action_head"),
        ("lr", job.point.lr),
        ("gamma", job.point.gamma),
        ("p", job.point.p),
        ("alpha", job.point.alpha),
        ("sgr_seed", SGR_SEED),
        ("seed", SEED),
        ("episodes", args.episodes),
        ("git_commit", provenance.git_commit),
        ("checkpoint_sha256", provenance.checkpoint_sha256[job.model]),
        ("dataset_index_sha256", provenance.dataset_index_sha256),
        ("stream_order_sha256", provenance.stream_order_sha256),
        ("stream_content_sha256", provenance.stream_content_sha256),
        ("expected_tensors", EXPECTED_TENSORS[job.model]),
        ("expected_parameters", EXPECTED_PARAMETERS[job.model]),
    )
    return "".join("{}={}\n".format(key, value) for key, value in values)


def archive_previous(job_dir: Path) -> None:
    stamp = timestamp_slug()
    names = (
        "console.log",
        "runner_exitcode",
        "exitcode",
        "validation",
        "manifest.path",
        "parameters.env",
        "metrics.json",
    )
    for name in names:
        source = job_dir / name
        if source.exists():
            destination = job_dir / "{}.previous.{}".format(name, stamp)
            counter = 1
            while destination.exists():
                destination = job_dir / "{}.previous.{}.{}".format(
                    name, stamp, counter
                )
                counter += 1
            source.rename(destination)


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


def expected_regularizer_variant(point: SearchPoint) -> str:
    if math.isclose(float(point.p), 0.0, abs_tol=1e-15):
        return "none"
    alpha = float(point.alpha)
    if alpha < 0.0:
        return "stochastic_gradient_reversion"
    if math.isclose(alpha, 0.0, abs_tol=1e-15):
        return "gradient_dropout"
    return "gradient_scaling"


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
    require(
        set(override_map) == set(expected_override_map(job, episodes)),
        "manifest contains unplanned config overrides",
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


def _finite_diagnostic(diagnostics: Mapping[str, object], name: str) -> float:
    try:
        value = float(diagnostics[name])
    except (KeyError, TypeError, ValueError) as error:
        raise UserError("missing or nonnumeric diagnostic: {}".format(name)) from error
    require(math.isfinite(value), "non-finite diagnostic: {}".format(name))
    return value


def validate_stats_and_diagnostics(
    manifest_path: Path,
    job: Job,
    episodes: int,
    allow_parameter_count_mismatch: bool = False,
) -> Tuple[Dict[str, float], Path, Mapping[str, object]]:
    run_dir = manifest_path.resolve().parent
    stats_path = run_dir / "raw" / "model" / "tb" / "val_stats_{}.json".format(SEED)
    diagnostics_path = (
        run_dir / "raw" / "model" / "tb" / "tta_diagnostics_{}.json".format(SEED)
    )
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError("cannot read FeedTTA result artifacts: {}".format(error)) from error

    require(isinstance(stats, dict), "episode statistics are not a dictionary")
    require(len(stats) == episodes, "episode statistics count mismatch")
    for episode_key, episode_stats in stats.items():
        require(isinstance(episode_stats, dict), "invalid episode statistics")
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
    require(diagnostics.get("updates") == episodes, "FeedTTA must update once per episode")
    require(diagnostics.get("optimizer") == OPTIMIZER, "FeedTTA optimizer mismatch")
    require(diagnostics.get("param_scope") == PARAM_SCOPE, "scope mode mismatch")
    require(diagnostics.get("trainable_prefixes") == list(PREFIXES), "prefix list mismatch")
    try:
        adapted_parameter_count = int(diagnostics["adapted_parameter_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise UserError("adapted parameter count is missing or invalid") from error
    require(adapted_parameter_count > 0, "adapted parameter count must be positive")
    if not allow_parameter_count_mismatch:
        require(
            adapted_parameter_count == EXPECTED_PARAMETERS[job.model],
            "adapted parameter count mismatch for {}".format(job.model),
        )
    names = diagnostics.get("adapted_parameter_names")
    require(isinstance(names, list), "adapted parameter names are missing")
    require(bool(names), "adapted parameter names are empty")
    if not allow_parameter_count_mismatch:
        require(len(names) == EXPECTED_TENSORS[job.model], "adapted tensor count mismatch")
    require(len(set(names)) == len(names), "duplicate adapted parameter names")
    require(
        all(
            any(name == prefix or name.startswith(prefix + ".") for prefix in PREFIXES)
            for name in names
        ),
        "adapted parameter outside declared prefixes",
    )

    require(diagnostics.get("feedback_type") == "binary_episode_success", "feedback type mismatch")
    require(diagnostics.get("feedback_values") == "+1_success_-1_failure", "feedback values mismatch")
    require(diagnostics.get("action_selection_protocol") == "sample_from_policy", "action protocol mismatch")
    require(diagnostics.get("update_timing") == "once_after_episode_feedback", "update timing mismatch")
    require(diagnostics.get("sgr_rng") == "dedicated_torch_generator", "SGR RNG mismatch")
    require(diagnostics.get("sgr_rule") == "main_text_eq5_unselected_coordinates_scaled", "SGR rule mismatch")
    require(diagnostics.get("regularizer_variant") == expected_regularizer_variant(job.point), "regularizer variant mismatch")
    require(diagnostics.get("sgr_seed") == SGR_SEED, "SGR seed mismatch")
    require(diagnostics.get("normalize_gradient") is False, "trajectory normalization must be disabled")
    require(diagnostics.get("episodic") is False, "FeedTTA must be continual")
    require(diagnostics.get("trajectory_gradient_storage") == "online_discounted_accumulator", "gradient storage mismatch")
    require(
        diagnostics.get("gradient_accumulator_elements") == adapted_parameter_count,
        "gradient accumulator size differs from the actual adapted parameter count",
    )
    require(diagnostics.get("current_trajectory_steps") == 0, "unfinished trajectory remains after evaluation")

    current_lr = _finite_diagnostic(diagnostics, "current_lr")
    gamma = _finite_diagnostic(diagnostics, "gamma")
    probability = _finite_diagnostic(diagnostics, "reversal_probability")
    alpha = _finite_diagnostic(diagnostics, "reversal_scale")
    selected_fraction = _finite_diagnostic(diagnostics, "mean_sgr_selected_fraction")
    require(math.isclose(current_lr, float(job.point.lr), rel_tol=1e-12), "FeedTTA LR mismatch")
    require(math.isclose(gamma, float(job.point.gamma), rel_tol=1e-12), "FeedTTA gamma mismatch")
    require(math.isclose(probability, float(job.point.p), rel_tol=1e-12, abs_tol=1e-15), "FeedTTA p mismatch")
    require(math.isclose(alpha, float(job.point.alpha), rel_tol=1e-12, abs_tol=1e-15), "FeedTTA alpha mismatch")
    require(0.0 <= selected_fraction <= 1.0, "invalid mean SGR selected fraction")
    if math.isclose(float(job.point.p), 0.0, abs_tol=1e-15):
        require(math.isclose(selected_fraction, 0.0, abs_tol=1e-15), "p=0 selected SGR dimensions")
    else:
        sample_count = adapted_parameter_count * episodes
        expected_p = float(job.point.p)
        standard_error = math.sqrt(
            expected_p * (1.0 - expected_p) / sample_count
        )
        tolerance = max(0.002 if episodes < 100 else 0.0005, 10.0 * standard_error)
        require(
            abs(selected_fraction - expected_p) <= tolerance,
            "observed SGR selection fraction does not match configured p",
        )

    successful = int(diagnostics.get("successful_feedback_episodes", -1))
    failed = int(diagnostics.get("failed_feedback_episodes", -1))
    require(successful + failed == episodes, "feedback episode accounting mismatch")
    stats_successes = sum(int(float(record["success"]) >= 0.5) for record in stats.values())
    require(successful == stats_successes, "feedback success count differs from episode statistics")
    action_steps = int(diagnostics.get("action_steps", 0))
    total_steps = int(diagnostics.get("total_trajectory_steps", -1))
    max_steps = int(diagnostics.get("max_trajectory_steps", 0))
    require(action_steps > 0, "no FeedTTA action steps recorded")
    require(total_steps == action_steps, "trajectory/action-step accounting mismatch")
    require(max_steps > 0, "no non-empty FeedTTA trajectory recorded")
    for name in (
        "mean_entropy",
        "last_entropy",
        "last_grad_norm",
        "relative_param_drift",
        "mean_trajectory_steps",
        "mean_action_nll",
        "last_action_nll",
        "mean_max_action_probability",
    ):
        _finite_diagnostic(diagnostics, name)

    aggregated = {}
    for metric in METRICS:
        values = [float(record[metric]) for record in stats.values()]
        value = sum(values) / len(values)
        require(math.isfinite(value), "non-finite aggregate metric: {}".format(metric))
        aggregated[metric] = value
    return aggregated, stats_path, diagnostics


def validate_job_artifacts(
    spec: ExperimentSpec,
    manifest_path: Path,
    job: Job,
    args: argparse.Namespace,
    provenance: Provenance,
) -> Tuple[Dict[str, float], Path, Mapping[str, object]]:
    if not manifest_path.is_file():
        raise UserError("run manifest is missing: {}".format(manifest_path))
    try:
        manifest_path.resolve().relative_to(
            (REPO_ROOT / "avn" / "results" / "runs").resolve()
        )
    except ValueError as error:
        raise UserError("run manifest is outside avn/results/runs") from error
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
            "feedtta",
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
    return validate_stats_and_diagnostics(
        manifest_path,
        job,
        args.episodes,
        allow_parameter_count_mismatch=(
            job.model in spec.provisional_parameter_count_models
        ),
    )


def write_job_metrics(
    spec: ExperimentSpec,
    manifest_path: Path,
    stats_path: Path,
    job: Job,
    job_dir: Path,
    episodes: int,
    metrics: Mapping[str, float],
    diagnostics: Mapping[str, object],
    validation: str,
) -> None:
    diagnostics_summary = {
        key: diagnostics.get(key)
        for key in (
            "updates",
            "action_steps",
            "successful_feedback_episodes",
            "failed_feedback_episodes",
            "relative_param_drift",
            "last_grad_norm",
            "mean_action_nll",
            "mean_trajectory_steps",
            "mean_sgr_selected_fraction",
            "adapted_parameter_count",
            "gradient_accumulator_elements",
        )
    }
    diagnostics_summary["adapted_tensor_count"] = len(
        diagnostics.get("adapted_parameter_names", ())
    )
    diagnostics_summary["expected_parameter_count"] = EXPECTED_PARAMETERS[job.model]
    diagnostics_summary["expected_tensor_count"] = EXPECTED_TENSORS[job.model]
    payload = {
        "job_id": job.job_id,
        "model_job_id": job.model_job_id,
        "run_tag": job.run_tag,
        "stage": spec.stage,
        "model": job.model,
        "source_setting": "single_source",
        "eval_split": "val",
        "result_role": "hyperparameter_search",
        "method": "feedtta",
        "feedback_supervision": "binary_episode_success_oracle",
        "validation": validation,
        "seed": SEED,
        "episodes": episodes,
        "manifest": str(manifest_path),
        "episode_stats": str(stats_path),
        "configuration": {
            "variant": job.point.variant,
            "lr": job.point.lr,
            "gamma": job.point.gamma,
            "p": job.point.p,
            "alpha": job.point.alpha,
            "sgr_seed": SGR_SEED,
            "scope": "full_state_fusion_plus_action_head",
        },
        "metrics": dict(metrics),
        "diagnostics": diagnostics_summary,
    }
    atomic_write(job_dir / "metrics.json", json_text(payload))
    run_dir = manifest_path.resolve().parent
    compact_summary = {
        "task": "avn",
        "benchmark": "mp3d",
        "stage": spec.stage,
        "model": job.model,
        "method": "feedtta",
        "source_setting": "single_source",
        "split": "val",
        "result_role": "hyperparameter_search",
        "feedback_supervision": "binary_episode_success_oracle",
        "validation": validation,
        "seed": SEED,
        "episodes": episodes,
        "run_tag": job.run_tag,
        "configuration": payload["configuration"],
        "metrics": dict(metrics),
        "diagnostics": diagnostics_summary,
    }
    atomic_write(run_dir / "summary.json", json_text(compact_summary))
    atomic_write(run_dir / "diagnostics.json", json_text(dict(diagnostics)))


def artifact_validation_label(
    spec: ExperimentSpec, job: Job, diagnostics: Mapping[str, object]
) -> str:
    actual = diagnostics.get("adapted_parameter_count")
    names = diagnostics.get("adapted_parameter_names")
    actual_tensors = len(names) if isinstance(names, list) else -1
    if (
        actual == EXPECTED_PARAMETERS[job.model]
        and actual_tensors == EXPECTED_TENSORS[job.model]
    ):
        return "ok"
    require(
        job.model in spec.provisional_parameter_count_models,
        "adapted parameter count mismatch for {}".format(job.model),
    )
    return PROVISIONAL_PARAMETER_COUNT_VALIDATION


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
    spec: ExperimentSpec,
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
    accepted_validation = {"ok"}
    if job.model in spec.provisional_parameter_count_models:
        accepted_validation.add(PROVISIONAL_PARAMETER_COUNT_VALIDATION)
    if exitcode != "0" or validation not in accepted_validation or manifest_path is None:
        return False
    try:
        metrics, stats_path, diagnostics = validate_job_artifacts(
            spec, manifest_path, job, args, provenance
        )
        current_validation = artifact_validation_label(spec, job, diagnostics)
        if current_validation != validation:
            return False
        write_job_metrics(
            spec,
            manifest_path,
            stats_path,
            job,
            job_dir,
            args.episodes,
            metrics,
            diagnostics,
            validation,
        )
    except UserError:
        return False
    return True


def launch_job(
    spec: ExperimentSpec,
    job: Job,
    batch_dir: Path,
    args: argparse.Namespace,
    provenance: Provenance,
    logger: Logger,
) -> Worker:
    job_dir = batch_dir / job.model / "jobs" / job.run_tag
    job_dir.mkdir(parents=True, exist_ok=True)
    archive_previous(job_dir)
    atomic_write(
        job_dir / "parameters.env",
        parameters_text(spec, job, args, provenance),
    )
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
        "{} launched job={} model={} tag={} gpu={} pid={}".format(
            utc_now(), job.job_id, job.model, job.run_tag, job.gpu, process.pid
        )
    )
    return Worker(job=job, process=process, output=output, job_dir=job_dir)


def finalize_worker(
    spec: ExperimentSpec,
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
            metrics, stats_path, diagnostics = validate_job_artifacts(
                spec, manifest_path, worker.job, args, provenance
            )
            validation = artifact_validation_label(spec, worker.job, diagnostics)
            write_job_metrics(
                spec,
                manifest_path,
                stats_path,
                worker.job,
                worker.job_dir,
                args.episodes,
                metrics,
                diagnostics,
                validation,
            )
        except UserError as error:
            with (worker.job_dir / "console.log").open("a", encoding="utf-8") as handle:
                handle.write("\n[launcher validation]\n{}\n".format(error))
            composite = 90
        else:
            with (worker.job_dir / "console.log").open("a", encoding="utf-8") as handle:
                if validation == "ok":
                    detail = str(manifest_path)
                else:
                    detail = (
                        "{}; actual/expected parameters={}/{}; "
                        "actual/expected tensors={}/{}; "
                        "metrics retained as provisional"
                    ).format(
                        validation,
                        diagnostics.get("adapted_parameter_count"),
                        EXPECTED_PARAMETERS[worker.job.model],
                        len(diagnostics.get("adapted_parameter_names", ())),
                        EXPECTED_TENSORS[worker.job.model],
                    )
                handle.write("\n[launcher validation]\n{}\n".format(detail))
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
    spec: ExperimentSpec,
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
        "stage",
        "model",
        "source_setting",
        "eval_split",
        "result_role",
        "method",
        "feedback_supervision",
        "variant",
        "seed",
        "episodes",
        "git_commit",
        "checkpoint_sha256",
        "dataset_index_sha256",
        "stream_order_sha256",
        "stream_content_sha256",
        "lr",
        "gamma",
        "p",
        "alpha",
        "sgr_seed",
        "scope",
        "manifest",
        "episode_stats",
        "updates",
        "action_steps",
        "successful_feedback_episodes",
        "failed_feedback_episodes",
        "relative_param_drift",
        "last_grad_norm",
        "mean_action_nll",
        "mean_trajectory_steps",
        "mean_sgr_selected_fraction",
        "adapted_parameter_count",
        "adapted_tensor_count",
        "expected_parameter_count",
        "expected_tensor_count",
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
            validation = (job_dir / "validation").read_text(encoding="utf-8").strip()
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
            "stage": spec.stage,
            "model": job.model,
            "source_setting": "single_source",
            "eval_split": "val",
            "result_role": "hyperparameter_search",
            "method": "feedtta",
            "feedback_supervision": "binary_episode_success_oracle",
            "variant": job.point.variant,
            "seed": SEED,
            "episodes": args.episodes,
            "git_commit": provenance.git_commit,
            "checkpoint_sha256": provenance.checkpoint_sha256[job.model],
            "dataset_index_sha256": provenance.dataset_index_sha256,
            "stream_order_sha256": provenance.stream_order_sha256,
            "stream_content_sha256": provenance.stream_content_sha256,
            "lr": job.point.lr,
            "gamma": job.point.gamma,
            "p": job.point.p,
            "alpha": job.point.alpha,
            "sgr_seed": SGR_SEED,
            "scope": "full_state_fusion_plus_action_head",
            "manifest": payload.get("manifest", ""),
            "episode_stats": payload.get("episode_stats", ""),
        }
        recorded_diagnostics = payload.get("diagnostics", {})
        if not isinstance(recorded_diagnostics, dict):
            recorded_diagnostics = {}
        for name in (
            "updates",
            "action_steps",
            "successful_feedback_episodes",
            "failed_feedback_episodes",
            "relative_param_drift",
            "last_grad_norm",
            "mean_action_nll",
            "mean_trajectory_steps",
            "mean_sgr_selected_fraction",
            "adapted_parameter_count",
            "adapted_tensor_count",
            "expected_parameter_count",
            "expected_tensor_count",
        ):
            row[name] = recorded_diagnostics.get(name, "")
        recorded_metrics = payload.get("metrics", {})
        if not isinstance(recorded_metrics, dict):
            recorded_metrics = {}
        for metric in METRICS:
            row[metric] = recorded_metrics.get(metric, "")
        writer.writerow(row)
    atomic_write(batch_dir / "metrics.csv", output.getvalue())


def summarize(
    spec: ExperimentSpec,
    batch_dir: Path,
    plan: Sequence[Job],
    args: argparse.Namespace,
    provenance: Provenance,
    launched: int,
    skipped: int,
) -> Tuple[str, bool]:
    write_batch_metrics(spec, batch_dir, plan, args, provenance)
    per_model = {}
    totals = {
        "successful": 0,
        "failed": 0,
        "missing": 0,
        "validated": 0,
        "provisional": 0,
        "metrics": 0,
    }
    for model in MODELS:
        counts = {
            "expected": 0,
            "successful": 0,
            "failed": 0,
            "missing": 0,
            "validated": 0,
            "provisional": 0,
            "metrics": 0,
        }
        for job in (item for item in plan if item.model == model):
            counts["expected"] += 1
            job_dir = batch_dir / model / "jobs" / job.run_tag
            try:
                status = (job_dir / "exitcode").read_text(encoding="utf-8").strip()
            except OSError:
                counts["missing"] += 1
                continue
            if status == "0":
                counts["successful"] += 1
            else:
                counts["failed"] += 1
            try:
                validation = (job_dir / "validation").read_text(
                    encoding="utf-8"
                ).strip()
                if validation == "ok":
                    counts["validated"] += 1
                elif validation == PROVISIONAL_PARAMETER_COUNT_VALIDATION:
                    counts["provisional"] += 1
            except OSError:
                pass
            if (job_dir / "metrics.json").is_file():
                counts["metrics"] += 1
        per_model[model] = counts
        for key in totals:
            totals[key] += counts[key]
    complete = (
        totals["successful"] == len(plan)
        and totals["failed"] == 0
        and totals["missing"] == 0
        and totals["validated"] + totals["provisional"] == len(plan)
        and totals["metrics"] == len(plan)
    )
    formal_validation_complete = totals["validated"] == len(plan)
    summary = {
        "batch_id": args.batch_id,
        "experiment": spec.experiment,
        "stage": spec.stage,
        "git_commit": provenance.git_commit,
        "result_role": "hyperparameter_search",
        "protocol": "source_tent_aligned_single_source_val_seed0",
        "completed_at": utc_now(),
        "tracked_worktree_dirty": provenance.tracked_worktree_dirty,
        "smoke": args.smoke,
        "expected": len(plan),
        "launched_this_invocation": launched,
        "skipped_validated": skipped,
        "successful": totals["successful"],
        "failed": totals["failed"],
        "missing": totals["missing"],
        "validated": totals["validated"],
        "provisional": totals["provisional"],
        "metrics": totals["metrics"],
        "complete": complete,
        "formal_validation_complete": formal_validation_complete,
        "per_model": per_model,
        "dataset_index_sha256": provenance.dataset_index_sha256,
        "stream_order_sha256": provenance.stream_order_sha256,
        "stream_content_sha256": provenance.stream_content_sha256,
    }
    return json_text(summary), complete


def run_grid(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    plan: Sequence[Job],
    provenance: Provenance,
    batch_dir: Path,
) -> int:
    batch_lock: Optional[Path] = None
    logger: Optional[Logger] = None
    workers: Dict[int, Worker] = {}
    old_handlers = {}
    try:
        batch_lock = acquire_lock(batch_dir / ".scheduler.lock", "FeedTTA batch")
        logger = Logger(batch_dir / "scheduler.log")
        pending: List[Job] = []
        skipped = 0
        for job in plan:
            job_dir = batch_dir / job.model / "jobs" / job.run_tag
            if reusable_job(spec, job, job_dir, args, provenance):
                skipped += 1
                logger.log("{} skip validated tag={}".format(utc_now(), job.run_tag))
            else:
                pending.append(job)

        active_per_gpu = [0 for _ in args.gpus]
        launched = 0
        failures = 0
        stop_signal: List[Optional[int]] = [None]

        def request_stop(received: int, _frame: object) -> None:
            stop_signal[0] = received

        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)

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
                    worker = launch_job(spec, job, batch_dir, args, provenance, logger)
                    workers[worker.process.pid] = worker
                    active_per_gpu[job.gpu_index] += 1
                    launched += 1
                    del pending[index]
                    launched_one = True
                    break

            finished = []
            for pid, worker in list(workers.items()):
                status = worker.process.poll()
                if status is None:
                    continue
                composite = finalize_worker(spec, worker, status, args, provenance)
                active_per_gpu[worker.job.gpu_index] -= 1
                if composite != 0:
                    failures += 1
                logger.log(
                    "{} finished tag={} runner_status={} status={} active_total={}".format(
                        utc_now(), worker.job.run_tag, status, composite, sum(active_per_gpu)
                    )
                )
                finished.append(pid)
            for pid in finished:
                del workers[pid]
            if (pending or workers) and not finished:
                time.sleep(1.0)

        require_repository_unchanged(provenance)
        summary, complete = summarize(
            spec, batch_dir, plan, args, provenance, launched, skipped
        )
        atomic_write(batch_dir / "SUMMARY.json", summary)
        for line in summary.rstrip().splitlines():
            logger.log(line)
        return 0 if complete and failures == 0 else 1
    except KeyboardInterrupt:
        if logger is not None:
            terminate_workers(workers.values(), logger)
        return 130
    except UserError as error:
        if logger is not None:
            terminate_workers(workers.values(), logger)
            logger.log("scheduler_integrity_failure={}".format(error))
        return 2
    except Exception as error:
        if logger is not None:
            terminate_workers(workers.values(), logger)
            logger.log("scheduler_unexpected_failure={!r}".format(error))
        return 1
    finally:
        for worker in workers.values():
            if not worker.output.closed:
                worker.output.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if logger is not None:
            logger.close()
        release_lock(batch_lock)


def run_experiment(
    spec: ExperimentSpec, args: argparse.Namespace
) -> int:
    plan = build_plan(spec, args)
    print_plan(spec, args, plan)
    if args.dry_run:
        print("Dry run complete: {} jobs, nothing launched.".format(len(plan)))
        return 0

    global_lock: Optional[Path] = None
    preflight_path: Optional[Path] = None
    old_handlers = {}

    def interrupt_preflight(_received: int, _frame: object) -> None:
        raise KeyboardInterrupt

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupt_preflight)
    try:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        global_lock = acquire_lock(
            LOG_ROOT / ".feedtta_scheduler.lock", "AVN FeedTTA scheduler"
        )
        log_base = LOG_ROOT / spec.log_dir_name
        log_base.mkdir(parents=True, exist_ok=True)
        preflight_path = log_base / "{}.preflight.log".format(args.batch_id)
        atomic_write(
            preflight_path,
            "preflight_started_at={}\nbatch_id={}\n".format(utc_now(), args.batch_id),
        )
        provenance = validate_preflight(spec, args)
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
        batch_dir = initialize_batch(spec, args, provenance, plan)
        return run_grid(spec, args, plan, provenance, batch_dir)
    except UserError as error:
        if preflight_path is not None:
            with preflight_path.open("a", encoding="utf-8") as handle:
                handle.write("error={}\n".format(error))
        raise
    except KeyboardInterrupt:
        return 130
    finally:
        release_lock(global_lock)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def cli_main(spec: ExperimentSpec, args: argparse.Namespace) -> int:
    try:
        return run_experiment(spec, args)
    except UserError as error:
        print("error: {}".format(error), file=sys.stderr)
        return 2
