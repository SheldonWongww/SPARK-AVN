#!/usr/bin/env python3
"""Export compact AVN hyperparameter analyses to a Git-trackable directory.

AVN schedulers keep their durable ``metrics.csv`` files below
``avn/results/logs`` so raw experiment evidence remains local.  This utility
reads one or more of those compact tables, joins the matched Source metrics,
and writes deterministic Markdown reports grouped by navigation model and TTA
method under ``avn/results/analysis/hparam_search/by_model``.

The exporter never follows manifest, checkpoint, episode-stat, or raw-log
paths recorded in input rows.  Only aggregate scalar metrics, hyperparameters,
run tags, and input-file hashes are included in the published report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import math
from pathlib import Path
import re
import sys
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_ROOT = REPO_ROOT / "avn" / "results" / "analysis" / "hparam_search"
DEFAULT_OUTPUT_ROOT = ANALYSIS_ROOT / "by_model"
DEFAULT_SOURCE_METRICS = (
    REPO_ROOT
    / "avn"
    / "results"
    / "logs"
    / "source_reval"
    / "source-reval-v1-seed0"
    / "metrics.csv"
)

SAFE_SLUG = re.compile(r"[a-z0-9][a-z0-9_]*")
SUCCESS_STATUSES = {"", "0", "ok", "success", "completed", "complete"}
MODEL_LABELS = {
    "smt_audio": "SMT+Audio",
    "enmus": "ENMuS",
    "av_nav": "AV-Nav",
    "savi": "SAVi",
}
METHOD_LABELS = {
    "tent": "Tent",
    "fstta": "FSTTA",
    "eam": "EAM",
    "feedtta": "FeedTTA",
    "atena": "ATENA",
}
FEEDBACK_METHODS = {"feedtta", "atena"}

METHOD_PARAMETERS = {
    "tent": (
        "norm_scope", "scope", "lr", "update_interval", "episodic",
    ),
    "fstta": (
        "fast_lr", "lr_fast", "M", "m", "slow_lr", "lr_slow", "N",
        "n", "q", "use_slow", "fast_grad_mode", "use_fast_lr_scaler",
        "slow_optimizer", "reset_slow_optimizer_each_window",
    ),
    "eam": (
        "lr", "update_interval", "confidence_scale", "memory_size",
        "batch_size", "scope",
    ),
    "feedtta": (
        "lr", "gamma", "p", "alpha", "sgr_seed", "scope",
    ),
    "atena": (
        "lr_query", "lr_self", "mix_lambda", "query_threshold",
        "self_loss_weight", "scope",
    ),
}

AGGREGATE_METRICS = {
    "reward", "distance_to_goal", "normalized_distance_to_goal", "success",
    "spl", "softspl", "na", "sna", "sws", "updates", "action_steps",
    "relative_param_drift", "last_grad_norm", "mean_action_nll",
    "mean_trajectory_steps", "mean_sgr_selected_fraction",
    "successful_feedback_episodes", "failed_feedback_episodes",
}
METADATA_COLUMNS = {
    "job_id", "model_job_id", "run_tag", "gpu", "status", "validation",
    "stage", "model", "source_setting", "eval_split", "result_role",
    "method", "feedback_supervision", "variant", "seed", "episodes",
    "git_commit", "checkpoint_sha256", "dataset_index_sha256",
    "stream_order_sha256", "stream_content_sha256", "manifest",
    "episode_stats", "exitcode",
}


class ExportError(RuntimeError):
    """Raised when compact evidence is incomplete or ambiguous."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise ExportError("missing CSV: {}".format(path))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ExportError("CSV has no header: {}".format(path))
        return list(reader)


def finite_number(row: Mapping[str, str], key: str, context: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ExportError("{} is missing {}".format(context, key))
    try:
        parsed = float(value)
    except ValueError as error:
        raise ExportError("{} has invalid {}={!r}".format(context, key, value)) from error
    if not math.isfinite(parsed):
        raise ExportError("{} has non-finite {}".format(context, key))
    return parsed


def validate_slug(value: str, kind: str) -> str:
    normalized = value.strip().lower()
    if not SAFE_SLUG.fullmatch(normalized):
        raise ExportError("unsafe {} slug: {!r}".format(kind, value))
    return normalized


def successful_row(row: Mapping[str, str]) -> bool:
    status = str(row.get("status", row.get("exitcode", ""))).strip().lower()
    return status in SUCCESS_STATUSES


def load_sources(path: Path, source_setting: str) -> Dict[str, Dict[str, float]]:
    sources: Dict[str, Dict[str, float]] = {}
    for index, row in enumerate(read_csv(path), start=2):
        if row.get("source_setting", source_setting) != source_setting:
            continue
        if not successful_row(row):
            continue
        model = validate_slug(row.get("model", ""), "model")
        context = "{}:{}".format(path, index)
        candidate = {
            "success": finite_number(row, "success", context),
            "spl": finite_number(row, "spl", context),
        }
        if model in sources and sources[model] != candidate:
            raise ExportError("conflicting Source metrics for {}".format(model))
        sources[model] = candidate
    return sources


def parameter_keys(row: Mapping[str, str], method: str) -> Tuple[str, ...]:
    preferred = tuple(
        key for key in METHOD_PARAMETERS.get(method, ()) if row.get(key, "") != ""
    )
    if preferred:
        return preferred
    return tuple(
        sorted(
            key for key, value in row.items()
            if value != ""
            and key not in METADATA_COLUMNS
            and key not in AGGREGATE_METRICS
        )
    )


def parameter_text(row: Mapping[str, str], method: str) -> str:
    values = ["{}={}".format(key, row[key]) for key in parameter_keys(row, method)]
    return ", ".join(values) if values else "(no explicit parameters)"


def markdown_code(value: object) -> str:
    return "`{}`".format(str(value).replace("`", "'").replace("|", "\\|"))


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def metric_scale(source: Mapping[str, float], rows: Sequence[Mapping[str, str]]) -> float:
    values = [source["success"], source["spl"]]
    for row in rows:
        values.extend((float(row["success"]), float(row["spl"])))
    return 100.0 if all(abs(value) <= 1.0 for value in values) else 1.0


def metric(value: float, scale: float) -> str:
    return "{:.2f}".format(value * scale)


def delta(value: float, source: float, scale: float) -> str:
    return "{:+.2f}".format((value - source) * scale)


def rank_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["success"]),
            -float(row["spl"]),
            row.get("run_tag", ""),
        ),
    )


def value_sort_key(value: str) -> Tuple[int, Union[float, str]]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def render_ranked_table(
    rows: Sequence[Dict[str, str]],
    source: Mapping[str, float],
    method: str,
    scale: float,
    top_k: int,
) -> List[str]:
    output = [
        "| Rank | SR | ΔSR | SPL | ΔSPL | Parameters | Run tag | Validation |",
        "|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for rank, row in enumerate(rows[:top_k], start=1):
        success = float(row["success"])
        spl = float(row["spl"])
        output.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                rank,
                metric(success, scale),
                delta(success, source["success"], scale),
                metric(spl, scale),
                delta(spl, source["spl"], scale),
                markdown_code(parameter_text(row, method)),
                markdown_code(row.get("run_tag", "")),
                markdown_code(row.get("validation", "unspecified")),
            )
        )
    return output


def render_marginals(
    rows: Sequence[Dict[str, str]], method: str, scale: float
) -> List[str]:
    keys: Set[str] = set()
    for row in rows:
        keys.update(parameter_keys(row, method))
    output: List[str] = []
    for key in sorted(keys):
        grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row.get(key, "") != "":
                grouped[row[key]].append(row)
        if not 2 <= len(grouped) <= 20:
            continue
        output.extend(
            [
                "### {}".format(markdown_code(key)),
                "",
                "| Value | Runs | Mean SR | Best SR | Mean SPL |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for value in sorted(grouped, key=value_sort_key):
            group = grouped[value]
            successes = [float(row["success"]) for row in group]
            spls = [float(row["spl"]) for row in group]
            output.append(
                "| {} | {} | {} | {} | {} |".format(
                    markdown_code(value),
                    len(group),
                    metric(sum(successes) / len(successes), scale),
                    metric(max(successes), scale),
                    metric(sum(spls) / len(spls), scale),
                )
            )
        output.append("")
    return output


def render_report(
    model: str,
    method: str,
    rows: Sequence[Dict[str, str]],
    source: Mapping[str, float],
    evidence: Sequence[Path],
    source_path: Path,
    source_setting: str,
    top_k: int,
) -> str:
    ranked = rank_rows(rows)
    scale = metric_scale(source, ranked)
    joint = [
        row for row in ranked
        if float(row["success"]) > source["success"]
        and float(row["spl"]) >= source["spl"]
    ]
    validations = Counter(row.get("validation", "unspecified") for row in rows)
    statuses = Counter(row.get("status", "unspecified") for row in rows)
    label = "{} × {}".format(
        MODEL_LABELS.get(model, model), METHOD_LABELS.get(method, method)
    )
    supervision = (
        "binary episode feedback" if method in FEEDBACK_METHODS else "unsupervised"
    )
    output = [
        "# {} AVN 超参数搜索分析".format(label),
        "",
        "状态：由聚合指标生成的可同步紧凑报告；原始日志、逐 episode 结果、权重和本地绝对路径不进入 Git。",
        "",
        "- Model: {}".format(markdown_code(model)),
        "- Method: {} ({})".format(markdown_code(method), supervision),
        "- Source setting: {}".format(markdown_code(source_setting)),
        "- Exported candidates: {}".format(len(rows)),
        "- Runner statuses: {}".format(
            ", ".join("{}={}".format(key, statuses[key]) for key in sorted(statuses))
        ),
        "- Validation labels: {}".format(
            ", ".join("{}={}".format(key, validations[key]) for key in sorted(validations))
        ),
        "- Matched Source SR/SPL: {}/{}".format(
            metric(source["success"], scale), metric(source["spl"], scale)
        ),
        "- Joint SR/SPL improvements: {}".format(len(joint)),
        "",
        "## Evidence",
        "",
    ]
    for path in sorted(evidence, key=lambda item: display_path(item)):
        output.append(
            "- {} — SHA256 {}".format(
                markdown_code(display_path(path)), markdown_code(sha256_file(path))
            )
        )
    output.append(
        "- {} — SHA256 {} (matched Source)".format(
            markdown_code(display_path(source_path)),
            markdown_code(sha256_file(source_path)),
        )
    )
    output.extend(
        [
            "",
            "## SR-first ranking",
            "",
            "Ranking: SR, then SPL, then run tag. Metrics are percentage points when inputs are rates in [0, 1].",
            "",
            *render_ranked_table(ranked, source, method, scale, top_k),
            "",
            "## Joint SR/SPL improvements",
            "",
        ]
    )
    if joint:
        output.extend(render_ranked_table(joint, source, method, scale, top_k))
    else:
        output.append("No candidate strictly improves SR while preserving or improving SPL.")
    marginals = render_marginals(rows, method, scale)
    if marginals:
        output.extend(["", "## Parameter marginals", "", *marginals])
    return "\n".join(output).rstrip() + "\n"


def export_reports(
    metrics_paths: Sequence[Path],
    source_path: Path,
    output_root: Path,
    source_setting: str = "single_source",
    models: Iterable[str] = (),
    methods: Iterable[str] = (),
    top_k: int = 10,
    include_nonzero_status: bool = False,
) -> List[Path]:
    selected_models = {validate_slug(value, "model") for value in models}
    selected_methods = {validate_slug(value, "method") for value in methods}
    sources = load_sources(source_path, source_setting)
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    evidence: Dict[Tuple[str, str], Set[Path]] = defaultdict(set)
    seen_run_tags: Set[str] = set()

    for metrics_path in metrics_paths:
        path = Path(metrics_path)
        for index, original in enumerate(read_csv(path), start=2):
            row = dict(original)
            if row.get("source_setting", source_setting) != source_setting:
                continue
            if row.get("result_role", "hyperparameter_search") != "hyperparameter_search":
                continue
            if not successful_row(row) and not include_nonzero_status:
                continue
            model = validate_slug(row.get("model", ""), "model")
            method = validate_slug(row.get("method", ""), "method")
            if selected_models and model not in selected_models:
                continue
            if selected_methods and method not in selected_methods:
                continue
            if model not in sources:
                raise ExportError("no matched Source metrics for {}".format(model))
            context = "{}:{}".format(path, index)
            row["success"] = str(finite_number(row, "success", context))
            row["spl"] = str(finite_number(row, "spl", context))
            run_tag = row.get("run_tag", "")
            if not run_tag:
                raise ExportError("{} is missing run_tag".format(context))
            if run_tag in seen_run_tags:
                raise ExportError("duplicate run_tag across inputs: {}".format(run_tag))
            seen_run_tags.add(run_tag)
            groups[(model, method)].append(row)
            evidence[(model, method)].add(path)

    if not groups:
        raise ExportError("no successful hyperparameter-search rows matched the filters")
    if top_k < 1:
        raise ExportError("top_k must be positive")

    written: List[Path] = []
    for model, method in sorted(groups):
        target = Path(output_root) / model / "{}.md".format(method)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = render_report(
            model=model,
            method=method,
            rows=groups[(model, method)],
            source=sources[model],
            evidence=tuple(evidence[(model, method)]),
            source_path=source_path,
            source_setting=source_setting,
            top_k=top_k,
        )
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        written.append(target)
    return written


def require_publish_root(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ANALYSIS_ROOT.resolve())
    except ValueError as error:
        raise ExportError(
            "output root must stay under {} so reports remain Git-trackable".format(
                ANALYSIS_ROOT
            )
        ) from error
    return resolved


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path, help="compact metrics.csv inputs")
    parser.add_argument(
        "--source-metrics", type=Path, default=DEFAULT_SOURCE_METRICS,
        help="matched Source metrics.csv",
    )
    parser.add_argument("--source-setting", default="single_source")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--include-nonzero-status",
        action="store_true",
        help=(
            "include rows with finite aggregate metrics even when launcher "
            "validation returned a nonzero status; reports retain the status"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        output_root = require_publish_root(args.output_root)
        written = export_reports(
            metrics_paths=args.metrics,
            source_path=args.source_metrics,
            output_root=output_root,
            source_setting=args.source_setting,
            models=args.model,
            methods=args.method,
            top_k=args.top_k,
            include_nonzero_status=args.include_nonzero_status,
        )
    except ExportError as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        return 2
    for path in written:
        print(display_path(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
