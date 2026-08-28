#!/usr/bin/env python3
"""Summarize and lint a canonical NavTTA experiment JSON spec without writing files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SHA256 = re.compile(r"^[0-9a-f]{64}$")
VALID_STATUS = {"active", "superseded"}
VALID_PHASE = {"smoke", "development", "confirmation", "transfer", "robustness", "formal"}
VALID_SUPERVISION = {"unsupervised", "pseudo_label", "feedback_supervised"}
REQUIRED_SOURCE_MATCHES = {
    "checkpoint_sha256",
    "dataset_sha256",
    "split",
    "episode_order_sha256",
    "model_seed",
    "action_protocol",
    "evaluator",
    "horizon",
}


@dataclass(frozen=True)
class Issue:
    severity: str
    path: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read, summarize, and lint a NavTTA experiment JSON spec."
    )
    parser.add_argument("spec", type=Path, help="JSON file, or - for stdin")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true", help="Return nonzero for warnings too")
    return parser.parse_args()


def at(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_seed_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in value)
        and len(value) == len(set(value))
    )


def lint(spec: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []

    def error(path: str, message: str) -> None:
        issues.append(Issue("error", path, message))

    def warning(path: str, message: str) -> None:
        issues.append(Issue("warning", path, message))

    for key in (
        "schema_version", "spec_id", "status", "purpose", "scope", "data",
        "randomness", "protocol", "source_control", "selection", "freeze",
        "leakage_controls", "stopping", "outputs", "provenance",
    ):
        if key not in spec:
            error(key, "required field is missing")

    if spec.get("schema_version") != 1:
        error("schema_version", "must equal 1")
    if not is_nonempty_string(spec.get("spec_id")):
        error("spec_id", "must be a non-empty string")

    status = spec.get("status")
    supersedes = spec.get("supersedes")
    superseded_by = spec.get("superseded_by")
    if status not in VALID_STATUS:
        error("status", "must be active or superseded")
    if not isinstance(supersedes, list) or not all(is_nonempty_string(item) for item in supersedes):
        error("supersedes", "must be a list of predecessor spec IDs")
    if status == "active" and superseded_by is not None:
        error("superseded_by", "must be null while status is active")
    if status == "superseded" and not is_nonempty_string(superseded_by):
        error("superseded_by", "must name the replacement when status is superseded")

    phase = at(spec, "purpose", "phase")
    if phase not in VALID_PHASE:
        error("purpose.phase", "must declare a recognized experiment phase")
    for path, value in (
        ("purpose.hypothesis", at(spec, "purpose", "hypothesis")),
        ("purpose.decision", at(spec, "purpose", "decision")),
        ("scope.task", at(spec, "scope", "task")),
        ("scope.benchmark", at(spec, "scope", "benchmark")),
    ):
        if not is_nonempty_string(value):
            error(path, "must be a non-empty string")
    if at(spec, "scope", "task") not in {"avn", "vln", "objectnav"}:
        error("scope.task", "must be avn, vln, or objectnav")
    for field in ("models", "methods"):
        value = at(spec, "scope", field)
        if not isinstance(value, list) or not value or not all(is_nonempty_string(item) for item in value):
            error(f"scope.{field}", "must be a non-empty list of strings")
    methods = at(spec, "scope", "methods")
    if isinstance(methods, list) and len(methods) > 1:
        warning(
            "scope.methods",
            "canonical contracts should be per method or supervision-homogeneous group",
        )

    split_name = at(spec, "data", "split", "name")
    split_role = at(spec, "data", "split", "role")
    if not is_nonempty_string(split_name):
        error("data.split.name", "must identify the physical split")
    if split_role not in {"development", "evaluation", "test"}:
        error("data.split.role", "must be development, evaluation, or test")
    if phase == "development" and split_role != "development":
        error("data.split.role", "development searches must use a development-role split")
    if phase in {"transfer", "robustness", "formal"} and split_role == "development":
        warning("data.split.role", "evaluation-like phase is still labeled development")

    for path, value in (
        ("data.dataset_sha256", at(spec, "data", "dataset_sha256")),
        ("data.episode_order.sha256", at(spec, "data", "episode_order", "sha256")),
        ("provenance.checkpoint_sha256", at(spec, "provenance", "checkpoint_sha256")),
        ("freeze.config_sha256", at(spec, "freeze", "config_sha256")),
    ):
        if not isinstance(value, str) or not SHA256.fullmatch(value):
            error(path, "must be a lowercase 64-character SHA256 digest")

    order_seeds = at(spec, "data", "episode_order", "order_seeds")
    model_seeds = at(spec, "randomness", "model_seeds")
    if not is_seed_list(order_seeds):
        error("data.episode_order.order_seeds", "must be a non-empty unique list of non-negative integers")
    if not is_seed_list(model_seeds):
        error("randomness.model_seeds", "must be a non-empty unique list of non-negative integers")
    for field in ("action_rng", "adaptation_rng", "augmentation_rng", "replay_rng"):
        if not is_nonempty_string(at(spec, "randomness", field)):
            error(f"randomness.{field}", "must declare an independent seed or derivation")

    horizon_unit = at(spec, "data", "horizon", "unit")
    horizon_value = at(spec, "data", "horizon", "value")
    if horizon_unit not in {"episodes", "actions", "scenes"}:
        error("data.horizon.unit", "must be episodes, actions, or scenes")
    if not is_positive_int(horizon_value):
        error("data.horizon.value", "must be a positive integer")

    dose = at(spec, "protocol", "adaptation", "update_dose")
    if not isinstance(dose, Mapping):
        error("protocol.adaptation.update_dose", "must define adaptation dose")
    else:
        if dose.get("trigger_unit") not in {"actions", "episodes", "batches"}:
            error("protocol.adaptation.update_dose.trigger_unit", "must be actions, episodes, or batches")
        for field in (
            "interval", "optimizer_steps_per_trigger", "backward_passes_per_trigger",
            "samples_per_step", "max_optimizer_steps",
        ):
            if not is_positive_int(dose.get(field)):
                error(f"protocol.adaptation.update_dose.{field}", "must be a positive integer")

    feedback = at(spec, "protocol", "feedback")
    supervision = at(spec, "protocol", "feedback", "supervision")
    fields = at(spec, "protocol", "feedback", "environment_fields")
    budget_max = at(spec, "protocol", "feedback", "budget", "maximum")
    if supervision not in VALID_SUPERVISION:
        error("protocol.feedback.supervision", "must be unsupervised, pseudo_label, or feedback_supervised")
    if not isinstance(fields, list) or not all(is_nonempty_string(item) for item in fields):
        error("protocol.feedback.environment_fields", "must be a list of explicitly named fields")
    if not isinstance(budget_max, (int, float)) or isinstance(budget_max, bool) or budget_max < 0:
        error("protocol.feedback.budget.maximum", "must be a non-negative number")
    if isinstance(feedback, Mapping) and supervision in {"unsupervised", "pseudo_label"}:
        if fields:
            error("protocol.feedback.environment_fields", "non-feedback supervision cannot consume evaluator fields")
        if budget_max != 0:
            error("protocol.feedback.budget.maximum", "non-feedback supervision must have zero query budget")
    if supervision == "feedback_supervised":
        for field in ("signal", "provider", "timing", "affects"):
            if not is_nonempty_string(feedback.get(field) if isinstance(feedback, Mapping) else None):
                error(f"protocol.feedback.{field}", "is required for feedback-supervised methods")
        if not fields:
            error("protocol.feedback.environment_fields", "feedback supervision must declare consumed fields")

    source = at(spec, "source_control")
    mode = at(spec, "source_control", "mode")
    if mode not in {"reuse", "rerun"}:
        error("source_control.mode", "must be reuse or rerun")
    if mode == "reuse" and isinstance(source, Mapping):
        if not is_nonempty_string(source.get("manifest")):
            error("source_control.manifest", "reused Source requires a validated manifest path")
        digest = source.get("manifest_sha256")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            error("source_control.manifest_sha256", "reused Source requires a SHA256 digest")
        matched = source.get("matched_on")
        matched_set = set(matched) if isinstance(matched, list) and all(isinstance(x, str) for x in matched) else set()
        missing = sorted(REQUIRED_SOURCE_MATCHES - matched_set)
        if missing:
            error("source_control.matched_on", "missing dimensions: " + ", ".join(missing))
        if source.get("rerun_on_mismatch") is not True:
            error("source_control.rerun_on_mismatch", "must be true for Source reuse")

    freeze = at(spec, "freeze")
    if not isinstance(freeze, Mapping):
        error("freeze", "must define the freeze boundary")
    else:
        if freeze.get("before_evaluation") is not True:
            error("freeze.before_evaluation", "must be true")
        immutable = freeze.get("immutable_fields")
        if not isinstance(immutable, list) or not immutable:
            error("freeze.immutable_fields", "must list frozen scientific fields")

    leakage = at(spec, "leakage_controls")
    if not isinstance(leakage, Mapping):
        error("leakage_controls", "must define leakage controls")
    else:
        for field in (
            "selection_rule_frozen_before_evaluation",
            "posthoc_grid_expansion_forbidden",
        ):
            if leakage.get(field) is not True:
                error(f"leakage_controls.{field}", "must be true")
        if leakage.get("state_transfer_across_splits") is not False:
            warning("leakage_controls.state_transfer_across_splits", "should be false unless transfer is the declared intervention")
        allowlist = leakage.get("feedback_field_allowlist")
        if not isinstance(allowlist, list):
            error("leakage_controls.feedback_field_allowlist", "must be a list")
        elif supervision != "feedback_supervised" and allowlist:
            error("leakage_controls.feedback_field_allowlist", "must be empty without feedback supervision")
        elif supervision == "feedback_supervised" and set(allowlist) != set(fields or []):
            error("leakage_controls.feedback_field_allowlist", "must exactly match declared feedback environment fields")

    if at(spec, "stopping", "algorithmic_failure_is_result") is not True:
        error("stopping.algorithmic_failure_is_result", "must be true")
    if at(spec, "outputs", "run_manifest_required") is not True:
        error("outputs.run_manifest_required", "must be true")

    return issues


def summarize(spec: Mapping[str, Any]) -> dict[str, Any]:
    dose = at(spec, "protocol", "adaptation", "update_dose")
    feedback = at(spec, "protocol", "feedback")
    return {
        "spec_id": spec.get("spec_id"),
        "status": spec.get("status"),
        "supersedes": spec.get("supersedes"),
        "superseded_by": spec.get("superseded_by"),
        "phase": at(spec, "purpose", "phase"),
        "task": at(spec, "scope", "task"),
        "benchmark": at(spec, "scope", "benchmark"),
        "models": at(spec, "scope", "models"),
        "methods": at(spec, "scope", "methods"),
        "split": at(spec, "data", "split"),
        "order_seeds": at(spec, "data", "episode_order", "order_seeds"),
        "model_seeds": at(spec, "randomness", "model_seeds"),
        "horizon": at(spec, "data", "horizon"),
        "update_dose": dict(dose) if isinstance(dose, Mapping) else dose,
        "source_mode": at(spec, "source_control", "mode"),
        "source_manifest": at(spec, "source_control", "manifest"),
        "feedback_supervision": feedback.get("supervision") if isinstance(feedback, Mapping) else None,
        "feedback_budget": feedback.get("budget") if isinstance(feedback, Mapping) else None,
        "frozen_before_evaluation": at(spec, "freeze", "before_evaluation"),
    }


def render_text(path: Path, summary: Mapping[str, Any], issues: Sequence[Issue]) -> str:
    lines = [f"spec: {path}"]
    for key, value in summary.items():
        lines.append(f"{key}: {json.dumps(value, sort_keys=True)}")
    errors = sum(issue.severity == "error" for issue in issues)
    warnings = sum(issue.severity == "warning" for issue in issues)
    lines.append(f"lint: {errors} error(s), {warnings} warning(s)")
    for issue in issues:
        lines.append(f"{issue.severity.upper()} {issue.path}: {issue.message}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    try:
        if str(args.spec) == "-":
            raw = sys.stdin.read()
            display_path = Path("<stdin>")
        else:
            if not args.spec.is_file():
                print(f"spec is not a file: {args.spec}", file=sys.stderr)
                return 2
            raw = args.spec.read_text(encoding="utf-8")
            display_path = args.spec
        loaded = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"cannot read JSON spec {args.spec}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(loaded, dict):
        print("experiment spec root must be a JSON object", file=sys.stderr)
        return 2

    issues = lint(loaded)
    summary = summarize(loaded)
    if args.format == "json":
        print(json.dumps({
            "path": "<stdin>" if str(args.spec) == "-" else str(args.spec.resolve()),
            "read_only": True,
            "summary": summary,
            "issues": [asdict(issue) for issue in issues],
        }, indent=2, sort_keys=True))
    else:
        print(render_text(display_path, summary, issues))

    has_errors = any(issue.severity == "error" for issue in issues)
    has_warnings = any(issue.severity == "warning" for issue in issues)
    return 1 if has_errors or (args.strict and has_warnings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
