#!/usr/bin/env python3
"""Cross-split consistency hyperparameter search for VLN TTA.

This runner implements the methodology in
``vln/experiments/VLN_TTA_HPARAM_METHODOLOGY.md``: for each (method, model) it
expands a small grid anchored on published values, evaluates every candidate on
BOTH ``val_seen`` and ``val_unseen`` (canonical ``order_seed 0``), and selects a
single frozen configuration by a no-regression floor plus a worst-split gain
objective -- never a single-split argmax.  Model execution stays in
``run_source_eval.sh``; this file only orchestrates configs, launches, metric
parsing, and selection.

It is deliberately independent of the older ``run_tta_hparam_search.py`` engine
(which hard-pins val_seen-only selection).  It reuses only stable primitives:
``run_source_eval.sh``, ``tta_config_cli.py`` (indirectly, via the launcher),
and the episode-order manifests.

Local use is limited to ``--dry-run`` (prints commands) and unit tests; real
runs happen on the AutoDL server.
"""
import argparse
import concurrent.futures
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "vln/scripts/run_source_eval.sh"
SEARCH_SCHEMA = "navtta.vln_tta_consistency_search.v1"
JOB_SCHEMA = "navtta.vln_tta_job.v1"
CONTINUOUS_SETTINGS = {"etpnav-r2r-ce", "bevbert-r2r-ce"}


class UserError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Spec expansion
# ---------------------------------------------------------------------------
def load_spec(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        spec = json.load(stream)
    if spec.get("schema") != SEARCH_SCHEMA:
        raise UserError("unsupported search spec schema: {}".format(spec.get("schema")))
    for key in ("benchmark", "splits", "settings", "methods", "selection"):
        if key not in spec:
            raise UserError("search spec missing required key: {}".format(key))
    if spec.get("order_seed", 0) != 0:
        raise UserError("this methodology selects on canonical order_seed 0 only")
    if list(spec["splits"]) != ["val_seen", "val_unseen"]:
        raise UserError("consistency search requires splits [val_seen, val_unseen]")
    return spec


def _slug(value):
    text = str(value).lower().replace("-", "m").replace(".", "p").replace("+", "")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def expand_candidates(method, method_spec, *, for_search):
    """Return the list of parameter dicts for one method's grid.

    ``for_search`` selects IDEA's cheaper ``search_opt_steps`` (vs the final
    ``final_opt_steps``) so the ranking pass stays tractable on long streams.
    """
    base = dict(method_spec.get("base", {}))
    grid = method_spec.get("grid", {})
    if method == "idea":
        steps = method_spec.get(
            "search_opt_steps" if for_search else "final_opt_steps", 50
        )
        base.setdefault("opt_steps", steps)
        base["opt_steps"] = steps
    # Grid values may be scalars-lists or lists-of-dicts (paired knobs like
    # FeedTTA (p, alpha)).  Each key contributes one axis of the product.
    axes = []
    keys = []
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise UserError("grid axis {} must be a non-empty list".format(key))
        keys.append(key)
        axes.append(values)
    candidates = []
    for combo in itertools.product(*axes) if axes else [()]:
        params = dict(base)
        for key, value in zip(keys, combo):
            if isinstance(value, dict):
                params.update(value)  # paired knobs expand into multiple params
            else:
                params[key] = value
        candidates.append(params)
    return candidates


def candidate_tag(method, params):
    parts = [method]
    for key in sorted(params):
        value = params[key]
        if key in ("action_selection", "norm_scope", "fast_grad_mode",
                   "scope_profile"):
            continue
        parts.append("{}{}".format(_slug(key), _slug(value)))
    return "-".join(parts)[:80]


def full_run_tag(run_tag, tag):
    """The RUN_TAG passed to run_source_eval.sh (result-root must end here)."""
    return "{}-{}".format(run_tag, tag)


def candidate_job_dir(out_dir, setting, method, run_tag, tag):
    """Directory holding the per-split result roots for one candidate.

    ``run_source_eval.sh`` requires ``--result-root`` to end with
    ``<RUN_TAG>/<SPLIT>``, so the split dir is nested under the full run tag.
    """
    return Path(out_dir) / setting / method / full_run_tag(run_tag, tag)


# ---------------------------------------------------------------------------
# Job config + launch
# ---------------------------------------------------------------------------
def write_job_config(job_dir, method, params):
    job_dir.mkdir(parents=True, exist_ok=True)
    config = {"schema": JOB_SCHEMA, "method": method, "parameters": params}
    path = job_dir / "tta_config.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
    return path


def build_command(setting, split, config_path, result_root, run_tag):
    # No --order-seed: run_source_eval.sh then defaults to the canonical
    # seed-0 episode-order manifest (the paper main-table order).  Passing
    # --order-seed 0 would instead demand an "orders"-stage robustness config.
    return [
        "bash", str(RUNNER), setting, split,
        "--run-tag", run_tag,
        "--tta-config", str(config_path),
        "--result-root", str(result_root),
    ]


# ---------------------------------------------------------------------------
# Metric parsing (discrete console line + continuous "Average episode X")
# ---------------------------------------------------------------------------
def parse_console_metrics(text, split):
    """Return an uppercased metric dict parsed from a run's console log."""
    values = {}
    marker = "Env name: {}".format(split)
    discrete = [line for line in text.splitlines() if marker in line]
    if discrete:
        for key, value in re.findall(
            r"([A-Za-z][A-Za-z0-9_]*)\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)", discrete[-1]
        ):
            values[key.upper()] = float(value)
        return values
    for key, value in re.findall(
        r"Average episode ([A-Za-z0-9_]+):\s*(-?[0-9]+(?:\.[0-9]+)?)", text
    ):
        metric = key.upper()
        number = float(value)
        values[metric] = 100.0 * number if metric in {
            "SUCCESS", "ORACLE_SUCCESS", "SPL", "NDTW", "SDTW"
        } else number
    if "SUCCESS" in values:
        values["SR"] = values["SUCCESS"]
    if "ORACLE_SUCCESS" in values:
        values["OSR"] = values["ORACLE_SUCCESS"]
    return values


def read_metrics(result_root, split):
    console = Path(result_root) / "console.log"
    if not console.is_file():
        raise UserError("missing console log: {}".format(console))
    values = parse_console_metrics(
        console.read_text(encoding="utf-8", errors="replace"), split
    )
    if not values:
        raise UserError("no validation metrics found in {}".format(console))
    return values


def read_adapter_diagnostics(result_root):
    path = Path(result_root) / "tta_diagnostics.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as stream:
        diag = json.load(stream)
    adapter = diag.get("adapter", diag) if isinstance(diag, dict) else {}
    return {
        "relative_param_drift": float(adapter.get("relative_param_drift", 0.0)),
        "updates": int(adapter.get("updates", 0)),
    }


# ---------------------------------------------------------------------------
# Selection (Section 2.2 of the methodology)
# ---------------------------------------------------------------------------
def select_config(candidate_results, source_metrics, selection):
    """Apply floor gate + worst-split gain objective + tie-breakers.

    ``candidate_results``: list of dicts with keys ``parameters``,
    ``metrics_seen``, ``metrics_unseen``, ``diagnostics``.
    ``source_metrics``: {"val_seen": {...}, "val_unseen": {...}}.
    Returns (winner_or_None, ranked_list).
    """
    floor_metric = selection.get("floor_metric", "SR")
    gain_metric = selection.get("gain_metric", "SR")
    epsilon = float(selection.get("floor_epsilon", 0.3))
    src_seen = source_metrics["val_seen"]
    src_unseen = source_metrics["val_unseen"]

    def gain(metrics, src, metric):
        return float(metrics.get(metric, 0.0)) - float(src.get(metric, 0.0))

    ranked = []
    for result in candidate_results:
        seen, unseen = result["metrics_seen"], result["metrics_unseen"]
        floor_seen = seen.get(floor_metric, float("-inf")) >= (
            src_seen.get(floor_metric, 0.0) - epsilon
        )
        floor_unseen = unseen.get(floor_metric, float("-inf")) >= (
            src_unseen.get(floor_metric, 0.0) - epsilon
        )
        g_seen = gain(seen, src_seen, gain_metric)
        g_unseen = gain(unseen, src_unseen, gain_metric)
        spl_gain = 0.5 * (gain(seen, src_seen, "SPL") + gain(unseen, src_unseen, "SPL"))
        diag = result.get("diagnostics", {})
        ranked.append({
            "parameters": result["parameters"],
            "metrics_seen": seen,
            "metrics_unseen": unseen,
            "passes_floor": bool(floor_seen and floor_unseen),
            "worst_split_gain": min(g_seen, g_unseen),
            "mean_gain": 0.5 * (g_seen + g_unseen),
            "mean_spl_gain": spl_gain,
            "param_drift": float(diag.get("relative_param_drift", 0.0)),
            "updates": int(diag.get("updates", 0)),
        })

    survivors = [item for item in ranked if item["passes_floor"]]
    survivors.sort(
        key=lambda item: (
            item["worst_split_gain"], item["mean_gain"], item["mean_spl_gain"],
            -item["param_drift"], -item["updates"],
        ),
        reverse=True,
    )
    ranked.sort(
        key=lambda item: (
            item["passes_floor"], item["worst_split_gain"], item["mean_gain"],
        ),
        reverse=True,
    )
    winner = survivors[0] if survivors else None
    return winner, ranked


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _run_one(command, dry_run):
    printable = " ".join(command)
    if dry_run:
        return 0, printable
    completed = subprocess.run(command, cwd=str(REPO_ROOT))
    return completed.returncode, printable


def run_search(spec_path, run_tag, methods_filter, settings_filter, out_dir,
               concurrency_override, dry_run, final_stage):
    spec = load_spec(spec_path)
    settings = [s for s in spec["settings"]
                if not settings_filter or s in settings_filter]
    method_names = [m for m in spec["methods"]
                    if not methods_filter or m in methods_filter]
    out_dir = Path(out_dir)
    commands_preview = []

    for setting in settings:
        for method in method_names:
            method_spec = spec["methods"][method]
            candidates = expand_candidates(
                method, method_spec, for_search=not final_stage
            )
            conc = concurrency_override or (
                spec.get("concurrency", {}).get(setting, {}).get(method, 1)
            )
            jobs = []
            for params in candidates:
                tag = candidate_tag(method, params)
                job_dir = candidate_job_dir(out_dir, setting, method, run_tag, tag)
                config_path = write_job_config(job_dir, method, params)
                for split in spec["splits"]:
                    result_root = job_dir / split
                    command = build_command(
                        setting, split, config_path, result_root,
                        full_run_tag(run_tag, tag),
                    )
                    jobs.append((command, params, split, result_root))
            # Launch with bounded concurrency (same-(model,method) only).
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                futures = {
                    pool.submit(_run_one, command, dry_run): (command, params, split)
                    for command, params, split, _ in jobs
                }
                for future in concurrent.futures.as_completed(futures):
                    code, printable = future.result()
                    commands_preview.append(printable)
                    if code != 0 and not dry_run:
                        print("[warn] nonzero exit ({}): {}".format(code, printable),
                              file=sys.stderr)

    if dry_run:
        print("# DRY RUN: {} commands".format(len(commands_preview)))
        for line in commands_preview:
            print(line)
    return commands_preview


def _read_source_metrics(source_root, setting, splits):
    metrics = {}
    for split in splits:
        result_root = Path(source_root) / setting / split
        metrics[split] = read_metrics(result_root, split)
    return metrics


def run_selection(spec_path, out_dir, source_root, run_tag, methods_filter,
                  settings_filter, final_stage):
    """Collect completed runs, apply the consistency rule, write winners.

    Expects the search runs and a Source eval to have completed, with results
    under ``out_dir/<setting>/<method>/<run_tag>-<tag>/<split>/console.log`` and
    ``source_root/<setting>/<split>/console.log``.  Writes
    ``out_dir/<setting>/<method>/selected_config.json`` per cell.
    """
    spec = load_spec(spec_path)
    out_dir = Path(out_dir)
    splits = spec["splits"]
    settings = [s for s in spec["settings"]
                if not settings_filter or s in settings_filter]
    method_names = [m for m in spec["methods"]
                    if not methods_filter or m in methods_filter]
    selections = {}
    for setting in settings:
        source_metrics = _read_source_metrics(source_root, setting, splits)
        for method in method_names:
            method_spec = spec["methods"][method]
            candidates = expand_candidates(
                method, method_spec, for_search=not final_stage
            )
            results = []
            for params in candidates:
                tag = candidate_tag(method, params)
                job_dir = candidate_job_dir(out_dir, setting, method, run_tag, tag)
                try:
                    seen = read_metrics(job_dir / "val_seen", "val_seen")
                    unseen = read_metrics(job_dir / "val_unseen", "val_unseen")
                except UserError as error:
                    print("[skip] {}/{}: {}".format(setting, tag, error),
                          file=sys.stderr)
                    continue
                results.append({
                    "parameters": params,
                    "metrics_seen": seen,
                    "metrics_unseen": unseen,
                    "diagnostics": read_adapter_diagnostics(job_dir / "val_unseen"),
                })
            winner, ranked = select_config(
                results, source_metrics, spec["selection"]
            )
            selected = {
                "schema": "navtta.vln_tta_selected_config.v1",
                "benchmark": spec["benchmark"],
                "setting": setting,
                "method": method,
                "order_seed": 0,
                "source_metrics": source_metrics,
                "selection_rule": spec["selection"],
                "winner": winner,
                "used_fallback": winner is None,
                "fallback_parameters": (
                    method_spec.get("paper_default") if winner is None else None
                ),
                "ranked": ranked,
            }
            cell_dir = out_dir / setting / method
            cell_dir.mkdir(parents=True, exist_ok=True)
            with (cell_dir / "selected_config.json").open("w", encoding="utf-8") as stream:
                json.dump(selected, stream, indent=2, sort_keys=True)
            selections[(setting, method)] = selected
            status = "FALLBACK(paper_default)" if winner is None else "selected"
            print("[select] {}/{}: {}".format(setting, method, status))
    return selections


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="path to a consistency search spec")
    parser.add_argument("--run-tag", default="consistency-v1")
    parser.add_argument("--out-dir", required=True,
                        help="scratch dir for job configs (under vln/results/tuning/...)")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--settings", nargs="*", default=None)
    parser.add_argument("--concurrency", type=int, default=None,
                        help="override same-(model,method) parallelism")
    parser.add_argument("--final-stage", action="store_true",
                        help="use IDEA final_opt_steps and re-eval winners")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--select-only", action="store_true",
                        help="skip launching; read completed runs and write "
                             "selected_config.json per cell")
    parser.add_argument("--source-root", default=None,
                        help="root of the Source eval results "
                             "(<setting>/<split>/console.log); required for --select-only")
    args = parser.parse_args(argv)
    if args.select_only:
        if not args.source_root:
            raise UserError("--select-only requires --source-root")
        run_selection(
            args.spec, args.out_dir, args.source_root, args.run_tag,
            set(args.methods or []), set(args.settings or []), args.final_stage,
        )
        return 0
    run_search(
        args.spec, args.run_tag, set(args.methods or []), set(args.settings or []),
        args.out_dir, args.concurrency, args.dry_run, args.final_stage,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
