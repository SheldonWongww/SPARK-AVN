#!/usr/bin/env python3
"""Summarize one four-GPU VLN Source evaluation batch."""

import argparse
import csv
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "vln" / "results" / "source"
DEFAULT_REFERENCE = (
    REPO_ROOT / "vln" / "results" / "legacy"
    / "excel_source_metrics.json"
)

SETTINGS = (
    "duet-r2r",
    "duet-reverie",
    "hamt-r2r",
    "hamt-reverie",
    "goat-r2r",
    "goat-reverie",
    "etpnav-r2r-ce",
    "bevbert-r2r-ce",
    "streamvln-r2r-ce",
)
SPLITS = ("val_seen", "val_unseen")
EXPECTED_EPISODES = {
    "duet-r2r": (1021, 2349),
    "hamt-r2r": (1021, 2349),
    "goat-r2r": (1021, 2349),
    "duet-reverie": (1423, 3521),
    "hamt-reverie": (1423, 3521),
    "goat-reverie": (1423, 3521),
    "etpnav-r2r-ce": (778, 1839),
    "bevbert-r2r-ce": (778, 1839),
    "streamvln-r2r-ce": (778, 1839),
}
REFERENCE_IDENTITY = {
    "duet-r2r": ("DUET", "R2R"),
    "duet-reverie": ("DUET", "REVERIE"),
    "hamt-r2r": ("HAMT", "R2R"),
    "hamt-reverie": ("HAMT", "REVERIE"),
    "goat-r2r": ("GOAT", "R2R"),
    "goat-reverie": ("GOAT", "REVERIE"),
    "etpnav-r2r-ce": ("ETPNav", "R2R-CE"),
    "bevbert-r2r-ce": ("BEVBert", "R2R-CE"),
    "streamvln-r2r-ce": ("StreamVLN", "R2R-CE"),
}
PAPER_NATIVE_PROTOCOL = {
    "duet-r2r": "discrete",
    "duet-reverie": "discrete",
    "hamt-r2r": "discrete",
    "hamt-reverie": "discrete",
    "goat-r2r": "discrete",
    "goat-reverie": "discrete",
    "etpnav-r2r-ce": "v1.2-native",
    "bevbert-r2r-ce": "v1.2-native",
    "streamvln-r2r-ce": "v1.3-native",
}
METRIC_ORDER = (
    "TL", "NE", "OSR", "SR", "SPL", "nDTW", "SDTW", "CLS",
    "RGS", "RGSPL",
)
ALIASES = {
    "tl": "TL",
    "lengths": "TL",
    "path_length": "TL",
    "ne": "NE",
    "nav_error": "NE",
    "distance_to_goal": "NE",
    "os": "OSR",
    "osr": "OSR",
    "oracle_sr": "OSR",
    "oracle_success": "OSR",
    "sr": "SR",
    "success": "SR",
    "spl": "SPL",
    "ndtw": "nDTW",
    "sdtw": "SDTW",
    "cls": "CLS",
    "rgs": "RGS",
    "rgspl": "RGSPL",
}
RATE_METRICS = {"OSR", "SR", "SPL", "nDTW", "SDTW", "RGS", "RGSPL"}
REQUIRED_METRICS = {
    "r2r": {"SR", "SPL"},
    "reverie": {"SR", "SPL", "RGSPL"},
    "r2r-ce": {"SR", "SPL"},
}
NUMBER = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"


class SummaryError(RuntimeError):
    pass


def decimal(value):
    return value if isinstance(value, Decimal) else Decimal(str(value))


def read_console(directory):
    path = directory / "console.log"
    if not path.is_file():
        raise SummaryError("missing console log: {}".format(path))
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def canonicalize(raw, ratios=False):
    result = {}
    for key, value in raw.items():
        name = ALIASES.get(str(key).lower())
        if name is None:
            continue
        value = decimal(value)
        if ratios and name in RATE_METRICS:
            value *= 100
        result[name] = value
    return result


def parse_discrete(directory, split):
    text = read_console(directory)
    rows = re.findall(r"^Env name:\s*([^,]+),\s*(.+)$", text, re.MULTILINE)
    rows = [tail for name, tail in rows if name.strip() == split]
    if not rows:
        raise SummaryError("no discrete aggregate row in {}".format(directory))
    raw = dict(re.findall(
        r"([A-Za-z][A-Za-z0-9_]*):\s*({})".format(NUMBER), rows[-1]
    ))
    counts = re.findall(r"eval\s+([0-9]+)\s+predictions", text)
    if not counts:
        raise SummaryError("no discrete episode count in {}".format(directory))
    return canonicalize(raw), int(counts[-1]), "discrete"


def parse_continuous(directory, split, expected_protocol):
    text = read_console(directory)
    files = list((directory / "metrics").rglob(
        "stats_ckpt_*_{}.json".format(split)
    ))
    if len(files) > 1:
        raise SummaryError("ambiguous continuous metrics: {}".format(files))
    if files:
        raw = json.loads(
            files[0].read_text(encoding="utf-8"),
            parse_float=Decimal,
            parse_int=Decimal,
        )
    else:
        pairs = re.findall(
            r"Average episode ([A-Za-z0-9_]+):\s*({})".format(NUMBER), text
        )
        if not pairs:
            raise SummaryError("no continuous aggregate row in {}".format(directory))
        raw = dict(pairs)
    counts = re.findall(r"Episodes evaluated:\s*([0-9.]+)", text)
    if not counts:
        raise SummaryError("no continuous episode count in {}".format(directory))
    if expected_protocol == "v1.2-native":
        marker = r"R2R_VLNCE_v1[-_]2"
    else:
        marker = r"R2R_VLNCE_v1[-_]3"
    if not re.search(marker, text):
        raise SummaryError(
            "{} does not show expected protocol {}".format(
                directory, expected_protocol
            )
        )
    return canonicalize(raw, ratios=True), int(decimal(counts[-1])), expected_protocol


def parse_streamvln(directory):
    path = directory / "result.json"
    if not path.is_file():
        raise SummaryError("missing StreamVLN result: {}".format(path))
    documents = [
        json.loads(line, parse_float=Decimal, parse_int=Decimal)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    aggregates = [
        item for item in documents
        if all(key in item for key in (
            "sucs_all", "spls_all", "oss_all", "ones_all", "length"
        ))
    ]
    if len(aggregates) != 1:
        raise SummaryError(
            "expected one StreamVLN aggregate in {}; found {}".format(
                path, len(aggregates)
            )
        )
    item = aggregates[0]
    raw = {
        "success": item["sucs_all"],
        "spl": item["spls_all"],
        "os": item["oss_all"],
        "distance_to_goal": item["ones_all"],
    }
    return canonicalize(raw, ratios=True), int(item["length"]), "v1.3-native"


def parse_result(setting, split, directory, ce_data_version):
    if setting in {"etpnav-r2r-ce", "bevbert-r2r-ce"}:
        return parse_continuous(directory, split, ce_data_version)
    if setting == "streamvln-r2r-ce":
        return parse_streamvln(directory)
    return parse_discrete(directory, split)


def validate_metrics(setting, split, metrics):
    if setting.endswith("-reverie"):
        family = "reverie"
    elif setting.endswith("-r2r-ce"):
        family = "r2r-ce"
    else:
        family = "r2r"
    missing = sorted(REQUIRED_METRICS[family] - set(metrics))
    if missing:
        raise SummaryError(
            "missing required metrics for {}/{}: {}".format(
                setting, split, ", ".join(missing)
            )
        )
    nonfinite = sorted(
        name for name, value in metrics.items() if not value.is_finite()
    )
    if nonfinite:
        raise SummaryError(
            "non-finite metrics for {}/{}: {}".format(
                setting, split, ", ".join(nonfinite)
            )
        )


def reference_records(reference_path):
    document = json.loads(
        reference_path.read_text(encoding="utf-8"),
        parse_float=Decimal,
        parse_int=Decimal,
    )
    return {
        (item["baseline"], item["benchmark"]["name"]): item
        for item in document["records"]
    }


def comparison_status(value, reference, integer=False):
    if integer:
        compared_value = value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        compared_reference = reference.quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
        status = (
            "MATCH_INTEGER"
            if compared_value == compared_reference
            else "MISMATCH_INTEGER"
        )
        return status, compared_value, compared_reference
    if value == reference:
        return "MATCH_EXACT", value, reference
    quantum = Decimal(1).scaleb(reference.as_tuple().exponent)
    if value.quantize(quantum, rounding=ROUND_HALF_UP) == reference:
        return "MATCH_ROUNDED", value.quantize(
            quantum, rounding=ROUND_HALF_UP
        ), reference
    return "MISMATCH", value, reference


def reference_protocol(setting, reference):
    declared = reference.get("benchmark", {}).get("protocol")
    if declared in {
        "discrete", "v1.2-native", "v1.3-native", "v1.3-unified"
    }:
        return declared
    return PAPER_NATIVE_PROTOCOL[setting]


def integer_primary_comparison(setting, metric):
    benchmark = REFERENCE_IDENTITY[setting][1]
    return benchmark in {"R2R", "R2R-CE"} and metric in {"SR", "SPL"}


def used_for_overall(setting, metric):
    benchmark = REFERENCE_IDENTITY[setting][1]
    if benchmark in {"R2R", "R2R-CE"}:
        return metric in {"SR", "SPL"}
    return True


def rounded_integer(metrics, name, setting):
    if REFERENCE_IDENTITY[setting][1] not in {"R2R", "R2R-CE"}:
        return ""
    value = metrics.get(name)
    if value is None:
        return ""
    return format(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP), "f")


def summarize(source_root, run_tag, output, ce_data_version, reference_path):
    references = reference_records(reference_path)
    rows = []
    summaries = []
    for setting in SETTINGS:
        reference = references[REFERENCE_IDENTITY[setting]]
        setting_tag = "{}-{}".format(run_tag, setting)
        for split_index, split in enumerate(SPLITS):
            directory = source_root / setting_tag / setting / split
            metrics, episodes, protocol = parse_result(
                setting, split, directory, ce_data_version
            )
            validate_metrics(setting, split, metrics)
            expected = EXPECTED_EPISODES[setting][split_index]
            if episodes != expected:
                raise SummaryError(
                    "incomplete {}/{}: expected {}, got {} episodes".format(
                        setting, split, expected, episodes
                    )
                )
            raw_reference = reference.get("metrics", {}).get(split)
            expected_protocol = reference_protocol(setting, reference)
            comparable = {} if raw_reference is None else {
                ALIASES[key.lower()]: decimal(value)
                for key, value in raw_reference.items()
                if key.lower() in ALIASES
            }
            missing_comparison_metrics = sorted(set(comparable) - set(metrics))
            if missing_comparison_metrics:
                raise SummaryError(
                    "missing reference comparison metrics for {}/{}: {}".format(
                        setting, split, ", ".join(missing_comparison_metrics)
                    )
                )
            statuses = []
            for metric in METRIC_ORDER:
                if metric not in metrics:
                    continue
                value = metrics[metric]
                expected_value = comparable.get(metric)
                if expected_value is None:
                    status = (
                        "NO_REFERENCE_SPLIT"
                        if raw_reference is None else "NO_REFERENCE_METRIC"
                    )
                    difference = ""
                    comparison_value = ""
                    comparison_reference = ""
                elif (
                    expected_protocol in {
                        "v1.2-native", "v1.3-native", "v1.3-unified"
                    }
                    and protocol != expected_protocol
                ):
                    status = "NOT_COMPARABLE_PROTOCOL"
                    difference = format(abs(value - expected_value), "f")
                    comparison_value = ""
                    comparison_reference = ""
                else:
                    status, compared_value, compared_reference = comparison_status(
                        value,
                        expected_value,
                        integer=integer_primary_comparison(setting, metric),
                    )
                    difference = format(abs(value - expected_value), "f")
                    comparison_value = format(compared_value, "f")
                    comparison_reference = format(compared_reference, "f")
                primary = expected_value is not None and used_for_overall(
                    setting, metric
                )
                if primary:
                    statuses.append(status)
                rows.append({
                    "setting": setting,
                    "split": split,
                    "protocol": protocol,
                    "episodes": episodes,
                    "metric": metric,
                    "value": format(value, "f"),
                    "reference": (
                        "" if expected_value is None
                        else format(expected_value, "f")
                    ),
                    "comparison_value": comparison_value,
                    "comparison_reference": comparison_reference,
                    "abs_delta": difference,
                    "status": status,
                    "used_for_overall": str(primary).lower(),
                    "reference_id": reference["id"],
                })
            if "NOT_COMPARABLE_PROTOCOL" in statuses:
                overall = "NOT_COMPARABLE_PROTOCOL"
            elif any(status.startswith("MISMATCH") for status in statuses):
                overall = "MISMATCH_INTEGER" if any(
                    status == "MISMATCH_INTEGER" for status in statuses
                ) else "MISMATCH"
            elif statuses and all(
                status == "MATCH_INTEGER" for status in statuses
            ):
                overall = "MATCH_INTEGER"
            elif "MATCH_ROUNDED" in statuses:
                overall = "MATCH_ROUNDED"
            elif "MATCH_EXACT" in statuses:
                overall = "MATCH_EXACT"
            else:
                overall = "NO_REFERENCE"
            summaries.append({
                "setting": setting,
                "split": split,
                "episodes": episodes,
                "protocol": protocol,
                "SR": format(metrics.get("SR", Decimal("NaN")), "f"),
                "SPL": format(metrics.get("SPL", Decimal("NaN")), "f"),
                "SR_integer": rounded_integer(metrics, "SR", setting),
                "SPL_integer": rounded_integer(metrics, "SPL", setting),
                "comparison": overall,
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "setting", "split", "protocol", "episodes", "metric", "value",
            "reference", "comparison_value", "comparison_reference",
            "abs_delta", "status", "used_for_overall", "reference_id",
        ))
        writer.writeheader()
        writer.writerows(rows)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--ce-data-version",
        choices=("v1.2-native", "v1.3-unified"),
        default="v1.2-native",
    )
    parser.add_argument(
        "--source-root", type=Path, default=DEFAULT_SOURCE_ROOT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reference", type=Path, default=DEFAULT_REFERENCE,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    summaries = summarize(
        args.source_root,
        args.run_tag,
        args.output,
        args.ce_data_version,
        args.reference,
    )
    print("reference={}".format(args.reference))
    print(
        "setting,split,episodes,protocol,SR,SPL,"
        "SR_integer,SPL_integer,comparison"
    )
    for item in summaries:
        print(",".join(str(item[key]) for key in (
            "setting", "split", "episodes", "protocol", "SR", "SPL",
            "SR_integer", "SPL_integer", "comparison",
        )))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, json.JSONDecodeError, SummaryError) as error:
        raise SystemExit("error: {}".format(error))
