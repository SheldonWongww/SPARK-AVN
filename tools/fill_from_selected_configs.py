"""Fill NavTTA_benchmark_results.xlsx from consistency-search selected configs.

Reads ``selected_config.json`` files produced by
``vln/scripts/run_consistency_hparam_search.py --select-only`` and writes the
frozen two-split metrics into the VLN sheets (R2R, REVERIE, R2R-CE) for every
method row except OURS.  When a cell used the honest paper-default fallback
(no candidate cleared the no-regression floor), its metrics are only written if
the fallback run's own metrics are supplied; otherwise the cell is left blank
and reported so it is never silently fabricated.

The row/column layout is imported from ``gen_vln_benchmark_xlsx`` so this filler
and the workbook generator cannot drift apart.

Usage:
  python tools/fill_from_selected_configs.py SELECTED_JSON [SELECTED_JSON ...]
      [--xlsx PATH] [--dry-run]

Each SELECTED_JSON is a ``navtta.vln_tta_selected_config.v1`` document; its
winner block carries ``metrics_seen`` / ``metrics_unseen`` (uppercased metric
keys) that map directly onto the sheet columns.
"""
import argparse
import json
import os
import sys

from openpyxl import load_workbook

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_vln_benchmark_xlsx import BENCHMARKS, METHODS  # noqa: E402

DEFAULT_XLSX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs/literature/NavTTA_benchmark_results.xlsx",
)

# setting token -> (sheet, base_model row label)
SETTING_MAP = {
    "duet-r2r": ("R2R", "DUET"),
    "hamt-r2r": ("R2R", "HAMT"),
    "goat-r2r": ("R2R", "GOAT"),
    "duet-reverie": ("REVERIE", "DUET"),
    "hamt-reverie": ("REVERIE", "HAMT"),
    "goat-reverie": ("REVERIE", "GOAT"),
    "etpnav-r2r-ce": ("R2R-CE", "ETPNav"),
    "bevbert-r2r-ce": ("R2R-CE", "BevBert"),
}
# method token -> data-key row label ("+ Tent" etc.)
METHOD_LABEL = {
    "tent": "Tent", "fstta": "FSTTA", "eam": "EAM",
    "feedtta": "FeedTTA", "atena": "ATENA", "idea": "IDEA",
}
# search-split name -> sheet split label
SPLIT_LABEL = {"val_seen": "Val Seen", "val_unseen": "Val Unseen"}


def _mkey(metric_header):
    return metric_header.replace("↑", "").replace("↓", "").strip()


def _method_row_offset(method_name):
    for i, m in enumerate(METHODS):
        if m == "Source" and method_name == "Source":
            return i
        if m.startswith("+") and m[1:].strip() == method_name:
            return i
    raise KeyError(method_name)


def selected_to_cells(document):
    """Yield (sheet, base_model, method_label, split_label, metrics) rows.

    Uses the winner's per-split metrics.  Fallback cells (no winner) are skipped
    unless the document explicitly carries ``fallback_metrics`` for both splits.
    """
    setting = document["setting"]
    method = document["method"]
    if setting not in SETTING_MAP:
        raise KeyError("unknown setting {}".format(setting))
    if method not in METHOD_LABEL:
        raise KeyError("unknown method {}".format(method))
    sheet, base_model = SETTING_MAP[setting]
    method_label = METHOD_LABEL[method]

    winner = document.get("winner")
    if winner is not None:
        per_split = {
            "val_seen": winner.get("metrics_seen", {}),
            "val_unseen": winner.get("metrics_unseen", {}),
        }
    else:
        fallback = document.get("fallback_metrics")
        if not fallback:
            return  # nothing to write; honest blank
        per_split = {
            "val_seen": fallback.get("val_seen", {}),
            "val_unseen": fallback.get("val_unseen", {}),
        }
    for split, metrics in per_split.items():
        if metrics:
            yield sheet, base_model, method_label, SPLIT_LABEL[split], metrics


def fill(documents, xlsx_path, dry_run=False):
    wb = load_workbook(xlsx_path)
    bench_map = {b[0]: b for b in BENCHMARKS}
    planned = []
    for document in documents:
        for sheet, base_model, method_label, split_label, metrics in \
                selected_to_cells(document):
            _, base_models, splits, metric_headers, _ = bench_map[sheet]
            if split_label not in splits:
                continue
            n_metrics = len(metric_headers)
            metric_pos = {_mkey(h): j for j, h in enumerate(metric_headers)}
            bi = base_models.index(base_model)
            row = 3 + bi * len(METHODS) + _method_row_offset(method_label)
            si = splits.index(split_label)
            ws = wb[sheet]
            for mk, value in metrics.items():
                key = _mkey(mk)
                if key not in metric_pos:
                    continue  # metric not present on this sheet (e.g. OSR on R2R)
                col = 2 + si * n_metrics + metric_pos[key]
                planned.append((sheet, row, col, key, float(value)))
                if not dry_run:
                    cell = ws.cell(row=row, column=col, value=float(value))
                    cell.number_format = "0.00"
    if dry_run:
        for sheet, row, col, key, value in planned:
            print("{}!R{}C{} {} = {:.2f}".format(sheet, row, col, key, value))
        print("# DRY RUN: {} cells".format(len(planned)))
    else:
        wb.save(xlsx_path)
        print("filled cells:", len(planned))
    return planned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selected", nargs="+", help="selected_config.json files")
    parser.add_argument("--xlsx", default=DEFAULT_XLSX)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    documents = []
    for path in args.selected:
        with open(path, "r", encoding="utf-8") as stream:
            documents.append(json.load(stream))
    fill(documents, args.xlsx, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
