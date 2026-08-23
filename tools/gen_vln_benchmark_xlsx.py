"""Generate a WPS/Excel workbook to collect reported Nav-TTA results.

Sheets cover VLN (REVERIE, R2R, R2R-CE), AVN and ObjectNav. Each sheet is laid
out like the paper tables: a merged group header per split/scenario, a
per-metric sub-header with arrows, and method rows grouped by navigation model.
Data cells are left blank for manual entry (literature reuse, not reproduction).
"""
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

# ---- shared styles ----
THIN = Side(style="thin", color="000000")
MED = Side(style="medium", color="000000")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center")
HDR_FILL = PatternFill("solid", fgColor="D9E1F2")
GROUP_FILL = PatternFill("solid", fgColor="F2F2F2")
BOLD = Font(bold=True)

# Method rows: Source baseline + reused TTA methods + our reserved row.
# "+ OURS" is intentionally left blank for future results.
METHODS = [
    "Source", "+ Tent", "+ FSTTA", "+ EAM", "+ FeedTTA",
    "+ ATENA", "+ IDEA", "+ OURS",
]

# benchmark definitions: (sheet, base_models, splits, metrics, has_time)
BENCHMARKS = [
    (
        "REVERIE",
        ["HAMT", "DUET", "GOAT"],
        ["Val Seen", "Val Unseen", "Test Unseen"],
        ["OSR↑", "SR↑", "SPL↑", "RGSPL↑"],
        True,
    ),
    (
        "R2R",
        ["HAMT", "DUET", "GOAT"],
        ["Val Seen", "Val Unseen"],
        ["TL", "NE↓", "SR↑", "SPL↑"],
        True,
    ),
    (
        "R2R-CE",
        ["BevBert", "ETPNav", "StreamVLN"],
        ["Val Seen", "Val Unseen"],
        ["TL", "NE↓", "OSR↑", "SR↑", "SPL↑"],
        True,
    ),
    (
        "AVN",
        ["SMT+AUDIO", "ENMuS^3"],
        ["Single-source", "Multi-source", "Noisy"],
        ["SR↑", "SPL↑", "SNA↑", "DTG↓"],
        False,
    ),
    (
        "ObjectNav",
        ["PONI", "GOAL"],
        ["Val"],
        ["SR↑", "SPL↑", "DTS↓"],
        False,
    ),
]


def build_sheet(ws, base_models, splits, metrics, has_time):
    n_metrics = len(metrics)
    # column 1 = Methods+Model, then splits*metrics, then optional Time
    total_cols = 1 + len(splits) * n_metrics + (1 if has_time else 0)

    # --- row 1: group header ---
    ws.cell(row=1, column=1, value="Methods+Model")
    ws.merge_cells(start_row=1, start_column=1, end_row=2, end_column=1)
    col = 2
    for split in splits:
        ws.cell(row=1, column=col, value=f"{ws.title} {split}")
        ws.merge_cells(start_row=1, start_column=col, end_row=1, end_column=col + n_metrics - 1)
        col += n_metrics
    if has_time:
        ws.cell(row=1, column=col, value="Time(ms)")
        ws.merge_cells(start_row=1, start_column=col, end_row=2, end_column=col)

    # --- row 2: metric sub-header ---
    col = 2
    for _ in splits:
        for m in metrics:
            ws.cell(row=2, column=col, value=m)
            col += 1

    # style headers
    for r in (1, 2):
        for cc in range(1, total_cols + 1):
            cell = ws.cell(row=r, column=cc)
            cell.font = BOLD
            cell.alignment = CENTER
            cell.fill = HDR_FILL
            cell.border = BORDER

    # --- data rows grouped by base model ---
    row = 3
    for bm in base_models:
        for method in METHODS:
            label = bm if method == "Source" else method
            cell = ws.cell(row=row, column=1, value=label)
            cell.alignment = LEFT
            # Force text so Excel/WPS keeps the literal "+" (leading apostrophe).
            if label.startswith("+"):
                cell.quotePrefix = True
            if method == "Source":
                cell.font = BOLD
                cell.fill = GROUP_FILL
            cell.border = BORDER
            for cc in range(2, total_cols + 1):
                dc = ws.cell(row=row, column=cc)
                dc.alignment = CENTER
                dc.border = BORDER
                if method == "Source":
                    dc.fill = GROUP_FILL
            row += 1

    # column widths
    ws.column_dimensions["A"].width = 20
    for cc in range(2, total_cols + 1):
        ws.column_dimensions[get_column_letter(cc)].width = 9
    ws.row_dimensions[1].height = 20
    ws.row_dimensions[2].height = 20

    # freeze panes below header and right of label column
    ws.freeze_panes = "B3"


def main():
    wb = Workbook()
    wb.remove(wb.active)
    for name, bases, splits, metrics, has_time in BENCHMARKS:
        ws = wb.create_sheet(title=name)
        build_sheet(ws, bases, splits, metrics, has_time)

    # a notes sheet documenting conventions
    notes = wb.create_sheet(title="Notes")
    lines = [
        "Nav-TTA reported results collection (literature reuse, not reproduction).",
        "",
        "Splits / metrics per benchmark:",
        "  REVERIE   : OSR / SR / SPL / RGSPL, arrows all higher-better.",
        "  R2R       : TL (neutral), NE (lower), OSR / SR / SPL.",
        "  R2R-CE    : TL (neutral), NE (lower), OSR / SR / SPL.",
        "  AVN       : SR / SPL / SNA (higher), DTG (lower); scenarios = Single-source / Multi-source / Noisy.",
        "  ObjectNav : SR / SPL (higher), DTS (lower).",
        "",
        "Navigation (base) models:",
        "  REVERIE / R2R : HAMT, DUET, GOAT.",
        "  R2R-CE        : BevBert, ETPNav, StreamVLN.",
        "  AVN           : SMT+AUDIO, ENMuS^3.",
        "  ObjectNav     : PONI, GOAL.",
        "",
        "Method rows: Source, Tent, FSTTA, EAM, FeedTTA, ATENA, IDEA, OURS (reserved, blank).",
        "'+' method labels use a leading apostrophe (quote prefix) so the plus stays literal text.",
        "",
        "Formatting: bold = best, underline = second (apply manually per column once filled).",
        "SR/SPL for R2R are often reported as integers in papers (rounding);",
        "  splits are not multiples of 100 (Val Seen=1021, Val Unseen=2349).",
        "Time(ms) columns exist only for the VLN sheets (per source-paper reporting).",
        "",
        "TODO: ObjectNav image was not received; the single 'Val' split and DTS naming",
        "  need confirmation once the reference table is available.",
    ]
    for i, ln in enumerate(lines, start=1):
        notes.cell(row=i, column=1, value=ln)
    notes.column_dimensions["A"].width = 100

    out = "/Users/bytedance/keyan/NaCoTTA/NavTTA/docs/literature/NavTTA_benchmark_results.xlsx"
    wb.save(out)
    print("saved:", out)


if __name__ == "__main__":
    main()
