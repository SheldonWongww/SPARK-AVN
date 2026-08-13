#!/usr/bin/env python3
"""Rebuild the VLN hyperparameter appendix with SR-first ranking.

The search scheduler promoted candidates with its historical benchmark metric
(SPL for R2R/R2R-CE and RGSPL for REVERIE).  This script does not alter that
provenance.  It re-ranks the five completed full-val finalists in every
method/model/benchmark group by SR first and SPL second, while retaining the
original scheduler rank in the companion CSV.  All reported gains use the
standard argmax Source.  FeedTTA's sampled no-update control is retained only
as an adaptation-control diagnostic.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


VLN_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = VLN_ROOT / "results" / "analysis" / "hparam_search"
CSV_PATH = ANALYSIS_ROOT / "val_seen_top5_by_model_benchmark.csv"
MARKDOWN_PATH = ANALYSIS_ROOT / "VAL_SEEN_TOP5_BY_MODEL_BENCHMARK.md"
REPORT_PATH = VLN_ROOT / "VLN_TTA_REPORT.md"

METHOD_ORDER = ("tent", "fstta", "eam", "feedtta", "atena")
METHOD_LABEL = {
    "tent": "Tent",
    "fstta": "FSTTA",
    "eam": "EAM",
    "feedtta": "FeedTTA†",
    "atena": "ATENA†",
}
SETTING_ORDER = (
    ("duet", "r2r"),
    ("hamt", "r2r"),
    ("goat", "r2r"),
    ("duet", "reverie"),
    ("hamt", "reverie"),
    ("goat", "reverie"),
    ("etpnav", "r2r-ce"),
    ("bevbert", "r2r-ce"),
)
MODEL_LABEL = {
    "duet": "DUET",
    "hamt": "HAMT",
    "goat": "GOAT",
    "etpnav": "ETPNav",
    "bevbert": "BEVBert",
}
BENCHMARK_LABEL = {"r2r": "R2R", "reverie": "REVERIE", "r2r-ce": "R2R-CE"}
AUDIT_LABEL = {
    "no_coarse_flag": "none",
    "no_fixed_order_collapse_signal": "none",
    "warning_fixed_order_degradation": "warning",
    "coarse_warning": "coarse warning",
    "coarse_flag": "coarse flag",
    "severe_fixed_order_collapse": "**severe collapse**",
}
ANALYSIS_METRICS = ("sr", "spl", "rgs", "rgspl")


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return math.nan
    return float(value)


def rank_key(row: dict[str, str]) -> tuple[float, float, str]:
    return (
        number(row, "delta_sr_pp"),
        number(row, "delta_spl_pp"),
        row.get("parameters_json", ""),
    )


def metric_precision(benchmark: str) -> int:
    return 4 if benchmark == "r2r-ce" else 2


def transition(row: dict[str, str], metric: str) -> str:
    benchmark = row["benchmark"]
    precision = metric_precision(benchmark)
    source = number(row, f"source_{metric}")
    target = number(row, f"tta_{metric}")
    delta = number(row, f"delta_{metric}_pp")
    if any(math.isnan(item) for item in (source, target, delta)):
        return "—"
    return (
        f"{source:.{precision}f}→{target:.{precision}f} "
        f"({delta:+.{precision}f})"
    )


def reverie_supplement(row: dict[str, str]) -> str:
    if row["benchmark"] != "reverie":
        return "—"
    return f"RGS {transition(row, 'rgs')}; RGSPL {transition(row, 'rgspl')}"


def standard_argmax_sources(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    """Recover each setting's canonical argmax Source from non-FeedTTA rows."""
    result: dict[tuple[str, str], dict[str, str]] = {}
    for model, benchmark in SETTING_ORDER:
        setting_rows = [
            row
            for row in rows
            if row["model"] == model
            and row["benchmark"] == benchmark
            and row["tta_method"] != "feedtta"
        ]
        metrics: dict[str, str] = {}
        for metric in ANALYSIS_METRICS:
            values = [row.get(f"source_{metric}", "") for row in setting_rows]
            values = [value for value in values if value != ""]
            if not values:
                metrics[metric] = ""
                continue
            reference = float(values[0])
            if any(not math.isclose(float(value), reference, abs_tol=1e-9) for value in values):
                raise ValueError(
                    f"conflicting standard Source {metric} values for {model}/{benchmark}"
                )
            metrics[metric] = values[0]
        result[(model, benchmark)] = metrics
    return result


def apply_standard_source_protocol(rows: list[dict[str, str]]) -> None:
    standards = standard_argmax_sources(rows)
    for row in rows:
        old_protocol = row.get("adaptation_control_protocol") or row.get(
            "source_protocol", "argmax"
        )
        row["adaptation_control_protocol"] = old_protocol
        standard = standards[(row["model"], row["benchmark"])]
        for metric in ANALYSIS_METRICS:
            source_key = f"source_{metric}"
            target_key = f"tta_{metric}"
            delta_key = f"delta_{metric}_pp"
            control_key = f"adaptation_control_{metric}"
            control_delta_key = f"delta_{metric}_vs_adaptation_control_pp"

            if not row.get(control_key, ""):
                row[control_key] = row.get(source_key, "")
            control = row.get(control_key, "")
            target = row.get(target_key, "")
            if control != "" and target != "":
                row[control_delta_key] = str(float(target) - float(control))
            else:
                row[control_delta_key] = ""

            row[source_key] = standard[metric]
            if standard[metric] != "" and target != "":
                row[delta_key] = str(float(target) - float(standard[metric]))
            else:
                row[delta_key] = ""
        row["source_protocol"] = "standard_argmax"


def load_and_rank() -> tuple[list[dict[str, str]], dict[tuple[str, str, str], list[dict[str, str]]]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 200:
        raise ValueError(f"expected 200 finalist rows, found {len(rows)}")

    apply_standard_source_protocol(rows)

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        scheduler_rank = row.get("scheduler_rank") or row.get("rank")
        if not scheduler_rank:
            raise ValueError("row lacks scheduler rank")
        row["scheduler_rank"] = scheduler_rank
        row["analysis_primary_metric"] = "SR"
        row["analysis_secondary_metric"] = "SPL"
        row["scheduler_primary_metric"] = (
            row.get("scheduler_primary_metric") or row.get("primary_metric", "")
        )
        row["source_scheduler_primary"] = (
            row.get("source_scheduler_primary") or row.get("source_primary", "")
        )
        row["tta_scheduler_primary"] = (
            row.get("tta_scheduler_primary") or row.get("tta_primary", "")
        )
        row["delta_scheduler_primary_pp"] = (
            row.get("delta_scheduler_primary_pp") or row.get("delta_primary_pp", "")
        )
        groups[(row["tta_method"], row["model"], row["benchmark"])].append(row)

    if len(groups) != 40 or any(len(items) != 5 for items in groups.values()):
        raise ValueError("expected 40 groups with exactly five finalists each")

    ranked_rows: list[dict[str, str]] = []
    for key, items in groups.items():
        ordered = sorted(items, key=rank_key, reverse=True)
        for sr_rank, row in enumerate(ordered, start=1):
            row["sr_rank"] = str(sr_rank)
            ranked_rows.append(row)
        groups[key] = ordered
    return ranked_rows, groups


def write_csv(rows: list[dict[str, str]]) -> None:
    prefix = [
        "setting",
        "model",
        "benchmark",
        "tta_method",
        "sr_rank",
        "scheduler_rank",
        "source_protocol",
        "adaptation_control_protocol",
        "analysis_primary_metric",
        "analysis_secondary_metric",
        "scheduler_primary_metric",
        "source_scheduler_primary",
        "tta_scheduler_primary",
        "delta_scheduler_primary_pp",
        "adaptation_control_sr",
        "adaptation_control_spl",
        "adaptation_control_rgs",
        "adaptation_control_rgspl",
        "delta_sr_vs_adaptation_control_pp",
        "delta_spl_vs_adaptation_control_pp",
        "delta_rgs_vs_adaptation_control_pp",
        "delta_rgspl_vs_adaptation_control_pp",
    ]
    omitted = {
        "rank",
        "primary_metric",
        "source_primary",
        "tta_primary",
        "delta_primary_pp",
        *prefix,
    }
    original = list(rows[0])
    fieldnames = prefix + [field for field in original if field not in omitted]

    method_index = {method: index for index, method in enumerate(METHOD_ORDER)}
    setting_index = {setting: index for index, setting in enumerate(SETTING_ORDER)}
    ordered = sorted(
        rows,
        key=lambda row: (
            setting_index[(row["model"], row["benchmark"])],
            method_index[row["tta_method"]],
            int(row["sr_rank"]),
        ),
    )
    temporary = CSV_PATH.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    temporary.replace(CSV_PATH)


def write_markdown(groups: dict[tuple[str, str, str], list[dict[str, str]]]) -> None:
    lines = [
        "# VLN `val_seen` hyperparameter finalists: SR-first ranking",
        "",
        "This appendix re-ranks the five completed full-`val_seen` finalists in each",
        "model/benchmark/method group. **SR is the primary analysis metric and SPL is",
        "the secondary metric.** REVERIE RGS/RGSPL are retained only as supplementary",
        "object-grounding metrics. The original scheduler rank is shown because candidate",
        "promotion used the historical SPL/RGSPL objective; this is a post-hoc re-ranking",
        "of completed finalists, not a new exhaustive SR-driven search.",
        "",
        "- Every reported Source→TTA delta uses the standard argmax Source.",
        "- FeedTTA† still executes sampled actions; its sampled no-update control is retained",
        "  in the CSV only as an internal adaptation-control diagnostic.",
        "- FeedTTA† and ATENA† consume binary episode feedback and are not unsupervised TTA.",
        "- Values are absolute percentage-point changes on canonical-order, seed-0 `val_seen`.",
        "",
    ]
    for model, benchmark in SETTING_ORDER:
        title = f"{MODEL_LABEL[model]}–{BENCHMARK_LABEL[benchmark]}"
        lines.extend(
            [
                f"## {title}",
                "",
                "| TTA | SR rank | Scheduler rank | Hyperparameters | SR standard Source→TTA (Δ) | SPL standard Source→TTA (Δ) | REVERIE supplement | Late-stream audit |",
                "|---|---:|---:|---|---:|---:|---|---|",
            ]
        )
        for method in METHOD_ORDER:
            for row in groups[(method, model, benchmark)]:
                audit = AUDIT_LABEL.get(row["audit"], row["audit"])
                lines.append(
                    "| {method} | {sr_rank} | {scheduler_rank} | `{parameters}` | "
                    "{sr} | {spl} | {supplement} | {audit} |".format(
                        method=METHOD_LABEL[method],
                        sr_rank=row["sr_rank"],
                        scheduler_rank=row["scheduler_rank"],
                        parameters=row["parameters"],
                        sr=transition(row, "sr"),
                        spl=transition(row, "spl"),
                        supplement=reverie_supplement(row),
                        audit=audit,
                    )
                )
        lines.append("")

    lines.extend(
        [
            "## Audit label legend",
            "",
            "- `none`: `no_coarse_flag` or `no_fixed_order_collapse_signal`.",
            "- `warning`: `warning_fixed_order_degradation`.",
            "- `coarse warning` / `coarse flag`: fixed-order coarse diagnostics for discrete environments.",
            "- `severe collapse`: statistically supported fixed-order degradation in the continuous environment.",
            "",
            "The companion CSV retains raw SR/SPL/RGS/RGSPL values, both ranking systems,",
            "full parameter JSON, run tags, batch IDs, and audit labels.",
            "",
        ]
    )
    temporary = MARKDOWN_PATH.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(MARKDOWN_PATH)


def signed_delta(row: dict[str, str], metric: str) -> str:
    precision = metric_precision(row["benchmark"])
    return f"{number(row, f'delta_{metric}_pp'):+.{precision}f}"


def sr_winners(
    groups: dict[tuple[str, str, str], list[dict[str, str]]]
) -> dict[tuple[str, str, str], dict[str, str]]:
    return {key: items[0] for key, items in groups.items()}


def shared_choice(
    groups: dict[tuple[str, str, str], list[dict[str, str]]],
    method: str,
    benchmark: str,
) -> tuple[str, list[dict[str, str]], bool]:
    models = {
        "r2r": ("duet", "hamt", "goat"),
        "reverie": ("duet", "hamt", "goat"),
        "r2r-ce": ("etpnav", "bevbert"),
    }[benchmark]
    by_model = {
        model: {row["parameters"]: row for row in groups[(method, model, benchmark)]}
        for model in models
    }
    common = set.intersection(*(set(items) for items in by_model.values()))
    if not common:
        raise ValueError(f"no common finalist for {method}/{benchmark}")

    candidates = []
    for parameters in common:
        rows = [by_model[model][parameters] for model in models]
        sr = [number(row, "delta_sr_pp") for row in rows]
        spl = [number(row, "delta_spl_pp") for row in rows]
        score = (min(sr), statistics.mean(sr), min(spl), statistics.mean(spl))
        candidates.append((score, parameters, rows))
    valid = [item for item in candidates if all(number(row, "delta_sr_pp") > 1e-9 for row in item[2])]
    _, parameters, rows = max(valid or candidates, key=lambda item: item[0])
    return parameters, rows, bool(valid)


def build_report_section(
    groups: dict[tuple[str, str, str], list[dict[str, str]]]
) -> str:
    winners = sr_winners(groups)
    lines = [
        "## 7. Tent、FSTTA、EAM、FeedTTA 与 ATENA 的 `val_seen` 超参数搜索",
        "",
        "本节统一采用 **SR 第一、SPL 第二** 的分析口径。逐 setting 排名先比较",
        "完整 `val_seen` 的 `ΔSR`，相同时再比较 `ΔSPL`；REVERIE 的 RGS/RGSPL",
        "只作为目标定位补充指标。**所有方法均相对标准 argmax Source 报告结果**；",
        "FeedTTA† 的 sampled no-update control 只保留为内部适配诊断。所有增益均为",
        "固定 canonical order、seed 0 下的绝对百分点。",
        "",
        "需要特别说明：历史调度器用 R2R/R2R-CE 的 SPL 和 REVERIE 的 RGSPL",
        "完成分阶段晋级。因此下面是对每组 **5 个已完成 full-val finalist** 的",
        "SR-first 事后重排，不是一次从原始搜索空间重新执行的 SR 驱动搜索。40 个",
        "逐 setting winner 中有 13 个与原 scheduler rank 1 不同；原 rank 仍保留在",
        "附表和 CSV 中以维护 provenance。",
        "",
        "### 7.1 批次与完整性",
        "",
        "五个方法批次的 `SUMMARY.json` 均为 `complete=true`、`terminal=true`、",
        "`errors=[]`。Tent/FSTTA/EAM 使用 search spec SHA256",
        "`389cd63a7d52f68040c920d41c9e0ead331189945e0c313e05f772f8bf645525`；",
        "FeedTTA/ATENA 使用精简 spec SHA256",
        "`24dee7711e9e3168c2340ffbace4678545ce8a8d4681cb5c0438cf36e1203219`。",
        "",
        "| 方法 | Batch | Git commit | smoke | controls | stage1 | stage2 | stage3 | final controls | final | 总任务 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| Tent | `vln-tta-hparam-final-20260810T182823Z` | `9ad8a42` | 8/8 | 8/8 | 280/280 | — | — | 8/8 | 40/40 | 344 |",
        "| FSTTA | `vln-tta-hparam-final-20260810T182823Z` | `9ad8a42` | 8/8 | 8/8 | 160/160 | 64/64 | 80/80 | 8/8 | 40/40 | 368 |",
        "| EAM | `vln-val-seen-hparam-v1-seed0` | `25ead04` | 8/8 | 8/8 | 280/280 | 80/80 | 64/64 | 8/8 | 40/40 | 488 |",
        "| FeedTTA† | `vln-val-seen-hparam-compact-v1-seed0` | `8572227` | 8/8 | 8/8 | 120/120 | 176/176 | — | 8/8 | 40/40 | 360 |",
        "| ATENA† | `vln-val-seen-hparam-compact-v1-seed0` | `8572227` | 8/8 | 8/8 | 40/40 | 80/80 | 64/64 | 8/8 | 40/40 | 248 |",
        "",
        "### 7.2 逐模型 SR-first 最优",
        "",
        "下表先汇总每种方法的 8 个 SR-first winner。`SR > 0` 和 `SPL > 0`",
        "均采用严格正增益；`0.00` 视为持平而非提升。FeedTTA†、ATENA† 的数值",
        "属于 feedback-supervised TTA，不能与前三种无监督方法直接合并排名。",
        "",
        "| 方法 | SR > 0 | 平均 ΔSR | 中位 ΔSR | SPL > 0 | 平均 ΔSPL | winner 改变 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        method_rows = [winners[(method, model, benchmark)] for model, benchmark in SETTING_ORDER]
        sr = [number(row, "delta_sr_pp") for row in method_rows]
        spl = [number(row, "delta_spl_pp") for row in method_rows]
        changed = sum(row["scheduler_rank"] != "1" for row in method_rows)
        lines.append(
            f"| {METHOD_LABEL[method]} | {sum(value > 1e-9 for value in sr)}/8 | "
            f"{statistics.mean(sr):+.2f} | {statistics.median(sr):+.2f} | "
            f"{sum(value > 1e-9 for value in spl)}/8 | {statistics.mean(spl):+.2f} | "
            f"{changed}/8 |"
        )
    lines.append("")

    for method in METHOD_ORDER:
        lines.extend(
            [
                f"#### {METHOD_LABEL[method]}",
                "",
                "| Setting | ΔSR（主） | ΔSPL（次） | 核心参数 | 原 scheduler rank | 后段审计 |",
                "|---|---:|---:|---|---:|---|",
            ]
        )
        for model, benchmark in SETTING_ORDER:
            row = winners[(method, model, benchmark)]
            setting = f"{MODEL_LABEL[model]}-{BENCHMARK_LABEL[benchmark]}"
            lines.append(
                f"| {setting} | {signed_delta(row, 'sr')} | {signed_delta(row, 'spl')} | "
                f"`{row['parameters']}` | {row['scheduler_rank']} | `{row['audit']}` |"
            )
        lines.append("")
        if method == "tent":
            lines.append(
                "Tent 在 6/8 个 setting 上提高 SR；DUET-R2R 虽提高 SPL `+1.11`，"
                "SR 仍下降 `0.19`，ETPNav-R2R-CE 则同时下降并出现 severe collapse。"
            )
        elif method == "fstta":
            lines.append(
                "FSTTA 在 5/8 个 setting 上提高 SR。ETPNav-R2R-CE 的 SR-first 配置"
                "为 `ΔSR=+0.2571`、`ΔSPL=-0.0064`，直接展示了 SR 优先时必须同时"
                "披露的路径效率代价。"
            )
        elif method == "eam":
            lines.append(
                "EAM 有 7/8 个严格 SR 正增益，GOAT-R2R 持平；8/8 个 SPL 均为正。"
                "按 SR 重排后，HAMT-REVERIE 从 scheduler rank 4 升为第一，两个 CE"
                " 模型也都改用原 rank 2，说明旧 SPL/RGSPL 排名会掩盖更好的成功率。"
            )
        elif method == "feedtta":
            lines.append(
                "以标准 argmax Source 为基线后，FeedTTA† 只有 4/8 个 setting 提高 SR，"
                "只有 2/8 个提高 SPL，正收益主要集中在 REVERIE。DUET-R2R 为"
                " `ΔSR=-1.46, ΔSPL=-2.58`；ETPNav-R2R-CE 虽相对 sampled control "
                "有改善，但相对标准 Source 仍为 `-1.1568/-8.2484`。"
            )
        else:
            lines.append(
                "ATENA† 有 7/8 个严格 SR 正增益；GOAT-R2R 的最佳 finalist 仍为"
                " `ΔSR=-0.39, ΔSPL=-0.16`。HAMT-REVERIE 按 SR 重排后选择 `δ=0.2`"
                "（原 scheduler rank 2），而不是 RGSPL 驱动的 `δ=0.1`。"
            )
        lines.append("")

    lines.extend(
        [
            "### 7.3 Benchmark 级共享参数：按 SR 判定",
            "",
            "同一 benchmark 的共同 tuple 必须在所有纳入模型上完成 full-val。有效性",
            "先要求每个模型 `ΔSR > 0`，再最大化最弱/平均 `ΔSR`，最后比较最弱/平均",
            "`ΔSPL`。增益顺序：R2R/REVERIE 为 DUET、HAMT、GOAT；R2R-CE 为",
            "ETPNav、BEVBert。表中的无效行展示当前共同 finalist 中 SR-first 得分最高",
            "的 tuple，不代表它可用于正式主表。",
            "",
            "| 方法 | Benchmark | SR-first 最佳共同 tuple | 各模型 ΔSR（主） | 各模型 ΔSPL（次） | 判定 |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for method in METHOD_ORDER:
        for benchmark in ("r2r", "reverie", "r2r-ce"):
            parameters, rows, valid = shared_choice(groups, method, benchmark)
            sr = "/".join(signed_delta(row, "sr") for row in rows)
            spl = "/".join(signed_delta(row, "spl") for row in rows)
            if valid:
                status = "**SR 有效；待 parity/多顺序审计**"
            else:
                failed = ", ".join(
                    MODEL_LABEL[row["model"]]
                    for row in rows
                    if number(row, "delta_sr_pp") <= 1e-9
                )
                status = f"无有效共享配置（{failed} 未严格提高 SR）"
            lines.append(
                f"| {METHOD_LABEL[method]} | {BENCHMARK_LABEL[benchmark]} | "
                f"`{parameters}` | `{sr}` | `{spl}` | {status} |"
            )

    audit_counts = Counter(row["audit"] for row in winners.values())
    clean = audit_counts["no_coarse_flag"] + audit_counts["no_fixed_order_collapse_signal"]
    warnings = audit_counts["coarse_warning"] + audit_counts["warning_fixed_order_degradation"]
    severe = audit_counts["severe_fixed_order_collapse"]
    lines.extend(
        [
            "",
            "SR-first 且统一标准 Source 后，当前 15 个“方法 × benchmark”组合中只有",
            "ATENA†-REVERIE 存在全模型 SR 严格为正的共同 full-val tuple。FeedTTA†",
            "原先相对 sampled control 的 REVERIE 共享结论不再成立。**当前没有任何",
            "无监督方法形成可直接冻结的 SR 有效共享配置。** 旧口径下 Tent-R2R/",
            "Tent-REVERIE、EAM-REVERIE/",
            "EAM-R2R-CE、ATENA-R2R-CE 等基于 SPL/RGSPL 的“有效”结论由本节取代。",
            "例如 ATENA-R2R-CE 的共同 tuple 虽有 `ΔSPL=+0.8134/+2.2479`，ETPNav",
            "的 `ΔSR=-0.5141`，因此不能冻结。",
            "",
            "### 7.4 反馈预算、稳定性与解释边界",
            "",
            "- FeedTTA† 的 40 个 full-val finalist 共消费 44,440 个二值反馈；8 个",
            "  SR-first winner 共 8,888 个反馈（6,692 成功、2,196 失败）。整个搜索",
            "  共消费 120,232 个反馈。FeedTTA† 运行本身使用 sampled policy，但正式",
            "  结果统一相对标准 argmax Source；sampled no-update control 只进入诊断附表。",
            "- ATENA† 的 40 个 finalist 在 44,440 episodes 中查询 30,335 次真实反馈",
            "  （68.26%）。8 个 SR-first winner 查询 4,635/8,888（52.15%；3,000 成功、",
            "  1,635 失败），另使用 4,253 个 self labels。全搜索查询 63,398/91,560",
            "  （69.24%）。",
            f"- 40 个 SR-first winner 中，{clean} 个无后段 flag，{warnings} 个为 warning，",
            f"  {severe} 个为 severe collapse。唯一 severe 项仍是 Tent-ETPNav；因此",
            "  SR-first 排名不能替代 late-collapse 与多顺序检查。",
            "- 由于分阶段晋级最初不是按 SR 完成，当前“无共享配置”只对已完成的",
            "  full-val finalist 交集成立。下一轮 targeted 补测应在 256-prefix 阶段就按",
            "  SR 第一、SPL 第二筛选共同 tuple，再运行完整 `val_seen`。",
            "- 上述数值仍是调参证据。零更新适配器一致性审计通过前，不进入正式主表；",
            "  无监督方法与 FeedTTA†/ATENA† 继续分表。",
            "",
            "逐 setting 的全部 200 行、绝对 SR/SPL、REVERIE RGS/RGSPL、两个 rank、",
            "完整参数 JSON、run tag 和 audit 标签见",
            "[`VAL_SEEN_TOP5_BY_MODEL_BENCHMARK.md`](results/analysis/hparam_search/VAL_SEEN_TOP5_BY_MODEL_BENCHMARK.md)",
            "及配套 CSV。",
            "",
        ]
    )
    return "\n".join(lines)


def rewrite_report(groups: dict[tuple[str, str, str], list[dict[str, str]]]) -> None:
    document = REPORT_PATH.read_text(encoding="utf-8")
    start = document.index("## 7. ")
    end = document.index("## 8. ", start)
    replacement = build_report_section(groups)
    REPORT_PATH.write_text(
        document[:start] + replacement + "\n" + document[end:], encoding="utf-8"
    )


def main() -> None:
    rows, groups = load_and_rank()
    write_csv(rows)
    write_markdown(groups)
    rewrite_report(groups)
    changed = sum(items[0]["scheduler_rank"] != "1" for items in groups.values())
    print(
        f"wrote {len(rows)} rows across {len(groups)} groups; "
        f"SR-first winner differs from scheduler winner in {changed} groups"
    )


if __name__ == "__main__":
    main()
