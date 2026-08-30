import importlib.util
import json
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln" / "scripts" / "summarize_source_eval.py"
SPEC = importlib.util.spec_from_file_location("summarize_source_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SummarizeSourceEvalTest(unittest.TestCase):
    def test_default_reference_is_excel_source_snapshot(self):
        self.assertEqual(
            MODULE.DEFAULT_REFERENCE.name,
            "excel_source_metrics.json",
        )
        document = json.loads(MODULE.DEFAULT_REFERENCE.read_text(encoding="utf-8"))
        self.assertEqual(document["kind"], "excel_source_reference_metrics")
        records = {
            (item["baseline"], item["benchmark"]["name"]): item
            for item in document["records"]
        }
        self.assertEqual(
            records[("HAMT", "R2R")]["metrics"]["val_unseen"]["SPL"],
            62,
        )

    def _fixture(self, root, tag):
        references = []
        for setting in MODULE.SETTINGS:
            baseline, benchmark = MODULE.REFERENCE_IDENTITY[setting]
            reference_metrics = {"SR": 50, "SPL": 40}
            if setting.endswith("-reverie"):
                reference_metrics["RGSPL"] = 30
            references.append({
                "id": setting + "-reference",
                "baseline": baseline,
                "benchmark": {
                    "name": benchmark,
                    "protocol": (
                        "v1.3-native" if baseline == "StreamVLN"
                        else "v1.2-native" if benchmark == "R2R-CE"
                        else "discrete"
                    ),
                },
                "metrics": {
                    "val_seen": dict(reference_metrics),
                    "val_unseen": dict(reference_metrics),
                },
            })
            for split_index, split in enumerate(MODULE.SPLITS):
                directory = root / f"{tag}-{setting}" / setting / split
                directory.mkdir(parents=True)
                count = MODULE.EXPECTED_EPISODES[setting][split_index]
                if setting in {"etpnav-r2r-ce", "bevbert-r2r-ce"}:
                    metrics = directory / "metrics" / f"source_{split}"
                    metrics.mkdir(parents=True)
                    (metrics / f"stats_ckpt_1_{split}.json").write_text(
                        json.dumps({"success": 0.5, "spl": 0.4}),
                        encoding="utf-8",
                    )
                    text = (
                        "R2R_VLNCE_v1-2_preprocessed_BERTidx\n"
                        f"Episodes evaluated: {count}.0\n"
                    )
                elif setting == "streamvln-r2r-ce":
                    (directory / "result.json").write_text(
                        json.dumps({
                            "sucs_all": 0.5,
                            "spls_all": 0.4,
                            "oss_all": 0.6,
                            "ones_all": 4.0,
                            "length": count,
                        }) + "\n",
                        encoding="utf-8",
                    )
                    text = "streamvln complete\n"
                else:
                    grounding = ", rgspl: 30.00" if setting.endswith(
                        "-reverie"
                    ) else ""
                    text = (
                        f"eval {count} predictions\n"
                        f"Env name: {split}, sr: 50.00, spl: 40.00"
                        f"{grounding}\n"
                    )
                (directory / "console.log").write_text(text, encoding="utf-8")
        return {"records": references}

    def test_all_settings_and_splits_are_summarized(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source_root = temporary / "source"
            tag = "unit-source"
            reference = temporary / "references.json"
            reference.write_text(
                json.dumps(self._fixture(source_root, tag)), encoding="utf-8"
            )
            output = temporary / "metrics.csv"
            summaries = MODULE.summarize(
                source_root, tag, output, "v1.2-native", reference
            )
            self.assertEqual(len(summaries), 18)
            self.assertTrue(output.is_file())
            for item in summaries:
                expected = (
                    "MATCH_EXACT"
                    if item["setting"].endswith("-reverie")
                    else "MATCH_INTEGER"
                )
                self.assertEqual(item["comparison"], expected, item)

    def test_selected_settings_allow_an_affected_model_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source_root = temporary / "source"
            tag = "unit-source"
            reference = temporary / "references.json"
            reference.write_text(
                json.dumps(self._fixture(source_root, tag)), encoding="utf-8"
            )
            output = temporary / "metrics.csv"
            selected = ("bevbert-r2r-ce", "streamvln-r2r-ce")
            summaries = MODULE.summarize(
                source_root,
                tag,
                output,
                "v1.2-native",
                reference,
                settings=selected,
            )
            self.assertEqual(len(summaries), 4)
            self.assertEqual(
                {item["setting"] for item in summaries}, set(selected)
            )

    def test_invalid_selected_settings_are_rejected(self):
        with self.assertRaisesRegex(MODULE.SummaryError, "unknown settings"):
            MODULE.validate_selected_settings(("bevbert-r2r-ce", "missing"))
        with self.assertRaisesRegex(MODULE.SummaryError, "duplicates"):
            MODULE.validate_selected_settings(
                ("bevbert-r2r-ce", "bevbert-r2r-ce")
            )

    def test_incomplete_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source_root = temporary / "source"
            tag = "unit-source"
            reference = temporary / "references.json"
            reference.write_text(
                json.dumps(self._fixture(source_root, tag)), encoding="utf-8"
            )
            path = (
                source_root / f"{tag}-duet-r2r" / "duet-r2r"
                / "val_seen" / "console.log"
            )
            path.write_text(
                "eval 2 predictions\n"
                "Env name: val_seen, sr: 50.00, spl: 40.00\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MODULE.SummaryError, "incomplete"):
                MODULE.summarize(
                    source_root,
                    tag,
                    temporary / "metrics.csv",
                    "v1.2-native",
                    reference,
                )

    def test_unified_ce_is_not_compared_to_native_paper_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source_root = temporary / "source"
            tag = "unit-source"
            reference = temporary / "references.json"
            reference.write_text(
                json.dumps(self._fixture(source_root, tag)), encoding="utf-8"
            )
            for setting in ("etpnav-r2r-ce", "bevbert-r2r-ce"):
                for split in MODULE.SPLITS:
                    path = (
                        source_root / f"{tag}-{setting}" / setting / split
                        / "console.log"
                    )
                    text = path.read_text(encoding="utf-8").replace(
                        "R2R_VLNCE_v1-2", "R2R_VLNCE_v1-3"
                    )
                    path.write_text(text, encoding="utf-8")
            summaries = MODULE.summarize(
                source_root,
                tag,
                temporary / "metrics.csv",
                "v1.3-unified",
                reference,
            )
            ce = [
                item for item in summaries
                if item["setting"] in {"etpnav-r2r-ce", "bevbert-r2r-ce"}
            ]
            self.assertTrue(all(
                item["comparison"] == "NOT_COMPARABLE_PROTOCOL"
                for item in ce
            ))

    def test_missing_required_metrics_are_rejected(self):
        with self.assertRaisesRegex(MODULE.SummaryError, "missing required"):
            MODULE.validate_metrics("duet-r2r", "val_seen", {"TL": Decimal("1")})

    def test_nonfinite_metrics_are_rejected(self):
        with self.assertRaisesRegex(MODULE.SummaryError, "non-finite"):
            MODULE.validate_metrics(
                "etpnav-r2r-ce",
                "val_seen",
                {"SR": Decimal("NaN"), "SPL": Decimal("40")},
            )

    def test_missing_reference_comparison_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source_root = temporary / "source"
            tag = "unit-source"
            document = self._fixture(source_root, tag)
            for item in document["records"]:
                if item["baseline"] == "DUET" and item["benchmark"]["name"] == "R2R":
                    item["metrics"]["val_seen"]["NE"] = 2.0
                    break
            reference = temporary / "references.json"
            reference.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                MODULE.SummaryError, "missing reference comparison metrics"
            ):
                MODULE.summarize(
                    source_root,
                    tag,
                    temporary / "metrics.csv",
                    "v1.2-native",
                    reference,
                )

    def test_r2r_and_r2r_ce_sr_spl_compare_as_rounded_integers(self):
        status, value, reference = MODULE.comparison_status(
            Decimal("71.52"), Decimal("72"), integer=True
        )
        self.assertEqual((status, value, reference), (
            "MATCH_INTEGER", Decimal("72"), Decimal("72")
        ))
        status, value, reference = MODULE.comparison_status(
            Decimal("67.58"), Decimal("67.34"), integer=True
        )
        self.assertEqual((status, value, reference), (
            "MISMATCH_INTEGER", Decimal("68"), Decimal("67")
        ))
        status, value, reference = MODULE.comparison_status(
            Decimal("56.5"), Decimal("57"), integer=True
        )
        self.assertEqual((status, value, reference), (
            "MATCH_INTEGER", Decimal("57"), Decimal("57")
        ))

    def test_only_sr_and_spl_decide_r2r_parity(self):
        self.assertTrue(MODULE.used_for_overall("duet-r2r", "SR"))
        self.assertTrue(MODULE.used_for_overall("etpnav-r2r-ce", "SPL"))
        self.assertFalse(MODULE.used_for_overall("duet-r2r", "NE"))
        self.assertFalse(MODULE.used_for_overall("bevbert-r2r-ce", "OSR"))


if __name__ == "__main__":
    unittest.main()
