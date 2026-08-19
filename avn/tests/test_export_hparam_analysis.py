import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "avn" / "scripts" / "export_hparam_analysis.py"
SPEC = importlib.util.spec_from_file_location("avn_hparam_export", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ExportHparamAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.csv"
        write_csv(
            self.source,
            ["model", "source_setting", "exitcode", "success", "spl"],
            [
                {
                    "model": "smt_audio", "source_setting": "single_source",
                    "exitcode": "0", "success": "0.50", "spl": "0.25",
                },
                {
                    "model": "enmus", "source_setting": "single_source",
                    "exitcode": "0", "success": "0.65", "spl": "0.35",
                },
            ],
        )
        self.metrics = self.root / "metrics.csv"
        fields = [
            "run_tag", "status", "validation", "model", "source_setting",
            "result_role", "method", "lr", "update_interval", "success",
            "spl", "manifest",
        ]
        write_csv(
            self.metrics,
            fields,
            [
                {
                    "run_tag": "tent-low", "status": "0", "validation": "ok",
                    "model": "smt_audio", "source_setting": "single_source",
                    "result_role": "hyperparameter_search", "method": "tent",
                    "lr": "1e-6", "update_interval": "2", "success": "0.51",
                    "spl": "0.26", "manifest": "/private/raw/manifest.json",
                },
                {
                    "run_tag": "tent-best", "status": "0", "validation": "ok",
                    "model": "smt_audio", "source_setting": "single_source",
                    "result_role": "hyperparameter_search", "method": "tent",
                    "lr": "3e-6", "update_interval": "4", "success": "0.53",
                    "spl": "0.27", "manifest": "/private/raw/manifest.json",
                },
                {
                    "run_tag": "feedtta-provisional", "status": "90",
                    "validation": "failed", "model": "enmus",
                    "source_setting": "single_source",
                    "result_role": "hyperparameter_search", "method": "feedtta",
                    "lr": "3e-8", "update_interval": "", "success": "0.66",
                    "spl": "0.36", "manifest": "/private/raw/manifest.json",
                },
            ],
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_exports_one_deterministic_report_per_model_method(self):
        output = self.root / "published"
        written = MODULE.export_reports(
            [self.metrics], self.source, output, top_k=5,
            include_nonzero_status=True,
        )
        self.assertEqual(
            [path.relative_to(output).as_posix() for path in written],
            ["enmus/feedtta.md", "smt_audio/tent.md"],
        )
        tent = (output / "smt_audio" / "tent.md").read_text(encoding="utf-8")
        self.assertIn("SMT+Audio × Tent", tent)
        self.assertLess(tent.index("tent-best"), tent.index("tent-low"))
        self.assertIn("+3.00", tent)
        self.assertNotIn("/private/raw", tent)
        before = tent.encode("utf-8")
        MODULE.export_reports(
            [self.metrics], self.source, output, top_k=5,
            include_nonzero_status=True,
        )
        self.assertEqual(
            before, (output / "smt_audio" / "tent.md").read_bytes()
        )

    def test_keeps_completed_rows_with_provisional_validation_label(self):
        output = self.root / "published"
        MODULE.export_reports(
            [self.metrics], self.source, output, include_nonzero_status=True
        )
        report = (output / "enmus" / "feedtta.md").read_text(encoding="utf-8")
        self.assertIn("Validation labels: failed=1", report)
        self.assertIn("Runner statuses: 90=1", report)
        self.assertIn("binary episode feedback", report)

    def test_excludes_nonzero_status_by_default(self):
        output = self.root / "published"
        written = MODULE.export_reports([self.metrics], self.source, output)
        self.assertEqual(
            [path.relative_to(output).as_posix() for path in written],
            ["smt_audio/tent.md"],
        )

    def test_rejects_duplicate_run_tags_across_inputs(self):
        duplicate = self.root / "duplicate.csv"
        duplicate.write_bytes(self.metrics.read_bytes())
        with self.assertRaisesRegex(MODULE.ExportError, "duplicate run_tag"):
            MODULE.export_reports(
                [self.metrics, duplicate], self.source, self.root / "published"
            )

    def test_cli_publish_root_must_be_inside_tracked_analysis_tree(self):
        with self.assertRaisesRegex(MODULE.ExportError, "must stay under"):
            MODULE.require_publish_root(self.root / "outside")


if __name__ == "__main__":
    unittest.main()
