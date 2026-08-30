from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "vln" / "scripts" / "run_hamt_e2e_source_eval.sh"
SOURCE_RUNNER = REPO_ROOT / "vln" / "scripts" / "run_source_eval.sh"


class HamtE2ESourceRunnerTest(unittest.TestCase):
    def test_help_identifies_both_benchmarks_and_e2e_assets(self):
        completed = subprocess.run(
            ["bash", str(RUNNER), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("R2R and REVERIE", completed.stdout)
        self.assertIn("vitbase-finetune-e2e", completed.stdout)

    def test_missing_option_values_are_rejected_before_path_guard(self):
        for option in ("--gpu", "--run-tag"):
            with self.subTest(option=option):
                completed = subprocess.run(
                    ["bash", str(RUNNER), option, "--dry-run"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("requires a value", completed.stderr)

    def test_runner_schedules_only_hamt_r2r_then_reverie(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("for setting in hamt-r2r hamt-reverie", source)
        self.assertIn('"${setting}" all "${GPU}"', source)
        self.assertNotIn("duet-r2r", source)
        self.assertNotIn("goat-r2r", source)

    def test_source_setting_uses_final_e2e_assets(self):
        source = SOURCE_RUNNER.read_text(encoding="utf-8")
        self.assertIn("--features vitbase_r2rfte2e", source)
        self.assertIn(
            "../datasets/R2R/trained_models/vitbase-finetune-e2e/ckpts/best_val_unseen",
            source,
        )
        self.assertIn(
            '"${CHECKPOINT_ROOT}/hamt/R2R/vitbase-finetune-e2e/best_val_unseen"',
            source,
        )
        self.assertIn('set_run_context "${R2R_DUET_HAMT_MANIFEST}"', source)

    def test_runner_uses_the_server_checkout_directly(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("REPO_ROOT=/data1/wxy/code/NavTTA", source)
        self.assertNotIn("runtime_paths.sh", source)


if __name__ == "__main__":
    unittest.main()
