from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "vln" / "scripts" / "run_four_gpu_source_eval.sh"


class FourGpuSourceRunnerTest(unittest.TestCase):
    def test_help_describes_validation_only_schedule(self):
        completed = subprocess.run(
            ["bash", str(RUNNER), "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("val_seen and val_unseen", completed.stdout)
        self.assertIn("never runs the hidden-label test split", completed.stdout)
        self.assertIn("--smoke", completed.stdout)
        self.assertIn("--gpus LIST", completed.stdout)
        self.assertIn("--only-queues LIST", completed.stdout)

    def test_invalid_gpu_and_queue_options_fail_before_server_access(self):
        cases = (
            (["--gpus", "0,1,2", "--dry-run"], "exactly four"),
            (["--gpus", "0,1,1,3", "--dry-run"], "must be distinct"),
            (["--gpus", "0,1,x,3", "--dry-run"], "non-negative integers"),
            (["--only-queue", "4", "--dry-run"], "invalid queue"),
            (["--only-queues", "1,4", "--dry-run"], "invalid queue"),
            (["--only-queues", "1,1", "--dry-run"], "duplicate queue"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    ["bash", str(RUNNER), *arguments],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr)

    def test_static_queue_and_split_contract(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('0) printf \'%s\\n\' etpnav-r2r-ce', source)
        self.assertIn('1) printf \'%s\\n\' bevbert-r2r-ce', source)
        self.assertIn('2) printf \'%s\\n\' streamvln-r2r-ce', source)
        for setting in (
            "duet-r2r", "duet-reverie", "hamt-r2r", "hamt-reverie",
            "goat-r2r", "goat-reverie",
        ):
            self.assertIn(setting, source)
        self.assertIn("SPLITS=(val_seen val_unseen)", source)
        self.assertNotIn("SPLITS=(val_seen val_unseen test)", source)
        self.assertIn('--smoke-episodes 2', source)

    def test_logging_cleanup_and_summary_contract(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn(
            'LOG_ROOT="${REPO_ROOT}/vln/results/source/${RUN_TAG}/_launcher"',
            source,
        )
        self.assertIn(
            'LOG_ROOT="${TMP_ROOT}/navtta-four-gpu-source/${RUN_TAG}"',
            source,
        )
        self.assertIn("plan.tsv", source)
        self.assertIn("SUMMARY.tsv", source)
        self.assertIn("metrics.csv", source)
        self.assertIn("paper_metrics.csv", source)
        self.assertIn("combined.log", source)
        self.assertIn("queue-${queue}.log", source)
        self.assertIn("queue-${queue}.exitcode", source)
        self.assertIn("metadata.tsv", source)
        self.assertIn("printf 'model_seed\\t0\\n'", source)
        self.assertIn("printf 'episode_order_seed\\t0\\n'", source)
        self.assertIn("setsid --fork --wait bash -c", source)
        self.assertIn('kill -TERM -- "-${pgid}"', source)
        self.assertIn('kill -KILL -- "-${pgid}"', source)
        waiter_clear = source.index(
            'CURRENT_SETTING_WAIT_PID=""\n'
            '    if ! wait_for_process_group_exit',
            source.index('if wait "${CURRENT_SETTING_WAIT_PID}"'),
        )
        group_check = source.index(
            'if ! wait_for_process_group_exit "${CURRENT_SETTING_PGID}"',
            source.index('if wait "${CURRENT_SETTING_WAIT_PID}"'),
        )
        self.assertLess(waiter_clear, group_check)

    def test_ce_version_is_only_forwarded_to_etpnav_and_bevbert(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("CE_DATA_VERSION=v1.2-native", source)
        self.assertIn("unset NAVTTA_CE_DATA_VERSION", source)
        self.assertIn("etpnav-r2r-ce|bevbert-r2r-ce", source)
        self.assertIn('command+=(--ce-data-version "${CE_DATA_VERSION}")', source)
        self.assertIn("streamvln-r2r-ce) protocol=v1.3-native", source)
        self.assertIn('--settings "${SUMMARY_SETTINGS_CSV}"', source)
        self.assertIn("verify_bevbert_source_pairing.py", source)
        self.assertIn('"${CE_DATA_VERSION}" == "v1.2-native"', source)


if __name__ == "__main__":
    unittest.main()
