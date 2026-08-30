from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "vln" / "scripts" / "run_grouped_source_eval.sh"
SOURCE_RUNNER = REPO_ROOT / "vln" / "scripts" / "run_source_eval.sh"


class GroupedSourceRunnerTest(unittest.TestCase):
    def test_help_describes_resource_groups(self):
        completed = subprocess.run(
            ["bash", str(RUNNER), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("DUET, HAMT, GOAT", completed.stdout)
        self.assertIn("ETPNav, BEVBert", completed.stdout)
        self.assertIn("StreamVLN", completed.stdout)
        self.assertIn("--skip-runtime-check", completed.stdout)
        self.assertNotIn("--skip-preflight", completed.stdout)

    def test_options_do_not_consume_the_next_option_as_a_value(self):
        for option in (
            "--gpu",
            "--run-tag",
            "--split",
            "--ce-data-version",
            "--only-group",
        ):
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

    def test_only_group_is_validated_and_documented(self):
        help_result = subprocess.run(
            ["bash", str(RUNNER), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        self.assertIn("--only-group INDEX", help_result.stdout)

        invalid = subprocess.run(
            ["bash", str(RUNNER), "--only-group", "4", "--dry-run"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("invalid resource group", invalid.stderr)

    def test_all_settings_are_scheduled_in_requested_workers(self):
        source = RUNNER.read_text(encoding="utf-8")
        worker_invocations = (
            "run_worker duet-r2r duet-reverie",
            "run_worker hamt-r2r hamt-reverie",
            "run_worker goat-r2r goat-reverie",
            "run_worker etpnav-r2r-ce",
            "run_worker bevbert-r2r-ce",
        )
        for invocation in worker_invocations:
            with self.subTest(invocation=invocation):
                self.assertEqual(source.count(invocation), 1)
        self.assertEqual(source.count("run_setting streamvln-r2r-ce"), 1)

        group_1 = source.index("[group 1/3]")
        group_1_wait = source.index('wait_group "group 1')
        group_2 = source.index("[group 2/3]")
        group_2_wait = source.index('wait_group "group 2')
        group_3 = source.index("[group 3/3]")
        self.assertLess(group_1, group_1_wait)
        self.assertLess(group_1_wait, group_2)
        self.assertLess(group_2, group_2_wait)
        self.assertLess(group_2_wait, group_3)

    def test_ce_version_and_failure_barrier_are_explicit(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('command+=(--ce-data-version "${CE_DATA_VERSION}")', source)
        self.assertIn("later groups were not started", source)
        self.assertIn('command+=(--dry-run)', source)
        self.assertIn("source result tag already exists", source)
        self.assertNotIn("run manifests already exist for tag", source)
        self.assertNotIn("verify_preflight.py", source)
        self.assertIn("verify_runtime_imports.sh", source)
        self.assertIn("claim_source_tag_lock", source)
        self.assertIn("flock -n", source)

    def test_server_paths_are_explicit(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("REPO_ROOT=/data1/wxy/code/NavTTA", source)
        self.assertIn("VLN_ROOT=/data1/wxy/exp_data/NavTTA/vln", source)
        self.assertIn('LOG_ROOT="${TMP_ROOT}/navtta-grouped-source/', source)
        self.assertNotIn("runtime_paths.sh", source)
        self.assertNotIn("/root/autodl-tmp", source)

    def test_group_and_child_share_the_same_tag_lock(self):
        grouped_source = RUNNER.read_text(encoding="utf-8")
        child_source = SOURCE_RUNNER.read_text(encoding="utf-8")
        self.assertIn('export NAVTTA_SOURCE_TAG_LOCKED="${RUN_TAG}"', grouped_source)
        self.assertIn("claim_or_verify_source_tag_lock", child_source)
        self.assertIn('NAVTTA_SOURCE_TAG_LOCKED:-', child_source)
        self.assertIn('readlink -f "/proc/$$/fd/${inherited_fd}"', child_source)

    def test_interrupt_cleanup_targets_child_process_groups(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("setsid --fork --wait bash -c", source)
        self.assertIn("trap 'parent_cleanup 130' INT", source)
        self.assertIn("trap 'parent_cleanup 143' TERM", source)
        self.assertIn('kill -TERM -- "-${pgid}"', source)
        self.assertIn('wait_for_process_group_exit "${CURRENT_SETTING_PGID}"', source)
        self.assertIn('kill -KILL "${CURRENT_SETTING_CHILD_PID}"', source)


if __name__ == "__main__":
    unittest.main()
