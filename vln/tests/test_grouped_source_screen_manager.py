from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
MANAGER = REPO_ROOT / "vln" / "scripts" / "manage_grouped_source_screen.sh"


class GroupedSourceScreenManagerTest(unittest.TestCase):
    def test_help_lists_lifecycle_actions(self):
        completed = subprocess.run(
            ["bash", str(MANAGER), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for action in ("start", "status", "logs", "attach", "stop"):
            with self.subTest(action=action):
                self.assertIn(action, completed.stdout)

    def test_manager_uses_one_screen_around_the_grouped_runner(self):
        source = MANAGER.read_text(encoding="utf-8")
        self.assertIn("run_grouped_source_eval.sh", source)
        self.assertIn('screen -dmS "${SESSION}"', source)
        self.assertEqual(source.count("-dmS"), 1)
        self.assertIn('os.setsid(); os.execv', source)
        self.assertIn('tee -a "${SCREEN_LOG}"', source)
        self.assertIn("runner_starttime", source)
        self.assertIn("boot_id", source)
        self.assertIn('kill -TERM "${runner_pid}"', source)
        self.assertIn("sweep_runner_process_group", source)
        self.assertIn("jobs -pr", source)
        self.assertIn("STOP_REQUEST_DIR", source)
        self.assertIn("dead_screen_socket_exists", source)
        self.assertIn("do not kill screen directly", source)

    def test_control_logs_stay_inside_vln_results(self):
        source = MANAGER.read_text(encoding="utf-8")
        self.assertIn("vln/results/logs/grouped_source_screen", source)
        self.assertIn("GROUP_LOG_ROOT", source)


if __name__ == "__main__":
    unittest.main()
