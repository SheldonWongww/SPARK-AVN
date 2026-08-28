import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "vln/scripts/run_r2r_ce_concurrency_calibration.py"
SPEC = importlib.util.spec_from_file_location(
    "run_r2r_ce_concurrency_calibration", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class R2RCEConcurrencyCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.spec_path = (
            ROOT / "vln/experiments/r2r_ce_concurrency_calibration_v1.json"
        )
        self.spec, self.search_path, self.search = MODULE.load_spec(
            self.spec_path
        )

    def test_reviewed_targets_and_limits_are_locked(self):
        self.assertEqual(
            self.spec["target_concurrency"]["etpnav-r2r-ce"],
            {"tent": 7, "fstta": 7, "eam": 5, "feedtta": 5, "atena": 4},
        )
        self.assertEqual(
            self.spec["target_concurrency"]["bevbert-r2r-ce"],
            {"tent": 6, "fstta": 6, "eam": 4, "feedtta": 4, "atena": 3},
        )
        self.assertEqual(
            self.spec["resource_limits"][
                "production_gpu_memory_mib_exclusive"
            ],
            29000,
        )
        self.assertEqual(self.spec["protocol"]["episode_count"], 100)

    def test_jobs_use_nonformal_smoke_prefix_and_unique_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = MODULE.prepare_jobs(
                "calib-test", directory, self.search,
                "etpnav-r2r-ce", "tent", 3, 1, 0, 100, write=False,
            )
        self.assertEqual(len(jobs), 3)
        self.assertEqual(len({job["run_tag"] for job in jobs}), 3)
        for index, job in enumerate(jobs, start=1):
            command = job["command"]
            self.assertIn("--smoke-episodes", command)
            self.assertEqual(command[command.index("--smoke-episodes") + 1], "100")
            self.assertNotIn("--order-seed", command)
            self.assertNotIn("--result-root", command)
            self.assertEqual(job["config"]["stage"], "resource_calibration")
            self.assertEqual(job["config"]["episodes"], 100)
            self.assertEqual(job["worker_index"], index)

    def test_feedtta_workers_have_sgr_but_no_action_sampling_seed(self):
        first = MODULE.paper_anchor_parameters(self.search, "feedtta", 1)
        second = MODULE.paper_anchor_parameters(self.search, "feedtta", 2)
        self.assertEqual(first["action_selection"], "argmax")
        self.assertEqual(first["sgr_seed"], 0)
        self.assertEqual(second["sgr_seed"], 1)
        self.assertNotIn("action_seed", first)

    def test_diagnostics_require_real_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tta_diagnostics.json"
            job = {"diagnostics_path": str(path)}
            valid = {
                "method": "tent",
                "baseline": "etpnav",
                "episode_count": 100,
                "action_selection": "target_native_argmax",
                "feedback_supervision": "none",
                "adapter": {"episodes": 100, "updates": 10},
            }
            path.write_text(json.dumps(valid), encoding="utf-8")
            result = MODULE.validate_diagnostics(
                job, "etpnav-r2r-ce", "tent", 100
            )
            self.assertTrue(result["valid"])
            valid["adapter"]["updates"] = 0
            path.write_text(json.dumps(valid), encoding="utf-8")
            result = MODULE.validate_diagnostics(
                job, "etpnav-r2r-ce", "tent", 100
            )
            self.assertFalse(result["valid"])
            self.assertIn("no positive method updates", result["errors"])

    def test_atena_requires_exact_replay_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tta_diagnostics.json"
            job = {"diagnostics_path": str(path)}
            document = {
                "method": "atena",
                "baseline": "bevbert",
                "episode_count": 100,
                "action_selection": "target_native_argmax",
                "feedback_supervision": "binary_episode_success",
                "atena_exact_replay_within_declared_scope": True,
                "atena_optimizer_scope_matches_reachable": True,
                "adapter": {"episodes": 100, "updates": 5},
            }
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertTrue(MODULE.validate_diagnostics(
                job, "bevbert-r2r-ce", "atena", 100
            )["valid"])
            document["atena_exact_replay_within_declared_scope"] = False
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertFalse(MODULE.validate_diagnostics(
                job, "bevbert-r2r-ce", "atena", 100
            )["valid"])


if __name__ == "__main__":
    unittest.main()
