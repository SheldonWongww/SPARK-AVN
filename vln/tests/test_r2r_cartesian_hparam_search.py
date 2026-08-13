from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/run_r2r_cartesian_hparam_search.py"
MODULE_SPEC = importlib.util.spec_from_file_location("r2r_cartesian", SCRIPT)
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)

TRANSLATOR_SCRIPT = REPO_ROOT / "vln/scripts/tta_config_cli.py"
TRANSLATOR_SPEC = importlib.util.spec_from_file_location(
    "r2r_cartesian_tta_config_cli", TRANSLATOR_SCRIPT
)
TRANSLATOR = importlib.util.module_from_spec(TRANSLATOR_SPEC)
TRANSLATOR_SPEC.loader.exec_module(TRANSLATOR)


class R2RCartesianSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = MODULE.load_spec()

    def test_exact_cartesian_counts(self):
        expected = {
            "tent": 40,
            "fstta": 81,
            "eam": 320,
            "feedtta": 250,
            "atena": 100,
        }
        self.assertEqual(self.spec["settings"], list(MODULE.SETTINGS))
        self.assertEqual(self.spec["episode_count"], 1021)
        for method, count in expected.items():
            with self.subTest(method=method):
                points = MODULE.expand_parameters(method, self.spec)
                self.assertEqual(len(points), count)
                self.assertEqual(
                    len({MODULE.canonical(point) for point in points}), count
                )
        self.assertEqual(
            sum(
                len(MODULE.expand_parameters(method, self.spec))
                * len(MODULE.SETTINGS)
                for method in MODULE.TTA_METHODS
            ),
            2373,
        )

    def test_special_axes_expand_to_runner_parameters(self):
        eam = MODULE.expand_parameters("eam", self.spec)[0]
        self.assertNotIn("memory_batch", eam)
        self.assertEqual((eam["memory_size"], eam["batch_size"]), (16, 4))

        feedtta = MODULE.expand_parameters("feedtta", self.spec)[0]
        self.assertNotIn("sgr_profile", feedtta)
        self.assertEqual((feedtta["p"], feedtta["alpha"]), (0.0, -0.2))
        self.assertEqual(feedtta["action_selection"], "sample")
        self.assertEqual(feedtta["action_seed"], 0)

        atena = MODULE.expand_parameters("atena", self.spec)[0]
        self.assertNotIn("lr_pair", atena)
        self.assertEqual((atena["lr_query"], atena["lr_self"]), (4e-7, 5e-8))
        self.assertEqual(atena["action_selection"], "argmax")

    def test_jobs_are_full_val_round_robin_and_benchmark_first(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(mock.patch.object(MODULE, "LOG_ROOT", root / "logs"))
            stack.enter_context(mock.patch.object(MODULE, "TUNING_ROOT", root / "tuning"))
            jobs = MODULE.build_jobs("tent", "batch-a", self.spec, gpu=2)
        self.assertEqual(len(jobs), 120)
        self.assertEqual([job["setting"] for job in jobs[:3]], list(MODULE.SETTINGS))
        self.assertEqual([job["point_index"] for job in jobs[:3]], [0, 0, 0])
        for job in jobs:
            self.assertEqual(job["episodes"], -1)
            self.assertEqual(job["stage"], "cartesian")
            self.assertNotIn("--episode-limit", job["command"])
            self.assertNotIn("--order-seed", job["command"])
            parts = Path(job["result_root"]).parts
            self.assertIn("batch-a", parts)
            self.assertIn(job["model"], parts)
            self.assertIn("tent", parts)
            self.assertEqual(Path(job["result_root"]).name, "val_seen")

    def test_controls_are_argmax_and_sampled_diagnostic(self):
        source = MODULE.build_jobs("source", "controls", self.spec)
        sampled = MODULE.build_jobs("feedtta_control", "controls", self.spec)
        self.assertEqual(len(source), 3)
        self.assertEqual(len(sampled), 3)
        self.assertTrue(all(job["config_method"] == "source" for job in source + sampled))
        self.assertTrue(
            all(job["parameters"]["action_selection"] == "argmax" for job in source)
        )
        self.assertTrue(
            all(job["parameters"]["action_selection"] == "sample" for job in sampled)
        )

    def test_generated_runtime_configs_translate_without_stage_or_order_seed(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(mock.patch.object(MODULE, "LOG_ROOT", root / "logs"))
            stack.enter_context(mock.patch.object(MODULE, "TUNING_ROOT", root / "tuning"))
            for method in MODULE.METHODS:
                with self.subTest(method=method):
                    job = MODULE.build_jobs(method, "translation", self.spec)[0]
                    MODULE.staged._write_job(job)
                    config = json.loads(
                        Path(job["config_path"]).read_text(encoding="utf-8")
                    )
                    self.assertNotIn("stage", config)
                    self.assertNotIn("order_seed", config)
                    translated_method, tokens = TRANSLATOR.translate(
                        job["setting"], job["config_path"],
                        str(root / "tta_diagnostics.json"),
                    )
                    self.assertEqual(translated_method, job["config_method"])
                    self.assertTrue(tokens)

    def test_path_components_reject_traversal(self):
        for value in ("../escape", "a/b", "", ".", ".."):
            with self.subTest(value=value), self.assertRaises(MODULE.UserError):
                MODULE.safe_component("batch", value)

    def test_plan_is_commit_and_spec_pinned_on_resume(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(mock.patch.object(MODULE, "LOG_ROOT", root / "logs"))
            stack.enter_context(mock.patch.object(MODULE, "TUNING_ROOT", root / "tuning"))
            method_root, jobs = MODULE.ensure_plan(
                "source", "resume-test", self.spec, gpu=0, resume=False
            )
            self.assertEqual(len(jobs), 3)
            with self.assertRaisesRegex(MODULE.UserError, "use --resume"):
                MODULE.ensure_plan(
                    "source", "resume-test", self.spec, gpu=0, resume=False
                )
            _, resumed = MODULE.ensure_plan(
                "source", "resume-test", self.spec, gpu=0, resume=True
            )
            self.assertEqual(len(resumed), 3)
            manifest_path = method_root / "GRID.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["spec_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.UserError, "spec_sha256"):
                MODULE.ensure_plan(
                    "source", "resume-test", self.spec, gpu=0, resume=True
                )

    def test_retry_preserves_cartesian_result_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir = root / "job"
            result_root = root / "old-result" / "val_seen"
            retry_parent = root / "new-results"
            job_dir.mkdir(parents=True)
            result_root.mkdir(parents=True)
            (job_dir / "console.log").write_text("failed", encoding="utf-8")
            (job_dir / "exitcode").write_text("1\n", encoding="utf-8")
            config_path = job_dir / "parameters.json"
            base = "retry-base"
            job = {
                "batch_id": "batch",
                "ordinal": 0,
                "base_run_tag": base,
                "run_tag": base,
                "attempt": 0,
                "setting": "duet-r2r",
                "model": "duet",
                "family": "discrete",
                "benchmark": "r2r",
                "search_method": "tent",
                "config_method": "tent",
                "result_layout": MODULE.RESULT_LAYOUT,
                "result_namespace": "tent",
                "stage": "cartesian",
                "episodes": -1,
                "order_seed": None,
                "parameters": {"lr": 1e-5},
                "parent_run_tags": [],
                "config_path": str(config_path),
                "job_dir": str(job_dir),
                "result_root": str(result_root),
                "retry_result_root_parent": str(retry_parent),
                "command": [
                    "runner", "duet-r2r", "val_seen", "0",
                    "--run-tag", base,
                    "--tta-config", str(config_path),
                    "--result-root", str(result_root),
                ],
            }
            MODULE.staged._bump_attempt(job)
            self.assertEqual(job["run_tag"], base + "-retry1")
            self.assertEqual(
                Path(job["result_root"]),
                retry_parent / (base + "-retry1") / "val_seen",
            )
            self.assertTrue((job_dir / "attempts/attempt-00/result_root").is_dir())
            persisted = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["run_tag"], base + "-retry1")

    def test_sr_then_spl_then_stability_score(self):
        def value(sr, spl, drift, updates):
            return {
                "metrics": {"SR": sr, "SPL": spl},
                "adapter_diagnostics": {
                    "relative_param_drift": drift,
                    "updates": updates,
                },
                "parameters": {"lr": drift},
            }

        candidates = [
            value(80.0, 72.0, 0.1, 10),
            value(80.1, 70.0, 0.2, 20),
            value(80.1, 71.0, 0.3, 30),
            value(80.1, 71.0, 0.1, 40),
        ]
        winner = max(candidates, key=MODULE.result_score)
        self.assertEqual(winner["metrics"], {"SR": 80.1, "SPL": 71.0})
        self.assertEqual(winner["adapter_diagnostics"]["relative_param_drift"], 0.1)


if __name__ == "__main__":
    unittest.main()
