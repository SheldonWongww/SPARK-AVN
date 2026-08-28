import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def _runner():
    path = ROOT / "avn/scripts/run_smt_audio_val_search.py"
    spec = importlib.util.spec_from_file_location("smt_audio_val_search", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SmtAudioValSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _runner()
        cls.spec = cls.runner.load_spec(cls.runner.DEFAULT_SPEC)

    def _args(self, **updates):
        values = {
            "batch_id": "unit-test",
            "gpus": ("0", "1", "2", "3"),
            "smoke": False,
            "smoke_setting": "single_source",
            "eam_concurrency": 4,
            "feedtta_concurrency": 4,
            "atena_concurrency": 3,
            "idea_concurrency": 2,
        }
        values.update(updates)
        return SimpleNamespace(**values)

    def test_frozen_plan_has_118_tta_jobs_and_no_source_jobs(self):
        jobs = self.runner.build_jobs(self.spec, self._args())
        self.assertEqual(len(jobs), 118)
        expected = {"eam": 24, "feedtta": 30, "atena": 40, "idea": 24}
        self.assertEqual(
            {method: sum(job.method == method for job in jobs)
             for method in self.runner.METHODS},
            expected,
        )
        self.assertNotIn("source", {job.method for job in jobs})
        for method in self.runner.METHODS:
            settings = [job.source_setting for job in jobs if job.method == method]
            first_multi = settings.index("multi_source")
            self.assertNotIn("single_source", settings[first_multi:])

    def test_all_method_overrides_use_task_native_sampling(self):
        jobs = self.runner.build_jobs(self.spec, self._args())
        asset = {
            "statistics_path": "/tmp/source.json",
            "statistics_sha256": "a" * 64,
            "manifest_path": "/tmp/manifest.json",
            "manifest_sha256": "b" * 64,
        }
        for method in self.runner.METHODS:
            job = next(item for item in jobs if item.method == method)
            values = self.runner.expected_overrides(
                job,
                2000,
                asset if method == "idea" else None,
            )
            overrides = dict(zip(values[0::2], values[1::2]))
            self.assertEqual(overrides["EVAL.ACTION_SELECTION"], "sample")
            if method == "atena":
                self.assertEqual(
                    overrides["TTA.ATENA.ACTION_SELECTION_PROTOCOL"],
                    "sample_from_policy",
                )

    def test_grid_sizes_and_no_result_dependent_expansion(self):
        expected = {"eam": 12, "feedtta": 15, "atena": 20, "idea": 12}
        for method, count in expected.items():
            self.assertEqual(len(self.runner.method_points(self.spec, method)), count)
        self.assertFalse(self.spec["freeze"]["grid_expansion_after_results"])

    def test_concurrency_can_only_be_reduced(self):
        self.runner.validate_runtime_limits(self.spec, self._args())
        with self.assertRaisesRegex(self.runner.UserError, "exceeds frozen cap"):
            self.runner.validate_runtime_limits(
                self.spec, self._args(atena_concurrency=4)
            )

    def test_smoke_selects_one_job_per_method_on_one_setting(self):
        jobs = self.runner.build_jobs(
            self.spec,
            self._args(smoke=True, smoke_setting="multi_source"),
        )
        self.assertEqual(len(jobs), 4)
        self.assertEqual({job.method for job in jobs}, set(self.runner.METHODS))
        self.assertEqual(
            {job.source_setting for job in jobs}, {"multi_source"}
        )
        for method in self.runner.METHODS:
            self.assertTrue(self.runner.phase_complete(
                jobs, set(), method, "single_source"
            ))
            self.assertFalse(self.runner.phase_complete(
                jobs, set(), method, "multi_source"
            ))

    def test_spec_is_valid_json_and_uses_only_sample(self):
        payload = json.loads(self.runner.DEFAULT_SPEC.read_text(encoding="utf-8"))
        self.assertEqual(payload["protocol"]["action_selection"], "sample")
        self.assertEqual(
            payload["protocol"]["method_rng"]["atena_self_head"],
            "forked_torch_rng_state_restored_before_next_action",
        )
        self.assertFalse(payload["source_control"]["rerun_in_campaign"])
        self.assertEqual(
            payload["protocol"]["methods"]["feedtta"]["action_port"],
            "sample_from_policy_and_optimize_executed_action",
        )
        self.assertEqual(
            payload["protocol"]["methods"]["atena"]["action_port"],
            "sample_from_policy_and_use_executed_action_as_pseudo_expert",
        )

    def test_status_rejects_unknown_batch(self):
        original = self.runner.LOG_ROOT
        try:
            with tempfile.TemporaryDirectory() as directory:
                self.runner.LOG_ROOT = Path(directory)
                with self.assertRaisesRegex(self.runner.UserError, "does not exist"):
                    self.runner.print_status("missing")
        finally:
            self.runner.LOG_ROOT = original


if __name__ == "__main__":
    unittest.main()
