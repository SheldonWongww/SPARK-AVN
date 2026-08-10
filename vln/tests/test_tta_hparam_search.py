import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/run_tta_hparam_search.py"
MODULE_SPEC = importlib.util.spec_from_file_location("tta_search", SCRIPT)
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)


def args(**overrides):
    values = {
        "method": "tent",
        "settings": ["duet-r2r", "etpnav-r2r-ce"],
        "smoke": False,
        "stage": "stage1",
        "batch_id": "unit-test",
        "episodes": 256,
        "gpu": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def result(run_tag, setting, parameters, primary, sr=80.0, drift=0.01,
           updates=10):
    metric = "RGSPL" if setting.endswith("reverie") else "SPL"
    metrics = {metric: primary, "SR": sr, "SPL": primary, "RGS": primary}
    return {
        "run_tag": run_tag,
        "setting": setting,
        "parameters": parameters,
        "metrics": metrics,
        "adapter_diagnostics": {
            "relative_param_drift": drift,
            "updates": updates,
        },
    }


class TTAHparamSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = MODULE.load_spec()

    def test_stage1_grids_are_complete_and_round_robin(self):
        expected_per_setting = {
            "tent": 35,
            "fstta": 20,
            "eam": 35,
            "feedtta": 35,
            "atena": 5,
        }
        for method, count in expected_per_setting.items():
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                jobs = MODULE.build_jobs(
                    args(method=method), self.spec, Path(directory)
                )
                self.assertEqual(len(jobs), count * 2)
                self.assertEqual([job["setting"] for job in jobs[:4]], [
                    "duet-r2r", "etpnav-r2r-ce",
                    "duet-r2r", "etpnav-r2r-ce",
                ])

    def test_tent_anchor_includes_fixed_protocol(self):
        anchor = MODULE.anchor_for("tent", "duet-r2r", self.spec)
        self.assertEqual(anchor["optimizer"], "AdamW")
        self.assertEqual(anchor["norm_scope"], "ln")
        self.assertEqual(anchor["max_grad_norm"], 0.0)
        self.assertFalse(anchor["episodic"])

    def test_smoke_has_one_anchor_per_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = MODULE.build_jobs(
                args(smoke=True, stage="smoke", episodes=2),
                self.spec,
                Path(directory),
            )
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(job["stage"] == "smoke" for job in jobs))
        self.assertTrue(all(job["episodes"] == 2 for job in jobs))

    def test_method_specific_stage_expansions(self):
        setting = "duet-r2r"
        cases = (
            ("fstta", "stage2", 8),
            ("fstta", "stage3", 10),
            ("eam", "stage2", 10),
            ("eam", "stage3", 8),
            ("feedtta", "stage2", 39),
            ("atena", "stage2", 23),
            ("atena", "stage3", 10),
        )
        for method, stage, expected in cases:
            anchor = MODULE.clean_parameters(
                MODULE.anchor_for(method, setting, self.spec)
            )
            second = dict(anchor)
            if method == "fstta":
                second["lr_fast"] *= 3
            elif method == "eam":
                second["lr"] *= 3
            elif method == "atena":
                second["lr_query"] *= 2
                second["lr_self"] *= 2
            promoted = [
                result("anchor", setting, anchor, 80),
                result("second", setting, second, 79),
            ]
            if method == "feedtta":
                promoted = promoted[:1]
            with self.subTest(method=method, stage=stage):
                expanded = MODULE.expand_stage(
                    method, stage, setting, promoted, self.spec
                )
                self.assertEqual(len(expanded), expected)

    def test_promotion_enforces_source_floor_and_keeps_anchor(self):
        setting = "duet-r2r"
        anchor = MODULE.clean_parameters(
            MODULE.anchor_for("tent", setting, self.spec)
        )
        other = dict(anchor)
        other["lr"] = 3e-6
        bad = dict(anchor)
        bad["lr"] = 1e-4
        candidates = [
            result("paper", setting, anchor, 70, sr=78),
            result("best", setting, other, 82, sr=80),
            result("unsafe", setting, bad, 95, sr=70),
        ]
        source = result("source", setting, {}, 75, sr=80)
        selected, record = MODULE.rank_and_promote(
            candidates, source, "tent", setting, 2, self.spec
        )
        self.assertEqual({item["run_tag"] for item in selected}, {"best", "paper"})
        unsafe = next(item for item in record["ranked"]
                      if item["run_tag"] == "unsafe")
        self.assertFalse(unsafe["eligible"])
        self.assertIn("below_matched_source_sr_floor", unsafe["reasons"])

    def test_order_jobs_are_exactly_three_and_pass_order_seed(self):
        candidates = {"duet-r2r": [
            MODULE._candidate({"lr": 1e-6}, order_seed=seed)
            for seed in (0, 1, 2)
        ]}
        with tempfile.TemporaryDirectory() as directory:
            jobs = MODULE.build_jobs(
                args(settings=["duet-r2r"], stage="orders", episodes=-1),
                self.spec,
                Path(directory),
                stage="orders",
                candidates_by_setting=candidates,
            )
        self.assertEqual([job["order_seed"] for job in jobs], [0, 1, 2])
        for seed, job in enumerate(jobs):
            self.assertNotIn("--episode-limit", job["command"])
            index = job["command"].index("--order-seed")
            self.assertEqual(job["command"][index + 1], str(seed))

    def test_retry_uses_new_run_tag_and_preserves_attempt_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            job_dir = Path(directory) / "job"
            job_dir.mkdir()
            config = job_dir / "parameters.json"
            config.write_text("{}", encoding="utf-8")
            for name in ("console.log", "exitcode", "worker_state.json"):
                (job_dir / name).write_text("old\n", encoding="utf-8")
            job = {
                "base_run_tag": "base", "run_tag": "base", "attempt": 0,
                "setting": "duet-r2r", "job_dir": str(job_dir),
                "config_path": str(config), "result_root": "/old/result",
                "command": ["runner", "--run-tag", "base"],
                "config_method": "tent", "search_method": "tent",
                "stage": "stage1", "episodes": 2, "order_seed": None,
                "parameters": {},
            }
            MODULE._bump_attempt(job)
            self.assertEqual(job["run_tag"], "base-retry1")
            self.assertIn("base-retry1", job["result_root"])
            self.assertEqual(job["command"][-1], "base-retry1")
            attempt = job_dir / "attempts" / "attempt-00"
            self.assertTrue((attempt / "console.log").is_file())
            self.assertTrue((attempt / "exitcode").is_file())

    def test_parse_metrics_rejects_incomplete_episode_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir = root / "job"
            result_root = root / "result"
            job_dir.mkdir()
            result_root.mkdir()
            (job_dir / "console.log").write_text(
                "Env name: val_seen, sr: 80.0, spl: 75.0\n",
                encoding="utf-8",
            )
            (result_root / "tta_diagnostics.json").write_text(json.dumps({
                "episode_count": 1,
                "adapter": {"episodes": 1, "updates": 1,
                            "relative_param_drift": 0.01},
            }), encoding="utf-8")
            job = {
                "job_dir": str(job_dir), "result_root": str(result_root),
                "episodes": 2, "setting": "duet-r2r", "config_method": "tent",
            }
            with self.assertRaisesRegex(MODULE.UserError, "incomplete episode stream"):
                MODULE.parse_metrics(job, self.spec)

    def test_screening_prerequisite_can_load_terminal_partial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            for ordinal in range(2):
                job_dir = stage / "jobs" / str(ordinal)
                job_dir.mkdir(parents=True)
                (job_dir / "job.json").write_text(json.dumps({
                    "ordinal": ordinal,
                }), encoding="utf-8")
            successful = result(
                "ok", "duet-r2r",
                MODULE.clean_parameters(
                    MODULE.anchor_for("tent", "duet-r2r", self.spec)
                ),
                75,
            )
            with mock.patch.object(
                MODULE, "write_summary",
                return_value=([successful], [{"run_tag": "oom", "exit_code": 1}]),
            ):
                loaded = MODULE.load_stage_results(
                    stage, self.spec, allow_partial=True
                )
                self.assertEqual(loaded, [successful])
                with self.assertRaisesRegex(MODULE.UserError, "incomplete"):
                    MODULE.load_stage_results(stage, self.spec)

    def test_workflow_has_controls_final_and_orders(self):
        self.assertEqual(MODULE.workflow_stages("tent"), (
            "smoke", "controls", "stage1", "final",
        ))
        self.assertEqual(MODULE.workflow_stages("fstta", include_orders=True), (
            "smoke", "controls", "stage1", "stage2", "stage3",
            "final", "orders",
        ))


if __name__ == "__main__":
    unittest.main()
