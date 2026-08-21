import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "vln/scripts/run_tta_hparam_search.py"
SPEC_PATH = REPO_ROOT / "vln/experiments/r2r_ce_small_hparam_search_v1.json"
SOURCE_MANIFEST = (
    REPO_ROOT / "vln/manifests/r2r_ce_reused_source_controls.json"
)
MODULE_SPEC = importlib.util.spec_from_file_location("r2r_ce_search", RUNNER)
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)


def args(method, stage="stage1", episodes=100):
    return SimpleNamespace(
        method=method,
        settings=["etpnav-r2r-ce", "bevbert-r2r-ce"],
        smoke=False,
        stage=stage,
        batch_id="r2r-ce-small-unit",
        episodes=episodes,
        gpu=0,
    )


class R2RCESmallSearchPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = MODULE.load_spec(SPEC_PATH)

    def test_scope_is_two_formal_models_and_streamvln_is_blocked(self):
        self.assertEqual(self.spec["settings"], [
            "etpnav-r2r-ce", "bevbert-r2r-ce",
        ])
        self.assertNotIn("streamvln-r2r-ce", self.spec["settings"])
        self.assertEqual(
            self.spec["profile"]["streamvln_status"],
            "blocked_no_formal_tta_runner",
        )
        for setting in self.spec["settings"]:
            self.assertEqual(self.spec["setting_episode_counts"][setting], 778)

    def test_explicit_spec_selects_reviewed_scheduler_defaults(self):
        parsed = MODULE.parse_args([
            "all", "--spec", str(SPEC_PATH), "--batch-id", "explicit-spec",
        ])
        self.assertEqual(parsed.spec, SPEC_PATH.resolve())
        self.assertEqual(parsed.max_workers, 1)
        self.assertEqual(parsed.max_continuous_workers, 1)
        self.assertEqual(parsed.max_gpu_memory_mib, 16000)

    def test_exact_two_stage_budget_executes_no_source(self):
        self.assertEqual(
            MODULE.workflow_stages("feedtta", spec=self.spec),
            ("stage1", "final"),
        )
        self.assertEqual(self.spec["screening_episodes"], 100)
        self.assertEqual(self.spec["budget"], {
            "screening_tta_jobs": 40,
            "full_val_seen_tta_jobs": 10,
            "source_execution_jobs": 0,
            "total_executed_jobs": 50,
        })
        for method in MODULE.METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                jobs = MODULE.build_jobs(
                    args(method), self.spec, Path(root), stage="stage1"
                )
                self.assertEqual(len(jobs), 8)
                self.assertTrue(all(job["config_method"] == method for job in jobs))
                self.assertTrue(all(job["episodes"] == 100 for job in jobs))
                self.assertTrue(all(
                    job["command"][job["command"].index("--episode-limit") + 1]
                    == "100" for job in jobs
                ))
                with self.assertRaisesRegex(
                    MODULE.UserError, "must not execute Source"
                ):
                    MODULE._stage_candidates(
                        method, "controls", self.spec["settings"],
                        self.spec, Path(root),
                    )

    def test_four_candidates_per_model_method_and_one_full_finalist(self):
        for method in MODULE.METHODS:
            for setting in self.spec["settings"]:
                with self.subTest(method=method, setting=setting):
                    candidates = list(MODULE.stage1_points(
                        method, setting, self.spec
                    ))
                    self.assertEqual(len(candidates), 4)
                    self.assertEqual(
                        len({MODULE._canonical(item) for item in candidates}), 4
                    )
        self.assertEqual(
            self.spec["protocol"]["full_val_seen_finalists_per_setting"], 1
        )
        self.assertFalse(
            self.spec["protocol"]["paper_anchor_always_promoted"]
        )
        self.assertFalse(
            self.spec["protocol"]["order_robustness_enabled"]
        )
        self.assertEqual(self.spec["final_order_seeds"], [0])
        with self.assertRaisesRegex(MODULE.UserError, "disables the orders"):
            MODULE.workflow_stages(
                "tent", include_orders=True, spec=self.spec
            )

    def test_low_learning_rates_and_fixed_method_contracts(self):
        for setting in self.spec["settings"]:
            tent = list(MODULE.stage1_points("tent", setting, self.spec))
            self.assertEqual({item["update_interval"] for item in tent}, {1})
            self.assertTrue({1e-8, 3e-8, 1e-7}.issubset(
                {item["lr"] for item in tent}
            ))

            fstta = list(MODULE.stage1_points("fstta", setting, self.spec))
            self.assertTrue(all(item["norm_scope"] == "last_k_ln"
                                for item in fstta))
            self.assertTrue(all(item["last_k_ln"] == 4 for item in fstta))
            self.assertTrue(any(item["lr_slow"] == 1e-5 for item in fstta))
            low_pairs = {
                (item["lr_fast"], item["lr_slow"])
                for item in fstta if item["lr_slow"] <= 1e-4
            }
            self.assertEqual(low_pairs, {
                (3e-7, 1e-5), (1e-6, 3e-5), (3e-6, 1e-4),
            })

            eam = list(MODULE.stage1_points("eam", setting, self.spec))
            self.assertTrue({3e-9, 1e-8, 3e-8}.issubset(
                {item["lr"] for item in eam}
            ))

            feedtta = list(MODULE.stage1_points(
                "feedtta", setting, self.spec
            ))
            self.assertTrue(all(item["action_selection"] == "argmax"
                                for item in feedtta))
            self.assertEqual(
                {item["scope_profile"] for item in feedtta},
                {"last_crossmodal", "action_head"},
            )
            self.assertTrue(all(item["lr"] <= 1e-7 for item in feedtta))

            atena = list(MODULE.stage1_points("atena", setting, self.spec))
            self.assertTrue(all(item["lr_self"] <= item["lr_query"]
                                for item in atena))
            self.assertTrue(all(item["action_selection"] == "argmax"
                                for item in atena))

    def test_reused_source_full_and_prefix_are_authenticated(self):
        expected = {
            "etpnav-r2r-ce": (60.0, 51.964427100351486,
                               67.48071979434447, 59.96944136961733),
            "bevbert-r2r-ce": (68.0, 59.10483880390991,
                                68.38046272493573, 59.9207010445596),
        }
        for setting, values in expected.items():
            with self.subTest(setting=setting):
                prefix = MODULE.reused_source_result(setting, 100, self.spec)
                full = MODULE.reused_source_result(setting, -1, self.spec)
                self.assertAlmostEqual(prefix["metrics"]["SR"], values[0])
                self.assertAlmostEqual(prefix["metrics"]["SPL"], values[1])
                self.assertAlmostEqual(full["metrics"]["SR"], values[2])
                self.assertAlmostEqual(full["metrics"]["SPL"], values[3])
                self.assertEqual(full["expected_episodes"], 778)
                self.assertEqual(full["parameters"]["action_selection"], "argmax")

    def test_reused_source_fails_closed_on_artifact_tampering(self):
        controls = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
        controls = copy.deepcopy(controls)
        controls["settings"]["etpnav-r2r-ce"]["per_episode_artifact"][
            "sha256"
        ] = "0" * 64
        with mock.patch.object(
            MODULE, "_load_reused_source_manifest", return_value=controls
        ), self.assertRaisesRegex(MODULE.UserError, "SHA256 mismatch"):
            MODULE.reused_source_result("etpnav-r2r-ce", 100, self.spec)

    def test_promotion_uses_prefix_source_floor_and_does_not_force_anchor(self):
        setting = "etpnav-r2r-ce"
        points = list(MODULE.stage1_points("tent", setting, self.spec))
        source = MODULE.reused_source_result(setting, 100, self.spec)
        results = []
        spl_values = (55.0, 54.0, 57.0, 52.0)
        for index, point in enumerate(points):
            results.append({
                "run_tag": "candidate-{}".format(index),
                "setting": setting,
                "parameters": point,
                "metrics": {
                    "SPL": spl_values[index],
                    "SR": 60.0,
                },
                "adapter_diagnostics": {
                    "relative_param_drift": 0.001 * index,
                    "updates": 10 + index,
                },
            })
        selected, _ = MODULE.rank_and_promote(
            results, source, "tent", setting, 1, self.spec,
            force_anchor=self.spec["protocol"]["paper_anchor_always_promoted"],
        )
        self.assertEqual(selected[0]["run_tag"], "candidate-2")

    def test_shared_gpu_caps_and_projected_memory_gate_are_conservative(self):
        coexistence = self.spec["protocol"]["shared_gpu_coexistence"]
        defaults = self.spec["scheduler_defaults"]
        self.assertEqual(coexistence["ce_worker_cap"], 1)
        self.assertEqual(coexistence["peer_worker_cap"], 1)
        self.assertEqual(defaults["max_workers"], 1)
        self.assertEqual(defaults["max_continuous_workers"], 1)
        self.assertTrue(coexistence["joint_readiness_ack_required"])
        self.assertTrue(coexistence["campaign_lifetime_lock_required"])
        self.assertTrue(coexistence["shared_active_reservation_required"])
        self.assertLessEqual(
            defaults["max_gpu_memory_mib_before_launch"]
            + defaults["estimated_job_gpu_memory_mib"],
            defaults["max_aggregate_gpu_memory_mib"],
        )
        self.assertLessEqual(
            defaults["max_cgroup_memory_gib_before_launch"]
            + defaults["estimated_job_memory_gib"],
            defaults["max_aggregate_cgroup_memory_gib"],
        )

    def test_formal_launch_requires_joint_entry_and_rejects_cli_overrides(self):
        defaults = self.spec["scheduler_defaults"]
        formal = SimpleNamespace(
            method="all", stage="all", settings=None, with_orders=False,
            dry_run=False, confirm_reviewed=False, joint_launch_manifest=None,
            batch_id="r2r-ce-small-unit", gpu=0,
            max_workers=defaults["max_workers"],
            max_per_model=defaults["max_per_model"],
            max_discrete_workers=defaults["max_discrete_workers"],
            max_continuous_workers=defaults["max_continuous_workers"],
            max_gpu_memory_mib=defaults["max_gpu_memory_mib_before_launch"],
            estimated_job_gpu_memory_mib=defaults[
                "estimated_job_gpu_memory_mib"
            ],
            max_aggregate_gpu_memory_mib=defaults[
                "max_aggregate_gpu_memory_mib"
            ],
            max_memory_gib=defaults["max_cgroup_memory_gib_before_launch"],
            estimated_job_memory_gib=defaults["estimated_job_memory_gib"],
            max_aggregate_memory_gib=defaults[
                "max_aggregate_cgroup_memory_gib"
            ],
            launch_stagger=defaults["launch_stagger_seconds"],
        )
        with self.assertRaisesRegex(MODULE.UserError, "confirm-reviewed"):
            MODULE._validate_formal_joint_launch(formal, self.spec)
        formal.confirm_reviewed = True
        formal.max_workers = 2
        with self.assertRaisesRegex(MODULE.UserError, "forbids CLI"):
            MODULE._validate_formal_joint_launch(formal, self.spec)


if __name__ == "__main__":
    unittest.main()
