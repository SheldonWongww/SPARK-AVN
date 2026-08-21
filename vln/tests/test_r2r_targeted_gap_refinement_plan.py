import copy
import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/run_r2r_local_refinement.py"
PLAN = REPO_ROOT / "vln/experiments/r2r_targeted_gap_refinement_v1.json"
MODULE_SPEC = importlib.util.spec_from_file_location("targeted_runner", SCRIPT)
RUNNER = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(RUNNER)


class R2RTargetedGapRefinementPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = RUNNER.load_spec(PLAN)

    def test_only_three_weak_model_method_pairs_are_enabled(self):
        self.assertEqual(
            RUNNER.enabled_phases_by_setting(self.plan),
            {
                "duet-r2r": ["tent"],
                "hamt-r2r": [],
                "goat-r2r": ["feedtta", "atena"],
            },
        )
        phases = RUNNER.phase_sequence(self.plan)
        self.assertEqual(
            [(item["setting"], item["method"], item["job_count"])
             for item in phases],
            [
                ("duet-r2r", "tent", 9),
                ("goat-r2r", "feedtta", 11),
                ("goat-r2r", "atena", 9),
            ],
        )
        self.assertEqual(sum(item["job_count"] for item in phases), 29)

    def test_source_and_sampled_controls_are_reused_not_rerun(self):
        self.assertEqual(self.plan["controls"]["candidate_count"], 0)
        self.assertFalse(
            self.plan["controls"]["feedtta_sampled_no_update_per_setting"]
        )
        self.assertTrue(all(
            "source" not in methods and "feedtta_control" not in methods
            for methods in RUNNER.enabled_phases_by_setting(self.plan).values()
        ))

    def test_tent_keeps_update_interval_one_and_localizes_scope(self):
        candidates = RUNNER.expand_candidates("tent", "duet-r2r", self.plan)
        self.assertEqual(len(candidates), 9)
        self.assertEqual(
            {item["parameters"]["lr"] for item in candidates},
            {3e-6, 1e-5, 3e-5},
        )
        self.assertTrue(all(
            item["parameters"]["update_interval"] == 1
            and item["parameters"]["norm_scope"] in {"last_ln", "last_k_ln"}
            for item in candidates
        ))
        self.assertTrue(all(
            item["parameters"]["norm_scope"] != "ln" for item in candidates
        ))

    def test_feedtta_stays_on_corrected_action_head_protocol(self):
        candidates = RUNNER.expand_candidates(
            "feedtta", "goat-r2r", self.plan
        )
        self.assertEqual(len(candidates), 11)
        self.assertTrue(all(
            item["parameters"]["action_selection"] == "argmax"
            and item["parameters"]["scope_profile"] == "action_head"
            and item["parameters"]["p"] == 0.05
            and item["parameters"]["alpha"] == 0.1
            for item in candidates
        ))
        self.assertFalse(any(
            item["parameters"]["lr"] == 1e-6
            and item["parameters"]["gamma"] == 0.9
            for item in candidates
        ))

    def test_atena_reruns_corrected_anchor_and_focuses_query_boundary(self):
        candidates = RUNNER.expand_candidates("atena", "goat-r2r", self.plan)
        self.assertEqual(len(candidates), 9)
        anchors = [
            item for item in candidates
            if item["role"] == "corrected_evaluator_feedback_anchor"
        ]
        self.assertEqual(len(anchors), 1)
        self.assertEqual(
            anchors[0]["candidate"],
            {
                "lr_query": 4e-7,
                "lr_self": 5e-8,
                "mix_lambda": 0.0,
                "query_threshold": 0.15,
            },
        )
        self.assertTrue(all(
            item["parameters"]["mix_lambda"] == 0.0
            and item["parameters"]["query_threshold"] in {0.13, 0.15, 0.17}
            for item in candidates
        ))

    def test_targeted_schema_rejects_protocol_drift(self):
        invalid = copy.deepcopy(self.plan)
        invalid["execution"]["enabled_methods_by_setting"]["hamt-r2r"] = [
            "tent"
        ]
        with self.assertRaisesRegex(
            RUNNER.UserError, "targeted gap refinement"
        ):
            RUNNER._validate_spec(invalid)

        invalid = copy.deepcopy(self.plan)
        invalid["methods"]["feedtta"]["fixed"]["action_selection"] = "sample"
        with self.assertRaisesRegex(RUNNER.UserError, "FeedTTA"):
            RUNNER._validate_spec(invalid)

    def test_measured_concurrency_caps_are_reused(self):
        expected = {
            ("tent", "duet-r2r"): 10,
            ("feedtta", "goat-r2r"): 6,
            ("atena", "goat-r2r"): 5,
        }
        for (method, setting), cap in expected.items():
            value = self.plan["execution"]["concurrency_calibration"][method][
                setting
            ]
            self.assertEqual(value["production_cap"], cap)
            self.assertIn("observed_peak_mib", value)
        self.assertEqual(
            self.plan["execution"]["gpu_safety"][
                "approved_projection_exceptions"
            ],
            [],
        )


if __name__ == "__main__":
    unittest.main()
