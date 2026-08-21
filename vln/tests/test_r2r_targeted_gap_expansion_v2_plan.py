import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/run_r2r_local_refinement.py"
PLAN = REPO_ROOT / "vln/experiments/r2r_targeted_gap_expansion_v2.json"
PARENT = REPO_ROOT / "vln/experiments/r2r_targeted_gap_refinement_v1.json"
POSTFIX = REPO_ROOT / "vln/experiments/r2r_fstta_feedtta_postfix_search_v1.json"
MODULE_SPEC = importlib.util.spec_from_file_location("expanded_runner", SCRIPT)
RUNNER = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(RUNNER)


class R2RTargetedGapExpansionV2PlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = RUNNER.load_spec(PLAN)
        cls.parent = RUNNER.load_spec(PARENT)
        cls.postfix = RUNNER.load_spec(POSTFIX)

    def test_three_phase_budget_is_exact(self):
        self.assertEqual(
            [(p["setting"], p["method"], p["job_count"])
             for p in RUNNER.phase_sequence(self.plan)],
            [
                ("duet-r2r", "tent", 24),
                ("goat-r2r", "feedtta", 36),
                ("goat-r2r", "atena", 37),
            ],
        )
        self.assertEqual(self.plan["budget"]["search_total"], 97)
        self.assertEqual(self.plan["controls"]["candidate_count"], 0)

    def test_every_candidate_is_new_relative_to_corrected_parent(self):
        pairs = (
            ("tent", "duet-r2r"),
            ("feedtta", "goat-r2r"),
            ("atena", "goat-r2r"),
        )
        for method, setting in pairs:
            current = {
                RUNNER.canonical(item["parameters"])
                for item in RUNNER.expand_candidates(method, setting, self.plan)
            }
            parent = {
                RUNNER.canonical(item["parameters"])
                for item in RUNNER.expand_candidates(method, setting, self.parent)
            }
            self.assertTrue(current.isdisjoint(parent), (method, setting))
            self.assertEqual(
                len(current),
                self.plan["methods"][method]["settings"][setting][
                    "candidate_count"
                ],
            )

        current_feedtta = {
            RUNNER.canonical(item["parameters"])
            for item in RUNNER.expand_candidates(
                "feedtta", "goat-r2r", self.plan
            )
        }
        postfix_feedtta = {
            RUNNER.canonical(item["parameters"])
            for item in RUNNER.expand_candidates(
                "feedtta", "goat-r2r", self.postfix
            )
        }
        self.assertTrue(current_feedtta.isdisjoint(postfix_feedtta))

    def test_tent_expands_only_replay_connected_scope_at_interval_one(self):
        candidates = RUNNER.expand_candidates("tent", "duet-r2r", self.plan)
        self.assertEqual(
            {item["parameters"]["last_k_ln"] for item in candidates},
            {4, 6, 9, 15, 29},
        )
        self.assertTrue(all(
            item["parameters"]["update_interval"] == 1
            and item["parameters"]["norm_scope"] == "last_k_ln"
            for item in candidates
        ))

    def test_feedtta_expands_gamma_and_limits_sgr_budget(self):
        candidates = RUNNER.expand_candidates(
            "feedtta", "goat-r2r", self.plan
        )
        self.assertTrue(all(
            item["parameters"]["action_selection"] == "argmax"
            and item["parameters"]["scope_profile"] == "action_head"
            for item in candidates
        ))
        diagnostic = [item for item in candidates if item["role"] is not None]
        self.assertEqual(len(diagnostic), 6)
        self.assertEqual(
            {item["parameters"]["gamma"] for item in candidates},
            {0.7, 0.75, 0.8, 0.85, 0.9, 0.95},
        )

    def test_atena_has_low_self_lr_and_corrected_high_spl_branches(self):
        candidates = RUNNER.expand_candidates("atena", "goat-r2r", self.plan)
        high = [
            item for item in candidates
            if item["parameters"]["query_threshold"] == 0.0
        ]
        low = [
            item for item in candidates
            if item["parameters"]["query_threshold"] > 0.0
        ]
        self.assertEqual((len(low), len(high)), (25, 12))
        self.assertEqual(
            {item["parameters"]["mix_lambda"] for item in high},
            {0.625, 0.75, 0.875},
        )
        self.assertTrue(all(
            item["parameters"]["mix_lambda"] == 0.0 for item in low
        ))

    def test_search_priors_are_tracked_formal_manifests(self):
        RUNNER._validate_search_prior_artifacts(self.plan)
        for method, settings in self.plan["search_priors"].items():
            for setting, prior in settings.items():
                self.assertIn("prior_manifest_path", prior)
                self.assertNotIn("prior_job_path", prior)
                phase = next(
                    item for item in RUNNER.phase_sequence(self.plan)
                    if item["method"] == method and item["setting"] == setting
                )
                jobs = RUNNER.build_phase_jobs(
                    phase, "expanded-prior-test", self.plan
                )
                bound = [job for job in jobs if job["search_prior"] is not None]
                self.assertEqual(len(bound), 1)
                self.assertEqual(
                    bound[0]["search_prior"]["manifest_path"],
                    prior["prior_manifest_path"],
                )

    def test_measured_concurrency_is_reused(self):
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
            self.assertIn("targeted_v1", value["cap_basis"])


if __name__ == "__main__":
    unittest.main()
