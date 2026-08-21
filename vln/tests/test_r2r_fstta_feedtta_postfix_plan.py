import copy
import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/run_r2r_local_refinement.py"
PLAN = REPO_ROOT / "vln/experiments/r2r_fstta_feedtta_postfix_search_v1.json"
MODULE_SPEC = importlib.util.spec_from_file_location("postfix_runner", SCRIPT)
RUNNER = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(RUNNER)


class R2RFSTTAFeedTTAPostfixPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = RUNNER.load_spec(PLAN)

    def test_plan_contains_only_the_two_requested_methods(self):
        self.assertEqual(RUNNER.active_tta_methods(self.plan), ("fstta", "feedtta"))
        phases = RUNNER.phase_sequence(self.plan)
        self.assertEqual(len(phases), 6)
        self.assertEqual(sum(item["job_count"] for item in phases), 78)
        self.assertTrue(all(item["method"] in ("fstta", "feedtta") for item in phases))

    def test_source_is_reused_and_sampled_control_is_removed(self):
        self.assertEqual(self.plan["controls"]["candidate_count"], 0)
        self.assertFalse(
            self.plan["controls"]["feedtta_sampled_no_update_per_setting"]
        )
        self.assertEqual(
            self.plan["controls"]["feedtta_action_protocol"],
            "target_native_argmax",
        )
        self.assertTrue(all(
            "source" not in methods and "feedtta_control" not in methods
            for methods in RUNNER.enabled_phases_by_setting(self.plan).values()
        ))

    def test_fstta_and_feedtta_candidate_contracts(self):
        for setting in RUNNER.SETTINGS:
            fstta = RUNNER.expand_candidates("fstta", setting, self.plan)
            feedtta = RUNNER.expand_candidates("feedtta", setting, self.plan)
            self.assertEqual(len(fstta), 12)
            self.assertEqual(len(feedtta), 14)
            self.assertTrue(all(
                item["parameters"]["action_selection"] == "argmax"
                for item in feedtta
            ))
            self.assertTrue(all(
                item["parameters"]["scope_profile"] in {
                    "paper_full", "last_crossmodal", "action_head"
                }
                for item in feedtta
            ))
            self.assertTrue(any(
                item["parameters"]["p"] == 0.05
                and item["parameters"]["alpha"] == 0.1
                for item in feedtta
            ))
        published = {
            "lr_fast": 0.0006, "lr_slow": 0.001, "m": 3, "n": 4,
        }
        duet = RUNNER.expand_candidates("fstta", "duet-r2r", self.plan)
        self.assertEqual(sum(
            all(item["parameters"][key] == value for key, value in published.items())
            for item in duet
        ), 1)

    def test_feedtta_priors_are_explicit_protocol_transfers(self):
        for setting in RUNNER.SETTINGS:
            prior = self.plan["search_priors"]["feedtta"][setting]
            self.assertEqual(
                prior["historical_parameters"]["action_selection"], "sample"
            )
            self.assertIn("not an exact parent replay", prior["transfer_note"])
            jobs = RUNNER.build_phase_jobs(
                next(
                    phase for phase in RUNNER.phase_sequence(self.plan)
                    if phase["setting"] == setting
                    and phase["method"] == "feedtta"
                ),
                "postfix-prior-test",
                self.plan,
            )
            bound = [job for job in jobs if job["search_prior"] is not None]
            self.assertEqual(len(bound), 1)
            self.assertEqual(bound[0]["parent_run_tags"], [])

    def test_v2_validator_rejects_protocol_drift(self):
        mutations = []

        sampled = copy.deepcopy(self.plan)
        sampled["methods"]["feedtta"]["fixed"]["action_selection"] = "sample"
        mutations.append(sampled)

        bad_scope = copy.deepcopy(self.plan)
        bad_scope["methods"]["feedtta"]["settings"]["duet-r2r"][
            "points"
        ][0]["scope_profile"] = "nonsense"
        mutations.append(bad_scope)

        supervised = copy.deepcopy(self.plan)
        supervised["protocol"]["supervision_groups"] = {
            "unsupervised": ["fstta", "feedtta"],
            "binary_feedback_supervised": [],
        }
        mutations.append(supervised)

        screened = copy.deepcopy(self.plan)
        screened["protocol"]["prefix_screening_forbidden"] = False
        mutations.append(screened)

        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with self.assertRaises(RUNNER.UserError):
                    RUNNER._validate_spec(mutation)

    def test_concurrency_preserves_model_method_barriers(self):
        self.assertTrue(self.plan["execution"]["strict_model_barrier_required"])
        self.assertTrue(self.plan["execution"]["strict_method_barrier_required"])
        expected = {
            "fstta": {"duet-r2r": 14, "hamt-r2r": 11, "goat-r2r": 14},
            "feedtta": {"duet-r2r": 6, "hamt-r2r": 5, "goat-r2r": 6},
        }
        for method, settings in expected.items():
            for setting, cap in settings.items():
                self.assertEqual(
                    self.plan["execution"]["concurrency_calibration"]
                    [method][setting]["production_cap"],
                    cap,
                )


if __name__ == "__main__":
    unittest.main()
