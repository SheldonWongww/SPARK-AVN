import copy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/run_r2r_local_refinement.py"
TRANSLATOR_SCRIPT = REPO_ROOT / "vln/scripts/tta_config_cli.py"
PLAN = REPO_ROOT / "vln/experiments/r2r_eam_atena_formal_confirmation_v1.json"

MODULE_SPEC = importlib.util.spec_from_file_location("formal_confirmation_runner", SCRIPT)
RUNNER = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(RUNNER)

TRANSLATOR_SPEC = importlib.util.spec_from_file_location(
    "formal_confirmation_translator", TRANSLATOR_SCRIPT
)
TRANSLATOR = importlib.util.module_from_spec(TRANSLATOR_SPEC)
TRANSLATOR_SPEC.loader.exec_module(TRANSLATOR)


class R2REAMATENAFormalConfirmationPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = RUNNER.load_spec(PLAN)

    def test_exact_five_singleton_phases(self):
        self.assertEqual(
            RUNNER.enabled_phases_by_setting(self.plan),
            {
                "duet-r2r": ["eam", "atena"],
                "hamt-r2r": ["eam", "atena"],
                "goat-r2r": ["eam"],
            },
        )
        phases = RUNNER.phase_sequence(self.plan)
        self.assertEqual(
            [
                (phase["phase_id"], phase["setting"], phase["method"], phase["job_count"])
                for phase in phases
            ],
            [
                ("00-duet-r2r-eam", "duet-r2r", "eam", 1),
                ("01-duet-r2r-atena", "duet-r2r", "atena", 1),
                ("02-hamt-r2r-eam", "hamt-r2r", "eam", 1),
                ("03-hamt-r2r-atena", "hamt-r2r", "atena", 1),
                ("04-goat-r2r-eam", "goat-r2r", "eam", 1),
            ],
        )
        self.assertEqual(self.plan["budget"]["total_before_confirmation"], 5)
        self.assertEqual(self.plan["controls"]["candidate_count"], 0)
        self.assertTrue(
            self.plan["controls"]["feedtta_sampled_no_update_per_setting"]
        )
        self.assertTrue(all(
            phase["method"] not in RUNNER.CONTROL_METHODS for phase in phases
        ))

    def test_singletons_are_exact_frozen_parameters(self):
        expected = {
            ("eam", "duet-r2r"): {
                "optimizer": "Adam",
                "max_grad_norm": 0.0,
                "episodic": False,
                "lr": 1e-5,
                "confidence_scale": 0.5,
                "memory_size": 64,
                "batch_size": 8,
                "update_interval": 8,
            },
            ("eam", "hamt-r2r"): {
                "optimizer": "Adam",
                "max_grad_norm": 0.0,
                "episodic": False,
                "lr": 1e-6,
                "confidence_scale": 0.6,
                "memory_size": 32,
                "batch_size": 8,
                "update_interval": 8,
            },
            ("eam", "goat-r2r"): {
                "optimizer": "Adam",
                "max_grad_norm": 0.0,
                "episodic": False,
                "lr": 3e-6,
                "confidence_scale": 0.3,
                "memory_size": 32,
                "batch_size": 8,
                "update_interval": 4,
            },
            ("atena", "duet-r2r"): {
                "self_loss_weight": 0.1,
                "optimizer": "AdamW",
                "weight_decay": 0.01,
                "max_grad_norm": 0.0,
                "action_selection": "argmax",
                "episodic": False,
                "lr_query": 1.6e-6,
                "lr_self": 2e-7,
                "mix_lambda": 0.5,
                "query_threshold": 0.0,
            },
            ("atena", "hamt-r2r"): {
                "self_loss_weight": 0.1,
                "optimizer": "AdamW",
                "weight_decay": 0.01,
                "max_grad_norm": 0.0,
                "action_selection": "argmax",
                "episodic": False,
                "lr_query": 3.2e-6,
                "lr_self": 4e-7,
                "mix_lambda": 0.25,
                "query_threshold": 0.1,
            },
        }
        for key, parameters in expected.items():
            method, setting = key
            candidates = RUNNER.expand_candidates(method, setting, self.plan)
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["parameters"], parameters)
            self.assertEqual(
                candidates[0]["candidate"],
                self.plan["parent_anchors"][method][setting]["candidate"],
            )

    def test_corrected_contract_is_explicit_and_typed(self):
        self.assertEqual(
            self.plan["protocol"]["supervision_groups"],
            {
                "unsupervised": ["eam"],
                "binary_feedback_supervised": ["atena"],
            },
        )
        self.assertTrue(RUNNER._enforces_corrected_result_contract(self.plan))

        without_switch = copy.deepcopy(self.plan)
        without_switch["protocol"]["enforce_corrected_result_contract"] = False
        self.assertFalse(RUNNER._enforces_corrected_result_contract(without_switch))

        invalid = copy.deepcopy(self.plan)
        invalid["protocol"]["enforce_corrected_result_contract"] = "true"
        with self.assertRaisesRegex(
            RUNNER.UserError, "enforce_corrected_result_contract must be boolean"
        ):
            RUNNER._validate_spec(invalid)

    def test_generated_full_split_configs_translate_at_one_worker(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(mock.patch.object(RUNNER, "LOG_ROOT", root / "logs"))
            stack.enter_context(
                mock.patch.object(RUNNER, "TUNING_ROOT", root / "tuning")
            )
            for phase in RUNNER.phase_sequence(self.plan):
                self.assertEqual(RUNNER._safe_worker_limit(phase, self.plan), 1)
                jobs = RUNNER.build_phase_jobs(
                    phase,
                    "vln-r2r-eam-atena-formal-confirm-v1-seed0",
                    self.plan,
                    gpu=2,
                )
                self.assertEqual(len(jobs), 1)
                job = jobs[0]
                self.assertEqual(job["episodes"], -1)
                self.assertIsNone(job["order_seed"])
                self.assertNotIn("--episode-limit", job["command"])
                self.assertNotIn("--order-seed", job["command"])
                RUNNER.staged._write_job(job)
                translated_method, tokens = TRANSLATOR.translate(
                    job["setting"], job["config_path"],
                    str(root / "tta_diagnostics.json"),
                )
                self.assertEqual(translated_method, phase["method"])
                self.assertTrue(tokens)

    def _atena_result(self, directory, setting, queries):
        phase = RUNNER._find_phase(self.plan, setting, "atena")
        parameters = RUNNER.expand_candidates("atena", setting, self.plan)[0][
            "parameters"
        ]
        expected = self.plan["protocol"]["episode_count"]
        self_labels = expected - queries
        adapter = {
            "queries": queries,
            "feedback_observed_episodes": queries,
            "query_gate_evaluations": expected,
            "self_label_episodes": self_labels,
            "queried_feedback_successes": min(600, queries),
            "self_feedback_successes": min(180, self_labels),
            "query_rate": queries / expected,
            "feedback_observation_rate": queries / expected,
        }
        diagnostics = {
            "binary_feedback_endpoint": (
                "r2r_submitted_trajectory_evaluator_success_lazy_query"
            ),
            "adapter": adapter,
        }
        path = Path(directory) / f"{setting}-tta_diagnostics.json"
        path.write_text(json.dumps(diagnostics), encoding="utf-8")
        result = {
            "run_tag": f"formal-confirm-{setting}-atena",
            "parameters": parameters,
            "diagnostics_path": str(path),
            "adapter_diagnostics": adapter,
            "metrics": {"SR": 80.0, "SPL": 75.0},
        }
        return phase, result, diagnostics, path

    def test_corrected_atena_contract_accepts_full_and_partial_query_budgets(self):
        with tempfile.TemporaryDirectory() as directory:
            for setting, queries in (("duet-r2r", 1021), ("hamt-r2r", 796)):
                with self.subTest(setting=setting):
                    phase, result, _, _ = self._atena_result(
                        directory, setting, queries
                    )
                    RUNNER._validate_postfix_result_contract(
                        phase, result, self.plan
                    )

    def test_corrected_atena_contract_rejects_endpoint_and_budget_drift(self):
        mutations = {
            "endpoint": lambda diagnostics, adapter: diagnostics.update({
                "binary_feedback_endpoint": "simulator_stop_distance"
            }),
            "observed": lambda diagnostics, adapter: adapter.update({
                "feedback_observed_episodes": adapter["queries"] - 1
            }),
            "gate": lambda diagnostics, adapter: adapter.update({
                "query_gate_evaluations": 1020
            }),
            "self_labels": lambda diagnostics, adapter: adapter.update({
                "self_label_episodes": adapter["self_label_episodes"] + 1
            }),
            "query_rate": lambda diagnostics, adapter: adapter.update({
                "query_rate": 0.5
            }),
            "queried_successes": lambda diagnostics, adapter: adapter.update({
                "queried_feedback_successes": adapter["queries"] + 1
            }),
        }
        with tempfile.TemporaryDirectory() as directory:
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    phase, result, diagnostics, path = self._atena_result(
                        directory, "hamt-r2r", 796
                    )
                    mutate(diagnostics, result["adapter_diagnostics"])
                    path.write_text(json.dumps(diagnostics), encoding="utf-8")
                    with self.assertRaises(RUNNER.UserError):
                        RUNNER._validate_postfix_result_contract(
                            phase, result, self.plan
                        )

            phase, result, diagnostics, path = self._atena_result(
                directory, "duet-r2r", 1020
            )
            path.write_text(json.dumps(diagnostics), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.UserError, "invalid lazy ATENA"):
                RUNNER._validate_postfix_result_contract(phase, result, self.plan)


if __name__ == "__main__":
    unittest.main()
