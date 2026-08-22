from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln/scripts"))

import run_r2r_val_unseen_frozen_eval as runner  # noqa: E402
from tta_config_cli import translate  # noqa: E402


SPEC_PATH = REPO_ROOT / "vln/experiments/r2r_val_unseen_frozen_eval_v1.json"


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class R2RValUnseenFrozenEvalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = runner.load_spec(SPEC_PATH)
        cls.jobs = runner.expand_jobs(cls.spec, "unit-test-batch", gpu=0)

    def test_spec_is_exact_model_major_fifteen_tta_zero_source_matrix(self):
        self.assertEqual(
            self.spec["budget"],
            {
                "source_execution_jobs": 0,
                "tta_jobs": 15,
                "total_executed_jobs": 15,
            },
        )
        self.assertEqual(len(self.jobs), 15)
        self.assertNotIn("source", {job["method"] for job in self.jobs})
        self.assertEqual(
            [(job["setting"], job["method"]) for job in self.jobs],
            [
                (setting, method)
                for setting in runner.SETTINGS
                for method in runner.METHODS
            ],
        )
        phases = runner.model_phases(self.jobs)
        self.assertEqual([phase[0]["model"] for phase in phases], list(runner.MODELS))
        for phase in phases:
            self.assertEqual([job["method"] for job in phase], list(runner.METHODS))
            self.assertEqual(len({job["model"] for job in phase}), 1)
            self.assertEqual(
                self.spec["execution"]["max_workers_by_model"][
                    phase[0]["model"]
                ],
                5,
            )

    def test_protocol_pins_canonical_val_unseen_without_order_seed_cli(self):
        protocol = self.spec["protocol"]
        self.assertEqual(protocol["split"], "val_unseen")
        self.assertEqual(protocol["episode_count"], 2349)
        self.assertEqual(protocol["canonical_order_seed"], 0)
        self.assertTrue(protocol["order_seed_cli_forbidden"])
        self.assertEqual(
            protocol["episode_order_sha256"],
            "bfaa25c07a8755e67585b1cde3c99833377005ea70017107457761e8ad82a390",
        )
        for binding in protocol["order_manifests"].values():
            document = json.loads(
                (REPO_ROOT / binding["path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(document["split"], "val_unseen")
            self.assertEqual(document["episode_count"], 2349)
            self.assertEqual(
                document["order_sha256"], protocol["episode_order_sha256"]
            )

    def test_parameters_are_copied_verbatim_from_frozen_registry(self):
        _, registry = runner.load_registry(self.spec)
        for job in self.jobs:
            entry = registry["records"][job["setting"]][job["method"]]
            self.assertEqual(job["parameters"], entry["parameters"])
            self.assertEqual(job["selected_anchor"], entry)
            self.assertEqual(
                job["selected_binding"]["selected_run_tag"], entry["run_tag"]
            )
            self.assertEqual(
                job["selected_binding"]["selected_formal_manifest_sha256"],
                entry["formal_manifest_sha256"],
            )

    def test_registry_binding_rejects_a_different_selected_run(self):
        candidate = deepcopy(self.spec)
        candidate["matrix"]["jobs"][0]["selected_run_tag"] = "wrong-run"
        _, registry = runner.load_registry(self.spec)
        with self.assertRaisesRegex(runner.UserError, "selected run mismatch"):
            runner._validate_matrix_against_registry(candidate, registry)

    def test_all_configs_translate_and_commands_omit_forbidden_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory)
            for job in self.jobs:
                attempt_dir, metadata = runner.materialize_attempt(
                    self.spec, SPEC_PATH, "unit-test-batch", batch_root, job, 0
                )
                config_path = attempt_dir / "parameters.json"
                config = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertNotIn("order_seed", config)
                self.assertEqual(config["stage"], "frozen_val_unseen")
                self.assertEqual(config["episodes"], -1)
                self.assertEqual(config["parameters"], job["parameters"])
                method, tokens = translate(
                    job["setting"], config_path, attempt_dir / "diag.json"
                )
                self.assertEqual(method, job["method"])
                self.assertIn("--tta_method", tokens)
                command = metadata["command"]
                self.assertEqual(command[1:3], [job["setting"], "val_unseen"])
                self.assertNotIn("--order-seed", command)
                self.assertNotIn("--episode-limit", command)
                self.assertNotIn("--result-root", command)
                self.assertEqual(metadata["canonical_order_seed"], 0)
                self.assertTrue(
                    metadata["result_root"].endswith(
                        "/{}/val_unseen".format(job["setting"])
                    )
                )

    def test_source_ledger_reuses_three_formal_controls_without_source_jobs(self):
        _, ledger, blockers = runner.validate_source_ledger(
            self.spec, require_ready=True, require_metrics=True
        )
        self.assertEqual(
            ledger["source_execution_policy"]["execution"], "reuse_only"
        )
        self.assertTrue(
            ledger["source_execution_policy"]["rerun_forbidden"]
        )
        self.assertEqual(
            {setting for setting, record in ledger["records"].items()
             if record["evidence_status"] == "ready"},
            set(runner.SETTINGS),
        )
        self.assertEqual(blockers, [])
        for setting in runner.SETTINGS:
            record = ledger["records"][setting]
            self.assertTrue(
                (REPO_ROOT / record["formal_manifest_path"]).is_file()
            )

    def test_missing_hamt_manifest_blocks_before_batch_materialization(self):
        ignored_root = REPO_ROOT / "vln/results/logs"
        ignored_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ignored_root) as evidence_directory, \
                tempfile.TemporaryDirectory() as runtime_directory:
            evidence_directory = Path(evidence_directory)
            source_path = REPO_ROOT / self.spec["source_control"]["manifest"]
            source = json.loads(source_path.read_text(encoding="utf-8"))
            hamt = source["records"]["hamt-r2r"]
            for key in (
                "formal_manifest_path", "formal_manifest_sha256",
                "immutable_identity_sha256",
            ):
                hamt.pop(key)
            hamt.update({
                "evidence_status": "missing_formal_manifest",
                "expected_formal_manifest_path": (
                    "vln/results/runs/synthetic-missing-hamt-val-unseen/"
                    "manifest.json"
                ),
                "blocking_reason": "synthetic missing HAMT formal manifest",
            })
            source_copy = evidence_directory / "source.json"
            _write_json(source_copy, source)
            candidate = deepcopy(self.spec)
            candidate["source_control"]["manifest"] = str(source_copy)
            candidate["source_control"]["sha256"] = runner._sha256(source_copy)
            spec_copy = evidence_directory / "spec.json"
            _write_json(spec_copy, candidate)

            batch_id = "unit-test-hamt-source-blocker"
            log_root = Path(runtime_directory) / "logs"
            batch_root = log_root / batch_id
            with mock.patch.object(runner, "LOG_ROOT", log_root):
                with self.assertRaisesRegex(
                    runner.UserError,
                    "recover the existing manifest without rerunning Source.*hamt-r2r",
                ):
                    runner.main([
                        "--spec", str(spec_copy),
                        "--batch-id", batch_id,
                        "--confirm-reviewed",
                    ])
            self.assertFalse(batch_root.exists())

    def test_feedback_contract_uses_r2r_evaluator_endpoints(self):
        episode_count = 2349
        successes = 1000
        feed_job = next(job for job in self.jobs if job["method"] == "feedtta")
        feed_metadata = {
            "episode_count": episode_count,
            "method": "feedtta",
            "parameters": feed_job["parameters"],
        }
        feed_diagnostics = {
            "method": "feedtta",
            "episode_count": episode_count,
            "supervision": "binary_navigation_success_feedback",
            "binary_feedback_endpoint": (
                "r2r_submitted_trajectory_evaluator_success_every_episode"
            ),
            "adapter": {
                "episodes": episode_count,
                "feedback_episodes": episode_count,
                "successful_feedback_episodes": successes,
                "failed_feedback_episodes": episode_count - successes,
                "action_selection_protocol": "target_native_argmax",
            },
        }
        metrics = {"SR": successes * 100.0 / episode_count, "SPL": 40.0}
        runner._validate_diagnostics(feed_metadata, feed_diagnostics, metrics)
        feed_diagnostics["binary_feedback_endpoint"] = (
            "final_simulator_observation_distance"
        )
        with self.assertRaisesRegex(
            runner.UserError, "binary-feedback endpoint"
        ):
            runner._validate_diagnostics(feed_metadata, feed_diagnostics, metrics)

        fstta_job = next(job for job in self.jobs if job["method"] == "fstta")
        fstta_metadata = {
            "episode_count": episode_count,
            "method": "fstta",
            "parameters": fstta_job["parameters"],
        }
        fstta_diagnostics = {
            "method": "fstta",
            "episode_count": episode_count,
            "supervision": "unsupervised",
            "binary_feedback_endpoint": None,
            "adapter": {
                "episodes": episode_count,
                "variance_history_lifetime": "episode",
            },
        }
        with self.assertRaisesRegex(runner.UserError, "stream variance"):
            runner._validate_diagnostics(
                fstta_metadata, fstta_diagnostics, {"SR": 1.0, "SPL": 1.0}
            )


if __name__ == "__main__":
    unittest.main()
