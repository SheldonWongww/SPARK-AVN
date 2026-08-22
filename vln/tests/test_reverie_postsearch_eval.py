from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln/scripts"))

import build_reverie_final_registry as registry_builder  # noqa: E402
import prepare_reverie_val_unseen_frozen_eval as plan_builder  # noqa: E402
import run_reverie_val_unseen_frozen_eval as runner  # noqa: E402
from tta_config_cli import translate  # noqa: E402


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReveriePostSearchEvalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads(
            registry_builder.DEFAULT_SOURCE.read_text(encoding="utf-8")
        )
        transfer = json.loads((
            REPO_ROOT / "vln/experiments/reverie_r2r_frozen_transfer_v1.json"
        ).read_text(encoding="utf-8"))
        parameters = {
            (item["setting"], item["method"]): item["parameters"]
            for item in transfer["jobs"]
        }
        records = {}
        for setting_index, setting in enumerate(registry_builder.SETTINGS):
            records[setting] = {}
            source_metrics = cls.source["records"][setting]["metrics"]
            for method_index, method in enumerate(registry_builder.METHODS):
                matches = []
                for path in registry_builder.FORMAL_ROOT.glob(
                    "vln-reverie-r2r-frozen-transfer-v1-seed0-*"
                    "-{}-val_seen-native/manifest.json".format(setting)
                ):
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    if manifest.get("method") == method:
                        matches.append((path, manifest))
                if len(matches) != 1:
                    raise AssertionError(
                        "expected one frozen-transfer manifest for {} {}, got {}"
                        .format(setting, method, len(matches))
                    )
                path, manifest = matches[0]
                metric_path, _ = registry_builder._manifest_artifact(
                    manifest, "logs/valid.txt", "fixture"
                )
                metrics = registry_builder._parse_reverie_metrics(
                    metric_path, "val_seen", "fixture"
                )
                records[setting][method] = {
                    "origin": "r2r_frozen_transfer_incumbent",
                    "run_tag": manifest["run_tag"],
                    "parameters": parameters[(setting, method)],
                    "metrics": metrics,
                    "formal_manifest": str(path),
                    "formal_manifest_sha256": _sha256(path),
                    "metric_artifact_sha256": _sha256(metric_path),
                    "source_metrics": source_metrics,
                    "delta_vs_source_pp": {
                        key: round(metrics[key] - source_metrics[key], 6)
                        for key in registry_builder.METRICS
                    },
                    "better_than_source_on_rgspl": (
                        metrics["RGSPL"] > source_metrics["RGSPL"]
                    ),
                    "new_candidate_eligible": False,
                }
        cls.search_selection = {
            "schema": registry_builder.SEARCH_SELECTION_SCHEMA,
            "experiment_id": "vln-reverie-val-seen-small-hparam-search-v1",
            "split": "val_seen",
            "git_commit": "a" * 40,
            "spec_sha256": _sha256(registry_builder.DEFAULT_SEARCH_SPEC),
            "primary_metric": "RGSPL",
            "source_sr_floor_tolerance_percentage_points": 1.0,
            "records": records,
        }

    def setUp(self):
        ignored_root = REPO_ROOT / "vln/results/logs/reverie"
        ignored_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ignored_root)
        self.root = Path(self.temporary.name)
        self.search_path = self.root / "FINAL_SELECTION.json"
        _write_json(self.search_path, self.search_selection)
        search_spec = json.loads(
            registry_builder.DEFAULT_SEARCH_SPEC.read_text(encoding="utf-8")
        )
        _write_json(self.root / "BATCH.json", {
            "schema": "navtta.vln_reverie_small_search_batch.v1",
            "batch_id": "unit-test-reverie-search-seed0",
            "experiment_id": search_spec["experiment_id"],
            "spec_path": str(registry_builder.DEFAULT_SEARCH_SPEC),
            "spec_sha256": _sha256(registry_builder.DEFAULT_SEARCH_SPEC),
            "git_commit": self.search_selection["git_commit"],
            "gpu": 0,
            "source_registry_sha256": search_spec["dependencies"][
                "source_registry"
            ]["sha256"],
            "incumbent_spec_sha256": search_spec["dependencies"][
                "incumbent_spec"
            ]["sha256"],
        })
        self.output = self.root / "final"
        self.selected_path, self.registry_path, self.registry = self._freeze(
            self.search_path, self.output
        )
        self.spec = plan_builder.build_spec(self.registry_path)
        self.spec_path = self.root / "reverie_val_unseen.json"
        _write_json(self.spec_path, self.spec)

    def tearDown(self):
        self.temporary.cleanup()

    def _full_candidates(self):
        search_spec = json.loads(
            registry_builder.DEFAULT_SEARCH_SPEC.read_text(encoding="utf-8")
        )
        candidates = {}
        for method in registry_builder.METHODS:
            for setting in search_spec["methods"][method][
                "candidates_by_setting"
            ]:
                incumbent = self.search_selection["records"][setting][method]
                candidates[(setting, method)] = {
                    "run_tag": "synthetic-unused-full-{}-{}".format(setting, method),
                    "parameters": incumbent["parameters"],
                    "metrics": {key: 0.0 for key in registry_builder.METRICS},
                    "formal_manifest_path": "unused",
                    "formal_manifest_sha256": "c" * 64,
                }
        return candidates

    def _freeze(self, selection_path, output):
        with mock.patch.object(
            registry_builder,
            "_load_full_candidates",
            return_value=self._full_candidates(),
        ):
            return registry_builder.freeze_from_search(selection_path, output)

    def test_offline_registry_is_complete_and_reproducible(self):
        self.assertEqual(self.registry["registry_status"], "complete")
        self.assertEqual(
            sum(len(value) for value in self.registry["records"].values()), 18
        )
        rebuilt = registry_builder.validate_registry(
            self.registry_path, self.selected_path
        )
        self.assertEqual(rebuilt, self.registry)
        for setting in registry_builder.SETTINGS:
            source_metrics = self.registry["records"][setting]["source"]["metrics"]
            for method in registry_builder.METHODS:
                record = self.registry["records"][setting][method]
                self.assertEqual(
                    record["parameters"],
                    self.search_selection["records"][setting][method]["parameters"],
                )
                self.assertEqual(
                    record["delta_vs_source_pp"],
                    {
                        key: round(record["metrics"][key] - source_metrics[key], 6)
                        for key in registry_builder.METRICS
                    },
                )

    def test_registry_rejects_tampered_winner_manifest_digest(self):
        bad = deepcopy(self.search_selection)
        bad["records"]["duet-reverie"]["tent"][
            "formal_manifest_sha256"
        ] = "0" * 64
        bad_path = self.root / "bad.json"
        _write_json(bad_path, bad)
        with self.assertRaisesRegex(
            registry_builder.RegistryError, "formal manifest SHA256 mismatch"
        ):
            self._freeze(bad_path, self.root / "bad-output")

    def test_registry_rejects_forged_metrics_and_parameters(self):
        bad_metrics = deepcopy(self.search_selection)
        record = bad_metrics["records"]["duet-reverie"]["tent"]
        record["metrics"]["RGSPL"] += 1.0
        record["delta_vs_source_pp"]["RGSPL"] += 1.0
        metrics_path = self.root / "bad-metrics.json"
        _write_json(metrics_path, bad_metrics)
        with self.assertRaisesRegex(
            registry_builder.RegistryError,
            "pinned incumbent|authenticated valid.txt",
        ):
            self._freeze(
                metrics_path, self.root / "bad-metrics-output"
            )

        bad_parameters = deepcopy(self.search_selection)
        bad_parameters["records"]["duet-reverie"]["tent"]["parameters"][
            "lr"
        ] *= 2
        parameters_path = self.root / "bad-parameters.json"
        _write_json(parameters_path, bad_parameters)
        with self.assertRaisesRegex(
            registry_builder.RegistryError,
            "pinned incumbent|config_overrides",
        ):
            self._freeze(
                parameters_path, self.root / "bad-parameters-output"
            )

        unknown = deepcopy(self.search_selection)
        unknown["records"]["duet-reverie"]["tent"]["parameters"][
            "bogus"
        ] = 1
        unknown_path = self.root / "unknown-parameter.json"
        _write_json(unknown_path, unknown)
        with self.assertRaisesRegex(
            registry_builder.RegistryError, "unknown parameters"
        ):
            self._freeze(
                unknown_path, self.root / "unknown-output"
            )

    def test_registry_recomputes_full_candidate_selection_policy(self):
        bad = deepcopy(self.search_selection)
        bad["records"]["duet-reverie"]["tent"]["origin"] = "new_full_candidate"
        bad["records"]["duet-reverie"]["tent"]["new_candidate_eligible"] = True
        path = self.root / "wrong-origin.json"
        _write_json(path, bad)
        with self.assertRaisesRegex(
            registry_builder.RegistryError, "selection policy"
        ):
            self._freeze(path, self.root / "wrong-origin-output")

    def test_source_ledgers_recover_existing_runs_without_rerun(self):
        _, unseen = plan_builder.validate_val_unseen_source(
            plan_builder.DEFAULT_SOURCE, require_metric_artifacts=True
        )
        _, test = plan_builder.validate_test_source(
            plan_builder.DEFAULT_TEST_SOURCE, require_submission_artifacts=True
        )
        self.assertEqual(unseen["source_execution_policy"]["execution"], "reuse_only")
        self.assertTrue(unseen["source_execution_policy"]["rerun_forbidden"])
        self.assertEqual(
            test["source_execution_policy"]["execution"],
            "reuse_existing_submission_only",
        )
        self.assertEqual(set(unseen["records"]), set(registry_builder.SETTINGS))
        self.assertEqual(set(test["records"]), set(registry_builder.SETTINGS))

        tampered = deepcopy(unseen)
        tampered["records"]["duet-reverie"]["metrics"]["RGSPL"] = 99.99
        tampered_path = self.root / "tampered-source.json"
        _write_json(tampered_path, tampered)
        with self.assertRaisesRegex(plan_builder.PlanError, "authenticated valid.txt"):
            plan_builder.validate_val_unseen_source(
                tampered_path, require_metric_artifacts=True
            )

    def test_generated_val_unseen_plan_is_exact_model_major_matrix(self):
        jobs = runner.expand_jobs(self.spec, "unit-test-batch", gpu=0)
        self.assertEqual(self.spec["budget"], {
            "source_execution_jobs": 0,
            "tta_jobs": 15,
            "total_executed_jobs": 15,
        })
        self.assertEqual(
            [(item["setting"], item["method"]) for item in jobs],
            [
                (setting, method)
                for setting in runner.SETTINGS
                for method in runner.METHODS
            ],
        )
        self.assertNotIn("source", {item["method"] for item in jobs})
        phases = runner.model_phases(jobs)
        self.assertEqual([phase[0]["model"] for phase in phases], list(runner.MODELS))
        self.assertEqual(
            self.spec["execution"]["max_workers_by_model"],
            {"duet": 2, "hamt": 2, "goat": 4},
        )

    def test_val_unseen_commands_use_full_split_seed0_and_frozen_parameters(self):
        jobs = runner.expand_jobs(self.spec, "unit-test-batch", gpu=0)
        for job in jobs:
            attempt, metadata = runner.materialize_attempt(
                self.spec, self.spec_path, "unit-test-batch", self.root / "batch",
                job, 0,
            )
            config = json.loads((attempt / "parameters.json").read_text())
            self.assertEqual(config["stage"], "frozen_val_unseen")
            self.assertEqual(config["episodes"], -1)
            self.assertNotIn("order_seed", config)
            self.assertEqual(config["parameters"], job["parameters"])
            method, tokens = translate(
                job["setting"], attempt / "parameters.json", attempt / "diag.json"
            )
            self.assertEqual(method, job["method"])
            self.assertIn("--tta_method", tokens)
            command = metadata["command"]
            self.assertEqual(command[1:3], [job["setting"], "val_unseen"])
            self.assertNotIn("--order-seed", command)
            self.assertNotIn("--episode-limit", command)
            self.assertEqual(metadata["canonical_order_seed"], 0)

        first = jobs[0]
        attempt, _ = runner.materialize_attempt(
            self.spec, self.spec_path, "binding-test", self.root / "binding-batch",
            first, 0,
        )
        metadata_path = attempt / "job.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["base_run_tag"] = "forged-plan-job"
        _write_json(metadata_path, metadata)
        with self.assertRaisesRegex(runner.UserError, "planned job"):
            runner._validate_attempt_binding(
                attempt, metadata, expected_job=first,
                expected_batch_id="binding-test", expected_spec_path=self.spec_path,
            )

    def test_hidden_test_is_submission_only_and_feedback_methods_fail_closed(self):
        policy = self.spec["test_submission_policy"]
        self.assertTrue(policy["hidden_ground_truth"])
        self.assertTrue(policy["local_metric_computation_forbidden"])
        self.assertTrue(policy["selection_or_ranking_on_test_forbidden"])
        self.assertEqual(policy["execution_purpose"], "submission_generation_only")
        self.assertEqual(policy["eligible_tta_methods"], ["tent", "fstta", "eam"])
        for method in policy["eligible_tta_methods"]:
            self.assertTrue(plan_builder.assert_test_method_allowed(policy, method))
        for method in ("feedtta", "atena"):
            self.assertEqual(
                policy["unavailable_without_legal_online_feedback"][method]["status"],
                "N/A",
            )
            with self.assertRaisesRegex(plan_builder.PlanError, "N/A"):
                plan_builder.assert_test_method_allowed(policy, method)
        self.assertEqual(policy["source"]["execution"], "reuse_existing_submission_only")

        jobs = runner.expand_test_jobs(self.spec, "unit-test-submit", gpu=0)
        self.assertEqual(len(jobs), 9)
        self.assertEqual(
            [(item["setting"], item["method"]) for item in jobs],
            [
                (setting, method)
                for setting in runner.SETTINGS
                for method in ("tent", "fstta", "eam")
            ],
        )
        for job in jobs:
            attempt, metadata = runner.materialize_test_attempt(
                self.spec, self.spec_path, "unit-test-submit",
                self.root / "test-batch", job, 0,
            )
            config = json.loads((attempt / "parameters.json").read_text())
            self.assertEqual(config["stage"], "frozen_test_submission")
            self.assertTrue(
                config["frozen_evaluation_provenance"]["hidden_ground_truth"]
            )
            self.assertTrue(
                config["frozen_evaluation_provenance"][
                    "submission_generation_only"
                ]
            )
            self.assertEqual(metadata["command"][1:3], [job["setting"], "test"])
            self.assertNotIn("--order-seed", metadata["command"])
            self.assertNotIn("--episode-limit", metadata["command"])
            method, _ = translate(
                job["setting"], attempt / "parameters.json", attempt / "diag.json"
            )
            self.assertEqual(method, job["method"])

        batch_root = self.root / "test-batch-binding"
        runner._prepare_batch(
            self.spec_path, self.spec, "unit-test-submit", batch_root, 0,
            resume=False, mode="test_submission",
        )
        batch = json.loads((batch_root / "BATCH.json").read_text())
        self.assertNotIn("source_ledger", batch)
        self.assertEqual(
            batch["test_source_submission_ledger"], policy["source"]
        )
        self.assertEqual(batch["tta_jobs"], 9)

    def test_retry_tags_are_new_and_model_phases_are_strict(self):
        jobs = runner.expand_jobs(self.spec, "retry-test", gpu=0)
        self.assertNotEqual(
            runner._attempt_tag(jobs[0]["base_run_tag"], 0),
            runner._attempt_tag(jobs[0]["base_run_tag"], 1),
        )
        phases = runner.model_phases(jobs)
        for first, second in zip(phases, phases[1:]):
            self.assertLess(
                max(item["ordinal"] for item in first),
                min(item["ordinal"] for item in second),
            )

    def test_plan_rejects_feedback_method_enabled_on_hidden_test(self):
        bad = deepcopy(self.spec)
        bad["test_submission_policy"]["eligible_tta_methods"].append("feedtta")
        with self.assertRaisesRegex(plan_builder.PlanError, "fail-closed"):
            plan_builder.validate_spec_document(bad)


if __name__ == "__main__":
    unittest.main()
