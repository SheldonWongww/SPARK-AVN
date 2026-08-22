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
import shared_gpu_launch_guard as gpu_guard  # noqa: E402
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
            self.assertEqual(command, [
                str(runner.RUNNER), job["setting"], "val_unseen", "0",
                "--run-tag", metadata["run_tag"], "--tta-config",
                str(attempt / "parameters.json"),
            ])
            self.assertNotIn("--order-seed", command)
            self.assertNotIn("--episode-limit", command)
            self.assertNotIn("--result-root", command)
            self.assertEqual(metadata["canonical_order_seed"], 0)
            self.assertEqual(
                Path(metadata["result_root"]),
                REPO_ROOT / "vln/results/tuning" / metadata["run_tag"]
                / job["setting"] / "val_unseen",
            )

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
            self.assertEqual(metadata["command"], [
                str(runner.RUNNER), job["setting"], "test", "0",
                "--run-tag", metadata["run_tag"], "--tta-config",
                str(attempt / "parameters.json"),
            ])
            self.assertNotIn("--order-seed", metadata["command"])
            self.assertNotIn("--episode-limit", metadata["command"])
            self.assertNotIn("--result-root", metadata["command"])
            self.assertEqual(
                Path(metadata["result_root"]),
                REPO_ROOT / "vln/results/tuning" / metadata["run_tag"]
                / job["setting"] / "test",
            )
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

    def test_attempts_get_unique_process_tokens_and_worker_exports_them(self):
        job = runner.expand_jobs(self.spec, "token-test", gpu=0)[0]
        first, first_metadata = runner.materialize_attempt(
            self.spec, self.spec_path, "token-test", self.root / "token-batch",
            job, 0,
        )
        second, second_metadata = runner.materialize_attempt(
            self.spec, self.spec_path, "token-test", self.root / "token-batch",
            job, 1,
        )
        self.assertEqual(
            first_metadata["process_group_token"],
            runner._expected_process_group_token(first_metadata, first),
        )
        self.assertNotEqual(
            first_metadata["process_group_token"],
            second_metadata["process_group_token"],
        )
        worker = runner._write_worker(first, first_metadata)
        source = worker.read_text(encoding="utf-8")
        self.assertIn(
            "export {}={}".format(
                runner.PROCESS_GROUP_ENV,
                first_metadata["process_group_token"],
            ),
            source,
        )
        self.assertIn("printf '%s\\n' \"$$\"", source)
        self.assertLess(source.index("export "), source.index("launch_authorized"))

    def test_launch_injects_token_and_requires_session_leader(self):
        job = runner.expand_jobs(self.spec, "launch-test", gpu=0)[0]
        attempt, metadata = runner.materialize_attempt(
            self.spec, self.spec_path, "launch-test", self.root / "launch-batch",
            job, 0,
        )
        process = mock.Mock(pid=123)
        identity = {
            "pid": 123, "start_token": "proc:1", "cmdline_sha256": "a" * 64,
        }
        with mock.patch.object(
            runner.subprocess, "Popen", return_value=process
        ) as popen, mock.patch.object(
            runner, "process_identity", return_value=identity
        ), mock.patch.object(
            runner, "_identity_is_group_member", return_value=True
        ):
            launched, handle = runner.launch_attempt(attempt, metadata)
            handle.close()
        self.assertIs(launched, process)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(
            popen.call_args.kwargs["env"][runner.PROCESS_GROUP_ENV],
            metadata["process_group_token"],
        )

    def test_proc_scan_requires_token_pgid_sid_and_non_zombie(self):
        token = "e" * 64
        child = {
            "pid": 777, "start_token": "python", "cmdline_sha256": "f" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            entries = (
                ("777", "S", 401, 401, token),
                ("778", "S", 402, 401, token),
                ("779", "S", 401, 402, token),
                ("780", "Z", 401, 401, token),
                ("781", "S", 401, 401, "d" * 64),
            )
            for pid, state, pgid, sid, process_token in entries:
                item = proc / pid
                item.mkdir()
                (item / "stat").write_text(
                    "{} (python worker) {} 1 {} {} 0 0 0\n".format(
                        pid, state, pgid, sid
                    ),
                    encoding="utf-8",
                )
                (item / "environ").write_bytes(
                    "{}={}".format(
                        runner.PROCESS_GROUP_ENV, process_token
                    ).encode("ascii") + b"\0"
                )
            with mock.patch.object(
                runner, "process_identity",
                side_effect=lambda pid: child if pid == 777 else None,
            ), mock.patch.object(
                runner, "process_identity_alive", return_value=True
            ):
                self.assertEqual(
                    runner._proc_group_member_identities(401, token, proc),
                    [child],
                )

    def test_exitcode_does_not_hide_live_descendant(self):
        wrapper = {
            "pid": 401, "start_token": "wrapper", "cmdline_sha256": "a" * 64,
        }
        descendant = {
            "pid": 777, "start_token": "python", "cmdline_sha256": "b" * 64,
        }
        attempt = self.root / "descendant-attempt"
        attempt.mkdir()
        metadata = {
            "batch_id": "batch", "run_tag": "run", "attempt": 0,
            "setting": "duet-reverie", "method": "tent",
            "spec_sha256": "c" * 64, "git_commit": "d" * 40,
        }
        metadata["process_group_token"] = runner._expected_process_group_token(
            metadata, attempt
        )
        _write_json(attempt / "job.json", metadata)
        _write_json(attempt / "process_identity.json", wrapper)
        (attempt / "pid").write_text("401\n", encoding="utf-8")
        (attempt / "exitcode").write_text("0\n", encoding="utf-8")
        with mock.patch.object(
            runner, "process_identity_alive", return_value=False
        ), mock.patch.object(
            runner, "_proc_group_member_identities", return_value=[descendant]
        ):
            self.assertEqual(runner._state_for_attempt(attempt), "running")
            self.assertEqual(runner._live_worker_identities(attempt), [descendant])

    def test_live_descendant_is_not_released_or_retried(self):
        job = runner.expand_jobs(self.spec, "live-test", gpu=0)[0]
        attempt = self.root / "live-attempt"
        attempt.mkdir()
        _write_json(attempt / "job.json", {"submission_generation_only": False})
        batch_root = self.root / "live-batch"
        _write_json(batch_root / "BATCH.json", {
            "git_commit": "a" * 40,
            "spec_sha256": runner._sha256(self.spec_path),
        })
        with mock.patch.object(
            runner, "_assert_clean_execution_tree"
        ), mock.patch.object(
            runner, "_git_commit", return_value="a" * 40
        ), mock.patch.object(
            runner, "_latest_attempt", return_value=(0, attempt)
        ), mock.patch.object(
            runner, "_recover_worker_identity", return_value=True
        ), mock.patch.object(
            runner, "_state_for_attempt", return_value="running"
        ), mock.patch.object(
            runner, "_validate_attempt_binding"
        ), mock.patch.object(
            runner, "_claim_running_reservation",
            side_effect=runner.UserError("audit stop after claim"),
        ) as claim, mock.patch.object(
            runner, "_release"
        ) as release:
            with self.assertRaisesRegex(runner.UserError, "audit stop"):
                runner.run_model_phase(
                    self.spec, self.spec_path, "live-test", batch_root, [job],
                    retry_failed=True,
                )
        claim.assert_called_once()
        release.assert_not_called()

    def test_cleanup_terminates_descendant_and_retains_failed_reservation(self):
        descendant = {
            "pid": 777, "start_token": "python", "cmdline_sha256": "b" * 64,
        }
        with mock.patch.object(
            runner, "_cleanup_process_groups",
            side_effect=[([descendant], {401}), ([], set())],
        ), mock.patch.object(runner.os, "killpg") as killpg:
            self.assertTrue(
                runner._terminate_attempt_process_group(Path("attempt"), None)
            )
        killpg.assert_called_once_with(401, runner.signal.SIGTERM)

        identity = {
            "pid": 777, "start_token": "python", "cmdline_sha256": "c" * 64,
        }
        ledger = mock.Mock()
        ledger.document = {"reservations": {"reservation": {"owners": []}}}
        process = mock.Mock(pid=777)
        process.poll.return_value = None
        with mock.patch.object(
            runner, "_terminate_attempt_process_group", return_value=False
        ), mock.patch.object(
            runner, "_live_worker_identities", return_value=[identity]
        ), mock.patch.object(
            runner, "process_identity", return_value=identity
        ), mock.patch.object(
            runner, "process_identity_alive", return_value=True
        ):
            with self.assertRaisesRegex(runner.UserError, "reservation retained"):
                runner._rollback_failed_launch(
                    ledger, "reservation", process, Path("attempt")
                )
        ledger.release.assert_not_called()
        ledger._write.assert_called_once_with()
        ledger.reset_mock()
        with mock.patch.object(
            runner, "_terminate_attempt_process_group", return_value=True
        ):
            runner._rollback_failed_launch(
                ledger, "reservation", process, Path("attempt")
            )
        ledger.release.assert_called_once_with("reservation")

    def test_release_refuses_live_descendants(self):
        job = {
            "gpu": 0,
            "base_run_tag": "candidate",
        }
        with mock.patch.object(
            runner, "_pid_alive", return_value=True
        ), mock.patch.object(
            runner, "release_shared_gpu_reservation"
        ) as release:
            with self.assertRaisesRegex(runner.UserError, "descendants are alive"):
                runner._release("batch", job, 0, Path("attempt"))
        release.assert_not_called()

        with mock.patch.object(
            runner, "_pid_alive", return_value=False
        ), mock.patch.object(
            runner, "release_shared_gpu_reservation"
        ) as release:
            runner._release("batch", job, 0, Path("attempt"))
        release.assert_called_once_with(
            0, runner._reservation_token("batch", "candidate")
        )

    def test_resumed_descendant_reclaims_reservation_and_authorizes(self):
        worker = {
            "pid": 101, "start_token": "worker", "cmdline_sha256": "a" * 64,
        }
        scheduler = {
            "pid": 202, "start_token": "scheduler", "cmdline_sha256": "b" * 64,
        }

        class Ledger:
            def __init__(self):
                self.document = {"reservations": {}}

            def reserve(self, token, **kwargs):
                self.document["reservations"][token] = {
                    "gpu_memory_mib": kwargs["gpu_memory_mib"],
                    "cgroup_memory_gib": kwargs["cgroup_memory_gib"],
                    "owners": [kwargs["owner"]],
                    "metadata": kwargs["metadata"],
                }

            def add_owner(self, token, owner):
                self.document["reservations"][token]["owners"].append(owner)

        ledger = Ledger()

        class Guard:
            def __enter__(self):
                return ledger

            def __exit__(self, *_args):
                return False

        spec = {"execution": {"estimated_gpu_memory_mib_by_method": {
            "tent": 4096,
        }}}
        job = {"gpu": 0, "method": "tent"}
        metadata = {"run_tag": "live-worker"}
        with mock.patch.object(
            runner, "_live_worker_identities", return_value=[worker]
        ), mock.patch.object(
            runner, "process_identity", return_value=scheduler
        ), mock.patch.object(
            runner, "shared_gpu_launch_guard", return_value=Guard()
        ), mock.patch.object(
            runner, "_gpu_memory_mib", return_value=(32000, 1000, 31000)
        ), mock.patch.object(runner, "_authorize_launch") as authorize:
            token = runner._claim_running_reservation(
                spec, "batch", job, Path("attempt"), metadata
            )
        self.assertEqual(
            ledger.document["reservations"][token]["owners"],
            [scheduler, worker],
        )
        authorize.assert_called_once_with(Path("attempt"))

    def test_retained_descendant_survives_scheduler_owner_pruning(self):
        scheduler = {
            "pid": 202, "start_token": "scheduler", "cmdline_sha256": "a" * 64,
        }
        descendant = {
            "pid": 303, "start_token": "worker", "cmdline_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            gpu_guard, "process_identity_alive",
            side_effect=lambda identity: identity.get("pid") == 303,
        ):
            ledger = gpu_guard.ReservationLedger(0, directory)
            ledger.reserve(
                "reservation", gpu_memory_mib=4096, cgroup_memory_gib=0.0,
                observed_gpu_memory_mib=100, observed_cgroup_memory_gib=0.0,
                owner=scheduler, metadata={"role": "reverie_val_unseen"},
            )
            with mock.patch.object(
                runner, "_live_worker_identities", return_value=[descendant]
            ):
                runner._retain_reservation_owners(
                    ledger, "reservation", Path("attempt")
                )
            reloaded = gpu_guard.ReservationLedger(0, directory)
            self.assertIn("reservation", reloaded.document["reservations"])
            self.assertEqual(
                reloaded.document["reservations"]["reservation"]["owners"][-1],
                descendant,
            )

    def test_launch_owner_failure_cleans_before_reservation_release(self):
        job = runner.expand_jobs(self.spec, "launch-failure", gpu=0)[0]
        batch_root = self.root / "launch-failure-batch"
        _write_json(batch_root / "BATCH.json", {
            "git_commit": "a" * 40,
            "spec_sha256": runner._sha256(self.spec_path),
        })
        attempt = self.root / "launch-failure-attempt"
        metadata = {"run_tag": job["base_run_tag"], "attempt": 0}
        process = mock.Mock(pid=303)
        process.poll.return_value = None
        handle = mock.Mock()
        ledger = mock.Mock()
        worker = {
            "pid": 303, "start_token": "worker", "cmdline_sha256": "b" * 64,
        }
        ledger.document = {"reservations": {
            runner._reservation_token("launch-failure", job["base_run_tag"]): {
                "owners": [],
            },
        }}
        ledger.snapshot.return_value = {"effective_gpu_memory_mib": 0}
        def fail_after_mutation(token, identity):
            ledger.document["reservations"][token]["owners"].append(identity)
            raise runner.UserError("owner write failed")

        ledger.add_owner.side_effect = fail_after_mutation
        _write_json(attempt / "process_identity.json", worker)

        class Guard:
            def __enter__(self):
                return ledger

            def __exit__(self, *_args):
                return False

        with mock.patch.object(
            runner, "_assert_clean_execution_tree"
        ), mock.patch.object(
            runner, "_git_commit", return_value="a" * 40
        ), mock.patch.object(
            runner, "_latest_attempt", return_value=None
        ), mock.patch.object(
            runner, "_gpu_memory_mib", return_value=(32000, 0, 32000)
        ), mock.patch.object(
            runner, "process_identity",
            side_effect=lambda pid=None: ({
                "pid": 1, "start_token": "scheduler", "cmdline_sha256": "a" * 64,
            } if pid is None else worker),
        ), mock.patch.object(
            runner, "shared_gpu_launch_guard", return_value=Guard()
        ), mock.patch.object(
            runner, "materialize_attempt", return_value=(attempt, metadata)
        ), mock.patch.object(
            runner, "launch_attempt", return_value=(process, handle)
        ), mock.patch.object(
            runner, "_terminate_attempt_process_group", return_value=False
        ) as terminate, mock.patch.object(
            runner, "_live_worker_identities", return_value=[worker]
        ), mock.patch.object(
            runner, "process_identity_alive", return_value=True
        ):
            with self.assertRaisesRegex(runner.UserError, "reservation retained"):
                runner.run_model_phase(
                    self.spec, self.spec_path, "launch-failure", batch_root,
                    [job],
                )
        terminate.assert_called_once_with(attempt, process)
        handle.close.assert_called_once()
        ledger.release.assert_not_called()
        ledger._write.assert_called_once_with()

    def test_plan_rejects_feedback_method_enabled_on_hidden_test(self):
        bad = deepcopy(self.spec)
        bad["test_submission_policy"]["eligible_tta_methods"].append("feedtta")
        with self.assertRaisesRegex(plan_builder.PlanError, "fail-closed"):
            plan_builder.validate_spec_document(bad)


if __name__ == "__main__":
    unittest.main()
