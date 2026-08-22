from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln/scripts"))

import run_reverie_small_hparam_search as runner  # noqa: E402
from tta_config_cli import translate  # noqa: E402


SPEC_PATH = (
    REPO_ROOT
    / "vln/experiments/reverie_val_seen_small_hparam_search_v1.json"
)


def _write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fake_promotions(spec):
    jobs = runner.expand_screening_jobs(spec, "test-batch")
    values = []
    seen = set()
    for job in jobs:
        cell = (job["setting"], job["method"])
        if cell in seen:
            continue
        seen.add(cell)
        values.append({
            "setting": job["setting"],
            "method": job["method"],
            "role": job["role"],
            "parameters": job["parameters"],
            "screening_run_tag": job["base_run_tag"],
        })
    return {"promotions": values}


class ReverieSmallHparamSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = runner.load_spec(SPEC_PATH)

    def test_budget_is_18_screening_plus_at_most_8_full(self):
        jobs = runner.expand_screening_jobs(self.spec, "test-batch")
        self.assertEqual(len(jobs), 18)
        cells = {(job["setting"], job["method"]) for job in jobs}
        self.assertEqual(len(cells), 8)
        self.assertNotIn(("duet-reverie", "eam"), cells)
        self.assertNotIn(("hamt-reverie", "atena"), cells)
        self.assertEqual(self.spec["budget"]["maximum_new_jobs"], 26)
        self.assertEqual(
            self.spec["budget"]["maximum_new_episode_evaluations"],
            15992,
        )

    def test_source_and_all_incumbents_are_reused_and_authenticated(self):
        source = runner._source_registry(self.spec)
        parameters = runner._incumbent_parameters(self.spec)
        incumbents = runner._validate_incumbents(
            self.spec, source, parameters
        )
        self.assertEqual(source["order_seed"], 0)
        self.assertEqual(
            source["episode_order_sha256"],
            self.spec["protocol"]["canonical_order_sha256"],
        )
        self.assertEqual(len(parameters), 15)
        self.assertEqual(len(incumbents), 15)

    def test_source_and_incumbent_metric_literals_cannot_override_artifacts(self):
        source = runner._source_registry(self.spec)
        parameters = runner._incumbent_parameters(self.spec)
        bad = deepcopy(self.spec)
        bad["incumbents"]["goat-reverie"]["atena"]["metrics"]["RGSPL"] = 99.99
        with self.assertRaisesRegex(
            runner.UserError, "differs from authenticated artifact"
        ):
            runner._validate_incumbents(bad, source, parameters)

        source_document = json.loads(
            (REPO_ROOT / self.spec["dependencies"]["source_registry"]["path"])
            .read_text(encoding="utf-8")
        )
        source_document["records"]["duet-reverie"]["metrics"]["RGSPL"] = 99.99
        with tempfile.TemporaryDirectory(
            dir=REPO_ROOT / "vln/results/logs"
        ) as directory:
            source_path = Path(directory) / "source.json"
            _write_json(source_path, source_document)
            bad = deepcopy(self.spec)
            bad["dependencies"]["source_registry"] = {
                "path": str(source_path.relative_to(REPO_ROOT)),
                "sha256": runner._sha256(source_path),
            }
            with self.assertRaisesRegex(
                runner.UserError, "differs from authenticated artifact"
            ):
                runner._source_registry(bad)

    def test_execution_order_and_caps_are_model_wise(self):
        execution = self.spec["execution"]
        self.assertEqual(tuple(execution["setting_order"]), runner.SETTINGS)
        self.assertTrue(execution["strict_model_barriers_required"])
        self.assertTrue(execution["parallel_methods_within_model"])
        self.assertEqual(
            execution["model_phase_pipeline"],
            ["screening", "promotion", "full", "summary"],
        )
        self.assertEqual(execution["max_workers_per_method"], 1)
        self.assertEqual(
            execution["max_workers_by_setting"],
            {"duet-reverie": 2, "hamt-reverie": 2, "goat-reverie": 4},
        )

    def test_tent_frequency_and_feedback_protocol_are_fixed(self):
        jobs = runner.expand_screening_jobs(self.spec, "test-batch")
        for job in jobs:
            if job["method"] == "tent":
                self.assertEqual(job["parameters"]["update_interval"], 1)
            if job["method"] in ("feedtta", "atena"):
                self.assertEqual(job["parameters"]["action_selection"], "argmax")
            if job["method"] == "feedtta":
                self.assertEqual(job["parameters"]["sgr_seed"], 0)

    def test_screening_configs_translate_and_use_only_canonical_prefix(self):
        jobs = runner.expand_screening_jobs(self.spec, "test-batch")
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory)
            runner._prepare_batch(
                SPEC_PATH, self.spec, "test-batch", batch_root, 0, False
            )
            with mock.patch.object(runner, "_assert_clean_execution_tree"):
                for job in jobs:
                    attempt_dir, metadata = runner.materialize_attempt(
                        self.spec, SPEC_PATH, "test-batch", batch_root, job, 0
                    )
                    method, tokens = translate(
                        job["setting"],
                        attempt_dir / "parameters.json",
                        attempt_dir / "diagnostics.json",
                    )
                    self.assertEqual(method, job["method"])
                    self.assertIn("--tta_method", tokens)
                    self.assertEqual(
                        metadata["command"][-2:], ["--episode-limit", "256"]
                    )
                    self.assertNotIn("--order-seed", metadata["command"])
                    self.assertIsNone(metadata["formal_manifest"])
                    config = json.loads(
                        (attempt_dir / "parameters.json").read_text(encoding="utf-8")
                    )
                    self.assertNotIn("role", config["parameters"])
                    self.assertTrue(
                        config["selection_provenance"]["fresh_source_restart"]
                    )

    def test_promoted_full_jobs_are_fresh_and_formal(self):
        full = runner.expand_full_jobs(
            self.spec, "test-batch", _fake_promotions(self.spec)
        )
        self.assertEqual(len(full), 8)
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory)
            runner._prepare_batch(
                SPEC_PATH, self.spec, "test-batch", batch_root, 0, False
            )
            with mock.patch.object(runner, "_assert_clean_execution_tree"):
                for job in full:
                    attempt_dir, metadata = runner.materialize_attempt(
                        self.spec, SPEC_PATH, "test-batch", batch_root, job, 0
                    )
                    self.assertNotIn("--episode-limit", metadata["command"])
                    self.assertNotIn("--order-seed", metadata["command"])
                    self.assertTrue(metadata["formal_manifest"].endswith(
                        "-val_seen-native/manifest.json"
                    ))
                    config = json.loads(
                        (attempt_dir / "parameters.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(config["episodes"], -1)
                    self.assertIsNotNone(
                        config["selection_provenance"]["screening_run_tag"]
                    )

    def test_screening_promotion_is_one_per_cell_and_rgspl_first(self):
        jobs = runner.expand_screening_jobs(self.spec, "test-batch")[:2]
        base = {
            "setting": jobs[0]["setting"],
            "method": jobs[0]["method"],
            "parameters": jobs[0]["parameters"],
            "role": jobs[0]["role"],
            "adapter_diagnostics": {"relative_param_drift": 0.01, "updates": 8},
        }
        results = [
            {
                **base,
                "run_tag": "lower",
                "metrics": {"SR": 90, "SPL": 90, "RGS": 90, "RGSPL": 50},
            },
            {
                **base,
                "run_tag": "higher-rgspl",
                "role": jobs[1]["role"],
                "parameters": jobs[1]["parameters"],
                "metrics": {"SR": 60, "SPL": 60, "RGS": 60, "RGSPL": 51},
            },
        ]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "_stage_results", return_value=results
        ), mock.patch.object(runner, "_assert_clean_execution_tree"):
            runner._prepare_batch(
                SPEC_PATH, self.spec, "test-batch", Path(directory), 0, False
            )
            output = runner.promote_screening(
                SPEC_PATH, self.spec, Path(directory), jobs
            )
        self.assertEqual(len(output["promotions"]), 1)
        self.assertEqual(
            output["promotions"][0]["screening_run_tag"], "higher-rgspl"
        )

    def test_diagnostics_enforce_supervision_boundary(self):
        unsupervised = {
            "method": "fstta",
            "episode_count": 256,
            "action_selection": "target_native_argmax",
            "supervision": "unsupervised",
            "binary_feedback_endpoint": None,
            "adapter": {
                "episodes": 256,
                "updates": 12,
                "relative_param_drift": 0.01,
            },
        }
        runner._validate_diagnostics(
            {"method": "fstta", "episode_count": 256}, unsupervised
        )
        invalid = deepcopy(unsupervised)
        invalid["binary_feedback_endpoint"] = "distance_oracle"
        with self.assertRaisesRegex(runner.UserError, "unexpectedly consumed"):
            runner._validate_diagnostics(
                {"method": "fstta", "episode_count": 256}, invalid
            )

    def test_feedback_counts_are_nonnegative_complete_and_cross_checked(self):
        feed_metadata = {
            "method": "feedtta",
            "episode_count": 100,
            "parameters": {"action_selection": "argmax"},
        }
        feed = {
            "method": "feedtta",
            "episode_count": 100,
            "action_selection": "target_native_argmax",
            "supervision": "binary_navigation_success_feedback",
            "binary_feedback_endpoint": runner.EXPECTED_FEEDBACK_ENDPOINT["feedtta"],
            "adapter": {
                "episodes": 100,
                "updates": 100,
                "relative_param_drift": 0.01,
                "action_selection_protocol": "target_native_argmax",
                "feedback_episodes": 100,
                "successful_feedback_episodes": 60,
                "failed_feedback_episodes": 40,
            },
        }
        runner._validate_diagnostics(
            feed_metadata, feed, metrics={"SR": 60.0}
        )
        bad_feed = deepcopy(feed)
        bad_feed["adapter"]["successful_feedback_episodes"] = -1
        bad_feed["adapter"]["failed_feedback_episodes"] = 101
        with self.assertRaisesRegex(runner.UserError, "integer in"):
            runner._validate_diagnostics(
                feed_metadata, bad_feed, metrics={"SR": 60.0}
            )

        atena_metadata = {
            "method": "atena",
            "episode_count": 100,
            "parameters": {"query_threshold": 0.1},
        }
        atena = {
            "method": "atena",
            "episode_count": 100,
            "action_selection": "target_native_argmax",
            "supervision": "binary_navigation_success_feedback",
            "binary_feedback_endpoint": runner.EXPECTED_FEEDBACK_ENDPOINT["atena"],
            "adapter": {
                "episodes": 100,
                "updates": 100,
                "relative_param_drift": 0.01,
                "queries": 40,
                "self_label_episodes": 60,
                "feedback_observed_episodes": 40,
                "query_gate_evaluations": 100,
                "self_prediction_evaluations": 100,
                "queried_feedback_successes": 20,
                "self_feedback_successes": 30,
            },
        }
        runner._validate_diagnostics(atena_metadata, atena)
        for key, value in (
            ("queries", -1),
            ("feedback_observed_episodes", 39),
            ("query_gate_evaluations", 99),
            ("self_prediction_evaluations", 99),
        ):
            invalid_atena = deepcopy(atena)
            invalid_atena["adapter"][key] = value
            with self.assertRaises(runner.UserError, msg=key):
                runner._validate_diagnostics(atena_metadata, invalid_atena)

    def test_invalid_feedback_action_and_tent_frequency_are_rejected(self):
        bad = deepcopy(self.spec)
        bad["methods"]["feedtta"]["fixed"]["action_selection"] = "sample"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            _write_json(path, bad)
            with self.assertRaisesRegex(runner.UserError, "argmax"):
                runner.load_spec(path, require_evidence=False)
        bad = deepcopy(self.spec)
        bad["methods"]["tent"]["fixed"]["update_interval"] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            _write_json(path, bad)
            with self.assertRaisesRegex(runner.UserError, "fixed at 1"):
                runner.load_spec(path, require_evidence=False)

    def test_final_selection_never_replaces_incumbent_with_worse_candidate(self):
        promotions = _fake_promotions(self.spec)
        full_jobs = runner.expand_full_jobs(self.spec, "test-batch", promotions)
        fake_results = []
        for job in full_jobs:
            incumbent = self.spec["incumbents"][job["setting"]][job["method"]]
            metrics = dict(incumbent["metrics"])
            if (job["setting"], job["method"]) == ("goat-reverie", "atena"):
                metrics["RGSPL"] += 1.0
            elif (job["setting"], job["method"]) == (
                "hamt-reverie", "fstta"
            ):
                metrics["RGSPL"] += 10.0
                metrics["SR"] = 0.0
            else:
                metrics["RGSPL"] -= 1.0
            fake_results.append({
                **job,
                "run_tag": job["base_run_tag"],
                "metrics": metrics,
                "formal_manifest": "/tmp/formal.json",
                "formal_manifest_sha256": "a" * 64,
            })
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "_stage_results", return_value=fake_results
        ), mock.patch.object(runner, "_assert_clean_execution_tree"):
            runner._prepare_batch(
                SPEC_PATH, self.spec, "test-batch", Path(directory), 0, False
            )
            selection = runner.finalize_selection(
                SPEC_PATH, self.spec, Path(directory), full_jobs
            )
        self.assertEqual(
            selection["records"]["goat-reverie"]["atena"]["origin"],
            "new_full_candidate",
        )
        self.assertEqual(
            selection["records"]["hamt-reverie"]["fstta"]["origin"],
            "r2r_frozen_transfer_incumbent",
        )
        self.assertEqual(
            selection["records"]["duet-reverie"]["eam"]["origin"],
            "r2r_frozen_transfer_incumbent",
        )
        self.assertEqual(
            sum(len(methods) for methods in selection["records"].values()),
            15,
        )
        for setting in runner.SETTINGS:
            self.assertEqual(
                set(selection["records"][setting]), set(runner.METHODS)
            )

    def test_batch_binding_rejects_head_drift_on_resume_and_materialize(self):
        job = runner.expand_screening_jobs(self.spec, "test-batch")[0]
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory)
            runner._prepare_batch(
                SPEC_PATH, self.spec, "test-batch", batch_root, 0, False
            )
            with mock.patch.object(
                runner, "_assert_clean_execution_tree"
            ), mock.patch.object(runner, "_git_commit", return_value="b" * 40):
                with self.assertRaisesRegex(runner.UserError, "HEAD drifted"):
                    runner._prepare_batch(
                        SPEC_PATH, self.spec, "test-batch", batch_root, 0, True
                    )
                with self.assertRaisesRegex(runner.UserError, "HEAD drifted"):
                    runner.materialize_attempt(
                        self.spec, SPEC_PATH, "test-batch", batch_root, job, 0
                    )

    def test_model_phase_checks_batch_before_any_gpu_access(self):
        job = runner.expand_screening_jobs(self.spec, "test-batch")[0]
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory)
            runner._prepare_batch(
                SPEC_PATH, self.spec, "test-batch", batch_root, 0, False
            )
            with mock.patch.object(
                runner, "_assert_clean_execution_tree"
            ), mock.patch.object(
                runner, "_git_commit", return_value="c" * 40
            ), mock.patch.object(
                runner.frozen_runner, "_gpu_memory_mib"
            ) as gpu_query:
                with self.assertRaisesRegex(runner.UserError, "HEAD drifted"):
                    runner.run_model_phase(
                        self.spec, SPEC_PATH, "test-batch", batch_root, [job]
                    )
                gpu_query.assert_not_called()

    def test_formal_launch_inputs_must_be_tracked(self):
        success = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(runner.subprocess, "run", return_value=success) as run:
            runner._require_tracked_launch_inputs(SPEC_PATH)
        command = run.call_args.args[0]
        self.assertEqual(command[:5], [
            "git", "-C", str(REPO_ROOT), "ls-files", "--error-unmatch"
        ])
        self.assertIn("vln/scripts/run_reverie_small_hparam_search.py", command)
        self.assertIn(
            "vln/experiments/reverie_val_seen_small_hparam_search_v1.json",
            command,
        )

        failure = mock.Mock(returncode=1, stdout="", stderr="not tracked")
        with mock.patch.object(runner.subprocess, "run", return_value=failure):
            with self.assertRaisesRegex(runner.UserError, "tracked by current HEAD"):
                runner._require_tracked_launch_inputs(SPEC_PATH)

    def test_execution_tree_rejects_untracked_execution_surface(self):
        with mock.patch.object(
            runner, "_tracked_worktree_dirty", return_value=False
        ), mock.patch.object(
            runner, "_require_tracked_launch_inputs"
        ), mock.patch.object(
            runner,
            "_untracked_execution_files",
            return_value=["vln/scripts/uncommitted.py"],
        ):
            with self.assertRaisesRegex(
                runner.UserError, "untracked execution files"
            ):
                runner._assert_clean_execution_tree(SPEC_PATH)

    def test_status_reads_per_model_full_plan_before_global_promotions(self):
        promotion = {
            "promotions": [
                item for item in _fake_promotions(self.spec)["promotions"]
                if item["setting"] == "duet-reverie"
            ]
        }
        full = runner.expand_full_jobs(
            self.spec,
            "status-test",
            promotion,
            settings=("duet-reverie",),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "LOG_ROOT", Path(directory)
        ), mock.patch.object(
            runner, "_job_state",
            side_effect=lambda unused_root, job: (
                "running" if job["stage"] == "full" else "completed",
                0,
                Path("/tmp/attempt"),
            ),
        ), mock.patch("builtins.print") as printer:
            plan_path = (
                Path(directory) / "status-test" / "stages" / "full"
                / "duet-reverie" / "stage_plan.json"
            )
            plan_path.parent.mkdir(parents=True)
            _write_json(plan_path, {"jobs": full})
            runner.show_status(self.spec, "status-test")
        snapshot = json.loads(printer.call_args.args[0])
        self.assertEqual(
            snapshot["models"]["duet-reverie"]["full"]["running"], 2
        )
        self.assertEqual(snapshot["stages"]["full"]["running"], 2)

    def test_status_reports_validation_failure_as_invalid(self):
        batch_id = "status-invalid"
        job = runner.expand_screening_jobs(self.spec, batch_id)[0]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "LOG_ROOT", Path(directory)
        ), mock.patch("builtins.print") as printer:
            attempt = runner._attempt_dir(
                Path(directory) / batch_id, job, 0
            )
            attempt.mkdir(parents=True)
            _write_json(attempt / "metrics.json", {})
            (attempt / "exitcode").write_text("0\n", encoding="utf-8")
            _write_json(
                attempt / "validation_error.json", {"error": "bad artifact"}
            )
            runner.show_status(self.spec, batch_id)
        snapshot = json.loads(printer.call_args.args[0])
        self.assertEqual(snapshot["stages"]["screening"]["invalid"], 1)
        self.assertEqual(
            snapshot["models"][job["setting"]]["screening"]["invalid"], 1
        )
        self.assertEqual(snapshot["stages"]["screening"]["completed"], 0)

    def test_retry_archives_persisted_exit_zero_validation_failure(self):
        job = runner.expand_screening_jobs(self.spec, "test-batch")[0]
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory)
            runner._prepare_batch(
                SPEC_PATH, self.spec, "test-batch", batch_root, 0, False
            )
            with mock.patch.object(runner, "_assert_clean_execution_tree"):
                attempt0, _ = runner.materialize_attempt(
                    self.spec, SPEC_PATH, "test-batch", batch_root, job, 0
                )
            (attempt0 / "exitcode").write_text("0\n", encoding="utf-8")
            (attempt0 / "metrics.json").write_text("{}\n", encoding="utf-8")

            ledger = mock.MagicMock()
            ledger.snapshot.return_value = {
                "effective_gpu_memory_mib": 0,
                "effective_cgroup_memory_gib": 0.0,
            }

            def launch(attempt_dir, metadata):
                _write_json(attempt_dir / "process_identity.json", {
                    "pid": 123,
                    "start_token": "proc:1",
                    "cmdline_sha256": "a" * 64,
                })
                (attempt_dir / "exitcode").write_text("0\n", encoding="utf-8")
                return mock.Mock(pid=123), mock.Mock()

            guard = mock.MagicMock()
            guard.return_value.__enter__.return_value = ledger
            with mock.patch.object(
                runner, "_assert_execution_invariants"
            ), mock.patch.object(
                runner, "_release_job_reservation"
            ), mock.patch.object(
                runner, "shared_gpu_launch_guard", guard
            ), mock.patch.object(
                runner.frozen_runner,
                "_gpu_memory_mib",
                return_value=(32768, 0, 32768),
            ), mock.patch.object(
                runner.frozen_runner, "_cgroup_memory_gib", return_value=0.0
            ), mock.patch.object(
                runner.frozen_runner, "launch_attempt", side_effect=launch
            ), mock.patch.object(
                runner,
                "validate_attempt",
                side_effect=[runner.UserError("bad artifact"), {}],
            ):
                with self.assertRaisesRegex(
                    runner.UserError, "finished but validation failed"
                ):
                    runner.run_model_phase(
                        self.spec,
                        SPEC_PATH,
                        "test-batch",
                        batch_root,
                        [job],
                        retry_failed=False,
                    )
                self.assertTrue((attempt0 / "validation_error.json").is_file())
                self.assertFalse((attempt0 / "archived_evidence.json").exists())
                runner.run_model_phase(
                    self.spec,
                    SPEC_PATH,
                    "test-batch",
                    batch_root,
                    [job],
                    retry_failed=True,
                )
            self.assertTrue((attempt0 / "archived_evidence.json").is_file())
            self.assertTrue((attempt0 / "invalid_evidence/metrics.json").is_file())
            self.assertTrue(
                runner._attempt_dir(batch_root, job, 1).joinpath("job.json").is_file()
            )

    def test_all_stage_runs_screening_and_full_model_major(self):
        calls = []

        def run_phase(unused_spec, unused_path, unused_batch, unused_root, jobs,
                      retry_failed=False):
            del unused_spec, unused_path, unused_batch, unused_root, retry_failed
            calls.append((jobs[0]["stage"], jobs[0]["setting"]))

        def promote(unused_path, unused_spec, unused_root, jobs, output_path=None):
            del unused_path, unused_spec, unused_root, output_path
            return _fake_promotions(self.spec) if len({
                job["setting"] for job in jobs
            }) > 1 else {
                "promotions": [
                    item for item in _fake_promotions(self.spec)["promotions"]
                    if item["setting"] == jobs[0]["setting"]
                ]
            }

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "LOG_ROOT", Path(directory)
        ), mock.patch.object(
            runner, "_assert_clean_execution_tree"
        ), mock.patch.object(
            runner, "run_model_phase", side_effect=run_phase
        ), mock.patch.object(
            runner, "_stage_summary", return_value={"complete": True}
        ), mock.patch.object(
            runner, "promote_screening", side_effect=promote
        ), mock.patch.object(
            runner, "finalize_selection"
        ):
            runner.main([
                "--stage", "all",
                "--batch-id", "model-major-test",
                "--confirm-reviewed",
            ])
        self.assertEqual(calls, [
            ("screening", "duet-reverie"),
            ("full", "duet-reverie"),
            ("screening", "hamt-reverie"),
            ("full", "hamt-reverie"),
            ("screening", "goat-reverie"),
            ("full", "goat-reverie"),
        ])

    def test_cached_full_result_is_reauthenticated_before_selection(self):
        job = runner.expand_full_jobs(
            self.spec, "test-batch", _fake_promotions(self.spec)
        )[0]
        authenticated = {
            **job,
            "run_tag": job["base_run_tag"],
            "metrics": {"SR": 1, "SPL": 1, "RGS": 1, "RGSPL": 1},
        }
        with mock.patch.object(
            runner,
            "_job_state",
            return_value=("completed", 0, Path("/tmp/cached-attempt")),
        ), mock.patch.object(
            runner, "validate_attempt", return_value=authenticated
        ) as validate:
            results = runner._stage_results(Path("/tmp/batch"), [job])
        self.assertEqual(results, [authenticated])
        validate.assert_called_once_with(Path("/tmp/cached-attempt"))

    def test_full_validation_delegates_manifest_and_artifact_authentication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = root / "attempt-00"
            result_root = root / "result"
            result_root.mkdir(parents=True)
            attempt.mkdir(parents=True)
            diagnostics_path = result_root / "tta_diagnostics.json"
            metric_path = result_root / "logs" / "valid.txt"
            metric_path.parent.mkdir()
            _write_json(diagnostics_path, {
                "method": "fstta",
                "episode_count": 2,
                "action_selection": "target_native_argmax",
                "supervision": "unsupervised",
                "binary_feedback_endpoint": None,
                "adapter": {
                    "episodes": 2,
                    "updates": 1,
                    "relative_param_drift": 0.01,
                },
            })
            metric_path.write_text(
                "Env name: val_seen, sr: 1.0, spl: 2.0, "
                "rgs: 3.0, rgspl: 4.0\n",
                encoding="utf-8",
            )
            formal_path = root / "manifest.json"
            _write_json(formal_path, {"placeholder": True})
            metadata = {
                "stage": "full",
                "method": "fstta",
                "episode_count": 2,
                "result_root": str(result_root),
                "formal_manifest": str(formal_path),
            }
            _write_json(attempt / "job.json", metadata)
            (attempt / "exitcode").write_text("0\n", encoding="utf-8")
            with mock.patch.object(
                runner.frozen_runner,
                "_validate_formal_manifest",
                return_value={"immutable_identity_sha256": "d" * 64},
            ) as validate:
                result = runner.validate_attempt(attempt)
            required = validate.call_args.kwargs["required_artifacts"]
            self.assertEqual(
                {Path(item) for item in required},
                {metric_path, diagnostics_path},
            )
            self.assertEqual(result["formal_manifest_sha256"], runner._sha256(formal_path))


if __name__ == "__main__":
    unittest.main()
