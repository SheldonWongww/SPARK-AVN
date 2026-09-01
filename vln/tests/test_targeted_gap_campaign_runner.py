import importlib.util
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "vln/scripts/run_targeted_gap_campaign.py"
BASE_SPEC = REPO_ROOT / "vln/experiments/vln_targeted_gap_campaign_v1.json"
SPEC = REPO_ROOT / "vln/experiments/vln_targeted_gap_campaign_v2.json"

MODULE_SPEC = importlib.util.spec_from_file_location(
    "targeted_gap_campaign_runner", MODULE_PATH
)
RUNNER = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(RUNNER)


class TargetedGapCampaignRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec_path, cls.spec = RUNNER.load_spec(SPEC)

    def test_exact_search_expansion_and_fixed_gpu_queues(self):
        jobs = RUNNER.expand_search_jobs(self.spec, (0, 1, 2, 3))
        self.assertEqual(len(jobs), 1024)
        self.assertEqual({job["queue_id"] for job in jobs}, set(range(16)))
        for job in jobs:
            self.assertEqual(job["gpu_slot"], job["queue_id"] % 4)
            self.assertEqual(job["gpu"], job["queue_id"] % 4)
            self.assertEqual(job["split"], "val_unseen")
        counts = {
            gpu: sum(job["gpu"] == gpu for job in jobs) for gpu in range(4)
        }
        self.assertEqual(counts, {0: 256, 1: 256, 2: 256, 3: 256})

    def test_manifest_bound_checkpoint_symlink_may_target_external_storage(self):
        with tempfile.TemporaryDirectory() as outside, \
                tempfile.TemporaryDirectory(dir=str(REPO_ROOT)) as inside:
            target = Path(outside) / "checkpoint.bin"
            target.write_bytes(b"checkpoint")
            link = Path(inside) / "checkpoint.bin"
            link.symlink_to(target)
            relative_link = link.relative_to(REPO_ROOT).as_posix()
            digest = RUNNER.sha256(target)
            spec = json.loads(json.dumps(self.spec))
            spec["data_bindings"]["checkpoints"]["hamt-reverie"] = digest
            assets = {
                "hamt_reverie_checkpoint": {
                    "path": relative_link,
                    "sha256": digest,
                    "size": target.stat().st_size,
                }
            }
            with mock.patch.object(
                RUNNER, "asset_index", return_value=(SPEC, assets)
            ):
                binding = RUNNER.checkpoint_binding(
                    spec, "hamt-reverie", require_file=True
                )
            self.assertEqual(binding["path"], target.resolve())
            with self.assertRaisesRegex(RUNNER.CampaignError, "escapes repository"):
                RUNNER.repo_file(relative_link, "tracked file", require=True)

    def test_canonical_seed_zero_commands_omit_order_seed_and_pin_ce_v12(self):
        jobs = RUNNER.expand_search_jobs(self.spec, (0, 1, 2, 3))
        for job in jobs:
            command = RUNNER._command(
                job, Path("/tmp/config.json"),
                Path("/tmp") / "run" / job["split"], "run",
            )
            self.assertNotIn("--order-seed", command)
            if job["benchmark"] == "r2r-ce":
                self.assertEqual(
                    command[-2:], ["--ce-data-version", "v1.2-native"]
                )
            else:
                self.assertNotIn("--ce-data-version", command)

    def test_discrete_configs_defer_idea_integrity_hash_to_full_horizon(self):
        jobs = RUNNER.expand_search_jobs(self.spec, (0, 1, 2, 3))
        discrete = next(job for job in jobs if job["method"] == "idea")
        config = RUNNER._job_config(self.spec, "batch", discrete)
        expected = RUNNER.order_binding(
            self.spec, discrete["setting"], discrete["split"]
        )["episode_count"]
        self.assertEqual(
            config["parameters"]["diagnostics_expected_episodes"], expected
        )
        continuous = next(job for job in jobs if job["benchmark"] == "r2r-ce")
        continuous_config = RUNNER._job_config(
            self.spec, "batch", continuous
        )
        self.assertNotIn(
            "diagnostics_expected_episodes", continuous_config["parameters"]
        )

    def test_reverie_selection_uses_all_four_metrics_before_diagnostics(self):
        base = {
            "stage": "search", "split": "val_unseen",
            "cell_id": "goat-reverie-eam", "benchmark": "reverie",
            "metrics": {"RGSPL": 20.0, "RGS": 30.0, "SPL": 40.0, "SR": 50.0},
            "diagnostics": {"relative_param_drift": 0.01, "updates": 1},
        }
        lower_spl = dict(base, candidate_id="a")
        lower_spl["metrics"] = dict(base["metrics"], SPL=39.0)
        higher_spl = dict(base, candidate_id="z")
        winners, rankings = RUNNER.select_winners([lower_spl, higher_spl])
        self.assertEqual(winners["goat-reverie-eam"]["candidate_id"], "z")
        self.assertEqual(rankings["goat-reverie-eam"][0]["candidate_id"], "z")

    def test_each_cell_queue_is_serial_even_when_other_queues_are_parallel(self):
        jobs = RUNNER.expand_search_jobs(self.spec, (0, 1, 2, 3))
        queue = [job for job in jobs if job["queue_id"] == 0]
        observed = []
        with mock.patch.object(
            RUNNER, "_choose_attempt", side_effect=lambda root, job, retry, reason: (
                "run", 1, None
            )
        ), mock.patch.object(
            RUNNER, "_run_attempt",
            side_effect=lambda *args: observed.append(args[-2]["candidate_id"]) or True,
        ):
            failures = RUNNER._run_cell_queue(
                SPEC, self.spec, "batch", Path("/tmp/batch"), queue, False
            )
        expected = [item["id"] for item in self.spec["candidate_grids"][
            "reverie_eam_v2"
        ]["candidates"]]
        self.assertEqual(failures, [])
        self.assertEqual(observed, expected)

    def test_live_worker_stops_the_rest_of_its_serial_queue(self):
        jobs = [
            job for job in RUNNER.expand_search_jobs(self.spec, (0, 1, 2, 3))
            if job["queue_id"] == 0
        ]
        with mock.patch.object(
            RUNNER, "_choose_attempt",
            return_value=("blocked", 1, "worker is still running"),
        ) as choose, mock.patch.object(RUNNER, "_run_attempt") as launch:
            failures = RUNNER._run_cell_queue(
                SPEC, self.spec, "batch", Path("/tmp/batch"), jobs, False
            )
        self.assertEqual(choose.call_count, 1)
        launch.assert_not_called()
        self.assertEqual(len(failures), 1)

    def test_failed_materialization_never_publishes_a_pending_attempt(self):
        job = RUNNER.expand_search_jobs(self.spec, (0, 1, 2, 3))[0]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            RUNNER, "_validate_config",
            side_effect=RUNNER.CampaignError("translator failed"),
        ):
            root = Path(directory)
            with self.assertRaisesRegex(RUNNER.CampaignError, "translator failed"):
                RUNNER.materialize_attempt(
                    SPEC, self.spec, "batch", root, job, 1
                )
            parent = RUNNER._job_parent(root, job)
            self.assertFalse((parent / "attempt-001").exists())
            self.assertEqual(
                len(list(parent.glob("attempt-001.preparing.*"))), 1
            )

    def test_pid_liveness_requires_matching_process_start_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            identity = RUNNER._process_identity(os.getpid())
            self.assertIsNotNone(identity)
            (attempt / "pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")
            RUNNER.atomic_json(attempt / "process_identity.json", identity)
            self.assertTrue(RUNNER._pid_alive(attempt))
            identity["start_token"] += "-reused"
            RUNNER.atomic_json(attempt / "process_identity.json", identity)
            self.assertFalse(RUNNER._pid_alive(attempt))

    def test_retry_requires_scheduler_attributed_signal_and_keeps_evidence(self):
        job = RUNNER.expand_search_jobs(self.spec, (0, 1, 2, 3))[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = RUNNER._job_parent(root, job) / "attempt-001"
            attempt.mkdir(parents=True)
            (attempt / "job.json").write_text(json.dumps({
                "job_identity_sha256": "a" * 64,
            }) + "\n", encoding="utf-8")
            (attempt / "exitcode").write_text("137\n", encoding="utf-8")
            action, number, reason = RUNNER._choose_attempt(root, job, False)
            self.assertEqual((action, number), ("blocked", 1))
            self.assertIn("retry-failed", reason)
            with self.assertRaisesRegex(
                RUNNER.CampaignError, "not independently marked"
            ):
                RUNNER._choose_attempt(
                    root, job, True, "worker host lost power"
                )
            process_identity = {
                "schema": RUNNER.PROCESS_SCHEMA,
                "hostname": "test-host", "pid": 123,
                "process_group_id": 123, "start_token": "ticks:1",
                "command_sha256": "b" * 64,
            }
            RUNNER.atomic_json(
                attempt / "process_identity.json", process_identity
            )
            RUNNER.atomic_json(attempt / "SCHEDULER_TERMINATION.json", {
                "schema": RUNNER.TERMINATION_SCHEMA,
                "reason": "operator_keyboard_interrupt",
                "job_identity_sha256": "a" * 64,
                "pid": 123, "process_group_id": 123,
                "start_token": "ticks:1",
                "termination_signal": 15,
                "recorded_at": "2026-08-31T00:00:00+00:00",
            })
            action, number, reason = RUNNER._choose_attempt(
                root, job, True, "worker host lost power"
            )
            self.assertEqual((action, number, reason), ("run", 2, None))
            self.assertTrue((attempt / "RETRY_ARCHIVE.json").is_file())
            self.assertTrue((attempt / "exitcode").is_file())
            archive = json.loads(
                (attempt / "RETRY_ARCHIVE.json").read_text(encoding="utf-8")
            )
            self.assertEqual(archive["failure_classification"], "infrastructure")
            self.assertTrue(archive["operator_authorized"])
            self.assertEqual(
                archive["classification_evidence"]["basis"],
                "scheduler_recorded_termination",
            )
            self.assertEqual(archive["job_key"], RUNNER._job_key(job))
            self.assertEqual(archive["retry_reason"], "worker host lost power")
            self.assertIn("job.json", archive["prior_evidence"])

    def test_retry_failed_requires_nonempty_operator_reason(self):
        job = RUNNER.expand_search_jobs(self.spec, (0, 1, 2, 3))[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = RUNNER._job_parent(root, job) / "attempt-001"
            attempt.mkdir(parents=True)
            RUNNER.atomic_json(attempt / "job.json", {
                "job_identity_sha256": "a" * 64,
            })
            (attempt / "exitcode").write_text("137\n", encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.CampaignError, "retry-reason"):
                RUNNER._choose_attempt(root, job, True, "   ")
            (attempt / "exitcode").write_text("1\n", encoding="utf-8")
            with self.assertRaisesRegex(
                RUNNER.CampaignError, "not independently marked"
            ):
                RUNNER._choose_attempt(root, job, True, "loss diverged")

    def test_feedtta_llm_test_config_injects_only_pinned_provider(self):
        search = RUNNER.expand_search_jobs(self.spec, (0, 1, 2, 3))
        by_cell = {}
        for job in search:
            by_cell.setdefault(job["cell_id"], job)
        frozen = {"winners": []}
        for cell in self.spec["cells"]:
            job = by_cell[cell["cell_id"]]
            frozen["winners"].append({
                "cell_id": cell["cell_id"],
                "candidate_id": job["candidate_id"],
                "candidate_role": job["candidate_role"],
                "parameters": job["parameters"],
                "frozen_config_sha256": "a" * 64,
            })
        jobs = RUNNER.expand_frozen_jobs(
            self.spec, (0, 1, 2, 3), frozen, "reverie-test", "batch"
        )
        self.assertEqual(len(jobs), 5)
        llm = next(job for job in jobs if job["reported_method_label"] == "FeedTTA-LLM")
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"NAVTTA_LLM_FEEDBACK_TOKEN_FILE": str(Path(directory) / "token")}
        ), mock.patch.object(
            RUNNER, "_validate_reference",
            return_value=REPO_ROOT / "vln/manifests/models/qwen2_vl_2b_instruct.json",
        ):
            config = RUNNER._job_config(self.spec, "batch", llm, Path(directory) / "result")
        params = config["parameters"]
        provider = self.spec["hidden_test_transfer"]["provider"]
        self.assertEqual(params["feedback_provider"], "qwen2_vl_2b_v1")
        self.assertEqual(params["llm_feedback_model_id"], provider["model_id"])
        self.assertEqual(params["llm_feedback_revision"], provider["revision"])
        self.assertEqual(params["llm_feedback_weights_sha256"], provider["weights_sha256"])
        self.assertEqual(params["llm_feedback_prompt_sha256"], provider["prompt_bundle_sha256"])
        self.assertEqual(
            params["llm_feedback_bundle_sha256"],
            "cf7dd27d27987b7ae71458529e6a72e3dcc3db6e91fb7298577e03f88e238a7e",
        )
        self.assertEqual(
            params["llm_feedback_transcript_path"],
            str((Path(directory) / "result/llm_feedback_transcript.ndjson").resolve()),
        )
        self.assertTrue(params["llm_feedback_abort_on_failure"])
        ordinary = next(job for job in jobs if job["reported_method_label"] == "EAM")
        ordinary_config = RUNNER._job_config(self.spec, "batch", ordinary)
        self.assertNotIn("feedback_provider", ordinary_config["parameters"])

    def test_llm_config_validation_uses_the_real_diagnostics_directory(self):
        diagnostics = Path("/tmp/job/result/tta_diagnostics.json")
        completed = mock.Mock(returncode=0, stdout="feedtta\n", stderr="")
        with mock.patch.object(RUNNER.subprocess, "run", return_value=completed) as run:
            RUNNER._validate_config(
                "goat-reverie", Path("/tmp/config.json"), "feedtta", diagnostics
            )
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--diagnostics") + 1], str(diagnostics.resolve()))

    def test_active_v2_is_launch_ready_without_source_gate(self):
        with mock.patch.object(RUNNER, "require_tracked"):
            RUNNER.require_launch_ready(SPEC, self.spec)
        self.assertFalse(RUNNER.source_controls_are_execution_gate(self.spec))
        self.assertNotIn(
            "matched_source_available", self.spec["selection"]["constraints"]
        )
        self.assertNotIn(
            "all_matched_source_manifest_sha256_values",
            self.spec["freeze"]["required_bindings"],
        )

    def test_direct_preflight_validates_inputs_without_reading_source_ledgers(self):
        with mock.patch.object(
            RUNNER, "assert_clean_formal_tree"
        ) as clean, mock.patch.object(
            RUNNER, "validate_runtime_assets"
        ) as runtime, mock.patch.object(
            RUNNER, "validate_idea_assets", return_value={}
        ) as ideas, \
                mock.patch.object(
                    RUNNER, "validate_source_controls",
                    side_effect=AssertionError("Source must not be consulted"),
                ):
            result = RUNNER.formal_preflight(SPEC, self.spec, "search")
        self.assertEqual(result["source_controls"], {})
        clean.assert_called_once_with(SPEC)
        runtime.assert_called_once_with(self.spec, "search")
        ideas.assert_called_once_with(self.spec)

    def test_run_command_dispatches_search_freeze_and_val_seen(self):
        entrypoint = RUNNER.main
        with mock.patch.object(RUNNER, "main", return_value=0) as phase:
            entrypoint([
                "run", "--spec", str(SPEC),
                "--batch-id", "vln-targeted-gap-campaign-v2-seed0",
                "--gpus", "0,1,2,3",
            ])
        self.assertEqual(
            [call.args[0][0] for call in phase.call_args_list],
            ["search", "freeze", "val-seen"],
        )

    def test_superseded_v1_is_inspectable_but_not_executable(self):
        _, predecessor = RUNNER.load_spec(BASE_SPEC)
        with self.assertRaisesRegex(RUNNER.CampaignError, "launch_readiness"):
            RUNNER.require_launch_ready(BASE_SPEC, predecessor)

    def test_successor_rejects_wrong_predecessor_digest(self):
        successor = json.loads(SPEC.read_text(encoding="utf-8"))
        successor["base_spec"]["sha256"] = "0" * 64
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", dir=str(REPO_ROOT),
            encoding="utf-8", delete=False,
        ) as stream:
            json.dump(successor, stream)
            successor_path = Path(stream.name)
        try:
            with self.assertRaisesRegex(RUNNER.CampaignError, "SHA256"):
                RUNNER.load_spec(successor_path)
        finally:
            successor_path.unlink()

    def test_optional_source_auditor_still_reports_unpromoted_ce_ledger(self):
        with self.assertRaisesRegex(
            RUNNER.CampaignError,
            "r2r_ce_v1_2_val_unseen.*not ready",
        ):
            RUNNER.validate_source_controls(self.spec)

    def test_source_manifest_reauthenticates_native_aggregate_fallback(self):
        from tools.run_manifest_identity import immutable_identity_sha256

        ledger = json.loads((
            REPO_ROOT / "vln/manifests/reverie_val_unseen_reused_source_controls.json"
        ).read_text(encoding="utf-8"))
        original_record = ledger["records"]["duet-reverie"]
        original_manifest = json.loads((
            REPO_ROOT / original_record["formal_manifest_path"]
        ).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(dir=str(REPO_ROOT)) as directory:
            root = Path(directory)
            native = root / "valid.txt"
            native.write_text(
                "Env name: val_unseen, sr: 46.98, spl: 33.73, "
                "rgs: 32.15, rgspl: 23.03\n",
                encoding="utf-8",
            )
            derived = root / "metrics.json"
            RUNNER.atomic_json(derived, {"metrics": original_record["metrics"]})
            manifest = json.loads(json.dumps(original_manifest))
            manifest["result_artifacts"] = [{
                "name": "logs/valid.txt", "path": str(native),
                "size": native.stat().st_size, "sha256": RUNNER.sha256(native),
            }]
            manifest["immutable_identity_sha256"] = immutable_identity_sha256(
                manifest
            )
            manifest_path = root / "manifest.json"
            RUNNER.atomic_json(manifest_path, manifest)
            record = dict(original_record)
            record.update({
                "git_commit": manifest["git_commit"],
                "formal_manifest_path": str(manifest_path),
                "formal_manifest_sha256": RUNNER.sha256(manifest_path),
                "immutable_identity_sha256": manifest[
                    "immutable_identity_sha256"
                ],
                "metrics_artifact_path": str(derived),
                "metrics_artifact_sha256": RUNNER.sha256(derived),
            })
            order = RUNNER.order_binding(
                self.spec, "duet-reverie", "val_unseen"
            )
            expected = {
                "setting": "duet-reverie", "split": "val_unseen",
                "data_version": "discrete-native", "model": "duet",
                "benchmark": order["benchmark"],
                "selection_benchmark": "reverie",
                "dataset_version": order["benchmark"],
                "checkpoint_sha256": self.spec["data_bindings"][
                    "checkpoints"
                ]["duet-reverie"],
                "dataset_sha256": order["dataset_sha256"],
                "order_sha256": order["order_sha256"],
                "order_manifest_sha256": order["sha256"],
                "metric_artifact_sha256": RUNNER.sha256(derived),
                "metric_artifact_size": derived.stat().st_size,
                "metric_artifact_path": derived.resolve(),
                "required_metrics": ("RGSPL", "RGS", "SPL", "SR"),
                "source_metrics": original_record["metrics"],
            }
            with mock.patch.object(RUNNER, "require_tracked"):
                RUNNER._validate_formal_source_manifest(
                    str(manifest_path), RUNNER.sha256(manifest_path), record,
                    expected,
                )
                native.write_text("tampered\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    RUNNER.CampaignError, "artifact changed"
                ):
                    RUNNER._validate_formal_source_manifest(
                        str(manifest_path), RUNNER.sha256(manifest_path), record,
                        expected,
                    )

    def test_source_validation_rejects_unpromoted_candidate_ledger(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", dir=str(REPO_ROOT),
            encoding="utf-8", delete=False,
        ) as stream:
            json.dump({
                "schema": (
                    "navtta.vln_r2r_ce_v1_2_source_controls_candidate.v1"
                ),
                "schema_version": 1,
                "candidate_status": "review_required_not_yet_promoted",
            }, stream)
            ledger_path = Path(stream.name)
        try:
            candidate = json.loads(ledger_path.read_text(encoding="utf-8"))
            with mock.patch.object(RUNNER, "require_tracked"):
                with self.assertRaisesRegex(
                    RUNNER.CampaignError, "schema mismatch|not been promoted"
                ):
                    RUNNER._validate_source_ledger_header(
                        "r2r_ce_v1_2_val_unseen", "r2r-ce", "val_unseen",
                        ledger_path, candidate,
                    )
        finally:
            ledger_path.unlink()

    def test_promoted_ce_ledger_binds_reviewed_candidate_bytes(self):
        with tempfile.TemporaryDirectory(dir=str(REPO_ROOT)) as directory:
            root = Path(directory)
            candidate_path = root / "candidate.json"
            candidate = {
                "schema": (
                    "navtta.vln_r2r_ce_v1_2_source_controls_candidate.v1"
                ),
                "schema_version": 1,
                "candidate_status": "review_required_not_yet_promoted",
                "split": "val_unseen", "records": {},
            }
            RUNNER.atomic_json(candidate_path, candidate)
            ledger = {
                "schema": "navtta.vln_r2r_ce_v1_2_source_controls.v1",
                "schema_version": 1,
                "promotion_status": "independently_reviewed_and_promoted",
                "split": "val_unseen", "records": {},
                "promotion_review": {
                    "decision": "approved", "reviewed_by": "reviewer",
                    "reviewed_at": "2026-08-31T00:00:00+08:00",
                    "candidate_ledger_path": str(
                        candidate_path.relative_to(REPO_ROOT)
                    ),
                    "candidate_ledger_sha256": "0" * 64,
                },
            }
            with mock.patch.object(RUNNER, "require_tracked"):
                with self.assertRaisesRegex(
                    RUNNER.CampaignError, "candidate ledger digest"
                ):
                    RUNNER._validate_source_ledger_header(
                        "r2r_ce_v1_2_val_unseen", "r2r-ce", "val_unseen",
                        root / "promoted.json", ledger,
                    )
                ledger["promotion_review"]["candidate_ledger_sha256"] = (
                    RUNNER.sha256(candidate_path)
                )
                RUNNER._validate_source_ledger_header(
                    "r2r_ce_v1_2_val_unseen", "r2r-ce", "val_unseen",
                    root / "promoted.json", ledger,
                )

    def test_missing_discrete_paired_metrics_is_an_explicit_freeze_gate(self):
        metadata = {
            "benchmark": "r2r", "result_root": "/does/not/matter",
        }
        path, ready = RUNNER._per_episode_evidence(
            metadata, Path("/does/not/matter")
        )
        self.assertIsNone(path)
        self.assertFalse(ready)

    def test_per_episode_sidecar_binds_source_order_and_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            order = root / "order.json"
            RUNNER.atomic_json(order, {
                "episodes": [
                    {"episode_id": "a", "scene_id": "s1"},
                    {"episode_id": "b", "scene_id": "s2"},
                ]
            })
            metadata = {
                "setting": "hamt-r2r", "split": "val_unseen",
                "run_tag": "run", "episode_count": 2,
                "order_sha256": "a" * 64,
                "order_manifest_sha256": RUNNER.sha256(order),
                "order_manifest_path": str(order), "benchmark": "r2r",
            }
            sidecar = root / "per_episode_metrics.json"
            document = {
                "schema": RUNNER.PER_EPISODE_SCHEMA,
                "setting": "hamt-r2r", "split": "val_unseen",
                "run_tag": "run", "episode_count": 2,
                "order_sha256": "a" * 64,
                "episode_order_manifest_sha256": RUNNER.sha256(order),
                "source": "native_discrete_evaluator_return",
                "episodes": [
                    {"ordinal": 0, "episode_id": "a", "metrics": {
                        "spl": 0.25, "success": 0.0,
                        "nDTW": 0.5, "SDTW": 0.0,
                    }},
                    {"ordinal": 1, "episode_id": "b", "metrics": {
                        "spl": 0.75, "success": 1.0,
                        "nDTW": 0.7, "SDTW": 0.7,
                    }},
                ],
            }
            RUNNER.atomic_json(sidecar, document)
            aggregates = {
                "SPL": 50.0, "SR": 50.0, "NDTW": 60.0, "SDTW": 35.0,
            }
            RUNNER._validate_per_episode_document(sidecar, metadata, aggregates)
            wrong = dict(aggregates, SR=75.0)
            with self.assertRaisesRegex(RUNNER.CampaignError, "SR disagrees"):
                RUNNER._validate_per_episode_document(sidecar, metadata, wrong)
            document["episodes"][1]["ordinal"] = 7
            RUNNER.atomic_json(sidecar, document)
            with self.assertRaisesRegex(RUNNER.CampaignError, "ordinal"):
                RUNNER._validate_per_episode_document(sidecar, metadata, aggregates)

    def test_adaptive_diagnostics_reject_empty_scope_and_noop(self):
        metadata = {"method": "tent"}
        base = {
            "trainable_prefixes": ["encoder.norm"],
            "adapter": {
                "adapted_parameter_names": ["encoder.norm.weight"],
                "adapted_parameter_count": 2,
                "action_steps": 3,
                "updates": 3,
            },
        }
        names, attempted, backward, accepted = RUNNER._diagnostic_accounting(
            metadata, base, base["adapter"]
        )
        self.assertEqual((attempted, backward, accepted), (3, 3, 3))
        self.assertTrue(names)
        base["adapter"]["updates"] = 0
        with self.assertRaisesRegex(RUNNER.CampaignError, "no-op"):
            RUNNER._diagnostic_accounting(metadata, base, base["adapter"])

    @staticmethod
    def _valid_idea_diagnostic_evidence():
        metadata = {
            "method": "idea", "benchmark": "r2r", "split": "val_unseen",
            "episode_count": 2, "feedback_provider": "none",
            "supervision": "unsupervised",
            "runtime_parameters": {
                "action_selection": "argmax", "lr": 3e-3,
                "prompt_length": 4, "k_max": 32, "lambda": 0.4,
                "tau": 0.7, "opt_steps": 50, "use_fisher": True,
                "diagnostics_expected_episodes": 2,
            },
        }
        adapter = {
            "episodes": 2,
            "adapted_parameter_names": ["external_soft_prompt"],
            "adapted_parameter_count": 32,
            "updates": 100,
            "new_domain_steps": 2,
            "opt_steps": 50,
            "prompt_optimizer_attempts": 100,
            "prompt_optimizer_updates": 100,
            "prompt_optimizations": 2,
            "relative_param_drift": 0.2,
            "max_prompt_relative_drift": 0.3,
            "current_lr": 3e-3,
            "prompt_length": 4,
            "library_capacity": 32,
            "lambda": 0.4,
            "tau": 0.7,
            "use_fisher": True,
            "base_parameter_integrity_schema": (
                "navtta.idea.base_parameter_integrity.v1"
            ),
            "base_parameter_integrity_complete": True,
            "base_parameter_unchanged": True,
            "trains_base_policy": False,
            "base_parameter_grads_none": True,
            "base_parameter_name_count_before": 10,
            "base_parameter_name_count_after": 10,
            "base_parameter_name_set_before_sha256": "a" * 64,
            "base_parameter_name_set_after_sha256": "a" * 64,
            "base_parameter_name_set_unchanged": True,
            "base_parameter_added_names": [],
            "base_parameter_removed_names": [],
            "base_parameter_content_before_sha256": "b" * 64,
            "base_parameter_content_after_sha256": "b" * 64,
            "base_parameter_content_hash_match": True,
            "base_parameter_content_modified_names": [],
            "base_parameter_versions_unchanged": True,
            "base_parameter_version_changed_names": [],
            "base_parameter_modified_names": [],
        }
        diagnostics = {
            "schema": "navtta.vln_discrete_tta.v1",
            "method": "idea", "episode_count": 2,
            "stream": "val_unseen", "batch_size": 1,
            "audit_zero_update": False, "audit_control": False,
            "action_selection": "target_native_argmax",
            "supervision": "unsupervised",
            "attempted_updates": 100,
            "backward_passes": 100,
            "accepted_updates": 100,
            "diagnostics_expected_episodes": 2,
            "adapter": adapter,
        }
        return metadata, diagnostics, adapter

    def test_idea_diagnostics_bind_prompt_updates_drift_and_frozen_policy(self):
        metadata, diagnostics, _ = self._valid_idea_diagnostic_evidence()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tta_diagnostics.json"
            RUNNER.atomic_json(path, diagnostics)
            _, compact = RUNNER._validate_diagnostics(metadata, path)
        self.assertEqual(compact["trainable_names"], ["external_soft_prompt"])
        self.assertEqual(compact["attempted_updates"], 100)
        self.assertEqual(compact["backward_passes"], 100)
        self.assertEqual(compact["updates"], 100)
        self.assertEqual(compact["relative_param_drift"], 0.2)

    def test_idea_diagnostics_reject_prompt_update_or_drift_noop(self):
        metadata, diagnostics, adapter = self._valid_idea_diagnostic_evidence()
        RUNNER._diagnostic_accounting(metadata, diagnostics, adapter)

        adapter["prompt_optimizer_updates"] = 99
        with self.assertRaisesRegex(RUNNER.CampaignError, "no-op"):
            RUNNER._diagnostic_accounting(metadata, diagnostics, adapter)

        _, diagnostics, adapter = self._valid_idea_diagnostic_evidence()
        adapter["prompt_optimizer_attempts"] = 100.0
        with self.assertRaisesRegex(RUNNER.CampaignError, "counter is invalid"):
            RUNNER._diagnostic_accounting(metadata, diagnostics, adapter)

        _, diagnostics, adapter = self._valid_idea_diagnostic_evidence()
        adapter["relative_param_drift"] = 0.4
        with self.assertRaisesRegex(RUNNER.CampaignError, "no-op"):
            RUNNER._diagnostic_accounting(metadata, diagnostics, adapter)

        _, diagnostics, adapter = self._valid_idea_diagnostic_evidence()
        adapter["relative_param_drift"] = 0.0
        with self.assertRaisesRegex(RUNNER.CampaignError, "no-op"):
            RUNNER._diagnostic_accounting(metadata, diagnostics, adapter)

    def test_idea_diagnostics_reject_base_policy_version_change(self):
        metadata, diagnostics, adapter = self._valid_idea_diagnostic_evidence()
        names = ["external_soft_prompt"]
        RUNNER._validate_method_diagnostic_contract(
            metadata, diagnostics, adapter, names
        )

        adapter["base_parameter_versions_unchanged"] = False
        adapter["base_parameter_modified_names"] = ["encoder.weight"]
        with self.assertRaisesRegex(
            RUNNER.CampaignError, "idea.base_parameter_versions_unchanged"
        ):
            RUNNER._validate_method_diagnostic_contract(
                metadata, diagnostics, adapter, names
            )
        adapter["base_parameter_versions_unchanged"] = True
        with self.assertRaisesRegex(
            RUNNER.CampaignError, "idea.base_parameter_modified_names"
        ):
            RUNNER._validate_method_diagnostic_contract(
                metadata, diagnostics, adapter, names
            )
        _, diagnostics, adapter = self._valid_idea_diagnostic_evidence()
        with self.assertRaisesRegex(
            RUNNER.CampaignError, "exactly external_soft_prompt"
        ):
            RUNNER._validate_method_diagnostic_contract(
                metadata, diagnostics, adapter, ["encoder.weight"]
            )
        _, diagnostics, adapter = self._valid_idea_diagnostic_evidence()
        adapter["base_parameter_integrity_complete"] = False
        with self.assertRaisesRegex(
            RUNNER.CampaignError, "base_parameter_integrity_complete"
        ):
            RUNNER._validate_method_diagnostic_contract(
                metadata, diagnostics, adapter, ["external_soft_prompt"]
            )
        _, diagnostics, adapter = self._valid_idea_diagnostic_evidence()
        adapter["base_parameter_content_after_sha256"] = "c" * 64
        with self.assertRaisesRegex(RUNNER.CampaignError, "content digest"):
            RUNNER._validate_method_diagnostic_contract(
                metadata, diagnostics, adapter, ["external_soft_prompt"]
            )

    def test_method_diagnostics_must_match_effective_hyperparameters(self):
        metadata = {
            "method": "eam", "benchmark": "r2r", "split": "val_unseen",
            "episode_count": 2,
            "runtime_parameters": {
                "action_selection": "argmax", "lr": 3e-6,
                "confidence_scale": 0.4, "memory_size": 32,
                "batch_size": 8, "update_interval": 1,
                "optimizer": "Adam",
                "diagnostics_expected_episodes": 2,
            },
        }
        diagnostics = {
            "schema": "navtta.vln_discrete_tta.v1",
            "stream": "val_unseen", "batch_size": 1,
            "audit_zero_update": False, "audit_control": False,
            "action_selection": "target_native_argmax",
            "diagnostics_expected_episodes": 2,
        }
        adapter = {
            "current_lr": 3e-6, "confidence_scale": 0.4,
            "memory_size_steps": 32, "batch_size_steps": 8,
            "update_interval": 1, "optimizer": "Adam",
            "param_scope": "module_prefixes",
            "trainable_prefixes": ["vln_bert.encoder"],
        }
        RUNNER._validate_method_diagnostic_contract(
            metadata, diagnostics, adapter,
            ["vln_bert.encoder.layer.0.weight"],
        )
        adapter["current_lr"] = 1e-2
        with self.assertRaisesRegex(RUNNER.CampaignError, "eam.current_lr"):
            RUNNER._validate_method_diagnostic_contract(
                metadata, diagnostics, adapter,
                ["vln_bert.encoder.layer.0.weight"],
            )

    def test_hidden_test_rejects_local_metric_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "valid.txt").write_text(
                "Env name: test, sr: 99.0\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RUNNER.CampaignError, "forbidden"):
                RUNNER._reject_test_metric_artifacts({
                    "result_root": str(root), "split": "test"
                })

    def test_hidden_test_rejects_metric_json_under_innocent_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "payload.json").write_text(
                json.dumps({"summary": {"success": 0.9}}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RUNNER.CampaignError, "forbidden"):
                RUNNER._reject_test_metric_artifacts({
                    "result_root": str(root), "split": "test"
                })

    def test_hidden_test_rejects_metric_text_under_innocent_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "output.log").write_text(
                "completed; spl: 88.0\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RUNNER.CampaignError, "forbidden"):
                RUNNER._reject_test_metric_artifacts({
                    "result_root": str(root), "split": "test"
                })

    def test_llm_transcript_is_bound_to_prediction_and_budget(self):
        provider = self.spec["hidden_test_transfer"]["provider"]
        bundle = json.loads((REPO_ROOT / provider["runtime"]["contract"]["path"]).read_text(
            encoding="utf-8"
        ))["bundle_sha256"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            order = root / "order.json"
            RUNNER.atomic_json(order, {
                "episodes": [{"episode_id": "ep", "scene_id": "scan"}]
            })
            prediction = root / "submit_test.json"
            prediction.write_text(json.dumps([{
                "instr_id": "ep",
                "trajectory": [["start", 0.0, 0.0], ["end", 0.0, 0.0]],
            }]) + "\n", encoding="utf-8")
            row = {
                "schema": "navtta.reverie_llm_feedback_transcript.v1",
                "sequence": 0, "previous_event_sha256": "0" * 64,
                "status": "label", "provider_id": "qwen2_vl_2b_v1",
                "model_id": provider["model_id"], "revision": provider["revision"],
                "weights_sha256": provider["weights_sha256"],
                "bundle_sha256": bundle,
                "prompt_sha256_values": [
                    item["prompt_sha256"] for item in provider["pipeline"]
                ],
                "episode_id": "ep", "scan_id": "scan",
                "endpoint_viewpoint_id": "end",
                "trajectory_sha256": hashlib.sha256(
                    RUNNER.canonical(["start", "end"]).encode("utf-8")
                ).hexdigest(),
                "panorama_png_sha256": "1" * 64,
                "cache_key": "2" * 64, "cache_hit": False,
                "request_sha256_values": ["4" * 64, "5" * 64],
                "response_sha256_values": ["6" * 64, "7" * 64],
                "parsed_label": "Yes", "latency_seconds_values": [0.1, 0.2],
                "token_count": {"prompt": 10, "completion": 2}, "cost": 0.0,
            }
            record = {
                "schema": "navtta.reverie_llm_feedback_cache.v1",
                "provider": {"provider_id": "qwen2_vl_2b_v1"},
                "cache_key": row["cache_key"],
                "input_sha256": row["cache_key"],
                "episode_id": "ep", "scan_id": "scan",
                "endpoint_viewpoint_id": "end",
                "trajectory_sha256": row["trajectory_sha256"],
                "panorama": {
                    "panorama_png_sha256": row["panorama_png_sha256"]
                },
                "stage1": {
                    "request_sha256": row["request_sha256_values"][0],
                    "response_sha256": row["response_sha256_values"][0],
                    "output": "target", "latency_seconds": 0.1,
                },
                "stage2": {
                    "request_sha256": row["request_sha256_values"][1],
                    "response_sha256": row["response_sha256_values"][1],
                    "output": "Yes", "latency_seconds": 0.2,
                },
                "parsed_label": "Yes",
                "token_count": row["token_count"], "cost": 0.0,
                "created_at_unix": 1.0,
            }
            row["record_sha256"] = hashlib.sha256(
                RUNNER.canonical(record).encode("utf-8")
            ).hexdigest()
            cache_root = root / "cache"
            cache_path = (
                cache_root / row["cache_key"][:2]
                / (row["cache_key"] + ".json")
            )
            RUNNER.atomic_json(cache_path, record)
            row["event_sha256"] = hashlib.sha256(
                RUNNER.canonical(row).encode("utf-8")
            ).hexdigest()
            transcript = root / "llm_feedback_transcript.ndjson"
            transcript.write_text(RUNNER.canonical(row) + "\n", encoding="utf-8")
            diagnostics = {
                "transcript_events": 1, "transcript_sha256": row["event_sha256"],
                "http_requests": 2, "requests_per_uncached_episode": 2,
                "positive_labels": 1, "negative_labels": 0,
                "cache_hits": 0, "cache_misses": 1,
                "prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0,
                "max_labels": provider["budgets"]["binary_labels_maximum"],
                "max_http_requests": provider["budgets"]["provider_requests_maximum"],
                "feedback_timing": "post_episode_affects_future_episodes_only",
            }
            metadata = {
                "episode_count": 1, "order_manifest_path": str(order),
                "result_root": str(root), "split": "test", "spec_path": str(SPEC),
                "runtime_parameters": {
                    "llm_feedback_bundle_sha256": bundle,
                    "llm_feedback_cache_dir": str(cache_root),
                },
            }
            RUNNER._validate_feedback_transcript(transcript, diagnostics, metadata)
            row["endpoint_viewpoint_id"] = "wrong"
            row.pop("event_sha256")
            row["event_sha256"] = hashlib.sha256(
                RUNNER.canonical(row).encode("utf-8")
            ).hexdigest()
            transcript.write_text(RUNNER.canonical(row) + "\n", encoding="utf-8")
            diagnostics["transcript_sha256"] = row["event_sha256"]
            with self.assertRaisesRegex(
                RUNNER.CampaignError, "cache/transcript|trajectory binding"
            ):
                RUNNER._validate_feedback_transcript(transcript, diagnostics, metadata)

            row["endpoint_viewpoint_id"] = "end"
            record["parsed_label"] = "No"
            RUNNER.atomic_json(cache_path, record)
            with self.assertRaisesRegex(RUNNER.CampaignError, "record digest"):
                RUNNER._validate_feedback_cache_record(row, metadata)

    def test_gpu_probe_fails_closed(self):
        failed = mock.Mock(returncode=1, stdout="", stderr="driver unavailable")
        with mock.patch.object(RUNNER.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(RUNNER.CampaignError, "failed closed"):
                RUNNER._hardware(0)

    def test_formal_manifest_uses_campaign_data_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.bin"
            dataset = root / "dataset.json"
            config = root / "config.json"
            order = root / "order.json"
            for path in (checkpoint, dataset, config, order):
                path.write_text("evidence\n", encoding="utf-8")
            metadata = {
                "formal_manifest": str(root / "formal/manifest.json"),
                "spec_path": str(SPEC), "order_manifest_path": str(order),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": RUNNER.sha256(checkpoint),
                "dataset_path": str(dataset),
                "dataset_sha256": RUNNER.sha256(dataset),
                "order_sha256": "a" * 64,
                "expected_benchmark": "r2r_discrete_duet_hamt",
                "data_version": "discrete-native", "model": "hamt",
                "reported_method_label": "tent", "run_tag": "run",
                "setting": "hamt-r2r", "split": "val_unseen",
                "command": ["runner"], "config_path": str(config),
                "runtime_parameters": {}, "git_commit": RUNNER.git_commit(),
                "gpu": 0, "batch_id": "batch", "stage": "search",
                "cell_id": "hamt-r2r-tent", "candidate_id": "candidate",
                "queue_id": 5, "gpu_slot": 1,
                "config_sha256": RUNNER.sha256(config),
                "job_identity_sha256": "b" * 64,
                "attempt": 1, "retry_count": 0,
                "retry_history_sha256": hashlib.sha256(b"[]").hexdigest(),
                "supervision": "unsupervised", "feedback_provider": "none",
            }
            hardware = {
                "hostname": "host", "platform": "linux", "python": "3",
                "cuda_visible_devices": "0", "gpu_name": "GPU",
                "gpu_uuid": "GPU-1", "gpu_memory_mib": 100,
            }
            with mock.patch.object(RUNNER, "_hardware", return_value=hardware):
                manifest_path = RUNNER._create_run_manifest(metadata)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset"]["version"], "discrete-native")

    def test_job_json_rebinds_to_independent_stage_plan(self):
        job = RUNNER.expand_search_jobs(self.spec, (0, 1, 2, 3))[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = {
                "schema": RUNNER.BATCH_SCHEMA,
                "batch_id": "batch", "git_commit": RUNNER.git_commit(),
                "spec_path": str(SPEC), "spec_sha256": RUNNER.sha256(SPEC),
                "runner_sha256": RUNNER.sha256(MODULE_PATH),
                "gpus": [0, 1, 2, 3],
                "model_seed": 0, "episode_order_seed": 0,
            }
            RUNNER.atomic_json(root / "BATCH.json", binding)
            jobs = RUNNER.expand_search_jobs(self.spec, (0, 1, 2, 3))
            RUNNER.write_stage_plan(root, binding, "search", jobs)
            attempt = RUNNER._job_parent(root, job) / "attempt-001"
            attempt.mkdir(parents=True)
            metadata = dict(job)
            metadata.update({
                "batch_id": "batch",
                "job_identity_sha256": RUNNER._job_identity(SPEC, "batch", job),
            })
            RUNNER._expected_attempt_job(attempt, metadata)
            metadata["ordinal"] += 1
            with self.assertRaisesRegex(RUNNER.CampaignError, "ordinal"):
                RUNNER._expected_attempt_job(attempt, metadata)

    def test_retry_history_detects_changed_prior_evidence(self):
        job = RUNNER.expand_search_jobs(self.spec, (0, 1, 2, 3))[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = {
                "schema": RUNNER.BATCH_SCHEMA,
                "batch_id": "batch", "git_commit": RUNNER.git_commit(),
                "spec_path": str(SPEC), "spec_sha256": RUNNER.sha256(SPEC),
                "runner_sha256": RUNNER.sha256(MODULE_PATH),
                "gpus": [0, 1, 2, 3], "model_seed": 0,
                "episode_order_seed": 0,
            }
            RUNNER.atomic_json(root / "BATCH.json", binding)
            jobs = RUNNER.expand_search_jobs(self.spec, (0, 1, 2, 3))
            RUNNER.write_stage_plan(root, binding, "search", jobs)
            attempt = RUNNER._job_parent(root, job) / "attempt-001"
            attempt.mkdir(parents=True)
            metadata = dict(job)
            metadata.update({
                "batch_id": "batch",
                "job_identity_sha256": RUNNER._job_identity(SPEC, "batch", job),
            })
            RUNNER.atomic_json(attempt / "job.json", metadata)
            (attempt / "exitcode").write_text("137\n", encoding="utf-8")
            (attempt / "launcher.log").write_text("oom\n", encoding="utf-8")
            RUNNER.atomic_json(attempt / "process_identity.json", {
                "schema": RUNNER.PROCESS_SCHEMA,
                "hostname": "test-host", "pid": 123,
                "process_group_id": 123, "start_token": "ticks:1",
                "command_sha256": "b" * 64,
            })
            RUNNER.atomic_json(attempt / "SCHEDULER_TERMINATION.json", {
                "schema": RUNNER.TERMINATION_SCHEMA,
                "reason": "scheduler_signal_15",
                "job_identity_sha256": metadata["job_identity_sha256"],
                "pid": 123, "process_group_id": 123,
                "start_token": "ticks:1", "termination_signal": 15,
                "recorded_at": "2026-08-31T00:00:00+00:00",
            })
            RUNNER._archive_for_retry(
                attempt, 2, "failed", job, "GPU allocation service failed"
            )
            history, digest = RUNNER._validated_retry_history(root, job, 2)
            self.assertEqual(len(history), 1)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            (attempt / "launcher.log").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.CampaignError, "prior_evidence"):
                RUNNER._validated_retry_history(root, job, 2)

    def test_completed_manifest_is_not_rewritten_to_accept_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            artifact = root / "metrics.json"
            RUNNER.atomic_json(manifest, {
                "status": "running", "exit_code": None, "completed_at": None,
            })
            artifact.write_text("original\n", encoding="utf-8")
            RUNNER._finish_run_manifest(
                manifest, 0, (("aggregate_metrics", artifact),)
            )
            original_manifest = manifest.read_bytes()
            artifact.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.CampaignError, "changed"):
                RUNNER._finish_run_manifest(
                    manifest, 0, (("aggregate_metrics", artifact),)
                )
            self.assertEqual(manifest.read_bytes(), original_manifest)


if __name__ == "__main__":
    unittest.main()
