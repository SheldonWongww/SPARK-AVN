from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln/scripts"))

import run_r2r_cross_split_robust_eval as runner  # noqa: E402
import run_tta_hparam_search as hparam_runner  # noqa: E402


def _sha256(path):
    return runner._sha256(REPO_ROOT / path)


def _candidate_parameters(method, index):
    if method == "tent":
        return {
            "lr": (index + 1) * 1e-6, "update_interval": 1,
            "episodic": False,
        }
    if method == "fstta":
        return {
            "lr_fast": (index + 1) * 1e-5, "lr_slow": 1e-5,
            "m": 3, "n": 4, "episodic": False,
        }
    if method == "eam":
        return {
            "lr": (index + 1) * 1e-6, "confidence_scale": 0.5,
            "batch_size": 8, "episodic": False,
        }
    if method == "feedtta":
        return {
            "lr": (index + 1) * 1e-6, "p": 0.05, "alpha": 0.1,
            "action_selection": "argmax", "episodic": False,
        }
    return {
        "lr_query": (index + 1) * 1e-7, "lr_self": 1e-7,
        "action_selection": "argmax", "episodic": False,
    }


def _spec_document():
    counts = {"tent": 2, "fstta": 2, "eam": 2, "feedtta": 3, "atena": 2}
    candidates = {}
    for setting in runner.SETTINGS:
        candidates[setting] = {}
        for method in runner.METHODS:
            candidates[setting][method] = [{
                "candidate_id": "{}-{}".format(method, index),
                "parameters": _candidate_parameters(method, index),
            } for index in range(counts[method])]
    seen = "vln/manifests/r2r_reused_source_controls.json"
    unseen = "vln/manifests/r2r_val_unseen_reused_source_controls.json"
    caps = {
        model: {method: 5 for method in runner.METHODS}
        for model in runner.MODELS
    }
    return {
        "schema": runner.SCHEMA,
        "experiment_id": "unit-r2r-cross-split",
        "protocol": {
            "selection_split": "val_seen",
            "evaluation_split": "val_unseen",
            "selection_order_seeds": [1, 2, 3],
            "evaluation_order_seeds": [1, 2, 3],
            "episodes_by_split": {"val_seen": 1021, "val_unseen": 2349},
            "source_execution_jobs": 0,
            "no_val_unseen_selection": True,
            "freeze_before_evaluation": True,
        },
        "source_controls": {
            "val_seen": {"path": seen, "sha256": _sha256(seen)},
            "val_unseen": {"path": unseen, "sha256": _sha256(unseen)},
        },
        "implementation_gates": {
            "duet_atena_self_prediction": (
                "concat_global_and_local_cls_matching_official_2D_representation"
            ),
            "eam_warmup": "no_optimizer_update_until_reservoir_reaches_batch_size",
            "feedtta_action_protocol": "target_native_argmax",
            "atena_feedback": "lazy_submitted_trajectory_evaluator_query",
            "fstta_variance_history": "test_stream",
            "nonfinite_metrics_forbidden": True,
            "complete_diagnostics_required": True,
        },
        "matrix": {
            "setting_order": list(runner.SETTINGS),
            "method_order": list(runner.METHODS),
            "candidates": candidates,
        },
        "execution": {
            "formal_requires_clean_tracked_tree": True,
            "strict_model_barrier": True,
            "strict_method_barrier": True,
            "max_workers_by_model": caps,
        },
    }


def _load_spec():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
        json.dump(_spec_document(), stream)
        path = Path(stream.name)
    try:
        return runner.load_spec(path)
    finally:
        path.unlink()


def _row(candidate, seed, sr, spl, drift=0.01):
    return {
        "campaign_stage": "selection",
        "split": "val_seen",
        "candidate_id": candidate,
        "candidate_parameters": {"lr": 1e-6 if candidate == "a" else 2e-6},
        "order_seed": seed,
        "metrics": {"SR": sr, "SPL": spl},
        "relative_param_drift": drift,
        "run_tag": "{}-{}".format(candidate, seed),
        "formal_manifest_path": "/tmp/{}-{}.json".format(candidate, seed),
        "formal_manifest_sha256": "0" * 64,
    }


def _full_selection_results(spec):
    jobs = runner.build_selection_jobs(
        spec, "unit-selection-matrix", gpu=0, batch_root="/tmp/unit-selection-matrix"
    )
    results = []
    for job in jobs:
        result = dict(job)
        result.update({
            "metrics": {
                "SR": job["source_metrics"]["SR"] + 0.1,
                "SPL": job["source_metrics"]["SPL"] + 0.1,
            },
            "relative_param_drift": 0.01,
            "formal_manifest_path": "/tmp/{}.json".format(job["run_tag"]),
            "formal_manifest_sha256": "0" * 64,
        })
        results.append(result)
    return results


class R2RCrossSplitRobustEvalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = _load_spec()

    def test_selection_matrix_is_candidate_cross_seed_with_strict_cell_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = runner.build_selection_jobs(
                self.spec, "unit-batch", gpu=0, batch_root=directory
            )
        self.assertEqual(len(jobs), 99)
        self.assertEqual(len(runner.phase_groups(jobs)), 15)
        self.assertNotIn("source", {job["config_method"] for job in jobs})
        for phase in runner.phase_groups(jobs):
            self.assertEqual(
                len({(job["setting"], job["method"]) for job in phase}), 1
            )
            for job in phase:
                self.assertEqual(job["split"], "val_seen")
                self.assertEqual(job["episodes"], -1)
                self.assertEqual(job["stage"], "orders")
                self.assertEqual(job["command"][1:3], [job["setting"], "val_seen"])
                position = job["command"].index("--order-seed")
                self.assertEqual(
                    job["command"][position + 1], str(job["order_seed"])
                )
                self.assertNotIn("--episode-limit", job["command"])

    def test_feedtta_rng_seeds_follow_order_seed_in_every_written_config(self):
        with tempfile.TemporaryDirectory() as directory:
            jobs = runner.build_selection_jobs(
                self.spec, "unit-feed", gpu=0, batch_root=directory
            )
            feed_jobs = [job for job in jobs if job["method"] == "feedtta"]
            for job in feed_jobs:
                config = hparam_runner._job_config(job)
                self.assertEqual(config["stage"], "orders")
                self.assertEqual(config["order_seed"], job["order_seed"])
                self.assertEqual(
                    config["parameters"]["action_seed"], job["order_seed"]
                )
                self.assertEqual(
                    config["parameters"]["sgr_seed"], job["order_seed"]
                )

    def test_feasible_candidate_precedes_failed_gate_and_drift_breaks_ties(self):
        source = {"SR": 70.0, "SPL": 60.0}
        rows = []
        for seed, sr, spl in ((1, 69.8, 59.8), (2, 70.2, 60.2), (3, 70.1, 60.1)):
            rows.append(_row("a", seed, sr, spl, drift=0.02))
        # Better median SR, but an inadmissible worst SPL: feasibility wins.
        for seed, sr, spl in ((1, 70.5, 59.0), (2, 70.5, 61.0), (3, 70.5, 61.0)):
            rows.append(_row("b", seed, sr, spl, drift=0.001))
        winner, ranked = runner.rank_cell(rows, source)
        self.assertEqual(winner["candidate_id"], "a")
        self.assertTrue(winner["feasible"])
        self.assertEqual(ranked[1]["gate_status"], "fail")
        self.assertAlmostEqual(
            winner["statistics"]["SR"]["std"], 0.20816659994661327
        )

        tied = []
        for candidate, drift in (("a", 0.02), ("b", 0.01)):
            for seed in (1, 2, 3):
                tied.append(_row(candidate, seed, 70.1, 60.1, drift=drift))
        winner, _ = runner.rank_cell(tied, source)
        self.assertEqual(winner["candidate_id"], "b")

    def test_no_feasible_candidate_is_frozen_with_explicit_failed_gate(self):
        rows = [
            _row("a", seed, 69.0 + seed / 100.0, 59.0, drift=0.01)
            for seed in (1, 2, 3)
        ]
        winner, _ = runner.rank_cell(rows, {"SR": 70.0, "SPL": 60.0})
        self.assertEqual(winner["gate_status"], "fail")
        self.assertTrue(winner["fallback_selected"])

    def test_evaluation_requires_freeze_and_is_exactly_45_non_source_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(runner.UserError, "blocked"):
                runner.require_frozen(directory, self.spec)
            with self.assertRaisesRegex(runner.UserError, "frozen selection"):
                runner.build_evaluation_jobs(
                    self.spec, None, "unit-eval", batch_root=directory
                )

            frozen = {
                "schema": runner.FROZEN_SCHEMA,
                "cells": {
                    setting: {
                        method: {
                            "candidate_id": "{}-0".format(method),
                            "parameters": _candidate_parameters(method, 0),
                        } for method in runner.METHODS
                    } for setting in runner.SETTINGS
                },
            }
            jobs = runner.build_evaluation_jobs(
                self.spec, frozen, "unit-eval", batch_root=directory
            )
        self.assertEqual(len(jobs), 45)
        self.assertNotIn("source", {job["config_method"] for job in jobs})
        self.assertEqual(len(runner.phase_groups(jobs)), 15)
        for job in jobs:
            self.assertEqual(job["split"], "val_unseen")
            self.assertEqual(job["command"][1:3], [job["setting"], "val_unseen"])
            self.assertEqual(job["stage"], "orders")

    def test_ranker_rejects_val_unseen_feedback(self):
        rows = [_row("a", seed, 70.1, 60.1) for seed in (1, 2, 3)]
        rows[0]["split"] = "val_unseen"
        with self.assertRaisesRegex(runner.UserError, "val_seen selection"):
            runner.rank_cell(rows, {"SR": 70.0, "SPL": 60.0})

    def test_freeze_matrix_rejects_missing_duplicate_injected_and_changed_candidates(self):
        source = {
            setting: self.spec["_source_records"]["val_seen"][setting]["metrics"]
            for setting in runner.SETTINGS
        }
        rows = _full_selection_results(self.spec)
        cells, _ = runner.select_frozen(self.spec, rows, source)
        self.assertEqual(set(cells), set(runner.SETTINGS))

        cases = {}
        cases["missing"] = rows[:-1]
        cases["duplicate"] = rows + [deepcopy(rows[0])]
        injected = deepcopy(rows)
        injected[0]["candidate_id"] = "not-registered"
        cases["unregistered"] = injected
        changed = deepcopy(rows)
        changed[0]["candidate_parameters"]["lr"] = 9e-4
        cases["parameters"] = changed
        for label, candidate_rows in cases.items():
            with self.subTest(label=label), self.assertRaises(runner.UserError):
                runner.select_frozen(self.spec, candidate_rows, source)

    def test_selection_authentication_rebuilds_summary_instead_of_trusting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory)
            (batch_root / "selection").mkdir()
            forged = {"scope": "selection", "complete": True, "results": ["forged"]}
            (batch_root / "selection" / "SUMMARY.json").write_text(
                json.dumps(forged), encoding="utf-8"
            )
            plan = {"schema": runner.PLAN_SCHEMA, "batch_id": "unit", "gpu": 0}
            (batch_root / "PLAN.json").write_text(json.dumps(plan), encoding="utf-8")
            authenticated = {
                "schema": runner.SUMMARY_SCHEMA,
                "scope": "selection",
                "split": "val_seen",
                "complete": True,
                "results": ["authenticated-from-artifacts"],
            }
            with mock.patch.object(runner, "build_selection_jobs", return_value=[{}]), \
                    mock.patch.object(runner, "_plan_payload", return_value=plan), \
                    mock.patch.object(
                        runner, "_authenticate_complete_stage",
                        return_value=([{}], authenticated),
                    ):
                _, summary_path, returned = runner._authenticate_selection_evidence(
                    batch_root, self.spec
                )
            self.assertEqual(returned, authenticated)
            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8")), authenticated
            )

    def test_successful_freeze_binds_plan_and_authenticated_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_root = Path(directory)
            summary_path = batch_root / "selection" / "SUMMARY.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text("{}\n", encoding="utf-8")
            plan_path = batch_root / "PLAN.json"
            plan_path.write_text("{}\n", encoding="utf-8")
            plan = {
                "git_commit": runner._git_commit(),
                "spec_sha256": "a" * 64,
            }
            summary = {"results": []}
            cells = {setting: {} for setting in runner.SETTINGS}
            rankings = {setting: {} for setting in runner.SETTINGS}
            with mock.patch.object(
                runner, "_authenticate_selection_evidence",
                return_value=(plan, summary_path, summary),
            ), mock.patch.object(
                runner, "select_frozen", return_value=(cells, rankings),
            ):
                frozen = runner.freeze_selection(batch_root, self.spec)
            self.assertEqual(frozen["plan_path"], str(plan_path.resolve()))
            self.assertEqual(frozen["plan_sha256"], runner._sha256(plan_path))
            self.assertEqual(
                json.loads((batch_root / "FROZEN.json").read_text(encoding="utf-8")),
                frozen,
            )

    def test_selection_stage_stops_before_freeze(self):
        with mock.patch.object(runner, "load_spec", return_value=self.spec), \
                mock.patch.object(runner, "_prepare_batch", return_value=[]), \
                mock.patch.object(runner, "_materialize_plans", return_value=[]), \
                mock.patch.object(runner, "_require_clean_execution_tree"), \
                mock.patch.object(
                    runner, "write_stage_summary", return_value={"complete": True}
                ), mock.patch.object(runner, "_write_progress"), \
                mock.patch.object(runner, "freeze_selection") as freeze:
            code = runner.main([
                "--batch-id", "unit-selection-stop-before-freeze",
                "--stage", "selection", "--confirm-reviewed",
            ])
        self.assertEqual(code, 0)
        freeze.assert_not_called()

    def test_formal_all_stage_is_rejected(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(["--stage", "all", "--confirm-reviewed"])
        args = runner.parse_args(["--stage", "all", "--plan-only"])
        self.assertTrue(args.plan_only)

    def test_method_specific_diagnostic_gates_are_enforced(self):
        eam_job = {
            "method": "eam", "setting": "duet-r2r",
            "parameters": {"batch_size": 8}, "expected_episode_count": 1021,
        }
        eam_adapter = {
            "short_buffer_behavior": "warmup_no_update",
            "warmup_no_update_steps": 7,
            "batch_size_steps": 8,
        }
        runner._validate_implementation_gates(eam_job, {}, eam_adapter)
        eam_adapter["warmup_no_update_steps"] = 8
        with self.assertRaisesRegex(runner.UserError, "warm-up"):
            runner._validate_implementation_gates(eam_job, {}, eam_adapter)

        duet_atena = {
            "method": "atena", "setting": "duet-r2r",
            "parameters": {}, "expected_episode_count": 1021,
        }
        runner._validate_implementation_gates(
            duet_atena, {}, {"self_prediction_feature_dim": 1536}
        )
        with self.assertRaisesRegex(runner.UserError, "1536-D"):
            runner._validate_implementation_gates(
                duet_atena, {}, {"self_prediction_feature_dim": 768}
            )

    def test_source_and_order_provenance_reject_tampering(self):
        ledger = json.loads(
            (REPO_ROOT / "vln/manifests/r2r_reused_source_controls.json")
            .read_text(encoding="utf-8")
        )
        record = ledger["records"]["duet-r2r"]
        runner._validate_source_record(ledger, "val_seen", "duet-r2r", record)

        missing_optional_copy = deepcopy(record)
        missing_optional_copy["formal_manifest_evidence_copy_path"] = (
            "vln/results/logs/ignored-copy-not-required/manifest.json"
        )
        runner._validate_source_record(
            ledger, "val_seen", "duet-r2r", missing_optional_copy
        )

        bad_record = deepcopy(record)
        bad_record["formal_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(runner.UserError, "formal manifest SHA256"):
            runner._validate_source_record(
                ledger, "val_seen", "duet-r2r", bad_record
            )

        bad_metrics = deepcopy(record)
        bad_metrics["metrics"]["SR"] += 1.0
        with self.assertRaisesRegex(runner.UserError, "Source SR metric mismatch"):
            runner._validate_source_record(
                ledger, "val_seen", "duet-r2r", bad_metrics
            )

        formal_path = (REPO_ROOT / record["formal_manifest_path"]).resolve()
        bad_manifest = json.loads(formal_path.read_text(encoding="utf-8"))
        bad_manifest["immutable_identity_sha256"] = "0" * 64
        original_read = runner._read_json

        def forged_read(path):
            if Path(path).resolve() == formal_path:
                return bad_manifest
            return original_read(path)

        with mock.patch.object(runner, "_read_json", side_effect=forged_read):
            with self.assertRaisesRegex(runner.UserError, "immutable identity"):
                runner._validate_source_record(
                    ledger, "val_seen", "duet-r2r", record
                )

        order_path = (
            REPO_ROOT
            / "vln/manifests/episode_order/order_seed_1/"
            "r2r_duet_hamt/val_seen.json"
        )
        order = json.loads(order_path.read_text(encoding="utf-8"))
        for bad_seed in (2, 1.0, True):
            order["order_seed"] = bad_seed
            with self.subTest(bad_seed=bad_seed), self.assertRaisesRegex(
                runner.UserError, "order_seed mismatch"
            ):
                runner._validate_order_document(
                    order, "duet-r2r", "val_seen", 1, 1021
                )
        bindings = {
            setting: {"order_sha256": "a" * 64} for setting in runner.SETTINGS
        }
        bindings["goat-r2r"]["order_sha256"] = "b" * 64
        with self.assertRaisesRegex(runner.UserError, "order hashes differ"):
            runner._validate_paired_order_hashes(bindings, "val_seen", 1)

    def test_retry_identity_and_progress_use_current_persisted_job(self):
        with tempfile.TemporaryDirectory() as directory:
            all_jobs = runner.build_selection_jobs(
                self.spec, "unit-retry-identity", gpu=3, batch_root=directory
            )
            phase = runner.phase_groups(all_jobs)[0]
            runner._ensure_phase_plan(directory, phase, resume=False)
            first = phase[0]
            Path(first["job_dir"], "exitcode").write_text("1\n", encoding="utf-8")
            runner._retry_failed_jobs(phase)
            persisted = runner._load_persisted_stage_jobs(
                directory, "selection", phase
            )
            retried = persisted[0]
            self.assertEqual(retried["attempt"], 1)
            self.assertTrue(retried["run_tag"].endswith("-retry1"))
            self.assertEqual(retried["command"][3], "3")
            runner._validate_job_contract(retried)

            tampered = deepcopy(retried)
            tampered["command"][3] = "0"
            with self.assertRaisesRegex(runner.UserError, "complete registered command"):
                runner._validate_job_contract(tampered)

            Path(retried["job_dir"], "exitcode").write_text("0\n", encoding="utf-8")
            second = persisted[1]
            Path(second["job_dir"], "exitcode").write_text("2\n", encoding="utf-8")
            progress = runner._progress_payload(directory, persisted)
            self.assertEqual(progress["selection"]["succeeded"], 0)
            self.assertEqual(progress["selection"]["invalid"], 1)
            self.assertEqual(progress["selection"]["failed"], 1)

    def test_formal_manifest_seed_is_the_order_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / "results"
            result_root.mkdir()
            artifact = result_root / "valid.txt"
            artifact.write_text("metrics\n", encoding="utf-8")
            run_tag = "unit-order-seed"
            run_id = "{}-duet-r2r-val_seen-native".format(run_tag)
            formal_path = root / "formal" / run_id / "manifest.json"
            formal_path.parent.mkdir(parents=True)
            job = {
                "formal_manifest": str(formal_path),
                "run_tag": run_tag,
                "setting": "duet-r2r",
                "split": "val_seen",
                "expected_benchmark": "r2r_discrete_duet_hamt",
                "model": "duet",
                "method": "tent",
                "order_seed": 2,
                "git_commit": "a" * 40,
                "expected_checkpoint_sha256": "b" * 64,
                "expected_dataset_sha256": "c" * 64,
                "expected_episode_order_sha256": "d" * 64,
                "expected_order_manifest_sha256": "e" * 64,
                "result_root": str(result_root),
            }
            manifest = {
                "run_id": run_id,
                "task": "vln",
                "benchmark": job["expected_benchmark"],
                "model": "duet",
                "method": "tent",
                "run_tag": run_tag,
                "source_setting": "duet-r2r:val_seen:native:tent",
                "seed": 2,
                "git_commit": job["git_commit"],
                "status": "completed",
                "exit_code": 0,
                "checkpoint": {"sha256": job["expected_checkpoint_sha256"]},
                "dataset": {
                    "stream_content_sha256": job["expected_dataset_sha256"],
                    "stream_order_sha256": job["expected_episode_order_sha256"],
                },
                "pinned_manifests": {
                    "episode_order": {"sha256": job["expected_order_manifest_sha256"]}
                },
                "result_artifacts": [{
                    "path": str(artifact),
                    "size": artifact.stat().st_size,
                    "sha256": runner._sha256(artifact),
                }],
            }
            manifest["immutable_identity_sha256"] = (
                runner.immutable_identity_sha256(manifest)
            )
            formal_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch.object(runner, "FORMAL_ROOT", root / "formal"):
                runner._validate_formal_manifest(job, (artifact,))
                manifest["seed"] = 0
                manifest["immutable_identity_sha256"] = (
                    runner.immutable_identity_sha256(manifest)
                )
                formal_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(runner.UserError, "seed mismatch"):
                    runner._validate_formal_manifest(job, (artifact,))

    def test_evaluation_summary_uses_sample_standard_deviation(self):
        results = []
        for setting in runner.SETTINGS:
            source = self.spec["_source_records"]["val_unseen"][setting]["metrics"]
            for method in runner.METHODS:
                for seed, delta in zip((1, 2, 3), (-0.2, 0.2, 0.1)):
                    results.append({
                        "setting": setting,
                        "method": method,
                        "order_seed": seed,
                        "metrics": {
                            "SR": source["SR"] + delta,
                            "SPL": source["SPL"] + delta,
                        },
                        "relative_param_drift": 0.01,
                        "run_tag": "{}-{}-{}".format(setting, method, seed),
                        "formal_manifest_sha256": "f" * 64,
                    })
        aggregates = runner._evaluation_aggregates(self.spec, results)
        self.assertAlmostEqual(
            aggregates["duet-r2r"]["tent"]["aggregate"]["SR"]["std"],
            0.20816659994661327,
        )


if __name__ == "__main__":
    unittest.main()
