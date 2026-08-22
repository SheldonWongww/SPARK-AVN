from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln/scripts"))

import build_r2r_ce_final_registry as builder  # noqa: E402
import run_r2r_ce_val_unseen_frozen_eval as runner  # noqa: E402
from tta_config_cli import translate  # noqa: E402


FINALIZATION_SPEC = (
    REPO_ROOT / "vln/experiments/r2r_ce_postsearch_finalization_v1.json"
)


def _candidate(origin, method, spl, sr, run_tag, updates=10):
    parameters = {
        "tent": {
            "lr": 1e-6, "norm_scope": "last_k_ln", "last_k_ln": 4,
            "update_interval": 1, "optimizer": "AdamW", "weight_decay": 0.0,
            "max_grad_norm": 0.0, "episodic": False,
        },
        "fstta": {
            "lr_fast": 1e-6, "lr_slow": 3e-5, "m": 4, "n": 16,
            "q": 0.1, "rho": 0.95, "tau": 0.7, "a": 0.9, "b": 1.1,
            "optimizer": "AdamW", "beta1": 0.9, "beta2": 0.99,
            "weight_decay": 0.0, "norm_scope": "last_k_ln", "last_k_ln": 4,
            "fast_grad_mode": "concordant", "use_fast_lr_scaler": True,
            "use_slow": True, "reset_optimizer_each_episode": True,
            "reset_slow_optimizer_each_window": False, "max_grad_norm": 0.0,
            "episodic": False,
        },
        "eam": {
            "lr": 1e-6, "memory_size": 32, "batch_size": 8,
            "update_interval": 1, "confidence_scale": 0.4,
            "optimizer": "Adam", "weight_decay": 0.0,
            "max_grad_norm": 0.0, "episodic": False,
        },
        "feedtta": {
            "lr": 1e-6, "scope_profile": "last_crossmodal", "p": 0.05,
            "alpha": -0.2, "gamma": 0.99, "optimizer": "Adam",
            "optimizer_eps": 1e-5, "max_grad_norm": 0.0,
            "action_selection": "argmax", "action_seed": 0, "sgr_seed": 0,
            "episodic": False,
        },
        "atena": {
            "lr_query": 5e-7, "lr_self": 1e-8, "mix_lambda": 0.5,
            "query_threshold": 0.3, "self_loss_weight": 0.1,
            "optimizer": "AdamW", "weight_decay": 0.01,
            "max_grad_norm": 0.0, "action_selection": "argmax",
            "episodic": False,
        },
    }[method]
    return {
        "origin": origin,
        "run_tag": run_tag,
        "parameters": parameters,
        "metrics": {"SR": sr, "SPL": spl},
        "checkpoint_sha256": "a" * 64,
        "formal_manifest_path": "vln/results/runs/{}/manifest.json".format(run_tag),
        "formal_manifest_sha256": "b" * 64,
        "formal_immutable_identity_sha256": "c" * 64,
        "adapter_diagnostics": {"updates": updates},
    }


def _candidate_matrix():
    return {
        setting: {
            method: _candidate(
                "initial_search_full", method, 60.0, 68.0,
                "initial-{}-{}".format(setting, method),
            )
            for method in builder.METHODS
        }
        for setting in builder.SETTINGS
    }


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _supplement_campaign_fixture(root, spec, source_records):
    root = Path(root)
    target_spec = json.loads(
        (REPO_ROOT / "vln/experiments/r2r_ce_targeted_supplement_v1.json")
        .read_text(encoding="utf-8")
    )
    commit = "a" * 40
    phases = []
    confirmations = {}
    for setting_index, setting in enumerate(builder.SETTINGS):
        model = builder.MODEL_FOR_SETTING[setting]
        enabled = [
            method for method in builder.METHODS
            if setting in target_spec["methods"].get(method, {}).get("settings", {})
        ]
        screening = {
            "index": setting_index * 2,
            "phase_id": "{:02d}-{}-screening".format(setting_index * 2, setting),
            "setting": setting,
            "model": model,
            "kind": "screening",
            "stage": "r2r_ce_targeted_screening",
            "planned_jobs": 3 * len(enabled),
            "max_jobs": None,
            "max_workers": 3,
        }
        full = {
            "index": setting_index * 2 + 1,
            "phase_id": "{:02d}-{}-full_confirmation".format(
                setting_index * 2 + 1, setting
            ),
            "setting": setting,
            "model": model,
            "kind": "full_confirmation",
            "stage": "r2r_ce_targeted_full",
            "planned_jobs": None,
            "max_jobs": len(enabled),
            "max_workers": len(enabled),
        }
        phases.extend((screening, full))
        screening_root = root / "phases" / screening["phase_id"]
        full_root = root / "phases" / full["phase_id"]
        for phase, phase_root, count in (
            (screening, screening_root, 3 * len(enabled)),
            (full, full_root, len(enabled)),
        ):
            _write_json(phase_root / "PHASE.json", {
                "schema": "navtta.vln_r2r_ce_targeted_supplement_phase.v1",
                "experiment_id": spec["targeted_supplement"]["experiment_id"],
                "batch_id": spec["targeted_supplement"]["batch_id"],
                "git_commit": commit,
                "spec_path": str(
                    REPO_ROOT / "vln/experiments/r2r_ce_targeted_supplement_v1.json"
                ),
                "spec_sha256": spec["targeted_supplement"]["spec"]["sha256"],
                "canonical_order_seed": 0,
                "canonical_order_sha256": target_spec["canonical_order"][
                    "manifest"
                ]["order_sha256"],
                "phase": phase,
                "source_execution_jobs": 0,
                "restart_from_source_checkpoint": True,
                "job_count": count,
                "job_count_cap": (
                    phase["planned_jobs"]
                    if phase["kind"] == "screening" else phase["max_jobs"]
                ),
                "resource_limits": builder.targeted_runner.runtime_limits(
                    phase, target_spec
                ),
            })
        promotion_cells = {}
        confirmation_cells = {}
        for method in enabled:
            method_spec = target_spec["methods"][method]["settings"][setting]
            parameters = dict(method_spec["fixed"])
            parameters.update(method_spec["candidates"][0])
            screening_tag = "screen-{}-{}".format(setting, method)
            full_tag = "full-{}-{}".format(setting, method)
            promotion_cells[method] = {
                "selected": {
                    "run_tag": screening_tag,
                    "point_index": 0,
                    "parameters": parameters,
                    "metrics": {"SR": 68.0, "SPL": 60.0},
                    "adapter_diagnostics": {"updates": 1},
                    "navigation_record_change_count": 1,
                },
                "selection": {
                    "method": method,
                    "setting": setting,
                    "decision": "fixture",
                    "selected_run_tag": screening_tag,
                    "ranked": [
                        {"run_tag": "{}-{}".format(screening_tag, index)}
                        for index in range(3)
                    ],
                },
            }
            confirmation_cells[method] = {
                "run_tag": full_tag,
                "parent_screening_run_tag": screening_tag,
                "parameters": parameters,
            }
        promotion = {
            "schema": "navtta.vln_r2r_ce_targeted_supplement_promotion.v1",
            "experiment_id": spec["targeted_supplement"]["experiment_id"],
            "batch_id": spec["targeted_supplement"]["batch_id"],
            "setting": setting,
            "git_commit": commit,
            "spec_sha256": spec["targeted_supplement"]["spec"]["sha256"],
            "canonical_prefix_episodes": 100,
            "canonical_order_seed": 0,
            "source": {
                "formal_manifest_sha256": source_records[setting][
                    "formal_manifest_sha256"
                ]
            },
            "cells": promotion_cells,
        }
        _write_json(screening_root / "PROMOTION.json", promotion)
        confirmation = {
            "schema": "navtta.vln_r2r_ce_targeted_supplement_confirmation.v1",
            "experiment_id": spec["targeted_supplement"]["experiment_id"],
            "batch_id": spec["targeted_supplement"]["batch_id"],
            "setting": setting,
            "git_commit": commit,
            "spec_sha256": spec["targeted_supplement"]["spec"]["sha256"],
            "canonical_full_episodes": 778,
            "canonical_order_seed": 0,
            "restart_from_source_checkpoint": True,
            "source": {
                "formal_manifest_sha256": source_records[setting][
                    "formal_manifest_sha256"
                ]
            },
            "cells": confirmation_cells,
        }
        _write_json(full_root / "CONFIRMATION.json", confirmation)
        confirmations[setting] = confirmation
    plan = {
        "schema": "navtta.vln_r2r_ce_targeted_supplement_plan.v1",
        "experiment_id": spec["targeted_supplement"]["experiment_id"],
        "batch_id": spec["targeted_supplement"]["batch_id"],
        "benchmark": spec["protocol"]["benchmark"],
        "split": "val_seen",
        "git_commit": commit,
        "spec_path": str(
            REPO_ROOT / "vln/experiments/r2r_ce_targeted_supplement_v1.json"
        ),
        "spec_sha256": spec["targeted_supplement"]["spec"]["sha256"],
        "canonical_order_seed": 0,
        "canonical_order_sha256": target_spec["canonical_order"][
            "manifest"
        ]["order_sha256"],
        "source_execution_jobs": 0,
        "screening_jobs": 15,
        "full_jobs_max": 5,
        "total_jobs_max": 20,
        "strict_model_barrier": True,
        "phases": phases,
    }
    _write_json(root / "PLAN.json", plan)
    results = {
        "schema": builder.SUPPLEMENT_SCHEMA,
        "experiment_id": spec["targeted_supplement"]["experiment_id"],
        "batch_id": spec["targeted_supplement"]["batch_id"],
        "benchmark": spec["protocol"]["benchmark"],
        "split": "val_seen",
        "git_commit": commit,
        "spec_sha256": spec["targeted_supplement"]["spec"]["sha256"],
        "canonical_order_seed": 0,
        "source_execution_jobs": 0,
        "supervision_groups": target_spec["protocol"]["supervision_groups"],
        "settings": confirmations,
    }
    _write_json(root / "RESULTS.json", results)
    updated_spec = deepcopy(spec)
    updated_spec["targeted_supplement"]["results_relative_path"] = (
        (root / "RESULTS.json").relative_to(REPO_ROOT).as_posix()
    )
    return updated_spec, target_spec, results


class R2RCEPostSearchFinalizationTest(unittest.TestCase):
    def test_workspace_spec_is_valid_but_freeze_waits_for_supplement(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        self.assertEqual(spec["status"], "awaiting_targeted_supplement")
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "RESULTS.json"
            with self.assertRaisesRegex(
                builder.RegistryError, "invalid targeted supplement results"
            ):
                builder.build_documents(
                    FINALIZATION_SPEC, supplement_results=missing
                )

    def test_existing_val_unseen_sources_are_reused_and_complete(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        _, ledger = builder.validate_val_unseen_source_ledger(
            spec, require_artifacts=False
        )
        self.assertEqual(ledger["episode_count"], 1839)
        self.assertEqual(set(ledger["records"]), set(builder.SETTINGS))
        self.assertTrue(all(
            item["evidence_status"] == "ready"
            for item in ledger["records"].values()
        ))
        self.assertEqual(
            ledger["source_execution_policy"]["execution"], "reuse_only"
        )
        self.assertTrue(ledger["source_execution_policy"]["rerun_forbidden"])

    def test_tracked_initial_inventory_has_ten_translatable_candidates(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        _, source = builder._val_seen_source_records(spec, REPO_ROOT)
        candidates = builder._initial_candidates(spec, REPO_ROOT, source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for setting in builder.SETTINGS:
                self.assertEqual(set(candidates[setting]), set(builder.METHODS))
                for method in builder.METHODS:
                    config = root / "{}-{}.json".format(setting, method)
                    config.write_text(json.dumps({
                        "method": method,
                        "parameters": candidates[setting][method]["parameters"],
                    }), encoding="utf-8")
                    translated, _ = translate(
                        setting, config,
                        root / "{}-{}-diagnostics.json".format(setting, method),
                    )
                    self.assertEqual(translated, method)

    def test_selection_uses_only_same_cell_val_seen_candidates(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        initial = _candidate_matrix()
        supplement = {setting: {} for setting in builder.SETTINGS}
        supplement["etpnav-r2r-ce"]["tent"] = _candidate(
            "targeted_supplement_full", "tent", 61.0, 67.5, "supplement-tent"
        )
        supplement["etpnav-r2r-ce"]["fstta"] = _candidate(
            "targeted_supplement_full", "fstta", 60.5, 68.0, "supplement-fstta"
        )
        supplement["bevbert-r2r-ce"]["fstta"] = _candidate(
            "targeted_supplement_full", "fstta", 60.5, 68.0, "supplement-bev-fstta"
        )
        for setting in builder.SETTINGS:
            supplement[setting]["feedtta"] = _candidate(
                "targeted_supplement_full", "feedtta", 60.0, 68.0,
                "supplement-{}-feedtta".format(setting),
            )
        # Reproduce the invalid zero-update M=16 first-round FSTTA incumbent.
        for setting in builder.SETTINGS:
            initial[setting]["fstta"]["parameters"]["m"] = 16
            initial[setting]["fstta"]["adapter_diagnostics"]["updates"] = 0

        source = {
            setting: {"metrics": {"SR": 68.0, "SPL": 59.0}}
            for setting in builder.SETTINGS
        }
        winners, decisions = builder.select_winners(
            spec, initial, supplement, source
        )
        self.assertEqual(
            winners["etpnav-r2r-ce"]["tent"]["run_tag"], "supplement-tent"
        )
        self.assertEqual(
            winners["etpnav-r2r-ce"]["feedtta"]["origin"],
            "targeted_supplement_full",
        )
        self.assertEqual(
            winners["bevbert-r2r-ce"]["eam"]["origin"],
            "initial_search_full",
        )
        fstta_review = decisions["etpnav-r2r-ce"]["fstta"]["ranked_candidates"]
        self.assertIn("invalid_fast_window", fstta_review[0]["reasons"])

    def test_invalid_fstta_cannot_be_frozen_when_supplement_promotes_none(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        initial = _candidate_matrix()
        for setting in builder.SETTINGS:
            initial[setting]["fstta"]["parameters"]["m"] = 16
        source = {
            setting: {"metrics": {"SR": 68.0, "SPL": 59.0}}
            for setting in builder.SETTINGS
        }
        with self.assertRaisesRegex(
            builder.RegistryError, "further search is required"
        ):
            builder.select_winners(
                spec,
                initial,
                {setting: {} for setting in builder.SETTINGS},
                source,
            )

    def test_all_five_target_cells_require_authenticated_dispositions(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        _, source = builder._val_seen_source_records(spec, REPO_ROOT)
        parent = REPO_ROOT / "vln/results/logs/r2r-ce/hparam_search"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            campaign = Path(directory)
            candidate_spec, target_spec, results = _supplement_campaign_fixture(
                campaign, spec, source
            )
            job_evidence = {
                tuple(cell): {"screening_jobs": []}
                for cell in candidate_spec["targeted_cells"]
            }
            with mock.patch.object(
                builder, "_authenticate_supplement_jobs",
                return_value=job_evidence,
            ):
                evidence = builder._validate_supplement_campaign_artifacts(
                    campaign / "RESULTS.json", results, candidate_spec,
                    target_spec, REPO_ROOT, source,
                )
            self.assertEqual(len(evidence["cells"]), 5)
            self.assertTrue(all(
                item["status"] == "full_confirmed"
                and item["promotion_sha256"]
                and item["confirmation_sha256"]
                for item in evidence["cells"].values()
            ))

            promotion_path = (
                campaign / "phases/00-etpnav-r2r-ce-screening/PROMOTION.json"
            )
            promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
            promotion["cells"].pop("tent")
            _write_json(promotion_path, promotion)
            with mock.patch.object(
                builder, "_authenticate_supplement_jobs",
                return_value=job_evidence,
            ):
                with self.assertRaisesRegex(
                    builder.RegistryError, "PROMOTION cell set mismatch"
                ):
                    builder._validate_supplement_campaign_artifacts(
                        campaign / "RESULTS.json", results, candidate_spec,
                        target_spec, REPO_ROOT, source,
                    )

    def test_targeted_disposition_requires_real_job_level_evidence(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        _, source = builder._val_seen_source_records(spec, REPO_ROOT)
        parent = REPO_ROOT / "vln/results/logs/r2r-ce/hparam_search"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            campaign = Path(directory)
            candidate_spec, target_spec, results = _supplement_campaign_fixture(
                campaign, spec, source
            )
            with self.assertRaisesRegex(
                builder.RegistryError, "screening has no persisted jobs"
            ):
                builder._validate_supplement_campaign_artifacts(
                    campaign / "RESULTS.json", results, candidate_spec,
                    target_spec, REPO_ROOT, source,
                )

    def test_targeted_job_loader_replays_the_producer_contract(self):
        target_spec = builder.targeted_runner.load_spec(
            REPO_ROOT / "vln/experiments/r2r_ce_targeted_supplement_v1.json"
        )
        phase = builder.targeted_runner.phase_sequence(target_spec)[0]
        parent = REPO_ROOT / "vln/results/logs/r2r-ce/hparam_search"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            campaign = Path(directory)
            with mock.patch.object(
                builder.targeted_runner, "LOG_ROOT", campaign.parent
            ), mock.patch.object(
                builder.targeted_runner, "TUNING_ROOT", campaign / "tuning"
            ):
                jobs = builder.targeted_runner.build_screening_jobs(
                    phase, campaign.name, target_spec, gpu=2
                )
                for job in jobs:
                    builder.staged_runner._write_job(job)
                    (Path(job["job_dir"]) / "exitcode").write_text(
                        "0\n", encoding="utf-8"
                    )
                    _write_json(Path(job["job_dir"]) / "metrics.json", {})
                loaded = builder._load_targeted_phase_jobs(
                    campaign / "phases" / phase["phase_id"], jobs,
                    REPO_ROOT,
                )
                self.assertEqual(
                    [item["base_run_tag"] for item in loaded],
                    [item["base_run_tag"] for item in jobs],
                )
                path = Path(jobs[0]["job_dir"]) / "job.json"
                changed = json.loads(path.read_text(encoding="utf-8"))
                changed["command"][3] = "7"
                _write_json(path, changed)
                with self.assertRaisesRegex(
                    builder.RegistryError, "targeted job command changed"
                ):
                    builder._load_targeted_phase_jobs(
                        campaign / "phases" / phase["phase_id"], jobs,
                        REPO_ROOT,
                    )

    def test_promotion_ranking_is_replayed_from_screening_results(self):
        target_spec = builder.targeted_runner.load_spec(
            REPO_ROOT / "vln/experiments/r2r_ce_targeted_supplement_v1.json"
        )
        target_spec["_batch_id"] = "promotion-replay-unit"
        phases = builder.targeted_runner.phase_sequence(target_spec)
        promotions = {}
        confirmations = {}
        synthetic = {}
        with tempfile.TemporaryDirectory(
            dir=REPO_ROOT / "vln/results/logs/r2r-ce/hparam_search"
        ) as directory:
            campaign = Path(directory)
            for setting_index, setting in enumerate(builder.SETTINGS):
                screening = phases[setting_index * 2]
                dummy = campaign / "phases" / screening["phase_id"] / "jobs/dummy/job.json"
                _write_json(dummy, {"command": ["runner", setting, "val_seen", "0"]})
                jobs = builder.targeted_runner.build_screening_jobs(
                    screening, target_spec["_batch_id"], target_spec, 0
                )
                source = builder.staged_runner.reused_source_result(
                    setting, 100, target_spec
                )
                cells = {}
                for job in jobs:
                    synthetic[job["base_run_tag"]] = {
                        **job,
                        "metrics": {
                            "SR": source["metrics"]["SR"] - 10.0,
                            "SPL": source["metrics"]["SPL"],
                        },
                        "adapter_diagnostics": {
                            "updates": 1, "relative_param_drift": 0.0,
                        },
                        "navigation_record_change_count": 0,
                    }
                for method in builder.targeted_runner.enabled_methods(
                    setting, target_spec
                ):
                    values = [
                        synthetic[job["base_run_tag"]]
                        for job in jobs if job["config_method"] == method
                    ]
                    selected, selection = builder.targeted_runner.select_finalist(
                        method, setting, values, source, target_spec
                    )
                    self.assertIsNone(selected)
                    cells[method] = {"selected": None, "selection": selection}
                promotions[setting] = {
                    "setting": setting,
                    "source": {
                        "run_tag": source["run_tag"],
                        "metrics": source["metrics"],
                        "formal_manifest": source["reused_formal_manifest"],
                        "formal_manifest_sha256": source[
                            "reused_formal_manifest_sha256"
                        ],
                    },
                    "cells": cells,
                }
                confirmations[setting] = {"cells": {}}

            def phase_jobs(_root, expected, _repo_root):
                return expected

            def result(job, _spec):
                return synthetic[job["base_run_tag"]]

            with mock.patch.object(
                builder, "_load_targeted_phase_jobs", side_effect=phase_jobs
            ), mock.patch.object(
                builder, "_screening_result", side_effect=result
            ), mock.patch.object(
                builder, "_targeted_job_evidence",
                side_effect=lambda job, _root: {"run_tag": job["run_tag"]},
            ):
                evidence = builder._authenticate_supplement_jobs(
                    campaign, phases, target_spec, promotions,
                    confirmations, REPO_ROOT,
                )
                self.assertEqual(len(evidence), 5)
                promotions["etpnav-r2r-ce"]["cells"]["tent"]["selection"][
                    "ranked"
                ][0]["run_tag"] = "invented"
                with self.assertRaisesRegex(
                    builder.RegistryError,
                    "PROMOTION does not replay from screening jobs",
                ):
                    builder._authenticate_supplement_jobs(
                        campaign, phases, target_spec, promotions,
                        confirmations, REPO_ROOT,
                    )

    def test_promotion_requires_confirmation_or_empty_completion(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        _, source = builder._val_seen_source_records(spec, REPO_ROOT)
        parent = REPO_ROOT / "vln/results/logs/r2r-ce/hparam_search"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            campaign = Path(directory)
            candidate_spec, target_spec, results = _supplement_campaign_fixture(
                campaign, spec, source
            )
            job_evidence = {
                tuple(cell): {"screening_jobs": []}
                for cell in candidate_spec["targeted_cells"]
            }
            confirmation_path = (
                campaign / "phases/01-etpnav-r2r-ce-full_confirmation/CONFIRMATION.json"
            )
            confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
            confirmation["cells"].pop("tent")
            results["settings"]["etpnav-r2r-ce"] = confirmation
            _write_json(confirmation_path, confirmation)
            _write_json(campaign / "RESULTS.json", results)
            with mock.patch.object(
                builder, "_authenticate_supplement_jobs",
                return_value=job_evidence,
            ):
                with self.assertRaisesRegex(
                    builder.RegistryError, "promotion lacks confirmation"
                ):
                    builder._validate_supplement_campaign_artifacts(
                        campaign / "RESULTS.json", results, candidate_spec,
                        target_spec, REPO_ROOT, source,
                    )

        with tempfile.TemporaryDirectory(dir=parent) as directory:
            campaign = Path(directory)
            candidate_spec, target_spec, results = _supplement_campaign_fixture(
                campaign, spec, source
            )
            setting = "bevbert-r2r-ce"
            promotion_path = campaign / (
                "phases/02-bevbert-r2r-ce-screening/PROMOTION.json"
            )
            promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
            for cell in promotion["cells"].values():
                cell["selected"] = None
                cell["selection"]["selected_run_tag"] = None
            _write_json(promotion_path, promotion)
            confirmation_path = campaign / (
                "phases/03-bevbert-r2r-ce-full_confirmation/CONFIRMATION.json"
            )
            confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
            confirmation["cells"] = {}
            _write_json(confirmation_path, confirmation)
            results["settings"][setting] = confirmation
            _write_json(campaign / "RESULTS.json", results)
            phase_path = confirmation_path.parent / "PHASE.json"
            phase = json.loads(phase_path.read_text(encoding="utf-8"))
            phase["job_count"] = 0
            _write_json(phase_path, phase)
            with self.assertRaisesRegex(builder.RegistryError, "EMPTY_COMPLETE"):
                builder._validate_supplement_campaign_artifacts(
                    campaign / "RESULTS.json", results, candidate_spec,
                    target_spec, REPO_ROOT, source,
                )

    def test_plan_and_phase_runtime_contracts_fail_closed(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        _, source = builder._val_seen_source_records(spec, REPO_ROOT)
        parent = REPO_ROOT / "vln/results/logs/r2r-ce/hparam_search"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            campaign = Path(directory)
            candidate_spec, target_spec, results = _supplement_campaign_fixture(
                campaign, spec, source
            )
            plan_path = campaign / "PLAN.json"
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["canonical_order_sha256"] = "0" * 64
            _write_json(plan_path, plan)
            with self.assertRaisesRegex(
                builder.RegistryError, "PLAN canonical_order_sha256 mismatch"
            ):
                builder._validate_supplement_campaign_artifacts(
                    campaign / "RESULTS.json", results, candidate_spec,
                    target_spec, REPO_ROOT, source,
                )

    def test_fstta_eligibility_uses_authenticated_diagnostics_updates(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        _, source = builder._val_seen_source_records(spec, REPO_ROOT)
        initial = builder._initial_candidates(spec, REPO_ROOT, source)
        candidate = deepcopy(initial["etpnav-r2r-ce"]["fstta"])
        self.assertEqual(candidate["adapter_diagnostics"]["updates"], 0)
        candidate["parameters"]["m"] = 4
        eligible, reasons = builder._eligible(
            candidate, "fstta", source["etpnav-r2r-ce"]["metrics"], spec
        )
        self.assertFalse(eligible)
        self.assertIn("no_effective_updates", reasons)

    def test_supplement_results_diagnostics_must_match_formal_artifact(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        _, source = builder._val_seen_source_records(spec, REPO_ROOT)
        parent = REPO_ROOT / "vln/results/logs/r2r-ce/hparam_search"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            campaign = Path(directory)
            candidate_spec, _, results = _supplement_campaign_fixture(
                campaign, spec, source
            )
            aggregate_path = campaign / "fake/stats_ckpt_1.json"
            aggregate_path.parent.mkdir(parents=True)
            aggregate_path.write_text("{}\n", encoding="utf-8")
            for setting, confirmation in results["settings"].items():
                for method, cell in confirmation["cells"].items():
                    run_id = "{}-{}-val_seen-v1.3-unified".format(
                        cell["run_tag"], setting
                    )
                    cell.update({
                        "metrics": {"SR": 68.0, "SPL": 60.0},
                        "adapter_diagnostics": {"updates": 1},
                        "job_config_path": str(campaign / "parameters.json"),
                        "job_config_sha256": "d" * 64,
                        "aggregate_artifact_path": str(aggregate_path),
                        "aggregate_artifact_sha256": "e" * 64,
                        "formal_manifest_path": str(
                            REPO_ROOT / "vln/results/runs" / run_id / "manifest.json"
                        ),
                        "formal_manifest_sha256": "b" * 64,
                        "formal_immutable_identity_sha256": "c" * 64,
                    })
                confirmation_path = campaign / "phases" / (
                    "01-etpnav-r2r-ce-full_confirmation"
                    if setting == "etpnav-r2r-ce"
                    else "03-bevbert-r2r-ce-full_confirmation"
                ) / "CONFIRMATION.json"
                _write_json(confirmation_path, confirmation)
            _write_json(campaign / "RESULTS.json", results)
            job_evidence = {
                tuple(cell): {"screening_jobs": []}
                for cell in candidate_spec["targeted_cells"]
            }
            fake_manifest = {
                "git_commit": results["git_commit"],
                "immutable_identity_sha256": "c" * 64,
                "result_artifacts": [{
                    "name": "metrics/source_val_seen/stats_ckpt_1.json",
                    "path": str(aggregate_path),
                    "size": aggregate_path.stat().st_size,
                    "sha256": "e" * 64,
                }],
            }
            with mock.patch.object(
                builder, "_authenticate_supplement_jobs",
                return_value=job_evidence,
            ), mock.patch.object(
                builder, "_validate_manifest",
                return_value=("formal/manifest.json", fake_manifest),
            ), mock.patch.object(
                builder, "_validate_manifest_parameters"
            ), mock.patch.object(
                builder, "_validate_aggregate_metrics",
                return_value={"SR": 68.0, "SPL": 60.0},
            ), mock.patch.object(
                builder, "_validate_diagnostics",
                return_value={
                    "path": "diagnostics.json", "sha256": "f" * 64,
                    "adapter": {"updates": 2},
                },
            ):
                with self.assertRaisesRegex(
                    builder.RegistryError,
                    "RESULTS diagnostics disagree with authenticated artifact",
                ):
                    builder._load_supplement(
                        campaign / "RESULTS.json", candidate_spec,
                        REPO_ROOT, source,
                    )

    def test_initial_inventory_is_bound_to_original_final_selection_files(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        _, source = builder._val_seen_source_records(spec, REPO_ROOT)
        inventory_path = REPO_ROOT / spec["initial_search"]["candidate_inventory"]["path"]
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(
            dir=REPO_ROOT / "vln/results/logs"
        ) as directory:
            changed = deepcopy(inventory)
            changed["records"]["etpnav-r2r-ce"]["tent"]["parameters"]["lr"] = 9e-3
            path = Path(directory) / "initial.json"
            _write_json(path, changed)
            candidate_spec = deepcopy(spec)
            candidate_spec["initial_search"]["candidate_inventory"] = {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "sha256": builder._sha256(path),
            }
            with self.assertRaisesRegex(
                builder.RegistryError, "inventory differs from authenticated"
            ):
                builder._initial_candidates(candidate_spec, REPO_ROOT, source)

    def test_initial_selection_source_provenance_is_authenticated(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        _, source = builder._val_seen_source_records(spec, REPO_ROOT)
        inventory_path = REPO_ROOT / spec["initial_search"]["candidate_inventory"]["path"]
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(
            dir=REPO_ROOT / "vln/results/logs"
        ) as directory:
            root = Path(directory)
            original_path = REPO_ROOT / inventory["source_selection_files"]["tent"]["path"]
            selection = json.loads(original_path.read_text(encoding="utf-8"))
            selection["settings"]["etpnav-r2r-ce"]["source_provenance"][
                "per_episode_artifact_sha256"
            ] = "0" * 64
            changed_selection = root / "FINAL_SELECTION.json"
            _write_json(changed_selection, selection)
            inventory["source_selection_files"]["tent"] = {
                "path": changed_selection.relative_to(REPO_ROOT).as_posix(),
                "sha256": builder._sha256(changed_selection),
            }
            changed_inventory = root / "initial.json"
            _write_json(changed_inventory, inventory)
            candidate_spec = deepcopy(spec)
            candidate_spec["initial_search"]["candidate_inventory"] = {
                "path": changed_inventory.relative_to(REPO_ROOT).as_posix(),
                "sha256": builder._sha256(changed_inventory),
            }
            with self.assertRaisesRegex(
                builder.RegistryError, "FINAL_SELECTION provenance mismatch"
            ):
                builder._initial_candidates(candidate_spec, REPO_ROOT, source)

    def test_val_seen_source_dataset_and_artifact_tampering_is_rejected(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        ledger_path = REPO_ROOT / spec["val_seen_source_control"]["path"]
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(
            dir=REPO_ROOT / "vln/results/logs"
        ) as directory:
            root = Path(directory)
            changed = deepcopy(ledger)
            changed["settings"]["etpnav-r2r-ce"]["aggregate_artifact"][
                "sha256"
            ] = "0" * 64
            changed_path = root / "source.json"
            _write_json(changed_path, changed)
            candidate_spec = deepcopy(spec)
            candidate_spec["val_seen_source_control"] = {
                "path": changed_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": builder._sha256(changed_path),
            }
            with self.assertRaisesRegex(
                builder.RegistryError, "digest-mismatched"
            ):
                builder._val_seen_source_records(candidate_spec, REPO_ROOT)

        source = ledger["settings"]["etpnav-r2r-ce"]
        run_tag = source["run_id"].rsplit("-etpnav-r2r-ce-val_seen", 1)[0]
        record = {
            "run_tag": run_tag,
            "checkpoint_sha256": source["checkpoint_sha256"],
            "formal_manifest_path": source["formal_manifest"]["path"],
            "formal_manifest_sha256": source["formal_manifest"]["sha256"],
            "episode_order_sha256": spec["protocol"]["selection_order_sha256"],
        }
        manifest_path = REPO_ROOT / source["formal_manifest"]["path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset"]["stream_content_sha256"] = "0" * 64
        with mock.patch.object(builder, "_read_json", return_value=manifest):
            with self.assertRaisesRegex(
                builder.RegistryError, "val_seen content digest mismatch"
            ):
                builder._validate_source_manifest(
                    REPO_ROOT, record, "etpnav-r2r-ce", "val_seen"
                )

    def test_generated_frozen_spec_is_ten_tta_zero_source_with_three_workers(self):
        _, spec = builder.load_spec(FINALIZATION_SPEC)
        initial = _candidate_matrix()
        registry = {
            "records": {
                setting: {
                    **{
                        method: {
                            **initial[setting][method],
                            "method": method,
                        }
                        for method in builder.METHODS
                    },
                    "source": {
                        "checkpoint_sha256": "a" * 64,
                        "formal_manifest_path": "source",
                    },
                }
                for setting in builder.SETTINGS
            }
        }
        frozen = builder._frozen_spec(
            spec,
            registry,
            REPO_ROOT / "vln/results/final/r2r-ce/registry.json",
            "d" * 64,
            REPO_ROOT / "vln/manifests/r2r_ce_val_unseen_reused_source_controls.json",
        )
        self.assertEqual(
            frozen["budget"],
            {"source_execution_jobs": 0, "tta_jobs": 10, "total_executed_jobs": 10},
        )
        self.assertEqual(frozen["matrix"]["max_workers"], 3)
        self.assertEqual(frozen["canonical_batch_id"], runner.CANONICAL_BATCH_ID)
        self.assertFalse(frozen["protocol"]["selection_on_val_unseen"])
        self.assertEqual(
            [(item["setting"], item["method"]) for item in frozen["matrix"]["jobs"]],
            [
                (setting, method)
                for setting in builder.SETTINGS
                for method in builder.METHODS
            ],
        )
        self.assertTrue(all(
            "parameters" not in item and "metrics" not in item
            for item in frozen["matrix"]["jobs"]
        ))
        with tempfile.TemporaryDirectory(
            dir=REPO_ROOT / "vln/results/logs"
        ) as directory:
            root = Path(directory)
            registry_path = root / "registry.json"
            source_path = root / "source.json"
            registry_path.write_text("{}\n", encoding="utf-8")
            source_path.write_text("{}\n", encoding="utf-8")
            frozen["registry_dependency"]["path"] = registry_path.relative_to(
                REPO_ROOT
            ).as_posix()
            frozen["registry_dependency"]["sha256"] = runner._sha256(
                registry_path
            )
            frozen["source_control"]["manifest"] = source_path.relative_to(
                REPO_ROOT
            ).as_posix()
            frozen["source_control"]["sha256"] = runner._sha256(source_path)
            runner._validate_spec_structure(frozen)
            duplicate = deepcopy(frozen)
            duplicate["matrix"]["jobs"][1] = deepcopy(
                duplicate["matrix"]["jobs"][0]
            )
            with self.assertRaisesRegex(runner.UserError, "model-major 2x5"):
                runner._validate_spec_structure(duplicate)

    def test_registry_rejects_any_change_from_deterministic_replay(self):
        initial = _candidate_matrix()
        source_records = {
            setting: {
                "run_tag": "source-{}".format(setting),
                "metrics": {"SR": 68.0, "SPL": 59.0},
                "checkpoint_sha256": "a" * 64,
                "episode_order_sha256": (
                    "93f44aab1be2e3d96b867a323172ab3bbbaa4dd97e5b3fa450fe839a6c4bd94e"
                ),
                "formal_manifest_path": "vln/results/runs/source-{}/manifest.json".format(setting),
                "formal_manifest_sha256": "b" * 64,
                "immutable_identity_sha256": "c" * 64,
            }
            for setting in builder.SETTINGS
        }
        records = {
            setting: {
                method: {
                    "selection_status": "ready",
                    "supervision_category": builder.SUPERVISION[method],
                    "selected_origin": initial[setting][method]["origin"],
                    "parameters": initial[setting][method]["parameters"],
                    "run_tag": initial[setting][method]["run_tag"],
                    "metrics": initial[setting][method]["metrics"],
                    "formal_manifest_path": initial[setting][method]["formal_manifest_path"],
                    "formal_manifest_sha256": initial[setting][method]["formal_manifest_sha256"],
                    "formal_immutable_identity_sha256": initial[setting][method]["formal_immutable_identity_sha256"],
                    "decision": {},
                }
                for method in builder.METHODS
            }
            for setting in builder.SETTINGS
        }
        with tempfile.TemporaryDirectory(
            dir=REPO_ROOT / "vln/results/logs"
        ) as directory:
            root = Path(directory)
            source_path = root / "source.json"
            _write_json(source_path, {
                "schema": builder.VAL_SEEN_SOURCE_SCHEMA,
                "settings": {
                    setting: {
                        "checkpoint_sha256": source_records[setting]["checkpoint_sha256"],
                        "formal_manifest": {
                            "sha256": source_records[setting]["formal_manifest_sha256"]
                        },
                    }
                    for setting in builder.SETTINGS
                },
            })
            selection_path = root / "selected.json"
            selection = {
                "schema": builder.SELECTION_SCHEMA,
                "selection_status": "complete",
                "protocol": {
                    "benchmark": "r2r_ce_v1_3_unified_etpnav_bevbert",
                    "split": "val_seen",
                    "episode_count": 778,
                    "order_seed": 0,
                    "episode_order_sha256": "93f44aab1be2e3d96b867a323172ab3bbbaa4dd97e5b3fa450fe839a6c4bd94e",
                    "selection_scope": "val_seen_seed0_initial_plus_targeted_supplement",
                    "selection_on_val_unseen": False,
                },
                "records": records,
            }
            _write_json(selection_path, selection)
            registry = builder._registry_document(
                selection, selection_path, source_records, source_path
            )
            registry_path = root / "registry.json"
            _write_json(registry_path, registry)
            def replay(_path, observed, _repo_root):
                if observed != selection:
                    raise builder.RegistryError(
                        "selected winners do not deterministically replay"
                    )
                return selection, source_path, source_records

            with mock.patch.object(
                builder, "_rebuild_selection", side_effect=replay,
            ), mock.patch.object(builder, "_validate_source_manifest"), \
                    mock.patch.object(builder, "_validate_manifest"):
                builder.validate_registry(registry_path)
                changed = deepcopy(registry)
                changed["records"]["etpnav-r2r-ce"]["tent"]["parameters"]["lr"] = 9e-3
                changed_selection = deepcopy(selection)
                changed_selection["records"]["etpnav-r2r-ce"]["tent"][
                    "parameters"
                ]["lr"] = 9e-3
                _write_json(selection_path, changed_selection)
                changed["selected_winners"]["sha256"] = builder._sha256(
                    selection_path
                )
                _write_json(registry_path, changed)
                with self.assertRaisesRegex(
                    builder.RegistryError, "deterministically replay",
                ):
                    builder.validate_registry(registry_path)

    def test_selection_rebuild_replays_metrics_parameters_and_eligibility(self):
        _, base_spec = builder.load_spec(FINALIZATION_SPEC)
        initial = _candidate_matrix()
        initial["etpnav-r2r-ce"]["fstta"]["parameters"]["m"] = 16
        initial["etpnav-r2r-ce"]["fstta"]["adapter_diagnostics"]["updates"] = 0
        supplement = {setting: {} for setting in builder.SETTINGS}
        supplement["etpnav-r2r-ce"]["fstta"] = _candidate(
            "targeted_supplement_full", "fstta", 61.0, 68.0,
            "authenticated-supplement-fstta", updates=7,
        )
        source_records = {
            setting: {
                "metrics": {"SR": 68.0, "SPL": 59.0},
                "checkpoint_sha256": "a" * 64,
            }
            for setting in builder.SETTINGS
        }
        winners, decisions = builder.select_winners(
            base_spec, initial, supplement, source_records
        )
        parent = REPO_ROOT / "vln/results/logs"
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            root = Path(directory)
            spec_path = root / "spec.json"
            results_path = root / "RESULTS.json"
            source_path = root / "source.json"
            selection_path = root / "selected.json"
            for path in (spec_path, results_path, source_path):
                _write_json(path, {})
            spec = deepcopy(base_spec)
            spec["outputs"]["selected_winners"] = selection_path.relative_to(
                REPO_ROOT
            ).as_posix()
            supplement_document = {"git_commit": "a" * 40}
            dispositions = {"cells": {}}
            selection = builder._selection_document(
                spec_path, spec, results_path, supplement_document,
                dispositions, winners, decisions, source_path, REPO_ROOT,
            )
            _write_json(selection_path, selection)

            def load_supplement(*_args):
                return (
                    results_path, supplement_document, supplement,
                    dispositions,
                )

            with mock.patch.object(
                builder, "load_spec", return_value=(spec_path, spec)
            ), mock.patch.object(
                builder, "_val_seen_source_records",
                return_value=(source_path, source_records),
            ), mock.patch.object(
                builder, "_initial_candidates", return_value=initial
            ), mock.patch.object(
                builder, "_load_supplement", side_effect=load_supplement
            ):
                rebuilt, _, _ = builder._rebuild_selection(
                    selection_path, selection, REPO_ROOT
                )
                self.assertEqual(rebuilt, selection)
                changed = deepcopy(selection)
                changed["records"]["etpnav-r2r-ce"]["fstta"][
                    "parameters"
                ]["m"] = 3
                with self.assertRaisesRegex(
                    builder.RegistryError, "deterministically replay"
                ):
                    builder._rebuild_selection(
                        selection_path, changed, REPO_ROOT
                    )
                supplement["etpnav-r2r-ce"]["fstta"][
                    "adapter_diagnostics"
                ]["updates"] = 0
                with self.assertRaisesRegex(
                    builder.RegistryError, "further search is required"
                ):
                    builder._rebuild_selection(
                        selection_path, selection, REPO_ROOT
                    )

    def test_runner_materializes_exact_frozen_parameters_without_source(self):
        initial = _candidate_matrix()
        registry = {
            "records": {
                setting: {
                    **{
                        method: {
                            **initial[setting][method],
                            "method": method,
                        }
                        for method in builder.METHODS
                    },
                    "source": {"checkpoint_sha256": "a" * 64},
                }
                for setting in builder.SETTINGS
            }
        }
        jobs_binding = [
            {
                "setting": setting,
                "model": builder.MODEL_FOR_SETTING[setting],
                "method": method,
                "selected_run_tag": initial[setting][method]["run_tag"],
                "selected_formal_manifest_sha256": "b" * 64,
            }
            for setting in builder.SETTINGS
            for method in builder.METHODS
        ]
        with tempfile.TemporaryDirectory(
            dir=REPO_ROOT / "vln/results/logs"
        ) as source_directory, tempfile.TemporaryDirectory() as directory:
            source_path = Path(source_directory) / "source.json"
            ledger = {
                "records": {
                    setting: {
                        "checkpoint_sha256": "a" * 64,
                        "dataset_sha256": "1" * 64,
                        "episode_order_sha256": "f" * 64,
                    }
                    for setting in builder.SETTINGS
                }
            }
            source_path.write_text(json.dumps(ledger), encoding="utf-8")
            spec = {
            "registry_dependency": {"path": "registry", "sha256": "d" * 64},
            "source_control": {
                "manifest": source_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": runner._sha256(source_path),
                "execution": "reuse_only",
                "rerun_forbidden": True,
            },
            "protocol": {
                "episode_order_sha256": "f" * 64,
                "order_manifest": {
                    "path": "vln/manifests/episode_order/r2r_ce_v1_3_unified/val_unseen.json",
                    "sha256": runner._sha256(
                        REPO_ROOT / "vln/manifests/episode_order/r2r_ce_v1_3_unified/val_unseen.json"
                    ),
                },
            },
            "matrix": {"jobs": jobs_binding},
            }
            with mock.patch.object(runner, "load_registry", return_value=(Path("registry"), registry)), \
                    mock.patch.object(runner, "validate_source_ledger", return_value=ledger), \
                    mock.patch.object(runner, "_git_commit", return_value="2" * 40):
                jobs = runner.expand_jobs(spec, "unit-r2r-ce", gpu=0)
                self.assertEqual(len(jobs), 10)
                self.assertEqual([len(item) for item in runner.model_phases(jobs)], [5, 5])
                self.assertNotIn("source", {item["method"] for item in jobs})
                for job in jobs:
                    attempt_dir, metadata = runner.materialize_attempt(
                        spec, Path(__file__), "unit-r2r-ce", Path(directory), job, 0
                    )
                    config = json.loads(
                        (attempt_dir / "parameters.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(config["parameters"], job["parameters"])
                    self.assertEqual(config["episodes"], -1)
                    self.assertNotIn("order_seed", config)
                    self.assertEqual(metadata["command"][1:3], [job["setting"], "val_unseen"])
                    self.assertNotIn("--order-seed", metadata["command"])
                    self.assertNotIn("--episode-limit", metadata["command"])
                    runner._validate_attempt_binding(
                        attempt_dir, metadata, expected_job=job,
                        expected_batch_id="unit-r2r-ce",
                    )
                    if job["ordinal"] == 0:
                        tampered = deepcopy(metadata)
                        tampered["ordinal"] = 9
                        with self.assertRaisesRegex(
                            runner.UserError, "differs from expanded job"
                        ):
                            runner._validate_attempt_binding(
                                attempt_dir, tampered, expected_job=job,
                                expected_batch_id="unit-r2r-ce",
                            )
                        runner._record_validation_error(
                            attempt_dir, runner.UserError("first attempt failed")
                        )
                        original_error = (
                            attempt_dir / "validation_error.json"
                        ).read_text(encoding="utf-8")
                        retry_dir, retry_metadata = runner.materialize_attempt(
                            spec, Path(__file__), "unit-r2r-ce", Path(directory),
                            job, 1,
                        )
                        self.assertNotEqual(attempt_dir, retry_dir)
                        self.assertEqual(
                            original_error,
                            (attempt_dir / "validation_error.json").read_text(
                                encoding="utf-8"
                            ),
                        )
                        runner._validate_attempt_binding(
                            retry_dir, retry_metadata, expected_job=job,
                            expected_batch_id="unit-r2r-ce",
                        )
                        with self.assertRaisesRegex(
                            runner.UserError, "retry budget"
                        ):
                            runner.materialize_attempt(
                                spec, Path(__file__), "unit-r2r-ce",
                                Path(directory), job, 2,
                            )
                    translated_method, _ = translate(
                        job["setting"], attempt_dir / "parameters.json",
                        attempt_dir / "diagnostics.json",
                    )
                    self.assertEqual(translated_method, job["method"])

                crash_root = Path(directory) / "crash-recovery"
                crash_job = jobs[1]
                original_atomic = runner._atomic_json

                def interrupt_between_files(path, value):
                    if Path(path).name == "job.json":
                        raise RuntimeError("injected materialization crash")
                    return original_atomic(path, value)

                with mock.patch.object(
                    runner, "_atomic_json", side_effect=interrupt_between_files
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "materialization crash"
                    ):
                        runner.materialize_attempt(
                            spec, Path(__file__), "unit-r2r-ce", crash_root,
                            crash_job, 0,
                        )
                self.assertFalse(
                    runner._attempt_dir(crash_root, crash_job, 0).exists()
                )
                recovered_dir, _ = runner.materialize_attempt(
                    spec, Path(__file__), "unit-r2r-ce", crash_root,
                    crash_job, 0,
                )
                self.assertTrue((recovered_dir / "job.json").is_file())

    def test_resource_gate_enforces_existing_three_worker_thresholds(self):
        spec = {
            "execution": {
                "resource_limits": {
                    "max_gpu_memory_mib_before_launch": 22000,
                    "estimated_job_gpu_memory_mib": 8000,
                    "max_aggregate_gpu_memory_mib": 30000,
                    "max_cgroup_memory_gib_before_launch": 55.0,
                    "estimated_job_memory_gib": 20.0,
                    "max_aggregate_cgroup_memory_gib": 75.0,
                }
            }
        }
        with mock.patch.object(runner.resources, "gpu_stats", return_value=(21000, 90)), \
                mock.patch.object(runner.resources, "cgroup_memory_gib", return_value=50.0):
            self.assertTrue(runner._resource_ok(spec, 0)[0])
        with mock.patch.object(runner.resources, "gpu_stats", return_value=(22500, 90)), \
                mock.patch.object(runner.resources, "cgroup_memory_gib", return_value=50.0):
            self.assertFalse(runner._resource_ok(spec, 0)[0])

        ledger = mock.Mock()
        ledger.snapshot.return_value = {
            "effective_gpu_memory_mib": 25000,
            "effective_cgroup_memory_gib": 50.0,
        }
        with mock.patch.object(runner.resources, "gpu_stats", return_value=(1000, 10)), \
                mock.patch.object(runner.resources, "cgroup_memory_gib", return_value=5.0):
            self.assertFalse(runner._reserved_resource_ok(spec, ledger, 0)[0])

    def test_process_bound_scheduler_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.lock"
            with runner.campaign_lifetime_lock(
                path, role="r2r_ce_val_unseen", batch_id="batch"
            ):
                with self.assertRaises(runner.JointLaunchError):
                    with runner.campaign_lifetime_lock(
                        path, role="r2r_ce_val_unseen", batch_id="batch"
                    ):
                        pass

    def test_tracked_gate_includes_both_val_unseen_source_manifests(self):
        _, finalization = builder.load_spec(FINALIZATION_SPEC)
        source_ledger_path = REPO_ROOT / finalization[
            "val_unseen_source_control"
        ]["path"]
        source_ledger = json.loads(source_ledger_path.read_text(encoding="utf-8"))
        placeholder = "vln/results/final/r2r-ce/initial_full_candidates.json"
        registry = {
            "selected_winners": {"path": placeholder},
            "source_ledger": {
                "path": finalization["val_seen_source_control"]["path"]
            },
            "records": {
                setting: {
                    method: {
                        "formal_manifest_path": source_ledger["records"][setting][
                            "formal_manifest_path"
                        ]
                    }
                    for method in ("source",) + builder.METHODS
                }
                for setting in builder.SETTINGS
            },
        }
        spec = {
            "registry_dependency": {"path": placeholder},
            "source_control": {"manifest": source_ledger_path.relative_to(REPO_ROOT).as_posix()},
            "protocol": {"order_manifest": {
                "path": "vln/manifests/episode_order/r2r_ce_v1_3_unified/val_unseen.json"
            }},
        }
        checked = []

        def tracked(command, **kwargs):
            del kwargs
            checked.append(command[-1])
            return mock.Mock(returncode=0, stdout="", stderr="")

        with mock.patch.object(runner.subprocess, "run", side_effect=tracked), \
                mock.patch.object(runner, "_untracked_execution_files", return_value=[]):
            runner._require_tracked_inputs(
                REPO_ROOT / "vln/experiments/r2r_ce_postsearch_finalization_v1.json",
                spec, registry,
            )
        expected = {
            item["formal_manifest_path"]
            for item in source_ledger["records"].values()
        }
        self.assertTrue(expected.issubset(set(checked)))

        missing = next(iter(expected))

        def one_missing(command, **kwargs):
            del kwargs
            return mock.Mock(
                returncode=int(command[-1] == missing), stdout="", stderr=""
            )

        with mock.patch.object(runner.subprocess, "run", side_effect=one_missing), \
                mock.patch.object(runner, "_untracked_execution_files", return_value=[]):
            with self.assertRaisesRegex(
                runner.UserError, "formal input must be committed"
            ):
                runner._require_tracked_inputs(
                    REPO_ROOT / "vln/experiments/r2r_ce_postsearch_finalization_v1.json",
                    spec, registry,
                )

    def test_validation_error_is_durable_and_marks_attempt_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt-00"
            attempt.mkdir()
            _write_json(attempt / "job.json", {"run_tag": "failed"})
            _write_json(attempt / "metrics.json", {"SR": 1.0})
            runner._record_validation_error(
                attempt, runner.UserError("authenticated artifact mismatch")
            )
            first = (attempt / "validation_error.json").read_text(encoding="utf-8")
            runner._record_validation_error(
                attempt, runner.UserError("replacement message")
            )
            self.assertEqual(
                first,
                (attempt / "validation_error.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(runner._state(attempt), "invalid")

            (attempt / "validation_error.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                runner.UserError, "validation-error record is malformed"
            ):
                runner._record_validation_error(
                    attempt, runner.UserError("must not trust an empty marker")
                )

    def test_attempt_history_rejects_gaps_and_excess_retries(self):
        job = {"model_index": 0, "model": "etpnav", "base_run_tag": "unit"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner._attempt_dir(root, job, 0).mkdir(parents=True)
            runner._attempt_dir(root, job, 2).mkdir(parents=True)
            with self.assertRaisesRegex(runner.UserError, "retry budget"):
                runner._existing_attempts(root, job)

    def test_scheduler_does_not_release_resumed_live_worker_on_peer_error(self):
        job = {"model": "etpnav", "base_run_tag": "external", "gpu": 0}
        spec = {
            "execution": {
                "max_attempts_per_job": 2,
                "max_workers": 1,
                "launch_stagger_seconds": 0,
                "poll_seconds": 0,
                "resource_limits": {"resource_wait_timeout_seconds": 1},
            }
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "_assert_batch_unchanged",
            side_effect=[None, runner.UserError("peer validation failed")],
        ), mock.patch.object(
            runner, "_latest_attempt", return_value=(0, Path(directory))
        ), mock.patch.object(
            runner, "_recover_worker_identity", return_value=True
        ), mock.patch.object(
            runner, "_state", return_value="running"
        ), mock.patch.object(
            runner, "_read_json", return_value={}
        ), mock.patch.object(
            runner, "_validate_attempt_binding"
        ), mock.patch.object(
            runner, "_claim_running_reservation"
        ), mock.patch.object(
            runner.time, "sleep"
        ), mock.patch.object(
            runner, "_release_reservation"
        ) as release:
            with self.assertRaisesRegex(runner.UserError, "peer validation failed"):
                runner.run_model_phase(
                    spec, Path(__file__), "batch", Path(directory), [job]
                )
        release.assert_not_called()

    def test_resumed_live_worker_reclaims_reservation_before_authorization(self):
        worker = {
            "pid": 101, "start_token": "worker", "cmdline_sha256": "a" * 64
        }
        scheduler = {
            "pid": 202, "start_token": "scheduler", "cmdline_sha256": "b" * 64
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

        spec = {"execution": {"resource_limits": {
            "estimated_job_gpu_memory_mib": 8000,
            "estimated_job_memory_gib": 20.0,
        }}}
        metadata = {"run_tag": "live-retry"}
        job = {"gpu": 0}
        with mock.patch.object(
            runner, "_live_worker_identities", return_value=[worker]
        ), mock.patch.object(
            runner, "process_identity", return_value=scheduler
        ), mock.patch.object(
            runner, "shared_gpu_launch_guard", return_value=Guard()
        ), mock.patch.object(
            runner.resources, "gpu_stats", return_value=(1000, 0)
        ), mock.patch.object(
            runner.resources, "cgroup_memory_gib", return_value=5.0
        ), mock.patch.object(runner, "_authorize_launch") as authorize:
            runner._claim_running_reservation(
                spec, "batch", job, Path("attempt"), metadata
            )
        token = runner._reservation_token("batch", "live-retry")
        self.assertIn(token, ledger.document["reservations"])
        self.assertIn(worker, ledger.document["reservations"][token]["owners"])
        authorize.assert_called_once_with(Path("attempt"))

    def test_dead_wrapper_does_not_hide_authenticated_live_descendant(self):
        wrapper = {
            "pid": 401, "start_token": "wrapper", "cmdline_sha256": "a" * 64
        }
        descendant = {
            "pid": 777, "start_token": "python", "cmdline_sha256": "b" * 64
        }
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt-00"
            attempt.mkdir()
            (attempt / "pid").write_text("401\n", encoding="utf-8")
            metadata = {
                "batch_id": "batch",
                "run_tag": "run",
                "spec_sha256": "c" * 64,
                "git_commit": "d" * 40,
            }
            metadata["process_group_token"] = (
                runner._expected_process_group_token(metadata, attempt)
            )
            (attempt / "job.json").write_text(
                json.dumps(metadata) + "\n", encoding="utf-8"
            )
            (attempt / "process_identity.json").write_text(
                json.dumps(wrapper) + "\n", encoding="utf-8"
            )
            with mock.patch.object(
                runner, "process_identity_alive", return_value=False
            ), mock.patch.object(
                runner, "_proc_group_member_identities",
                return_value=[descendant],
            ):
                self.assertTrue(runner._pid_alive(attempt))
                self.assertEqual(
                    runner._live_worker_identities(attempt), [descendant]
                )

    def test_proc_group_recovery_requires_group_session_and_attempt_token(self):
        token = "e" * 64
        child = {
            "pid": 777, "start_token": "python", "cmdline_sha256": "f" * 64
        }
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            valid = proc / "777"
            valid.mkdir()
            (valid / "stat").write_text(
                "777 (python worker) S 1 401 401 0 0 0\n", encoding="utf-8"
            )
            (valid / "environ").write_bytes(
                "{}={}".format(runner.PROCESS_GROUP_ENV, token).encode("ascii")
                + b"\0"
            )
            wrong_token = proc / "778"
            wrong_token.mkdir()
            (wrong_token / "stat").write_text(
                "778 (python worker) S 1 401 401 0 0 0\n", encoding="utf-8"
            )
            (wrong_token / "environ").write_bytes(b"OTHER=value\0")
            zombie = proc / "779"
            zombie.mkdir()
            (zombie / "stat").write_text(
                "779 (python worker) Z 1 401 401 0 0 0\n", encoding="utf-8"
            )
            (zombie / "environ").write_bytes(
                "{}={}".format(runner.PROCESS_GROUP_ENV, token).encode("ascii")
                + b"\0"
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

    def test_worker_exports_group_token_and_persists_session_leader_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            worker = runner._write_worker(attempt, {
                "command": ["true"], "process_group_token": "a" * 64,
            })
            source = worker.read_text(encoding="utf-8")
        self.assertIn(
            "export {}={}".format(runner.PROCESS_GROUP_ENV, "a" * 64), source
        )
        self.assertIn("printf '%s\\n' \"$$\"", source)
        self.assertLess(source.index("export "), source.index("for ((gate_wait="))

    def test_cleanup_terminates_descendants_after_wrapper_exit(self):
        descendant = {
            "pid": 777, "start_token": "python", "cmdline_sha256": "b" * 64
        }
        with mock.patch.object(
            runner, "_live_worker_identities",
            side_effect=[[descendant], [], []],
        ), mock.patch.object(
            runner, "_attempt_pgid", return_value=401
        ), mock.patch.object(runner.os, "killpg") as killpg:
            self.assertTrue(
                runner._terminate_attempt_process_group(Path("attempt"), 0)
            )
        killpg.assert_called_once_with(401, runner.signal.SIGTERM)

    def test_status_mode_is_read_only(self):
        spec = {"canonical_batch_id": runner.CANONICAL_BATCH_ID}
        with mock.patch.object(
            runner, "load_spec", return_value=spec
        ), mock.patch.object(
            runner, "expand_jobs", return_value=[]
        ), mock.patch.object(
            runner, "collect_states", return_value=({"pending": 10}, [])
        ) as collect, mock.patch("builtins.print"):
            self.assertEqual(runner.main(["--status"]), 0)
        self.assertFalse(collect.call_args.kwargs["revalidate"])

    def test_continuous_feedback_and_update_postconditions_fail_closed(self):
        feed = {
            "method": "feedtta",
            "episode_count": 1839,
            "parameters": {"scope_profile": "last_crossmodal"},
        }
        diagnostics = {
            "method": "feedtta",
            "episode_count": 1839,
            "action_selection": "target_native_argmax",
            "feedback_supervision": "binary_episode_success",
            "binary_feedback_endpoint": None,
            "feedtta_scope_profile": "last_crossmodal",
            "adapter": {
                "episodes": 1839,
                "updates": 1839,
                "relative_param_drift": 0.1,
                "feedback_episodes": 1839,
                "successful_feedback_episodes": 1000,
                "failed_feedback_episodes": 839,
                "feedback_type": "binary_episode_success",
                "action_selection_protocol": "target_native_argmax",
            },
        }
        runner._validate_diagnostics(
            feed, diagnostics, {"SR": 1000 * 100.0 / 1839, "SPL": 40.0}
        )
        diagnostics["adapter"]["feedback_episodes"] = 1838
        with self.assertRaisesRegex(runner.UserError, "feedback accounting"):
            runner._validate_diagnostics(
                feed, diagnostics, {"SR": 1000 * 100.0 / 1839, "SPL": 40.0}
            )

        fstta = {
            "method": "fstta", "episode_count": 1839, "parameters": {}
        }
        fstta_diagnostics = {
            "method": "fstta",
            "episode_count": 1839,
            "action_selection": "target_native_argmax",
            "feedback_supervision": "none",
            "binary_feedback_endpoint": None,
            "adapter": {
                "episodes": 1839,
                "updates": 0,
                "relative_param_drift": 0.0,
                "variance_history_lifetime": "test_stream",
            },
        }
        with self.assertRaisesRegex(runner.UserError, "no effective updates"):
            runner._validate_diagnostics(
                fstta, fstta_diagnostics, {"SR": 50.0, "SPL": 40.0}
            )
        fstta_diagnostics["adapter"]["updates"] = 1
        fstta_diagnostics["adapter"]["feedback_episodes"] = 1
        with self.assertRaisesRegex(runner.UserError, "consumed episode feedback"):
            runner._validate_diagnostics(
                fstta, fstta_diagnostics, {"SR": 50.0, "SPL": 40.0}
            )


if __name__ == "__main__":
    unittest.main()
