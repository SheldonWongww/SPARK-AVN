import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/promote_targeted_gap_source_controls.py"
SPEC = importlib.util.spec_from_file_location("source_promotion", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SourcePromotionTest(unittest.TestCase):
    def test_plan_identity_requires_exact_four_job_native_v12_matrix(self):
        jobs = []
        for setting in ("etpnav-r2r-ce", "bevbert-r2r-ce"):
            for split in ("val_unseen", "val_seen"):
                jobs.append({"setting": setting, "split": split})
        plan = {
            "schema": "navtta.vln_r2r_ce_v1_2_source_control_workflow.v1",
            "schema_version": 1,
            "protocol": {"data_version": "v1.2-native"},
            "jobs": jobs,
        }
        plan["plan_identity_sha256"] = MODULE.sha256_bytes(
            MODULE.canonical_bytes(plan)
        )
        MODULE.validate_plan_identity(plan)

        changed = copy.deepcopy(plan)
        changed["jobs"][0]["split"] = "val_seen"
        with self.assertRaisesRegex(MODULE.PromotionError, "identity mismatch"):
            MODULE.validate_plan_identity(changed)

    def test_native_aggregate_is_recomputed_in_canonical_episode_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate = root / "aggregate.json"
            episodes = root / "episodes.json"
            aggregate.write_text(
                json.dumps({"success": 0.5, "spl": 0.4}), encoding="utf-8"
            )
            episodes.write_text(json.dumps({
                "10": {"success": 1.0, "spl": 0.8},
                "11": {"success": 0.0, "spl": 0.0},
            }), encoding="utf-8")
            validation = {
                "metrics": {"SR": 50.0, "SPL": 40.0},
                "aggregate_artifact": {
                    "path": str(aggregate), "size": aggregate.stat().st_size,
                    "sha256": MODULE.sha256_file(aggregate),
                },
                "per_episode_artifact": {
                    "path": str(episodes), "size": episodes.stat().st_size,
                    "sha256": MODULE.sha256_file(episodes),
                },
            }
            order = {
                "episode_count": 2,
                "episodes": [{"episode_id": 10}, {"episode_id": 11}],
            }
            result = MODULE._validate_ce_aggregate_and_episodes(validation, order)
            self.assertEqual(result, (aggregate.resolve(), episodes.resolve()))

            order["episodes"].reverse()
            with self.assertRaisesRegex(MODULE.PromotionError, "IDs/order"):
                MODULE._validate_ce_aggregate_and_episodes(validation, order)

    def test_promoted_ledger_preserves_records_and_binds_candidate_review(self):
        candidate = {
            "schema": MODULE.CANDIDATE_SCHEMA,
            "schema_version": 1,
            "candidate_status": "review_required_not_yet_promoted",
            "split": "val_unseen",
            "records": {"etpnav-r2r-ce": {"run_id": "run"}},
        }
        candidate_path = (
            REPO_ROOT / "vln/manifests/source_candidates/candidate.json"
        )
        promoted = MODULE.build_promoted_ledger(
            candidate, candidate_path, "a" * 64, "reviewer",
            "2026-08-31T12:00:00+08:00", {"etpnav-r2r-ce": {}},
        )
        self.assertEqual(promoted["schema"], MODULE.PROMOTED_SCHEMA)
        self.assertNotIn("candidate_status", promoted)
        self.assertEqual(promoted["records"], candidate["records"])
        self.assertEqual(
            promoted["promotion_status"],
            "independently_reviewed_and_promoted",
        )
        review = promoted["promotion_review"]
        self.assertEqual(review["decision"], "approved")
        self.assertEqual(review["candidate_ledger_sha256"], "a" * 64)
        self.assertEqual(
            review["candidate_ledger_path"],
            "vln/manifests/source_candidates/candidate.json",
        )

    def test_successor_changes_only_reviewed_lifecycle_and_source_fields(self):
        v1 = json.loads(MODULE.BASE_SPEC_PATH.read_text(encoding="utf-8"))
        v1["status"] = "active"
        v1["superseded_by"] = None
        bindings = {
            key: {
                "mode": "reuse",
                "path": "vln/manifests/{}.json".format(key),
                "sha256": str(index) * 64,
            }
            for index, key in enumerate(
                (
                    "reverie_val_unseen", "reverie_val_seen",
                    "r2r_val_unseen", "r2r_val_seen",
                    "r2r_ce_v1_2_val_unseen", "r2r_ce_v1_2_val_seen",
                ),
                start=1,
            )
        }
        predecessor, successor = MODULE.build_successor_documents(
            v1, bindings, "reviewer", "2026-08-31T12:00:00+08:00",
            "f" * 64,
        )
        self.assertEqual(predecessor["status"], "superseded")
        self.assertEqual(
            predecessor["superseded_by"], "vln-targeted-gap-campaign-v2"
        )
        self.assertEqual(successor["status"], "active")
        self.assertEqual(successor["launch_readiness"]["status"], "ready")
        self.assertEqual(successor["source_control"]["bindings"], bindings)
        self.assertEqual(
            successor["supersedes"][0]["sha256"],
            MODULE.sha256_bytes(MODULE.spec_bytes(predecessor)),
        )
        runner = MODULE.campaign_runner()
        self.assertEqual(
            MODULE.canonical_bytes(runner._successor_scientific_projection(predecessor)),
            MODULE.canonical_bytes(runner._successor_scientific_projection(successor)),
        )
        self.assertEqual(v1["status"], "active")
        self.assertIsNone(v1["superseded_by"])

    def test_v1_supersession_preserves_hand_formatted_scientific_bytes(self):
        original = MODULE.BASE_SPEC_PATH.read_bytes().replace(
            b'  "status": "superseded",\n'
            b'  "supersedes": [],\n'
            b'  "superseded_by": "vln-targeted-gap-campaign-v2",',
            b'  "status": "active",\n'
            b'  "supersedes": [],\n'
            b'  "superseded_by": null,',
            1,
        )
        changed = MODULE.superseded_v1_bytes(
            original, "vln-targeted-gap-campaign-v2"
        )
        self.assertEqual(
            changed.count(b'"status": "superseded"'), 1
        )
        self.assertEqual(
            changed.count(b'"superseded_by": "vln-targeted-gap-campaign-v2"'),
            1,
        )
        restored = changed.replace(
            b'  "status": "superseded",\n'
            b'  "supersedes": [],\n'
            b'  "superseded_by": "vln-targeted-gap-campaign-v2",',
            b'  "status": "active",\n'
            b'  "supersedes": [],\n'
            b'  "superseded_by": null,',
            1,
        )
        self.assertEqual(restored, original)

    def test_review_timestamp_requires_timezone(self):
        with self.assertRaisesRegex(MODULE.PromotionError, "timezone"):
            MODULE.validate_review_identity("reviewer", "2026-08-31T12:00:00")
        MODULE.validate_review_identity(
            "reviewer", "2026-08-31T12:00:00+08:00"
        )

    def test_discrete_audit_fills_only_manifest_authenticated_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "manifest.json"
            metric = root / "metrics.json"
            metric.write_text(
                json.dumps({"SPL": 72.0, "SR": 75.0}), encoding="utf-8"
            )
            manifest = {
                "model": "hamt", "run_id": "run", "run_tag": "tag",
                "git_commit": "c" * 40,
                "immutable_identity_sha256": "d" * 64,
                "checkpoint": {"sha256": "1" * 64},
                "dataset": {
                    "stream_content_sha256": "2" * 64,
                    "stream_order_sha256": "3" * 64,
                },
            }
            formal.write_text(json.dumps(manifest), encoding="utf-8")
            record = {
                "parameters": {"action_selection": "argmax", "action_seed": 0},
                "metrics": {"SPL": 72.0, "SR": 75.0},
                "formal_manifest_path": "formal",
                "formal_manifest_sha256": MODULE.sha256_file(formal),
                "metrics_json_path": "metric",
                "metrics_json_sha256": MODULE.sha256_file(metric),
            }
            spec = {
                "cells": [{"setting": "hamt-r2r", "model": "hamt"}],
                "data_bindings": {"checkpoints": {"hamt-r2r": "1" * 64}},
            }

            class FakeCampaign:
                @staticmethod
                def order_binding(*_args, **_kwargs):
                    return {
                        "episode_count": 1021,
                        "benchmark": "r2r_discrete_duet_hamt",
                        "dataset_sha256": "2" * 64,
                        "order_sha256": "3" * 64,
                        "sha256": "4" * 64,
                    }

                @staticmethod
                def _source_metrics_from_artifact(*_args, **_kwargs):
                    return {"SPL": 72.0, "SR": 75.0}

                @staticmethod
                def _validate_formal_source_manifest(*_args, **_kwargs):
                    return formal, "d" * 64, MODULE.sha256_file(formal)

            def resolve(value, _label, require=True):
                del require
                return formal if value == "formal" else metric

            with mock.patch.object(MODULE, "repo_path", side_effect=resolve), \
                    mock.patch.object(MODULE, "require_tracked"):
                upgraded = MODULE.audit_discrete_record(
                    record, "hamt-r2r", "val_seen", "r2r", spec,
                    FakeCampaign(),
                )
            self.assertEqual(upgraded["git_commit"], "c" * 40)
            self.assertEqual(upgraded["immutable_identity_sha256"], "d" * 64)
            self.assertEqual(upgraded["dataset_sha256"], "2" * 64)
            self.assertEqual(upgraded["episode_order_sha256"], "3" * 64)
            self.assertEqual(upgraded["episode_count"], 1021)
            self.assertEqual(upgraded["evidence_status"], "ready")

    def test_successor_requires_explicit_supersession_confirmation(self):
        with self.assertRaisesRegex(
            MODULE.PromotionError, "confirm-v1-supersession"
        ):
            MODULE.create_successor(
                MODULE.BASE_SPEC_PATH, MODULE.SUCCESSOR_SPEC_PATH,
                "reviewer", "2026-08-31T12:00:00+08:00", False,
            )

    def test_successor_validates_all_source_controls_before_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "vln/experiments/vln_targeted_gap_campaign_v1.json"
            output = root / "vln/experiments/vln_targeted_gap_campaign_v2.json"
            runner_path = root / "vln/scripts/run_targeted_gap_campaign.py"
            base.parent.mkdir(parents=True)
            runner_path.parent.mkdir(parents=True)
            original = MODULE.BASE_SPEC_PATH.read_bytes().replace(
                b'  "status": "superseded",\n'
                b'  "supersedes": [],\n'
                b'  "superseded_by": "vln-targeted-gap-campaign-v2",',
                b'  "status": "active",\n'
                b'  "supersedes": [],\n'
                b'  "superseded_by": null,',
                1,
            )
            base.write_bytes(original)
            runner_path.write_text("# runner\n", encoding="utf-8")

            class FakeCampaign:
                @staticmethod
                def validate_source_controls(_spec):
                    raise MODULE.PromotionError("all-six-source gate failed")

                @staticmethod
                def _successor_scientific_projection(value):
                    return value

            with mock.patch.object(MODULE, "REPO_ROOT", root), \
                    mock.patch.object(MODULE, "BASE_SPEC_PATH", base), \
                    mock.patch.object(MODULE, "SUCCESSOR_SPEC_PATH", output), \
                    mock.patch.object(
                        MODULE, "CAMPAIGN_RUNNER_PATH", runner_path
                    ), mock.patch.object(MODULE, "require_tracked"), \
                    mock.patch.object(
                        MODULE, "source_bindings_for_successor",
                        return_value={},
                    ), mock.patch.object(
                        MODULE, "campaign_runner", return_value=FakeCampaign()
                    ):
                with self.assertRaisesRegex(
                    MODULE.PromotionError, "all-six-source gate failed"
                ):
                    MODULE.create_successor(
                        base, output, "reviewer",
                        "2026-08-31T12:00:00+08:00", True,
                    )
            self.assertEqual(base.read_bytes(), original)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
