import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/analyze_tta_late_collapse.py"
SPEC = importlib.util.spec_from_file_location("late_collapse", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def canonical_order(count):
    return {
        "order_sha256": "a" * 64,
        "episodes": [{
            "ordinal": index,
            "episode_id": "ep-{:03d}".format(index),
            "scene_id": "scene-{:02d}".format(index // 8),
        } for index in range(count)],
    }


def portable_document(order, run_tag="run", setting="etpnav-r2r-ce",
                      spl=None, success=None):
    count = len(order["episodes"])
    spl = spl if spl is not None else [0.5] * count
    success = success if success is not None else [1.0] * count
    return {
        "schema": MODULE.PER_EPISODE_SCHEMA,
        "setting": setting,
        "run_tag": run_tag,
        "split": "val_seen",
        "episode_count": count,
        "order_sha256": order["order_sha256"],
        "episodes": [{
            "ordinal": index,
            "episode_id": episode["episode_id"],
            "scene_id": episode["scene_id"],
            "metrics": {
                "spl": spl[index],
                "success": success[index],
                "ndtw": 0.6,
            },
        } for index, episode in enumerate(order["episodes"])],
    }


class DiscreteDerivationTest(unittest.TestCase):
    def test_suffix_and_matched_erosion_formula(self):
        result = MODULE.derive_discrete_metric(
            tta_prefix=62.0,
            tta_full=45.0,
            source_prefix=60.0,
            source_full=50.0,
            episode_count=1000,
            prefix_count=250,
        )
        self.assertAlmostEqual(result["tta_suffix_mean"], 39.333333333333336)
        self.assertAlmostEqual(result["source_suffix_mean"], 46.666666666666664)
        self.assertAlmostEqual(result["matched_prefix_delta"], 2.0)
        self.assertAlmostEqual(result["matched_suffix_delta"], -7.333333333333329)
        self.assertAlmostEqual(result["matched_delta_erosion"], -9.333333333333329)
        self.assertAlmostEqual(
            result["rounding_bounds"]["derived_suffix_mean"],
            (1250.0 / 750.0) * 0.005,
        )
        self.assertAlmostEqual(
            result["rounding_bounds"]["matched_delta_erosion"],
            0.01 + 2.0 * (1250.0 / 750.0) * 0.005,
        )

    def test_rounding_robust_coarse_flag(self):
        # First 20 episodes: TTA 65 vs Source 60.  Remaining 80 episodes:
        # TTA 50 vs Source 60.  The resulting full TTA mean is 53.
        result = MODULE.analyze_discrete_aggregates(
            {"SPL": 65.0, "SR": 65.0},
            {"SPL": 53.0, "SR": 53.0},
            {"SPL": 60.0, "SR": 60.0},
            {"SPL": 60.0, "SR": 60.0},
            setting="duet-r2r",
            episode_count=100,
            prefix_count=20,
        )
        self.assertTrue(result["coarse_2pp_flag_nominal"])
        self.assertTrue(result["coarse_2pp_flag"])
        self.assertEqual(result["classification"], "coarse_flag")
        self.assertAlmostEqual(
            result["metrics"]["SPL"]["matched_delta_erosion"], -15.0
        )

    def test_nonfinite_aggregate_is_rejected(self):
        with self.assertRaisesRegex(MODULE.CollapseError, "non-finite"):
            MODULE.derive_discrete_metric(
                float("nan"), 1.0, 1.0, 1.0, 100, 20
            )

    def test_finalist_parent_must_be_exact_last_screening_config(self):
        final = {
            "job": {
                "setting": "duet-r2r",
                "search_method": "fstta",
                "config_method": "fstta",
                "parameters": {"lr": 1e-5},
            },
        }
        parent = {
            "job": {
                "setting": "duet-r2r",
                "search_method": "fstta",
                "config_method": "fstta",
                "stage": "stage2",
                "parameters": {"lr": 1e-5},
            },
            "result": {"expected_episodes": 256},
        }
        with self.assertRaisesRegex(MODULE.CollapseError, "last screening"):
            MODULE._validate_parent(final, parent, 256)
        parent["job"]["stage"] = "stage3"
        MODULE._validate_parent(final, parent, 256)


class PerEpisodeValidationTest(unittest.TestCase):
    def test_canonical_order_must_be_val_seen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            order_dir = root / MODULE.SETTING_ORDER_DIRECTORY["etpnav-r2r-ce"]
            order_dir.mkdir(parents=True)
            (order_dir / "val_seen.json").write_text(json.dumps({
                "schema": "navtta.episode_order.v1",
                "split": "val_unseen",
                "episode_count": 1,
                "episodes": [{"episode_id": "ep-0"}],
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.CollapseError, "not val_seen"):
                MODULE.load_canonical_order(root, "etpnav-r2r-ce")

    def test_order_mismatch_is_rejected(self):
        order = canonical_order(4)
        document = portable_document(order)
        document["episodes"][1], document["episodes"][2] = (
            document["episodes"][2], document["episodes"][1]
        )
        # Preserve internally plausible ordinals so the episode-id order is
        # what fails, not merely the ordinal field.
        document["episodes"][1]["ordinal"] = 1
        document["episodes"][2]["ordinal"] = 2
        with self.assertRaisesRegex(MODULE.CollapseError, "order mismatch"):
            MODULE.validate_per_episode_document(
                document, order, setting="etpnav-r2r-ce", run_tag="run"
            )

    def test_missing_episode_is_rejected(self):
        order = canonical_order(4)
        document = portable_document(order)
        document["episodes"].pop()
        with self.assertRaisesRegex(MODULE.CollapseError, "missing or incomplete"):
            MODULE.validate_per_episode_document(
                document, order, setting="etpnav-r2r-ce", run_tag="run"
            )

    def test_nonfinite_episode_metric_is_rejected(self):
        order = canonical_order(4)
        document = portable_document(order)
        document["episodes"][2]["metrics"]["spl"] = float("inf")
        with self.assertRaisesRegex(MODULE.CollapseError, "non-finite"):
            MODULE.validate_per_episode_document(
                document, order, setting="etpnav-r2r-ce", run_tag="run"
            )

    def test_native_fallback_requires_canonical_key_order(self):
        order = canonical_order(4)
        stats = {
            episode["episode_id"]: {"success": 1.0, "spl": 0.5}
            for episode in reversed(order["episodes"])
        }
        with self.assertRaisesRegex(MODULE.CollapseError, "out of order"):
            MODULE.validate_raw_stats_episode(stats, order)

    def test_compact_loader_verifies_inventory_and_schema(self):
        order = canonical_order(4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_dir = (
                root / "methods/tent/stages/final/jobs/0000-etpnav"
            )
            job_dir.mkdir(parents=True)
            evidence = job_dir / "per_episode_metrics.json"
            evidence.write_text(
                json.dumps(portable_document(order), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            relative = evidence.relative_to(root).as_posix()
            report = {
                "schema": "navtta.vln_tta_hparam_compact_export.v1",
                "batch_id": "batch",
                "methods": {},
                "per_episode_metrics": [{
                    "exported_path": relative,
                    "exported_size": evidence.stat().st_size,
                    "exported_sha256": MODULE.sha256(evidence),
                }],
            }
            (root / "report.json").write_text(
                json.dumps(report) + "\n", encoding="utf-8"
            )
            store = MODULE.EvidenceStore(root, "batch")
            normalized = store.load_per_episode({
                "job": {
                    "setting": "etpnav-r2r-ce",
                    "run_tag": "run",
                    "result_root": "/remote/result/that/is/not/local",
                },
                "job_dir": job_dir,
                "result": {},
            }, order)
            self.assertEqual(normalized["episode_count"], 4)
            self.assertEqual(normalized["schema"], MODULE.PER_EPISODE_SCHEMA)

            evidence.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.CollapseError, "hash/size mismatch"):
                store.load_per_episode({
                    "job": {
                        "setting": "etpnav-r2r-ce",
                        "run_tag": "run",
                    },
                    "job_dir": job_dir,
                    "result": {},
                }, order)

    def test_raw_store_falls_back_to_native_stats_ep(self):
        order = canonical_order(4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_root = root / "result"
            stats_path = (
                result_root / "metrics/source_val_seen/"
                "stats_ep_ckpt_59_val_seen_r0_w1.json"
            )
            stats_path.parent.mkdir(parents=True)
            stats = {
                episode["episode_id"]: {"success": 1.0, "spl": 0.5}
                for episode in order["episodes"]
            }
            stats_path.write_text(json.dumps(stats) + "\n", encoding="utf-8")
            job_dir = root / "jobs/job"
            job_dir.mkdir(parents=True)
            store = MODULE.EvidenceStore(root, "batch")
            normalized = store.load_per_episode({
                "job": {
                    "setting": "etpnav-r2r-ce",
                    "run_tag": "run",
                    "result_root": str(result_root),
                },
                "job_dir": job_dir,
                "result": {},
            }, order)
            self.assertEqual(normalized["schema"], "native_stats_ep_fallback")
            self.assertEqual(normalized["episode_count"], 4)


class BootstrapAndClassificationTest(unittest.TestCase):
    def test_continuous_threshold_must_be_positive(self):
        order = canonical_order(4)
        source = MODULE.validate_per_episode_document(
            portable_document(order, run_tag="source"), order,
            setting="etpnav-r2r-ce", run_tag="source",
        )
        tta = MODULE.validate_per_episode_document(
            portable_document(order, run_tag="tta"), order,
            setting="etpnav-r2r-ce", run_tag="tta",
        )
        with self.assertRaisesRegex(MODULE.CollapseError, "must be positive"):
            MODULE.analyze_continuous_per_episode(
                tta, source, "tent", "etpnav-r2r-ce", "tta",
                bootstrap_samples=10, block_length=1, rolling_window=1,
                threshold_pp=0,
            )

    def test_bootstrap_is_deterministic(self):
        first = [float(index % 5) for index in range(96)]
        last = [value - 3.0 for value in first]
        first_result = MODULE.block_bootstrap_contrast(
            first, last, samples=250, block_length=16, seed=91
        )
        second_result = MODULE.block_bootstrap_contrast(
            first, last, samples=250, block_length=16, seed=91
        )
        self.assertEqual(first_result, second_result)
        self.assertAlmostEqual(first_result["estimate_percentage_points"], -3.0)
        self.assertEqual(first_result["bootstrap_seed"], 91)

    def test_holm_adjustment(self):
        adjusted = MODULE.holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.06)

    @staticmethod
    def _classification_record(tag, erosion, erosion_upper, erosion_p,
                               tail, tail_upper, tail_p):
        def test(estimate, upper, p_value):
            return {
                "estimate_percentage_points": estimate,
                "one_sided_upper95_percentage_points": upper,
                "one_sided_negative_p_value": p_value,
            }

        return {
            "run_tag": tag,
            "analysis": {
                "materiality_threshold_percentage_points": 2.0,
                "tests": {
                    "primary_erosion": test(erosion, erosion_upper, erosion_p),
                    "primary_tail": test(tail, tail_upper, tail_p),
                    "sr_erosion": test(erosion, erosion_upper, erosion_p),
                    "sr_tail": test(tail, tail_upper, tail_p),
                },
            },
        }

    def test_severe_warning_and_stable_flags_after_holm(self):
        records = [
            self._classification_record(
                "severe", -5.0, -1.0, 0.001, -3.0, -0.5, 0.001
            ),
            self._classification_record(
                "warning", -3.0, 1.0, 0.20, 0.0, 1.0, 0.50
            ),
            self._classification_record(
                "stable-1", -1.0, 1.0, 0.30, -1.0, 1.0, 0.30
            ),
            self._classification_record(
                "stable-2", 0.0, 1.0, 0.50, 0.0, 1.0, 0.50
            ),
            self._classification_record(
                "stable-3", 1.0, 2.0, 0.80, 1.0, 2.0, 0.80
            ),
        ]
        MODULE.apply_continuous_holm(records, expected_count=5)
        by_tag = {record["run_tag"]: record["analysis"] for record in records}
        self.assertTrue(by_tag["severe"]["severe_fixed_order_collapse"])
        self.assertEqual(
            by_tag["severe"]["classification"],
            "severe_fixed_order_collapse",
        )
        self.assertTrue(by_tag["warning"]["warning_fixed_order_degradation"])
        self.assertEqual(
            by_tag["warning"]["classification"],
            "warning_fixed_order_degradation",
        )
        self.assertEqual(
            by_tag["stable-1"]["classification"],
            "no_fixed_order_collapse_signal",
        )

    def test_paired_analysis_quartiles_and_rolling_64(self):
        count = 128
        order = canonical_order(count)
        source_document = portable_document(
            order, run_tag="source", spl=[0.7] * count,
            success=[1.0] * count,
        )
        tta_spl = [0.75] * 32 + [0.7] * 64 + [0.60] * 32
        tta_success = [1.0] * 96 + [0.0] * 32
        tta_document = portable_document(
            order, run_tag="tta", spl=tta_spl, success=tta_success,
        )
        source = MODULE.validate_per_episode_document(
            source_document, order, setting="etpnav-r2r-ce", run_tag="source"
        )
        tta = MODULE.validate_per_episode_document(
            tta_document, order, setting="etpnav-r2r-ce", run_tag="tta"
        )
        analysis = MODULE.analyze_continuous_per_episode(
            tta, source, "tent", "etpnav-r2r-ce", "tta",
            bootstrap_samples=100, block_length=16, rolling_window=64,
            bootstrap_seed=7,
        )
        self.assertAlmostEqual(
            analysis["first_quartile_primary_delta_percentage_points"], 5.0
        )
        self.assertAlmostEqual(
            analysis["last_quartile_primary_delta_percentage_points"], -10.0
        )
        self.assertAlmostEqual(
            analysis["tests"]["primary_erosion"][
                "estimate_percentage_points"
            ],
            -15.0,
        )
        self.assertEqual(len(analysis["rolling_deltas"]), count - 64 + 1)
        self.assertEqual(analysis["rolling_deltas"][0]["start_ordinal"], 0)
        self.assertEqual(analysis["rolling_deltas"][-1]["end_ordinal"], 127)


class OutputTest(unittest.TestCase):
    def test_outputs_are_json_and_csv_only_and_require_overwrite(self):
        report = {
            "schema": MODULE.OUTPUT_SCHEMA,
            "records": [],
            "rolling_rows": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            targets = MODULE.write_outputs(directory, report)
            self.assertEqual(
                {path.suffix for path in targets.values()}, {".json", ".csv"}
            )
            self.assertEqual(
                sorted(path.name for path in Path(directory).iterdir()),
                [
                    "late_collapse_analysis.json",
                    "late_collapse_rolling64.csv",
                    "late_collapse_summary.csv",
                ],
            )
            with self.assertRaisesRegex(MODULE.CollapseError, "output files exist"):
                MODULE.write_outputs(directory, report)
            MODULE.write_outputs(directory, report, overwrite=True)

        custom = dict(report)
        custom["policy"] = {"continuous_rolling_window": 32}
        with tempfile.TemporaryDirectory() as directory:
            targets = MODULE.write_outputs(directory, custom)
            self.assertEqual(targets["rolling"].name, "late_collapse_rolling32.csv")


class CampaignSyntheticIntegrationTest(unittest.TestCase):
    class Store:
        mode = "synthetic"
        batch_id = "synthetic-batch"

        def __init__(self, by_tag, by_stage, winners):
            self.by_tag = by_tag
            self.by_stage = by_stage
            self.winners = winners

        def load_method_jobs(self, method):
            self.test_method = method
            return self.by_tag, self.by_stage

        def winner_tags(self, method):
            return self.winners

        def load_per_episode(self, item, canonical):
            return item["per_episode"]

    @staticmethod
    def _metrics(setting, value):
        if setting.endswith("reverie"):
            return {"SR": value, "SPL": value, "RGS": value,
                    "RGSPL": value}
        return {"SR": value, "SPL": value, "OSR": value}

    def test_one_method_complete_campaign_path(self):
        settings = list(MODULE.SETTING_ORDER_DIRECTORY)
        count = 80
        prefix = 20
        final_count = 5
        spec = {
            "settings": settings,
            "setting_episode_counts": {setting: count for setting in settings},
            "screening_episodes": prefix,
            "protocol": {"full_val_seen_finalists_per_setting": final_count},
        }
        by_tag = {}
        by_stage = {"controls": [], "stage1": [], "final_controls": [],
                    "final": []}
        winners = {}

        with tempfile.TemporaryDirectory() as directory:
            order_root = Path(directory)
            normalized_orders = {}
            for setting in settings:
                order_dir = order_root / MODULE.SETTING_ORDER_DIRECTORY[setting]
                order_dir.mkdir(parents=True, exist_ok=True)
                episodes = [{
                    "episode_id": "{}-{:03d}".format(setting, index),
                    "scene_id": "scene-{:02d}".format(index // 8),
                } for index in range(count)]
                manifest = {
                    "schema": "navtta.episode_order.v1",
                    "split": "val_seen",
                    "episode_count": count,
                    "order_sha256": "b" * 64,
                    "episodes": episodes,
                }
                (order_dir / "val_seen.json").write_text(
                    json.dumps(manifest) + "\n", encoding="utf-8"
                )
                normalized_orders[setting] = [{
                    "ordinal": index,
                    "episode_id": episode["episode_id"],
                    "scene_id": episode["scene_id"],
                    "metrics": {"spl": 0.6, "success": 1.0},
                } for index, episode in enumerate(episodes)]

                source_parameters = {"action_selection": "argmax",
                                     "action_seed": 0}
                source_prefix_tag = "source-prefix-{}".format(setting)
                source_full_tag = "source-full-{}".format(setting)
                source_prefix = {
                    "job": {
                        "run_tag": source_prefix_tag, "setting": setting,
                        "search_method": "tent", "config_method": "source",
                        "stage": "controls", "parameters": source_parameters,
                    },
                    "result": {
                        "expected_episodes": prefix,
                        "metrics": self._metrics(setting, 60.0),
                    },
                }
                source_full = {
                    "job": {
                        "run_tag": source_full_tag, "setting": setting,
                        "search_method": "tent", "config_method": "source",
                        "stage": "final_controls", "parameters": source_parameters,
                    },
                    "result": {
                        "expected_episodes": count,
                        "metrics": self._metrics(setting, 60.0),
                    },
                }
                if setting in MODULE.CONTINUOUS_SETTINGS:
                    source_full["per_episode"] = {
                        "schema": MODULE.PER_EPISODE_SCHEMA,
                        "setting": setting, "run_tag": source_full_tag,
                        "episode_count": count,
                        "episodes": copy.deepcopy(normalized_orders[setting]),
                    }
                by_stage["controls"].append(source_prefix)
                by_stage["final_controls"].append(source_full)
                by_tag[source_prefix_tag] = source_prefix
                by_tag[source_full_tag] = source_full

                for candidate in range(final_count):
                    parameters = {"lr": 1e-6, "candidate": candidate}
                    parent_tag = "parent-{}-{}".format(setting, candidate)
                    final_tag = "final-{}-{}".format(setting, candidate)
                    parent = {
                        "job": {
                            "run_tag": parent_tag, "setting": setting,
                            "search_method": "tent", "config_method": "tent",
                            "stage": "stage1", "parameters": parameters,
                        },
                        "result": {
                            "expected_episodes": prefix,
                            "metrics": self._metrics(setting, 61.0),
                        },
                    }
                    final = {
                        "job": {
                            "run_tag": final_tag, "setting": setting,
                            "search_method": "tent", "config_method": "tent",
                            "stage": "final", "parameters": parameters,
                            "parent_run_tags": [parent_tag],
                        },
                        "result": {
                            "expected_episodes": count,
                            "metrics": self._metrics(setting, 61.0),
                        },
                    }
                    if setting in MODULE.CONTINUOUS_SETTINGS:
                        episodes = copy.deepcopy(normalized_orders[setting])
                        for episode in episodes:
                            episode["metrics"]["spl"] = 0.61
                        final["per_episode"] = {
                            "schema": MODULE.PER_EPISODE_SCHEMA,
                            "setting": setting, "run_tag": final_tag,
                            "episode_count": count, "episodes": episodes,
                        }
                    by_stage["stage1"].append(parent)
                    by_stage["final"].append(final)
                    by_tag[parent_tag] = parent
                    by_tag[final_tag] = final
                    if candidate == 0:
                        winners[setting] = final_tag

            store = self.Store(by_tag, by_stage, winners)
            report = MODULE.analyze_campaign(
                store, spec, order_root, methods=("tent",),
                bootstrap_samples=20, block_length=4, rolling_window=8,
                bootstrap_seed=3,
            )
            self.assertEqual(len(report["records"]), len(settings) * final_count)
            self.assertEqual(
                sum(record["is_scheduler_winner"] for record in report["records"]),
                len(settings),
            )
            self.assertEqual(
                len(report["rolling_rows"]),
                len(MODULE.CONTINUOUS_SETTINGS) * final_count * (count - 8 + 1),
            )
            self.assertTrue(all(
                record["analysis"]["classification"]
                != "pending_holm_adjustment"
                for record in report["records"]
                if record["family"] == "continuous"
            ))


if __name__ == "__main__":
    unittest.main()
