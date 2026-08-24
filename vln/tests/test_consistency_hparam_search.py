"""Tests for the fail-closed staged VLN consistency search."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln" / "scripts"))

import run_consistency_hparam_search as runner  # noqa: E402


SPECS = (
    REPO_ROOT / "vln/experiments/r2r_consistency_search_v2.json",
    REPO_ROOT / "vln/experiments/reverie_consistency_search_v2.json",
    REPO_ROOT / "vln/experiments/r2r_ce_consistency_search_v2.json",
)


class SpecContractTest(unittest.TestCase):
    @staticmethod
    def _different_value(value):
        if isinstance(value, bool):
            return not value
        if isinstance(value, (int, float)):
            return value + 1
        if isinstance(value, str):
            return value + "_changed"
        raise AssertionError("unsupported invariant value: {!r}".format(value))

    def test_all_specs_use_correct_split_and_metric_protocol(self):
        expected = {
            "r2r": ("SPL", "SR"),
            "reverie": ("RGSPL", "RGS"),
            "r2r-ce": ("SPL", "SR"),
        }
        for path in SPECS:
            spec = runner.load_spec(path)
            protocol = spec["protocol"]
            self.assertEqual(protocol["development_split"], "val_unseen")
            self.assertEqual(protocol["retention_split"], "val_seen")
            self.assertEqual(protocol["order_seeds"], [1, 2, 3])
            self.assertFalse(protocol["positive_result_required"])
            self.assertEqual(
                protocol["source_policy"],
                "reuse_authenticated_order_invariant_argmax",
            )
            self.assertEqual(
                (protocol["primary_metric"], protocol["secondary_metric"]),
                expected[spec["benchmark"]],
            )

    def test_v1_specs_remain_historical_and_campaign_targets_v2(self):
        for path in SPECS:
            v1 = Path(str(path).replace("_v2.json", "_v1.json"))
            self.assertEqual(json.loads(v1.read_text())["schema"],
                             "navtta.vln_tta_consistency_search.v1")
        campaign = (REPO_ROOT / "vln/scripts/run_consistency_campaign.sh").read_text()
        self.assertNotIn("consistency_search_v1.json", campaign)
        self.assertEqual(campaign.count("consistency_search_v2.json"), 3)

    def test_candidate_budgets_are_explicit_and_constrained(self):
        for path in SPECS:
            spec = runner.load_spec(path)
            expected_fstta = 4 if spec["benchmark"] == "r2r-ce" else 3
            expected = {
                "tent": 4, "fstta": expected_fstta, "eam": 3,
                "feedtta": 3, "atena": 6, "idea": 4,
            }
            for method, method_spec in spec["methods"].items():
                records = runner.expand_candidate_records(method, method_spec)
                self.assertEqual(len(records), expected[method])
                self.assertEqual(
                    sum(item["role"] == "paper_anchor" for item in records), 1
                )
                self.assertNotIn("grid", method_spec)

    def test_every_fixed_base_key_and_value_is_fail_closed(self):
        for path in SPECS:
            document = json.loads(path.read_text(encoding="utf-8"))
            benchmark = document["benchmark"]
            for method in runner.METHOD_ORDER:
                expected = runner._expected_base(method, benchmark)
                self.assertEqual(document["methods"][method]["base"], expected)
                for key, value in expected.items():
                    with self.subTest(
                        benchmark=benchmark, method=method, changed=key
                    ):
                        method_spec = json.loads(json.dumps(
                            document["methods"][method]
                        ))
                        method_spec["base"][key] = self._different_value(value)
                        with self.assertRaisesRegex(
                            runner.UserError,
                            r"{}\.base.*changed={}".format(method, key),
                        ):
                            runner._validate_candidates(
                                method, method_spec, benchmark
                            )

                with self.subTest(
                    benchmark=benchmark, method=method, failure="missing"
                ):
                    method_spec = json.loads(json.dumps(
                        document["methods"][method]
                    ))
                    missing_key = next(iter(expected))
                    del method_spec["base"][missing_key]
                    with self.assertRaisesRegex(
                        runner.UserError, r"{}\.base.*missing=".format(method)
                    ):
                        runner._validate_candidates(method, method_spec, benchmark)

                with self.subTest(
                    benchmark=benchmark, method=method, failure="extra"
                ):
                    method_spec = json.loads(json.dumps(
                        document["methods"][method]
                    ))
                    method_spec["base"]["undeclared_fixed_knob"] = 0
                    with self.assertRaisesRegex(
                        runner.UserError, r"{}\.base.*extra=".format(method)
                    ):
                        runner._validate_candidates(method, method_spec, benchmark)

    def test_gradient_clipping_and_atena_benchmark_values_are_locked(self):
        expected_self_weight = {"r2r": 0.1, "reverie": 0.25, "r2r-ce": 0.1}
        for path in SPECS:
            spec = runner.load_spec(path)
            self.assertEqual(spec["methods"]["tent"]["base"]["max_grad_norm"], 0.0)
            self.assertEqual(spec["methods"]["fstta"]["base"]["max_grad_norm"], 0.0)
            self.assertEqual(
                spec["methods"]["atena"]["base"]["self_loss_weight"],
                expected_self_weight[spec["benchmark"]],
            )

        # JSON/Python normally considers True equal to 1; exact invariants must not.
        document = json.loads(SPECS[0].read_text(encoding="utf-8"))
        document["methods"]["tent"]["base"]["update_interval"] = True
        with self.assertRaisesRegex(runner.UserError, r"tent\.base.*update_interval"):
            runner._validate_candidates("tent", document["methods"]["tent"], "r2r")

    def test_candidate_parameters_must_be_exact_declared_search_keys(self):
        for path in SPECS:
            document = json.loads(path.read_text(encoding="utf-8"))
            benchmark = document["benchmark"]
            for method in runner.METHOD_ORDER:
                expected = runner.SEARCH_PARAMETER_KEYS[method]
                for candidate in document["methods"][method]["candidates"]:
                    self.assertEqual(set(candidate["parameters"]), expected)

                for failure in ("missing", "extra"):
                    with self.subTest(
                        benchmark=benchmark, method=method, failure=failure
                    ):
                        method_spec = json.loads(json.dumps(
                            document["methods"][method]
                        ))
                        parameters = method_spec["candidates"][0]["parameters"]
                        if failure == "missing":
                            del parameters[next(iter(expected))]
                        else:
                            parameters["undeclared_search_knob"] = 0
                        with self.assertRaisesRegex(
                            runner.UserError, "exactly the search keys"
                        ):
                            runner._validate_candidates(
                                method, method_spec, benchmark
                            )

    def test_atena_required_diagnostics_are_exact_and_benchmark_specific(self):
        for path in SPECS:
            document = json.loads(path.read_text(encoding="utf-8"))
            benchmark = document["benchmark"]
            expected = runner.ATENA_REQUIRED_DIAGNOSTICS[benchmark]
            self.assertEqual(
                document["methods"]["atena"]["required_diagnostics"], expected
            )
            for key, value in expected.items():
                with self.subTest(benchmark=benchmark, changed=key):
                    method_spec = json.loads(json.dumps(
                        document["methods"]["atena"]
                    ))
                    method_spec["required_diagnostics"][key] = (
                        self._different_value(value)
                    )
                    with self.assertRaisesRegex(
                        runner.UserError,
                        r"atena\.required_diagnostics.*changed={}".format(key),
                    ):
                        runner._validate_candidates("atena", method_spec, benchmark)

            method_spec = json.loads(json.dumps(document["methods"]["atena"]))
            method_spec["required_diagnostics"]["unexpected_diagnostic"] = True
            with self.assertRaisesRegex(
                runner.UserError, r"atena\.required_diagnostics.*extra="
            ):
                runner._validate_candidates("atena", method_spec, benchmark)

            method_spec = json.loads(json.dumps(document["methods"]["atena"]))
            del method_spec["required_diagnostics"][next(iter(expected))]
            with self.assertRaisesRegex(
                runner.UserError, r"atena\.required_diagnostics.*missing="
            ):
                runner._validate_candidates("atena", method_spec, benchmark)

    def test_fstta_uses_paired_scales_and_eam_only_varies_lr(self):
        spec = runner.load_spec(SPECS[0])
        fstta = runner.expand_candidate_records("fstta", spec["methods"]["fstta"])
        self.assertEqual(
            [(row["parameters"]["lr_fast"], row["parameters"]["lr_slow"])
             for row in fstta],
            [(6e-5, 1e-4), (1.8e-4, 3e-4), (6e-4, 1e-3)],
        )
        self.assertEqual(
            {key: fstta[0]["parameters"][key] for key in ("rho", "tau", "a", "b")},
            {"rho": 0.95, "tau": 0.7, "a": 0.9, "b": 1.1},
        )
        self.assertFalse(fstta[0]["parameters"]["reset_var_hist_each_episode"])
        eam = runner.expand_candidate_records("eam", spec["methods"]["eam"])
        fixed = {key: eam[0]["parameters"][key]
                 for key in ("confidence_scale", "memory_size", "batch_size", "update_interval")}
        self.assertEqual(fixed, {
            "confidence_scale": 0.4, "memory_size": 32,
            "batch_size": 8, "update_interval": 1,
        })
        self.assertTrue(all(
            {key: row["parameters"][key] for key in fixed} == fixed for row in eam
        ))

    def test_feedtta_only_sgr_seed_is_bound_under_native_argmax(self):
        spec = runner.load_spec(SPECS[0])
        params = runner.expand_candidates("feedtta", spec["methods"]["feedtta"])[0]
        seeded = runner._parameters_for_seed("feedtta", params, 3)
        self.assertNotIn("action_seed", seeded)
        self.assertEqual(seeded["sgr_seed"], 3)
        self.assertEqual(seeded["alpha"], 0.1)
        self.assertEqual(seeded["sgr_mode"], "paper_main")
        self.assertEqual(seeded["action_selection"], "argmax")
        reverie = runner.load_spec(SPECS[1])
        self.assertEqual(
            runner.expand_candidates("feedtta", reverie["methods"]["feedtta"])[0]["alpha"],
            -0.2,
        )

    def test_idea_is_o50_but_explicitly_blocked_without_source_stats(self):
        spec = runner.load_spec(SPECS[0])
        candidates = runner.expand_candidates("idea", spec["methods"]["idea"])
        self.assertEqual(len(candidates), 4)
        self.assertTrue(all(item["opt_steps"] == 50 for item in candidates))
        with self.assertRaisesRegex(runner.UserError, "Source-statistics"):
            runner._preflight_method(spec, "duet-r2r", "idea")

    def test_ready_idea_binding_is_validated_and_injected(self):
        spec = json.loads(SPECS[0].read_text(encoding="utf-8"))
        trajectory_ids = sorted(
            "trajectory-{}".format(index) for index in range(128)
        )
        ids_sha = hashlib.sha256(json.dumps(
            sorted(trajectory_ids), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        artifact = {
            "schema": runner.IDEA_SOURCE_STATS_SCHEMA,
            "version": runner.IDEA_SOURCE_STATS_VERSION,
            "feature_dim": 768,
            "num_layers": 4,
            "moment_estimator": runner.IDEA_SOURCE_STATS_ESTIMATOR,
            "provenance": {
                "checkpoint_sha256": "a" * 64,
                "dataset": "R2R",
                "dataset_version": "sha256:" + "d" * 64,
                "split": "train",
                "setting": "duet-r2r",
                "model": "duet",
                "trajectory_count": 128,
                "trajectory_ids": trajectory_ids,
                "trajectory_ids_sha256": ids_sha,
                "collection_scope": runner.IDEA_SOURCE_STATS_SCOPE,
                "collection_policy": runner.IDEA_SOURCE_COLLECTION_POLICY,
            },
            "layers": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source_stats.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            order = root / "train.json"
            order.write_text(json.dumps({
                "split": "train",
                "episode_count": 128,
                "order_sha256": "e" * 64,
                "dataset": {"path": "R2R", "sha256": "d" * 64},
                "episodes": [
                    {"episode_id": item, "scene_id": "scene"}
                    for item in trajectory_ids
                ],
            }), encoding="utf-8")
            config = root / "collection_config.json"
            config.write_text(json.dumps({
                "schema": runner.JOB_SCHEMA,
                "namespace": "idea_source_statistics",
                "stage": "source_statistics",
                "method": "idea",
                "parameters": {
                    "collect_source_stats": True,
                    "action_selection": "argmax",
                    "collection_policy": runner.IDEA_SOURCE_COLLECTION_POLICY,
                    "source_checkpoint_sha256": "a" * 64,
                    "source_stats_output": str(path.resolve()),
                },
            }), encoding="utf-8")
            diagnostics = root / "tta_diagnostics.json"
            diagnostics.write_text(json.dumps({
                "schema": "navtta.vln_discrete_tta.v1",
                "method": "idea",
                "episode_count": 128,
                "adapter": {
                    "complete": True,
                    "trajectory_count": 128,
                    "artifact_sha256": runner._sha256(path),
                    "collection_scope": runner.IDEA_SOURCE_STATS_SCOPE,
                    "provenance": {
                        "collection_policy": runner.IDEA_SOURCE_COLLECTION_POLICY,
                    },
                },
            }), encoding="utf-8")
            formal = root / "manifest.json"
            formal_document = {
                "task": "vln",
                "model": "duet",
                "method": "idea",
                "source_setting": "duet-r2r:train:native:idea",
                "status": "completed",
                "exit_code": 0,
                "checkpoint": {"sha256": "a" * 64},
                "dataset": {
                    "stream_order_sha256": "e" * 64,
                    "stream_content_sha256": "d" * 64,
                },
                "pinned_manifests": {
                    "episode_order": {"sha256": runner._sha256(order)},
                },
                "auxiliary_checkpoints": [{
                    "name": "tta_job_config",
                    "sha256": runner._sha256(config),
                }],
                "result_artifacts": [
                    {"name": path.name, "sha256": runner._sha256(path)},
                    {
                        "name": diagnostics.name,
                        "sha256": runner._sha256(diagnostics),
                    },
                ],
            }
            formal_document["immutable_identity_sha256"] = (
                runner.immutable_identity_sha256(formal_document)
            )
            formal.write_text(json.dumps(formal_document), encoding="utf-8")
            spec["methods"]["idea"]["availability"] = {"status": "ready"}
            spec["methods"]["idea"]["source_statistics"] = {
                "duet-r2r": {
                    "path": str(path),
                    "sha256": runner._sha256(path),
                    "trajectory_count": 128,
                    "collection_policy": runner.IDEA_SOURCE_COLLECTION_POLICY,
                    "checkpoint_sha256": "a" * 64,
                    "order_manifest": str(order),
                    "order_manifest_sha256": runner._sha256(order),
                    "collection_config": str(config),
                    "collection_config_sha256": runner._sha256(config),
                    "diagnostics": str(diagnostics),
                    "diagnostics_sha256": runner._sha256(diagnostics),
                    "formal_manifest": str(formal),
                    "formal_manifest_sha256": runner._sha256(formal),
                    "formal_immutable_identity_sha256": formal_document[
                        "immutable_identity_sha256"
                    ],
                }
            }
            runner._preflight_method(spec, "duet-r2r", "idea", "a" * 64)
            records = runner._candidate_records_for_setting(
                spec, "duet-r2r", "idea"
            )
            self.assertEqual(len(records), 4)
            for record in records:
                self.assertEqual(
                    record["parameters"]["source_stats_path"], str(path.resolve())
                )
                self.assertEqual(
                    record["parameters"]["source_stats_sha256"],
                    runner._sha256(path),
                )
                self.assertEqual(record["parameters"]["source_trajectories"], 128)
            with self.assertRaisesRegex(runner.UserError, "differs from Source"):
                runner._preflight_method(
                    spec, "duet-r2r", "idea", "b" * 64
                )

    def test_atena_forbids_zero_delta_and_lambda(self):
        for path in SPECS:
            spec = runner.load_spec(path)
            for params in runner.expand_candidates("atena", spec["methods"]["atena"]):
                self.assertGreater(params["query_threshold"], 0.0)
                self.assertGreater(params["mix_lambda"], 0.0)
            # Runtime diagnostics, not a brittle source-text sentinel, attest
            # exact replay after the first real episode.
            runner._preflight_method(spec, spec["settings"][0], "atena")

    def test_concurrency_contract(self):
        r2r = runner.load_spec(SPECS[0])
        self.assertEqual(r2r["concurrency"]["duet-r2r"]["tent"], 10)
        self.assertEqual(r2r["concurrency"]["hamt-r2r"]["fstta"], 11)
        self.assertEqual(r2r["concurrency"]["goat-r2r"]["fstta"], 12)
        reverie = runner.load_spec(SPECS[1])
        self.assertTrue(all(
            cap == 1 for methods in reverie["concurrency"].values()
            for cap in methods.values()
        ))
        ce = runner.load_spec(SPECS[2])
        self.assertTrue(ce["model_barrier"])
        for methods in ce["concurrency"].values():
            self.assertEqual(methods["idea"], 1)
            self.assertTrue(all(cap == 3 for name, cap in methods.items() if name != "idea"))

    def test_streamvln_request_fails_clearly(self):
        spec = runner.load_spec(SPECS[2])
        with self.assertRaisesRegex(runner.UserError, "StreamVLN TTA search is blocked"):
            runner._selected(spec["settings"], {"streamvln-r2r-ce"}, "settings")

    def test_all_tracked_source_ledgers_authenticate_per_setting(self):
        for path in SPECS:
            spec = runner.load_spec(path)
            metrics = (
                spec["protocol"]["primary_metric"],
                spec["protocol"]["secondary_metric"],
            )
            for setting in spec["settings"]:
                for split in ("val_unseen", "val_seen"):
                    evidence = runner._source_evidence(None, setting, split, metrics)
                    self.assertEqual(len(evidence["checkpoint_sha256"]), 64)
                    self.assertEqual(len(evidence["source_ledger_sha256"]), 64)
                    self.assertEqual(
                        evidence["source_action_protocol"],
                        "target_native_argmax",
                    )
                    self.assertFalse(evidence["matched_feedtta_source"])
                    self.assertEqual(len(evidence["dataset_sha256"]), 64)
                    self.assertTrue(evidence["benchmark"])

    def test_spec_rejects_non_argmax_source_policy(self):
        document = json.loads(SPECS[0].read_text(encoding="utf-8"))
        document["protocol"]["source_policy"] = "matched_sample"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-source-policy.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                runner.UserError, "protocol.source_policy"
            ):
                runner.load_spec(path)

    def test_source_evidence_rejects_sampling_ledger_claims(self):
        ledger_path = REPO_ROOT / runner.SOURCE_LEDGER[("r2r", "val_unseen")]
        original = json.loads(ledger_path.read_text(encoding="utf-8"))
        bad_protocol = json.loads(json.dumps(original))
        bad_protocol["source_protocol"] = "policy_sampling"
        with mock.patch.object(runner, "_read_json", return_value=bad_protocol):
            with self.assertRaisesRegex(runner.UserError, "action protocol"):
                runner._source_evidence(
                    None, "duet-r2r", "val_unseen", ("SPL", "SR")
                )

        bad_record = json.loads(json.dumps(original))
        bad_record["records"]["duet-r2r"]["parameters"][
            "action_selection"
        ] = "sample"
        with mock.patch.object(runner, "_read_json", return_value=bad_record):
            with self.assertRaisesRegex(runner.UserError, "unmatched argmax"):
                runner._source_evidence(
                    None, "duet-r2r", "val_unseen", ("SPL", "SR")
                )


class CommandPlanTest(unittest.TestCase):
    def test_search_dry_run_is_val_unseen_three_global_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            commands = runner.run_search(
                str(SPECS[0]), "unit", {"tent"}, {"duet-r2r"}, directory,
                None, True, False, 2,
            )
            self.assertEqual(len(commands), 12)
            self.assertTrue(all(" duet-r2r val_unseen 2 " in cmd for cmd in commands))
            self.assertEqual(sum("--order-seed 1" in cmd for cmd in commands), 4)
            self.assertEqual(sum("--order-seed 2" in cmd for cmd in commands), 4)
            self.assertEqual(sum("--order-seed 3" in cmd for cmd in commands), 4)
            self.assertTrue(all("val_seen" not in cmd for cmd in commands))
            configs = sorted(Path(directory).rglob("tta_config.json"))
            self.assertEqual(len(configs), 12)
            for path in configs:
                config = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(config["stage"], "orders")
                self.assertEqual(config["episodes"], -1)
                self.assertIn(config["order_seed"], (1, 2, 3))

    def test_override_cannot_exceed_registered_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(runner.UserError, r"\[1, 10\]"):
                runner.run_search(
                    str(SPECS[0]), "unit", {"tent"}, {"duet-r2r"}, directory,
                    11, True, False, 0,
                )

    def test_nonzero_worker_exit_aborts(self):
        job = {"command": ["false"], "result_root": "/unused"}
        with mock.patch.object(
            runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 9)
        ):
            with self.assertRaisesRegex(runner.UserError, "job failed with exit 9"):
                runner._run_one(job, False)

    def test_every_ready_setting_method_builds_translatable_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            for spec_path in SPECS:
                spec = runner.load_spec(spec_path)
                for setting in spec["settings"]:
                    for method in runner.METHOD_ORDER:
                        if method == "idea":
                            continue
                        candidate = runner.expand_candidate_records(
                            method, spec["methods"][method]
                        )[0]
                        jobs = runner._build_jobs(
                            spec, directory, "matrix", setting, method,
                            "search", [candidate], 0,
                        )
                        self.assertEqual(len(jobs), 3)
                        for job in jobs:
                            self.assertEqual(job["command"][2:5], [
                                setting, "val_unseen", "0",
                            ])
                            config = json.loads(Path(
                                job["config_path"]
                            ).read_text(encoding="utf-8"))
                            if method == "feedtta":
                                self.assertEqual(
                                    config["parameters"]["action_selection"],
                                    "argmax",
                                )
                                self.assertNotIn(
                                    "action_seed", config["parameters"]
                                )

    def test_launcher_preflight_is_dry_run_only_and_probes_one_job(self):
        jobs = [
            {"command": ["runner", "a"], "result_root": "/unused-a"},
            {"command": ["runner", "b"], "result_root": "/unused-b"},
        ]
        with self.assertRaisesRegex(runner.UserError, "requires --dry-run"):
            runner.run_search(
                str(SPECS[0]), "unit", {"tent"}, {"duet-r2r"},
                tempfile.gettempdir(), None, False,
                launcher_preflight=True,
            )
        with mock.patch.object(
            runner.subprocess, "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            runner._run_jobs(jobs, 1, True, launcher_preflight=True)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["runner", "a", "--dry-run"])

    def test_loaded_plan_rejects_split_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            runner.run_search(
                str(SPECS[0]), "tamper", {"tent"}, {"duet-r2r"},
                directory, None, True,
            )
            spec = runner.load_spec(SPECS[0])
            path = runner._plan_path(
                directory, spec, "tamper", "duet-r2r", "tent", "search"
            )
            plan = json.loads(path.read_text(encoding="utf-8"))
            plan["jobs"][0]["split"] = "val_seen"
            path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(runner.UserError, "job split mismatch"):
                runner._load_plan(
                    path, SPECS[0], spec, "tamper", "duet-r2r", "tent",
                    "search",
                )


class MetricParsingTest(unittest.TestCase):
    def test_discrete_reverie_metrics(self):
        values = runner.parse_console_metrics(
            "Env name: val_unseen, sr: 62.1, spl: 55.2, rgs: 49.3, rgspl: 43.4\n",
            "val_unseen",
        )
        self.assertEqual(values["SPL"], 55.2)
        self.assertEqual(values["RGSPL"], 43.4)

    def test_continuous_metrics_are_scaled(self):
        values = runner.parse_console_metrics(
            "Average episode success: 0.5824\nAverage episode spl: 0.4826\n",
            "val_unseen",
        )
        self.assertAlmostEqual(values["SR"], 58.24)
        self.assertAlmostEqual(values["SPL"], 48.26)


class SelectionTest(unittest.TestCase):
    @staticmethod
    def _rows(candidate_id, primary, secondary, drift=0.1, updates=10):
        return [
            {
                "split": "val_unseen",
                "candidate_id": candidate_id,
                "candidate_role": "paper_anchor" if candidate_id == "anchor" else "bounded_variant",
                "candidate_parameters": {"name": candidate_id},
                "order_seed": seed,
                "metrics": {"SPL": p, "SR": s},
                "diagnostics": {"relative_param_drift": drift, "updates": updates},
                "run_tag": "{}-{}".format(candidate_id, seed),
                "formal_manifest": "/manifest/{}-{}".format(candidate_id, seed),
                "formal_manifest_sha256": "a" * 64,
                "diagnostics_sha256": "b" * 64,
                "metric_sha256": "c" * 64,
            }
            for seed, p, s in zip((1, 2, 3), primary, secondary)
        ]

    def test_lcb_prefers_stable_candidate_over_higher_noisy_mean(self):
        rows = self._rows("noisy", [70, 100, 70], [70, 100, 70])
        rows += self._rows("stable", [73, 73, 73], [72, 72, 72])
        winner, _ = runner.select_config(
            rows, {"SPL": 70, "SR": 70},
            {"primary_metric": "SPL", "secondary_metric": "SR", "order_seeds": [1, 2, 3]},
        )
        self.assertEqual(winner["candidate_id"], "stable")

    def test_negative_lcb_still_selects_best_candidate(self):
        rows = self._rows("worse", [65, 65, 65], [66, 66, 66])
        rows += self._rows("least_bad", [69, 69, 69], [68, 68, 68])
        winner, ranked = runner.select_config(
            rows, {"SPL": 70, "SR": 70},
            {"primary_metric": "SPL", "secondary_metric": "SR", "order_seeds": [1, 2, 3]},
        )
        self.assertEqual(winner["candidate_id"], "least_bad")
        self.assertLess(winner["primary"]["lcb"], 0.0)
        self.assertEqual(len(ranked), 2)

    def test_secondary_then_drift_then_updates_break_ties(self):
        rows = self._rows("low_secondary", [72, 72, 72], [70, 70, 70], drift=0.01, updates=1)
        rows += self._rows("high_secondary", [72, 72, 72], [71, 71, 71], drift=9, updates=999)
        winner, _ = runner.select_config(
            rows, {"SPL": 70, "SR": 70},
            {"primary_metric": "SPL", "secondary_metric": "SR", "order_seeds": [1, 2, 3]},
        )
        self.assertEqual(winner["candidate_id"], "high_secondary")

        rows = self._rows("high_drift", [72] * 3, [71] * 3, drift=0.2, updates=1)
        rows += self._rows("low_drift", [72] * 3, [71] * 3, drift=0.1, updates=100)
        winner, _ = runner.select_config(
            rows, {"SPL": 70, "SR": 70},
            {"primary_metric": "SPL", "secondary_metric": "SR", "order_seeds": [1, 2, 3]},
        )
        self.assertEqual(winner["candidate_id"], "low_drift")

    def test_missing_seed_and_val_seen_input_fail_closed(self):
        rows = self._rows("incomplete", [72, 72, 72], [71, 71, 71])[:-1]
        with self.assertRaisesRegex(runner.UserError, "lacks complete seeds"):
            runner.select_config(
                rows, {"SPL": 70, "SR": 70},
                {"primary_metric": "SPL", "secondary_metric": "SR", "order_seeds": [1, 2, 3]},
            )
        rows = self._rows("wrong_split", [72] * 3, [71] * 3)
        rows[0]["split"] = "val_seen"
        with self.assertRaisesRegex(runner.UserError, "val_unseen development"):
            runner.select_config(
                rows, {"SPL": 70, "SR": 70},
                {"primary_metric": "SPL", "secondary_metric": "SR", "order_seeds": [1, 2, 3]},
            )


class DiagnosticsContractTest(unittest.TestCase):
    def test_missing_diagnostics_aborts(self):
        job = {
            "method": "tent", "expected_episode_count": 3,
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(runner.UserError, "cannot read JSON"):
                runner._validate_diagnostics(job, Path(directory) / "missing.json")

    def test_atena_requires_full_episode_replay_marker(self):
        job = {"method": "atena", "expected_episode_count": 3}
        diagnostics = {
            "method": "atena", "episode_count": 3,
            "supervision": "binary_navigation_success_feedback",
            "binary_feedback_endpoint": "episode_success",
            "adapter": {"episodes": 3, "updates": 1, "relative_param_drift": 0.1},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tta_diagnostics.json"
            path.write_text(json.dumps(diagnostics), encoding="utf-8")
            with self.assertRaisesRegex(runner.UserError, "full-replay diagnostic"):
                runner._validate_diagnostics(job, path)

    def test_feedtta_requires_native_not_paper_sampling_protocol(self):
        job = {
            "method": "feedtta", "setting": "duet-r2r",
            "expected_episode_count": 3,
        }
        diagnostics = {
            "method": "feedtta",
            "episode_count": 3,
            "supervision": "binary_navigation_success_feedback",
            "binary_feedback_endpoint": "episode_success",
            "action_selection": "target_native_argmax",
            "feedtta_sgr_mode": "paper_main",
            "feedtta_scope_profile": "paper_full",
            "feedtta_protocol": "task_adapted_target_native_argmax",
            "feedtta_native_action_protocol": True,
            "feedtta_paper_sampling_protocol": False,
            "feedtta_canonical_protocol": False,
            "adapter": {
                "episodes": 3,
                "updates": 3,
                "relative_param_drift": 0.1,
                "sgr_mode": "paper_main",
                "action_selection_protocol": "target_native_argmax",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tta_diagnostics.json"
            path.write_text(json.dumps(diagnostics), encoding="utf-8")
            runner._validate_diagnostics(job, path)
            diagnostics["feedtta_canonical_protocol"] = True
            path.write_text(json.dumps(diagnostics), encoding="utf-8")
            with self.assertRaisesRegex(
                runner.UserError, "native-action protocol mismatch"
            ):
                runner._validate_diagnostics(job, path)

    def test_idea_diagnostics_are_bound_to_the_planned_source_artifact(self):
        job = {
            "method": "idea",
            "model": "duet",
            "setting": "duet-r2r",
            "expected_episode_count": 3,
            "parameters": {
                "source_stats_sha256": "a" * 64,
                "source_trajectories": 128,
            },
        }
        diagnostics = {
            "method": "idea",
            "episode_count": 3,
            "supervision": "unsupervised",
            "binary_feedback_endpoint": None,
            "adapter": {
                "episodes": 3,
                "updates": 3,
                "relative_param_drift": 0.1,
                "source_statistics_mode": "offline_artifact",
                "source_statistics_schema": runner.IDEA_SOURCE_STATS_SCHEMA,
                "source_statistics_sha256": "a" * 64,
                "source_statistics_trajectory_count": 128,
                "source_statistics_moment_estimator": (
                    runner.IDEA_SOURCE_STATS_ESTIMATOR
                ),
                "source_statistics_provenance": {
                    "checkpoint_sha256": "b" * 64,
                    "collection_scope": runner.IDEA_SOURCE_STATS_SCOPE,
                    "collection_policy": runner.IDEA_SOURCE_COLLECTION_POLICY,
                    "setting": "duet-r2r",
                    "model": "duet",
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tta_diagnostics.json"
            path.write_text(json.dumps(diagnostics), encoding="utf-8")
            result = runner._validate_diagnostics(job, path)
            self.assertEqual(result["source_checkpoint_sha256"], "b" * 64)
            diagnostics["adapter"]["source_statistics_sha256"] = "c" * 64
            path.write_text(json.dumps(diagnostics), encoding="utf-8")
            with self.assertRaisesRegex(runner.UserError, "identity mismatch"):
                runner._validate_diagnostics(job, path)


if __name__ == "__main__":
    unittest.main()
