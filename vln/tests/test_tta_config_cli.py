import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln/scripts"))

from tta_config_cli import translate  # noqa: E402


class TTAConfigCLITest(unittest.TestCase):
    def _translate(self, setting, method, parameters, **extra):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            document = {"method": method, "parameters": parameters}
            document.update(extra)
            path.write_text(json.dumps(document), encoding="utf-8")
            return translate(setting, str(path), str(Path(directory) / "diag.json"))

    def test_discrete_fstta_mapping(self):
        method, tokens = self._translate(
            "duet-r2r", "fstta",
            {"lr_fast": 6e-4, "lr_slow": 1e-3, "m": 3, "n": 4,
             "use_fast_lr_scaler": False,
             "reset_var_hist_each_episode": True},
        )
        self.assertEqual(method, "fstta")
        self.assertEqual(tokens[tokens.index("--tta_lr") + 1], "0.0006")
        self.assertEqual(tokens[tokens.index("--tta_fstta_m") + 1], "3")
        self.assertIn("--tta_fstta_no_fast_lr_scaler", tokens)
        self.assertIn("--tta_fstta_reset_var_hist_each_episode", tokens)

    def test_continuous_fstta_records_variance_history_lifetime(self):
        _, tokens = self._translate(
            "etpnav-r2r-ce", "fstta",
            {"reset_var_hist_each_episode": True},
        )
        self.assertEqual(
            tokens[
                tokens.index("TTA.FSTTA.RESET_VAR_HIST_EACH_EPISODE") + 1
            ],
            "True",
        )

    def test_formal_fstta_uses_paper_stream_history(self):
        common = {
            "schema": "navtta.vln_tta_job.v1",
            "stage": "final",
        }
        _, default_tokens = self._translate(
            "duet-r2r", "fstta", {}, **common
        )
        self.assertNotIn(
            "--tta_fstta_reset_var_hist_each_episode", default_tokens
        )
        with self.assertRaisesRegex(ValueError, "must not reset"):
            self._translate(
                "duet-r2r", "fstta",
                {"reset_var_hist_each_episode": True},
                **common
            )
        _, tokens = self._translate(
            "duet-r2r", "fstta",
            {"reset_var_hist_each_episode": False},
            **common
        )
        self.assertNotIn("--tta_fstta_reset_var_hist_each_episode", tokens)

    def test_continuous_feedtta_mapping(self):
        method, tokens = self._translate(
            "etpnav-r2r-ce", "feedtta",
            {"lr": 5e-6, "gamma": 0.95, "p": 0.05, "alpha": -0.2,
             "action_selection": "sample", "action_seed": 2,
             "sgr_mode": "paper_main"},
        )
        self.assertEqual(method, "feedtta")
        self.assertEqual(tokens[tokens.index("TTA.FEEDTTA.LR") + 1], "5e-06")
        self.assertEqual(tokens[tokens.index("TTA.FEEDTTA.GAMMA") + 1], "0.95")
        self.assertEqual(tokens[tokens.index("TTA.ACTION_SEED") + 1], "2")
        self.assertEqual(
            tokens[tokens.index("TTA.FEEDTTA.SGR_MODE") + 1],
            "paper_main",
        )

    def test_discrete_feedtta_scope_profile_mapping(self):
        method, tokens = self._translate(
            "duet-r2r", "feedtta",
            {
                "lr": 5e-6,
                "action_selection": "argmax",
                "scope_profile": "last_crossmodal",
            },
        )
        self.assertEqual(method, "feedtta")
        self.assertEqual(
            tokens[tokens.index("--tta_feedtta_scope_profile") + 1],
            "last_crossmodal",
        )
        self.assertEqual(
            tokens[tokens.index("--tta_action_selection") + 1], "argmax"
        )

    def test_continuous_feedtta_scope_profile_mapping(self):
        method, tokens = self._translate(
            "etpnav-r2r-ce", "feedtta",
            {"scope_profile": "last_crossmodal"},
        )
        self.assertEqual(method, "feedtta")
        self.assertEqual(
            tokens[tokens.index("TTA.FEEDTTA.SCOPE_PROFILE") + 1],
            "last_crossmodal",
        )

    def test_continuous_feedtta_rejects_unknown_scope_profile(self):
        with self.assertRaisesRegex(ValueError, "invalid continuous"):
            self._translate(
                "etpnav-r2r-ce", "feedtta",
                {"scope_profile": "all_the_things"},
            )

    def test_unknown_parameter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown tent parameters"):
            self._translate("hamt-r2r", "tent", {"typo": 1})

    def test_adapter_parity_schema_maps_only_explicit_audit_modes(self):
        common = {
            "schema": "navtta.vln_tta_adapter_parity_job.v1",
            "namespace": "adapter_parity_audit",
            "episodes": 256,
            "order_seed": 0,
        }
        method, discrete = self._translate(
            "duet-r2r", "tent", {"lr": 1e-6},
            audit_zero_update=True, audit_control=False, **common
        )
        self.assertEqual(method, "tent")
        self.assertIn("--tta_audit_zero_update", discrete)
        method, continuous = self._translate(
            "etpnav-r2r-ce", "source",
            {"action_selection": "argmax", "action_seed": 0},
            audit_zero_update=False, audit_control=True, **common
        )
        self.assertEqual(method, "source")
        self.assertEqual(
            continuous[continuous.index("TTA.AUDIT_CONTROL") + 1], "True"
        )
        self.assertEqual(
            continuous[
                continuous.index("TTA.AUDIT_EXPECTED_EPISODES") + 1
            ],
            "256",
        )

    def test_adapter_parity_order_seed_requires_exact_integer_zero(self):
        common = {
            "schema": "navtta.vln_tta_adapter_parity_job.v1",
            "namespace": "adapter_parity_audit",
            "episodes": 256,
            "audit_zero_update": True,
            "audit_control": False,
        }
        self._translate(
            "duet-r2r", "tent", {"lr": 1e-6}, order_seed=0, **common
        )
        for invalid_seed in (1, 2, True, False, 0.0, None, "0"):
            with self.subTest(order_seed=invalid_seed), self.assertRaisesRegex(
                    ValueError, "exact integer 0"):
                self._translate(
                    "duet-r2r", "tent", {"lr": 1e-6},
                    order_seed=invalid_seed, **common
                )

    def test_ordinary_search_schema_cannot_enable_audit(self):
        with self.assertRaisesRegex(ValueError, "adapter-parity job schema"):
            self._translate(
                "hamt-r2r", "tent", {},
                schema="navtta.vln_tta_job.v1",
                audit_zero_update=True,
            )

    def test_ordinary_schema_cannot_spoof_audit_namespace(self):
        with self.assertRaisesRegex(ValueError, "cannot claim"):
            self._translate(
                "hamt-r2r", "tent", {},
                schema="navtta.vln_tta_job.v1",
                namespace="adapter_parity_audit",
            )

    def test_ordinary_configs_omit_order_seed_except_exact_orders_jobs(self):
        for value in (None, 0, 1.0, False):
            with self.subTest(value=value), self.assertRaisesRegex(
                    ValueError, "must omit order_seed"):
                self._translate(
                    "hamt-r2r", "tent", {},
                    schema="navtta.vln_tta_job.v1",
                    stage="final",
                    order_seed=value,
                )
        self._translate(
            "hamt-r2r", "tent", {},
            schema="navtta.vln_tta_job.v1",
            stage="orders",
            order_seed=3,
        )
        with self.assertRaisesRegex(ValueError, "exact integer"):
            self._translate(
                "hamt-r2r", "tent", {},
                schema="navtta.vln_tta_job.v1",
                stage="orders",
            )
        for value in (True, 1.0, "1", None):
            with self.subTest(orders_value=value), self.assertRaisesRegex(
                    ValueError, "exact integer"):
                self._translate(
                    "hamt-r2r", "tent", {},
                    schema="navtta.vln_tta_job.v1",
                    stage="orders",
                    order_seed=value,
                )
        with self.assertRaisesRegex(ValueError, r"\[0, 1, 2, 3\]"):
            self._translate(
                "hamt-r2r", "tent", {},
                schema="navtta.vln_tta_job.v1",
                stage="orders",
                order_seed=4,
            )

    def test_rng_seed_parameters_require_exact_integers(self):
        for key in ("action_seed", "sgr_seed"):
            for value in (True, 1.0, "1", None):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(
                        ValueError, "must be an exact integer"):
                    self._translate(
                        "duet-r2r", "feedtta", {key: value}
                    )

    def test_feedtta_orders_rng_seeds_match_order_seed(self):
        common = {
            "schema": "navtta.vln_tta_job.v1",
            "stage": "orders",
            "order_seed": 2,
        }
        self._translate(
            "duet-r2r", "feedtta",
            {
                "action_seed": 2,
                "sgr_seed": 2,
                "sgr_mode": "paper_main",
                "scope_profile": "paper_full",
                "action_selection": "sample",
            },
            **common
        )
        for key in ("action_seed", "sgr_seed"):
            parameters = {
                "action_seed": 2,
                "sgr_seed": 2,
                "sgr_mode": "paper_main",
                "scope_profile": "paper_full",
                "action_selection": "sample",
            }
            parameters[key] = 1
            with self.subTest(key=key), self.assertRaisesRegex(
                    ValueError, "must equal order_seed"):
                self._translate(
                    "duet-r2r", "feedtta", parameters, **common
                )
        seed_three = dict(common, order_seed=3)
        self._translate(
            "duet-r2r", "feedtta",
            {
                "action_seed": 3,
                "sgr_seed": 3,
                "sgr_mode": "paper_main",
                "scope_profile": "paper_full",
                "action_selection": "sample",
            },
            **seed_three
        )

    def test_native_argmax_orders_require_only_the_sgr_seed(self):
        common = {
            "schema": "navtta.vln_tta_job.v1",
            "stage": "orders",
            "order_seed": 2,
        }
        _, tokens = self._translate(
            "duet-r2r", "feedtta",
            {
                "sgr_seed": 2,
                "sgr_mode": "paper_main",
                "scope_profile": "paper_full",
                "action_selection": "argmax",
            },
            **common
        )
        self.assertNotIn("--tta_action_seed", tokens)
        with self.assertRaisesRegex(ValueError, "must omit action_seed"):
            self._translate(
                "duet-r2r", "feedtta",
                {
                    "sgr_seed": 2,
                    "action_seed": 2,
                    "sgr_mode": "paper_main",
                    "scope_profile": "paper_full",
                    "action_selection": "argmax",
                },
                **common
            )
        with self.assertRaisesRegex(ValueError, "sgr_seed must equal"):
            self._translate(
                "duet-r2r", "feedtta",
                {
                    "sgr_seed": 1,
                    "sgr_mode": "paper_main",
                    "scope_profile": "paper_full",
                    "action_selection": "argmax",
                },
                **common
            )

    def test_formal_feedtta_native_argmax_and_sampling_ablation_both_translate(self):
        common = {
            "schema": "navtta.vln_tta_job.v1",
            "stage": "final",
        }
        _, native_port = self._translate(
            "duet-r2r", "feedtta",
            {
                "action_selection": "argmax",
                "scope_profile": "paper_full",
                "sgr_mode": "paper_main",
            },
            **common
        )
        self.assertEqual(
            native_port[native_port.index("--tta_action_selection") + 1],
            "argmax",
        )
        _, sampling_ablation = self._translate(
            "duet-r2r", "feedtta",
            {
                "action_selection": "sample",
                "scope_profile": "paper_full",
                "sgr_mode": "paper_main",
            },
            **common
        )
        self.assertEqual(
            sampling_ablation[
                sampling_ablation.index("--tta_action_selection") + 1
            ],
            "sample",
        )

    def test_explicit_native_argmax_feedtta_mapping(self):
        _, tokens = self._translate(
            "duet-r2r", "feedtta",
            {
                "action_selection": "argmax",
                "sgr_mode": "paper_main",
            },
        )
        self.assertEqual(
            tokens[tokens.index("--tta_action_selection") + 1], "argmax"
        )

    def test_matched_sampled_source_is_explicitly_mapped(self):
        _, discrete = self._translate(
            "duet-r2r", "source",
            {
                "action_selection": "sample",
                "matched_feedtta_source": True,
            },
        )
        self.assertIn("--tta_matched_feedtta_source", discrete)
        _, continuous = self._translate(
            "etpnav-r2r-ce", "source",
            {
                "action_selection": "sample",
                "matched_feedtta_source": True,
            },
        )
        self.assertEqual(
            continuous[
                continuous.index("TTA.MATCHED_FEEDTTA_SOURCE") + 1
            ],
            "True",
        )

    def test_legacy_sampled_source_is_normalized_to_matched_control(self):
        _, discrete = self._translate(
            "duet-r2r", "source", {"action_selection": "sample"}
        )
        self.assertIn("--tta_matched_feedtta_source", discrete)
        _, continuous = self._translate(
            "etpnav-r2r-ce", "source", {"action_selection": "sample"}
        )
        self.assertEqual(
            continuous[
                continuous.index("TTA.MATCHED_FEEDTTA_SOURCE") + 1
            ],
            "True",
        )

    def test_matched_source_marker_rejects_argmax(self):
        with self.assertRaisesRegex(ValueError, "requires action_selection=sample"):
            self._translate(
                "duet-r2r", "source",
                {
                    "action_selection": "argmax",
                    "matched_feedtta_source": True,
                },
            )

    def test_legacy_formal_tent_ablation_remains_translatable(self):
        common = {
            "schema": "navtta.vln_tta_job.v1",
            "stage": "final",
        }
        _, tokens = self._translate(
            "duet-r2r", "tent", {"update_interval": 4}, **common
        )
        self.assertEqual(
            tokens[tokens.index("--tta_update_interval") + 1], "4"
        )

    def test_atena_full_policy_claim_fails_before_launch(self):
        with self.assertRaisesRegex(ValueError, "unreplayed full policy"):
            self._translate(
                "duet-r2r", "atena", {"update_scope": "full_policy"}
            )

    def test_discrete_idea_mapping(self):
        method, tokens = self._translate(
            "duet-r2r", "idea",
            {"lr": 3e-3, "tau": 0.7, "opt_steps": 50, "k_max": 32,
             "use_fisher": False, "prompt_layers": 0,
             "source_stats_path": "/artifacts/duet-source.json",
             "source_stats_sha256": "a" * 64},
        )
        self.assertEqual(method, "idea")
        self.assertEqual(tokens[tokens.index("--tta_idea_lr") + 1], "0.003")
        self.assertEqual(tokens[tokens.index("--tta_idea_tau") + 1], "0.7")
        self.assertEqual(tokens[tokens.index("--tta_idea_opt_steps") + 1], "50")
        self.assertEqual(
            tokens[tokens.index("--tta_idea_source_stats_sha256") + 1],
            "a" * 64,
        )
        self.assertIn("--tta_idea_no_fisher", tokens)
        # IDEA must not leak FSTTA tau naming.
        self.assertNotIn("--tta_fstta_tau", tokens)

    def test_discrete_idea_use_fisher_true_emits_no_flag(self):
        _, tokens = self._translate(
            "duet-r2r", "idea", {
                "lr": 3e-3, "use_fisher": True,
                "source_stats_path": "/artifacts/duet-source.json",
                "source_stats_sha256": "a" * 64,
            },
        )
        self.assertNotIn("--tta_idea_no_fisher", tokens)

    def test_continuous_idea_mapping(self):
        method, tokens = self._translate(
            "etpnav-r2r-ce", "idea",
            {"lr": 3e-3, "tau": 0.5, "lambda": 0.4, "k_max": 32,
             "source_stats_path": "/artifacts/etp-source.json",
             "source_stats_sha256": "b" * 64},
        )
        self.assertEqual(method, "idea")
        self.assertEqual(tokens[tokens.index("TTA.IDEA.LR") + 1], "0.003")
        self.assertEqual(tokens[tokens.index("TTA.IDEA.TAU") + 1], "0.5")
        self.assertEqual(tokens[tokens.index("TTA.IDEA.LAMBDA") + 1], "0.4")
        self.assertEqual(
            tokens[tokens.index("TTA.IDEA.SOURCE_STATS_PATH") + 1],
            "/artifacts/etp-source.json",
        )

    def test_idea_without_offline_source_artifact_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "source_stats_path"):
            self._translate("duet-r2r", "idea", {"lr": 3e-3})

    def test_discrete_idea_source_collection_mapping(self):
        _, tokens = self._translate(
            "hamt-r2r", "idea", {
                "collect_source_stats": True,
                "source_stats_output": "/artifacts/hamt-source.json",
                "source_checkpoint_sha256": "c" * 64,
                "source_dataset": "R2R",
                "source_dataset_version": "v1",
                "source_split": "train",
                "source_trajectories": 128,
                "collection_policy": "frozen_source_argmax_rollout",
            },
            schema="navtta.vln_tta_job.v1",
            namespace="idea_source_statistics",
            stage="source_statistics",
        )
        self.assertIn("--tta_idea_collect_source_stats", tokens)
        self.assertEqual(
            tokens[tokens.index("--tta_idea_source_stats_output") + 1],
            "/artifacts/hamt-source.json",
        )

    def test_idea_source_collection_contract_is_fail_closed(self):
        parameters = {
            "action_selection": "argmax",
            "collect_source_stats": True,
            "source_stats_output": "/artifacts/hamt-source.json",
            "source_checkpoint_sha256": "c" * 64,
            "source_dataset": "vln/data/train.json",
            "source_dataset_version": "sha256:" + "d" * 64,
            "source_split": "train",
            "source_trajectories": 128,
            "opt_steps": 50,
            "collection_policy": "frozen_source_argmax_rollout",
        }
        metadata = {
            "schema": "navtta.vln_tta_job.v1",
            "namespace": "idea_source_statistics",
            "stage": "source_statistics",
        }
        self._translate("hamt-r2r", "idea", parameters, **metadata)
        for key, value, message in (
            ("source_trajectories", 128.0, "128 trajectories"),
            ("source_trajectories", True, "128 trajectories"),
            ("opt_steps", 50.0, "opt_steps=50"),
            ("opt_steps", True, "opt_steps=50"),
            ("source_split", "val_train_seen", "training split"),
            ("action_selection", "sample", "native argmax"),
        ):
            changed = dict(parameters, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaisesRegex(
                    ValueError, message):
                self._translate("hamt-r2r", "idea", changed, **metadata)

        with self.assertRaisesRegex(ValueError, "namespace=idea_source_statistics"):
            self._translate(
                "hamt-r2r", "idea", parameters,
                schema="navtta.vln_tta_job.v1", stage="source_statistics",
            )
        with self.assertRaisesRegex(ValueError, "requires collect_source_stats"):
            self._translate(
                "hamt-r2r", "idea", {
                    "source_stats_path": "/artifacts/hamt-source.json",
                    "source_stats_sha256": "d" * 64,
                },
                schema="navtta.vln_tta_job.v1",
                namespace="idea_source_statistics",
                stage="source_statistics",
            )

    def test_discrete_fstta_tau_unaffected_by_idea_entry(self):
        _, tokens = self._translate(
            "duet-r2r", "fstta", {"lr_fast": 6e-4, "tau": 0.5},
        )
        self.assertEqual(tokens[tokens.index("--tta_fstta_tau") + 1], "0.5")
        self.assertNotIn("--tta_idea_tau", tokens)


if __name__ == "__main__":
    unittest.main()
