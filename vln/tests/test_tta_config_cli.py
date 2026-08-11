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
             "use_fast_lr_scaler": False},
        )
        self.assertEqual(method, "fstta")
        self.assertEqual(tokens[tokens.index("--tta_lr") + 1], "0.0006")
        self.assertEqual(tokens[tokens.index("--tta_fstta_m") + 1], "3")
        self.assertIn("--tta_fstta_no_fast_lr_scaler", tokens)

    def test_continuous_feedtta_mapping(self):
        method, tokens = self._translate(
            "etpnav-r2r-ce", "feedtta",
            {"lr": 5e-6, "gamma": 0.95, "p": 0.05, "alpha": -0.2,
             "action_seed": 2},
        )
        self.assertEqual(method, "feedtta")
        self.assertEqual(tokens[tokens.index("TTA.FEEDTTA.LR") + 1], "5e-06")
        self.assertEqual(tokens[tokens.index("TTA.FEEDTTA.GAMMA") + 1], "0.95")
        self.assertEqual(tokens[tokens.index("TTA.ACTION_SEED") + 1], "2")

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
            order_seed=1,
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
            {"action_seed": 2, "sgr_seed": 2}, **common
        )
        for key in ("action_seed", "sgr_seed"):
            parameters = {"action_seed": 2, "sgr_seed": 2}
            parameters[key] = 1
            with self.subTest(key=key), self.assertRaisesRegex(
                    ValueError, "must equal order_seed"):
                self._translate(
                    "duet-r2r", "feedtta", parameters, **common
                )


if __name__ == "__main__":
    unittest.main()
