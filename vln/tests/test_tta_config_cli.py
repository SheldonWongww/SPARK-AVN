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


if __name__ == "__main__":
    unittest.main()
