import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vln/scripts"))

from tta_config_cli import translate  # noqa: E402


class TTAConfigCLITest(unittest.TestCase):
    def _translate(self, setting, method, parameters):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({
                "method": method, "parameters": parameters,
            }), encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main()
