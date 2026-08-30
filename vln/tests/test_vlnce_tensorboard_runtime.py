from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_MODULES = (
    "vln/baselines/etpnav/vlnce_baselines/ss_trainer_ETP.py",
    "vln/baselines/etpnav/vlnce_baselines/dagger_trainer.py",
    "vln/baselines/etpnav/vlnce_baselines/common/base_il_trainer.py",
    "vln/baselines/bevbert/bevbert_ce/vlnce_baselines/ss_trainer_ETP.py",
    "vln/baselines/bevbert/bevbert_ce/vlnce_baselines/ss_trainer_BEV.py",
    "vln/baselines/bevbert/bevbert_ce/vlnce_baselines/dagger_trainer.py",
    "vln/baselines/bevbert/bevbert_ce/vlnce_baselines/common/base_il_trainer.py",
)


class VlnceTensorboardRuntimeTest(unittest.TestCase):
    def test_evaluation_trainers_do_not_import_unused_tensorflow(self):
        for relative_path in TRAINER_MODULES:
            source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
            self.assertNotIn("import tensorflow", source, relative_path)


if __name__ == "__main__":
    unittest.main()
