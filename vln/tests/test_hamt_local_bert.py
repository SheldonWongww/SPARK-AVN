from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_CHECK = REPO_ROOT / "vln" / "scripts" / "verify_runtime_imports.sh"
SOURCE_RUNNER = REPO_ROOT / "vln" / "scripts" / "run_source_eval.sh"
HAMT_R2R_MODEL_INIT = (
    REPO_ROOT / "vln" / "baselines" / "hamt" / "finetune_src"
    / "models" / "vlnbert_init.py"
)
HAMT_REVERIE_MODEL = (
    REPO_ROOT / "vln" / "baselines" / "hamt" / "finetune_src"
    / "reverie" / "model_navref.py"
)


class HamtLocalBertTest(unittest.TestCase):
    def test_launchers_pin_existing_local_snapshot(self):
        expected_revision = "86b5e0934494bd15c9632b12f734a8a67f723594"
        for script in (RUNTIME_CHECK, SOURCE_RUNNER):
            source = script.read_text(encoding="utf-8")
            self.assertIn(expected_revision, source)
            self.assertIn(
                'export NAVTTA_BERT_BASE_UNCASED="${BERT_BASE_UNCASED_ROOT}"',
                source,
            )

    def test_hamt_uses_local_snapshot_override(self):
        for module in (HAMT_R2R_MODEL_INIT, HAMT_REVERIE_MODEL):
            source = module.read_text(encoding="utf-8")
            self.assertIn("NAVTTA_BERT_BASE_UNCASED", source)
            self.assertIn("'bert-base-uncased'", source)

    def test_runtime_check_uses_local_path_for_hamt(self):
        source = RUNTIME_CHECK.read_text(encoding="utf-8")
        hamt_branch = source[source.index('elif setting == "hamt":'):]
        self.assertIn('root = os.environ["NAVTTA_BERT_BASE_UNCASED"]', hamt_branch)
        self.assertIn(
            "AutoTokenizer.from_pretrained(root, local_files_only=True)",
            hamt_branch,
        )
        self.assertIn(
            "PretrainedConfig.from_pretrained(root, local_files_only=True)",
            hamt_branch,
        )


if __name__ == "__main__":
    unittest.main()
