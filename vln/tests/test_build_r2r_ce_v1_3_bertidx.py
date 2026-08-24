import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln" / "scripts" / "build_r2r_ce_v1_3_bertidx.py"


def _write(path, token):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "instruction_vocab": {"word_list": ["old"]},
        "episodes": [{
            "episode_id": "1",
            "scene_id": "data/scene/scene.glb",
            "start_rotation": [0, 0, 0, 1],
            "instruction": {
                "instruction_text": "go",
                "instruction_tokens": [token],
            },
        }],
    }
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)


class BuildR2RCEV13BertIndexTest(unittest.TestCase):
    def test_include_train_builds_train_and_evaluation_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v13 = root / "v13"
            v12 = root / "v12"
            output = root / "output"
            for split in ("train", "val_seen", "val_unseen", "test"):
                _write(v13 / split / (split + ".json.gz"), 7)
                _write(v12 / split / (split + "_bertidx.json.gz"), 101)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--v13-root", str(v13),
                    "--v12-bert-root", str(v12),
                    "--output-root", str(output),
                    "--include-train",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            summary = json.loads(completed.stdout)

            self.assertEqual(
                set(summary), {"train", "val_seen", "val_unseen", "test"}
            )
            with gzip.open(
                output / "train" / "train_bertidx.json.gz",
                "rt",
                encoding="utf-8",
            ) as stream:
                built = json.load(stream)
            self.assertEqual(
                built["episodes"][0]["instruction"]["instruction_tokens"],
                [101],
            )

    def test_default_still_builds_evaluation_splits_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v13 = root / "v13"
            v12 = root / "v12"
            output = root / "output"
            for split in ("val_seen", "val_unseen", "test"):
                _write(v13 / split / (split + ".json.gz"), 7)
                _write(v12 / split / (split + "_bertidx.json.gz"), 101)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--v13-root", str(v13),
                    "--v12-bert-root", str(v12),
                    "--output-root", str(output),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            self.assertFalse((output / "train").exists())
            self.assertTrue((output / "val_unseen" / "val_unseen_bertidx.json.gz").is_file())


if __name__ == "__main__":
    unittest.main()
