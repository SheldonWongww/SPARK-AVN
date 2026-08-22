import importlib.util
import json
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "vln/scripts/build_order_seed_manifests.py"
SPEC = importlib.util.spec_from_file_location("order_seed_builder", SCRIPT)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


CANONICAL_FILE_SHA256 = {
    "r2r_duet_hamt": "ce1112520da64c70121a76ae604fc33b96a931f86830a21b63c452d01a8d8833",
    "r2r_goat": "b0ac1a9103a9b48741dfbf32f6a835af8e1125021c803dddf72e35f02b6daf68",
    "r2r_ce_v1_3_unified": "3c3edb55e4f7bf2812b3863f15c0bd7c56a230311023aafcd265d7f4a1106a49",
    "reverie_duet_hamt": "71585f6f2193a80c43c2fb99cb843ad3a3c300aaf0bcac553ee2d37bbb34952a",
    "reverie_goat": "35ac1a383a1635155899e10669b14e8924d8cbcaf4eff1b22700e19c868ac037",
}
VAL_UNSEEN_CANONICAL_FILE_SHA256 = {
    "r2r_duet_hamt": "0608b494878bd6a87e65de78069461e8ae86680ed7038bf6eeebf5549a513ad7",
    "r2r_goat": "61d283bdd5699ca8ba7de4ca4cbba686cc3430919614c91da85ccee4bf12258d",
    "r2r_ce_v1_3_unified": "dc9600fd768f1c885601928c5599e7dc8833f07da57c0aa22d01a9d4fe79047f",
    "reverie_duet_hamt": "09b2d4984d0073a6b31dc0ea6057a70c1436b8d54ebb463b901065b23f0c9e47",
    "reverie_goat": "1e5ca8cd785c54def1a7225ecc3cf420a1ef28c1e90e8d02780d58f2709c2db3",
}


class OrderSeedManifestTest(unittest.TestCase):
    def test_builder_requires_exact_supported_integer_seed(self):
        family = BUILDER.FAMILIES[0]
        for seed in (True, 1.0, "1", 0, 4, None):
            with self.subTest(seed=seed), self.assertRaisesRegex(
                    ValueError, "exact integer 1, 2, or 3"):
                BUILDER.expected_manifest(family, seed)
        with self.assertRaisesRegex(ValueError, "val_seen or val_unseen"):
            BUILDER.expected_manifest(family, 1, "test")

    def test_builder_outputs_three_orders_for_both_validation_splits(self):
        outputs = sorted(
            path
            for seed in BUILDER.ORDER_SEEDS
            for path in (BUILDER.ORDER_ROOT / "order_seed_{}".format(seed)).glob(
                "*/*.json"
            )
        )
        self.assertEqual(len(outputs), 30)
        self.assertEqual(
            {path.name for path in outputs},
            {"val_seen.json", "val_unseen.json"},
        )
        subprocess.run(
            ["python3", str(SCRIPT), "--check"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_seed_zero_canonical_files_remain_byte_compatible(self):
        self.assertEqual(set(BUILDER.FAMILIES), set(CANONICAL_FILE_SHA256))
        self.assertEqual(
            set(BUILDER.FAMILIES), set(VAL_UNSEEN_CANONICAL_FILE_SHA256)
        )
        for split, expected_by_family in (
            ("val_seen", CANONICAL_FILE_SHA256),
            ("val_unseen", VAL_UNSEEN_CANONICAL_FILE_SHA256),
        ):
            for family, expected in expected_by_family.items():
                with self.subTest(family=family, split=split):
                    self.assertEqual(
                        BUILDER.sha256_file(
                            str(BUILDER.canonical_path(family, split))
                        ),
                        expected,
                    )

    def test_derived_orders_pin_parent_and_preserve_episode_set(self):
        for family in BUILDER.FAMILIES:
            for split in BUILDER.EVALUATION_SPLITS:
                parent_path = BUILDER.canonical_path(family, split)
                parent = json.loads(parent_path.read_text(encoding="utf-8"))
                orders = []
                for seed in BUILDER.ORDER_SEEDS:
                    derived_path = BUILDER.derived_path(family, seed, split)
                    derived = json.loads(
                        derived_path.read_text(encoding="utf-8")
                    )
                    provenance = derived["derivation"]
                    self.assertEqual(derived["split"], split)
                    self.assertEqual(derived["order_seed"], seed)
                    self.assertEqual(
                        provenance["parent_manifest_path"],
                        parent_path.relative_to(REPO_ROOT).as_posix(),
                    )
                    self.assertEqual(
                        provenance["parent_manifest_sha256"],
                        BUILDER.sha256_file(str(parent_path)),
                    )
                    self.assertEqual(
                        provenance["parent_order_sha256"],
                        parent["order_sha256"],
                    )
                    self.assertEqual(derived["dataset"], parent["dataset"])
                    self.assertEqual(
                        {
                            json.dumps(item, sort_keys=True)
                            for item in derived["episodes"]
                        },
                        {
                            json.dumps(item, sort_keys=True)
                            for item in parent["episodes"]
                        },
                    )
                    orders.append(derived["order_sha256"])
                self.assertEqual(len(set(orders)), 3)


if __name__ == "__main__":
    unittest.main()
