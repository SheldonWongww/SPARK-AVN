import copy
import json
import unittest
from pathlib import Path

from tools.verify_layout import validate_reference_catalog


REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "references" / "catalog.json"


class ReferenceCatalogValidationTest(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    def test_current_catalog_and_future_additions_are_allowed(self):
        self.assertEqual(validate_reference_catalog(self.catalog), [])

        expanded = copy.deepcopy(self.catalog)
        expanded["repositories"].append(
            {
                "name": "future-method",
                "category": "tta/general",
                "url": "https://example.com/future-method.git",
                "commit": "a" * 40,
                "path": "references/repos/tta/general/future-method",
            }
        )
        self.assertEqual(validate_reference_catalog(expanded), [])

    def test_missing_required_repository_is_rejected(self):
        reduced = copy.deepcopy(self.catalog)
        reduced["repositories"] = [
            entry for entry in reduced["repositories"] if entry["name"] != "Tent"
        ]

        errors = validate_reference_catalog(reduced)

        self.assertIn("reference catalog missing required repositories: Tent", errors)

    def test_duplicate_names_and_paths_are_rejected(self):
        duplicated = copy.deepcopy(self.catalog)
        duplicated["repositories"].append(copy.deepcopy(duplicated["repositories"][0]))

        errors = validate_reference_catalog(duplicated)

        self.assertTrue(any("duplicate reference repository name" in error for error in errors))
        self.assertTrue(any("duplicate reference repository path" in error for error in errors))

    def test_missing_and_empty_required_fields_are_rejected(self):
        invalid = copy.deepcopy(self.catalog)
        del invalid["repositories"][0]["url"]
        invalid["repositories"][1]["category"] = ""

        errors = validate_reference_catalog(invalid)

        self.assertTrue(any("missing required fields: url" in error for error in errors))
        self.assertTrue(
            any(
                "field category must be a non-empty string" in error
                for error in errors
            )
        )

    def test_invalid_commit_and_out_of_tree_path_are_rejected(self):
        invalid = copy.deepcopy(self.catalog)
        invalid["repositories"][0]["commit"] = "main"
        invalid["repositories"][0]["path"] = "../VLN-DUET"

        errors = validate_reference_catalog(invalid)

        self.assertTrue(any("40-character lowercase SHA-1" in error for error in errors))
        self.assertTrue(any("normalized under references/repos/" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
