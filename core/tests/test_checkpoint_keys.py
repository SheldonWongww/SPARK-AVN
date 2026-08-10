from collections import OrderedDict
import unittest

from navtta_core.experiment.checkpoint_keys import (
    normalize_strict_checkpoint_state_dict,
    validate_strict_checkpoint_loading_info,
)


class CheckpointKeysTest(unittest.TestCase):
    def test_exact_keys_are_preserved(self):
        checkpoint = OrderedDict((key, object()) for key in ("a", "b.weight"))
        normalized = normalize_strict_checkpoint_state_dict(
            {"a": None, "b.weight": None}, checkpoint, "policy"
        )
        self.assertEqual(list(normalized), ["a", "b.weight"])
        self.assertIs(normalized["a"], checkpoint["a"])

    def test_whole_module_prefix_is_stripped(self):
        checkpoint = OrderedDict(
            (key, object()) for key in ("module.a", "module.b.weight")
        )
        normalized = normalize_strict_checkpoint_state_dict(
            {"a": None, "b.weight": None}, checkpoint
        )
        self.assertEqual(list(normalized), ["a", "b.weight"])

    def test_whole_module_prefix_is_added(self):
        normalized = normalize_strict_checkpoint_state_dict(
            {"module.a": None, "module.b": None},
            OrderedDict((key, object()) for key in ("a", "b")),
        )
        self.assertEqual(list(normalized), ["module.a", "module.b"])

    def test_missing_key_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, r"missing=\['b'\]"):
            normalize_strict_checkpoint_state_dict(
                {"a": None, "b": None}, {"a": object()}, "policy"
            )

    def test_unexpected_key_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, r"unexpected=\['b'\]"):
            normalize_strict_checkpoint_state_dict(
                {"a": None}, {"a": object(), "b": object()}, "policy"
            )

    def test_mixed_module_prefix_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "key mismatch"):
            normalize_strict_checkpoint_state_dict(
                {"a": None, "b": None},
                {"module.a": object(), "b": object()},
            )

    def test_module_prefix_does_not_hide_real_difference(self):
        with self.assertRaisesRegex(RuntimeError, r"missing=\['b'\]"):
            normalize_strict_checkpoint_state_dict(
                {"a": None, "b": None}, {"module.a": object()}
            )

    def test_allowlisted_auxiliary_prefix_is_filtered(self):
        checkpoint = OrderedDict(
            (
                ("policy.weight", object()),
                ("auxiliary.head.weight", object()),
            )
        )
        normalized = normalize_strict_checkpoint_state_dict(
            {"policy.weight": None},
            checkpoint,
            "policy",
            allowed_unexpected_prefixes=("auxiliary.",),
        )
        self.assertEqual(list(normalized), ["policy.weight"])

    def test_allowlisted_auxiliary_prefix_does_not_hide_missing_key(self):
        with self.assertRaisesRegex(RuntimeError, r"missing=\['policy.bias'\]"):
            normalize_strict_checkpoint_state_dict(
                {"policy.weight": None, "policy.bias": None},
                {
                    "policy.weight": object(),
                    "auxiliary.head.weight": object(),
                },
                allowed_unexpected_prefixes=("auxiliary.",),
            )

    def test_allowlist_does_not_hide_other_unexpected_key(self):
        with self.assertRaisesRegex(RuntimeError, r"unexpected=\['other.weight'\]"):
            normalize_strict_checkpoint_state_dict(
                {"policy.weight": None},
                {
                    "policy.weight": object(),
                    "auxiliary.head.weight": object(),
                    "other.weight": object(),
                },
                allowed_unexpected_prefixes=("auxiliary.",),
            )

    def test_empty_allowlist_prefix_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            normalize_strict_checkpoint_state_dict(
                {"policy.weight": None},
                {"policy.weight": object()},
                allowed_unexpected_prefixes=("",),
            )

    def test_metadata_follows_prefix_conversion(self):
        checkpoint = OrderedDict((("module.a", object()),))
        checkpoint._metadata = OrderedDict(
            (("", {"version": 1}), ("module", {"version": 2}))
        )
        normalized = normalize_strict_checkpoint_state_dict(
            {"a": None}, checkpoint
        )
        self.assertEqual(list(normalized._metadata), [""])
        self.assertEqual(normalized._metadata[""], {"version": 2})

    def test_empty_transformers_loading_info_passes(self):
        validate_strict_checkpoint_loading_info(
            {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            },
            "streamvln",
        )

    def test_nonempty_transformers_loading_info_fails(self):
        with self.assertRaisesRegex(RuntimeError, "missing_keys"):
            validate_strict_checkpoint_loading_info(
                {"missing_keys": ["model.layer.weight"]}, "streamvln"
            )


if __name__ == "__main__":
    unittest.main()
