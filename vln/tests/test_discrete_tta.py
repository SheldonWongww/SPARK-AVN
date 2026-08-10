import argparse
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as error:  # Host-side layout checks need no Torch.
    raise unittest.SkipTest("discrete TTA CPU tests require torch") from error


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "core"))
sys.path.insert(0, str(REPO_ROOT / "vln"))

from navtta_vln.discrete_tta import (  # noqa: E402
    DiscreteTTAController,
    add_discrete_tta_args,
    discrete_forward_policy,
)


class _TinyInner(nn.Module):
    def __init__(self, width=6):
        super().__init__()
        self.global_encoder = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.Tanh()
        )
        self.local_encoder = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.Tanh()
        )
        self.global_sap_head = nn.Linear(width, 3)
        self.local_sap_head = nn.Linear(width, 3)
        self.sap_fuse_linear = nn.Linear(width * 2, 1)


class _TinyGraphPolicy(nn.Module):
    def __init__(self, width=6):
        super().__init__()
        self.vln_bert = _TinyInner(width)

    def forward(self, mode, batch):
        if mode != "navigation":
            raise ValueError(mode)
        global_feature = self.vln_bert.global_encoder(batch["x"])
        local_feature = self.vln_bert.local_encoder(batch["x"] * 0.7)
        weight = torch.sigmoid(
            self.vln_bert.sap_fuse_linear(
                torch.cat([global_feature, local_feature], dim=-1)
            )
        )
        global_logits = self.vln_bert.global_sap_head(global_feature) * weight
        local_logits = self.vln_bert.local_sap_head(local_feature) * (1.0 - weight)
        fused_logits = global_logits + local_logits
        invalid = torch.tensor([False, False, True]).expand_as(fused_logits)
        return {
            "gmap_embeds": global_feature.unsqueeze(1),
            "vp_embeds": local_feature.unsqueeze(1),
            "global_logits": global_logits.masked_fill(invalid, -float("inf")),
            "local_logits": local_logits.masked_fill(invalid, -float("inf")),
            "fused_logits": fused_logits.masked_fill(invalid, -float("inf")),
        }


def _args(output_dir, method, *extra):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=output_dir)
    add_discrete_tta_args(parser)
    return parser.parse_args(["--tta_method", method, *extra])


class DiscreteTTAControllerTest(unittest.TestCase):
    def _step(self, controller, policy, x):
        inputs = controller.policy_inputs(
            {"x": x}, family="graph", fusion="dynamic"
        )
        controller.before_inference(inputs)
        outputs = policy("navigation", {"x": x})
        source_logits = outputs["fused_logits"]
        decision_logits = controller.prepare_action(source_logits, inputs)
        action = controller.select_action(decision_logits)
        controller.adapt_step(
            decision_logits,
            action=action,
            features=0.5 * (
                outputs["gmap_embeds"][:, 0]
                + outputs["vp_embeds"][:, 0]
            ),
            policy_inputs=inputs,
        )
        return decision_logits, action

    def test_replay_callback_preserves_action_dimension_and_is_finite(self):
        policy = _TinyGraphPolicy()
        features, logits = discrete_forward_policy(
            policy,
            {
                "family": "graph",
                "fusion": "dynamic",
                "model_inputs": {"x": torch.randn(1, 6)},
            },
        )
        self.assertEqual(tuple(features.shape), (1, 6))
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertTrue(torch.isfinite(logits).all())

    def test_replay_callback_preserves_runner_level_action_mask(self):
        policy = _TinyGraphPolicy()
        _, logits = discrete_forward_policy(
            policy,
            {
                "family": "graph",
                "fusion": "dynamic",
                "model_inputs": {"x": torch.randn(1, 6)},
                # Action one is finite in the model output but masked by a
                # runner-level constraint such as HAMT no-backtrack.
                "invalid_action_mask": torch.tensor(
                    [[False, True, True]], dtype=torch.bool
                ),
            },
        )
        self.assertEqual(int(logits.argmax(dim=-1).item()), 0)
        self.assertLess(float(logits[0, 1].item()), -1000.0)
        self.assertLess(float(logits[0, 2].item()), -1000.0)

    def test_tent_and_fstta_run_one_cpu_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            for method, extra in (
                ("tent", ("--tta_last_k_ln", "2")),
                (
                    "fstta",
                    (
                        "--tta_last_k_ln", "2", "--tta_fstta_m", "1",
                        "--tta_fstta_n", "2",
                    ),
                ),
            ):
                policy = _TinyGraphPolicy()
                controller = DiscreteTTAController(
                    _args(directory, method, *extra), policy, "val_unseen"
                )
                controller.begin_episode()
                logits, _ = self._step(
                    controller, policy, torch.randn(1, 6)
                )
                self.assertTrue(torch.isfinite(logits).all())
                controller.end_episode(observations=[{"distance": 9.0}])
                self.assertEqual(controller.adapter.diagnostics()["action_steps"], 1)

    def test_replay_and_feedback_methods_run_on_cpu(self):
        prefixes = (
            "--tta_trainable_prefixes",
            "vln_bert.global_encoder",
            "vln_bert.local_encoder",
            "vln_bert.global_sap_head",
            "vln_bert.local_sap_head",
            "vln_bert.sap_fuse_linear",
        )
        with tempfile.TemporaryDirectory() as directory:
            for method in ("eam", "feedtta", "atena"):
                policy = _TinyGraphPolicy()
                extra = prefixes if method in ("eam", "feedtta") else ()
                controller = DiscreteTTAController(
                    _args(
                        directory,
                        method,
                        "--tta_action_seed", "17",
                        *extra,
                    ),
                    policy,
                    "val_unseen",
                )
                controller.begin_episode()
                self._step(controller, policy, torch.randn(1, 6))
                controller.end_episode(observations=[{"distance": 2.5}])
                diagnostics = controller.adapter.diagnostics()
                self.assertEqual(diagnostics["action_steps"], 1)

    def test_source_sampling_is_seeded_and_never_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(
                directory,
                "source",
                "--tta_action_selection", "sample",
                "--tta_action_seed", "23",
            )
            first_policy = _TinyGraphPolicy()
            second_policy = deepcopy(first_policy)
            first = DiscreteTTAController(args, first_policy, "val_unseen")
            second = DiscreteTTAController(args, second_policy, "val_unseen")
            sequences = []
            for controller, policy in (
                (first, first_policy), (second, second_policy)
            ):
                controller.begin_episode()
                actions = []
                for _ in range(5):
                    _, action = self._step(
                        controller, policy, torch.ones(1, 6)
                    )
                    actions.append(int(action.item()))
                controller.end_episode(observations=[{"distance": 8.0}])
                sequences.append(actions)
                self.assertEqual(controller.adapter.diagnostics()["updates"], 0)
            self.assertEqual(sequences[0], sequences[1])

    def test_source_and_feed_reseed_each_matched_episode(self):
        prefixes = (
            "--tta_trainable_prefixes",
            "vln_bert.global_encoder",
            "vln_bert.local_encoder",
            "vln_bert.global_sap_head",
            "vln_bert.local_sap_head",
            "vln_bert.sap_fuse_linear",
        )
        with tempfile.TemporaryDirectory() as directory:
            source_policy = _TinyGraphPolicy()
            feed_policy = deepcopy(source_policy)
            source = DiscreteTTAController(
                _args(
                    directory,
                    "source",
                    "--tta_action_selection", "sample",
                    "--tta_action_seed", "31",
                ),
                source_policy,
                "val_unseen",
            )
            feed = DiscreteTTAController(
                _args(
                    directory,
                    "feedtta",
                    "--tta_action_seed", "31",
                    "--tta_feedtta_lr", "0",
                    *prefixes,
                ),
                feed_policy,
                "val_unseen",
            )

            # Deliberately consume different amounts of randomness in episode
            # zero; episode one must still start from the same RNG state.
            for controller, policy, length in (
                (source, source_policy, 1), (feed, feed_policy, 4)
            ):
                controller.begin_episode()
                for _ in range(length):
                    self._step(controller, policy, torch.ones(1, 6))
                controller.end_episode(observations=[{"distance": 2.0}])

            matched = []
            for controller, policy in (
                (source, source_policy), (feed, feed_policy)
            ):
                controller.begin_episode()
                actions = [
                    int(self._step(
                        controller, policy, torch.ones(1, 6)
                    )[1].item())
                    for _ in range(6)
                ]
                matched.append(actions)
                controller.end_episode(observations=[{"distance": 2.0}])
            self.assertEqual(matched[0], matched[1])


if __name__ == "__main__":
    unittest.main()
