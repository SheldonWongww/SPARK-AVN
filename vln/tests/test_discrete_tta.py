import argparse
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
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
    DiscreteTTAAgentMixin,
    DiscreteTTAController,
    _feedtta_trainable_prefixes,
    _graph_decision_features,
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
            features=_graph_decision_features(outputs),
            policy_inputs=inputs,
        )
        return decision_logits, action

    def test_duet_online_and_replay_use_concatenated_global_local_cls(self):
        policy = _TinyGraphPolicy()
        x = torch.randn(1, 6)
        outputs = policy("navigation", {"x": x})
        online_features = DiscreteTTAAgentMixin().tta_graph_features(outputs)
        features, logits = discrete_forward_policy(
            policy,
            {
                "family": "graph",
                "fusion": "dynamic",
                "model_inputs": {"x": x},
            },
        )
        expected = torch.cat(
            [outputs["gmap_embeds"][:, 0], outputs["vp_embeds"][:, 0]],
            dim=-1,
        )
        self.assertEqual(tuple(features.shape), (1, 12))
        self.assertTrue(torch.equal(online_features, expected))
        self.assertTrue(torch.equal(features, expected))
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertTrue(torch.isfinite(logits).all())

    def test_graph_feature_contract_preserves_native_duet_and_goat_states(self):
        global_cls = torch.tensor([[1.0, 2.0]])
        local_cls = torch.tensor([[3.0, 4.0]])
        branches = {
            "gmap_embeds": global_cls.unsqueeze(1),
            "vp_embeds": local_cls.unsqueeze(1),
        }

        # An ATENA-instrumented DUET exposes the same concatenation directly.
        duet_state = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
        self.assertIs(
            _graph_decision_features({**branches, "curr_state": duet_state}),
            duet_state,
        )

        # GOAT's recurrent history state is model-specific and must not be
        # replaced by DUET's global/local concatenation.
        goat_state = torch.tensor([[9.0, 10.0, 11.0]])
        self.assertIs(
            _graph_decision_features({**branches, "cls_embeds": goat_state}),
            goat_state,
        )

    def test_atena_head_accepts_duet_double_width_feature_and_replays_it(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = _TinyGraphPolicy(width=6)
            controller = DiscreteTTAController(
                _args(directory, "atena"), policy, "val_unseen"
            )
            controller.begin_episode()
            self._step(controller, policy, torch.randn(1, 6))
            self.assertEqual(
                controller.adapter.self_prediction_head[0].in_features, 12
            )
            controller.end_episode(observations=[{"distance": 2.5}])
            diagnostics = controller.adapter.diagnostics()
            self.assertEqual(diagnostics["replayed_steps"], 1)
            self.assertEqual(diagnostics["self_prediction_feature_dim"], 12)
            self.assertEqual(
                diagnostics["max_replay_feature_abs_error"], 0.0
            )

    def test_feedtta_duet_scope_profiles_start_at_crossmodal_stacks(self):
        class CrossmodalStack(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.x_layers = nn.ModuleList([
                    nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)
                ])

        policy = nn.Module()
        policy.vln_bert = nn.Module()
        policy.vln_bert.global_encoder = CrossmodalStack()
        policy.vln_bert.local_encoder = CrossmodalStack()
        policy.vln_bert.global_sap_head = nn.Linear(2, 1)
        policy.vln_bert.local_sap_head = nn.Linear(2, 1)
        policy.vln_bert.sap_fuse_linear = nn.Linear(4, 1)
        policy.vln_bert.gmap_pooler = nn.Linear(2, 2)

        full = _feedtta_trainable_prefixes(policy, "paper_full")
        local = _feedtta_trainable_prefixes(policy, "last_crossmodal")
        heads = _feedtta_trainable_prefixes(policy, "action_head")
        self.assertNotIn("vln_bert.gmap_pooler", full)
        self.assertNotIn("vln_bert.global_encoder", full)
        self.assertIn(
            "vln_bert.global_encoder.encoder.x_layers", full
        )
        self.assertIn(
            "vln_bert.global_encoder.encoder.x_layers.2", local
        )
        self.assertNotIn("vln_bert.global_encoder", heads)
        self.assertEqual(len(heads), 3)

    def test_feedtta_goat_scope_uses_crossattention_and_excludes_poolers(self):
        class GoatCrossmodalStack(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.crossattention = nn.ModuleList([
                    nn.Linear(2, 2), nn.Linear(2, 2)
                ])

        policy = nn.Module()
        policy.vln_bert = nn.Module()
        policy.vln_bert.global_encoder = GoatCrossmodalStack()
        policy.vln_bert.local_encoder = GoatCrossmodalStack()
        policy.vln_bert.global_sap_head = nn.Linear(2, 1)
        policy.vln_bert.local_sap_head = nn.Linear(2, 1)
        policy.vln_bert.sap_fuse_linear = nn.Linear(4, 1)
        policy.vln_bert.gmap_pooler = nn.Linear(2, 2)
        policy.vln_bert.local_his_map = nn.Linear(6, 2)

        full = _feedtta_trainable_prefixes(policy, "paper_full")
        local = _feedtta_trainable_prefixes(policy, "last_crossmodal")
        self.assertIn(
            "vln_bert.global_encoder.encoder.crossattention", full
        )
        self.assertIn(
            "vln_bert.local_encoder.encoder.crossattention.1", local
        )
        self.assertFalse(any("pooler" in prefix for prefix in full))
        self.assertFalse(any("local_his" in prefix for prefix in full))

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

    def test_feedtta_auto_preserves_target_native_argmax(self):
        prefixes = (
            "--tta_trainable_prefixes",
            "vln_bert.global_encoder",
            "vln_bert.local_encoder",
            "vln_bert.global_sap_head",
            "vln_bert.local_sap_head",
            "vln_bert.sap_fuse_linear",
        )
        with tempfile.TemporaryDirectory() as directory:
            policy = _TinyGraphPolicy()
            controller = DiscreteTTAController(
                _args(directory, "feedtta", *prefixes), policy, "val_unseen"
            )
            controller.begin_episode()
            logits, action = self._step(
                controller, policy, torch.randn(1, 6)
            )
            self.assertEqual(int(action.item()), int(logits.argmax(dim=-1).item()))
            controller.end_episode(episode_stats={"success": 1.0})
            self.assertEqual(
                controller.diagnostics()["action_selection_protocol"],
                "target_native_argmax",
            )

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
                    "--tta_action_selection", "sample",
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

    def test_r2r_feedback_uses_submitted_trajectory_endpoint(self):
        class FakeEnvironment:
            gt_trajs = {"instruction": ("scan", ["start", "goal"])}
            calls = 0

            @classmethod
            def _eval_item(cls, scan, path, ground_truth):
                cls.calls += 1
                self.assertEqual(scan, "scan")
                self.assertEqual(ground_truth, ["start", "goal"])
                self.assertEqual(path[-1][-1], "goal")
                return {"success": 1.0}

        agent = DiscreteTTAAgentMixin()
        agent.tta_controller = SimpleNamespace(method="feedtta")
        agent.env = FakeEnvironment()
        stats = agent.tta_r2r_episode_stats(
            [{
                "instr_id": "instruction",
                "path": [["start"], ["reranked", "goal"]],
            }],
            "nested_graph_path",
        )
        self.assertEqual(stats, {"success": 1.0})
        self.assertEqual(FakeEnvironment.calls, 1)
        self.assertEqual(
            agent.tta_controller.binary_feedback_endpoint,
            "r2r_submitted_trajectory_evaluator_success_every_episode",
        )

        agent.tta_controller = SimpleNamespace(method="atena")
        feedback = agent.tta_r2r_episode_stats(
            [{
                "instr_id": "instruction",
                "path": [["start"], ["reranked", "goal"]],
            }],
            "nested_graph_path",
        )
        self.assertTrue(callable(feedback))
        self.assertEqual(FakeEnvironment.calls, 1)
        self.assertEqual(feedback(), {"success": 1.0})
        self.assertEqual(FakeEnvironment.calls, 2)
        self.assertEqual(
            agent.tta_controller.binary_feedback_endpoint,
            "r2r_submitted_trajectory_evaluator_success_lazy_query",
        )

    def test_reverie_feedback_uses_only_submitted_navigation_success(self):
        class GraphEnvironment:
            gt_trajs = {
                "instruction": ("scan", ["start", "goal"], "target")
            }
            calls = 0

            @classmethod
            def _eval_item(
                cls, scan, path, predicted_object, ground_truth, target_object
            ):
                cls.calls += 1
                self.assertEqual(scan, "scan")
                self.assertEqual(path[-1][-1], "goal")
                self.assertEqual(predicted_object, "wrong-object")
                self.assertEqual(ground_truth, ["start", "goal"])
                self.assertEqual(target_object, "target")
                # Grounding deliberately fails.  Binary-feedback TTA must not
                # receive or infer that label.
                return {"success": 1.0, "rgs": 0.0, "rgspl": 0.0}

        agent = DiscreteTTAAgentMixin()
        agent.env = GraphEnvironment()
        trajectory = [{
            "instr_id": "instruction",
            "path": [["start"], ["reranked", "goal"]],
            "pred_objid": "wrong-object",
        }]

        agent.tta_controller = SimpleNamespace(method="feedtta")
        stats = agent.tta_reverie_episode_stats(
            trajectory, "nested_graph_path"
        )
        self.assertEqual(stats, {"success": 1.0})
        self.assertEqual(GraphEnvironment.calls, 1)
        self.assertEqual(
            agent.tta_controller.binary_feedback_endpoint,
            "reverie_submitted_trajectory_evaluator_navigation_"
            "success_every_episode",
        )

        agent.tta_controller = SimpleNamespace(method="atena")
        callback = agent.tta_reverie_episode_stats(
            trajectory, "nested_graph_path"
        )
        self.assertTrue(callable(callback))
        self.assertEqual(GraphEnvironment.calls, 1)
        self.assertEqual(callback(), {"success": 1.0})
        self.assertEqual(GraphEnvironment.calls, 2)
        self.assertEqual(
            agent.tta_controller.binary_feedback_endpoint,
            "reverie_submitted_trajectory_evaluator_navigation_"
            "success_lazy_query",
        )

    def test_reverie_hamt_feedback_uses_submitted_tuple_endpoint(self):
        class HamtEnvironment:
            gt_trajs = {
                "instruction": ("scan", ["start", "goal"], "target")
            }

            @staticmethod
            def _eval_item(
                scan, path, ground_truth, predicted_object, target_object
            ):
                self.assertEqual(path, ["start", "goal"])
                self.assertEqual(predicted_object, "prediction")
                return {"success": 0.0, "rgs": 1.0}

        agent = DiscreteTTAAgentMixin()
        agent.tta_controller = SimpleNamespace(method="feedtta")
        agent.env = HamtEnvironment()
        stats = agent.tta_reverie_episode_stats(
            [{
                "instr_id": "instruction",
                "path": [("start", 0.0, 0.0), ("goal", 0.0, 0.0)],
                "predObjId": "prediction",
            }],
            "viewpoint_tuples",
        )
        self.assertEqual(stats, {"success": 0.0})

    def test_reverie_controller_rejects_simulator_distance_fallback(self):
        prefixes = (
            "--tta_trainable_prefixes",
            "vln_bert.global_encoder",
            "vln_bert.local_encoder",
            "vln_bert.global_sap_head",
            "vln_bert.local_sap_head",
            "vln_bert.sap_fuse_linear",
        )
        with tempfile.TemporaryDirectory() as directory:
            policy = _TinyGraphPolicy()
            args = _args(directory, "feedtta", *prefixes)
            args.dataset = "reverie"
            controller = DiscreteTTAController(
                args, policy, "val_seen"
            )
            controller.begin_episode()
            self._step(controller, policy, torch.randn(1, 6))
            with self.assertRaisesRegex(
                ValueError, "simulator-distance fallback is forbidden"
            ):
                controller.end_episode(observations=[{"distance": 0.0}])


if __name__ == "__main__":
    unittest.main()
