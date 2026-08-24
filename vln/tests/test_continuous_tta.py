import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "core"))
sys.path.insert(0, str(REPO_ROOT / "vln"))

try:
    import torch
    import torch.nn as nn
except ImportError:  # Lightweight repository checks need not install PyTorch.
    torch = None
    nn = None


if torch is not None:
    from navtta_vln.continuous_tta import (
        ContinuousVLNTTA,
        _feedtta_trainable_prefixes,
        make_continuous_tta_config,
    )


    class TinyDecisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.global_encoder = nn.Sequential(
                nn.Linear(4, 4),
                nn.LayerNorm(4),
                nn.Tanh(),
                nn.LayerNorm(4),
            )
            self.global_sap_head = nn.Linear(4, 1)

        def forward_navigation(
            self,
            txt_embeds,
            txt_masks,
            gmap_vp_ids,
            gmap_step_ids,
            gmap_img_fts,
            gmap_pos_fts,
            gmap_masks,
            gmap_visited_masks,
            gmap_pair_dists,
            *bev_inputs
        ):
            del (
                txt_embeds,
                txt_masks,
                gmap_vp_ids,
                gmap_step_ids,
                gmap_pos_fts,
                gmap_masks,
                gmap_pair_dists,
                bev_inputs,
            )
            embeds = self.global_encoder(gmap_img_fts)
            logits = self.global_sap_head(embeds).squeeze(-1)
            logits = logits.masked_fill(gmap_visited_masks, -float("inf"))
            return {
                "gmap_embeds": embeds,
                "global_logits": logits,
                "fused_logits": logits,
            }


def _config(method):
    prefixes = ("global_encoder", "global_sap_head")
    return SimpleNamespace(
        METHOD=method,
        ACTION_SELECTION="argmax",
        MATCHED_FEEDTTA_SOURCE=False,
        ACTION_SEED=19,
        LR=1e-3,
        STEPS=1,
        EPISODIC=False,
        RESET_BN_STATS=True,
        NORM_SCOPE="last_k_ln",
        LAST_K_LN=1,
        OPTIMIZER="SGD",
        MOMENTUM=0.0,
        BETA1=0.9,
        BETA2=0.999,
        WEIGHT_DECAY=0.0,
        MAX_GRAD_NORM=1.0,
        UPDATE_INTERVAL=1,
        MAX_UPDATES_PER_EPISODE=-1,
        FSTTA=SimpleNamespace(
            LR_SLOW=1e-3,
            M=1,
            N=2,
            Q=0.1,
            RHO=0.95,
            TAU=0.7,
            A=0.9,
            B=1.1,
            USE_SLOW=False,
            FAST_GRAD_MODE="mean",
            USE_FAST_LR_SCALER=False,
            OPTIMIZER="SGD",
            BETA1=0.9,
            BETA2=0.99,
            WEIGHT_DECAY=0.0,
            SLOW_OPTIMIZER="SGD",
            SLOW_MOMENTUM=0.0,
            RESET_OPTIMIZER_EACH_EPISODE=True,
            RESET_VAR_HIST_EACH_EPISODE=False,
            RESET_SLOW_OPTIMIZER_EACH_WINDOW=False,
            EIGEN_EPS=1e-6,
        ),
        EAM=SimpleNamespace(
            LR=1e-3,
            CONFIDENCE_SCALE=1.0,
            MEMORY_SIZE=4,
            BATCH_SIZE=2,
            UPDATE_INTERVAL=1,
            PARAM_SCOPE="module_prefixes",
            TRAINABLE_PREFIXES=prefixes,
            OPTIMIZER="SGD",
            MOMENTUM=0.0,
            BETA1=0.9,
            BETA2=0.999,
            WEIGHT_DECAY=0.0,
            MAX_GRAD_NORM=0.0,
        ),
        FEEDTTA=SimpleNamespace(
            LR=1e-3,
            P=0.0,
            ALPHA=-0.2,
            SGR_SEED=0,
            SGR_MODE="paper_main",
            GAMMA=0.99,
            NORMALIZE_GRADIENT=False,
            PARAM_SCOPE="module_prefixes",
            TRAINABLE_PREFIXES=prefixes,
            SCOPE_PROFILE="configured_prefixes",
            OPTIMIZER="SGD",
            BETA1=0.9,
            BETA2=0.999,
            WEIGHT_DECAY=0.0,
            EPS=1e-5,
            MAX_GRAD_NORM=0.0,
        ),
        ATENA=SimpleNamespace(
            LR_QUERY=1e-3,
            LR_SELF=1e-4,
            MIX_LAMBDA=0.5,
            QUERY_THRESHOLD=0.0,
            SELF_LOSS_WEIGHT=0.1,
            PARAM_SCOPE="all",
            TASK_UPDATE_SCOPE="replay_reachable_high_level_navigation",
            OPTIMIZER="SGD",
            BETA1=0.9,
            BETA2=0.999,
            WEIGHT_DECAY=0.0,
            MAX_GRAD_NORM=0.0,
        ),
    )


def _nav_inputs(visited_mask=None):
    image_features = torch.tensor(
        [[
            [0.2, -0.1, 0.4, 0.7],
            [0.5, 0.3, -0.2, 0.1],
            [-0.4, 0.8, 0.2, -0.3],
            [0.1, 0.6, -0.5, 0.9],
        ]],
        dtype=torch.float32,
    )
    return {
        "mode": "navigation",
        "txt_embeds": torch.zeros(1, 2, 4),
        "txt_masks": torch.ones(1, 2, dtype=torch.bool),
        "gmap_vp_ids": [[None, "visited", "candidate-a", "candidate-b"]],
        "gmap_step_ids": torch.zeros(1, 4, dtype=torch.long),
        "gmap_img_fts": image_features,
        "gmap_pos_fts": torch.zeros(1, 4, 1),
        "gmap_masks": torch.ones(1, 4, dtype=torch.bool),
        "gmap_visited_masks": torch.tensor(
            [[False, True, False, False]
             if visited_mask is None else visited_mask],
            dtype=torch.bool,
        ),
        "gmap_pair_dists": torch.zeros(1, 4, 4),
        "bev_fts": torch.zeros(1, 1, 4),
        "bev_pos_fts": torch.zeros(1, 1, 1),
        "bev_masks": torch.ones(1, 1, dtype=torch.bool),
        "bev_nav_masks": torch.ones(1, 1, dtype=torch.bool),
        "bev_cand_idxs": torch.zeros(1, 1, dtype=torch.long),
        "bev_cand_vpids": [[None]],
    }


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ContinuousVLNTTATest(unittest.TestCase):
    def _controller(self, method, variant="etpnav", model=None):
        if not hasattr(self, "temporary_directory"):
            self.temporary_directory = tempfile.TemporaryDirectory()
            self.addCleanup(self.temporary_directory.cleanup)
        return ContinuousVLNTTA(
            TinyDecisionModel() if model is None else model,
            _config(method),
            variant,
            Path(self.temporary_directory.name) / "diagnostics.json",
            stream_name="unit-test",
        )

    def test_config_defaults_record_canonical_method_protocols(self):
        config = make_continuous_tta_config(
            SimpleNamespace, ("global_encoder", "global_sap_head")
        )
        self.assertEqual(config.FSTTA.RHO, 0.95)
        self.assertEqual(config.FSTTA.TAU, 0.7)
        self.assertEqual(config.FSTTA.A, 0.9)
        self.assertEqual(config.FSTTA.B, 1.1)
        self.assertFalse(config.FSTTA.RESET_VAR_HIST_EACH_EPISODE)
        self.assertEqual(config.FEEDTTA.SGR_MODE, "paper_main")
        self.assertEqual(config.FEEDTTA.SCOPE_PROFILE, "paper_full")

    def test_source_compacts_masked_actions_and_maps_back_to_native_index(self):
        controller = self._controller("source")
        controller.begin_episode()
        _, native_logits, native_action = controller.step(_nav_inputs())
        self.assertTrue(torch.isneginf(native_logits[0, 1]))
        self.assertNotEqual(int(native_action.item()), 1)
        controller.end_episode({"success": 1.0})
        diagnostics = controller.diagnostics()
        self.assertEqual(diagnostics["batch_size"], 1)
        self.assertEqual(
            diagnostics["tta_step_unit"],
            "high_level_waypoint_navigation_decision",
        )
        self.assertFalse(diagnostics["low_level_controller_adapted"])

    def test_source_controller_is_prediction_and_parameter_equivalent(self):
        model = TinyDecisionModel()
        native_model = copy.deepcopy(model)
        inputs = _nav_inputs()
        native = native_model.forward_navigation(*(
            inputs[key] for key in ContinuousVLNTTA._ETP_KEYS
        ))
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        controller = self._controller("source", model=model)
        controller.begin_episode()
        outputs, logits, action = controller.step(inputs)
        controller.end_episode(None)
        torch.testing.assert_close(logits, native["global_logits"])
        torch.testing.assert_close(
            outputs["gmap_embeds"], native["gmap_embeds"]
        )
        self.assertEqual(
            int(action.item()), int(native["global_logits"].argmax(dim=-1).item())
        )
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.detach(), before[name])

    def test_tent_updates_once_on_one_high_level_decision(self):
        controller = self._controller("tent")
        controller.begin_episode()
        controller.step(_nav_inputs())
        controller.end_episode({"success": 0.0})
        self.assertEqual(controller.adapter.diagnostics()["updates"], 1)
        self.assertEqual(controller.adapter.diagnostics()["action_steps"], 1)

    def test_fstta_fast_window_is_counted_in_high_level_decisions(self):
        controller = self._controller("fstta")
        controller.begin_episode()
        controller.step(_nav_inputs())
        controller.end_episode({"success": 0.0})
        diagnostics = controller.adapter.diagnostics()
        self.assertEqual(diagnostics["fast_window"], 1)
        self.assertEqual(diagnostics["updates"], 1)

    def test_bevbert_replay_callback_uses_fused_logits(self):
        controller = self._controller("eam", variant="bevbert")
        controller.begin_episode()
        controller.step(_nav_inputs())
        with mock.patch("random.sample", side_effect=lambda population, k: population[:k]):
            controller.step(
                _nav_inputs([False, True, True, False])
            )
        controller.end_episode({"success": 1.0})
        diagnostics = controller.adapter.diagnostics()
        self.assertEqual(diagnostics["action_steps"], 2)
        # The first decision is reservoir warm-up only.  On the second
        # decision Algorithm 3 forms B from one sampled history item plus the
        # explicitly appended current item, so exactly two rows are replayed.
        self.assertEqual(diagnostics["replayed_steps"], 2)
        self.assertEqual(diagnostics["valid_action_metadata_steps"], 2)
        self.assertEqual(diagnostics["masked_invalid_action_slots"], 0)

    def test_feedtta_argmax_port_is_native_and_not_paper_sampling(self):
        controller = self._controller("feedtta")
        controller.begin_episode()
        _, logits, action = controller.step(_nav_inputs())
        self.assertEqual(int(action.item()), int(logits.argmax(dim=-1).item()))
        controller.end_episode({"success": 1.0})
        self.assertEqual(
            controller.diagnostics()["action_selection"],
            "target_native_argmax",
        )
        self.assertEqual(
            controller.adapter.diagnostics()["action_selection_protocol"],
            "target_native_argmax",
        )
        self.assertFalse(
            controller.diagnostics()["feedtta_canonical_protocol"]
        )
        self.assertTrue(
            controller.diagnostics()["feedtta_native_action_protocol"]
        )
        self.assertFalse(
            controller.diagnostics()["feedtta_paper_sampling_protocol"]
        )
        self.assertEqual(
            controller.diagnostics()["feedtta_protocol"],
            "task_adapted_target_native_argmax",
        )

    def test_feedtta_continuous_scope_profiles_are_model_aware(self):
        stack = lambda: SimpleNamespace(  # noqa: E731
            encoder=SimpleNamespace(x_layers=[object(), object(), object()])
        )
        etpnav = SimpleNamespace(
            global_encoder=stack(), global_sap_head=object()
        )
        self.assertEqual(
            _feedtta_trainable_prefixes(
                etpnav, "etpnav", "last_crossmodal", ()
            ),
            ("global_encoder.encoder.x_layers.2", "global_sap_head"),
        )
        bevbert = SimpleNamespace(
            global_encoder=stack(),
            local_encoder=stack(),
            global_sap_head=object(),
            local_sap_head=object(),
            sap_fuse_linear=object(),
        )
        self.assertEqual(
            _feedtta_trainable_prefixes(
                bevbert, "bevbert", "paper_full", ()
            ),
            (
                "global_encoder.encoder.x_layers",
                "local_encoder.encoder.x_layers",
                "global_sap_head",
                "local_sap_head",
                "sap_fuse_linear",
            ),
        )
        self.assertEqual(
            _feedtta_trainable_prefixes(
                bevbert, "bevbert", "action_head", ()
            ),
            ("global_sap_head", "local_sap_head", "sap_fuse_linear"),
        )

    def test_feedtta_explicit_sampling_uses_a_dedicated_stream(self):
        source_model = TinyDecisionModel()
        config = _config("feedtta")
        config.ACTION_SELECTION = "sample"
        if not hasattr(self, "temporary_directory"):
            self.temporary_directory = tempfile.TemporaryDirectory()
            self.addCleanup(self.temporary_directory.cleanup)
        first = ContinuousVLNTTA(
            copy.deepcopy(source_model), config, "etpnav",
            Path(self.temporary_directory.name) / "feed-first.json",
        )
        second = ContinuousVLNTTA(
            copy.deepcopy(source_model), config, "etpnav",
            Path(self.temporary_directory.name) / "feed-second.json",
        )
        first.begin_episode()
        second.begin_episode()
        first_actions, second_actions = [], []
        for _ in range(3):
            first_actions.append(int(first.step(_nav_inputs())[2].item()))
            second_actions.append(int(second.step(_nav_inputs())[2].item()))
        self.assertEqual(first_actions, second_actions)
        first.end_episode({"success": 1.0})
        second.end_episode({"success": 1.0})
        self.assertEqual(
            first.diagnostics()["action_selection"],
            "policy_sampling",
        )
        self.assertFalse(
            first.diagnostics()["feedtta_native_action_protocol"]
        )
        self.assertTrue(
            first.diagnostics()["feedtta_paper_sampling_protocol"]
        )
        self.assertTrue(first.diagnostics()["feedtta_canonical_protocol"])
        self.assertEqual(
            first.diagnostics()["feedtta_protocol"],
            "paper_policy_sampling_ablation",
        )

    def test_matched_source_control_samples_identically_without_updates(self):
        source_model = TinyDecisionModel()
        config = _config("source")
        config.ACTION_SELECTION = "sample"
        config.MATCHED_FEEDTTA_SOURCE = True
        first_model = copy.deepcopy(source_model)
        second_model = copy.deepcopy(source_model)
        before = {
            name: parameter.detach().clone()
            for name, parameter in first_model.named_parameters()
        }
        if not hasattr(self, "temporary_directory"):
            self.temporary_directory = tempfile.TemporaryDirectory()
            self.addCleanup(self.temporary_directory.cleanup)
        first = ContinuousVLNTTA(
            first_model,
            config,
            "etpnav",
            Path(self.temporary_directory.name) / "source-first.json",
        )
        second = ContinuousVLNTTA(
            second_model,
            config,
            "etpnav",
            Path(self.temporary_directory.name) / "source-second.json",
        )
        first.begin_episode()
        second.begin_episode()
        first_actions, second_actions = [], []
        for _ in range(4):
            first_actions.append(int(first.step(_nav_inputs())[2].item()))
            second_actions.append(int(second.step(_nav_inputs())[2].item()))
        first.end_episode({"success": 0.0})
        second.end_episode({"success": 0.0})
        self.assertEqual(first_actions, second_actions)
        self.assertIsNone(first.adapter)
        for name, parameter in first_model.named_parameters():
            torch.testing.assert_close(parameter.detach(), before[name])
        self.assertEqual(
            first.diagnostics()["action_selection"],
            "matched_policy_sampling_source_control",
        )

    def test_legacy_sampled_source_is_normalized_to_matched_control(self):
        config = _config("source")
        config.ACTION_SELECTION = "sample"
        config.MATCHED_FEEDTTA_SOURCE = False
        if not hasattr(self, "temporary_directory"):
            self.temporary_directory = tempfile.TemporaryDirectory()
            self.addCleanup(self.temporary_directory.cleanup)
        controller = ContinuousVLNTTA(
            TinyDecisionModel(),
            config,
            "etpnav",
            Path(self.temporary_directory.name) / "legacy-source.json",
        )
        self.assertTrue(controller.matched_feedtta_source)
        self.assertEqual(
            controller.diagnostics()["action_selection"],
            "matched_policy_sampling_source_control",
        )
        self.assertTrue(controller.diagnostics()["matched_feedtta_source"])

    def test_matched_source_marker_rejects_argmax(self):
        config = _config("source")
        config.MATCHED_FEEDTTA_SOURCE = True
        if not hasattr(self, "temporary_directory"):
            self.temporary_directory = tempfile.TemporaryDirectory()
            self.addCleanup(self.temporary_directory.cleanup)
        with self.assertRaisesRegex(ValueError, "must use policy sampling"):
            ContinuousVLNTTA(
                TinyDecisionModel(),
                config,
                "etpnav",
                Path(self.temporary_directory.name) / "bad-source.json",
            )

    def test_atena_replays_high_level_inputs_at_episode_end(self):
        controller = self._controller("atena")
        controller.begin_episode()
        controller.step(_nav_inputs())
        controller.step(_nav_inputs([False, True, True, False]))
        controller.end_episode({"success": 1.0})
        diagnostics = controller.adapter.diagnostics()
        self.assertEqual(diagnostics["updates"], 1)
        self.assertEqual(diagnostics["max_trajectory_steps"], 2)
        self.assertEqual(diagnostics["gradient_reconstruction"],
                         "exact_step_replay_in_eval_mode")
        task_diagnostics = controller.diagnostics()
        self.assertEqual(
            task_diagnostics["atena_update_scope"],
            "replay_reachable_high_level_navigation",
        )
        self.assertFalse(
            task_diagnostics["atena_full_end_to_end_policy_claimed"]
        )

    def test_atena_optimizer_is_limited_to_continuous_replay_reachability(self):
        model = TinyDecisionModel()
        model.unused_navigation_head = nn.Linear(4, 2)
        controller = self._controller("atena", model=model)
        controller.begin_episode()
        controller.step(_nav_inputs())
        diagnostics = controller.adapter.diagnostics()
        self.assertIsNone(controller.adapter.optimizer)
        self.assertEqual(
            diagnostics["replay_reachability_validation_result"], "passed"
        )
        self.assertIn(
            "unused_navigation_head.weight",
            diagnostics["replay_unreachable_parameter_names"],
        )
        self.assertNotIn(
            "unused_navigation_head.weight",
            diagnostics["optimizer_policy_parameter_names"],
        )
        controller.end_episode({"success": 1.0})
        task_diagnostics = controller.diagnostics()
        self.assertGreater(
            task_diagnostics["atena_replay_unreachable_parameter_count"], 0
        )
        self.assertTrue(
            task_diagnostics["atena_optimizer_scope_matches_reachable"]
        )
        self.assertTrue(
            task_diagnostics["atena_exact_replay_within_declared_scope"]
        )

    def test_atena_enables_declared_high_level_stack_and_rejects_full_policy(self):
        model = TinyDecisionModel()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        controller = self._controller("atena", model=model)
        self.assertTrue(all(
            parameter.requires_grad for parameter in model.parameters()
        ))

        config = _config("atena")
        config.ATENA.TASK_UPDATE_SCOPE = "full_policy"
        with self.assertRaisesRegex(ValueError, "full_policy"):
            ContinuousVLNTTA(
                TinyDecisionModel(),
                config,
                "etpnav",
                Path(self.temporary_directory.name) / "bad-atena.json",
            )


class ContinuousTTAWiringTest(unittest.TestCase):
    def test_both_ce_trainers_use_the_shared_high_level_controller(self):
        trainers = (
            REPO_ROOT
            / "vln/baselines/etpnav/vlnce_baselines/ss_trainer_ETP.py",
            REPO_ROOT
            / "vln/baselines/bevbert/bevbert_ce/vlnce_baselines/ss_trainer_BEV.py",
        )
        for trainer in trainers:
            source = trainer.read_text(encoding="utf-8")
            with self.subTest(trainer=trainer):
                self.assertIn("ContinuousVLNTTA", source)
                self.assertIn("self.tta_controller.step(", source)
                self.assertIn("self.tta_controller.end_episode(metric)", source)


if __name__ == "__main__":
    unittest.main()
