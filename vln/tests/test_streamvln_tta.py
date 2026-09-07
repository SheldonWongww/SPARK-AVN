"""CPU regressions for the StreamVLN first-action adapter boundary.

The small policy reproduces Qwen's RMSNorm and BF16 action head arithmetic;
these checks require neither a simulator nor a Transformers installation.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as error:
    raise unittest.SkipTest("StreamVLN TTA CPU tests require torch") from error


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "core"))
sys.path.insert(0, str(REPO_ROOT / "vln"))
sys.path.insert(
    0, str(REPO_ROOT / "vln/baselines/streamvln/streamvln")
)

from navtta_vln.discrete_tta import add_discrete_tta_args  # noqa: E402
from streamvln_tta import (  # noqa: E402
    ACTION_TEXT,
    StreamActionReadout,
    StreamVLNTTAController,
    _stream_forward_policy,
    add_streamvln_tta_args,
)


class _TinyQwen2RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.variance_epsilon = eps

    def forward(self, hidden):
        input_dtype = hidden.dtype
        value = hidden.float()
        variance = value.square().mean(-1, keepdim=True)
        value = value * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * value.to(input_dtype)


class _TinyTokenizer:
    # Deliberately different from ACTION_TEXT order: a full-vocabulary argmax
    # breaks an exact tie by token ID, not by STOP/UP/LEFT/RIGHT ordering.
    token_ids = {"STOP": 9, "↑": 2, "←": 12, "→": 5}

    def encode(self, text, add_special_tokens=False):
        return [self.token_ids[text]]


class _UnreadableFeedback:
    def __float__(self):
        raise AssertionError("this episode route must not read evaluator feedback")

    def __bool__(self):
        raise AssertionError("this episode route must not read evaluator feedback")


class _TinyPolicy(nn.Module):
    def __init__(self, bf16_tie=False):
        super().__init__()
        width = 2 if bf16_tie else 4
        self.model = nn.Module()
        self.model.norm = _TinyQwen2RMSNorm(
            width, eps=0.0 if bf16_tie else 1e-6
        )
        self.lm_head = nn.Linear(width, 16, bias=False)
        rows = (
            ((0.0, 0.0), (0.5, 0.5), (-0.5, -0.5), (0.5, 0.50390625))
            if bf16_tie else
            ((0.6, 0.2, 0.1, -0.3), (-0.2, 0.4, 0.1, 0.2),
             (0.3, -0.1, 0.3, 0.1), (-0.1, -0.4, 0.2, 0.5))
        )
        with torch.no_grad():
            self.lm_head.weight.zero_()
            for text, row in zip(ACTION_TEXT, rows):
                self.lm_head.weight[_TinyTokenizer.token_ids[text]].copy_(
                    torch.tensor(row)
                )
        self.to(dtype=torch.bfloat16)
        self.eval()
        self.requires_grad_(False)

    def forward(self, hidden):
        return self.lm_head(self.model.norm(hidden))


def _args(directory, method, *extra):
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_split", default="val_unseen")
    add_discrete_tta_args(parser)
    add_streamvln_tta_args(parser)
    return parser.parse_args([
        "--tta_method", method,
        "--tta_norm_scope", "last_ln",
        "--tta_diagnostics", str(Path(directory) / "diagnostics.json"),
        *extra,
    ])


def _native_action(model, hidden, action_token_ids):
    with torch.no_grad():
        native = model(hidden)[:, list(action_token_ids)]
    # The residual protocol sorts these columns by token ID.
    token_id = action_token_ids[int(native.argmax(-1).item())]
    return native, token_id


def _hidden(step=0):
    values = (
        (1.0, 2.0, -0.5, 0.25),
        (0.5, 1.0, 0.25, -1.0),
        (1.25, -0.5, 0.75, 0.5),
    )
    return torch.tensor([values[step % len(values)]], dtype=torch.bfloat16)


class StreamActionReadoutTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.token_ids = tuple(sorted(_TinyTokenizer.token_ids.values()))

    def test_bf16_precision_mismatch_is_removed_at_zero_update(self):
        model = _TinyPolicy(bf16_tie=True)
        hidden = torch.ones(1, 2, dtype=torch.bfloat16)
        native, native_token = _native_action(model, hidden, self.token_ids)
        legacy = StreamActionReadout(model, self.token_ids)
        residual = StreamActionReadout(
            model, self.token_ids, protocol="native_residual"
        )

        self.assertEqual(native_token, 2)
        self.assertEqual(native[0, 0].item(), native[0, 1].item())
        self.assertGreater(legacy(hidden)[0, 1], legacy(hidden)[0, 0])
        self.assertEqual(self.token_ids[legacy(hidden).argmax(-1).item()], 5)
        self.assertTrue(torch.equal(residual(hidden, native), native.float()))

    def test_residual_zero_update_preserves_all_native_logits(self):
        model = _TinyPolicy()
        readout = StreamActionReadout(
            model, self.token_ids, protocol="native_residual"
        )
        for step in range(3):
            with self.subTest(step=step):
                hidden = _hidden(step)
                native, _ = _native_action(model, hidden, self.token_ids)
                self.assertTrue(torch.equal(readout(hidden, native), native.float()))

    def test_residual_requires_native_logits_and_legacy_does_not(self):
        model = _TinyPolicy()
        residual = StreamActionReadout(
            model, self.token_ids, protocol="native_residual"
        )
        with self.assertRaisesRegex(ValueError, "native"):
            residual(_hidden())
        legacy = StreamActionReadout(model, self.token_ids)
        self.assertEqual(tuple(legacy(_hidden()).shape), (1, 4))

    def test_frozen_baseline_survives_updates_and_replay_is_exact(self):
        model = _TinyPolicy()
        source_state = deepcopy(model.state_dict())
        readout = StreamActionReadout(
            model, self.token_ids, protocol="native_residual"
        )
        readout.norm.weight.requires_grad_(True)
        hidden = _hidden()
        native, _ = _native_action(model, hidden, self.token_ids)
        inputs = {"pre_norm_hidden": hidden, "native_action_logits": native}
        initial_logits = readout.initial_logits(hidden).detach().clone()
        initial_weight = readout.norm.weight.detach().clone()
        optimizer = torch.optim.SGD([readout.norm.weight], lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(readout(hidden, native), torch.tensor([2]))
        loss.backward()
        self.assertGreater(readout.norm.weight.grad.abs().max().item(), 0.0)
        optimizer.step()

        self.assertFalse(torch.equal(readout.norm.weight, initial_weight))
        self.assertTrue(torch.equal(readout.initial_logits(hidden), initial_logits))
        self.assertFalse(torch.equal(readout(hidden, native), native.float()))
        replay_features, replay_logits = _stream_forward_policy(readout, inputs)
        self.assertTrue(torch.equal(replay_logits, readout(hidden, native)))
        self.assertTrue(torch.equal(replay_features, readout.policy_features(hidden)))
        for key, value in source_state.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value), key)

    def test_residual_self_prediction_features_reach_norm_weight(self):
        model = _TinyPolicy()
        readout = StreamActionReadout(
            model, self.token_ids, protocol="native_residual"
        )
        readout.norm.weight.requires_grad_(True)
        hidden = _hidden()
        native, _ = _native_action(model, hidden, self.token_ids)
        features, _ = _stream_forward_policy(
            readout,
            {"pre_norm_hidden": hidden, "native_action_logits": native},
        )
        self.assertTrue(torch.equal(features, readout.norm(hidden).float()))
        # ATENA reconstructs this feature-gradient term at episode end.
        gradient = torch.autograd.grad(features.square().sum(), readout.norm.weight)[0]
        self.assertGreater(gradient.abs().max().item(), 0.0)
        legacy = StreamActionReadout(model, self.token_ids)
        self.assertTrue(torch.equal(legacy.policy_features(hidden), hidden.float()))

    def test_native_baseline_has_no_gradient_path_into_source_graph(self):
        model = _TinyPolicy()
        readout = StreamActionReadout(
            model, self.token_ids, protocol="native_residual"
        )
        readout.norm.weight.requires_grad_(True)
        native, _ = _native_action(model, _hidden(), self.token_ids)
        native = native.float().requires_grad_(True)
        loss = F.cross_entropy(readout(_hidden(), native), torch.tensor([2]))
        norm_gradient, native_gradient = torch.autograd.grad(
            loss, (readout.norm.weight, native), allow_unused=True
        )
        self.assertGreater(norm_gradient.abs().max().item(), 0.0)
        self.assertIsNone(native_gradient)

    def test_only_the_live_norm_is_selected_as_a_parameter(self):
        readout = StreamActionReadout(
            _TinyPolicy(), self.token_ids, protocol="native_residual"
        )
        self.assertEqual(list(dict(readout.named_parameters())), ["norm.weight"])


class StreamVLNTTAControllerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        random.seed(23)

    def _step(self, controller, model, hidden):
        native, native_token = _native_action(
            model, hidden, controller.action_token_ids
        )
        return controller.adapt_first_token(
            hidden,
            native_action_logits=native,
            native_token_id=native_token,
        )

    def test_legacy_remains_the_default_and_accepts_hidden_only(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(directory, "tent", "--tta_audit_zero_update")
            self.assertEqual(args.tta_streamvln_readout_protocol, "legacy_v4")
            model = _TinyPolicy(bf16_tie=True)
            controller = StreamVLNTTAController(args, model, _TinyTokenizer())
            expected_order = tuple(_TinyTokenizer.token_ids[text] for text in ACTION_TEXT)
            self.assertEqual(controller.action_token_ids, expected_order)
            hidden = torch.ones(1, 2, dtype=torch.bfloat16)
            expected = int(controller.readout(hidden).argmax(-1).item())
            controller.begin_episode()
            self.assertEqual(controller.adapt_first_token(hidden), expected)
            controller.end_episode(success=True)

    def test_frozen_methods_preserve_native_token_id_ties(self):
        for method in ("tent", "fstta", "eam", "feedtta", "atena"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                model = _TinyPolicy(bf16_tie=True)
                args = _args(
                    directory, method,
                    "--tta_streamvln_readout_protocol", "native_residual",
                    "--tta_audit_zero_update",
                    "--tta_fstta_m", "1",
                    "--tta_eam_batch_size", "1",
                    "--tta_eam_confidence_scale", "1.0",
                )
                controller = StreamVLNTTAController(args, model, _TinyTokenizer())
                self.assertEqual(controller.action_token_ids, (2, 5, 9, 12))
                initial_weight = controller.readout.norm.weight.detach().clone()
                controller.begin_episode()
                hidden = torch.ones(1, 2, dtype=torch.bfloat16)
                for _ in range(2):
                    action = self._step(controller, model, hidden)
                    self.assertEqual(controller.action_token_id(action), 2)
                controller.end_episode(success=True)
                self.assertTrue(torch.equal(controller.readout.norm.weight, initial_weight))
                payload = json.loads(Path(args.tta_diagnostics).read_text(encoding="utf-8"))
                self.assertEqual(payload["schema"], "navtta.streamvln_tta_diagnostics.v2")
                self.assertEqual(payload["readout_protocol"], "native_residual")
                comparisons = payload["action_comparisons"]
                self.assertEqual(comparisons["native_vs_frozen_fp32_flips"], 2)
                self.assertEqual(comparisons["native_vs_adapted_flips"], 0)
                self.assertEqual(comparisons["native_vs_prepared_flips"], 0)
                self.assertEqual(comparisons["frozen_fp32_vs_adapted_fp32_flips"], 0)
                self.assertEqual(len(payload["episode_diagnostics"]), 1)

    def test_tiny_lr_can_attempt_updates_without_changing_fp32_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _TinyPolicy()
            args = _args(
                directory, "tent", "--tta_lr", "1e-8",
                "--tta_streamvln_readout_protocol", "native_residual",
            )
            controller = StreamVLNTTAController(args, model, _TinyTokenizer())
            initial_weight = controller.readout.norm.weight.detach().clone()
            controller.begin_episode()
            self._step(controller, model, _hidden())
            controller.end_episode(success=True)
            self.assertGreater(controller.adapter.diagnostics()["updates"], 0)
            self.assertTrue(torch.equal(controller.readout.norm.weight, initial_weight))
            payload = json.loads(Path(args.tta_diagnostics).read_text(encoding="utf-8"))
            self.assertEqual(payload["parameter_changes"]["changed_boundaries"], 0)

    def test_all_methods_can_update_and_keep_the_native_checkpoint_frozen(self):
        for method in ("tent", "fstta", "eam", "feedtta", "atena"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                model = _TinyPolicy()
                source_state = deepcopy(model.state_dict())
                args = _args(
                    directory, method,
                    "--tta_streamvln_readout_protocol", "native_residual",
                    "--tta_lr", "1e-3",
                    "--tta_fstta_m", "1", "--tta_fstta_n", "2",
                    "--tta_fstta_lr_slow", "1e-3",
                    "--tta_eam_lr", "1e-3",
                    "--tta_eam_batch_size", "2",
                    "--tta_eam_confidence_scale", "1.0",
                    "--tta_feedtta_lr", "1e-3", "--tta_feedtta_p", "0",
                    "--tta_atena_lr_query", "1e-3",
                    "--tta_atena_lr_self", "1e-3",
                )
                controller = StreamVLNTTAController(args, model, _TinyTokenizer())
                adapted_model = (
                    controller.adapter.aux_model if method == "eam"
                    else controller.readout
                )
                initial_weight = adapted_model.norm.weight.detach().clone()
                for episode in range(2):
                    controller.begin_episode()
                    for step in range(3):
                        self._step(controller, model, _hidden(step))
                    controller.end_episode(success=episode == 0)
                diagnostics = controller.adapter.diagnostics()
                self.assertGreater(diagnostics["updates"], 0)
                self.assertFalse(torch.equal(adapted_model.norm.weight, initial_weight))
                if method == "fstta":
                    self.assertEqual(diagnostics["completed_slow_windows"], 1)
                if method in ("eam", "atena"):
                    self.assertGreater(diagnostics["replayed_steps"], 0)
                if method == "atena":
                    self.assertTrue(diagnostics["exact_episode_replay_validated"])
                    self.assertEqual(diagnostics["max_replay_feature_abs_error"], 0.0)
                    self.assertEqual(diagnostics["self_prediction_feature_dim"], 4)
                for key, value in source_state.items():
                    self.assertTrue(torch.equal(model.state_dict()[key], value), key)

    def test_eam_replay_keeps_each_steps_native_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _TinyPolicy()
            args = _args(
                directory, "eam",
                "--tta_streamvln_readout_protocol", "native_residual",
                "--tta_eam_batch_size", "2",
                "--tta_eam_confidence_scale", "1.0",
            )
            controller = StreamVLNTTAController(args, model, _TinyTokenizer())
            controller.begin_episode()
            online_auxiliary_logits = []
            for step in range(3):
                native, _ = _native_action(
                    model, _hidden(step), controller.action_token_ids
                )
                with torch.no_grad():
                    online_auxiliary_logits.append(
                        controller.adapter.aux_model(_hidden(step), native).clone()
                    )
                self._step(controller, model, _hidden(step))
            for step, entry in enumerate(controller.adapter.replay):
                inputs = entry["policy_inputs"]
                expected, _ = _native_action(model, _hidden(step), controller.action_token_ids)
                self.assertTrue(torch.equal(inputs["native_action_logits"], expected))
                self.assertFalse(inputs["native_action_logits"].requires_grad)
                self.assertEqual(inputs["native_action_logits"].device.type, "cpu")
                _, replay_logits = _stream_forward_policy(controller.readout, inputs)
                self.assertTrue(torch.equal(replay_logits, expected.float()))
                self.assertTrue(torch.equal(entry["source_decision"], expected.float()))
                self.assertTrue(torch.equal(
                    entry["auxiliary_decision"], online_auxiliary_logits[step]
                ))
                _, auxiliary_replay = _stream_forward_policy(
                    controller.adapter.aux_model, inputs
                )
                # Reconstruct the frozen reference from the original model;
                # it must stay frozen when EAM updates its auxiliary copy.
                checkpoint_norm = deepcopy(model.model.norm).float()
                action_rows = model.lm_head.weight[
                    list(controller.action_token_ids)
                ].float()
                frozen_reference = F.linear(
                    checkpoint_norm(_hidden(step)).float(), action_rows
                )
                adapted_reference = F.linear(
                    controller.adapter.aux_model.norm(_hidden(step)).float(),
                    action_rows,
                )
                self.assertTrue(torch.equal(
                    auxiliary_replay,
                    expected.float() + (adapted_reference - frozen_reference),
                ))
            controller.end_episode(success=True)

    def test_atena_records_post_norm_features_for_exact_episode_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _TinyPolicy()
            args = _args(
                directory, "atena",
                "--tta_streamvln_readout_protocol", "native_residual",
            )
            controller = StreamVLNTTAController(args, model, _TinyTokenizer())
            controller.begin_episode()
            self._step(controller, model, _hidden())
            stored_feature = controller.adapter.trajectory_features[0]
            expected = controller.readout.norm(_hidden()).float()
            self.assertTrue(torch.equal(stored_feature, expected))
            self.assertFalse(torch.equal(stored_feature, _hidden().float()))
            inputs = controller.adapter.trajectory[0]["policy_inputs"]
            self.assertIn("native_action_logits", inputs)
            controller.end_episode(success=True)
            self.assertTrue(controller.adapter.diagnostics()["exact_episode_replay_validated"])

    def test_atena_episode_routes_follow_actual_query_and_self_label_updates(self):
        for route, threshold, feedback in (
            ("query", "0", True),
            ("self", "100", _UnreadableFeedback()),
        ):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as directory:
                model = _TinyPolicy()
                args = _args(
                    directory, "atena",
                    "--tta_streamvln_readout_protocol", "native_residual",
                    "--tta_atena_query_threshold", threshold,
                    "--tta_atena_lr_query", "1e-3",
                    "--tta_atena_lr_self", "1e-3",
                )
                controller = StreamVLNTTAController(args, model, _TinyTokenizer())
                controller.begin_episode()
                self._step(controller, model, _hidden())
                before_end = controller.readout.norm.weight.detach().clone()
                controller.end_episode(success=feedback)
                delta = controller.readout.norm.weight.detach() - before_end
                changed_elements = int(torch.count_nonzero(delta).item())
                self.assertGreater(changed_elements, 0)
                self.assertEqual(controller.adapter.query_count, int(route == "query"))
                self.assertEqual(controller.adapter.self_label_count, int(route == "self"))
                payload = json.loads(Path(args.tta_diagnostics).read_text(encoding="utf-8"))
                episode = payload["episode_diagnostics"][0]
                self.assertEqual(episode["episode_route"], route)
                if route == "query":
                    self.assertIs(episode["feedback_success"], True)
                else:
                    self.assertNotIn("feedback_success", episode)
                    self.assertEqual(controller.adapter.feedback_observed_count, 0)
                boundary = "episode_end_" + route
                changes = episode["parameter_changes"]["by_boundary"]
                self.assertNotIn("episode_end", changes)
                self.assertEqual(changes[boundary]["observed_boundaries"], 1)
                self.assertEqual(changes[boundary]["changed_boundaries"], 1)
                self.assertEqual(changes[boundary]["changed_elements"], changed_elements)
                self.assertEqual(changes[boundary]["max_abs_change"], delta.abs().max().item())
                self.assertEqual(
                    payload["parameter_changes"]["by_boundary"][boundary],
                    changes[boundary],
                )

    def test_atena_empty_episode_reports_no_route_and_never_reads_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _TinyPolicy()
            args = _args(
                directory, "atena",
                "--tta_streamvln_readout_protocol", "native_residual",
            )
            controller = StreamVLNTTAController(args, model, _TinyTokenizer())
            controller.begin_episode()
            controller.end_episode(success=_UnreadableFeedback())
            self.assertEqual(controller.adapter.query_count, 0)
            self.assertEqual(controller.adapter.self_label_count, 0)
            payload = json.loads(Path(args.tta_diagnostics).read_text(encoding="utf-8"))
            episode = payload["episode_diagnostics"][0]
            self.assertEqual(episode["episode_route"], "none")
            self.assertNotIn("feedback_success", episode)
            changes = episode["parameter_changes"]["by_boundary"]["episode_end_none"]
            self.assertEqual(changes["observed_boundaries"], 1)
            self.assertEqual(changes["changed_boundaries"], 0)
            self.assertEqual(changes["changed_elements"], 0)

    def test_feedtta_success_and_failure_updates_have_distinct_episode_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _TinyPolicy()
            args = _args(
                directory, "feedtta",
                "--tta_streamvln_readout_protocol", "native_residual",
                "--tta_feedtta_lr", "1e-3", "--tta_feedtta_p", "0",
            )
            controller = StreamVLNTTAController(args, model, _TinyTokenizer())
            expected_changes = []
            for feedback in (True, False):
                controller.begin_episode()
                self._step(controller, model, _hidden())
                before_end = controller.readout.norm.weight.detach().clone()
                controller.end_episode(success=feedback)
                delta = controller.readout.norm.weight.detach() - before_end
                expected_changes.append(int(torch.count_nonzero(delta).item()))
            self.assertEqual(controller.adapter.successful_episodes, 1)
            self.assertEqual(controller.adapter.failed_episodes, 1)
            payload = json.loads(Path(args.tta_diagnostics).read_text(encoding="utf-8"))
            for index, (route, feedback) in enumerate((("success", True), ("failure", False))):
                episode = payload["episode_diagnostics"][index]
                self.assertEqual(episode["episode_route"], route)
                self.assertIs(episode["feedback_success"], feedback)
                boundary = "episode_end_" + route
                changes = episode["parameter_changes"]["by_boundary"]
                self.assertNotIn("episode_end", changes)
                self.assertEqual(changes[boundary]["observed_boundaries"], 1)
                self.assertEqual(changes[boundary]["changed_boundaries"], 1)
                self.assertEqual(changes[boundary]["changed_elements"], expected_changes[index])
                self.assertGreater(changes[boundary]["changed_elements"], 0)
                self.assertEqual(
                    payload["parameter_changes"]["by_boundary"][boundary],
                    changes[boundary],
                )

    def test_feedtta_empty_episode_still_records_the_feedback_it_consumes(self):
        with tempfile.TemporaryDirectory() as directory:
            model = _TinyPolicy()
            args = _args(
                directory, "feedtta",
                "--tta_streamvln_readout_protocol", "native_residual",
            )
            controller = StreamVLNTTAController(args, model, _TinyTokenizer())
            controller.begin_episode()
            controller.end_episode(success=False)
            self.assertEqual(controller.adapter.feedback_episode_count, 1)
            payload = json.loads(Path(args.tta_diagnostics).read_text(encoding="utf-8"))
            episode = payload["episode_diagnostics"][0]
            self.assertEqual(episode["episode_route"], "failure")
            self.assertIs(episode["feedback_success"], False)
            changes = episode["parameter_changes"]["by_boundary"]["episode_end_failure"]
            self.assertEqual(changes["observed_boundaries"], 1)
            self.assertEqual(changes["changed_boundaries"], 0)

    def test_unsupervised_methods_never_read_or_record_feedback(self):
        for method in ("tent", "fstta", "eam"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                model = _TinyPolicy()
                args = _args(
                    directory, method,
                    "--tta_streamvln_readout_protocol", "native_residual",
                )
                controller = StreamVLNTTAController(args, model, _TinyTokenizer())
                controller.begin_episode()
                self._step(controller, model, _hidden())
                controller.end_episode(success=_UnreadableFeedback())
                payload = json.loads(Path(args.tta_diagnostics).read_text(encoding="utf-8"))
                episode = payload["episode_diagnostics"][0]
                self.assertNotIn("feedback_success", episode)
                self.assertNotIn("episode_route", episode)
                changes = episode["parameter_changes"]["by_boundary"]
                self.assertIn("episode_end", changes)
                self.assertFalse(any(key.startswith("episode_end_") for key in changes))


if __name__ == "__main__":
    unittest.main()
