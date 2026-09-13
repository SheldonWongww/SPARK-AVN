"""Exercise ENMuS's eval-mode recurrent ATENA path without Habitat/assets.

Run on CPU or the evaluation server with::

    PYTHONPATH=core python -m unittest avn.tests.test_enmus_atena_replay -v

CUDA/cuDNN checks run automatically when that backend is available.
"""

import ast
from pathlib import Path
import unittest

import torch
from torch import nn

from avn.navtta_avn.enmus_replay import (
    forward_policy_for_replay,
    policy_forward_context,
)
from navtta_core.tta import ATENAAdapter


class _RecurrentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(0.25)
        self.gru = nn.GRU(4, 6, num_layers=2, dropout=0.25,
                          bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(12, 5, batch_first=True)
        self.calls = []
        self.require_native = True
        self.fail_on = None
        for name in ("gru", "lstm"):
            getattr(self, name).register_forward_pre_hook(self._observe(name))

    def _observe(self, name):
        def hook(module, inputs):
            self.calls.append((name, torch.is_grad_enabled(),
                               torch.backends.cudnn.enabled, module.training))
            if self.require_native and torch.backends.cudnn.enabled:
                raise AssertionError("recurrent forward still uses cuDNN")
            if name == self.fail_on:
                raise RuntimeError("injected recurrent forward failure")
        return hook

    def forward(self, observations, rnn_hidden_states, prev_actions, masks,
                ext_memory=None, ext_memory_masks=None):
        inputs = observations["audio"] + prev_actions.float().view(-1, 1, 1) * 0.01
        inputs = self.dropout(self.norm(inputs.transpose(1, 2)).transpose(1, 2))
        encoded, _ = self.gru(inputs)
        _, (hidden, _) = self.lstm(
            encoded, (rnn_hidden_states * masks.view(1, -1, 1),
                      torch.zeros_like(rnn_hidden_states)))
        features = hidden[-1]
        if ext_memory is not None:
            features = features + ext_memory.mean(0, keepdim=True) * ext_memory_masks.mean()
        return features, hidden, features


class _ActionHead(nn.Linear):
    def forward(self, features):
        return torch.distributions.Categorical(logits=super().forward(features))


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _RecurrentNet()
        self.action_distribution = _ActionHead(5, 4)


def _inputs(device):
    return {
        "observations": {"audio": torch.randn(1, 5, 4, device=device)},
        "rnn_hidden_states": torch.zeros(1, 1, 5, device=device),
        "prev_actions": torch.ones(1, 1, dtype=torch.long, device=device),
        "masks": torch.ones(1, 1, device=device),
        "ext_memory": torch.randn(2, 5, device=device),
        "ext_memory_masks": torch.ones(1, 2, device=device),
    }


def _behavior_forward(policy, inputs):
    # Match the trainer's no-grad action pass, independently of the callback.
    with policy_forward_context("atena"), torch.no_grad():
        features, _, _ = policy.net(
            inputs["observations"], inputs["rnn_hidden_states"],
            inputs["prev_actions"], inputs["masks"],
            inputs["ext_memory"], inputs["ext_memory_masks"])
        return features, policy.action_distribution(features).logits


class EnmusAtenaReplayTest(unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(31)
        self.policy = _Policy().to(self.device).eval()

    def test_behavior_and_gradient_replay_preserve_eval_and_restore_backend(self):
        inputs = _inputs(self.device)
        buffers = {name: value.clone() for name, value in self.policy.named_buffers()}
        with torch.backends.cudnn.flags(enabled=True):
            behavior, _ = _behavior_forward(self.policy, inputs)
            self.assertTrue(torch.backends.cudnn.enabled)
            features, logits = forward_policy_for_replay(self.policy, inputs)
            self.assertTrue(torch.backends.cudnn.enabled)
            # Backward deliberately happens after the context has exited.
            gradients = torch.autograd.grad(
                features.sum() + logits.square().sum(), tuple(self.policy.parameters()))
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        torch.testing.assert_close(features.detach(), behavior, rtol=0, atol=1e-5)
        self.assertEqual({entry[1] for entry in self.policy.net.calls}, {False, True})
        self.assertTrue(all(not cudnn and not training
                            for _, _, cudnn, training in self.policy.net.calls))
        self.assertTrue(all(not module.training for module in self.policy.modules()))
        for name, value in self.policy.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)

    def test_callback_restores_backend_after_forward_exception(self):
        self.policy.net.fail_on = "lstm"
        with torch.backends.cudnn.flags(enabled=True):
            with self.assertRaisesRegex(RuntimeError, "injected recurrent forward"):
                forward_policy_for_replay(self.policy, _inputs(self.device))
            self.assertTrue(torch.backends.cudnn.enabled)
        self.assertTrue(all(not module.training for module in self.policy.modules()))

    def test_non_atena_methods_keep_the_existing_backend(self):
        self.policy.net.require_native = False
        for enabled in (False, True):
            for method in ("source", "none", "tent", "fstta", "eam", "feedtta", "idea"):
                with self.subTest(enabled=enabled, method=method):
                    with torch.backends.cudnn.flags(enabled=enabled):
                        with policy_forward_context(method), torch.no_grad():
                            inputs = _inputs(self.device)
                            self.policy.net(inputs["observations"], inputs["rnn_hidden_states"],
                                            inputs["prev_actions"], inputs["masks"])
                        self.assertEqual(torch.backends.cudnn.enabled, enabled)
                        self.assertEqual(self.policy.net.calls[-1][2], enabled)

    def test_unwrapped_forward_is_detected_even_without_cuda(self):
        inputs = _inputs(self.device)
        with torch.backends.cudnn.flags(enabled=True):
            with self.assertRaisesRegex(AssertionError, "still uses cuDNN"):
                self.policy.net(inputs["observations"], inputs["rnn_hidden_states"],
                                inputs["prev_actions"], inputs["masks"])

    def _exercise_episodes(self, query):
        adapter = ATENAAdapter(
            self.policy, lr_query=1e-2, lr_self=7e-3,
            query_threshold=0.0 if query else 100.0,
            weight_decay=0.0, action_selection_protocol="sample_from_policy",
            forward_policy=forward_policy_for_replay)
        buffers = {name: value.clone() for name, value in self.policy.named_buffers()}
        with torch.backends.cudnn.flags(enabled=True):
            for episode in range(2):
                before = {name: parameter.detach().clone()
                          for name, parameter in self.policy.named_parameters()}
                adapter.episode_start()
                for step in range(3):
                    inputs = _inputs(self.device)
                    features, logits = _behavior_forward(self.policy, inputs)
                    actions = adapter.select_action(self.policy.action_distribution(features))
                    adapter.adapt(logits, action=actions, features=features, policy_inputs=inputs)
                    self.assertTrue(torch.backends.cudnn.enabled)
                    if episode == 0 and step == 0:
                        self.assertTrue(adapter.replay_reachability_validated)
                        self.assertIsNone(adapter.optimizer)
                        for name, parameter in self.policy.named_parameters():
                            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
                adapter.episode_end({"success": 1.0} if query else None)
                self.assertTrue(torch.backends.cudnn.enabled)
                for prefix in ("net.gru.", "net.lstm."):
                    self.assertTrue(any(
                        not torch.equal(parameter.detach(), before[name])
                        for name, parameter in self.policy.named_parameters()
                        if name.startswith(prefix)), prefix + " did not adapt")
        diagnostics = adapter.diagnostics()
        self.assertEqual(diagnostics["updates"], 2)
        self.assertEqual(diagnostics["replayed_steps"], 6)
        self.assertEqual(diagnostics["queries"], 2 if query else 0)
        self.assertEqual(diagnostics["feedback_observed_episodes"], 2 if query else 0)
        self.assertEqual(diagnostics["self_label_episodes"], 0 if query else 2)
        self.assertTrue(diagnostics["optimizer_policy_scope_matches_reachable"])
        self.assertEqual(diagnostics["replay_determinism_validated_episodes"], 2)
        self.assertLessEqual(diagnostics["max_replay_feature_abs_error"], 1e-5)
        self.assertTrue(all(not cudnn and not training
                            for _, _, cudnn, training in self.policy.net.calls))
        self.assertTrue(all(not module.training for module in self.policy.modules()))
        for name, value in self.policy.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)

    def test_query_episodes_replay_and_update_both_recurrent_modules(self):
        self._exercise_episodes(query=True)

    def test_self_label_episodes_replay_and_update_without_feedback(self):
        self._exercise_episodes(query=False)


@unittest.skipUnless(torch.cuda.is_available() and torch.backends.cudnn.is_available(),
                     "requires CUDA and cuDNN")
class EnmusAtenaCudaReplayTest(EnmusAtenaReplayTest):
    device = "cuda"

    def test_unwrapped_cuda_eval_rnn_reproduces_original_backward_error(self):
        self.policy.net.require_native = False
        inputs = _inputs(self.device)
        with torch.backends.cudnn.flags(enabled=True):
            features, _, _ = self.policy.net(
                inputs["observations"], inputs["rnn_hidden_states"],
                inputs["prev_actions"], inputs["masks"])
            with self.assertRaisesRegex(RuntimeError, "cudnn RNN backward.*training mode"):
                torch.autograd.grad(features.sum(), tuple(self.policy.parameters()),
                                    allow_unused=True)


class EnmusAtenaTrainerWiringTest(unittest.TestCase):
    def test_trainer_uses_callback_and_matching_behavior_context(self):
        trainer = Path(__file__).resolve().parents[1] / (
            "baselines/enmus/sen_baselines/enmus/ddppo/ddppo_enmus_trainer.py")
        tree = ast.parse(trainer.read_text(encoding="utf-8"))
        builders = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "build_adapter"]
        self.assertEqual(len(builders), 1)
        callback = next((item.value for item in builders[0].keywords
                         if item.arg == "forward_policy"), None)
        self.assertIsNotNone(callback, "trainer must connect the recurrent replay callback")
        expression = compile(ast.Expression(callback), str(trainer), "eval")
        for method in ("atena", "feedtta", "tent", "eam", "idea"):
            selected = eval(expression, {"tta_method": method,
                                        "forward_policy_for_replay": forward_policy_for_replay})
            self.assertIs(selected, forward_policy_for_replay if method == "atena" else None)
        guarded_forwards = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.With):
                continue
            contexts = [item.context_expr for item in node.items]
            if not any(isinstance(context, ast.Call) and isinstance(context.func, ast.Name)
                       and context.func.id == "policy_forward_context"
                       and len(context.args) == 1 and isinstance(context.args[0], ast.Name)
                       and context.args[0].id == "tta_method" for context in contexts):
                continue
            guarded_forwards.extend(call for call in ast.walk(node)
                                    if isinstance(call, ast.Call)
                                    and isinstance(call.func, ast.Attribute)
                                    and call.func.attr == "net")
        self.assertTrue(guarded_forwards, "behavior net forward must share the replay backend")


if __name__ == "__main__":
    unittest.main()
