import random
import unittest

import torch
import torch.nn as nn

from navtta_core.tta.tta_core import (
    ATENAAdapter,
    EAMAdapter,
    FEEDTTAAdapter,
    FSTTAAdapter,
    TentAdapter,
    _concordant_grad_and_trace,
    configure_tta_model,
)


class _TinyNet(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])

    def forward(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        ext_memory=None,
        ext_memory_masks=None,
    ):
        x = observations["x"]
        for norm in self.norms:
            x = torch.tanh(norm(x))
        return x, rnn_hidden_states, x


class _DistributionHead(nn.Module):
    def __init__(self, width, actions):
        super().__init__()
        self.linear = nn.Linear(width, actions)

    def forward(self, features):
        return torch.distributions.Categorical(logits=self.linear(features))


class _TinyPolicy(nn.Module):
    def __init__(self, width=8, actions=4):
        super().__init__()
        self.net = _TinyNet(width)
        self.action_distribution = _DistributionHead(width, actions)
        with torch.no_grad():
            self.action_distribution.linear.bias.copy_(
                torch.tensor([5.0, 0.0, 0.0, 0.0])
            )

    def distribution(self, features):
        return self.action_distribution(features)


def _inputs(width=8):
    return {
        "observations": {"x": torch.randn(1, width)},
        "rnn_hidden_states": torch.zeros(1, 1),
        "prev_actions": torch.zeros(1, 1, dtype=torch.long),
        "masks": torch.ones(1, 1),
        "ext_memory": None,
        "ext_memory_masks": None,
    }


def _forward(policy, inputs):
    features, _, _ = policy.net(
        inputs["observations"],
        inputs["rnn_hidden_states"],
        inputs["prev_actions"],
        inputs["masks"],
        inputs["ext_memory"],
        inputs["ext_memory_masks"],
    )
    distribution = policy.action_distribution(features)
    return features, distribution.logits


class TTACoreTest(unittest.TestCase):
    def setUp(self):
        random.seed(7)
        torch.manual_seed(7)

    def test_layernorm_scope_variants_select_exact_modules(self):
        cases = {
            "first_ln": ["norms.0.weight", "norms.0.bias"],
            "last_ln": ["norms.3.weight", "norms.3.bias"],
            "last_k_ln": [
                "norms.2.weight", "norms.2.bias",
                "norms.3.weight", "norms.3.bias",
            ],
            "ln": [
                "norms.0.weight", "norms.0.bias",
                "norms.1.weight", "norms.1.bias",
                "norms.2.weight", "norms.2.bias",
                "norms.3.weight", "norms.3.bias",
            ],
        }
        for scope, expected_names in cases.items():
            with self.subTest(scope=scope):
                model = _TinyNet()
                _, names = configure_tta_model(
                    model, scope=scope, last_k=2
                )
                trainable_names = [
                    name for name, param in model.named_parameters()
                    if param.requires_grad
                ]
                self.assertEqual(names, expected_names)
                self.assertEqual(trainable_names, expected_names)

    def test_fstta_concordant_gradient_is_finite_and_length_calibrated(self):
        grads = [torch.randn(32) for _ in range(4)]
        concordant, trace = _concordant_grad_and_trace(grads)
        mean_grad = torch.stack(grads).mean(0)
        self.assertTrue(torch.isfinite(concordant).all())
        self.assertTrue(torch.isfinite(trace))
        self.assertAlmostEqual(
            concordant.norm().item(), mean_grad.norm().item(), places=5
        )

    def test_fstta_fast_and_slow_schedules_use_steps_and_episodes(self):
        policy = _TinyPolicy()
        adapter = FSTTAAdapter(
            policy, M=2, N=2, lr_fast=1e-3, lr_slow=1e-3,
            last_k=1, max_grad_norm=10.0,
        )
        for _ in range(2):
            adapter.episode_start()
            for _ in range(2):
                features, logits = _forward(policy, _inputs())
                adapter.adapt(logits)
            adapter.episode_end()
        self.assertEqual(adapter.update_count, 2)
        self.assertEqual(adapter.slow_update_count, 1)

    def test_tent_caps_updates_per_episode_without_resetting_model(self):
        policy = _TinyPolicy()
        adapter = TentAdapter(
            policy,
            lr=1e-3,
            update_interval=1,
            max_updates_per_episode=2,
            last_k=1,
            max_grad_norm=10.0,
        )
        adapter.episode_start()
        for _ in range(5):
            _, logits = _forward(policy, _inputs())
            adapter.adapt(logits)
        self.assertEqual(adapter.update_count, 2)
        self.assertEqual(adapter.episode_update_count, 2)
        self.assertEqual(adapter.skipped_updates_by_budget, 3)

        adapter.episode_end()
        adapter.episode_start()
        _, logits = _forward(policy, _inputs())
        adapter.adapt(logits)
        self.assertEqual(adapter.update_count, 3)
        self.assertEqual(adapter.episode_update_count, 1)
        self.assertEqual(adapter.skipped_updates_by_budget, 3)

    def test_eam_current_only_then_replay_updates_auxiliary_branch_only(self):
        policy = _TinyPolicy()
        source_before = [p.detach().clone() for p in policy.parameters()]
        adapter = EAMAdapter(
            policy, batch_size=2, memory_size=4, lr=1e-2,
            trainable_prefixes=("net.norms", "action_distribution"),
        )
        self.assertEqual(adapter.update_interval, 1)
        self.assertEqual(adapter.param_scope, "module_prefixes")
        self.assertEqual(adapter.max_grad_norm, 0.0)
        aux_before = [p.detach().clone() for p in adapter.params]
        adapter.episode_start()
        for step in range(2):
            inputs = _inputs()
            adapter.before_inference(policy_inputs=inputs)
            with torch.no_grad():
                _, logits = _forward(policy, inputs)
            adapter.prepare_action(logits, policy_inputs=inputs)
            adapter.adapt(logits, action=torch.tensor([[0]]))
            self.assertEqual(adapter.update_count, step + 1)
        adapter.episode_end()
        self.assertEqual(adapter.update_count, 2)
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(aux_before, adapter.params)
        ))
        self.assertTrue(all(
            torch.equal(before, after)
            for before, after in zip(source_before, policy.parameters())
        ))

    def test_eam_applies_paper_confidence_gates(self):
        adapter = EAMAdapter(
            _TinyPolicy(),
            batch_size=1,
            trainable_prefixes=("net.norms", "action_distribution"),
        )
        source_logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
        confident_aux = torch.tensor([[10.0, -10.0, -10.0, -10.0]])
        combined, use_aux = adapter._combine(source_logits, confident_aux)
        expected = source_logits.softmax(dim=-1) + confident_aux.softmax(dim=-1)
        expected = expected / expected.sum(dim=-1, keepdim=True)
        self.assertTrue(bool(use_aux.item()))
        torch.testing.assert_close(combined.exp(), expected)

        uncertain_aux = torch.zeros_like(source_logits)
        source_only, use_aux = adapter._combine(source_logits, uncertain_aux)
        self.assertFalse(bool(use_aux.item()))
        torch.testing.assert_close(source_only.exp(), source_logits.softmax(dim=-1))

    def test_eam_rejects_high_entropy_source_sample(self):
        policy = _TinyPolicy()
        adapter = EAMAdapter(
            policy,
            batch_size=1,
            memory_size=4,
            lr=1e-2,
            trainable_prefixes=("net.norms", "action_distribution"),
        )
        adapter.episode_start()
        inputs = _inputs()
        uncertain_source = torch.zeros(1, 4)
        adapter.before_inference(policy_inputs=inputs)
        adapter.prepare_action(
            uncertain_source,
            policy_inputs=inputs,
        )
        adapter.adapt(uncertain_source, action=torch.tensor([[0]]))
        adapter.episode_end()
        self.assertEqual(adapter.update_count, 0)
        self.assertEqual(adapter.accepted_samples, 0)
        self.assertEqual(len(adapter.replay), 1)

    def test_feedtta_updates_once_from_episode_feedback(self):
        policy = _TinyPolicy()
        adapter = FEEDTTAAdapter(
            policy, lr=1e-2, reversal_probability=0.0,
            trainable_prefixes=("net.norms.3", "action_distribution"),
            max_grad_norm=10.0,
        )
        self.assertEqual(adapter.gamma, 0.99)
        self.assertFalse(adapter.normalize_gradient)
        self.assertEqual(adapter.param_scope, "module_prefixes")
        self.assertTrue(all(
            name.startswith(("net.norms.3.", "action_distribution."))
            for name in adapter.names
        ))
        before = [p.detach().clone() for p in adapter.params]
        adapter.episode_start()
        features, logits = _forward(policy, _inputs())
        action = torch.tensor([[0]])
        adapter.adapt(logits, action=action, features=features)
        adapter.episode_end({"success": 1.0})
        self.assertEqual(adapter.update_count, 1)
        self.assertTrue(any(
            not torch.equal(old, new) for old, new in zip(before, adapter.params)
        ))

    def test_feedtta_trajectory_gradient_matches_discounted_sum(self):
        adapter = FEEDTTAAdapter(
            _TinyPolicy(), gamma=0.5,
            trainable_prefixes=("action_distribution",),
        )
        width = sum(param.numel() for param in adapter.params)
        adapter._accumulate_step_gradient(torch.ones(width))
        adapter._accumulate_step_gradient(torch.full((width,), 2.0))
        expected = torch.full((width,), 2.5)
        torch.testing.assert_close(adapter._aggregate_trajectory(), expected)

    def test_atena_uses_argmax_and_joint_episode_update(self):
        policy = _TinyPolicy()
        adapter = ATENAAdapter(
            policy, lr_query=1e-2, query_threshold=0.0,
            weight_decay=0.0, max_grad_norm=0.0,
        )
        self.assertEqual(adapter.param_scope, "all")
        adapter.episode_start()
        features, logits = _forward(policy, _inputs())
        distribution = policy.action_distribution(features)
        action = adapter.select_action(distribution)
        torch.testing.assert_close(
            action, distribution.probs.argmax(dim=-1, keepdim=True)
        )
        adapter.adapt(logits, action=action, features=features)
        self.assertTrue(adapter.trajectory_mixture_entropies[0].requires_grad)
        adapter.episode_end({"success": 1.0})
        self.assertEqual(adapter.update_count, 1)
        self.assertEqual(adapter.query_count, 1)
        self.assertIsInstance(adapter.self_prediction_head, nn.Sequential)
        self.assertIsNotNone(adapter.optimizer)
        self.assertEqual(adapter.trajectory_mixture_entropies, [])

    def test_atena_self_prediction_loss_updates_policy_representation(self):
        policy = _TinyPolicy()
        adapter = ATENAAdapter(
            policy, lr_query=1e-2, mix_lambda=1.0,
            query_threshold=0.0, self_loss_weight=1.0,
            weight_decay=0.0, max_grad_norm=0.0,
        )
        before = [
            param.detach().clone() for param in policy.net.parameters()
        ]
        adapter.episode_start()
        features, logits = _forward(policy, _inputs())
        action = adapter.select_action(policy.action_distribution(features))
        adapter.adapt(logits, action=action, features=features)
        adapter.episode_end({"success": 1.0})
        self.assertTrue(any(
            not torch.equal(old, new)
            for old, new in zip(before, policy.net.parameters())
        ))


if __name__ == "__main__":
    unittest.main()
