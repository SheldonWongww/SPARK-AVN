from copy import deepcopy
from types import SimpleNamespace
import random
import unittest
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

from navtta_core.tta.tta_core import (
    ATENAAdapter,
    EAMAdapter,
    FEEDTTAAdapter,
    FSTTAAdapter,
    TentAdapter,
    _concordant_grad_and_trace,
    _small_row_svd,
    build_adapter,
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


class _ReplayStateEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fusion_encoder = nn.Linear(width, width)
        self.transformer = nn.Linear(width, width)


class _ReplayNet(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.visual_encoder = nn.Linear(width, width)
        self.smt_state_encoder = _ReplayStateEncoder(width)

    def forward(self, observations, hidden, prev_actions, masks, *unused):
        memory_feature = self.visual_encoder(observations["x"])
        fused = self.smt_state_encoder.fusion_encoder(memory_feature)
        features = self.smt_state_encoder.transformer(fused)
        return features, hidden, memory_feature


class _ReplayPolicy(nn.Module):
    def __init__(self, width=8, actions=4):
        super().__init__()
        self.net = _ReplayNet(width)
        self.action_distribution = _DistributionHead(width, actions)
        self.critic = nn.Linear(width, 1)


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


def _run_fast_update(adapter, policy):
    """Run one complete fast window for adapters configured with ``M=1``."""
    features, logits = _forward(policy, _inputs())
    adapter.adapt(logits)
    return features


def _run_nondegenerate_slow_window(adapter, scale=0.05):
    """Install a deterministic, full-rank-enough trajectory and update SLOW."""
    if adapter.N != 2:
        raise ValueError("The test helper requires an FSTTA slow window of N=2")
    anchor = adapter.slow_anchor.detach().clone()
    first = torch.zeros_like(anchor)
    second = torch.zeros_like(anchor)
    first[0] = scale
    second[1] = 2.0 * scale
    adapter.slow_trajectory = [anchor + first, anchor + second]
    adapter._slow_step()


def _optimizer_step_value(state):
    step = state["step"]
    return float(step.item()) if torch.is_tensor(step) else float(step)


class TTACoreTest(unittest.TestCase):
    def test_fstta_small_row_svd_uses_float32_cpu_lapack(self):
        matrix = torch.randn(4, 12, dtype=torch.float64)
        singular_values, vh = _small_row_svd(matrix)

        self.assertEqual(singular_values.device, matrix.device)
        self.assertEqual(vh.device, matrix.device)
        self.assertEqual(singular_values.dtype, matrix.dtype)
        self.assertEqual(vh.dtype, matrix.dtype)
        reconstructed_gram = vh.t() @ torch.diag(
            singular_values.square()
        ) @ vh
        torch.testing.assert_close(
            reconstructed_gram,
            matrix.t() @ matrix,
            rtol=1e-5,
            atol=1e-5,
        )

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

    def test_tent_nonpositive_clip_limit_disables_clipping(self):
        policy = _TinyPolicy()
        adapter = TentAdapter(
            policy, lr=1e-2, scope="last_ln", max_grad_norm=0.0
        )
        before = [parameter.detach().clone() for parameter in adapter.params]
        _, logits = _forward(policy, _inputs())
        adapter.adapt(logits)
        self.assertTrue(any(
            not torch.equal(old, parameter)
            for old, parameter in zip(before, adapter.params)
        ))

    def test_fstta_concordant_gradient_is_finite_and_length_calibrated(self):
        grads = [torch.randn(32) for _ in range(4)]
        concordant, trace = _concordant_grad_and_trace(grads)
        mean_grad = torch.stack(grads).mean(0)
        self.assertTrue(torch.isfinite(concordant).all())
        self.assertTrue(torch.isfinite(trace))
        self.assertAlmostEqual(
            concordant.norm().item(), mean_grad.norm().item(), places=5
        )

    def test_fstta_fast_gradient_modes_select_expected_values_and_trace(self):
        grads = [
            torch.tensor([1.0, 3.0]),
            torch.tensor([3.0, 7.0]),
            torch.tensor([5.0, 11.0]),
        ]
        expected = {
            "concordant": _concordant_grad_and_trace(grads)[0],
            "mean": torch.tensor([3.0, 7.0]),
            "last": torch.tensor([5.0, 11.0]),
        }
        for mode, expected_grad in expected.items():
            with self.subTest(mode=mode):
                adapter = FSTTAAdapter(
                    _TinyPolicy(), M=3, fast_grad_mode=mode,
                    use_slow=False, last_k=1,
                )
                fast_grad, sigma = adapter._fast_grad_and_trace(grads)
                torch.testing.assert_close(fast_grad, expected_grad)
                torch.testing.assert_close(sigma, torch.tensor(20.0))

    def test_fstta_all_fast_gradient_modes_update_every_m_steps(self):
        for mode in ("concordant", "mean", "last"):
            with self.subTest(mode=mode):
                policy = _TinyPolicy()
                adapter = FSTTAAdapter(
                    policy, M=2, fast_grad_mode=mode, use_slow=False,
                    last_k=1, max_grad_norm=10.0,
                )
                for step in range(1, 6):
                    _, logits = _forward(policy, _inputs())
                    adapter.adapt(logits)
                    self.assertEqual(adapter.update_count, step // 2)
                    self.assertEqual(len(adapter.grad_buffer), step % 2)

    def test_fstta_disabled_fast_lr_scaler_records_unit_scale_and_sigma(self):
        policy = _TinyPolicy()
        adapter = FSTTAAdapter(
            policy, M=2, use_fast_lr_scaler=False, use_slow=False,
            lr_fast=2e-3, last_k=1, max_grad_norm=10.0,
        )
        for _ in range(2):
            _, logits = _forward(policy, _inputs())
            adapter.adapt(logits)

        diagnostics = adapter.diagnostics()
        self.assertGreater(diagnostics["last_sigma"], 0.0)
        self.assertEqual(adapter.lr_scale_count, 1)
        self.assertEqual(diagnostics["lr_scale_mean"], 1.0)
        self.assertEqual(diagnostics["lr_scale_min"], 1.0)
        self.assertEqual(diagnostics["lr_scale_max"], 1.0)
        self.assertEqual(diagnostics["current_lr"], 2e-3)
        self.assertEqual(adapter.optimizer.param_groups[0]["lr"], 2e-3)

    def test_fstta_factory_reads_fast_controls_and_reports_diagnostics(self):
        config = SimpleNamespace(
            METHOD="fstta",
            LR=1e-3,
            LAST_K_LN=1,
            FSTTA=SimpleNamespace(
                FAST_GRAD_MODE="last",
                USE_FAST_LR_SCALER=False,
                USE_SLOW=False,
                SLOW_OPTIMIZER="SGD",
                SLOW_MOMENTUM=0.0,
                RESET_SLOW_OPTIMIZER_EACH_WINDOW=True,
            ),
        )
        adapter = build_adapter(_TinyPolicy(), config)
        diagnostics = adapter.diagnostics()
        self.assertEqual(diagnostics["fast_grad_mode"], "last")
        self.assertFalse(diagnostics["use_fast_lr_scaler"])
        self.assertIsInstance(adapter.slow_optimizer, torch.optim.SGD)
        self.assertEqual(diagnostics["slow_momentum"], 0.0)
        self.assertTrue(diagnostics["reset_slow_optimizer_each_window"])

    def test_fstta_rejects_unknown_fast_gradient_mode(self):
        with self.assertRaisesRegex(ValueError, "FAST_GRAD_MODE"):
            FSTTAAdapter(
                _TinyPolicy(), fast_grad_mode="sum", use_slow=False,
                last_k=1,
            )

    def test_fstta_legacy_positional_arguments_keep_their_meaning(self):
        adapter = FSTTAAdapter(
            _TinyPolicy(),
            1e-3, 2e-3, 1, 2, 0.1, 0.95, 0.7, 0.9, 1.1,
            1, False, True, True, "last_k_ln", 1,
            "SGD", 0.25, 0.9, 0.99, 0.0, 2.0, True, 1e-5,
        )

        self.assertIsInstance(adapter.optimizer, torch.optim.SGD)
        self.assertIsInstance(adapter.slow_optimizer, torch.optim.SGD)
        self.assertEqual(adapter.slow_momentum, 0.25)
        self.assertEqual(adapter.max_grad_norm, 2.0)
        self.assertEqual(adapter.eigen_eps, 1e-5)

    def test_fstta_legacy_config_inherits_fast_optimizer_for_slow(self):
        config = SimpleNamespace(
            METHOD="fstta",
            LR=1e-3,
            MOMENTUM=0.25,
            LAST_K_LN=1,
            FSTTA=SimpleNamespace(
                OPTIMIZER="SGD",
                USE_SLOW=False,
            ),
        )
        adapter = build_adapter(_TinyPolicy(), config)

        self.assertIsInstance(adapter.optimizer, torch.optim.SGD)
        self.assertIsInstance(adapter.slow_optimizer, torch.optim.SGD)
        self.assertEqual(adapter.slow_momentum, 0.25)
        self.assertEqual(adapter.fast_grad_mode, "concordant")
        self.assertTrue(adapter.use_fast_lr_scaler)
        self.assertFalse(adapter.reset_slow_optimizer_each_window)

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

    def test_fstta_fast_and_slow_adamw_optimizers_are_independent(self):
        policy = _TinyPolicy()
        adapter = FSTTAAdapter(
            policy, M=1, N=2, lr_fast=1e-3, lr_slow=3e-3,
            last_k=1, max_grad_norm=10.0,
        )

        self.assertIsInstance(adapter.optimizer, torch.optim.AdamW)
        self.assertIsInstance(adapter.slow_optimizer, torch.optim.AdamW)
        fast_params = {
            id(param)
            for group in adapter.optimizer.param_groups
            for param in group["params"]
        }
        slow_params = {
            id(param)
            for group in adapter.slow_optimizer.param_groups
            for param in group["params"]
        }
        self.assertTrue(fast_params)
        self.assertEqual(slow_params, {id(adapter.slow_anchor)})
        self.assertTrue(fast_params.isdisjoint(slow_params))
        self.assertEqual(adapter.optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(adapter.slow_optimizer.param_groups[0]["lr"], 3e-3)

    def test_fstta_slow_optimizer_supports_plain_sgd(self):
        adapter = FSTTAAdapter(
            _TinyPolicy(), M=1, N=2, lr_fast=1e-3, lr_slow=3e-3,
            last_k=1, max_grad_norm=10.0,
            slow_optimizer_name="SGD", slow_momentum=0.0,
        )

        self.assertIsInstance(adapter.optimizer, torch.optim.AdamW)
        self.assertIsInstance(adapter.slow_optimizer, torch.optim.SGD)
        _run_nondegenerate_slow_window(adapter)
        self.assertAlmostEqual(
            adapter.last_slow_step_norm,
            adapter.lr_slow * adapter.last_slow_grad_norm,
            places=6,
        )
        diagnostics = adapter.diagnostics()
        self.assertEqual(diagnostics["slow_optimizer"], "SGD")
        self.assertEqual(diagnostics["slow_momentum"], 0.0)
        self.assertEqual(diagnostics["slow_optimizer_resets"], 0)

    def test_fstta_can_reset_slow_adamw_moments_each_window(self):
        adapter = FSTTAAdapter(
            _TinyPolicy(), M=1, N=2, lr_fast=1e-3, lr_slow=1e-3,
            last_k=1, max_grad_norm=10.0,
            reset_slow_optimizer_each_window=True,
        )

        _run_nondegenerate_slow_window(adapter)
        first_state = adapter.slow_optimizer.state[adapter.slow_anchor]
        self.assertEqual(_optimizer_step_value(first_state), 1.0)
        _run_nondegenerate_slow_window(adapter, scale=0.025)
        second_state = adapter.slow_optimizer.state[adapter.slow_anchor]
        self.assertEqual(_optimizer_step_value(second_state), 1.0)
        diagnostics = adapter.diagnostics()
        self.assertEqual(diagnostics["slow_updates"], 2)
        self.assertEqual(diagnostics["slow_optimizer_resets"], 2)
        self.assertTrue(diagnostics["reset_slow_optimizer_each_window"])

    def test_fstta_window_reset_clears_moments_for_degenerate_window(self):
        adapter = FSTTAAdapter(
            _TinyPolicy(), M=1, N=2, lr_fast=1e-3, lr_slow=1e-3,
            last_k=1, max_grad_norm=10.0,
            reset_slow_optimizer_each_window=True,
        )
        _run_nondegenerate_slow_window(adapter)
        self.assertTrue(adapter.slow_optimizer.state)

        anchor = adapter.slow_anchor.detach().clone()
        adapter.slow_trajectory = [anchor.clone(), anchor.clone()]
        adapter._slow_step()

        self.assertFalse(adapter.slow_optimizer.state)
        diagnostics = adapter.diagnostics()
        self.assertEqual(diagnostics["slow_attempts"], 2)
        self.assertEqual(diagnostics["slow_updates"], 1)
        self.assertEqual(diagnostics["slow_skipped_updates"], 1)
        self.assertEqual(diagnostics["slow_optimizer_resets"], 2)

    def test_fstta_slow_adamw_moments_persist_across_windows(self):
        adapter = FSTTAAdapter(
            _TinyPolicy(), M=1, N=2, lr_fast=1e-3, lr_slow=1e-3,
            last_k=1, max_grad_norm=10.0,
        )

        anchor_before = adapter.slow_anchor.detach().clone()
        _run_nondegenerate_slow_window(adapter)
        first_state = adapter.slow_optimizer.state[adapter.slow_anchor]
        self.assertEqual(_optimizer_step_value(first_state), 1.0)
        self.assertEqual(first_state["exp_avg"].shape, adapter.slow_anchor.shape)
        self.assertEqual(
            first_state["exp_avg_sq"].shape, adapter.slow_anchor.shape
        )
        self.assertGreater(float(first_state["exp_avg"].norm().item()), 0.0)
        self.assertGreater(float(first_state["exp_avg_sq"].norm().item()), 0.0)

        # Reconstruct the first bias-corrected AdamW step. This catches the
        # previous implementation, which applied LR_SLOW as plain SGD.
        group = adapter.slow_optimizer.param_groups[0]
        beta1, beta2 = group["betas"]
        exp_avg = first_state["exp_avg"]
        exp_avg_sq = first_state["exp_avg_sq"]
        expected = anchor_before - group["lr"] * (
            exp_avg / (1.0 - beta1)
        ) / (
            (exp_avg_sq / (1.0 - beta2)).sqrt() + group["eps"]
        )
        torch.testing.assert_close(adapter.slow_anchor.detach(), expected)
        reconstructed_grad = exp_avg / (1.0 - beta1)
        plain_sgd = anchor_before - group["lr"] * reconstructed_grad
        self.assertFalse(torch.allclose(adapter.slow_anchor.detach(), plain_sgd))

        first_exp_avg = first_state["exp_avg"].detach().clone()
        _run_nondegenerate_slow_window(adapter, scale=0.025)
        second_state = adapter.slow_optimizer.state[adapter.slow_anchor]
        self.assertEqual(_optimizer_step_value(second_state), 2.0)
        self.assertFalse(torch.equal(second_state["exp_avg"], first_exp_avg))
        self.assertEqual(adapter.slow_update_count, 2)
        self.assertEqual(adapter.slow_attempt_count, 2)
        self.assertEqual(adapter.slow_skip_count, 0)

    def test_fstta_episode_start_clears_only_fast_adamw_moments(self):
        policy = _TinyPolicy()
        adapter = FSTTAAdapter(
            policy, M=1, N=2, lr_fast=1e-3, lr_slow=1e-3,
            last_k=1, max_grad_norm=10.0,
            reset_optimizer_each_episode=True,
        )
        _run_nondegenerate_slow_window(adapter)
        _run_fast_update(adapter, policy)
        self.assertTrue(adapter.optimizer.state)
        self.assertTrue(adapter.slow_optimizer.state)

        slow_state = adapter.slow_optimizer.state[adapter.slow_anchor]
        slow_step = _optimizer_step_value(slow_state)
        slow_exp_avg = slow_state["exp_avg"].detach().clone()
        slow_exp_avg_sq = slow_state["exp_avg_sq"].detach().clone()
        slow_anchor = adapter.slow_anchor.detach().clone()
        pending = slow_anchor + 0.01
        adapter.slow_trajectory = [pending]
        adapter.var_hist = torch.tensor(2.0)
        adapter.episode_start()

        self.assertEqual(len(adapter.optimizer.state), 0)
        persistent_state = adapter.slow_optimizer.state[adapter.slow_anchor]
        self.assertEqual(_optimizer_step_value(persistent_state), slow_step)
        torch.testing.assert_close(persistent_state["exp_avg"], slow_exp_avg)
        torch.testing.assert_close(
            persistent_state["exp_avg_sq"], slow_exp_avg_sq
        )
        torch.testing.assert_close(adapter.slow_anchor.detach(), slow_anchor)
        torch.testing.assert_close(adapter.slow_trajectory[0], pending)
        self.assertIsNone(adapter.var_hist)

    def test_fstta_reset_restores_anchor_in_place_and_clears_all_state(self):
        policy = _TinyPolicy()
        adapter = FSTTAAdapter(
            policy, M=1, N=2, lr_fast=1e-3, lr_slow=1e-3,
            last_k=1, max_grad_norm=10.0,
        )
        anchor_object = adapter.slow_anchor
        _run_nondegenerate_slow_window(adapter)
        _run_fast_update(adapter, policy)
        adapter.episode_count = 3
        self.assertTrue(adapter.optimizer.state)
        self.assertTrue(adapter.slow_optimizer.state)
        self.assertGreater(adapter.update_count, 0)
        self.assertGreater(adapter.slow_update_count, 0)

        adapter.reset()

        self.assertIs(adapter.slow_anchor, anchor_object)
        self.assertIs(
            adapter.slow_optimizer.param_groups[0]["params"][0],
            anchor_object,
        )
        torch.testing.assert_close(
            adapter.slow_anchor.detach(), adapter._source_flat
        )
        current = torch.cat([
            param.detach().reshape(-1) for param in adapter.params
        ])
        torch.testing.assert_close(current, adapter._source_flat)
        self.assertEqual(len(adapter.optimizer.state), 0)
        self.assertEqual(len(adapter.slow_optimizer.state), 0)
        self.assertEqual(adapter.grad_buffer, [])
        self.assertEqual(adapter.slow_trajectory, [])
        self.assertIsNone(adapter.var_hist)
        self.assertEqual(adapter.action_steps, 0)
        self.assertEqual(adapter.update_count, 0)
        self.assertEqual(adapter.episode_count, 0)
        self.assertEqual(adapter.slow_update_count, 0)
        self.assertEqual(adapter.slow_attempt_count, 0)
        self.assertEqual(adapter.slow_skip_count, 0)
        self.assertEqual(adapter.discarded_fast_gradients, 0)
        self.assertEqual(
            adapter.diagnostics()["relative_slow_anchor_drift"], 0.0
        )

    def test_fstta_degenerate_slow_window_is_skipped_and_cleared(self):
        adapter = FSTTAAdapter(
            _TinyPolicy(), M=1, N=2, lr_fast=1e-3, lr_slow=1e-3,
            last_k=1, max_grad_norm=10.0,
        )
        source_anchor = adapter.slow_anchor.detach().clone()

        # No action-level updates make both episode snapshots identical to the
        # slow anchor, so PDA has neither variance nor a reference direction.
        for _ in range(2):
            adapter.episode_start()
            adapter.episode_end()

        diagnostics = adapter.diagnostics()
        self.assertEqual(diagnostics["episodes"], 2)
        self.assertEqual(diagnostics["slow_attempts"], 1)
        self.assertEqual(diagnostics["slow_skipped_updates"], 1)
        self.assertEqual(diagnostics["slow_updates"], 0)
        self.assertEqual(diagnostics["slow_pending_episodes"], 0)
        self.assertEqual(len(adapter.slow_optimizer.state), 0)
        torch.testing.assert_close(adapter.slow_anchor.detach(), source_anchor)

    def test_fstta_rejects_episodic_slow_configuration(self):
        with self.assertRaisesRegex(ValueError, "EPISODIC=False"):
            FSTTAAdapter(
                _TinyPolicy(), episodic=True, use_slow=True, last_k=1
            )

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
        self.assertTrue(all(not name.startswith("critic.") for name in adapter.names))
        aux_before = [p.detach().clone() for p in adapter.params]
        adapter.episode_start()
        for step in range(2):
            inputs = _inputs()
            adapter.before_inference(policy_inputs=inputs)
            with torch.no_grad():
                _, logits = _forward(policy, inputs)
            adapter.prepare_action(logits, policy_inputs=inputs)
            adapter.adapt(logits, action=torch.tensor([[0]]))
            # Strict Algorithm 3: when m < K, B=x still reaches Model Update.
            self.assertEqual(adapter.update_count, step + 1)
        adapter.episode_end()
        self.assertEqual(adapter.update_count, 2)
        self.assertEqual(adapter.seen_samples, 2)
        self.assertEqual(len(adapter.replay), 2)
        self.assertEqual(adapter.update_attempt_count, 2)
        self.assertEqual(adapter.current_only_batches, 1)
        self.assertEqual(adapter.replayed_step_count, 3)
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

    def test_eam_identical_initial_branches_preserve_source_distribution(self):
        policy = _TinyPolicy()
        adapter = EAMAdapter(
            policy,
            batch_size=2,
            trainable_prefixes=("net.norms", "action_distribution"),
        )
        inputs = _inputs()
        adapter.before_inference(policy_inputs=inputs)
        with torch.no_grad():
            _, source_logits = _forward(policy, inputs)
        combined = adapter.prepare_action(
            source_logits,
            policy_inputs=inputs,
        )
        torch.testing.assert_close(
            combined.exp(), source_logits.softmax(dim=-1)
        )
        adapter.adapt(source_logits, action=torch.tensor([[0]]))

    def test_eam_accepts_task_specific_forward_policy(self):
        policy = _TinyPolicy()
        calls = []

        def forward_policy(model, inputs):
            calls.append(model)
            return _forward(model, inputs)

        adapter = EAMAdapter(
            policy,
            batch_size=1,
            trainable_prefixes=("net.norms", "action_distribution"),
            forward_policy=forward_policy,
        )
        inputs = _inputs()
        adapter.before_inference(policy_inputs=inputs)
        with torch.no_grad():
            _, source_logits = _forward(policy, inputs)
        adapter.prepare_action(source_logits, policy_inputs=inputs)
        adapter.adapt(source_logits, action=torch.tensor([[0]]))
        self.assertTrue(calls)

    def test_eam_replay_supports_variable_action_dimensions(self):
        policy = _TinyPolicy(actions=4)

        def forward_policy(model, inputs):
            features, logits = _forward(model, inputs)
            return features, logits[:, :int(inputs["action_count"])]

        adapter = EAMAdapter(
            policy,
            batch_size=2,
            memory_size=4,
            confidence_scale=1.0,
            trainable_prefixes=("net.norms", "action_distribution"),
            forward_policy=forward_policy,
        )
        for action_count in (3, 4):
            inputs = _inputs()
            inputs["action_count"] = action_count
            adapter.before_inference(policy_inputs=inputs)
            with torch.no_grad():
                _, logits = forward_policy(policy, inputs)
            prepared = adapter.prepare_action(logits, policy_inputs=inputs)
            self.assertEqual(prepared.shape[-1], action_count)
            adapter.adapt(logits, action=torch.tensor([[0]]))
        self.assertEqual(adapter.replayed_step_count, 3)

    def test_eam_rejects_high_entropy_source_sample(self):
        policy = _TinyPolicy()
        adapter = EAMAdapter(
            policy, batch_size=1, memory_size=4, lr=1e-2,
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
        self.assertEqual(adapter.update_attempt_count, 1)
        self.assertEqual(adapter.no_reliable_update_count, 1)

    def test_eam_explicit_scope_freezes_pretransformer_and_critic(self):
        policy = _ReplayPolicy()
        with torch.no_grad():
            policy.action_distribution.linear.bias.copy_(
                torch.tensor([5.0, 0.0, 0.0, 0.0])
            )
        adapter = EAMAdapter(
            policy,
            batch_size=1,
            trainable_prefixes=(
                "net.smt_state_encoder.transformer",
                "action_distribution",
            ),
        )
        expected = {
            "net.smt_state_encoder.transformer.weight",
            "net.smt_state_encoder.transformer.bias",
            "action_distribution.linear.weight",
            "action_distribution.linear.bias",
        }
        self.assertEqual(set(adapter.names), expected)
        trainable = {
            name for name, parameter in adapter.aux_model.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(trainable, expected)
        self.assertFalse(adapter.aux_model.net.visual_encoder.weight.requires_grad)
        self.assertFalse(
            adapter.aux_model.net.smt_state_encoder.fusion_encoder.weight.requires_grad
        )
        self.assertFalse(adapter.aux_model.critic.weight.requires_grad)

        visual_before = adapter.aux_model.net.visual_encoder.weight.detach().clone()
        fusion_before = (
            adapter.aux_model.net.smt_state_encoder.fusion_encoder.weight
            .detach().clone()
        )
        inputs = _inputs()
        adapter.before_inference(policy_inputs=inputs)
        with torch.no_grad():
            _, logits = _forward(policy, inputs)
        adapter.prepare_action(logits, policy_inputs=inputs)
        action = torch.tensor([[2]])
        adapter.adapt(logits, action=action)
        self.assertEqual(adapter.update_count, 1)
        torch.testing.assert_close(
            adapter.aux_model.net.visual_encoder.weight, visual_before
        )
        torch.testing.assert_close(
            adapter.aux_model.net.smt_state_encoder.fusion_encoder.weight,
            fusion_before,
        )
        self.assertIsNone(adapter.aux_model.net.visual_encoder.weight.grad)
        self.assertIsNone(
            adapter.aux_model.net.smt_state_encoder.fusion_encoder.weight.grad
        )

    def test_eam_stores_each_action_as_an_independent_replay_sample(self):
        policy = _TinyPolicy()
        adapter = EAMAdapter(
            policy,
            batch_size=4,
            memory_size=4,
            trainable_prefixes=("net.norms", "action_distribution"),
        )
        adapter.episode_start()
        expected_actions = []
        for step in range(3):
            inputs = _inputs()
            adapter.before_inference(policy_inputs=inputs)
            with torch.no_grad():
                _, logits = _forward(policy, inputs)
            adapter.prepare_action(
                logits,
                policy_inputs=inputs,
            )
            action = torch.tensor([[step % 4]])
            expected_actions.append(action)
            adapter.adapt(logits, action=action)
        adapter.episode_end()

        self.assertEqual(adapter.action_steps, 3)
        self.assertEqual(adapter.seen_samples, 3)
        self.assertEqual(len(adapter.replay), 3)
        self.assertTrue(all(
            "policy_inputs" in entry and "steps" not in entry
            for entry in adapter.replay
        ))
        for entry, action in zip(adapter.replay, expected_actions):
            torch.testing.assert_close(entry["action"], action)
        self.assertEqual(adapter.update_count, 3)
        self.assertEqual(adapter.current_only_batches, 3)

    def test_eam_update_interval_counts_action_steps(self):
        policy = _TinyPolicy()
        adapter = EAMAdapter(
            policy, batch_size=1, memory_size=2, update_interval=2,
            trainable_prefixes=("net.norms", "action_distribution"),
        )
        adapter.episode_start()
        for step in range(2):
            inputs = _inputs()
            adapter.before_inference(policy_inputs=inputs)
            with torch.no_grad():
                _, logits = _forward(policy, inputs)
            adapter.prepare_action(
                logits,
                policy_inputs=inputs,
            )
            adapter.adapt(logits, action=torch.tensor([[0]]))
            self.assertEqual(adapter.update_count, step)
        adapter.episode_end()

    def test_eam_rejects_invalid_or_episodic_replay_configs(self):
        with self.assertRaisesRegex(ValueError, "MEMORY_SIZE"):
            EAMAdapter(
                _TinyPolicy(), memory_size=0,
                trainable_prefixes=("action_distribution",),
            )
        with self.assertRaisesRegex(ValueError, "BATCH_SIZE"):
            EAMAdapter(
                _TinyPolicy(), batch_size=0,
                trainable_prefixes=("action_distribution",),
            )
        with self.assertRaisesRegex(ValueError, "EPISODIC=False"):
            EAMAdapter(
                _TinyPolicy(),
                episodic=True,
                trainable_prefixes=("action_distribution",),
            )
        with self.assertRaisesRegex(ValueError, "module_prefixes"):
            EAMAdapter(_TinyPolicy(), param_scope="all")

    def test_eam_requires_buffer_hook_before_source_forward(self):
        policy = _TinyPolicy()
        adapter = EAMAdapter(
            policy,
            trainable_prefixes=("net.norms", "action_distribution"),
        )
        inputs = _inputs()
        with torch.no_grad():
            _, logits = _forward(policy, inputs)
        with self.assertRaisesRegex(RuntimeError, "before_inference"):
            adapter.prepare_action(logits, policy_inputs=inputs)
        self.assertEqual(adapter.seen_samples, 0)
        self.assertEqual(adapter.replay, [])

    def test_eam_snapshots_the_exact_step_state_on_cpu(self):
        policy = _TinyPolicy()
        adapter = EAMAdapter(
            policy,
            batch_size=2,
            trainable_prefixes=("net.norms", "action_distribution"),
        )
        inputs = _inputs()
        inputs["ext_memory"] = torch.randn(3, 1, 8)
        inputs["ext_memory_masks"] = torch.tensor([[1.0, 1.0, 0.0]])
        expected_x = inputs["observations"]["x"].clone()
        expected_memory = inputs["ext_memory"].clone()
        expected_masks = inputs["ext_memory_masks"].clone()
        adapter.before_inference(policy_inputs=inputs)
        with torch.no_grad():
            _, logits = _forward(policy, inputs)
        adapter.prepare_action(logits, policy_inputs=inputs)
        action = torch.tensor([[2]])
        adapter.adapt(logits, action=action)

        inputs["observations"]["x"].zero_()
        inputs["ext_memory"].zero_()
        inputs["ext_memory_masks"].zero_()
        stored = adapter.replay[0]["policy_inputs"]
        torch.testing.assert_close(stored["observations"]["x"], expected_x)
        torch.testing.assert_close(stored["ext_memory"], expected_memory)
        torch.testing.assert_close(stored["ext_memory_masks"], expected_masks)
        torch.testing.assert_close(adapter.replay[0]["action"], action)
        self.assertNotIn("action", stored)
        self.assertEqual(stored["ext_memory"].device.type, "cpu")

    def test_eam_replay_draw_uses_updated_reservoir_without_excluding_current(self):
        policy = _TinyPolicy()
        adapter = EAMAdapter(
            policy,
            batch_size=2,
            memory_size=3,
            trainable_prefixes=("net.norms", "action_distribution"),
        )

        old_inputs = _inputs()
        old_inputs["sample_id"] = "old"
        adapter.before_inference(policy_inputs=old_inputs)
        with torch.no_grad():
            _, old_logits = _forward(policy, old_inputs)
        adapter.prepare_action(old_logits, policy_inputs=old_inputs)
        adapter.adapt(old_logits, action=torch.tensor([[0]]))

        current_inputs = _inputs()
        current_inputs["sample_id"] = "current"

        def choose_current(population, count):
            self.assertEqual(count, 1)
            current = next(
                entry for entry in population
                if entry["policy_inputs"]["sample_id"] == "current"
            )
            return [current]

        with mock.patch(
            "navtta_core.tta.tta_core.random.sample",
            side_effect=choose_current,
        ) as sample:
            adapter.before_inference(policy_inputs=current_inputs)
            with torch.no_grad():
                _, current_logits = _forward(policy, current_inputs)
            adapter.prepare_action(current_logits, policy_inputs=current_inputs)

        sample.assert_called_once()
        cached_source = adapter._cached_current["source_batch"]
        self.assertEqual(len(cached_source), 2)
        torch.testing.assert_close(cached_source[0], cached_source[1])
        self.assertEqual(adapter.current_replay_duplicates, 1)
        adapter.adapt(current_logits, action=torch.tensor([[1]]))

    def test_eam_recomputes_both_branches_for_historical_replay(self):
        policy = _TinyPolicy()
        adapter = EAMAdapter(
            policy,
            batch_size=2,
            memory_size=3,
            trainable_prefixes=("net.norms", "action_distribution"),
        )

        old_inputs = _inputs()
        old_inputs["sample_id"] = "old"
        adapter.before_inference(policy_inputs=old_inputs)
        with torch.no_grad():
            _, old_logits = _forward(policy, old_inputs)
        adapter.prepare_action(old_logits, policy_inputs=old_inputs)
        adapter.adapt(old_logits, action=torch.tensor([[0]]))

        current_inputs = _inputs()
        current_inputs["sample_id"] = "current"

        def choose_old(population, count):
            self.assertEqual(count, 1)
            old = next(
                entry for entry in population
                if entry["policy_inputs"]["sample_id"] == "old"
            )
            return [old]

        with mock.patch(
            "navtta_core.tta.tta_core.random.sample",
            side_effect=choose_old,
        ), mock.patch.object(
            adapter, "_forward_source", wraps=adapter._forward_source
        ) as source_forward, mock.patch.object(
            adapter, "_forward_aux", wraps=adapter._forward_aux
        ) as aux_forward:
            adapter.before_inference(policy_inputs=current_inputs)
            with torch.no_grad():
                _, current_logits = _forward(policy, current_inputs)
            adapter.prepare_action(current_logits, policy_inputs=current_inputs)

        source_forward.assert_called_once()
        self.assertEqual(aux_forward.call_count, 2)
        adapter.adapt(current_logits, action=torch.tensor([[1]]))

    def test_eam_reservoir_replacement_and_rejection_follow_algorithm_1(self):
        adapter = EAMAdapter(
            _TinyPolicy(),
            memory_size=2,
            batch_size=3,
            trainable_prefixes=("net.norms", "action_distribution"),
        )
        first = adapter._reservoir_add(_inputs())
        second = adapter._reservoir_add(_inputs())

        with mock.patch(
            "navtta_core.tta.tta_core.random.randrange", return_value=0
        ) as randrange:
            replacement = adapter._reservoir_add(_inputs())
        randrange.assert_called_once_with(3)
        self.assertIs(adapter.replay[0], replacement)
        self.assertIs(adapter.replay[1], second)

        retained = tuple(adapter.replay)
        with mock.patch(
            "navtta_core.tta.tta_core.random.randrange", return_value=2
        ) as randrange:
            rejected = adapter._reservoir_add(_inputs())
        randrange.assert_called_once_with(4)
        self.assertIsNone(rejected)
        self.assertIs(adapter.replay[0], retained[0])
        self.assertIs(adapter.replay[1], retained[1])
        self.assertIsNot(first, adapter.replay[0])

    def test_eam_algorithm_3_orders_buffer_inference_action_then_update(self):
        policy = _TinyPolicy()
        adapter = EAMAdapter(
            policy,
            batch_size=1,
            trainable_prefixes=("net.norms", "action_distribution"),
        )
        inputs = _inputs()

        events = []
        reservoir_add = adapter._reservoir_add
        forward_aux = adapter._forward_aux
        source_forward = policy.net.forward
        select_action = adapter.select_action
        optimizer_step = adapter.optimizer.step

        def record_buffer(*args, **kwargs):
            events.append("buffer")
            return reservoir_add(*args, **kwargs)

        def record_inference(*args, **kwargs):
            events.append("auxiliary_inference")
            return forward_aux(*args, **kwargs)

        def record_source_inference(*args, **kwargs):
            events.append("source_inference")
            return source_forward(*args, **kwargs)

        def record_action(*args, **kwargs):
            events.append("action")
            return select_action(*args, **kwargs)

        def record_update(*args, **kwargs):
            events.append("update")
            return optimizer_step(*args, **kwargs)

        with mock.patch.object(
            adapter, "_reservoir_add", side_effect=record_buffer
        ), mock.patch.object(
            adapter, "_forward_aux", side_effect=record_inference
        ), mock.patch.object(
            policy.net, "forward", side_effect=record_source_inference
        ), mock.patch.object(
            adapter, "select_action", side_effect=record_action
        ), mock.patch.object(
            adapter.optimizer, "step", side_effect=record_update
        ):
            adapter.before_inference(policy_inputs=inputs)
            with torch.no_grad():
                _, logits = _forward(policy, inputs)
            action_logits = adapter.prepare_action(
                logits, policy_inputs=inputs
            )
            action = adapter.select_action(
                torch.distributions.Categorical(logits=action_logits)
            )
            adapter.adapt(logits, action=action)

        self.assertEqual(events, [
            "buffer",
            "source_inference",
            "auxiliary_inference",
            "action",
            "update",
        ])

    def test_eam_factory_reads_step_replay_scope(self):
        config = SimpleNamespace(
            METHOD="eam",
            EPISODIC=False,
            EAM=SimpleNamespace(
                MEMORY_SIZE=32,
                BATCH_SIZE=8,
                PARAM_SCOPE="module_prefixes",
                TRAINABLE_PREFIXES=[
                    "net.smt_state_encoder.transformer",
                    "action_distribution",
                ],
            ),
        )
        adapter = build_adapter(_ReplayPolicy(), config)
        self.assertEqual(adapter.param_scope, "module_prefixes")
        self.assertEqual(adapter.memory_size, 32)
        self.assertEqual(adapter.batch_size, 8)
        diagnostics = adapter.diagnostics()
        self.assertEqual(diagnostics["replay_unit"], "action_step")
        self.assertEqual(
            diagnostics["replay_storage"],
            "full_policy_input_action_and_decision_snapshot",
        )
        self.assertEqual(
            diagnostics["replay_sampling"],
            "updated_reservoir_including_current",
        )
        self.assertEqual(
            diagnostics["short_buffer_behavior"],
            "current_only_update",
        )
        self.assertEqual(diagnostics["update_interval_unit"], "action_step")

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

    def test_feedtta_trajectory_accumulator_has_constant_memory(self):
        adapter = FEEDTTAAdapter(
            _TinyPolicy(), gamma=0.99,
            trainable_prefixes=("action_distribution",),
        )
        width = sum(param.numel() for param in adapter.params)
        for _ in range(500):
            adapter._accumulate_step_gradient(torch.ones(width))
        self.assertEqual(adapter._trajectory_step_count, 500)
        self.assertEqual(adapter._trajectory_gradient.numel(), width)
        self.assertFalse(hasattr(adapter, "trajectory_grads"))

    def test_feedtta_sgr_matches_main_text_equation_five(self):
        adapter = FEEDTTAAdapter(
            _TinyPolicy(), reversal_probability=0.5, reversal_scale=-0.2,
            trainable_prefixes=("action_distribution",),
        )
        gradient = torch.tensor([1.0, 2.0, 3.0, 4.0])
        selected = torch.tensor([True, False, True, False])
        with mock.patch.object(
            adapter, "_sample_sgr_mask", return_value=selected
        ):
            actual = adapter._apply_sgr(gradient)
        torch.testing.assert_close(
            actual, torch.tensor([-0.2, 5.0, -0.6, 10.0])
        )
        self.assertEqual(adapter.regularizer_variant, "stochastic_gradient_reversion")
        self.assertEqual(adapter.last_sgr_selected_dimensions, 2)
        self.assertEqual(adapter.last_sgr_selected_fraction, 0.5)

    def test_feedtta_sgr_does_not_advance_global_torch_rng(self):
        adapter = FEEDTTAAdapter(
            _TinyPolicy(), reversal_probability=0.5, reversal_scale=-0.2,
            sgr_seed=17,
            trainable_prefixes=("action_distribution",),
        )
        gradient = torch.ones(32)
        torch.manual_seed(123)
        expected = torch.rand(8)
        torch.manual_seed(123)
        adapter._apply_sgr(gradient)
        actual = torch.rand(8)
        torch.testing.assert_close(actual, expected)

    def test_feedtta_success_and_failure_updates_have_opposite_signs(self):
        source = _TinyPolicy()
        success_policy = deepcopy(source)
        failure_policy = deepcopy(source)
        inputs = _inputs()

        success_adapter = FEEDTTAAdapter(
            success_policy, lr=1e-2, reversal_probability=0.0, gamma=1.0,
            trainable_prefixes=("action_distribution",),
            optimizer_name="SGD", momentum=0.0,
        )
        failure_adapter = FEEDTTAAdapter(
            failure_policy, lr=1e-2, reversal_probability=0.0, gamma=1.0,
            trainable_prefixes=("action_distribution",),
            optimizer_name="SGD", momentum=0.0,
        )
        success_before = [param.detach().clone() for param in success_adapter.params]
        failure_before = [param.detach().clone() for param in failure_adapter.params]

        _, success_logits = _forward(success_policy, inputs)
        success_adapter.episode_start()
        success_adapter.adapt(success_logits, action=torch.tensor([[0]]))
        success_adapter.episode_end({"success": 1.0})

        _, failure_logits = _forward(failure_policy, inputs)
        failure_adapter.episode_start()
        failure_adapter.adapt(failure_logits, action=torch.tensor([[0]]))
        failure_adapter.episode_end({"success": 0.0})

        for before_success, after_success, before_failure, after_failure in zip(
            success_before,
            success_adapter.params,
            failure_before,
            failure_adapter.params,
        ):
            success_delta = after_success.detach() - before_success
            failure_delta = after_failure.detach() - before_failure
            torch.testing.assert_close(success_delta, -failure_delta)

    def test_feedtta_rejects_behaviorally_inert_episodic_mode(self):
        with self.assertRaisesRegex(ValueError, "EPISODIC=True"):
            FEEDTTAAdapter(
                _TinyPolicy(), episodic=True,
                trainable_prefixes=("action_distribution",),
            )

    def test_atena_uses_argmax_and_joint_episode_update(self):
        policy = _TinyPolicy()
        adapter = ATENAAdapter(
            policy, lr_query=1e-2, query_threshold=0.0,
            weight_decay=0.0, max_grad_norm=0.0,
        )
        self.assertEqual(adapter.param_scope, "all")
        adapter.episode_start()
        inputs = _inputs()
        features, logits = _forward(policy, inputs)
        distribution = policy.action_distribution(features)
        action = adapter.select_action(distribution)
        torch.testing.assert_close(
            action, distribution.probs.argmax(dim=-1, keepdim=True)
        )
        adapter.adapt(
            logits, action=action, features=features, policy_inputs=inputs
        )
        self.assertIsInstance(adapter.trajectory_mixture_entropies[0], float)
        adapter.episode_end({"success": 1.0})
        self.assertEqual(adapter.update_count, 1)
        self.assertEqual(adapter.query_count, 1)
        self.assertIsInstance(adapter.self_prediction_head, nn.Sequential)
        self.assertIsNotNone(adapter.optimizer)
        self.assertEqual(adapter.trajectory_mixture_entropies, [])
        self.assertFalse(adapter.diagnostics()["retains_episode_graph"])
        self.assertEqual(
            adapter.diagnostics()["gradient_reconstruction"],
            "exact_step_replay_in_eval_mode",
        )

    def test_atena_accepts_lambda_one_and_task_forward_policy(self):
        policy = _TinyPolicy()
        calls = []

        def forward_policy(model, inputs):
            calls.append(model)
            return _forward(model, inputs)

        adapter = ATENAAdapter(
            policy,
            lr_query=1e-2,
            mix_lambda=1.0,
            query_threshold=0.0,
            weight_decay=0.0,
            max_grad_norm=0.0,
            forward_policy=forward_policy,
        )
        adapter.episode_start()
        inputs = _inputs()
        features, logits = _forward(policy, inputs)
        action = adapter.select_action(policy.action_distribution(features))
        adapter.adapt(
            logits, action=action, features=features, policy_inputs=inputs
        )
        adapter.episode_end({"success": 1.0})
        self.assertTrue(calls)

    def test_atena_self_prediction_loss_updates_policy_representation(self):
        policy = _TinyPolicy()
        adapter = ATENAAdapter(
            policy, lr_query=1e-2, mix_lambda=0.75,
            query_threshold=0.0, self_loss_weight=1.0,
            weight_decay=0.0, max_grad_norm=0.0,
        )
        before = [
            param.detach().clone() for param in policy.net.parameters()
        ]
        adapter.episode_start()
        inputs = _inputs()
        features, logits = _forward(policy, inputs)
        action = adapter.select_action(policy.action_distribution(features))
        adapter.adapt(
            logits, action=action, features=features, policy_inputs=inputs
        )
        adapter.episode_end({"success": 1.0})
        self.assertTrue(any(
            not torch.equal(old, new)
            for old, new in zip(before, policy.net.parameters())
        ))

    def test_atena_nonquery_episode_does_not_read_feedback(self):
        policy = _TinyPolicy()
        adapter = ATENAAdapter(
            policy,
            lr_query=1e-2,
            lr_self=1e-3,
            # Categorical entropy cannot reach this threshold, so the episode
            # must remain self-labelled and require no success oracle.
            query_threshold=100.0,
            weight_decay=0.0,
            max_grad_norm=0.0,
        )
        adapter.episode_start()
        inputs = _inputs()
        features, logits = _forward(policy, inputs)
        action = adapter.select_action(policy.action_distribution(features))
        adapter.adapt(
            logits, action=action, features=features, policy_inputs=inputs
        )
        adapter.episode_end(None)
        diagnostics = adapter.diagnostics()
        self.assertEqual(diagnostics["queries"], 0)
        self.assertEqual(diagnostics["feedback_observed_episodes"], 0)
        self.assertEqual(diagnostics["self_label_episodes"], 1)

    def test_atena_rejects_behaviorally_inert_episodic_mode(self):
        with self.assertRaisesRegex(ValueError, "EPISODIC=False"):
            ATENAAdapter(_TinyPolicy(), episodic=True)

    def test_atena_all_scope_excludes_value_only_critic(self):
        policy = _ReplayPolicy()
        adapter = ATENAAdapter(policy)
        self.assertTrue(adapter.names)
        self.assertFalse(any(name.startswith("critic.") for name in adapter.names))

    def test_atena_step_replay_matches_joint_episode_gradient(self):
        policy = _TinyPolicy()
        learning_rate = 1e-3
        adapter = ATENAAdapter(
            policy,
            lr_query=learning_rate,
            lr_self=learning_rate,
            mix_lambda=0.5,
            query_threshold=0.0,
            self_loss_weight=0.3,
            optimizer_name="SGD",
            momentum=0.0,
            weight_decay=0.0,
            max_grad_norm=0.0,
        )
        adapter.episode_start()
        inputs = [_inputs(), _inputs()]
        actions = []
        for policy_inputs in inputs:
            features, logits = _forward(policy, policy_inputs)
            action = adapter.select_action(policy.action_distribution(features))
            actions.append(action)
            adapter.adapt(
                logits,
                action=action,
                features=features,
                policy_inputs=policy_inputs,
            )

        joint_features = []
        joint_entropies = []
        for policy_inputs, action in zip(inputs, actions):
            features, logits = _forward(policy, policy_inputs)
            joint_features.append(features)
            joint_entropies.append(adapter._mixture_entropy(logits, action))
        mean_feature = torch.cat(joint_features, dim=0).mean(0, keepdim=True)
        prediction_logit = adapter.self_prediction_head(mean_feature).view(())
        expected_loss = torch.stack(joint_entropies).mean() + 0.3 * (
            F.binary_cross_entropy_with_logits(
                prediction_logit, torch.ones_like(prediction_logit)
            )
        )
        optimized = adapter.params + list(adapter.self_prediction_head.parameters())
        expected_gradients = [
            gradient.detach().clone()
            for gradient in torch.autograd.grad(expected_loss, optimized)
        ]
        before = [parameter.detach().clone() for parameter in optimized]

        adapter.episode_end({"success": 1.0})

        for old, parameter, gradient in zip(
            before, optimized, expected_gradients
        ):
            torch.testing.assert_close(
                parameter.detach() - old,
                -learning_rate * gradient,
                rtol=1e-4,
                atol=1e-7,
            )


if __name__ == "__main__":
    unittest.main()
