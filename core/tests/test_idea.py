"""Unit tests for the task-agnostic IDEA adapter (ICML 2026).

These tests use a tiny promptable fusion model and a concrete
:class:`IDEAFusionProtocol` implementation so that the algorithm can be
exercised end-to-end without any navigation dependency.  They assert the
Algorithm-1 control flow (cold start -> asset creation -> bridge reuse), the
closed-form bridge weights, the capacity-bounded library, the Fisher-guided
weighting, and -- most importantly -- that IDEA never mutates the base policy.
"""
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn

from navtta_core.tta import build_adapter
from navtta_core.tta.idea import (
    IDEAAdapter,
    IDEAFusionProtocol,
    solve_bridge_weights,
)
from navtta_core.tta.fusion import (
    SourceStatisticsAccumulator,
    SourceStatisticsArtifact,
    SourceStatisticsCollectionSession,
    TransformerFusionProtocol,
    build_multiscale_memory_key_padding_mask,
    collect_source_statistics,
    pool_tokens_to_stats,
)
from navtta_core.tta.tta_core import module_state_sha256


FEATURE_DIM = 6
NUM_LAYERS = 3
NUM_NODES = 5
NUM_ACTIONS = 4


class _PromptableFusion(nn.Module):
    """A minimal multi-layer fusion transformer that accepts a soft prompt.

    Each layer is a LayerNorm+linear block.  A prompt is prepended to the node
    tokens; per-layer statistics are pooled over the node dimension only (the
    prompt rows are excluded), matching IDEA's node-dimension pooling.
    """

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(FEATURE_DIM),
                nn.Linear(FEATURE_DIM, FEATURE_DIM),
                nn.Tanh(),
            )
            for _ in range(NUM_LAYERS)
        ])
        self.head = nn.Linear(FEATURE_DIM, NUM_ACTIONS)

    def forward(self, tokens, prompt=None, num_prompt=0):
        if prompt is not None:
            tokens = torch.cat([prompt, tokens], dim=0)
        layer_features = []
        hidden = tokens
        for block in self.blocks:
            # Cross-token mixing makes the external prompt influence real nodes.
            hidden = block(hidden + hidden.mean(dim=0, keepdim=True))
            layer_features.append(hidden)
        pooled = layer_features[-1][num_prompt:].mean(dim=0, keepdim=True)
        logits = self.head(pooled)
        return layer_features, logits


class _TinyIDEAPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = _PromptableFusion()


def _layer_stats(layer_features, num_prompt=0):
    stats = []
    for feature in layer_features:
        feature = feature[num_prompt:]
        stats.append(pool_tokens_to_stats(feature))
    return stats


class _TinyProtocol(IDEAFusionProtocol):
    """Concrete protocol that drives ``_PromptableFusion`` by submodule calls."""

    def __init__(self, policy, source_tokens):
        self.policy = policy
        # Precomputed source anchor statistics (prompt-free source subset).
        with torch.no_grad():
            features, _ = policy.fusion(source_tokens)
        self._source_stats = [
            (mu.detach(), sigma.detach()) for mu, sigma in _layer_stats(features)
        ]
        self._source_metadata = {
            "mode": "offline_artifact",
            "schema": "navtta.idea.source_statistics",
            "sha256": "0" * 64,
            "trajectory_count": 128,
            "provenance": {},
        }

    @property
    def feature_dim(self):
        return FEATURE_DIM

    @property
    def num_layers(self):
        return NUM_LAYERS

    @property
    def source_statistics_metadata(self):
        return dict(self._source_metadata)

    def source_statistics(self, device, dtype):
        return [
            (mu.to(device=device, dtype=dtype), sigma.to(device=device, dtype=dtype))
            for mu, sigma in self._source_stats
        ]

    def fused_forward(self, policy_inputs, prompt):
        tokens = policy_inputs["tokens"]
        num_prompt = 0 if prompt is None else prompt.shape[0]
        features, logits = self.policy.fusion(
            tokens, prompt=prompt, num_prompt=num_prompt
        )
        return _layer_stats(features, num_prompt=num_prompt), logits

    def fisher_forward(self, policy_inputs):
        tokens = policy_inputs["tokens"].clone().requires_grad_(True)
        features, logits = self.policy.fusion(tokens)
        # Return graph-connected per-layer features for the Fisher trace.
        valid = torch.ones(NUM_NODES, dtype=torch.bool, device=tokens.device)
        return features, logits, [valid] * NUM_LAYERS


def _inputs(shift=0.0):
    torch.manual_seed(0)
    tokens = torch.randn(NUM_NODES, FEATURE_DIM) + shift
    return {"tokens": tokens}


def _source_tokens():
    torch.manual_seed(123)
    return torch.randn(NUM_NODES, FEATURE_DIM)


def _make_adapter(policy, protocol, **overrides):
    kwargs = dict(
        prompt_length=4,
        capacity=8,
        lam=0.4,
        tau=0.7,
        opt_steps=3,
        lr=3e-3,
    )
    kwargs.update(overrides)
    return IDEAAdapter(policy, protocol, **kwargs)


class IDEASolverTest(unittest.TestCase):
    def test_bridge_weights_sum_to_one_and_are_nonnegative(self):
        torch.manual_seed(0)
        gammas = torch.randn(5, 2 * FEATURE_DIM)
        target = torch.randn(2 * FEATURE_DIM)
        uncertainties = torch.rand(5) + 0.1
        weights = solve_bridge_weights(gammas, target, uncertainties, lam=0.4)
        self.assertEqual(weights.shape, (5,))
        self.assertTrue(torch.all(weights >= 0))
        self.assertAlmostEqual(float(weights.sum().item()), 1.0, places=5)

    def test_single_asset_bridge_is_trivial(self):
        gammas = torch.randn(1, 2 * FEATURE_DIM)
        target = torch.randn(2 * FEATURE_DIM)
        weights = solve_bridge_weights(
            gammas, target, torch.tensor([0.5]), lam=0.4
        )
        self.assertEqual(weights.shape, (1,))
        self.assertAlmostEqual(float(weights.item()), 1.0, places=6)

    def test_bridge_recovers_exact_convex_target(self):
        # If the target equals a convex combination of two assets, the weights
        # should recover that combination (closed form is exact here).
        torch.manual_seed(1)
        gammas = torch.randn(2, 2 * FEATURE_DIM)
        true_w = torch.tensor([0.3, 0.7])
        target = true_w @ gammas
        weights = solve_bridge_weights(
            gammas, target, torch.zeros(2), lam=0.0, ridge=1e-8
        )
        torch.testing.assert_close(weights, true_w, rtol=1e-3, atol=1e-3)


class IDEAAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.policy = _TinyIDEAPolicy()
        self.protocol = _TinyProtocol(self.policy, _source_tokens())

    def test_first_step_is_cold_start_and_creates_one_asset(self):
        adapter = _make_adapter(self.policy, self.protocol)
        logits = adapter.prepare_action(
            torch.zeros(1, NUM_ACTIONS), policy_inputs=_inputs(shift=3.0)
        )
        self.assertEqual(logits.shape, (1, NUM_ACTIONS))
        diag = adapter.diagnostics()
        self.assertEqual(diag["library_size"], 1)
        self.assertEqual(diag["cold_start_steps"], 1)
        self.assertEqual(diag["new_domain_steps"], 1)
        self.assertEqual(diag["covered_steps"], 0)
        self.assertEqual(diag["updates"], adapter.opt_steps)
        self.assertEqual(diag["prompt_optimizer_attempts"], adapter.opt_steps)
        self.assertEqual(diag["prompt_optimizer_updates"], adapter.opt_steps)
        self.assertEqual(
            diag["adapted_parameter_count"],
            adapter.prompt_length * adapter.feature_dim,
        )
        self.assertEqual(
            diag["adapted_parameter_names"], ["external_soft_prompt"]
        )
        self.assertGreater(diag["relative_param_drift"], 0.0)
        self.assertEqual(
            diag["base_parameter_integrity_schema"],
            "navtta.idea.base_parameter_integrity.v1",
        )
        self.assertTrue(diag["base_parameter_integrity_complete"])
        self.assertTrue(diag["base_parameter_unchanged"])
        self.assertFalse(diag["trains_base_policy"])
        self.assertEqual(diag["base_parameter_requires_grad_names"], [])
        self.assertTrue(diag["base_parameter_grads_none"])
        self.assertEqual(diag["base_parameter_gradient_names"], [])
        self.assertEqual(
            diag["base_parameter_name_count_before"],
            diag["base_parameter_name_count_after"],
        )
        self.assertTrue(diag["base_parameter_name_set_unchanged"])
        self.assertEqual(diag["base_parameter_added_names"], [])
        self.assertEqual(diag["base_parameter_removed_names"], [])
        self.assertEqual(
            len(diag["base_parameter_name_set_before_sha256"]), 64
        )
        self.assertEqual(
            diag["base_parameter_name_set_before_sha256"],
            diag["base_parameter_name_set_after_sha256"],
        )
        self.assertEqual(
            len(diag["base_parameter_content_before_sha256"]), 64
        )
        self.assertEqual(
            diag["base_parameter_content_before_sha256"],
            diag["base_parameter_content_after_sha256"],
        )
        self.assertTrue(diag["base_parameter_content_hash_match"])
        self.assertEqual(diag["base_parameter_content_modified_names"], [])
        self.assertEqual(
            diag["base_parameter_content_hash_algorithm"],
            "sha256_of_sorted_named_parameter_sha256_v1",
        )
        self.assertTrue(diag["base_parameter_versions_unchanged"])
        self.assertEqual(diag["base_parameter_version_changed_names"], [])
        self.assertEqual(diag["base_parameter_modified_names"], [])

    def test_repeated_identical_domain_is_eventually_covered(self):
        # A wide coverage threshold plus a repeated domain should trigger the
        # training-free bridge reuse branch at least once.
        adapter = _make_adapter(self.policy, self.protocol, tau=100.0)
        same = _inputs(shift=2.0)
        adapter.prepare_action(torch.zeros(1, NUM_ACTIONS), policy_inputs=same)
        adapter.prepare_action(torch.zeros(1, NUM_ACTIONS), policy_inputs=same)
        diag = adapter.diagnostics()
        self.assertGreaterEqual(diag["covered_steps"], 1)

    def test_never_covered_when_threshold_is_zero(self):
        # tau -> 0 makes coverage effectively impossible, so every step forms a
        # new asset (bounded by capacity via merge).
        adapter = _make_adapter(self.policy, self.protocol, tau=1e-9, capacity=3)
        for shift in (0.0, 1.0, 2.0, 3.0, 4.0):
            adapter.prepare_action(
                torch.zeros(1, NUM_ACTIONS), policy_inputs=_inputs(shift=shift)
            )
        diag = adapter.diagnostics()
        self.assertEqual(diag["covered_steps"], 0)
        self.assertEqual(diag["library_size"], 3)
        self.assertGreaterEqual(diag["asset_merges"], 1)

    def test_library_capacity_is_never_exceeded(self):
        adapter = _make_adapter(self.policy, self.protocol, tau=1e-9, capacity=2)
        for shift in range(6):
            adapter.prepare_action(
                torch.zeros(1, NUM_ACTIONS),
                policy_inputs=_inputs(shift=float(shift)),
            )
        self.assertLessEqual(len(adapter.library), 2)

    def test_base_policy_is_never_mutated(self):
        before = module_state_sha256(self.policy)
        adapter = _make_adapter(self.policy, self.protocol, tau=1e-9)
        for shift in (0.0, 1.5, 3.0):
            adapter.prepare_action(
                torch.zeros(1, NUM_ACTIONS), policy_inputs=_inputs(shift=shift)
            )
            adapter.adapt(torch.zeros(1, NUM_ACTIONS))
            adapter.episode_end({"success": 1.0})
        after = module_state_sha256(self.policy)
        self.assertEqual(before, after)
        self.assertTrue(all(not p.requires_grad for p in self.policy.parameters()))
        self.assertTrue(all(p.grad is None for p in self.policy.parameters()))

    def test_content_hash_detects_write_that_can_bypass_version_counter(self):
        adapter = _make_adapter(self.policy, self.protocol)
        parameter_name, parameter = next(self.policy.named_parameters())
        with torch.no_grad():
            parameter.data.add_(1.0)
        diag = adapter.diagnostics()
        self.assertTrue(diag["base_parameter_name_set_unchanged"])
        self.assertTrue(diag["base_parameter_versions_unchanged"])
        self.assertFalse(diag["base_parameter_unchanged"])
        self.assertFalse(diag["base_parameter_content_hash_match"])
        self.assertIn(
            parameter_name, diag["base_parameter_content_modified_names"]
        )
        self.assertIn(parameter_name, diag["base_parameter_modified_names"])

    def test_intermediate_diagnostics_defer_full_content_hash(self):
        adapter = _make_adapter(self.policy, self.protocol)
        diag = adapter.diagnostics(verify_base_content=False)
        self.assertFalse(diag["base_parameter_integrity_complete"])
        self.assertIsNone(diag["base_parameter_unchanged"])
        self.assertIsNone(diag["base_parameter_content_after_sha256"])
        self.assertIsNone(diag["base_parameter_content_hash_match"])
        self.assertTrue(diag["base_parameter_name_set_unchanged"])
        self.assertTrue(diag["base_parameter_versions_unchanged"])

    def test_parameter_name_set_change_is_reported(self):
        adapter = _make_adapter(self.policy, self.protocol)
        self.policy.register_parameter(
            "unexpected_parameter",
            nn.Parameter(torch.zeros(1), requires_grad=False),
        )
        diag = adapter.diagnostics()
        self.assertFalse(diag["base_parameter_unchanged"])
        self.assertFalse(diag["base_parameter_name_set_unchanged"])
        self.assertEqual(
            diag["base_parameter_name_count_after"],
            diag["base_parameter_name_count_before"] + 1,
        )
        self.assertEqual(
            diag["base_parameter_added_names"], ["unexpected_parameter"]
        )
        self.assertEqual(diag["base_parameter_removed_names"], [])
        self.assertFalse(diag["base_parameter_content_hash_match"])
        self.assertFalse(diag["base_parameter_versions_unchanged"])
        self.assertIn(
            "unexpected_parameter", diag["base_parameter_modified_names"]
        )

    def test_fisher_weights_stay_normalised_and_respond(self):
        adapter = _make_adapter(self.policy, self.protocol, tau=1e-9)
        uniform = adapter.alpha.clone()
        adapter.prepare_action(
            torch.zeros(1, NUM_ACTIONS), policy_inputs=_inputs(shift=2.0)
        )
        self.assertAlmostEqual(float(adapter.alpha.sum().item()), 1.0, places=5)
        self.assertTrue(torch.all(adapter.alpha >= 0))
        # With a non-degenerate model the trace is not perfectly uniform, so the
        # EMA should move the weights away from the uniform initialisation.
        self.assertFalse(torch.allclose(adapter.alpha, uniform, atol=1e-4))

    def test_disabling_fisher_keeps_uniform_alpha(self):
        adapter = _make_adapter(
            self.policy, self.protocol, tau=1e-9, use_fisher=False
        )
        adapter.prepare_action(
            torch.zeros(1, NUM_ACTIONS), policy_inputs=_inputs(shift=2.0)
        )
        torch.testing.assert_close(
            adapter.alpha,
            torch.full((NUM_LAYERS,), 1.0 / NUM_LAYERS),
        )

    def test_reset_clears_library_and_weights(self):
        adapter = _make_adapter(self.policy, self.protocol, tau=1e-9)
        adapter.prepare_action(
            torch.zeros(1, NUM_ACTIONS), policy_inputs=_inputs(shift=2.0)
        )
        self.assertGreaterEqual(len(adapter.library), 1)
        adapter.reset()
        self.assertEqual(len(adapter.library), 0)
        torch.testing.assert_close(
            adapter.alpha, torch.full((NUM_LAYERS,), 1.0 / NUM_LAYERS)
        )

    def test_episodic_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            IDEAAdapter(self.policy, self.protocol, episodic=True)

    def test_zero_update_audit_confirms_frozen_base_policy(self):
        adapter = _make_adapter(self.policy, self.protocol, tau=1e-9)
        adapter.enable_zero_update_audit(expected_episodes=1)
        adapter.prepare_action(
            torch.zeros(1, NUM_ACTIONS), policy_inputs=_inputs(shift=2.0)
        )
        adapter.episode_end({"success": 1.0})
        diag = adapter.diagnostics()
        self.assertTrue(diag["primary_model_state_hash_match"])

    def test_asset_round_trip_is_digest_pinned(self):
        adapter = _make_adapter(self.policy, self.protocol, tau=1e-9)
        adapter.prepare_action(
            torch.zeros(1, NUM_ACTIONS), policy_inputs=_inputs(shift=2.0)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assets.json"
            digest = adapter.save_assets(path, {"run": "unit-test"})
            restored = _make_adapter(self.policy, self.protocol, tau=1e-9)
            restored.load_assets(path, digest)
            self.assertEqual(len(restored.library), len(adapter.library))
            with self.assertRaises(ValueError):
                restored.load_assets(path, "f" * 64)


class IDEAFactoryTest(unittest.TestCase):
    def test_build_adapter_requires_fusion_protocol(self):
        config = SimpleNamespace(METHOD="idea", EPISODIC=False)
        with self.assertRaises(ValueError):
            build_adapter(_TinyIDEAPolicy(), config)

    def test_build_adapter_reads_idea_config(self):
        policy = _TinyIDEAPolicy()
        protocol = _TinyProtocol(policy, _source_tokens())
        config = SimpleNamespace(
            METHOD="idea",
            EPISODIC=False,
            IDEA=SimpleNamespace(
                PROMPT_LENGTH=4,
                K_MAX=16,
                LAMBDA=0.4,
                TAU=0.7,
                OPT_STEPS=2,
                LR=3e-3,
            ),
        )
        adapter = build_adapter(policy, config, fusion_protocol=protocol)
        self.assertIsInstance(adapter, IDEAAdapter)
        self.assertEqual(adapter.library.capacity, 16)
        self.assertEqual(adapter.prompt_length, 4)
        diag = adapter.diagnostics()
        self.assertEqual(diag["method"], "idea")
        self.assertFalse(diag["trains_base_policy"])


class _NNTransformerPolicy(nn.Module):
    """A policy whose fusion core is a real ``torch.nn.Transformer`` (AVN-like)."""

    def __init__(self, dim=FEATURE_DIM, layers=NUM_LAYERS):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=dim,
            nhead=2,
            num_encoder_layers=layers,
            num_decoder_layers=1,
            dim_feedforward=16,
            dropout=0.0,
        )
        self.head = nn.Linear(dim, NUM_ACTIONS)

    def forward_logits(self, policy_inputs):
        memory = policy_inputs["memory"]  # [S, 1, C]
        padding = policy_inputs.get("padding")
        decoded = self.transformer(
            memory,
            memory[:1],
            src_key_padding_mask=padding,
            memory_key_padding_mask=padding,
        )
        return self.head(decoded[-1])


def _seq_inputs(shift=0.0):
    torch.manual_seed(0)
    memory = torch.randn(NUM_NODES, 1, FEATURE_DIM) + shift
    return {"memory": memory}


class TransformerFusionProtocolTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.policy = _NNTransformerPolicy()
        self.policy.eval()
        self._tempdir = tempfile.TemporaryDirectory()
        collector_protocol = TransformerFusionProtocol.for_source_collection(
            self.policy.transformer,
            self.policy.forward_logits,
            feature_dim=FEATURE_DIM,
        )
        self.source_path = Path(self._tempdir.name) / "source.json"
        self.source_sha = collect_source_statistics(
            collector_protocol,
            [("source-trajectory", [_seq_inputs()])],
            self.source_path,
            {
                "checkpoint_sha256": "a" * 64,
                "dataset": "unit",
                "dataset_version": "v1",
                "split": "train",
            },
            expected_trajectory_count=1,
        )
        self.protocol = TransformerFusionProtocol(
            self.policy.transformer,
            self.policy.forward_logits,
            feature_dim=FEATURE_DIM,
            source_stats_path=self.source_path,
            source_stats_sha256=self.source_sha,
            expected_source_trajectories=1,
        )

    def tearDown(self):
        self._tempdir.cleanup()

    def test_prompt_injection_changes_logits_but_not_prompt_free_pass(self):
        inputs = _seq_inputs(shift=1.0)
        base_stats, base_logits = self.protocol.fused_forward(inputs, None)
        self.assertEqual(len(base_stats), NUM_LAYERS)
        self.assertEqual(base_logits.shape, (1, NUM_ACTIONS))
        prompt = torch.randn(4, FEATURE_DIM)
        _, prompted_logits = self.protocol.fused_forward(inputs, prompt)
        # Injecting a prompt must change the decision; a prompt-free re-run must
        # reproduce the original logits (hooks are fully removed each call).
        self.assertFalse(torch.allclose(base_logits, prompted_logits, atol=1e-5))
        _, base_again = self.protocol.fused_forward(inputs, None)
        torch.testing.assert_close(base_logits, base_again)

    def test_stats_pooled_over_nodes_have_feature_dim(self):
        _, _ = self.protocol.fused_forward(_seq_inputs(), None)
        stats, _ = self.protocol.fused_forward(_seq_inputs(), torch.randn(4, FEATURE_DIM))
        for mu, sigma in stats:
            self.assertEqual(mu.shape, (FEATURE_DIM,))
            self.assertEqual(sigma.shape, (FEATURE_DIM,))

    def test_fisher_features_are_grad_connected(self):
        # The adapter calls fisher_forward under enable_grad; do the same here.
        with torch.enable_grad():
            features, logits, valid_masks = self.protocol.fisher_forward(_seq_inputs())
            self.assertEqual(len(features), NUM_LAYERS)
            self.assertEqual(len(valid_masks), NUM_LAYERS)
            self.assertEqual(tuple(valid_masks[0].shape), tuple(features[0].shape[:-1]))
            grads = torch.autograd.grad(
                logits.log_softmax(-1)[0, 0], features, allow_unused=True,
                retain_graph=True,
            )
        # At least one aligned layer must influence the decision output.
        self.assertTrue(any(g is not None for g in grads))

    def test_end_to_end_with_idea_adapter_never_mutates_policy(self):
        before = module_state_sha256(self.policy)
        adapter = IDEAAdapter(
            self.policy, self.protocol,
            prompt_length=4, capacity=4, opt_steps=2, tau=1e-9,
        )
        for shift in (0.0, 1.0, 2.0):
            adapter.prepare_action(
                torch.zeros(1, NUM_ACTIONS), policy_inputs=_seq_inputs(shift=shift)
            )
        self.assertEqual(module_state_sha256(self.policy), before)
        self.assertGreaterEqual(len(adapter.library), 1)

    def test_align_last_m_layers_when_fewer_requested(self):
        collector_protocol = TransformerFusionProtocol.for_source_collection(
            self.policy.transformer,
            self.policy.forward_logits,
            feature_dim=FEATURE_DIM,
            num_layers=2,
        )
        collector = SourceStatisticsAccumulator(2, FEATURE_DIM)
        collector.begin_trajectory("source-trajectory")
        collector_protocol.collect_source_step(_seq_inputs(), collector)
        path = Path(self._tempdir.name) / "source-two.json"
        digest = collector.save(
            path,
            {
                "checkpoint_sha256": "a" * 64,
                "dataset": "unit",
                "dataset_version": "v1",
                "split": "train",
            },
            expected_trajectory_count=1,
        )
        protocol = TransformerFusionProtocol(
            self.policy.transformer,
            self.policy.forward_logits,
            feature_dim=FEATURE_DIM,
            num_layers=2,
            source_stats_path=path,
            source_stats_sha256=digest,
            expected_source_trajectories=1,
        )
        self.assertEqual(protocol.num_layers, 2)
        stats, _ = protocol.fused_forward(_seq_inputs(), None)
        self.assertEqual(len(stats), 2)

    def test_missing_offline_artifact_fails_closed(self):
        with self.assertRaises(ValueError):
            TransformerFusionProtocol(
                self.policy.transformer,
                self.policy.forward_logits,
                feature_dim=FEATURE_DIM,
            )

    def test_padding_is_excluded_from_layer_statistics(self):
        inputs = _seq_inputs()
        inputs["padding"] = torch.tensor([[False, False, False, True, True]])
        inputs["memory"][3:] = 10000.0
        stats, _ = self.protocol.fused_forward(inputs, None)
        for mu, sigma in stats:
            self.assertTrue(torch.isfinite(mu).all())
            self.assertTrue(torch.isfinite(sigma).all())


class SourceStatisticsArtifactTest(unittest.TestCase):
    def test_global_moments_are_not_an_average_of_step_means(self):
        collector = SourceStatisticsAccumulator(1, 1)
        collector.begin_trajectory("a")
        collector.update([torch.tensor([[0.0]])])
        collector.begin_trajectory("b")
        collector.update([torch.tensor([[2.0], [4.0], [6.0]])])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.json"
            digest = collector.save(
                path,
                {
                    "checkpoint_sha256": "b" * 64,
                    "dataset": "unit",
                    "dataset_version": "v1",
                    "split": "source_train",
                },
                expected_trajectory_count=2,
            )
            artifact = SourceStatisticsArtifact.load(
                path,
                digest,
                expected_feature_dim=1,
                expected_num_layers=1,
                expected_trajectory_count=2,
            )
            (mean, sigma), = artifact.statistics("cpu", torch.float32)
            torch.testing.assert_close(mean, torch.tensor([3.0]))
            torch.testing.assert_close(
                sigma,
                torch.tensor([20.0 / 3.0 + 1e-6]).sqrt(),
                rtol=1e-5,
                atol=1e-5,
            )
            with self.assertRaises(ValueError):
                SourceStatisticsArtifact.load(path, "f" * 64)

    def test_pooling_uses_only_valid_tokens_across_batch(self):
        tokens = torch.tensor([
            [[1.0], [100.0]],
            [[3.0], [5.0]],
        ])
        valid = torch.tensor([[True, False], [True, True]])
        mean, _ = pool_tokens_to_stats(tokens, valid)
        torch.testing.assert_close(mean, torch.tensor([3.0]))

    def test_pooling_uses_paper_epsilon_for_constant_features(self):
        _, sigma = pool_tokens_to_stats(torch.ones(3, 2))
        torch.testing.assert_close(sigma, torch.full((2,), 1e-3))

    def test_multiscale_mask_matches_enmus_lengths(self):
        padding = torch.zeros(1, 32, dtype=torch.bool)
        padding[:, -1] = True
        result = build_multiscale_memory_key_padding_mask(padding)
        self.assertEqual(result.shape, (1, 32 + 8 + 4 + 2 + 1))
        # The last padded base token contaminates the final window at each scale.
        self.assertTrue(result[0, 31])
        self.assertTrue(result[0, -1])

    def test_collection_session_writes_only_after_128_real_ids(self):
        class Protocol:
            source_collection = True
            num_layers = 1
            feature_dim = 1

            @staticmethod
            def collect_source_step(policy_inputs, accumulator):
                accumulator.update([policy_inputs["tokens"]])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            session = SourceStatisticsCollectionSession(
                Protocol(),
                path,
                {
                    "checkpoint_sha256": "d" * 64,
                    "dataset": "unit",
                    "dataset_version": "v1",
                    "split": "train",
                },
            )
            for index in range(127):
                session.begin_trajectory("trajectory-{}".format(index))
                session.observe_step({"tokens": torch.tensor([[float(index)]])})
                self.assertIsNone(session.end_trajectory())
            self.assertFalse(path.exists())
            session.begin_trajectory("trajectory-127")
            session.observe_step({"tokens": torch.tensor([[127.0]])})
            digest = session.end_trajectory()
            self.assertTrue(path.is_file())
            self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
