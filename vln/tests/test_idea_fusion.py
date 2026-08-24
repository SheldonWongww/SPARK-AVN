"""IDEA fusion-protocol binding tests for the discrete VLN models.

These use DUET-shaped and HAMT-shaped mock policies with real cross-modal layer
stacks so the ``GraphIDEAProtocol`` / ``HAMTIDEAProtocol`` bindings can be
exercised end-to-end (prompt injection, per-layer capture, action-dim
preservation, frozen policy) without a Habitat runtime.
"""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as error:  # Host-side layout checks need no Torch.
    raise unittest.SkipTest("VLN IDEA tests require torch") from error


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "core"))
sys.path.insert(0, str(REPO_ROOT / "vln"))

from navtta_core.tta import (  # noqa: E402
    IDEAAdapter,
    SourceStatisticsAccumulator,
    module_state_sha256,
)
from navtta_vln.idea_fusion import (  # noqa: E402
    ContinuousIDEAProtocol,
    GraphIDEAProtocol,
    HAMTIDEAProtocol,
    make_idea_fusion_protocol,
)


HID = 6
XLAYERS = 4
NODES = 5


class _MockXLayer(nn.Module):
    """DUET ``GraphLXRTXLayer``-shaped: fuses visn against language tokens."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HID, HID)

    def forward(self, txt, txt_mask, visn, visn_mask, graph_sprels=None):
        if graph_sprels is not None:
            assert graph_sprels.shape[-1] == visn.shape[1]
        valid = visn_mask.bool()
        context = (visn * valid.unsqueeze(-1)).sum(1, keepdim=True)
        context = context / valid.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        return torch.tanh(self.proj(visn + context))


class _MockCrossmodalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_layers = nn.ModuleList([_MockXLayer() for _ in range(XLAYERS)])

    def forward(self, txt, txt_mask, visn, visn_mask, graph_sprels=None):
        for layer in self.x_layers:
            visn = layer(txt, txt_mask, visn, visn_mask, graph_sprels=graph_sprels)
        return visn


class _MockBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _MockCrossmodalEncoder()


class _MockDuetInner(nn.Module):
    def __init__(self):
        super().__init__()
        # Real DUET/HAMT expose a HF-style config with hidden_size.
        self.config = SimpleNamespace(hidden_size=HID)
        self.global_encoder = _MockBranch()
        self.local_encoder = _MockBranch()
        self.global_sap_head = nn.Linear(HID, 1)
        self.local_sap_head = nn.Linear(HID, 1)


class _MockDuetPolicy(nn.Module):
    """Minimal DUET-shaped policy exercising the ``family='graph'`` replay."""

    def __init__(self):
        super().__init__()
        self.vln_bert = _MockDuetInner()

    def forward(self, mode, model_inputs):
        assert mode == "navigation"
        txt = model_inputs["txt_embeds"]
        txt_mask = model_inputs["txt_masks"]
        gmap = model_inputs["gmap_embeds"]
        vp = model_inputs["vp_embeds"]
        node_mask = model_inputs["vp_masks"]
        gmap_out = self.vln_bert.global_encoder.encoder(
            txt, txt_mask, gmap, node_mask, graph_sprels=None
        )
        vp_out = self.vln_bert.local_encoder.encoder(
            txt, txt_mask, vp, node_mask, graph_sprels=model_inputs.get("sprels")
        )
        global_logits = self.vln_bert.global_sap_head(gmap_out).squeeze(-1)
        local_logits = self.vln_bert.local_sap_head(vp_out).squeeze(-1)
        return {
            "gmap_embeds": gmap_out,
            "vp_embeds": vp_out,
            "global_logits": global_logits,
            "local_logits": local_logits,
            "fused_logits": global_logits + local_logits,
        }


def _duet_policy_inputs():
    torch.manual_seed(0)
    return {
        "family": "graph",
        "fusion": "dynamic",
        "model_inputs": {
            "txt_embeds": torch.randn(1, 3, HID),
            "txt_masks": torch.ones(1, 3, dtype=torch.bool),
            "gmap_embeds": torch.randn(1, NODES, HID),
            "vp_embeds": torch.randn(1, NODES, HID),
            "vp_masks": torch.ones(1, NODES, dtype=torch.bool),
            "sprels": torch.randn(1, 1, NODES, NODES),
        },
    }


def _write_source_artifact(protocol, policy_inputs, directory, name, layers):
    collector = SourceStatisticsAccumulator(layers, HID)
    collector.begin_trajectory("source-trajectory")
    protocol.collect_source_step(policy_inputs, collector)
    path = Path(directory) / (name + ".json")
    digest = collector.save(
        path,
        {
            "checkpoint_sha256": "a" * 64,
            "dataset": "unit-vln",
            "dataset_version": "v1",
            "split": "train",
        },
        expected_trajectory_count=1,
    )
    return path, digest


class GraphIDEAProtocolTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.policy = _MockDuetPolicy()
        self.policy.eval()
        self._tempdir = tempfile.TemporaryDirectory()
        collector_protocol = GraphIDEAProtocol(
            self.policy, source_collection=True
        )
        self.source_path, self.source_digest = _write_source_artifact(
            collector_protocol,
            _duet_policy_inputs(),
            self._tempdir.name,
            "duet",
            XLAYERS,
        )
        self.protocol = GraphIDEAProtocol(
            self.policy,
            source_stats_path=self.source_path,
            source_stats_sha256=self.source_digest,
            expected_source_trajectories=1,
        )

    def tearDown(self):
        self._tempdir.cleanup()

    def test_family_inference_selects_graph(self):
        protocol = make_idea_fusion_protocol(
            self.policy,
            "graph",
            source_stats_path=self.source_path,
            source_stats_sha256=self.source_digest,
            expected_source_trajectories=1,
        )
        self.assertIs(type(protocol), GraphIDEAProtocol)
        self.assertEqual(protocol.num_layers, XLAYERS)
        self.assertEqual(protocol.feature_dim, HID)

    def test_prompt_free_logits_match_native_and_action_dim_preserved(self):
        inputs = _duet_policy_inputs()
        native = self.policy("navigation", inputs["model_inputs"])["fused_logits"]
        stats, logits = self.protocol.fused_forward(inputs, None)
        self.assertEqual(logits.shape, native.shape)  # action dim preserved
        torch.testing.assert_close(logits, native)
        self.assertEqual(len(stats), XLAYERS)

    def test_prompt_injection_preserves_action_dim(self):
        inputs = _duet_policy_inputs()
        prompt = torch.randn(4, HID)
        _, logits = self.protocol.fused_forward(inputs, prompt)
        self.assertEqual(logits.shape, (1, NODES))

    def test_missing_artifact_fails_closed(self):
        with self.assertRaises(ValueError):
            GraphIDEAProtocol(self.policy)

    def test_padded_nodes_are_excluded_from_statistics(self):
        inputs = _duet_policy_inputs()
        inputs["model_inputs"]["vp_masks"][0, -1] = False
        first, _ = self.protocol.fused_forward(inputs, None)
        inputs["model_inputs"]["vp_embeds"][0, -1] = 1e6
        second, _ = self.protocol.fused_forward(inputs, None)
        for (mu_a, std_a), (mu_b, std_b) in zip(first, second):
            torch.testing.assert_close(mu_a, mu_b)
            torch.testing.assert_close(std_a, std_b)

    def test_end_to_end_idea_never_mutates_policy(self):
        before = module_state_sha256(self.policy)
        adapter = IDEAAdapter(
            self.policy, self.protocol,
            prompt_length=4, capacity=4, opt_steps=2, tau=1e-9,
        )
        for _ in range(3):
            adapter.prepare_action(
                torch.zeros(1, NODES), policy_inputs=_duet_policy_inputs()
            )
        self.assertEqual(module_state_sha256(self.policy), before)
        self.assertGreaterEqual(len(adapter.library), 1)


class _MockGoatLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HID, HID)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        encoder_attention_mask,
        graph_sprels=None,
    ):
        valid = attention_mask.bool()
        context = (hidden_states * valid.unsqueeze(-1)).sum(1, keepdim=True)
        context = context / valid.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        return (torch.tanh(self.proj(hidden_states + context)),)


class _MockGoatEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.crossattention = nn.ModuleList(
            [_MockGoatLayer() for _ in range(XLAYERS)]
        )

    def forward(self, q, q_mask, kv, kv_mask, graph_sprels=None):
        for layer in self.crossattention:
            q = layer(
                hidden_states=q,
                encoder_hidden_states=kv,
                attention_mask=q_mask,
                encoder_attention_mask=kv_mask,
                graph_sprels=graph_sprels,
            )[0]
        return q


class _MockGoatBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _MockGoatEncoder()


class _MockGoatPolicy(_MockDuetPolicy):
    def __init__(self):
        nn.Module.__init__(self)
        self.vln_bert = _MockDuetInner()
        self.vln_bert.global_encoder = _MockGoatBranch()
        self.vln_bert.local_encoder = _MockGoatBranch()

    def forward(self, mode, model_inputs):
        assert mode == "navigation"
        txt, txt_mask = model_inputs["txt_embeds"], model_inputs["txt_masks"]
        gmap, vp = model_inputs["gmap_embeds"], model_inputs["vp_embeds"]
        mask = model_inputs["vp_masks"]
        gmap_out = self.vln_bert.global_encoder.encoder(
            gmap, mask, txt, txt_mask
        )
        vp_out = self.vln_bert.local_encoder.encoder(
            vp, mask, txt, txt_mask, graph_sprels=model_inputs.get("sprels")
        )
        global_logits = self.vln_bert.global_sap_head(gmap_out).squeeze(-1)
        local_logits = self.vln_bert.local_sap_head(vp_out).squeeze(-1)
        return {
            "gmap_embeds": gmap_out,
            "vp_embeds": vp_out,
            "global_logits": global_logits,
            "local_logits": local_logits,
            "fused_logits": global_logits + local_logits,
        }


class GOATIDEAProtocolTest(unittest.TestCase):
    def test_goat_uses_crossattention_visual_query_binding(self):
        policy = _MockGoatPolicy().eval()
        with tempfile.TemporaryDirectory() as directory:
            collector_protocol = GraphIDEAProtocol(
                policy, source_collection=True
            )
            self.assertEqual(
                collector_protocol.binding, "goat_crossattention"
            )
            path, digest = _write_source_artifact(
                collector_protocol, _duet_policy_inputs(), directory, "goat", XLAYERS
            )
            protocol = GraphIDEAProtocol(
                policy,
                source_stats_path=path,
                source_stats_sha256=digest,
                expected_source_trajectories=1,
            )
            base_stats, base = protocol.fused_forward(_duet_policy_inputs(), None)
            prompt_stats, prompted = protocol.fused_forward(
                _duet_policy_inputs(), torch.randn(4, HID)
            )
            self.assertEqual(prompted.shape, base.shape)
            self.assertFalse(torch.allclose(prompted, base))
            self.assertEqual(len(prompt_stats), len(base_stats))


class _MockHAMTLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(HID, HID)

    def forward(self, language, language_mask, visual, visual_mask):
        valid = visual_mask[:, 0, 0] == 0
        context = (visual * valid.unsqueeze(-1)).sum(1, keepdim=True)
        context = context / valid.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        return language, torch.tanh(self.proj(visual + context))


class _MockHAMTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_layers = nn.ModuleList([_MockHAMTLayer() for _ in range(XLAYERS)])

    def forward(self, *args, **kwargs):
        raise AssertionError("HAMT navigation must bypass encoder.forward")


class _MockHAMTInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HID)
        self.encoder = _MockHAMTEncoder()
        self.next_action = nn.Linear(HID, 1)
        self.ref_object = nn.Linear(HID, 1)


class _MockHAMTPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vln_bert = _MockHAMTInner()

    def forward(
        self,
        hist_embeds,
        ob_embeds,
        ob_masks,
        txt_embeds,
        ob_nav_types=None,
        obj_embeds=None,
        obj_masks=None,
        return_states=False,
    ):
        hist_len = hist_embeds.shape[1]
        visual_parts = [hist_embeds, ob_embeds]
        valid_parts = [
            torch.ones(hist_embeds.shape[:2], dtype=torch.bool), ob_masks
        ]
        if obj_embeds is not None:
            visual_parts.append(obj_embeds)
            valid_parts.append(obj_masks)
        visual = torch.cat(visual_parts, dim=1)
        valid = torch.cat(valid_parts, dim=1).to(visual.device)
        additive = (~valid).to(visual.dtype)[:, None, None] * -10000.0
        language_mask = torch.zeros(
            txt_embeds.shape[0], 1, 1, txt_embeds.shape[1],
            dtype=visual.dtype, device=visual.device,
        )
        language = txt_embeds
        for layer in self.vln_bert.encoder.x_layers:
            language, visual = layer(
                language, language_mask, visual, additive
            )
        ob_len = ob_embeds.shape[1]
        observations = visual[:, hist_len:hist_len + ob_len]
        objects = visual[:, hist_len + ob_len:]
        act_logits = self.vln_bert.next_action(observations).squeeze(-1)
        obj_logits = self.vln_bert.ref_object(objects).squeeze(-1)
        nav_valid = ob_masks if ob_nav_types is None else (
            ob_masks & ob_nav_types.ne(0)
        )
        act_logits = act_logits.masked_fill(~nav_valid, -float("inf"))
        obj_logits = obj_logits.masked_fill(~obj_masks, -float("inf"))
        return {
            "act_logits": act_logits,
            "obj_logits": obj_logits,
            "states": observations[:, 0],
        }


def _hamt_inputs():
    torch.manual_seed(19)
    return {
        "family": "hamt_reverie",
        "model_inputs": {
            "hist_embeds": torch.randn(1, 3, HID),
            "ob_embeds": torch.randn(1, NODES, HID),
            "ob_masks": torch.ones(1, NODES, dtype=torch.bool),
            "ob_nav_types": torch.tensor([[1, 1, 1, 0, 0]]),
            "obj_embeds": torch.randn(1, 2, HID),
            "obj_masks": torch.tensor([[True, False]]),
            "txt_embeds": torch.randn(1, 4, HID),
        },
    }


class HAMTIDEAProtocolTest(unittest.TestCase):
    def test_direct_x_layers_and_reverie_object_stop_are_prompted(self):
        policy = _MockHAMTPolicy().eval()
        inputs = _hamt_inputs()
        with tempfile.TemporaryDirectory() as directory:
            collector_protocol = HAMTIDEAProtocol(
                policy, source_collection=True
            )
            path, digest = _write_source_artifact(
                collector_protocol, inputs, directory, "hamt", XLAYERS
            )
            protocol = HAMTIDEAProtocol(
                policy,
                source_stats_path=path,
                source_stats_sha256=digest,
                expected_source_trajectories=1,
            )
            native_outputs = policy(**inputs["model_inputs"])
            native = torch.cat([
                native_outputs["act_logits"],
                native_outputs["obj_logits"].max(dim=1).indices.unsqueeze(1),
            ], dim=1)
            _, base = protocol.fused_forward(inputs, None)
            torch.testing.assert_close(base, native)
            base_objects = protocol.prompted_object_logits.detach().clone()
            _, prompted = protocol.fused_forward(inputs, torch.randn(4, HID))
            # N action logits plus HAMT's native object-index stop score.
            self.assertEqual(prompted.shape, (1, NODES + 1))
            self.assertFalse(torch.allclose(prompted, base))
            self.assertFalse(torch.allclose(
                protocol.prompted_object_logits, base_objects
            ))
            self.assertEqual(protocol.binding, "hamt_direct_x_layers")

            with torch.enable_grad():
                features, _, masks = protocol.fisher_forward(inputs)
            self.assertEqual(len(features), XLAYERS)
            expected = torch.tensor([
                [False, False, False, True, True, True, False, False,
                 False, False]
            ])
            for feature, mask in zip(features, masks):
                self.assertEqual(tuple(mask.shape), tuple(feature.shape[:-1]))
                torch.testing.assert_close(mask.cpu(), expected)


class _MockContinuousModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HID)
        self.global_encoder = _MockBranch()
        self.local_encoder = _MockBranch()
        self.global_sap_head = nn.Linear(HID, 1)
        self.local_sap_head = nn.Linear(HID, 1)


def _continuous_inputs():
    torch.manual_seed(23)
    return {
        "txt_embeds": torch.randn(1, 3, HID),
        "txt_masks": torch.ones(1, 3, dtype=torch.bool),
        "gmap_embeds": torch.randn(1, NODES, HID),
        "gmap_masks": torch.ones(1, NODES, dtype=torch.bool),
        "bev_embeds": torch.randn(1, NODES, HID),
        "bev_masks": torch.ones(1, NODES, dtype=torch.bool),
        "bev_nav_masks": torch.ones(1, NODES, dtype=torch.bool),
    }


def _continuous_forward(model, inputs):
    global_tokens = model.global_encoder.encoder(
        inputs["txt_embeds"], inputs["txt_masks"],
        inputs["gmap_embeds"], inputs["gmap_masks"],
    )
    local_tokens = model.local_encoder.encoder(
        inputs["txt_embeds"], inputs["txt_masks"],
        inputs["bev_embeds"], inputs["bev_masks"],
    )
    logits = (
        model.global_sap_head(global_tokens).squeeze(-1)
        + model.local_sap_head(local_tokens).squeeze(-1)
    )
    return global_tokens[:, 0], logits


class ContinuousIDEAProtocolTest(unittest.TestCase):
    def test_etpnav_and_bevbert_bind_real_decision_branches(self):
        for variant in ("etpnav", "bevbert"):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                model = _MockContinuousModel().eval()
                collector_protocol = ContinuousIDEAProtocol(
                    model,
                    _continuous_forward,
                    variant,
                    source_collection=True,
                )
                path, digest = _write_source_artifact(
                    collector_protocol,
                    _continuous_inputs(),
                    directory,
                    variant,
                    XLAYERS,
                )
                protocol = ContinuousIDEAProtocol(
                    model,
                    _continuous_forward,
                    variant,
                    source_stats_path=path,
                    source_stats_sha256=digest,
                    expected_source_trajectories=1,
                )
                _, base = protocol.fused_forward(_continuous_inputs(), None)
                _, prompted = protocol.fused_forward(
                    _continuous_inputs(), torch.randn(4, HID)
                )
                self.assertEqual(prompted.shape, base.shape)
                self.assertFalse(torch.allclose(prompted, base))


if __name__ == "__main__":
    unittest.main()
