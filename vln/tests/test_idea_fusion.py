"""IDEA fusion-protocol binding tests for the discrete VLN models.

These use DUET-shaped and HAMT-shaped mock policies with real cross-modal layer
stacks so the ``GraphIDEAProtocol`` / ``HAMTIDEAProtocol`` bindings can be
exercised end-to-end (prompt injection, per-layer capture, action-dim
preservation, frozen policy) without a Habitat runtime.
"""
from pathlib import Path
import sys
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

from navtta_core.tta import IDEAAdapter, module_state_sha256  # noqa: E402
from navtta_vln.idea_fusion import (  # noqa: E402
    GraphIDEAProtocol,
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
        return torch.tanh(self.proj(visn))


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


class GraphIDEAProtocolTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.policy = _MockDuetPolicy()
        self.policy.eval()
        self.protocol = GraphIDEAProtocol(self.policy, warmup_steps=1)

    def test_family_inference_selects_graph(self):
        protocol = make_idea_fusion_protocol(self.policy, "graph", warmup_steps=1)
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


if __name__ == "__main__":
    unittest.main()
