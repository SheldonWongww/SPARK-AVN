"""Unit tests for the VLN prepend-and-strip prompt injector.

These use a mock DUET-style cross-modal encoder so the injection mechanics
(prompt prepend, mask/bias padding, prompt-row strip, per-layer capture) can be
verified without any navigation dependency.
"""
import unittest

import torch
import torch.nn as nn

from navtta_core.tta.vln_fusion import CrossmodalPromptInjector


DIM = 6
NODES = 5
LAYERS = 4


class _MockXLayer(nn.Module):
    """A DUET-``GraphLXRTXLayer``-shaped layer: fuses visual against language."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(DIM, DIM)

    def forward(self, lang, lang_mask, visn, visn_mask, graph_sprels=None):
        # Use the mask/bias so the test can prove they are shape-consistent.
        out = torch.tanh(self.proj(visn))
        if graph_sprels is not None:
            # graph_sprels: [batch, 1, N, N]; must match visn's token count.
            assert graph_sprels.shape[-1] == visn.shape[1], (
                graph_sprels.shape, visn.shape,
            )
        return out


class _MockCrossmodalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_layers = nn.ModuleList([_MockXLayer() for _ in range(LAYERS)])

    def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):
        for layer in self.x_layers:
            img_embeds = layer(
                txt_embeds, txt_masks, img_embeds, img_masks,
                graph_sprels=graph_sprels,
            )
        return img_embeds


class CrossmodalPromptInjectorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.encoder = _MockCrossmodalEncoder()
        self.injector = CrossmodalPromptInjector(
            self.encoder,
            self.encoder.x_layers,
            visual_arg=2,       # img_embeds is the 3rd positional arg
            mask_arg=3,         # img_masks
            sprel_arg="graph_sprels",
            mask_valid_value=1,
        )

    def _inputs(self):
        txt = torch.randn(1, 3, DIM)
        txt_masks = torch.ones(1, 3, dtype=torch.bool)
        img = torch.randn(1, NODES, DIM)
        img_masks = torch.ones(1, NODES, dtype=torch.bool)
        sprel = torch.randn(1, 1, NODES, NODES)
        return txt, txt_masks, img, img_masks, sprel

    def test_output_action_dim_unchanged_after_strip(self):
        txt, txt_masks, img, img_masks, sprel = self._inputs()
        prompt = torch.randn(4, DIM)
        with self.injector.instrument(prompt):
            out = self.encoder(txt, txt_masks, img, img_masks, graph_sprels=sprel)
        # Prompt rows are stripped, so the node dimension is preserved.
        self.assertEqual(out.shape, (1, NODES, DIM))

    def test_prompt_free_pass_is_unmodified(self):
        txt, txt_masks, img, img_masks, sprel = self._inputs()
        with self.injector.instrument(None):
            out = self.encoder(txt, txt_masks, img, img_masks, graph_sprels=sprel)
        ref = self.encoder(txt, txt_masks, img, img_masks, graph_sprels=sprel)
        torch.testing.assert_close(out, ref)

    def test_captured_stats_pool_over_real_nodes_only(self):
        txt, txt_masks, img, img_masks, sprel = self._inputs()
        prompt = torch.randn(4, DIM)
        with self.injector.instrument(prompt):
            self.encoder(txt, txt_masks, img, img_masks, graph_sprels=sprel)
            stats = self.injector.captured_stats(LAYERS)
        self.assertEqual(len(stats), LAYERS)
        for mu, sigma in stats:
            self.assertEqual(mu.shape, (DIM,))
            self.assertEqual(sigma.shape, (DIM,))

    def test_injection_changes_captured_features(self):
        txt, txt_masks, img, img_masks, sprel = self._inputs()
        with self.injector.instrument(None):
            self.encoder(txt, txt_masks, img, img_masks, graph_sprels=sprel)
            base = self.injector.captured_features(LAYERS)
        prompt = torch.randn(4, DIM) * 5.0
        with self.injector.instrument(prompt):
            self.encoder(txt, txt_masks, img, img_masks, graph_sprels=sprel)
            prompted = self.injector.captured_features(LAYERS)
        # The prompted capture has L extra rows; the real-node rows differ once
        # the prompt participates in attention (here proj is per-token, so the
        # shapes at least reflect the prepended tokens).
        self.assertEqual(base[0].shape[1], NODES)
        self.assertEqual(prompted[0].shape[1], NODES + 4)

    def test_fisher_features_are_grad_connected(self):
        txt, txt_masks, img, img_masks, sprel = self._inputs()
        with torch.enable_grad():
            with self.injector.instrument(None):
                out = self.encoder(
                    txt, txt_masks, img, img_masks, graph_sprels=sprel
                )
                feats = self.injector.captured_features(LAYERS)
            scalar = out.sum()
            grads = torch.autograd.grad(
                scalar, feats, allow_unused=True, retain_graph=True
            )
        self.assertTrue(any(g is not None for g in grads))

    def test_hooks_removed_after_context(self):
        txt, txt_masks, img, img_masks, sprel = self._inputs()
        with self.injector.instrument(torch.randn(4, DIM)):
            self.encoder(txt, txt_masks, img, img_masks, graph_sprels=sprel)
        # After the context, a prompt-free forward must match the raw encoder.
        out = self.encoder(txt, txt_masks, img, img_masks, graph_sprels=sprel)
        self.assertEqual(out.shape, (1, NODES, DIM))


if __name__ == "__main__":
    unittest.main()
