"""IDEA fusion-protocol bindings for the discrete VLN navigation models.

These realise :class:`navtta_core.tta.IDEAFusionProtocol` for the discrete VLN
policies by driving each model's *frozen* cross-modal fusion stack through
:class:`navtta_core.tta.CrossmodalPromptInjector`.  The paper's soft prompt is
prepended to the visual/candidate token sequence and stripped from the fused
output, so the navigation action space is unchanged while the fusion layers
receive the prompt.  No model ``forward`` is edited and no parameters unfrozen.

Coverage:

* ``graph`` (DUET, GOAT): the injector wraps ``vln_bert.local_encoder.encoder``
  -- the local-viewpoint branch whose per-node tokens are the immediate
  candidate actions -- and aligns its ``x_layers``.  The global map branch is
  left prompt-free (its nodes are not the step's action candidates).
* ``hamt_r2r`` / ``hamt_reverie`` (HAMT): the injector wraps
  ``vln_bert.encoder`` and aligns its ``x_layers``; HAMT concatenates history
  and observation tokens *inside* its encoder, so injection uses layer-input
  capture with HAMT's additive mask convention.

RUNTIME VALIDATION: the mechanics are unit-tested in
``core/tests/test_vln_fusion.py`` against mock encoders.  End-to-end behaviour
on real DUET/HAMT/GOAT checkpoints must be validated on the VLN eval hardware
(Habitat + released weights); this repo checkout cannot execute those policies.
"""
from navtta_core.tta import (
    CrossmodalPromptInjector,
    IDEAFusionProtocol,
    SourceStatisticsAccumulator,
)

from .discrete_tta import discrete_forward_policy


class _InjectedIDEAProtocol(IDEAFusionProtocol):
    """Shared IDEA protocol driven by a :class:`CrossmodalPromptInjector`.

    Subclasses only provide the injector construction (which encoder, arg
    positions, mask convention).  ``forward_logits`` reuses the exact detached
    replay path (:func:`discrete_forward_policy`) that EAM/ATENA already use, so
    the prompt-free logits equal the model's native decision.
    """

    def __init__(self, model, injector, feature_dim, num_layers, warmup_steps=64):
        self.model = model
        self.injector = injector
        self._feature_dim = int(feature_dim)
        self._num_layers = int(num_layers)
        self._source = SourceStatisticsAccumulator(
            self._num_layers, warmup_steps
        )

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def num_layers(self):
        return self._num_layers

    def source_statistics(self, device, dtype):
        return self._source.statistics(self._feature_dim, device, dtype)

    def fused_forward(self, policy_inputs, prompt):
        with self.injector.instrument(prompt):
            _, logits = discrete_forward_policy(self.model, policy_inputs)
            layer_stats = self.injector.captured_stats(self._num_layers)
        if prompt is None:
            self._source.observe(layer_stats)
        return layer_stats, logits

    def fisher_forward(self, policy_inputs):
        with self.injector.instrument(None):
            _, logits = discrete_forward_policy(self.model, policy_inputs)
            features = self.injector.captured_features(self._num_layers)
        return features, logits


class GraphIDEAProtocol(_InjectedIDEAProtocol):
    """DUET / GOAT (``family='graph'``) IDEA protocol."""

    def __init__(self, model, num_layers=None, warmup_steps=64):
        inner = getattr(model, "vln_bert", None)
        if inner is None or not hasattr(inner, "local_encoder"):
            raise ValueError("graph IDEA protocol requires a DUET/GOAT vln_bert")
        encoder = inner.local_encoder.encoder  # CrossmodalEncoder
        layers = list(encoder.x_layers)
        feature_dim = _infer_hidden_size(inner)
        injector = CrossmodalPromptInjector(
            encoder,
            layers,
            # CrossmodalEncoder.forward(txt_embeds, txt_masks, img_embeds,
            #   img_masks, graph_sprels=None): img_embeds is positional index 2.
            visual_arg=2,
            mask_arg=3,
            sprel_arg="graph_sprels",
            mask_valid_value=1,
        )
        resolved_layers = len(layers) if num_layers is None else int(num_layers)
        super().__init__(
            model, injector, feature_dim, resolved_layers, warmup_steps
        )


class HAMTIDEAProtocol(_InjectedIDEAProtocol):
    """HAMT (``family='hamt_r2r'`` / ``'hamt_reverie'``) IDEA protocol.

    HAMT builds its fused ``hist_img_embeds`` inside ``LxmertEncoder.forward``
    and uses an additive attention mask, so the injector wraps the encoder and
    prepends to the visual argument with an additive-mask "keep" value of 0.
    """

    def __init__(self, model, num_layers=None, warmup_steps=64):
        inner = getattr(model, "vln_bert", None)
        if inner is None or not hasattr(inner, "encoder"):
            raise ValueError("HAMT IDEA protocol requires a HAMT vln_bert")
        encoder = inner.encoder  # LxmertEncoder
        layers = list(encoder.x_layers)
        feature_dim = _infer_hidden_size(inner)
        injector = CrossmodalPromptInjector(
            encoder,
            layers,
            # LxmertEncoder.forward(txt_embeds, extended_txt_masks, hist_embeds,
            #   extended_hist_masks, img_embeds=None, extended_img_masks=None):
            # the candidate observation tokens are passed as img_embeds (index 4)
            # with an additive mask (index 5); additive masks keep with value 0.
            visual_arg=4,
            mask_arg=5,
            sprel_arg=None,
            mask_valid_value=0,
            layer_output_index=1,  # x_layers return (txt, hist_img)
        )
        resolved_layers = len(layers) if num_layers is None else int(num_layers)
        super().__init__(
            model, injector, feature_dim, resolved_layers, warmup_steps
        )


def _infer_hidden_size(inner):
    config = getattr(inner, "config", None)
    if config is not None and hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    # Fall back to a common BERT-base width used by DUET/HAMT/GOAT.
    return 768


class ContinuousIDEAProtocol(_InjectedIDEAProtocol):
    """ETPNav / BEVBert (VLN-CE) IDEA protocol.

    The continuous decision model exposes the same ``CrossmodalEncoder`` shape as
    DUET.  ETPNav decides on the global-map branch (``global_encoder``); BEVBert
    fuses a local branch too, but the immediate waypoint candidates come from the
    branch that feeds ``logits_key``.  We inject on ``global_encoder`` for ETPNav
    and ``local_encoder`` for BEVBert, matching each variant's decision branch.
    """

    def __init__(self, decision_model, forward_logits, variant, num_layers=None,
                 warmup_steps=64):
        variant = str(variant).lower()
        if variant == "etpnav":
            branch = getattr(decision_model, "global_encoder", None)
        elif variant == "bevbert":
            branch = getattr(decision_model, "local_encoder", None)
        else:
            raise ValueError("Unknown continuous variant: {}".format(variant))
        encoder = getattr(branch, "encoder", None)
        if encoder is None or not hasattr(encoder, "x_layers"):
            raise ValueError(
                "continuous IDEA protocol requires {}.encoder.x_layers".format(
                    "global_encoder" if variant == "etpnav" else "local_encoder"
                )
            )
        layers = list(encoder.x_layers)
        feature_dim = _infer_hidden_size(decision_model)
        injector = CrossmodalPromptInjector(
            encoder,
            layers,
            visual_arg=2,
            mask_arg=3,
            sprel_arg="graph_sprels",
            mask_valid_value=1,
        )
        resolved_layers = len(layers) if num_layers is None else int(num_layers)
        # The continuous controller drives the forward with (model, nav_inputs);
        # bind the model so the protocol's forward takes only policy_inputs.
        self._bound_model = decision_model
        self._bound_forward = forward_logits
        super().__init__(
            decision_model, injector, feature_dim, resolved_layers, warmup_steps
        )

    def fused_forward(self, policy_inputs, prompt):
        with self.injector.instrument(prompt):
            _, logits = self._bound_forward(self._bound_model, policy_inputs)
            layer_stats = self.injector.captured_stats(self._num_layers)
        if prompt is None:
            self._source.observe(layer_stats)
        return layer_stats, logits

    def fisher_forward(self, policy_inputs):
        with self.injector.instrument(None):
            _, logits = self._bound_forward(self._bound_model, policy_inputs)
            features = self.injector.captured_features(self._num_layers)
        return features, logits


def make_idea_fusion_protocol(model, family, num_layers=None, warmup_steps=64):
    """Return the IDEA fusion protocol for a discrete VLN ``family``."""
    family = str(family)
    if family == "graph":
        return GraphIDEAProtocol(model, num_layers, warmup_steps)
    if family in ("hamt_r2r", "hamt_reverie"):
        return HAMTIDEAProtocol(model, num_layers, warmup_steps)
    raise ValueError(
        "No IDEA fusion protocol for discrete family {!r}".format(family)
    )


__all__ = [
    "GraphIDEAProtocol",
    "HAMTIDEAProtocol",
    "make_idea_fusion_protocol",
]
