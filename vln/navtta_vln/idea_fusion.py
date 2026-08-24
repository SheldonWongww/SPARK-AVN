"""Fail-closed IDEA fusion bindings for VLN policies.

Evaluation requires a digest-pinned offline source-statistics artifact.  The
same bindings expose ``collect_source_step`` in an explicit collection mode so
task runners can accumulate global moments over 128 source-training
trajectories without ever learning an anchor from target evaluation data.

DUET and GOAT deliberately use different bindings: DUET exposes ``x_layers``
whose visual stream is argument 2, while GOAT exposes ``crossattention`` whose
visual/query stream is argument 0.  HAMT navigation bypasses
``LxmertEncoder.forward`` and iterates ``encoder.x_layers`` directly, so HAMT is
instrumented at the layer stack rather than at the never-called encoder.
"""
import contextlib

import torch

from navtta_core.tta import CrossmodalPromptInjector, IDEAFusionProtocol
from navtta_core.tta.fusion import (
    SourceStatisticsArtifact,
    pool_tokens_to_stats,
)

from .discrete_tta import (
    _graph_decision_features,
    _sanitize_action_logits,
    discrete_forward_policy,
)


def _infer_hidden_size(inner):
    config = getattr(inner, "config", None)
    if config is not None and hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    return 768


class _OfflineSourceProtocol(IDEAFusionProtocol):
    """Common artifact and offline-collection state for VLN protocols."""

    def _configure_source(
        self,
        feature_dim,
        num_layers,
        source_stats_path,
        source_stats_sha256,
        expected_source_trajectories,
        expected_source_provenance,
        source_collection,
    ):
        self._feature_dim = int(feature_dim)
        self._num_layers = int(num_layers)
        if self._feature_dim < 1 or self._num_layers < 1:
            raise ValueError("IDEA protocol dimensions must be positive")
        self.source_collection = bool(source_collection)
        if self.source_collection:
            if source_stats_path is not None or source_stats_sha256 is not None:
                raise ValueError("source collection mode cannot consume an artifact")
            self._source = None
        else:
            if source_stats_path is None or source_stats_sha256 is None:
                raise ValueError(
                    "VLN IDEA evaluation requires source_stats_path and "
                    "source_stats_sha256; target warmup is forbidden"
                )
            self._source = SourceStatisticsArtifact.load(
                source_stats_path,
                source_stats_sha256,
                expected_feature_dim=self._feature_dim,
                expected_num_layers=self._num_layers,
                expected_trajectory_count=expected_source_trajectories,
                expected_provenance=expected_source_provenance,
            )

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def num_layers(self):
        return self._num_layers

    @property
    def source_statistics_metadata(self):
        if self._source is None:
            return {"mode": "offline_collection"}
        return {
            "mode": "offline_artifact",
            "schema": "navtta.idea.source_statistics",
            "moment_estimator": "global_sum_sumsq_count_sample_std",
            "sha256": self._source.sha256,
            "trajectory_count": self._source.provenance["trajectory_count"],
            "provenance": dict(self._source.provenance),
        }

    def source_statistics(self, device, dtype):
        if self._source is None:
            raise RuntimeError("source collection mode has no evaluation anchor")
        return self._source.statistics(device, dtype)


class _InjectedIDEAProtocol(_OfflineSourceProtocol):
    """IDEA protocol for an encoder supported by CrossmodalPromptInjector."""

    def __init__(
        self,
        model,
        injector,
        feature_dim,
        num_layers,
        valid_mask_key,
        source_stats_path=None,
        source_stats_sha256=None,
        expected_source_trajectories=128,
        expected_source_provenance=None,
        source_collection=False,
    ):
        self.model = model
        self.injector = injector
        self.valid_mask_keys = (
            tuple(valid_mask_key)
            if isinstance(valid_mask_key, (tuple, list))
            else (str(valid_mask_key),)
        )
        self._configure_source(
            feature_dim,
            num_layers,
            source_stats_path,
            source_stats_sha256,
            expected_source_trajectories,
            expected_source_provenance,
            source_collection,
        )

    def _forward(self, policy_inputs):
        return discrete_forward_policy(self.model, policy_inputs)

    def _valid_mask(self, policy_inputs):
        model_inputs = policy_inputs.get("model_inputs", policy_inputs)
        selected_key = next(
            (key for key in self.valid_mask_keys if model_inputs.get(key) is not None),
            self.valid_mask_keys[0],
        )
        mask = model_inputs.get(selected_key)
        if not torch.is_tensor(mask) or mask.ndim != 2:
            raise ValueError(
                "IDEA binding requires a [batch,tokens] {} mask".format(
                    "/".join(self.valid_mask_keys)
                )
            )
        return mask.bool()

    def _captured(self):
        if len(self.injector._captured) < self._num_layers:
            raise RuntimeError(
                "captured {} fusion layers, expected {}".format(
                    len(self.injector._captured), self._num_layers
                )
            )
        indices = sorted(self.injector._captured)[-self._num_layers:]
        return [self.injector._captured[index] for index in indices]

    def _stats(self, policy_inputs):
        valid = self._valid_mask(policy_inputs)
        if self.injector._num_prompt:
            valid = torch.cat([
                torch.zeros(
                    valid.shape[0], self.injector._num_prompt,
                    dtype=torch.bool, device=valid.device,
                ),
                valid,
            ], dim=1)
        stats = []
        for tokens in self._captured():
            if tokens.ndim != 3 or tuple(tokens.shape[:2]) != tuple(valid.shape):
                raise ValueError(
                    "captured VLN tokens do not match the original validity mask"
                )
            stats.append(pool_tokens_to_stats(tokens, valid))
        return stats

    @contextlib.contextmanager
    def _fisher_input_probe(self):
        """Make the frozen encoder's visual input an autograd leaf."""
        def pre_hook(module, args, kwargs):
            args = list(args)
            visual = self.injector._get(args, kwargs, self.injector.visual_arg)
            if visual is None:
                raise RuntimeError("IDEA Fisher probe could not find visual tokens")
            visual = visual.detach().requires_grad_(True)
            self.injector._set(args, kwargs, self.injector.visual_arg, visual)
            return tuple(args), kwargs

        handle = self.injector.encoder.register_forward_pre_hook(
            pre_hook, with_kwargs=True
        )
        try:
            yield
        finally:
            handle.remove()

    def fused_forward(self, policy_inputs, prompt):
        if self.source_collection:
            raise RuntimeError("collection protocol cannot run IDEA evaluation")
        with self.injector.instrument(prompt):
            _, logits = self._forward(policy_inputs)
            stats = self._stats(policy_inputs)
        return stats, logits

    def fisher_forward(self, policy_inputs):
        if self.source_collection:
            raise RuntimeError("collection protocol cannot run IDEA evaluation")
        # Register the probe before the capture context.  Prompt-free capture has
        # no injector pre-hook, and the probe makes every aligned activation a
        # differentiable ancestor of the decision logits despite frozen weights.
        with self._fisher_input_probe(), self.injector.instrument(None):
            _, logits = self._forward(policy_inputs)
            features = self._captured()
            valid = self._valid_mask(policy_inputs)
        if not all(feature.requires_grad for feature in features):
            raise RuntimeError("VLN IDEA Fisher activations are disconnected")
        return features, logits, [valid] * self._num_layers

    def collect_source_step(self, policy_inputs, accumulator):
        if not self.source_collection:
            raise RuntimeError("collect_source_step requires source_collection=True")
        with torch.no_grad(), self.injector.instrument(None):
            self._forward(policy_inputs)
            features = self._captured()
            valid = self._valid_mask(policy_inputs)
            accumulator.update(features, valid)


class GraphIDEAProtocol(_InjectedIDEAProtocol):
    """DUET or GOAT local-candidate IDEA protocol.

    The binding is selected by the actual stack name.  GOAT is never treated as
    a DUET ``x_layers`` encoder: its visual query and language key/value argument
    ordering is the reverse of DUET's.
    """

    def __init__(
        self,
        model,
        num_layers=None,
        source_stats_path=None,
        source_stats_sha256=None,
        expected_source_trajectories=128,
        expected_source_provenance=None,
        source_collection=False,
    ):
        inner = getattr(model, "vln_bert", None)
        branch = getattr(inner, "local_encoder", None)
        encoder = getattr(branch, "encoder", None)
        if encoder is None:
            raise ValueError("graph IDEA protocol requires a local fusion encoder")
        if hasattr(encoder, "x_layers"):
            self.binding = "duet_x_layers"
            layers = list(encoder.x_layers)
            injector = CrossmodalPromptInjector(
                encoder,
                layers,
                visual_arg=2,
                mask_arg=3,
                sprel_arg="graph_sprels",
                mask_valid_value=1,
            )
        elif hasattr(encoder, "crossattention"):
            self.binding = "goat_crossattention"
            layers = list(encoder.crossattention)
            injector = CrossmodalPromptInjector(
                encoder,
                layers,
                # GOAT CrossmodalEncoder.forward(q_visual, q_mask,
                # language_kv, language_mask, graph_sprels=None).
                visual_arg=0,
                mask_arg=1,
                sprel_arg="graph_sprels",
                mask_valid_value=1,
                layer_output_index=0,
            )
        else:
            raise ValueError(
                "graph IDEA supports DUET x_layers or GOAT crossattention only"
            )
        resolved = len(layers) if num_layers is None else int(num_layers)
        if resolved < 1 or resolved > len(layers):
            raise ValueError("invalid graph IDEA layer count")
        super().__init__(
            model,
            injector,
            _infer_hidden_size(inner),
            resolved,
            ("vp_nav_masks", "vp_masks"),
            source_stats_path,
            source_stats_sha256,
            expected_source_trajectories,
            expected_source_provenance,
            source_collection,
        )
        self.last_prompted_object_logits = None

    def _forward(self, policy_inputs):
        model_inputs = policy_inputs["model_inputs"]
        outputs = self.model("navigation", model_inputs)
        fusion = policy_inputs["fusion"]
        if fusion == "global":
            raise ValueError(
                "graph IDEA is bound to the local candidate branch and cannot "
                "claim prompt-conditioned global-only navigation"
            )
        key = (
            "local_logits" if fusion == "local" else
            "global_logits" if fusion == "global" else "fused_logits"
        )
        logits = outputs[key]
        self.last_prompted_object_logits = outputs.get("obj_logits")
        invalid = policy_inputs.get("invalid_action_mask")
        if invalid is not None:
            if not torch.is_tensor(invalid) or invalid.shape != logits.shape:
                raise ValueError("graph IDEA invalid-action mask shape mismatch")
            logits = logits.masked_fill(invalid.bool(), -float("inf"))
        return _graph_decision_features(outputs), _sanitize_action_logits(logits)

    @property
    def prompted_object_logits(self):
        """Latest prompt-conditioned DUET/GOAT object logits, if emitted."""
        return self.last_prompted_object_logits


class _HAMTDirectInjector:
    """Prompt the x_layers that HAMT navigation invokes directly."""

    def __init__(self, layers):
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("HAMT IDEA requires nonempty encoder.x_layers")
        self.captured = {}
        self.num_prompt = 0
        self.hist_len = 0
        self.prompt = None
        self.fisher_probe = False

    def _first_pre_hook(self, module, args, kwargs):
        args = list(args)
        if len(args) < 4:
            raise RuntimeError("HAMT x_layer binding requires positional inputs")
        visual = args[2]
        mask = args[3]
        if self.fisher_probe:
            visual = visual.detach().requires_grad_(True)
        if self.prompt is not None:
            prompt = self.prompt.to(device=visual.device, dtype=visual.dtype)
            prompt = prompt.unsqueeze(0).expand(visual.shape[0], -1, -1)
            visual = torch.cat([
                visual[:, :self.hist_len], prompt, visual[:, self.hist_len:]
            ], dim=1)
            if mask is None or mask.ndim != 4:
                raise ValueError("HAMT IDEA requires additive [B,1,1,N] mask")
            keep = torch.zeros(
                mask.shape[0], mask.shape[1], mask.shape[2], self.num_prompt,
                dtype=mask.dtype, device=mask.device,
            )
            mask = torch.cat([
                mask[..., :self.hist_len], keep, mask[..., self.hist_len:]
            ], dim=-1)
        args[2], args[3] = visual, mask
        return tuple(args), kwargs

    def _layer_hook(self, index):
        def hook(module, inputs, output):
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError("HAMT x_layer must return (language, visual)")
            visual = output[1]
            self.captured[index] = visual
            if index == len(self.layers) - 1 and self.num_prompt:
                stripped = torch.cat([
                    visual[:, :self.hist_len],
                    visual[:, self.hist_len + self.num_prompt:],
                ], dim=1)
                return (output[0], stripped) + tuple(output[2:])
            return None
        return hook

    @contextlib.contextmanager
    def instrument(self, prompt, hist_len, fisher_probe=False):
        self.prompt = prompt
        self.num_prompt = 0 if prompt is None else int(prompt.shape[0])
        self.hist_len = int(hist_len)
        self.fisher_probe = bool(fisher_probe)
        self.captured = {}
        handles = [self.layers[0].register_forward_pre_hook(
            self._first_pre_hook, with_kwargs=True
        )]
        handles.extend(
            layer.register_forward_hook(self._layer_hook(index))
            for index, layer in enumerate(self.layers)
        )
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            self.prompt = None
            self.fisher_probe = False


class HAMTIDEAProtocol(_OfflineSourceProtocol):
    """HAMT binding at the directly invoked ``encoder.x_layers`` stack."""

    def __init__(
        self,
        model,
        num_layers=None,
        source_stats_path=None,
        source_stats_sha256=None,
        expected_source_trajectories=128,
        expected_source_provenance=None,
        source_collection=False,
    ):
        self.model = model
        inner = getattr(model, "vln_bert", None)
        encoder = getattr(inner, "encoder", None)
        layers = list(getattr(encoder, "x_layers", ()))
        resolved = len(layers) if num_layers is None else int(num_layers)
        if resolved != len(layers):
            # Prompt stripping must happen after the real final layer.  We still
            # capture only the requested suffix below, but instrument all layers.
            if resolved < 1 or resolved > len(layers):
                raise ValueError("invalid HAMT IDEA layer count")
        self._capture_layers = resolved
        self.injector = _HAMTDirectInjector(layers)
        self.binding = "hamt_direct_x_layers"
        self.last_prompted_object_logits = None
        self._configure_source(
            _infer_hidden_size(inner),
            resolved,
            source_stats_path,
            source_stats_sha256,
            expected_source_trajectories,
            expected_source_provenance,
            source_collection,
        )

    def _forward(self, policy_inputs):
        family = policy_inputs["family"]
        if family != "hamt_reverie":
            self.last_prompted_object_logits = None
            return discrete_forward_policy(self.model, policy_inputs)
        replay_inputs = dict(policy_inputs["model_inputs"])
        replay_inputs["return_states"] = True
        outputs = self.model(**replay_inputs)
        self.last_prompted_object_logits = outputs["obj_logits"]
        # Preserve the released HAMT-REVERIE decision rule exactly.  Its agent
        # appends the argmax *object index* (the second torch.max return value)
        # as the final action score.  Replacing it with the maximum value makes
        # even prompt-free IDEA choose a different navigation action than
        # Source, invalidating the required zero-prompt parity check.
        object_stop = outputs["obj_logits"].max(dim=1).indices.unsqueeze(1)
        logits = torch.cat([outputs["act_logits"], object_stop], dim=1)
        invalid = policy_inputs.get("invalid_action_mask")
        if invalid is not None:
            if not torch.is_tensor(invalid) or invalid.shape != logits.shape:
                raise ValueError("HAMT IDEA invalid-action mask shape mismatch")
            logits = logits.masked_fill(invalid.bool(), -float("inf"))
        return outputs["states"], _sanitize_action_logits(logits)

    @property
    def prompted_object_logits(self):
        """Latest prompt-conditioned REVERIE grounding logits, if applicable."""
        return self.last_prompted_object_logits

    def _layout(self, policy_inputs):
        model_inputs = policy_inputs.get("model_inputs", policy_inputs)
        hist = model_inputs.get("hist_embeds")
        if torch.is_tensor(hist) and hist.ndim == 3:
            hist_len = int(hist.shape[1])
        elif isinstance(hist, (list, tuple)) and hist and all(
            torch.is_tensor(item) for item in hist
        ):
            # Official HAMT wrappers stack their per-step history list on dim 1
            # immediately before directly iterating encoder.x_layers.
            hist_len = len(hist)
        else:
            raise ValueError("HAMT IDEA requires tensor/list model_inputs['hist_embeds']")
        observation_valid = model_inputs.get("ob_masks")
        if not torch.is_tensor(observation_valid) or observation_valid.ndim != 2:
            raise ValueError("HAMT IDEA requires model_inputs['ob_masks']")
        navigation_types = model_inputs.get("ob_nav_types")
        if navigation_types is not None:
            if (
                not torch.is_tensor(navigation_types)
                or navigation_types.shape != observation_valid.shape
            ):
                raise ValueError("HAMT IDEA ob_nav_types shape must match ob_masks")
            observation_valid = observation_valid.bool() & navigation_types.ne(0)
        object_valid = model_inputs.get("obj_masks")
        if object_valid is not None:
            if not torch.is_tensor(object_valid) or object_valid.ndim != 2:
                raise ValueError("HAMT IDEA obj_masks must have shape [batch,tokens]")
            if object_valid.shape[0] != observation_valid.shape[0]:
                raise ValueError("HAMT IDEA object/observation batch mismatch")
            object_len = int(object_valid.shape[1])
        else:
            object_len = 0
        return hist_len, observation_valid.bool(), object_len

    def _feature_valid_mask(self, policy_inputs, prompt_length=0):
        """Select only navigable observation tokens from HAMT's visual stream.

        HAMT's cross-modal sequence is ``history + prompt + observation +
        object`` for REVERIE and has no object suffix for R2R.  IDEA Eq. (41)
        and Eq. (7) are defined over navigable candidate nodes, so history,
        prompts, padded panoramas and object-grounding tokens are excluded.
        """
        hist_len, observation_valid, object_len = self._layout(policy_inputs)
        batch = observation_valid.shape[0]
        device = observation_valid.device
        return hist_len, torch.cat([
            torch.zeros(batch, hist_len, dtype=torch.bool, device=device),
            torch.zeros(batch, int(prompt_length), dtype=torch.bool, device=device),
            observation_valid,
            torch.zeros(batch, object_len, dtype=torch.bool, device=device),
        ], dim=1)

    def _features(self):
        if len(self.injector.captured) != len(self.injector.layers):
            raise RuntimeError("HAMT IDEA direct x_layers were not all executed")
        indices = sorted(self.injector.captured)[-self._capture_layers:]
        return [self.injector.captured[index] for index in indices]

    def _stats(self, policy_inputs):
        _, valid = self._feature_valid_mask(
            policy_inputs, prompt_length=self.injector.num_prompt
        )
        return [pool_tokens_to_stats(tokens, valid) for tokens in self._features()]

    def fused_forward(self, policy_inputs, prompt):
        if self.source_collection:
            raise RuntimeError("collection protocol cannot run IDEA evaluation")
        hist_len, _, _ = self._layout(policy_inputs)
        with self.injector.instrument(prompt, hist_len):
            _, logits = self._forward(policy_inputs)
            stats = self._stats(policy_inputs)
        return stats, logits

    def fisher_forward(self, policy_inputs):
        if self.source_collection:
            raise RuntimeError("collection protocol cannot run IDEA evaluation")
        hist_len, _, _ = self._layout(policy_inputs)
        with self.injector.instrument(None, hist_len, fisher_probe=True):
            _, logits = self._forward(policy_inputs)
            features = self._features()
            _, valid = self._feature_valid_mask(policy_inputs)
        if not all(feature.requires_grad for feature in features):
            raise RuntimeError("HAMT IDEA Fisher activations are disconnected")
        return features, logits, [valid] * self._num_layers

    def collect_source_step(self, policy_inputs, accumulator):
        if not self.source_collection:
            raise RuntimeError("collect_source_step requires source_collection=True")
        hist_len, _, _ = self._layout(policy_inputs)
        with torch.no_grad(), self.injector.instrument(None, hist_len):
            self._forward(policy_inputs)
            features = self._features()
            _, valid = self._feature_valid_mask(policy_inputs)
            accumulator.update(features, valid)


class ContinuousIDEAProtocol(_InjectedIDEAProtocol):
    """ETPNav global-map or BEVBert local-BEV IDEA protocol."""

    def __init__(
        self,
        decision_model,
        forward_logits,
        variant,
        num_layers=None,
        source_stats_path=None,
        source_stats_sha256=None,
        expected_source_trajectories=128,
        expected_source_provenance=None,
        source_collection=False,
    ):
        variant = str(variant).lower()
        if variant == "etpnav":
            branch_name, valid_mask_key = "global_encoder", "gmap_masks"
        elif variant == "bevbert":
            branch_name = "local_encoder"
            valid_mask_key = ("bev_nav_masks", "bev_masks")
        else:
            raise ValueError("unknown continuous IDEA variant: {}".format(variant))
        branch = getattr(decision_model, branch_name, None)
        encoder = getattr(branch, "encoder", None)
        layers = list(getattr(encoder, "x_layers", ()))
        if not layers:
            raise ValueError(
                "continuous IDEA requires {}.encoder.x_layers".format(branch_name)
            )
        resolved = len(layers) if num_layers is None else int(num_layers)
        if resolved < 1 or resolved > len(layers):
            raise ValueError("invalid continuous IDEA layer count")
        injector = CrossmodalPromptInjector(
            encoder,
            layers,
            visual_arg=2,
            mask_arg=3,
            sprel_arg="graph_sprels",
            mask_valid_value=1,
        )
        self._bound_model = decision_model
        self._bound_forward = forward_logits
        self.variant = variant
        self.binding = "{}_{}".format(variant, branch_name)
        super().__init__(
            decision_model,
            injector,
            _infer_hidden_size(decision_model),
            resolved,
            valid_mask_key,
            source_stats_path,
            source_stats_sha256,
            expected_source_trajectories,
            expected_source_provenance,
            source_collection,
        )

    def _forward(self, policy_inputs):
        return self._bound_forward(self._bound_model, policy_inputs)


def make_idea_fusion_protocol(
    model,
    family,
    num_layers=None,
    source_stats_path=None,
    source_stats_sha256=None,
    expected_source_trajectories=128,
    expected_source_provenance=None,
    source_collection=False,
):
    """Build a discrete VLN IDEA protocol for evaluation or offline collection."""
    kwargs = dict(
        num_layers=num_layers,
        source_stats_path=source_stats_path,
        source_stats_sha256=source_stats_sha256,
        expected_source_trajectories=expected_source_trajectories,
        expected_source_provenance=expected_source_provenance,
        source_collection=source_collection,
    )
    family = str(family)
    if family == "graph":
        return GraphIDEAProtocol(model, **kwargs)
    if family in ("hamt_r2r", "hamt_reverie"):
        return HAMTIDEAProtocol(model, **kwargs)
    raise ValueError("no IDEA fusion protocol for family {!r}".format(family))


__all__ = [
    "ContinuousIDEAProtocol",
    "GraphIDEAProtocol",
    "HAMTIDEAProtocol",
    "make_idea_fusion_protocol",
]
