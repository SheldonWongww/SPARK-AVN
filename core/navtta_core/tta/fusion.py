#!/usr/bin/env python3
"""Reusable :class:`IDEAFusionProtocol` bindings for real navigation policies.

IDEA needs a model-specific way to (a) inject a soft prompt into a fusion
transformer's token sequence, (b) read per-layer pooled statistics, and (c)
read prompt-free per-layer features connected to the decision logits.  The
core algorithm never touches the model; these helpers realise the seam by
*calling frozen sub-modules and attaching temporary forward hooks* -- the
underlying model ``forward`` is never edited or its parameters unfrozen.

``TransformerFusionProtocol`` targets any ``torch.nn.Transformer`` (the AVN
SMT/ENMuS state encoders both use one) and relies only on the stable seq-first
``nn.Transformer`` contract, so it is task-agnostic and lives in ``core``.
Task packages provide a thin builder that locates the transformer module and a
``forward_logits`` callable; VLN's cross-modal stack (a ``ModuleList`` rather
than an ``nn.Transformer``) is handled by its own protocol on the task side.
"""
import contextlib

import torch

from .idea import IDEAFusionProtocol


def pool_tokens_to_stats(tokens, valid_mask=None, eps=1e-6):
    """Pool ``[S, C]`` tokens over the token dimension into ``(mu, sigma)``.

    ``valid_mask`` (``[S]`` bool, True = keep) restricts pooling to real tokens
    (dropping prompt rows and padded positions).  Statistics are pooled over the
    candidate-node dimension exactly as IDEA specifies for the streaming setting.
    """
    if valid_mask is not None:
        tokens = tokens[valid_mask]
    if tokens.shape[0] == 0:
        raise ValueError("cannot pool statistics over zero tokens")
    mu = tokens.mean(dim=0)
    if tokens.shape[0] < 2:
        sigma = torch.zeros_like(mu)
    else:
        var = tokens.var(dim=0, unbiased=True).clamp_min(0.0)
        sigma = torch.sqrt(var + eps)
    return mu, sigma


class SourceStatisticsAccumulator:
    """Running per-layer ``(mu, sigma)`` anchor for the source-domain space.

    IDEA precomputes ``Gamma_S`` offline from a small source subset.  When such
    an artifact is supplied it is used verbatim; otherwise this accumulator
    captures prompt-free statistics over the first ``warmup_steps`` streamed
    observations and then freezes, which is a documented online proxy for the
    offline source anchor (the frozen policy is the source model).
    """

    def __init__(self, num_layers, warmup_steps=64, precomputed=None):
        self.num_layers = int(num_layers)
        self.warmup_steps = int(warmup_steps)
        self._precomputed = precomputed
        self._sum = None
        self._count = 0
        self._frozen = precomputed is not None

    @property
    def ready(self):
        return self._frozen or self._count > 0

    def observe(self, layer_stats):
        if self._frozen:
            return
        vector = [
            torch.cat([mu.detach(), sigma.detach()]) for mu, sigma in layer_stats
        ]
        if self._sum is None:
            self._sum = [value.clone() for value in vector]
        else:
            for index, value in enumerate(vector):
                self._sum[index] += value
        self._count += 1
        if self._count >= self.warmup_steps:
            self._frozen = True

    def statistics(self, feature_dim, device, dtype):
        if self._precomputed is not None:
            return [
                (mu.to(device=device, dtype=dtype), sigma.to(device=device, dtype=dtype))
                for mu, sigma in self._precomputed
            ]
        if self._sum is None:
            raise RuntimeError(
                "source statistics requested before any prompt-free observation"
            )
        stats = []
        for value in self._sum:
            mean = (value / max(1, self._count)).to(device=device, dtype=dtype)
            stats.append((mean[:feature_dim], mean[feature_dim:]))
        return stats


class TransformerFusionProtocol(IDEAFusionProtocol):
    """IDEA protocol for a policy whose fusion core is an ``nn.Transformer``.

    Args:
        transformer: the ``nn.Transformer`` module to hook (its ``encoder``
            self-attends over the fused token sequence).
        forward_logits: ``callable(policy_inputs) -> logits[1, A]`` that runs the
            full (frozen) policy forward.  The transformer prompt is injected via
            a hook, so this callable is prompt-agnostic.  It must stay on the
            autograd graph when grad is enabled.
        feature_dim: fused feature dimension ``C``.
        num_layers: number of aligned encoder layers ``M``; defaults to the
            transformer's encoder depth.
        source_stats / source_warmup_steps: offline anchor or warmup fallback.
    """

    def __init__(
        self,
        transformer,
        forward_logits,
        feature_dim,
        num_layers=None,
        source_stats=None,
        source_warmup_steps=64,
    ):
        self.transformer = transformer
        self.forward_logits = forward_logits
        self._feature_dim = int(feature_dim)
        encoder_layers = list(transformer.encoder.layers)
        if not encoder_layers:
            raise ValueError("transformer encoder has no layers to align")
        self._num_layers = (
            len(encoder_layers) if num_layers is None else int(num_layers)
        )
        if self._num_layers > len(encoder_layers):
            raise ValueError(
                "requested {} align layers but encoder has {}".format(
                    self._num_layers, len(encoder_layers)
                )
            )
        # Align the last M encoder layers (closest to the decision head).
        self._aligned_layers = encoder_layers[-self._num_layers:]
        self._source = SourceStatisticsAccumulator(
            self._num_layers, source_warmup_steps, precomputed=source_stats
        )
        # Per-call transient state set by the injection/capture context.
        self._active_prompt = None
        self._num_prompt = 0
        self._captured = {}

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def num_layers(self):
        return self._num_layers

    # -- hooks --------------------------------------------------------------
    def _prompt_pre_hook(self, module, args, kwargs):
        """Prepend the active prompt to the encoder source tokens and masks."""
        if self._active_prompt is None:
            return None
        prompt = self._active_prompt
        length = prompt.shape[0]
        args = list(args)
        # nn.Transformer.forward(src, tgt, ...) -- src is seq-first [S, bs, C].
        src = args[0] if args else kwargs["src"]
        batch = src.shape[1]
        prompt_tokens = prompt.unsqueeze(1).expand(length, batch, prompt.shape[-1])
        new_src = torch.cat([prompt_tokens.to(src.dtype), src], dim=0)
        if args:
            args[0] = new_src
        else:
            kwargs["src"] = new_src

        # Prepend L valid (non-padding) columns to any source/memory key masks.
        for key in ("src_key_padding_mask", "memory_key_padding_mask"):
            mask = kwargs.get(key)
            if mask is not None:
                pad = torch.zeros(
                    mask.shape[0], length, dtype=mask.dtype, device=mask.device
                )
                kwargs[key] = torch.cat([pad, mask], dim=1)
        return tuple(args), kwargs

    def _make_layer_hook(self, index):
        def hook(module, inputs, output):
            tokens = output[0] if isinstance(output, tuple) else output
            self._captured[index] = tokens
        return hook

    @contextlib.contextmanager
    def _instrumented(self, prompt):
        self._active_prompt = prompt
        self._num_prompt = 0 if prompt is None else int(prompt.shape[0])
        self._captured = {}
        handles = [
            self.transformer.register_forward_pre_hook(
                self._prompt_pre_hook, with_kwargs=True
            )
        ]
        for index, layer in enumerate(self._aligned_layers):
            handles.append(layer.register_forward_hook(self._make_layer_hook(index)))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            self._active_prompt = None

    def _captured_layer_stats(self):
        if len(self._captured) != self._num_layers:
            raise RuntimeError(
                "captured {} aligned layers, expected {}; the transformer "
                "encoder may not have run".format(
                    len(self._captured), self._num_layers
                )
            )
        stats = []
        for index in range(self._num_layers):
            tokens = self._captured[index]
            # Squeeze the streaming batch dimension: [S, 1, C] -> [S, C].
            tokens = tokens[:, 0, :] if tokens.dim() == 3 else tokens
            valid = torch.ones(
                tokens.shape[0], dtype=torch.bool, device=tokens.device
            )
            if self._num_prompt:
                valid[: self._num_prompt] = False
            stats.append(pool_tokens_to_stats(tokens, valid_mask=valid))
        return stats

    def _captured_layer_features(self):
        if len(self._captured) != self._num_layers:
            raise RuntimeError(
                "captured {} aligned layers, expected {}; the transformer "
                "encoder may not have run".format(
                    len(self._captured), self._num_layers
                )
            )
        # Return the exact captured tensors so they remain on the autograd path
        # from the logits (a fresh slice would be a sibling branch that
        # ``autograd.grad`` cannot reach).  Fisher only needs the squared-grad
        # magnitude, which is invariant to the extra prompt-free token layout;
        # the prompt-free pass has no prompt rows to exclude.
        return [self._captured[index] for index in range(self._num_layers)]

    # -- IDEAFusionProtocol -------------------------------------------------
    def source_statistics(self, device, dtype):
        return self._source.statistics(self._feature_dim, device, dtype)

    def fused_forward(self, policy_inputs, prompt):
        with self._instrumented(prompt):
            logits = self.forward_logits(policy_inputs)
            layer_stats = self._captured_layer_stats()
        if prompt is None:
            # A prompt-free pass is exactly the source-anchor observation.
            self._source.observe(layer_stats)
        return layer_stats, logits

    def fisher_forward(self, policy_inputs):
        with self._instrumented(None):
            logits = self.forward_logits(policy_inputs)
            features = self._captured_layer_features()
        return features, logits


__all__ = [
    "TransformerFusionProtocol",
    "SourceStatisticsAccumulator",
    "pool_tokens_to_stats",
]
