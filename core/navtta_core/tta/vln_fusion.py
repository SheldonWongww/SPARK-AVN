#!/usr/bin/env python3
"""Prepend-and-strip prompt injection for VLN cross-modal fusion stacks.

VLN policies (DUET/HAMT/GOAT and the continuous ETPNav/BEVBert) fuse a set of
candidate visual tokens against the instruction with a stack of cross-modal
layers, and the per-token fused embeddings are *themselves* the candidate
actions.  IDEA prescribes prepending ``L`` soft-prompt tokens to the visual
token sequence and aligning the fusion layers' statistics.  Because the tokens
double as actions, a naive prepend would change the action count and collide
with the node masks (and, for DUET, the ``graph_sprels`` spatial-bias tensor).

``CrossmodalPromptInjector`` performs the paper's injection safely by wrapping a
cross-modal encoder module with a forward hook that:

* prepends ``L`` prompt rows to the visual-token argument,
* pads the visual key/pad mask with ``L`` valid columns,
* pads any square spatial-bias tensor (``graph_sprels``) with ``L`` zero-bias
  rows/columns,
* captures each aligned layer's fused tokens for the statistics/Fisher reads,
* strips the ``L`` prompt rows from the returned embeddings,

so everything downstream (logits, node masks, decision features) sees the
original action dimension.  The wrapped model's ``forward`` is never edited and
no parameters are unfrozen.

The mechanics live in ``core`` and are unit-tested against a mock encoder; the
task packages supply thin adapters that locate the encoder module(s), the
visual/mask/bias argument positions, and a ``forward_logits`` callable.
"""
import contextlib

import torch

from .fusion import pool_tokens_to_stats


class CrossmodalPromptInjector:
    """Inject a soft prompt into one cross-modal encoder via a forward hook.

    Args:
        encoder: the ``nn.Module`` whose ``forward`` fuses visual tokens against
            language (e.g. DUET ``CrossmodalEncoder``, HAMT ``LxmertEncoder``).
        layers: ordered list of the aligned inner layer modules whose outputs
            provide per-layer statistics (e.g. ``encoder.x_layers``).
        visual_arg: name (str) or positional index (int) of the visual-token
            argument in the encoder ``forward`` -- shape ``[batch, N, C]``.
        mask_arg: name/index of the visual key-pad mask argument, or ``None``.
            The mask is padded with ``L`` "keep" entries; ``mask_valid_value``
            sets what "keep" means for this model's mask convention.
        sprel_arg: name/index of a square spatial-bias argument shaped
            ``[batch, heads, N, N]`` (DUET ``graph_sprels``), or ``None``.
        mask_valid_value: the value that marks a *valid/keep* position in the
            visual mask (DUET uses 1 in a boolean keep-mask).
        layer_output_index: if a layer returns a tuple, the index of the token
            tensor; ``None`` means the layer returns the tensor directly.
    """

    def __init__(
        self,
        encoder,
        layers,
        visual_arg,
        mask_arg=None,
        sprel_arg=None,
        mask_valid_value=1,
        layer_output_index=None,
    ):
        self.encoder = encoder
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("cross-modal injector needs at least one layer")
        self.visual_arg = visual_arg
        self.mask_arg = mask_arg
        self.sprel_arg = sprel_arg
        self.mask_valid_value = mask_valid_value
        self.layer_output_index = layer_output_index
        self._active_prompt = None
        self._num_prompt = 0
        self._captured = {}

    # -- argument helpers ---------------------------------------------------
    @staticmethod
    def _get(args, kwargs, key):
        if isinstance(key, int):
            return args[key] if key < len(args) else None
        return kwargs.get(key)

    @staticmethod
    def _set(args, kwargs, key, value):
        if isinstance(key, int):
            args[key] = value
        else:
            kwargs[key] = value

    def _encoder_pre_hook(self, module, args, kwargs):
        if self._active_prompt is None:
            return None
        prompt = self._active_prompt
        length = prompt.shape[0]
        args = list(args)

        visual = self._get(args, kwargs, self.visual_arg)
        if visual is None:
            raise RuntimeError("cross-modal injector could not find visual tokens")
        batch = visual.shape[0]
        prompt_tokens = prompt.unsqueeze(0).expand(batch, length, prompt.shape[-1])
        new_visual = torch.cat([prompt_tokens.to(visual.dtype), visual], dim=1)
        self._set(args, kwargs, self.visual_arg, new_visual)

        if self.mask_arg is not None:
            mask = self._get(args, kwargs, self.mask_arg)
            if mask is not None:
                pad = torch.full(
                    (mask.shape[0], length),
                    self.mask_valid_value,
                    dtype=mask.dtype,
                    device=mask.device,
                )
                self._set(
                    args, kwargs, self.mask_arg,
                    torch.cat([pad, mask], dim=1),
                )

        if self.sprel_arg is not None:
            sprel = self._get(args, kwargs, self.sprel_arg)
            if sprel is not None:
                # Square bias [batch, heads, N, N] -> pad the last two dims with
                # L zero-bias rows/cols so prompt tokens add no spatial prior.
                pad_rows = torch.nn.functional.pad(
                    sprel, (0, 0, length, 0), value=0.0
                )
                new_sprel = torch.nn.functional.pad(
                    pad_rows, (length, 0), value=0.0
                )
                self._set(args, kwargs, self.sprel_arg, new_sprel)

        return tuple(args), kwargs

    def _encoder_post_hook(self, module, args, kwargs, output):
        if self._active_prompt is None:
            return output
        length = self._num_prompt
        # Strip the prepended prompt rows from the returned fused tokens so the
        # action/candidate dimension downstream is unchanged.
        if isinstance(output, tuple):
            first = output[0]
            return (first[:, length:],) + tuple(output[1:])
        return output[:, length:]

    def _make_layer_hook(self, index):
        def hook(module, inputs, output):
            tokens = (
                output if self.layer_output_index is None
                else output[self.layer_output_index]
            )
            self._captured[index] = tokens
        return hook

    @contextlib.contextmanager
    def instrument(self, prompt):
        """Activate injection for one forward; always removes its hooks."""
        self._active_prompt = prompt
        self._num_prompt = 0 if prompt is None else int(prompt.shape[0])
        self._captured = {}
        handles = []
        if prompt is not None:
            handles.append(
                self.encoder.register_forward_pre_hook(
                    self._encoder_pre_hook, with_kwargs=True
                )
            )
            handles.append(
                self.encoder.register_forward_hook(
                    self._encoder_post_hook, with_kwargs=True
                )
            )
        for index, layer in enumerate(self.layers):
            handles.append(layer.register_forward_hook(self._make_layer_hook(index)))
        try:
            yield self
        finally:
            for handle in handles:
                handle.remove()
            self._active_prompt = None

    # -- capture readers ----------------------------------------------------
    def captured_stats(self, num_layers):
        """Return ``num_layers`` ``(mu, sigma)`` pairs pooled over real nodes."""
        if len(self._captured) < num_layers:
            raise RuntimeError(
                "captured {} layers, expected at least {}".format(
                    len(self._captured), num_layers
                )
            )
        indices = sorted(self._captured)[-num_layers:]
        stats = []
        for index in indices:
            tokens = self._captured[index]
            tokens = tokens[0] if tokens.dim() == 3 else tokens  # [N(+L), C]
            valid = torch.ones(
                tokens.shape[0], dtype=torch.bool, device=tokens.device
            )
            if self._num_prompt:
                valid[: self._num_prompt] = False
            stats.append(pool_tokens_to_stats(tokens, valid_mask=valid))
        return stats

    def captured_features(self, num_layers):
        """Return the raw per-layer captured tensors (on the autograd graph)."""
        if len(self._captured) < num_layers:
            raise RuntimeError(
                "captured {} layers, expected at least {}".format(
                    len(self._captured), num_layers
                )
            )
        indices = sorted(self._captured)[-num_layers:]
        return [self._captured[index] for index in indices]


__all__ = ["CrossmodalPromptInjector"]
