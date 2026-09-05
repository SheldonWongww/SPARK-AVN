"""Online TTA binding for StreamVLN's generated action-token policy.

StreamVLN emits one short autoregressive response at each policy call.  The
response contains a Qwen chat prefix followed by an action chunk; the first
action token is one of STOP/UP/LEFT/RIGHT and is the online decision used by
the shared adapters.  We adapt Qwen's final RMSNorm, which keeps replay small:
replay inputs are the detached pre-norm hidden state rather than RGB-D frames
or a language-model computation graph.
"""

import json
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from navtta_core.tta import build_adapter
from navtta_vln.discrete_tta import _adapter_config


ACTION_TEXT = ("STOP", "↑", "←", "→")
FEEDBACK_METHODS = ("feedtta", "atena")


class StreamActionReadout(nn.Module):
    """Small replayable view of the deployed first-action token head."""

    def __init__(self, model, action_token_ids):
        super().__init__()
        # Keep a float32 adaptation copy. Updating a bfloat16 RMSNorm with the
        # deliberately small search LRs would quantize most steps to zero.
        # The selected action is injected into greedy generation by the logits
        # processor, so this readout is the deployed first-action policy.
        self.norm = deepcopy(model.model.norm).float()
        weights = (
            model.lm_head.weight.detach()[list(action_token_ids)].float().clone()
        )
        self.register_buffer("action_weight", weights)

    def forward(self, hidden):
        normalized = self.norm(hidden)
        # Float logits keep entropy/replay numerically stable while gradients
        # still flow to the bfloat16 RMSNorm affine weight.
        return F.linear(normalized.float(), self.action_weight.float())


def _stream_forward_policy(readout, policy_inputs):
    hidden = policy_inputs["pre_norm_hidden"]
    return hidden.float(), readout(hidden)


def _single_token_id(tokenizer, text):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            "StreamVLN action {!r} must be one tokenizer token, got {}".format(
                text, token_ids
            )
        )
    return int(token_ids[0])


class StreamVLNTTAController(object):
    """Bind shared TTA algorithms to StreamVLN action-generation calls."""

    def __init__(self, args, model, tokenizer):
        self.method = str(args.tta_method).lower()
        if self.method == "source":
            raise ValueError("Source evaluation does not construct a TTA adapter")
        if self.method not in ("tent", "fstta", "eam", "feedtta", "atena"):
            raise ValueError("unsupported StreamVLN TTA method: {}".format(
                self.method
            ))
        if args.eval_split == "test" and self.method in FEEDBACK_METHODS:
            raise ValueError(
                "{} requires labeled episode success and cannot run on test"
                .format(self.method.upper())
            )
        self.action_token_ids = tuple(
            _single_token_id(tokenizer, text) for text in ACTION_TEXT
        )
        if len(set(self.action_token_ids)) != len(self.action_token_ids):
            raise ValueError("StreamVLN action tokens must have distinct IDs")

        self.readout = StreamActionReadout(model, self.action_token_ids)
        if self.method == "atena":
            # ATENA preserves pre-existing trainability when binding scope=all.
            # The cloned readout inherits the frozen checkpoint flag, so expose
            # only its lightweight RMSNorm affine parameter.
            self.readout.norm.weight.requires_grad_(True)
        config = _adapter_config(
            args,
            trainable_prefixes=("norm",),
            action_selection="argmax",
        )
        self.adapter = build_adapter(
            self.readout,
            config,
            forward_policy=_stream_forward_policy,
        )
        if self.adapter is None:
            raise RuntimeError("failed to build StreamVLN TTA adapter")
        self.diagnostics_path = Path(args.tta_diagnostics)
        self.episode_count = 0
        self.generation_calls = 0
        self.overridden_first_actions = 0
        self._episode_open = False

    def begin_episode(self):
        if self._episode_open:
            raise RuntimeError("StreamVLN TTA episode already open")
        self.adapter.episode_start()
        self._episode_open = True

    @torch.enable_grad()
    def adapt_first_token(self, hidden):
        """Adapt from one generation call and return its first action."""
        if not self._episode_open:
            raise RuntimeError("StreamVLN TTA step outside an episode")
        if not torch.is_tensor(hidden) or hidden.ndim != 2:
            raise ValueError("StreamVLN action feature must be [batch, hidden]")
        policy_inputs = {"pre_norm_hidden": hidden.detach()}
        if self.method == "eam":
            self.adapter.before_inference(policy_inputs=policy_inputs)
        else:
            self.adapter.before_inference(policy_inputs=policy_inputs)
        source_logits = self.readout(hidden)
        source_action = int(source_logits.detach().argmax(dim=-1).item())
        prepared = self.adapter.prepare_action(
            source_logits, policy_inputs=policy_inputs
        )

        # EAM and ATENA define their own action decision. Other methods retain
        # the action already produced by StreamVLN's native greedy decoder and
        # apply their update only to future generation calls.
        if self.method in ("eam", "atena"):
            executed = int(prepared.detach().argmax(dim=-1).item())
        else:
            executed = source_action
        action = torch.tensor([executed], device=prepared.device)
        self.adapter.adapt(
            prepared,
            action=action,
            features=hidden.float(),
            policy_inputs=policy_inputs,
        )
        self.generation_calls += 1
        if executed != source_action:
            self.overridden_first_actions += 1
        return executed

    def action_token_id(self, action):
        return self.action_token_ids[int(action)]

    def end_episode(self, success):
        if not self._episode_open:
            raise RuntimeError("StreamVLN TTA episode is not open")
        stats = {"success": float(success)} if self.method in FEEDBACK_METHODS else None
        self.adapter.episode_end(stats)
        self._episode_open = False
        self.episode_count += 1
        self.write_diagnostics()

    def write_diagnostics(self):
        diagnostics = self.adapter.diagnostics()
        diagnostics.update({
            "schema": "navtta.streamvln_tta_diagnostics.v1",
            "method": self.method,
            "decision_space": "first_generated_action_token",
            "action_text": list(ACTION_TEXT),
            "action_token_ids": list(self.action_token_ids),
            "adaptation_scope": "float32_qwen_final_rmsnorm_action_readout",
            "generation_calls": self.generation_calls,
            "overridden_first_actions": self.overridden_first_actions,
            "episodes": self.episode_count,
        })
        path = self.diagnostics_path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                diagnostics, stream, indent=2, sort_keys=True,
                ensure_ascii=False, allow_nan=False,
            )
            stream.write("\n")
        os.replace(str(temporary), str(path))
