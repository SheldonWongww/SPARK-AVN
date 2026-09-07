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
from torch.func import functional_call

from navtta_core.tta import build_adapter
from navtta_vln.discrete_tta import _adapter_config


ACTION_TEXT = ("STOP", "↑", "←", "→")
FEEDBACK_METHODS = ("feedtta", "atena")
READOUT_PROTOCOLS = ("legacy_v4", "native_residual")


def add_streamvln_tta_args(parser):
    parser.add_argument(
        "--tta_streamvln_readout_protocol", choices=READOUT_PROTOCOLS,
        default="legacy_v4",
        help=(
            "legacy_v4 preserves the historical FP32 replacement policy; "
            "native_residual adds only the learned FP32 logit difference "
            "to the original decoder scores"
        ),
    )


class StreamActionReadout(nn.Module):
    """Small replayable view of the deployed first-action token head."""

    def __init__(self, model, action_token_ids, protocol="legacy_v4"):
        super().__init__()
        if protocol not in READOUT_PROTOCOLS:
            raise ValueError("unknown StreamVLN readout protocol: {}".format(protocol))
        self.protocol = protocol
        # Keep a float32 adaptation copy. Updating a bfloat16 RMSNorm with the
        # deliberately small search LRs would quantize most steps to zero.
        # The selected action is injected into greedy generation by the logits
        # processor, so this readout is the deployed first-action policy.
        self.norm = deepcopy(model.model.norm).float()
        # Buffers keep the checkpoint affine values out of the optimizer and
        # out of normalization-module selection. Registering a second RMSNorm
        # would otherwise make scope=last_ln select the frozen reference.
        self._initial_norm_names = tuple(dict(self.norm.named_parameters()))
        for index, (_, parameter) in enumerate(self.norm.named_parameters()):
            self.register_buffer(
                "initial_norm_{}".format(index), parameter.detach().clone()
            )
        weights = (
            model.lm_head.weight.detach()[list(action_token_ids)].float().clone()
        )
        self.register_buffer("action_weight", weights)

    def fp32_logits(self, hidden):
        normalized = self.norm(hidden)
        # Both the affine copy and the four-row head use FP32. The backbone
        # remains frozen in its original deployment precision.
        return F.linear(normalized.float(), self.action_weight.float())

    @torch.no_grad()
    def initial_logits(self, hidden):
        parameters = {
            name: getattr(self, "initial_norm_{}".format(index))
            for index, name in enumerate(self._initial_norm_names)
        }
        normalized = functional_call(self.norm, parameters, (hidden,))
        return F.linear(normalized.float(), self.action_weight.float())

    def policy_features(self, hidden):
        if self.protocol == "native_residual":
            # ATENA's self-prediction loss must be able to reach the same
            # affine parameter as its action-policy replay loss.
            return self.norm(hidden).float()
        return hidden.float()

    def forward(self, hidden, native_action_logits=None):
        logits = self.fp32_logits(hidden)
        if self.protocol == "legacy_v4":
            return logits
        if (
            not torch.is_tensor(native_action_logits)
            or native_action_logits.shape != logits.shape
            or not bool(torch.isfinite(native_action_logits).all())
        ):
            raise ValueError(
                "native_residual requires finite native action logits "
                "matching the readout shape"
            )
        # Parentheses matter: adding and then subtracting the full readout
        # can round away bits of the native baseline even at initialization.
        residual = logits - self.initial_logits(hidden)
        return native_action_logits.detach().float() + residual


def _stream_forward_policy(readout, policy_inputs):
    hidden = policy_inputs["pre_norm_hidden"]
    return readout.policy_features(hidden), readout(
        hidden, policy_inputs.get("native_action_logits")
    )


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
        self.readout_protocol = getattr(
            args, "tta_streamvln_readout_protocol", "legacy_v4"
        )
        if self.readout_protocol not in READOUT_PROTOCOLS:
            raise ValueError("unknown StreamVLN readout protocol")
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
        action_tokens = [
            (_single_token_id(tokenizer, text), text) for text in ACTION_TEXT
        ]
        if self.readout_protocol == "native_residual":
            # A full-vocabulary greedy tie selects the smallest token ID.
            # Keep this exact tie order in every adapter, including ATENA's
            # executed-action == policy-argmax assertion.
            action_tokens.sort(key=lambda item: item[0])
        self.action_token_ids = tuple(item[0] for item in action_tokens)
        self.action_text = tuple(item[1] for item in action_tokens)
        if len(set(self.action_token_ids)) != len(self.action_token_ids):
            raise ValueError("StreamVLN action tokens must have distinct IDs")

        self.readout = StreamActionReadout(
            model, self.action_token_ids, self.readout_protocol
        )
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
        self.action_comparisons = self._new_action_comparisons()
        self.parameter_changes = self._new_parameter_changes()
        self.episode_diagnostics = []
        self._initial_parameters = self._parameter_snapshot()
        self._episode_open = False

    @staticmethod
    def _new_action_comparisons():
        return {
            "native_comparisons": 0,
            "native_vs_frozen_fp32_flips": 0,
            "native_vs_adapted_flips": 0,
            "native_vs_prepared_flips": 0,
            "native_vs_executed_flips": 0,
            "frozen_fp32_vs_adapted_fp32_flips": 0,
            "baseline_logit_linf_sum": 0.0,
            "baseline_logit_linf_max": 0.0,
            "residual_logit_linf_sum": 0.0,
            "residual_logit_linf_max": 0.0,
            "native_margin_sum": 0.0,
            "native_margin_min": None,
        }

    @staticmethod
    def _new_parameter_changes():
        return {
            "observed_boundaries": 0,
            "changed_boundaries": 0,
            "changed_elements": 0,
            "max_abs_change": 0.0,
            "by_boundary": {},
        }

    def _parameter_values(self):
        values = {"primary.norm.weight": self.readout.norm.weight}
        if self.method == "eam":
            values["auxiliary.norm.weight"] = self.adapter.aux_model.norm.weight
        if self.method == "fstta":
            values["slow_anchor"] = self.adapter.slow_anchor
        return values

    def _parameter_snapshot(self):
        return {
            name: value.detach().clone()
            for name, value in self._parameter_values().items()
        }

    def _observe_parameter_call(self, boundary, callback):
        before = self._parameter_snapshot()
        output = callback()
        # Episode-end routing is decided inside ATENA. Resolve the diagnostic
        # label afterwards so observing a route never duplicates its policy.
        if callable(boundary):
            boundary = boundary()
        changed = 0
        max_change = 0.0
        for name, value in self._parameter_values().items():
            delta = value.detach() - before[name]
            changed += int(torch.count_nonzero(delta).item())
            max_change = max(max_change, float(delta.abs().max().item()))
        for counters in (self.parameter_changes, self._episode_parameter_changes):
            for target in (
                counters,
                counters["by_boundary"].setdefault(boundary, {
                    "observed_boundaries": 0, "changed_boundaries": 0,
                    "changed_elements": 0, "max_abs_change": 0.0,
                }),
            ):
                target["observed_boundaries"] += 1
                target["changed_boundaries"] += int(changed > 0)
                target["changed_elements"] += changed
                target["max_abs_change"] = max(
                    target["max_abs_change"], max_change
                )
        return output

    def _parameter_drift(self):
        output = {}
        for name, value in self._parameter_values().items():
            initial = self._initial_parameters[name]
            delta = value.detach() - initial
            output[name] = {
                "changed_elements": int(torch.count_nonzero(delta).item()),
                "max_abs_change": float(delta.abs().max().item()),
                "relative_l2": float(
                    (delta.norm() / initial.norm().clamp_min(1e-12)).item()
                ),
            }
        return output

    def begin_episode(self):
        if self._episode_open:
            raise RuntimeError("StreamVLN TTA episode already open")
        self._episode_action_comparisons = self._new_action_comparisons()
        self._episode_parameter_changes = self._new_parameter_changes()
        self._episode_start_calls = self.generation_calls
        self._observe_parameter_call("episode_start", self.adapter.episode_start)
        self._episode_open = True

    @torch.enable_grad()
    def adapt_first_token(
        self, hidden, native_action_logits=None, native_token_id=None
    ):
        """Adapt from one generation call and return its first action."""
        if not self._episode_open:
            raise RuntimeError("StreamVLN TTA step outside an episode")
        if not torch.is_tensor(hidden) or hidden.ndim != 2:
            raise ValueError("StreamVLN action feature must be [batch, hidden]")
        policy_inputs = {"pre_norm_hidden": hidden.detach()}
        if native_action_logits is not None:
            policy_inputs["native_action_logits"] = native_action_logits.detach()
        if self.readout_protocol == "native_residual":
            if native_token_id not in self.action_token_ids:
                raise ValueError("native_residual requires the native action token")
            if native_action_logits is None:
                raise ValueError("native_residual requires native action logits")
            native_argmax = int(native_action_logits.argmax(dim=-1).item())
            if self.action_token_ids[native_argmax] != native_token_id:
                raise ValueError("native action logits do not preserve vocabulary tie order")
        self._observe_parameter_call(
            "before_inference",
            lambda: self.adapter.before_inference(policy_inputs=policy_inputs),
        )
        source_logits = self.readout(hidden, native_action_logits)
        source_action = int(source_logits.detach().argmax(dim=-1).item())
        prepared = self.adapter.prepare_action(
            source_logits, policy_inputs=policy_inputs
        )

        # The decision is fixed before this call's update; it already includes
        # all updates made on earlier generation calls/episodes. EAM supplies
        # its source/auxiliary mixture here.
        if self.method in ("eam", "atena"):
            executed = int(prepared.detach().argmax(dim=-1).item())
        else:
            executed = source_action
        action = torch.tensor([executed], device=prepared.device)
        self._record_action_comparisons(
            hidden, native_action_logits, native_token_id, prepared, executed
        )
        features = self.readout.policy_features(hidden)
        self._observe_parameter_call(
            "adapt",
            lambda: self.adapter.adapt(
                prepared, action=action, features=features,
                policy_inputs=policy_inputs,
            ),
        )
        self.generation_calls += 1
        return executed

    @torch.no_grad()
    def _record_action_comparisons(
        self, hidden, native_logits, native_token_id, prepared, executed
    ):
        if native_logits is None or native_token_id is None:
            return
        native_action = self.action_token_ids.index(int(native_token_id))
        adapted_readout = (
            self.adapter.aux_model if self.method == "eam" else self.readout
        )
        frozen_logits = self.readout.initial_logits(hidden)
        adapted_fp32 = adapted_readout.fp32_logits(hidden)
        adapted_logits = adapted_readout(hidden, native_logits)
        frozen_action = int(frozen_logits.argmax(dim=-1).item())
        prepared_action = int(prepared.detach().argmax(dim=-1).item())
        flips = {
            "native_vs_frozen_fp32_flips": frozen_action != native_action,
            "native_vs_adapted_flips": (
                int(adapted_logits.argmax(dim=-1).item()) != native_action
            ),
            "native_vs_prepared_flips": prepared_action != native_action,
            "native_vs_executed_flips": executed != native_action,
            "frozen_fp32_vs_adapted_fp32_flips": (
                int(adapted_fp32.argmax(dim=-1).item()) != frozen_action
            ),
        }
        baseline_error = float((frozen_logits - native_logits.float()).abs().max().item())
        residual_size = float((adapted_fp32 - frozen_logits).abs().max().item())
        top_two = native_logits.float().topk(2, dim=-1).values
        native_margin = float((top_two[:, 0] - top_two[:, 1]).item())
        for counters in (self.action_comparisons, self._episode_action_comparisons):
            counters["native_comparisons"] += 1
            for key, flipped in flips.items():
                counters[key] += int(flipped)
            for prefix, value in (
                ("baseline_logit_linf", baseline_error),
                ("residual_logit_linf", residual_size),
            ):
                counters[prefix + "_sum"] += value
                counters[prefix + "_max"] = max(counters[prefix + "_max"], value)
            counters["native_margin_sum"] += native_margin
            previous_min = counters["native_margin_min"]
            counters["native_margin_min"] = (
                native_margin if previous_min is None
                else min(previous_min, native_margin)
            )
        self.overridden_first_actions += int(executed != native_action)

    def action_token_id(self, action):
        return self.action_token_ids[int(action)]

    def end_episode(self, success):
        if not self._episode_open:
            raise RuntimeError("StreamVLN TTA episode is not open")
        feedback = {}

        def feedback_provider():
            # The shared ATENA adapter calls this only for queried episodes;
            # its self-labelled path never reads evaluator success.
            value = float(success)
            feedback["feedback_success"] = bool(value >= 0.5)
            return {"success": value}

        stats = feedback_provider if self.method in FEEDBACK_METHODS else None
        if self.method == "atena":
            before_query = self.adapter.query_count
            before_self = self.adapter.self_label_count
        elif self.method == "feedtta":
            before_success = self.adapter.successful_episodes
            before_failure = self.adapter.failed_episodes

        def finish_episode():
            self.adapter.episode_end(stats)
            if self.method == "atena":
                if self.adapter.query_count > before_query:
                    route = "query"
                elif self.adapter.self_label_count > before_self:
                    route = "self"
                else:
                    route = "none"
                feedback["episode_route"] = route
            elif self.method == "feedtta":
                if self.adapter.successful_episodes > before_success:
                    route = "success"
                elif self.adapter.failed_episodes > before_failure:
                    route = "failure"
                else:
                    route = "none"
                feedback["episode_route"] = route

        self._observe_parameter_call(
            lambda: (
                "episode_end_" + feedback["episode_route"]
                if "episode_route" in feedback else "episode_end"
            ),
            finish_episode,
        )
        self._episode_open = False
        self.episode_count += 1
        self.episode_diagnostics.append({
            "episode": self.episode_count,
            "generation_calls": self.generation_calls - self._episode_start_calls,
            "action_comparisons": self._episode_action_comparisons,
            "parameter_changes": self._episode_parameter_changes,
            "policy_parameter_drift": self._parameter_drift(),
            **feedback,
        })
        self.write_diagnostics()

    def write_diagnostics(self):
        diagnostics = self.adapter.diagnostics()
        diagnostics.update({
            "schema": "navtta.streamvln_tta_diagnostics.v2",
            "method": self.method,
            "decision_space": "first_generated_action_token",
            "action_text": list(self.action_text),
            "action_token_ids": list(self.action_token_ids),
            "adaptation_scope": "float32_qwen_final_rmsnorm_action_readout",
            "readout_protocol": self.readout_protocol,
            "policy_parameter_features": (
                "post_norm_hidden" if self.readout_protocol == "native_residual"
                else "pre_norm_hidden"
            ),
            "action_comparisons": self.action_comparisons,
            "parameter_changes": self.parameter_changes,
            "policy_parameter_drift": self._parameter_drift(),
            "parameter_observation_semantics": "net_change_across_adapter_hook_boundaries",
            "optimizer_step_attempts": int(self.adapter.optimizer_step_attempts),
            "episode_diagnostics": self.episode_diagnostics,
            "generation_calls": self.generation_calls,
            "overridden_first_actions": self.overridden_first_actions,
            "overridden_first_actions_reference": "native_full_vocabulary_argmax",
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
