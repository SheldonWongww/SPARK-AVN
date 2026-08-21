"""Online TTA glue for the batch-one DUET/HAMT/GOAT evaluators.

The discrete VLN baselines expose a high-level navigation distribution rather
than Habitat's ``actor_critic`` interface.  Tent and FSTTA only need that
distribution, so their task-agnostic implementations can be reused directly.
EAM and ATENA use detached decision-input replay through a discrete-policy
callback; FeedTTA and ATENA receive only binary navigation-success feedback.
"""

import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from navtta_core.tta import build_adapter, module_state_sha256


TTA_METHODS = ("source", "tent", "fstta", "eam", "feedtta", "atena")


def add_discrete_tta_args(parser):
    """Install the identical TTA CLI on every discrete VLN parser."""
    group = parser.add_argument_group("discrete VLN test-time adaptation")
    group.add_argument("--tta_method", choices=TTA_METHODS, default="source")
    group.add_argument(
        "--tta_audit_zero_update", action="store_true", default=False,
        help=(
            "exercise the selected adapter while suppressing and auditing "
            "every parameter write (formal adapter-parity audit only)"
        ),
    )
    group.add_argument(
        "--tta_audit_control", action="store_true", default=False,
        help="emit no-update Source evidence for an adapter-parity audit",
    )
    group.add_argument(
        "--tta_audit_expected_episodes", type=int, default=-1,
        help="defer the expensive final parameter-state hash to this episode",
    )
    group.add_argument("--tta_diagnostics", default=None)
    group.add_argument(
        "--tta_action_selection",
        choices=("auto", "argmax", "sample"),
        default="auto",
        help=(
            "auto preserves the discrete VLN evaluator's native argmax "
            "policy; explicit sample is retained only for controlled ablations"
        ),
    )
    group.add_argument("--tta_action_seed", type=int, default=0)
    group.add_argument("--tta_lr", type=float, default=1e-6)
    group.add_argument("--tta_norm_scope", choices=(
        "first_ln", "last_ln", "last_k_ln", "ln", "gn", "bn", "all"
    ), default="last_k_ln")
    group.add_argument("--tta_last_k_ln", type=int, default=4)
    group.add_argument(
        "--tta_optimizer", choices=("Adam", "AdamW", "SGD"), default="Adam"
    )
    group.add_argument("--tta_momentum", type=float, default=0.9)
    group.add_argument("--tta_beta1", type=float, default=0.9)
    group.add_argument("--tta_beta2", type=float, default=0.999)
    group.add_argument("--tta_weight_decay", type=float, default=None)
    group.add_argument("--tta_max_grad_norm", type=float, default=None)
    group.add_argument("--tta_update_interval", type=int, default=1)
    group.add_argument("--tta_max_updates_per_episode", type=int, default=-1)
    group.add_argument("--tta_episodic", action="store_true", default=False)
    group.add_argument(
        "--tta_trainable_prefixes",
        nargs="+",
        default=None,
        help=(
            "explicit post-encoder module prefixes for EAM/FeedTTA; defaults "
            "to the baseline's cross-modal decision modules"
        ),
    )

    fstta = parser.add_argument_group("FSTTA")
    fstta.add_argument("--tta_fstta_lr_slow", type=float, default=1e-4)
    fstta.add_argument("--tta_fstta_m", type=int, default=3)
    fstta.add_argument("--tta_fstta_n", type=int, default=4)
    fstta.add_argument("--tta_fstta_q", type=float, default=0.1)
    fstta.add_argument("--tta_fstta_rho", type=float, default=0.95)
    fstta.add_argument("--tta_fstta_tau", type=float, default=0.7)
    fstta.add_argument("--tta_fstta_a", type=float, default=0.9)
    fstta.add_argument("--tta_fstta_b", type=float, default=1.1)
    fstta.add_argument(
        "--tta_fstta_fast_grad_mode",
        choices=("concordant", "mean", "last"),
        default="concordant",
    )
    fstta.add_argument(
        "--tta_fstta_no_fast_lr_scaler", action="store_true", default=False
    )
    fstta.add_argument("--tta_fstta_no_slow", action="store_true", default=False)
    fstta.add_argument(
        "--tta_fstta_slow_optimizer",
        choices=("", "Adam", "AdamW", "SGD"),
        default="",
    )
    fstta.add_argument("--tta_fstta_slow_momentum", type=float, default=-1.0)
    fstta.add_argument(
        "--tta_fstta_reset_slow_optimizer_each_window",
        action="store_true",
        default=False,
    )
    fstta.add_argument(
        "--tta_fstta_no_reset_optimizer_each_episode",
        action="store_true",
        default=False,
    )
    fstta.add_argument("--tta_fstta_eigen_eps", type=float, default=1e-6)

    eam = parser.add_argument_group("EAM")
    eam.add_argument("--tta_eam_lr", type=float, default=1e-5)
    eam.add_argument("--tta_eam_confidence_scale", type=float, default=0.4)
    eam.add_argument("--tta_eam_memory_size", type=int, default=32)
    eam.add_argument("--tta_eam_batch_size", type=int, default=8)
    eam.add_argument("--tta_eam_update_interval", type=int, default=1)

    feed = parser.add_argument_group("FeedTTA")
    feed.add_argument("--tta_feedtta_lr", type=float, default=5e-6)
    feed.add_argument("--tta_feedtta_p", type=float, default=0.05)
    feed.add_argument("--tta_feedtta_alpha", type=float, default=-0.2)
    feed.add_argument("--tta_feedtta_sgr_seed", type=int, default=0)
    feed.add_argument("--tta_feedtta_gamma", type=float, default=0.99)
    feed.add_argument(
        "--tta_feedtta_scope_profile",
        choices=("paper_full", "last_crossmodal", "action_head"),
        default="paper_full",
        help=(
            "model-aware FeedTTA parameter scope; paper_full updates the full "
            "cross-modal decision stack, last_crossmodal updates its final "
            "cross-modal layer and action heads, and action_head updates only "
            "the action heads"
        ),
    )
    feed.add_argument(
        "--tta_feedtta_normalize_gradient", action="store_true", default=False
    )
    feed.add_argument("--tta_feedtta_eps", type=float, default=1e-5)

    atena = parser.add_argument_group("ATENA")
    atena.add_argument("--tta_atena_lr_query", type=float, default=1e-6)
    atena.add_argument("--tta_atena_lr_self", type=float, default=1e-7)
    atena.add_argument("--tta_atena_mix_lambda", type=float, default=0.5)
    atena.add_argument("--tta_atena_query_threshold", type=float, default=0.1)
    atena.add_argument("--tta_atena_self_loss_weight", type=float, default=0.1)
    return parser


def _default_trainable_prefixes(model):
    inner = getattr(model, "vln_bert", None)
    if inner is None:
        raise ValueError("Discrete policy does not expose a vln_bert module")
    if hasattr(inner, "global_encoder") and hasattr(inner, "local_encoder"):
        prefixes = [
            "vln_bert.global_encoder",
            "vln_bert.local_encoder",
            "vln_bert.global_sap_head",
            "vln_bert.local_sap_head",
            "vln_bert.sap_fuse_linear",
        ]
        # GOAT's recurrent decision feature is produced after the two
        # cross-modal branches. These modules are absent in DUET and harmless
        # to omit there.
        for name in (
            "gmap_pooler", "vp_pooler", "txt_pooler", "local_his_map",
            "local_his_ln",
        ):
            if hasattr(inner, name):
                prefixes.append("vln_bert." + name)
        return tuple(prefixes)
    if hasattr(inner, "encoder") and hasattr(inner, "next_action"):
        prefixes = ["vln_bert.encoder.x_layers", "vln_bert.next_action"]
        if hasattr(inner, "ref_object"):
            prefixes.append("vln_bert.ref_object")
        return tuple(prefixes)
    raise ValueError(
        "Could not infer discrete cross-modal decision module prefixes; pass "
        "--tta_trainable_prefixes explicitly"
    )


def _feedtta_trainable_prefixes(model, profile):
    """Map the paper's cross-modal-and-downstream scope onto each VLN model."""
    profile = str(profile).lower()
    inner = getattr(model, "vln_bert", None)
    if inner is None:
        raise ValueError("Discrete policy does not expose a vln_bert module")
    if hasattr(inner, "global_encoder") and hasattr(inner, "local_encoder"):
        heads = [
            "vln_bert.global_sap_head",
            "vln_bert.local_sap_head",
        ]
        if getattr(inner, "sap_fuse_linear", None) is not None:
            heads.append("vln_bert.sap_fuse_linear")
        if profile == "action_head":
            return tuple(heads)

        # DUET exposes its cross-modal blocks as ``x_layers`` while GOAT uses
        # ``crossattention``.  Freeze the positional/map embeddings that live
        # beside these stacks: the paper updates from the cross-modal encoder
        # onward, not the complete global/local input encoders.
        stack_specs = []
        for branch in ("global_encoder", "local_encoder"):
            encoder = getattr(getattr(inner, branch), "encoder")
            if hasattr(encoder, "x_layers"):
                stack_name = "x_layers"
            elif hasattr(encoder, "crossattention"):
                stack_name = "crossattention"
            else:
                raise ValueError(
                    "Could not infer {} FeedTTA cross-modal stack".format(
                        branch
                    )
                )
            layers = getattr(encoder, stack_name)
            stack_specs.append((branch, stack_name, layers))
        if profile == "paper_full":
            return tuple(
                "vln_bert.{}.encoder.{}".format(branch, stack_name)
                for branch, stack_name, _ in stack_specs
            ) + tuple(heads)
        global_branch, global_name, global_layers = stack_specs[0]
        local_branch, local_name, local_layers = stack_specs[1]
        if len(global_layers) < 1 or len(local_layers) < 1:
            raise ValueError("FeedTTA requires nonempty cross-modal layer stacks")
        return (
            "vln_bert.{}.encoder.{}.{}".format(
                global_branch, global_name, len(global_layers) - 1
            ),
            "vln_bert.{}.encoder.{}.{}".format(
                local_branch, local_name, len(local_layers) - 1
            ),
            *tuple(heads),
        )
    if hasattr(inner, "encoder") and hasattr(inner, "next_action"):
        heads = ["vln_bert.next_action"]
        if hasattr(inner, "ref_object"):
            heads.append("vln_bert.ref_object")
        if profile == "action_head":
            return tuple(heads)
        layers = inner.encoder.x_layers
        if len(layers) < 1:
            raise ValueError("FeedTTA requires a nonempty cross-modal layer stack")
        if profile == "paper_full":
            return ("vln_bert.encoder.x_layers", *heads)
        return (
            "vln_bert.encoder.x_layers.{}".format(len(layers) - 1),
            *heads,
        )
    raise ValueError("Could not infer FeedTTA module prefixes")


def _adapter_config(args, trainable_prefixes, action_selection="argmax"):
    common_weight_decay = (
        0.0 if args.tta_weight_decay is None else args.tta_weight_decay
    )
    feedback_weight_decay = (
        0.01 if args.tta_weight_decay is None else args.tta_weight_decay
    )
    norm_max_grad_norm = (
        1.0 if args.tta_max_grad_norm is None else args.tta_max_grad_norm
    )
    replay_max_grad_norm = (
        0.0 if args.tta_max_grad_norm is None else args.tta_max_grad_norm
    )
    fstta = SimpleNamespace(
        LR_SLOW=args.tta_fstta_lr_slow,
        M=args.tta_fstta_m,
        N=args.tta_fstta_n,
        Q=args.tta_fstta_q,
        RHO=args.tta_fstta_rho,
        TAU=args.tta_fstta_tau,
        A=args.tta_fstta_a,
        B=args.tta_fstta_b,
        OPTIMIZER="AdamW" if args.tta_method == "fstta" else args.tta_optimizer,
        BETA1=args.tta_beta1,
        BETA2=0.99 if args.tta_method == "fstta" else args.tta_beta2,
        WEIGHT_DECAY=common_weight_decay,
        USE_SLOW=not args.tta_fstta_no_slow,
        FAST_GRAD_MODE=args.tta_fstta_fast_grad_mode,
        USE_FAST_LR_SCALER=not args.tta_fstta_no_fast_lr_scaler,
        SLOW_OPTIMIZER=args.tta_fstta_slow_optimizer,
        SLOW_MOMENTUM=args.tta_fstta_slow_momentum,
        RESET_SLOW_OPTIMIZER_EACH_WINDOW=(
            args.tta_fstta_reset_slow_optimizer_each_window
        ),
        RESET_OPTIMIZER_EACH_EPISODE=(
            not args.tta_fstta_no_reset_optimizer_each_episode
        ),
        EIGEN_EPS=args.tta_fstta_eigen_eps,
    )
    eam = SimpleNamespace(
        LR=args.tta_eam_lr,
        CONFIDENCE_SCALE=args.tta_eam_confidence_scale,
        MEMORY_SIZE=args.tta_eam_memory_size,
        BATCH_SIZE=args.tta_eam_batch_size,
        UPDATE_INTERVAL=args.tta_eam_update_interval,
        PARAM_SCOPE="module_prefixes",
        TRAINABLE_PREFIXES=trainable_prefixes,
        OPTIMIZER=args.tta_optimizer,
        MOMENTUM=args.tta_momentum,
        BETA1=args.tta_beta1,
        BETA2=args.tta_beta2,
        WEIGHT_DECAY=common_weight_decay,
        MAX_GRAD_NORM=replay_max_grad_norm,
    )
    feedtta = SimpleNamespace(
        LR=args.tta_feedtta_lr,
        P=args.tta_feedtta_p,
        ALPHA=args.tta_feedtta_alpha,
        SGR_SEED=args.tta_feedtta_sgr_seed,
        GAMMA=args.tta_feedtta_gamma,
        NORMALIZE_GRADIENT=args.tta_feedtta_normalize_gradient,
        PARAM_SCOPE="module_prefixes",
        TRAINABLE_PREFIXES=trainable_prefixes,
        OPTIMIZER=args.tta_optimizer,
        BETA1=args.tta_beta1,
        BETA2=args.tta_beta2,
        WEIGHT_DECAY=common_weight_decay,
        EPS=args.tta_feedtta_eps,
        MAX_GRAD_NORM=replay_max_grad_norm,
        ACTION_SELECTION_PROTOCOL=(
            "sample_from_policy"
            if action_selection == "sample" else "target_native_argmax"
        ),
    )
    atena = SimpleNamespace(
        LR_QUERY=args.tta_atena_lr_query,
        LR_SELF=args.tta_atena_lr_self,
        MIX_LAMBDA=args.tta_atena_mix_lambda,
        QUERY_THRESHOLD=args.tta_atena_query_threshold,
        SELF_LOSS_WEIGHT=args.tta_atena_self_loss_weight,
        PARAM_SCOPE="all",
        OPTIMIZER="AdamW",
        BETA1=args.tta_beta1,
        BETA2=args.tta_beta2,
        WEIGHT_DECAY=feedback_weight_decay,
        MAX_GRAD_NORM=replay_max_grad_norm,
    )
    return SimpleNamespace(
        METHOD=args.tta_method,
        AUDIT_ZERO_UPDATE=bool(args.tta_audit_zero_update),
        AUDIT_EXPECTED_EPISODES=(
            None if args.tta_audit_expected_episodes < 0
            else int(args.tta_audit_expected_episodes)
        ),
        LR=args.tta_lr,
        STEPS=1,
        EPISODIC=args.tta_episodic,
        RESET_BN_STATS=True,
        NORM_SCOPE=args.tta_norm_scope,
        LAST_K_LN=args.tta_last_k_ln,
        OPTIMIZER=args.tta_optimizer,
        MOMENTUM=args.tta_momentum,
        BETA1=args.tta_beta1,
        BETA2=args.tta_beta2,
        WEIGHT_DECAY=common_weight_decay,
        MAX_GRAD_NORM=norm_max_grad_norm,
        UPDATE_INTERVAL=args.tta_update_interval,
        MAX_UPDATES_PER_EPISODE=args.tta_max_updates_per_episode,
        FSTTA=fstta,
        EAM=eam,
        FEEDTTA=feedtta,
        ATENA=atena,
    )


def _tree_detach(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _tree_detach(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_detach(item) for item in value)
    if isinstance(value, list):
        return [_tree_detach(item) for item in value]
    return value


def _sanitize_action_logits(logits):
    """Keep the action index space while replacing invalid-mask infinities."""
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError(
            "Discrete TTA expects [batch, actions] logits; got {}".format(
                type(logits).__name__ if not torch.is_tensor(logits)
                else tuple(logits.shape)
            )
        )
    if bool(torch.isnan(logits).any()):
        raise FloatingPointError("Navigation logits contain NaN")
    if bool(torch.isposinf(logits).any()):
        raise FloatingPointError("Navigation logits contain positive infinity")
    finite = torch.isfinite(logits)
    if not bool(finite.any(dim=1).all()):
        raise FloatingPointError("Navigation decision has no finite action logits")
    # -1e4 has effectively zero softmax mass in fp32/fp16, while unlike -inf
    # it keeps p * log(p) finite in the shared entropy implementation.
    return torch.where(finite, logits, torch.full_like(logits, -1e4))


def discrete_forward_policy(model, policy_inputs):
    """Replay one detached DUET/GOAT/HAMT high-level decision."""
    family = policy_inputs["family"]
    model_inputs = policy_inputs["model_inputs"]
    if family == "graph":
        outputs = model("navigation", model_inputs)
        fusion = policy_inputs["fusion"]
        logit_key = (
            "local_logits" if fusion == "local"
            else "global_logits" if fusion == "global"
            else "fused_logits"
        )
        logits = outputs[logit_key]
        if "cls_embeds" in outputs:
            features = outputs["cls_embeds"]
        else:
            features = 0.5 * (
                outputs["gmap_embeds"][:, 0] + outputs["vp_embeds"][:, 0]
            )
    elif family == "hamt_r2r":
        replay_inputs = dict(model_inputs)
        replay_inputs["return_states"] = True
        outputs = model(**replay_inputs)
        logits, features = outputs[0], outputs[1]
    elif family == "hamt_reverie":
        replay_inputs = dict(model_inputs)
        replay_inputs["return_states"] = True
        outputs = model(**replay_inputs)
        object_stop_logits = outputs["obj_logits"].max(dim=1).values.unsqueeze(1)
        logits = torch.cat([outputs["act_logits"], object_stop_logits], dim=1)
        features = outputs["states"]
    else:
        raise ValueError("Unknown discrete policy replay family: {}".format(family))
    invalid_action_mask = policy_inputs.get("invalid_action_mask")
    if invalid_action_mask is not None:
        if (
            not torch.is_tensor(invalid_action_mask)
            or invalid_action_mask.shape != logits.shape
        ):
            raise ValueError(
                "Discrete replay action mask shape must match logits: {} vs {}"
                .format(
                    getattr(invalid_action_mask, "shape", None),
                    tuple(logits.shape),
                )
            )
        logits = logits.masked_fill(invalid_action_mask.bool(), -float("inf"))
    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError("Discrete policy replay must produce 2-D decision features")
    return features, _sanitize_action_logits(logits)


class _SourceAdapter:
    """No-update adapter used only for the matched sampling control."""

    requires_source_grad = False

    def __init__(
        self, model, control="source_no_update", audit_expected_episodes=None
    ):
        self.model = model
        self.action_steps = 0
        self.episode_count = 0
        self.control = str(control)
        self.audit_expected_episodes = audit_expected_episodes
        self.model_state_before_sha256 = (
            module_state_sha256(model)
            if audit_expected_episodes is not None else None
        )
        self.model_state_after_sha256 = None

    def reset(self):
        self.action_steps = 0
        self.episode_count = 0
        self.model_state_after_sha256 = None

    def episode_start(self):
        return None

    def episode_end(self, episode_stats=None):
        self.episode_count += 1

    def before_inference(self, **kwargs):
        return None

    def prepare_action(self, source_logits, **kwargs):
        return source_logits

    def adapt(self, logits, **kwargs):
        self.action_steps += int(logits.shape[0])

    def diagnostics(self):
        if (
            self.audit_expected_episodes is not None
            and self.episode_count >= self.audit_expected_episodes
            and self.model_state_after_sha256 is None
        ):
            self.model_state_after_sha256 = module_state_sha256(self.model)
        return {
            "action_steps": self.action_steps,
            "episodes": self.episode_count,
            "updates": 0,
            "relative_param_drift": 0.0,
            "control": self.control,
            "audit_expected_episodes": self.audit_expected_episodes,
            "model_state_before_sha256": self.model_state_before_sha256,
            "model_state_after_sha256": self.model_state_after_sha256,
            "model_state_hash_match": (
                None if self.model_state_after_sha256 is None else
                self.model_state_after_sha256
                == self.model_state_before_sha256
            ),
        }


class DiscreteTTAController:
    """Own one adapter and enforce the canonical discrete-VLN protocol."""

    def __init__(self, args, model, stream_name=None):
        self.args = args
        self.model = model
        self.method = str(args.tta_method).lower()
        audit_zero_update = bool(args.tta_audit_zero_update)
        audit_control = bool(args.tta_audit_control)
        if audit_zero_update and self.method == "source":
            raise ValueError("zero-update adapter audit requires a TTA method")
        if audit_control and self.method != "source":
            raise ValueError("adapter audit controls require method=source")
        if audit_zero_update and audit_control:
            raise ValueError("audit job cannot be both adapter and Source control")
        if (audit_zero_update or audit_control) and int(
            args.tta_audit_expected_episodes
        ) <= 0:
            raise ValueError("adapter audit requires expected episodes > 0")
        requested_selection = str(args.tta_action_selection).lower()
        self.action_selection = (
            "argmax"
        ) if requested_selection == "auto" else requested_selection
        if (
            self.method not in ("source", "feedtta")
            and self.action_selection != "argmax"
        ):
            raise ValueError(
                "Only source controls and FeedTTA support sampled evaluation"
            )
        self.action_seed = int(args.tta_action_seed)
        self.current_action_seed = self.action_seed
        self._action_generator = torch.Generator(device="cpu")
        self._action_generator.manual_seed(self.action_seed)
        self.stream_name = stream_name
        self.episode_count = 0
        self._episode_open = False
        self._pending_policy_inputs = None
        self.binary_feedback_endpoint = None
        self._trajectory_hasher = hashlib.sha256()
        self.trajectory_steps = 0
        self.diagnostics_path = args.tta_diagnostics or os.path.join(
            args.output_dir, "tta_diagnostics.json"
        )
        self.feedtta_scope_profile = (
            (
                "explicit_prefixes"
                if args.tta_trainable_prefixes is not None
                else str(args.tta_feedtta_scope_profile)
            )
            if self.method == "feedtta" else None
        )
        if self.method == "eam":
            self.trainable_prefixes = tuple(
                args.tta_trainable_prefixes
                if args.tta_trainable_prefixes is not None
                else _default_trainable_prefixes(model)
            )
        elif self.method == "feedtta":
            self.trainable_prefixes = tuple(
                args.tta_trainable_prefixes
                if args.tta_trainable_prefixes is not None
                else _feedtta_trainable_prefixes(
                    model, self.feedtta_scope_profile
                )
            )
        else:
            self.trainable_prefixes = tuple(args.tta_trainable_prefixes or ())
        if self.method == "source":
            self.model.eval()
            self.model.requires_grad_(False)
            self.adapter = _SourceAdapter(
                self.model,
                "matched_source_sampling_no_update"
                if self.action_selection == "sample"
                else "source_argmax_no_update",
                audit_expected_episodes=(
                    int(args.tta_audit_expected_episodes)
                    if audit_control else None
                ),
            )
        else:
            self.adapter = build_adapter(
                model,
                _adapter_config(
                    args, self.trainable_prefixes, self.action_selection
                ),
                forward_policy=discrete_forward_policy,
            )

    def reset(self):
        self.adapter.reset()
        self.episode_count = 0
        self.current_action_seed = self.action_seed
        self._action_generator.manual_seed(self.action_seed)
        self._episode_open = False
        self._pending_policy_inputs = None
        self.binary_feedback_endpoint = None
        self._trajectory_hasher = hashlib.sha256()
        self.trajectory_steps = 0

    def diagnostics(self):
        return self.adapter.diagnostics()

    def begin_episode(self):
        if self._episode_open:
            raise RuntimeError("TTA episode_start called twice")
        # Explicit sampling ablations use an independent per-episode stream so
        # different prior trajectory lengths cannot desynchronize paired runs.
        if self.action_selection == "sample":
            self.current_action_seed = (
                self.action_seed + self.episode_count * 1000003
            )
            self._action_generator.manual_seed(self.current_action_seed)
        self.adapter.episode_start()
        self._trajectory_hasher.update(
            "episode:{}\0".format(self.episode_count).encode("ascii")
        )
        self._episode_open = True

    def detach_state(self, value):
        """Cut cached language/map/history features from the prior step graph."""
        return _tree_detach(value)

    def policy_inputs(self, model_inputs, family, fusion=None):
        return {
            "family": str(family),
            "fusion": None if fusion is None else str(fusion),
            "model_inputs": _tree_detach(model_inputs),
        }

    def before_inference(self, policy_inputs):
        # Defer replay-buffer insertion until the native runner has applied
        # every action mask. HAMT's no-backtrack mask is computed after the
        # policy forward and is not part of its model inputs; snapshotting here
        # would therefore replay a different decision problem.
        if self._pending_policy_inputs is not None:
            raise RuntimeError(
                "TTA before_inference called twice without prepare_action"
            )
        self._pending_policy_inputs = policy_inputs

    def prepare_action(self, source_logits, policy_inputs):
        if self._pending_policy_inputs is not policy_inputs:
            raise RuntimeError(
                "TTA prepare_action must follow before_inference for the same "
                "policy-input snapshot"
            )
        # Preserve the exact native action space for EAM/ATENA replay. This is
        # redundant for DUET/GOAT (their model inputs regenerate the masks),
        # but essential for HAMT's runner-level visited-candidate mask.
        policy_inputs["invalid_action_mask"] = (~torch.isfinite(
            source_logits.detach()
        )).detach()
        self.adapter.before_inference(policy_inputs=policy_inputs)
        source_logits = _sanitize_action_logits(source_logits)
        prepared = self.adapter.prepare_action(
            source_logits, policy_inputs=policy_inputs
        )
        self._pending_policy_inputs = None
        return _sanitize_action_logits(prepared)

    def select_action(self, logits):
        if self.action_selection == "sample":
            # Use the same inverse-CDF sampler as VLN-CE. Keeping its uniform
            # stream on CPU makes paired runs invariant to CUDA RNG state.
            probabilities = logits.detach().float().softmax(dim=-1).cpu()
            uniforms = torch.rand(
                (probabilities.shape[0], 1),
                generator=self._action_generator,
            )
            cumulative = probabilities.cumsum(dim=-1)
            sampled = (uniforms > cumulative).sum(dim=-1)
            sampled.clamp_(max=probabilities.shape[-1] - 1)
            selected = sampled.to(device=logits.device)
        else:
            selected = logits.argmax(dim=-1).detach()
        values = selected.detach().cpu().view(-1).tolist()
        self._trajectory_hasher.update(
            ("actions:" + ",".join(map(str, values)) + "\0").encode("ascii")
        )
        self.trajectory_steps += len(values)
        return selected

    def adapt_step(self, logits, action=None, **context):
        if not self._episode_open:
            raise RuntimeError("TTA action step occurred outside an episode")
        logits = _sanitize_action_logits(logits)
        if self.adapter.requires_source_grad and not logits.requires_grad:
            raise RuntimeError(
                "Navigation logits are detached from every selected TTA parameter; "
                "choose a norm scope used by the high-level decision module"
            )
        self.adapter.adapt(logits, action=action, **context)

    def end_episode(self, episode_stats=None, observations=None):
        if not self._episode_open:
            raise RuntimeError("TTA episode_end called without episode_start")
        if self._pending_policy_inputs is not None:
            raise RuntimeError("TTA episode ended before prepare_action")
        if self.method in ("feedtta", "atena"):
            if episode_stats is None:
                if observations is None or len(observations) != 1:
                    raise ValueError(
                        "Binary-feedback TTA requires one final observation"
                    )
                observation = observations[0]
                if "distance" not in observation:
                    raise ValueError(
                        "Binary-feedback TTA requires final geodesic distance"
                    )
                if self.method == "atena":
                    # Do not inspect the held-out outcome unless ATENA's
                    # entropy gate actively queries this episode.
                    episode_stats = lambda: {
                        "success": float(observation["distance"]) < 3.0
                    }
                else:
                    episode_stats = {
                        "success": float(observation["distance"]) < 3.0
                    }
                self.binary_feedback_endpoint = (
                    "final_simulator_observation_distance"
                )
        self.adapter.episode_end(episode_stats=episode_stats)
        self._trajectory_hasher.update(b"episode_end\0")
        self._episode_open = False
        self.episode_count += 1
        self.write_diagnostics()

    def write_diagnostics(self):
        diagnostics = self.adapter.diagnostics()
        payload = {
            "schema": "navtta.vln_discrete_tta.v1",
            "method": self.method,
            "audit_zero_update": bool(
                getattr(self.args, "tta_audit_zero_update", False)
            ),
            "audit_control": bool(
                getattr(self.args, "tta_audit_control", False)
            ),
            "stream": self.stream_name,
            "episode_count": self.episode_count,
            "action_steps": diagnostics["action_steps"],
            "batch_size": 1,
            "tta_step_unit": "high_level_navigation_decision",
            "action_selection": (
                "policy_sampling"
                if self.action_selection == "sample"
                else "target_native_argmax"
            ),
            "action_seed": (
                self.action_seed if self.action_selection == "sample" else None
            ),
            "current_episode_action_seed": (
                self.current_action_seed
                if self.action_selection == "sample" else None
            ),
            "action_seed_schedule": (
                "base_plus_episode_index_times_1000003"
                if self.action_selection == "sample" else None
            ),
            "action_rng": (
                "dedicated_cpu_uniform_inverse_cdf"
                if self.action_selection == "sample" else None
            ),
            "masked_action_entropy": "finite_logits_only",
            "cached_policy_state": "detached_between_decisions",
            "trajectory_steps": self.trajectory_steps,
            "trajectory_sha256": self._trajectory_hasher.hexdigest(),
            "supervision": (
                "binary_navigation_success_feedback"
                if self.method in ("feedtta", "atena")
                else "unsupervised"
            ),
            "binary_feedback_endpoint": self.binary_feedback_endpoint,
            "trainable_prefixes": list(self.trainable_prefixes),
            "feedtta_scope_profile": self.feedtta_scope_profile,
            "adapter": diagnostics,
        }
        path = Path(self.diagnostics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(str(temporary), str(path))


class DiscreteTTAAgentMixin:
    """Mixin used by all six DUET/HAMT/GOAT agent bases."""

    def activate_discrete_tta(self, feedback):
        method = str(getattr(self.args, "tta_method", "source")).lower()
        action_selection = str(
            getattr(self.args, "tta_action_selection", "auto")
        ).lower()
        audit_control = bool(getattr(self.args, "tta_audit_control", False))
        if method == "source" and action_selection != "sample" and not audit_control:
            return
        if not bool(getattr(self.args, "test", False)):
            raise ValueError("Discrete TTA is evaluation-only and requires --test")
        if int(getattr(self.args, "world_size", 1)) != 1:
            raise ValueError("Discrete TTA requires --world_size 1")
        if int(getattr(self.args, "batch_size", 0)) != 1:
            raise ValueError("Discrete TTA requires --batch_size 1")
        if getattr(self.args, "episode_order_manifest", None) is None:
            raise ValueError("Discrete TTA requires --episode_order_manifest")
        if feedback != "argmax":
            raise ValueError(
                "Discrete TTA evaluation must be invoked with feedback='argmax'; "
                "FeedTTA sampling is selected internally"
            )
        stream_name = getattr(getattr(self, "env", None), "name", None)
        if method in ("feedtta", "atena") and stream_name == "test":
            raise ValueError(
                "{} consumes binary navigation-success feedback and cannot run "
                "on the unlabeled test split".format(method.upper())
            )
        if getattr(self, "tta_controller", None) is None:
            model = self.vln_bert.module if hasattr(self.vln_bert, "module") else self.vln_bert
            self.tta_controller = DiscreteTTAController(
                self.args, model, stream_name=stream_name
            )
            # run_exact_agent_epoch resets this task-agnostic adapter once at
            # the start of each canonical stream.
            self.tta_adapter = self.tta_controller
        else:
            self.tta_controller.stream_name = stream_name

    def tta_episode_start(self):
        controller = getattr(self, "tta_controller", None)
        if controller is not None:
            controller.begin_episode()

    def tta_detach_state(self, value):
        controller = getattr(self, "tta_controller", None)
        return value if controller is None else controller.detach_state(value)

    def tta_policy_inputs(self, model_inputs, family, fusion=None):
        controller = getattr(self, "tta_controller", None)
        if controller is None:
            return None
        return controller.policy_inputs(model_inputs, family, fusion=fusion)

    def tta_before_inference(self, policy_inputs):
        controller = getattr(self, "tta_controller", None)
        if controller is not None:
            controller.before_inference(policy_inputs)

    def tta_prepare_action(self, source_logits, policy_inputs):
        controller = getattr(self, "tta_controller", None)
        if controller is None:
            return source_logits
        return controller.prepare_action(source_logits, policy_inputs)

    def tta_graph_features(self, outputs):
        if "cls_embeds" in outputs:
            return outputs["cls_embeds"]
        return 0.5 * (
            outputs["gmap_embeds"][:, 0] + outputs["vp_embeds"][:, 0]
        )

    def tta_select_action(self, logits):
        controller = getattr(self, "tta_controller", None)
        return None if controller is None else controller.select_action(logits)

    def tta_adapt_step(self, logits, action=None, **context):
        controller = getattr(self, "tta_controller", None)
        if controller is not None:
            controller.adapt_step(logits, action=action, **context)

    def tta_episode_end(self, episode_stats=None, observations=None):
        controller = getattr(self, "tta_controller", None)
        if controller is not None:
            controller.end_episode(
                episode_stats=episode_stats, observations=observations
            )

    def tta_r2r_episode_stats(self, trajectories, path_format):
        """Return the exact R2R evaluator success for binary-feedback TTA.

        DUET and GOAT may rerank the submitted endpoint after the simulator has
        stopped.  Feedback must therefore be computed from the submitted path,
        not from the simulator's current observation.  HAMT uses a flat list of
        viewpoint tuples, while DUET/GOAT use nested graph paths.
        """
        controller = getattr(self, "tta_controller", None)
        if controller is None or controller.method not in ("feedtta", "atena"):
            return None
        if len(trajectories) != 1:
            raise ValueError(
                "Canonical binary-feedback R2R TTA requires batch size one"
            )
        item = trajectories[0]
        instr_id = item["instr_id"]
        scan, ground_truth = self.env.gt_trajs[instr_id]
        if path_format == "nested_graph_path":
            evaluator_path = item["path"]
        elif path_format == "viewpoint_tuples":
            evaluator_path = [step[0] for step in item["path"]]
        else:
            raise ValueError("Unknown R2R trajectory format: {}".format(path_format))
        def evaluate_submitted_endpoint():
            scores = self.env._eval_item(scan, evaluator_path, ground_truth)
            if "success" not in scores:
                raise ValueError("R2R evaluator did not return episode success")
            return {"success": float(scores["success"])}

        # FeedTTA consumes feedback after every episode.  ATENA must preserve
        # its query budget, so expose the same exact evaluator endpoint as a
        # lazy callback and let its entropy gate decide whether to call it.
        if controller.method == "atena":
            controller.binary_feedback_endpoint = (
                "r2r_submitted_trajectory_evaluator_success_lazy_query"
            )
            return evaluate_submitted_endpoint
        controller.binary_feedback_endpoint = (
            "r2r_submitted_trajectory_evaluator_success_every_episode"
        )
        return evaluate_submitted_endpoint()
