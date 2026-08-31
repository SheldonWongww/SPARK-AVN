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


TTA_METHODS = ("source", "tent", "fstta", "eam", "feedtta", "atena", "idea")


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
    group.add_argument(
        "--tta_diagnostics_expected_episodes", type=int, default=-1,
        help=(
            "full-stream episode count; IDEA defers its content hash until "
            "this final episode"
        ),
    )
    group.add_argument("--tta_diagnostics", default=None)
    group.add_argument(
        "--tta_action_selection",
        choices=("auto", "argmax", "sample"),
        default="auto",
        help=(
            "auto preserves each discrete VLN evaluator's native argmax "
            "policy; explicit FeedTTA sampling is retained as a labelled "
            "paper-protocol ablation"
        ),
    )
    group.add_argument(
        "--tta_matched_feedtta_source", action="store_true", default=False,
        help=(
            "mark a Source run as a matched sampled-policy ablation control; "
            "this also makes action_selection=auto resolve to sample"
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
    fstta.add_argument(
        "--tta_fstta_reset_var_hist_each_episode",
        action="store_true",
        default=False,
        help=(
            "released-code rollout-reset ablation; canonical paper Eq. (6) "
            "retains the historical variance EMA across the test stream"
        ),
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
    feed.add_argument(
        "--tta_feedtta_sgr_mode",
        choices=("paper_main", "appendix_b1"),
        default="paper_main",
    )
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
    feed.add_argument(
        "--tta_feedback_provider",
        choices=("task_evaluator", "qwen2_vl_2b_v1"),
        default="task_evaluator",
        help=(
            "binary-feedback source; qwen2_vl_2b_v1 is the separately "
            "reported GOAT-REVERIE hidden-test FeedTTA-LLM variant"
        ),
    )
    feed.add_argument("--tta_llm_feedback_url", default="")
    feed.add_argument(
        "--tta_llm_feedback_model_id",
        default="Qwen/Qwen2-VL-2B-Instruct",
    )
    feed.add_argument("--tta_llm_feedback_revision", default="")
    feed.add_argument("--tta_llm_feedback_weights_sha256", default="")
    feed.add_argument("--tta_llm_feedback_bundle_sha256", default="")
    feed.add_argument("--tta_llm_feedback_prompt_sha256", default="")
    feed.add_argument("--tta_llm_feedback_token_file", default="")
    feed.add_argument("--tta_llm_feedback_cache_dir", default="")
    feed.add_argument("--tta_llm_feedback_transcript_path", default="")
    feed.add_argument(
        "--tta_llm_feedback_timeout_seconds", type=float, default=120.0
    )
    feed.add_argument(
        "--tta_llm_feedback_allow_failure",
        action="store_false",
        dest="tta_llm_feedback_abort_on_failure",
        default=True,
        help="debug-only: continue after recording a failed-closed provider episode",
    )

    atena = parser.add_argument_group("ATENA")
    atena.add_argument("--tta_atena_lr_query", type=float, default=1e-6)
    atena.add_argument("--tta_atena_lr_self", type=float, default=1e-7)
    atena.add_argument("--tta_atena_mix_lambda", type=float, default=0.5)
    atena.add_argument("--tta_atena_query_threshold", type=float, default=0.1)
    atena.add_argument("--tta_atena_self_loss_weight", type=float, default=0.1)
    atena.add_argument(
        "--tta_atena_update_scope",
        choices=("replay_reachable_high_level_navigation", "full_policy"),
        default="replay_reachable_high_level_navigation",
        help=(
            "the discrete callback can replay only the high-level navigation "
            "model; full_policy is rejected instead of making an end-to-end "
            "replay claim"
        ),
    )

    idea = parser.add_argument_group("IDEA")
    idea.add_argument("--tta_idea_prompt_length", type=int, default=4)
    idea.add_argument("--tta_idea_k_max", type=int, default=32)
    idea.add_argument("--tta_idea_lambda", type=float, default=0.4)
    idea.add_argument("--tta_idea_tau", type=float, default=0.7)
    idea.add_argument("--tta_idea_fisher_beta", type=float, default=0.1)
    idea.add_argument("--tta_idea_opt_steps", type=int, default=50)
    idea.add_argument("--tta_idea_lr", type=float, default=3e-3)
    idea.add_argument("--tta_idea_no_fisher", action="store_true", default=False)
    idea.add_argument("--tta_idea_ridge", type=float, default=1e-4)
    # 0 aligns every fusion layer; DUET/HAMT/GOAT default to num_x_layers=4.
    idea.add_argument("--tta_idea_prompt_layers", type=int, default=0)
    idea.add_argument("--tta_idea_source_stats_path", default="")
    idea.add_argument("--tta_idea_source_stats_sha256", default="")
    idea.add_argument(
        "--tta_idea_source_trajectories", type=int, default=128
    )
    idea.add_argument(
        "--tta_idea_collect_source_stats", action="store_true", default=False
    )
    idea.add_argument("--tta_idea_source_stats_output", default="")
    idea.add_argument("--tta_idea_source_checkpoint_sha256", default="")
    idea.add_argument("--tta_idea_source_dataset", default="")
    idea.add_argument("--tta_idea_source_dataset_version", default="")
    idea.add_argument("--tta_idea_source_split", default="train")
    idea.add_argument("--tta_idea_collection_policy", default="")
    return parser


def _infer_discrete_family(model):
    """Infer the discrete replay family from the policy's fusion structure."""
    inner = getattr(model, "vln_bert", None)
    if inner is None:
        raise ValueError("Discrete policy does not expose a vln_bert module")
    if hasattr(inner, "global_encoder") and hasattr(inner, "local_encoder"):
        return "graph"
    if hasattr(inner, "encoder") and hasattr(inner, "next_action"):
        return "hamt_reverie" if hasattr(inner, "ref_object") else "hamt_r2r"
    raise ValueError("Could not infer discrete fusion family for IDEA")


def _infer_discrete_model_id(model):
    """Return the public baseline name used in artifact provenance.

    DUET and GOAT share the high-level ``graph`` replay family, so the family
    identifier is not sufficient for provenance.  Distinguish their concrete
    local cross-modal stacks and keep HAMT's task-specific family internal.
    """
    family = _infer_discrete_family(model)
    if family.startswith("hamt_"):
        return "hamt"
    inner = model.vln_bert
    encoder = getattr(getattr(inner, "local_encoder", None), "encoder", None)
    if hasattr(encoder, "x_layers"):
        return "duet"
    if hasattr(encoder, "crossattention"):
        return "goat"
    raise ValueError("Could not infer graph baseline identity for IDEA")


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
        RESET_VAR_HIST_EACH_EPISODE=(
            args.tta_fstta_reset_var_hist_each_episode
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
        SGR_MODE=args.tta_feedtta_sgr_mode,
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
        TASK_UPDATE_SCOPE=args.tta_atena_update_scope,
        OPTIMIZER="AdamW",
        BETA1=args.tta_beta1,
        BETA2=args.tta_beta2,
        WEIGHT_DECAY=feedback_weight_decay,
        MAX_GRAD_NORM=replay_max_grad_norm,
    )
    idea = SimpleNamespace(
        PROMPT_LENGTH=args.tta_idea_prompt_length,
        K_MAX=args.tta_idea_k_max,
        LAMBDA=args.tta_idea_lambda,
        TAU=args.tta_idea_tau,
        FISHER_BETA=args.tta_idea_fisher_beta,
        OPT_STEPS=args.tta_idea_opt_steps,
        LR=args.tta_idea_lr,
        OPTIMIZER="AdamW",
        BETA1=args.tta_beta1,
        BETA2=args.tta_beta2,
        WEIGHT_DECAY=0.0,
        USE_FISHER=not bool(args.tta_idea_no_fisher),
        RIDGE=args.tta_idea_ridge,
        MAX_GRAD_NORM=0.0,
        PROMPT_LAYERS=args.tta_idea_prompt_layers,
        SOURCE_STATS_PATH=getattr(args, "tta_idea_source_stats_path", ""),
        SOURCE_STATS_SHA256=getattr(args, "tta_idea_source_stats_sha256", ""),
        SOURCE_TRAJECTORIES=int(
            getattr(args, "tta_idea_source_trajectories", 128)
        ),
        COLLECT_SOURCE_STATS=bool(
            getattr(args, "tta_idea_collect_source_stats", False)
        ),
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
        IDEA=idea,
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


def _graph_decision_features(outputs):
    """Return the model-native graph state used by ATENA's success head.

    The official ATENA DUET implementation represents a navigation decision
    by concatenating the global-map and local-view CLS tokens.  GOAT exposes a
    different, recurrent history representation as ``cls_embeds`` and must
    keep using that model-native state.  Prefer an explicit ``curr_state``
    when an ATENA-instrumented DUET model provides it; otherwise reconstruct
    the same DUET state from the standard model outputs.

    This helper is shared by the online pass and the episode-end replay so the
    self-prediction gate and its replay gradient see exactly the same feature.
    """
    if not isinstance(outputs, dict):
        raise ValueError("Graph policy outputs must be a dictionary")

    if "curr_state" in outputs:
        features = outputs["curr_state"]
    elif "cls_embeds" in outputs:
        # GOAT's navigation model explicitly constructs this state from its
        # global, local, language, and recurrent-history representations.
        features = outputs["cls_embeds"]
    else:
        try:
            global_embeds = outputs["gmap_embeds"]
            local_embeds = outputs["vp_embeds"]
        except KeyError as error:
            raise ValueError(
                "Graph policy does not expose a recognized decision feature"
            ) from error
        if (
            not torch.is_tensor(global_embeds)
            or not torch.is_tensor(local_embeds)
            or global_embeds.ndim != 3
            or local_embeds.ndim != 3
            or global_embeds.shape[0] != local_embeds.shape[0]
            or global_embeds.shape[1] < 1
            or local_embeds.shape[1] < 1
        ):
            raise ValueError(
                "DUET graph features require compatible [batch, tokens, dim] "
                "global and local embeddings"
            )
        # Official ATENA_DUET/map_nav_src/models/vilmodel.py constructs
        # curr_state exactly this way (feature width = 2 * hidden_size).
        features = torch.cat(
            [global_embeds[:, 0], local_embeds[:, 0]], dim=-1
        )

    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError("Graph policy decision features must be 2-D")
    return features


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
        features = _graph_decision_features(outputs)
    elif family == "hamt_r2r":
        replay_inputs = dict(model_inputs)
        replay_inputs["return_states"] = True
        outputs = model(**replay_inputs)
        logits, features = outputs[0], outputs[1]
    elif family == "hamt_reverie":
        replay_inputs = dict(model_inputs)
        replay_inputs["return_states"] = True
        outputs = model(**replay_inputs)
        # Preserve HAMT-REVERIE's native agent contract exactly: its stop
        # slot is the argmax object *index* (the second torch.max return), not
        # the maximum object-logit value.
        object_stop_logits = (
            outputs["obj_logits"].max(dim=1).indices.unsqueeze(1)
        )
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
        self.benchmark = str(getattr(args, "dataset", "")).lower()
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
        declared_matched_source = bool(
            getattr(args, "tta_matched_feedtta_source", False)
        )
        if declared_matched_source and self.method != "source":
            raise ValueError(
                "--tta_matched_feedtta_source is valid only for method=source"
            )
        if declared_matched_source and requested_selection == "argmax":
            raise ValueError(
                "a matched FeedTTA Source control must use policy sampling"
            )
        # ``sample`` is itself an explicit ablation request.  Normalize legacy
        # sampled-Source configs to the matched-control marker so diagnostics
        # cannot describe a sampled Source as an ordinary Source baseline.
        self.matched_feedtta_source = bool(
            declared_matched_source
            or (self.method == "source" and requested_selection == "sample")
        )
        self.action_selection = (
            "sample"
            if requested_selection == "auto" and self.matched_feedtta_source
            else (
                "argmax" if requested_selection == "auto"
                else requested_selection
            )
        )
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
        expected_diagnostics = int(getattr(
            args, "tta_diagnostics_expected_episodes", -1
        ))
        if expected_diagnostics == 0 or expected_diagnostics < -1:
            raise ValueError("diagnostics expected episodes must be -1 or positive")
        self.diagnostics_expected_episodes = (
            None if expected_diagnostics < 0 else expected_diagnostics
        )
        self._episode_open = False
        self._pending_policy_inputs = None
        self.binary_feedback_endpoint = None
        requested_feedback_provider = str(
            getattr(args, "tta_feedback_provider", "task_evaluator")
        ).lower()
        self.feedback_provider_name = (
            requested_feedback_provider
            if self.method in ("feedtta", "atena") else "none"
        )
        self.llm_feedback_provider = None
        self.last_pseudo_feedback = None
        self.failed_closed_feedback_episodes = 0
        self._trajectory_hasher = hashlib.sha256()
        self.trajectory_steps = 0
        self.idea_source_collection = None
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
        self.feedtta_sgr_mode = (
            str(args.tta_feedtta_sgr_mode).lower()
            if self.method == "feedtta" else None
        )
        self.feedtta_native_action_protocol = (
            self.method == "feedtta"
            and self.action_selection == "argmax"
        )
        self.feedtta_paper_sampling_protocol = (
            self.method == "feedtta"
            and self.action_selection == "sample"
        )
        # Backward-compatible field: canonical means the paper's on-policy
        # sampling action protocol, never the task-adapted VLN argmax port.
        self.feedtta_canonical_protocol = (
            self.feedtta_paper_sampling_protocol
        )
        self.atena_update_scope = None
        if requested_feedback_provider != "task_evaluator":
            if requested_feedback_provider != "qwen2_vl_2b_v1":
                raise ValueError("unknown discrete feedback provider")
            if self.method != "feedtta" or self.benchmark != "reverie":
                raise ValueError(
                    "qwen2_vl_2b_v1 is only valid for FeedTTA on REVERIE"
                )
            if str(self.stream_name).lower() not in ("test", "test_unseen"):
                raise ValueError(
                    "qwen2_vl_2b_v1 is restricted to the REVERIE hidden test stream"
                )
            if _infer_discrete_model_id(model) != "goat":
                raise ValueError(
                    "qwen2_vl_2b_v1 is restricted to the GOAT-REVERIE variant"
                )
            from .reverie_llm_feedback import (
                MODEL_ID,
                Qwen2VLFeedbackProvider,
            )

            if str(args.tta_llm_feedback_model_id) != MODEL_ID:
                raise ValueError(
                    "FeedTTA-LLM model id must be {}".format(MODEL_ID)
                )
            required = {
                "service URL": args.tta_llm_feedback_url,
                "model revision": args.tta_llm_feedback_revision,
                "weights SHA256": args.tta_llm_feedback_weights_sha256,
                "bundle SHA256": args.tta_llm_feedback_bundle_sha256,
                "prompt SHA256": args.tta_llm_feedback_prompt_sha256,
                "bearer token file": args.tta_llm_feedback_token_file,
                "cache directory": args.tta_llm_feedback_cache_dir,
                "transcript path": args.tta_llm_feedback_transcript_path,
            }
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError(
                    "FeedTTA-LLM requires pinned {}".format(
                        ", ".join(missing)
                    )
                )
            self.llm_feedback_provider = Qwen2VLFeedbackProvider(
                service_url=args.tta_llm_feedback_url,
                revision=args.tta_llm_feedback_revision,
                weights_sha256=args.tta_llm_feedback_weights_sha256,
                bundle_sha256=args.tta_llm_feedback_bundle_sha256,
                prompt_bundle_sha256=args.tta_llm_feedback_prompt_sha256,
                token_file=args.tta_llm_feedback_token_file,
                cache_dir=args.tta_llm_feedback_cache_dir,
                transcript_path=args.tta_llm_feedback_transcript_path,
                connectivity_dir=args.connectivity_dir,
                scan_data_dir=args.scan_data_dir,
                timeout_seconds=args.tta_llm_feedback_timeout_seconds,
                abort_on_failure=args.tta_llm_feedback_abort_on_failure,
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
            if self.method == "atena":
                self.atena_update_scope = str(
                    args.tta_atena_update_scope
                ).lower()
                if self.atena_update_scope != (
                    "replay_reachable_high_level_navigation"
                ):
                    raise ValueError(
                        "discrete VLN ATENA cannot replay the upstream visual "
                        "feature extractors; request "
                        "replay_reachable_high_level_navigation instead of "
                        "claiming full_policy"
                    )
                # These are candidates only. The first real replay callback
                # differentiates its features/logits and removes unreachable
                # parameters before any episode optimizer is constructed.
                for parameter in self.model.parameters():
                    parameter.requires_grad_(True)
            fusion_protocol = None
            if self.method == "idea":
                # IDEA injects a soft prompt into the frozen cross-modal fusion
                # stack; the family is inferred from the policy structure.
                from .idea_fusion import make_idea_fusion_protocol

                prompt_layers = int(getattr(args, "tta_idea_prompt_layers", 0))
                collect_source = bool(
                    getattr(args, "tta_idea_collect_source_stats", False)
                )
                if collect_source and (
                    not self.stream_name
                    or "train" not in str(self.stream_name).lower()
                ):
                    raise ValueError(
                        "IDEA source statistics can only be collected on a "
                        "source-training stream"
                    )
                idea_family = _infer_discrete_family(model)
                idea_model_id = _infer_discrete_model_id(model)
                idea_setting = "{}-{}".format(
                    idea_model_id, self.benchmark
                )
                fusion_protocol = make_idea_fusion_protocol(
                    model,
                    idea_family,
                    num_layers=(prompt_layers if prompt_layers > 0 else None),
                    source_stats_path=(None if collect_source else (
                        getattr(args, "tta_idea_source_stats_path", "") or None
                    )),
                    source_stats_sha256=(None if collect_source else (
                        getattr(args, "tta_idea_source_stats_sha256", "") or None
                    )),
                    expected_source_trajectories=int(
                        getattr(args, "tta_idea_source_trajectories", 128)
                    ),
                    expected_source_provenance=(
                        None if collect_source else {
                            "model": idea_model_id,
                            "setting": idea_setting,
                            "collection_policy": (
                                "frozen_source_argmax_rollout"
                            ),
                        }
                    ),
                    source_collection=collect_source,
                )
                if collect_source:
                    from navtta_core.tta import SourceStatisticsCollectionSession

                    if self.action_selection != "argmax":
                        raise ValueError(
                            "VLN IDEA source collection requires native argmax actions"
                        )
                    collection_policy = getattr(
                        args, "tta_idea_collection_policy", ""
                    )
                    if collection_policy != "frozen_source_argmax_rollout":
                        raise ValueError(
                            "VLN IDEA source collection requires "
                            "collection_policy=frozen_source_argmax_rollout"
                        )
                    self.model.eval()
                    self.model.requires_grad_(False)
                    self.idea_source_collection = SourceStatisticsCollectionSession(
                        fusion_protocol,
                        getattr(args, "tta_idea_source_stats_output", ""),
                        {
                            "checkpoint_sha256": getattr(
                                args, "tta_idea_source_checkpoint_sha256", ""
                            ),
                            "dataset": getattr(
                                args, "tta_idea_source_dataset", ""
                            ),
                            "dataset_version": getattr(
                                args, "tta_idea_source_dataset_version", ""
                            ),
                            "split": getattr(
                                args, "tta_idea_source_split", "train"
                            ),
                            "collection_policy": collection_policy,
                            "model": idea_model_id,
                            "setting": idea_setting,
                        },
                        expected_trajectory_count=int(
                            getattr(args, "tta_idea_source_trajectories", 128)
                        ),
                    )
            if self.idea_source_collection is not None:
                self.adapter = _SourceAdapter(
                    model, control="idea_source_statistics_collection"
                )
            else:
                self.adapter = build_adapter(
                    model,
                    _adapter_config(
                        args, self.trainable_prefixes, self.action_selection
                    ),
                    forward_policy=discrete_forward_policy,
                    fusion_protocol=fusion_protocol,
                )
            if self.method == "atena":
                atena_diagnostics = self.adapter.diagnostics()
                if atena_diagnostics.get(
                    "requires_task_trainability_wiring", False
                ):
                    raise RuntimeError(
                        "ATENA high-level navigation parameters are not all "
                        "trainable after task-level wiring"
                    )

    def reset(self):
        self.adapter.reset()
        if self.idea_source_collection is not None:
            self.idea_source_collection.reset()
        self.episode_count = 0
        self.current_action_seed = self.action_seed
        self._action_generator.manual_seed(self.action_seed)
        self._episode_open = False
        self._pending_policy_inputs = None
        self.binary_feedback_endpoint = None
        self.last_pseudo_feedback = None
        self.failed_closed_feedback_episodes = 0
        if self.llm_feedback_provider is not None:
            self.llm_feedback_provider.reset()
        self._trajectory_hasher = hashlib.sha256()
        self.trajectory_steps = 0

    def diagnostics(self, final_integrity=None):
        if final_integrity is None:
            final_integrity = (
                self.diagnostics_expected_episodes is None
                or self.episode_count >= self.diagnostics_expected_episodes
            )
        output = (
            self.adapter.diagnostics(verify_base_content=final_integrity)
            if self.method == "idea" else self.adapter.diagnostics()
        )
        if self.idea_source_collection is not None:
            output.update(self.idea_source_collection.diagnostics())
        return output

    def query_reverie_llm_feedback(
        self, environment, trajectories, path_format
    ):
        """Query the post-episode pseudo-label without evaluator access."""
        if self.llm_feedback_provider is None:
            raise RuntimeError("FeedTTA-LLM provider is not active")
        from .reverie_llm_feedback import (
            BINARY_FEEDBACK_ENDPOINT,
            deploy_time_episode_inputs,
        )

        self.binary_feedback_endpoint = BINARY_FEEDBACK_ENDPOINT
        try:
            episode = deploy_time_episode_inputs(
                environment, trajectories, path_format
            )
            result = self.llm_feedback_provider.query(episode)
        except Exception as error:
            submitted = (
                trajectories[0]
                if isinstance(trajectories, (list, tuple))
                and len(trajectories) == 1
                and isinstance(trajectories[0], dict)
                else {}
            )
            result = self.llm_feedback_provider.fail_closed(
                error,
                episode={
                    "episode_id": submitted.get("instr_id"),
                    "scan_id": None,
                    "endpoint_viewpoint_id": None,
                },
            )
        self.last_pseudo_feedback = result
        return {"_navtta_llm_pseudo_feedback": result}

    def _discard_feedtta_episode_gradient(self):
        """Fail closed without treating an absent pseudo-label as failure."""
        clear = getattr(self.adapter, "_clear_trajectory", None)
        if clear is None:
            raise RuntimeError("FeedTTA adapter cannot discard an episode")
        trajectory_steps = int(
            getattr(self.adapter, "_trajectory_step_count", 0)
        )
        clear()
        # Preserve lifecycle accounting without inventing a success/failure
        # label or executing the optimizer.
        self.adapter.episode_count += 1
        self.adapter.total_trajectory_steps += trajectory_steps
        self.adapter.max_trajectory_steps = max(
            self.adapter.max_trajectory_steps, trajectory_steps
        )
        self.failed_closed_feedback_episodes += 1

    def begin_episode(self, trajectory_id=None):
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
        if self.idea_source_collection is not None:
            if trajectory_id is None:
                raise ValueError("IDEA source collection requires a trajectory id")
            self.idea_source_collection.begin_trajectory(trajectory_id)
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
        valid_action_mask = torch.isfinite(source_logits.detach()).detach()
        policy_inputs["invalid_action_mask"] = (~valid_action_mask).detach()
        if self.method == "eam":
            # EAM's entropy gate is defined against each decision's real
            # candidate set, not the padded tensor width.
            policy_inputs["valid_action_mask"] = valid_action_mask
            self.adapter.before_inference(
                policy_inputs=policy_inputs,
                valid_action_mask=valid_action_mask,
            )
        else:
            self.adapter.before_inference(policy_inputs=policy_inputs)
        source_logits = _sanitize_action_logits(source_logits)
        prepare_kwargs = {"policy_inputs": policy_inputs}
        if self.method == "eam":
            prepare_kwargs["valid_action_mask"] = valid_action_mask
        if self.idea_source_collection is not None:
            self.idea_source_collection.observe_step(policy_inputs)
            prepared = source_logits
        else:
            prepared = self.adapter.prepare_action(source_logits, **prepare_kwargs)
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

    def prompted_object_logits(self, source_object_logits):
        """Return IDEA's prompt-conditioned REVERIE logits when available."""
        if self.method != "idea" or self.idea_source_collection is not None:
            return source_object_logits
        protocol = getattr(self.adapter, "protocol", None)
        prompted = getattr(protocol, "prompted_object_logits", None)
        if prompted is None:
            raise RuntimeError(
                "REVERIE IDEA binding did not produce prompt-conditioned "
                "object logits"
            )
        if prompted.shape != source_object_logits.shape:
            raise RuntimeError(
                "prompted REVERIE object-logit shape changed: {} vs {}".format(
                    tuple(prompted.shape), tuple(source_object_logits.shape)
                )
            )
        return prompted.detach()

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
        discard_without_update = False
        if self.llm_feedback_provider is not None:
            if not isinstance(episode_stats, dict) or set(episode_stats) != {
                "_navtta_llm_pseudo_feedback"
            }:
                raise ValueError(
                    "FeedTTA-LLM requires one provider-bound post-episode result"
                )
            result = episode_stats["_navtta_llm_pseudo_feedback"]
            if not isinstance(result, dict) or result.get("available") not in (
                True, False
            ):
                raise ValueError("FeedTTA-LLM returned an invalid result")
            self.last_pseudo_feedback = result
            if result["available"]:
                if type(result.get("success")) is not bool:
                    raise ValueError("FeedTTA-LLM pseudo-label must be boolean")
                episode_stats = {"success": result["success"]}
            else:
                discard_without_update = True
                episode_stats = None
        if self.method in ("feedtta", "atena") and not discard_without_update:
            if episode_stats is None:
                if self.benchmark == "reverie":
                    raise ValueError(
                        "REVERIE binary-feedback TTA requires submitted-"
                        "trajectory evaluator success; simulator-distance "
                        "fallback is forbidden"
                    )
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
            elif self.benchmark == "reverie":
                expected_endpoint = (
                    (
                        "reverie_submitted_endpoint_panorama_llm_"
                        "pseudo_success_every_episode"
                    )
                    if self.llm_feedback_provider is not None else (
                        "reverie_submitted_trajectory_evaluator_navigation_"
                        "success_lazy_query"
                        if self.method == "atena" else
                        "reverie_submitted_trajectory_evaluator_navigation_"
                        "success_every_episode"
                    )
                )
                if self.binary_feedback_endpoint != expected_endpoint:
                    raise ValueError(
                        "REVERIE binary feedback did not come from the "
                        "submitted-trajectory navigation-success evaluator"
                    )
                if self.method == "atena" and not callable(episode_stats):
                    raise ValueError(
                        "REVERIE ATENA feedback must remain a lazy evaluator "
                        "callback until the entropy gate queries it"
                    )
                if self.method == "feedtta" and callable(episode_stats):
                    raise ValueError(
                        "REVERIE FeedTTA feedback must be evaluated eagerly "
                        "once per submitted trajectory"
                    )
        abort_after_cleanup = bool(
            discard_without_update
            and self.llm_feedback_provider is not None
            and self.llm_feedback_provider.abort_on_failure
        )
        if discard_without_update:
            self._discard_feedtta_episode_gradient()
        else:
            self.adapter.episode_end(episode_stats=episode_stats)
        if self.idea_source_collection is not None:
            self.idea_source_collection.end_trajectory()
        self._trajectory_hasher.update(b"episode_end\0")
        self._episode_open = False
        self.episode_count += 1
        self.write_diagnostics()
        if abort_after_cleanup:
            raise RuntimeError(
                "FeedTTA-LLM provider failed closed; no parameter update was "
                "performed and the formal run is invalid"
            )

    def write_diagnostics(self):
        diagnostics = self.diagnostics()
        atena_reachability_validated = (
            self.method == "atena"
            and diagnostics.get("replay_reachability_validated") is True
        )
        atena_replay_validated = (
            atena_reachability_validated
            and int(diagnostics.get(
                "replay_determinism_validated_episodes", 0
            )) > 0
        )
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
            "diagnostics_expected_episodes": self.diagnostics_expected_episodes,
            "action_steps": diagnostics["action_steps"],
            "batch_size": 1,
            "tta_step_unit": "high_level_navigation_decision",
            "action_selection": (
                "matched_policy_sampling_source_control"
                if self.method == "source" and self.matched_feedtta_source
                else (
                "policy_sampling"
                if self.action_selection == "sample"
                else "target_native_argmax"
                )
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
                "external_mllm_pseudo_feedback"
                if self.llm_feedback_provider is not None else (
                "binary_navigation_success_feedback"
                if self.method in ("feedtta", "atena")
                else "unsupervised"
                )
            ),
            "binary_feedback_endpoint": self.binary_feedback_endpoint,
            "feedback_provider": self.feedback_provider_name,
            "reported_method_label": (
                "FeedTTA-LLM"
                if self.llm_feedback_provider is not None else self.method.upper()
            ),
            "pseudo_feedback": (
                self.llm_feedback_provider.diagnostics()
                if self.llm_feedback_provider is not None else None
            ),
            "last_pseudo_feedback": self.last_pseudo_feedback,
            "failed_closed_feedback_episodes": (
                self.failed_closed_feedback_episodes
                if self.llm_feedback_provider is not None else 0
            ),
            "trainable_prefixes": list(self.trainable_prefixes),
            "feedtta_scope_profile": self.feedtta_scope_profile,
            "feedtta_sgr_mode": self.feedtta_sgr_mode,
            "feedtta_protocol": (
                "task_adapted_target_native_argmax"
                if self.feedtta_native_action_protocol else (
                    "paper_policy_sampling_ablation"
                    if self.method == "feedtta" else None
                )
            ),
            "feedtta_native_action_protocol": (
                self.feedtta_native_action_protocol
                if self.method == "feedtta" else None
            ),
            "feedtta_paper_sampling_protocol": (
                self.feedtta_paper_sampling_protocol
                if self.method == "feedtta" else None
            ),
            "feedtta_canonical_protocol": (
                self.feedtta_canonical_protocol
                if self.method == "feedtta" else None
            ),
            "matched_feedtta_source": self.matched_feedtta_source,
            "tent_canonical_update_interval": (
                int(self.args.tta_update_interval) == 1
                if self.method == "tent" else None
            ),
            "fstta_reset_var_hist_each_episode": (
                bool(self.args.tta_fstta_reset_var_hist_each_episode)
                if self.method == "fstta" else None
            ),
            "fstta_variance_history_profile": (
                "released_code_rollout_reset_ablation"
                if self.method == "fstta"
                and self.args.tta_fstta_reset_var_hist_each_episode
                else (
                    "paper_eq6_test_stream_history"
                    if self.method == "fstta" else None
                )
            ),
            "atena_update_scope": self.atena_update_scope,
            "atena_exact_replay_within_declared_scope": (
                atena_replay_validated if self.method == "atena" else None
            ),
            "atena_replay_reachability_validation_result": (
                diagnostics.get("replay_reachability_validation_result")
                if self.method == "atena" else None
            ),
            "atena_replay_reachable_parameter_count": (
                diagnostics.get("replay_reachable_parameter_count")
                if self.method == "atena" else None
            ),
            "atena_replay_reachable_parameter_names": (
                diagnostics.get("replay_reachable_parameter_names")
                if self.method == "atena" else None
            ),
            "atena_replay_unreachable_parameter_count": (
                diagnostics.get("replay_unreachable_parameter_count")
                if self.method == "atena" else None
            ),
            "atena_replay_unreachable_parameter_names": (
                diagnostics.get("replay_unreachable_parameter_names")
                if self.method == "atena" else None
            ),
            "atena_optimizer_scope_matches_reachable": (
                diagnostics.get("optimizer_policy_scope_matches_reachable")
                if self.method == "atena" else None
            ),
            "atena_full_end_to_end_policy_claimed": (
                False if self.method == "atena" else None
            ),
            "atena_upstream_feature_extractors_adapted": (
                False if self.method == "atena" else None
            ),
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
        matched_source = bool(
            getattr(self.args, "tta_matched_feedtta_source", False)
        )
        if (
            method == "source"
            and action_selection != "sample"
            and not matched_source
            and not audit_control
        ):
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
        feedback_provider = str(
            getattr(self.args, "tta_feedback_provider", "task_evaluator")
        ).lower()
        hidden_test = str(stream_name).lower() in ("test", "test_unseen")
        allowed_llm_proxy = (
            method == "feedtta"
            and hidden_test
            and feedback_provider == "qwen2_vl_2b_v1"
        )
        if method in ("feedtta", "atena") and hidden_test and not allowed_llm_proxy:
            raise ValueError(
                "{} consumes binary navigation-success feedback and cannot run "
                "on the unlabeled test split".format(method.upper())
            )
        if feedback_provider != "task_evaluator" and not allowed_llm_proxy:
            raise ValueError(
                "external pseudo-feedback is restricted to GOAT-REVERIE "
                "FeedTTA on the hidden test split"
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

    def tta_episode_start(self, trajectory_id=None):
        controller = getattr(self, "tta_controller", None)
        if controller is not None:
            controller.begin_episode(trajectory_id=trajectory_id)

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

    def tta_prompted_object_logits(self, source_object_logits):
        controller = getattr(self, "tta_controller", None)
        if controller is None:
            return source_object_logits
        return controller.prompted_object_logits(source_object_logits)

    def tta_graph_features(self, outputs):
        return _graph_decision_features(outputs)

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

    def tta_reverie_episode_stats(self, trajectories, path_format):
        """Return REVERIE navigation success at the submitted endpoint.

        The official REVERIE evaluator separates navigation success (the
        submitted endpoint can see the referred object) from remote grounding
        success (the predicted object id is correct).  FeedTTA and ATENA are
        permitted to consume only the former.  Graph agents may rerank their
        endpoint after the simulator stops, so the final submitted trajectory
        must be evaluated instead of reading ``observation['distance']``.
        """
        controller = getattr(self, "tta_controller", None)
        if controller is None or controller.method not in ("feedtta", "atena"):
            return None
        if getattr(controller, "llm_feedback_provider", None) is not None:
            # This branch must remain before every evaluator/ground-truth access.
            # It uses only instruction text, submitted trajectory identifiers,
            # and an explicitly rendered final-endpoint panorama.
            return controller.query_reverie_llm_feedback(
                self.env, trajectories, path_format
            )
        if len(trajectories) != 1:
            raise ValueError(
                "Canonical binary-feedback REVERIE TTA requires batch size one"
            )
        item = trajectories[0]
        instr_id = item["instr_id"]
        scan, ground_truth, ground_truth_object = self.env.gt_trajs[instr_id]

        if path_format == "nested_graph_path":
            evaluator_path = item["path"]
            predicted_object = item.get("pred_objid")

            def evaluate_submitted_endpoint():
                scores = self.env._eval_item(
                    scan,
                    evaluator_path,
                    predicted_object,
                    ground_truth,
                    ground_truth_object,
                )
                if "success" not in scores:
                    raise ValueError(
                        "REVERIE evaluator did not return navigation success"
                    )
                return {"success": float(scores["success"])}
        elif path_format == "viewpoint_tuples":
            evaluator_path = [step[0] for step in item["path"]]
            predicted_object = item.get("predObjId")

            def evaluate_submitted_endpoint():
                scores = self.env._eval_item(
                    scan,
                    evaluator_path,
                    ground_truth,
                    predicted_object,
                    ground_truth_object,
                )
                if "success" not in scores:
                    raise ValueError(
                        "REVERIE evaluator did not return navigation success"
                    )
                return {"success": float(scores["success"])}
        else:
            raise ValueError(
                "Unknown REVERIE trajectory format: {}".format(path_format)
            )

        # ATENA must not inspect the held-out outcome unless its entropy gate
        # spends a query.  FeedTTA consumes exactly one eager navigation label
        # after every completed submitted trajectory.
        if controller.method == "atena":
            controller.binary_feedback_endpoint = (
                "reverie_submitted_trajectory_evaluator_navigation_"
                "success_lazy_query"
            )
            return evaluate_submitted_endpoint
        controller.binary_feedback_endpoint = (
            "reverie_submitted_trajectory_evaluator_navigation_"
            "success_every_episode"
        )
        return evaluate_submitted_endpoint()
