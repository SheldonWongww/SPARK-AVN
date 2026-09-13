"""Keep ENMuS ATENA inference and gradient replay on the same eval backend."""

from contextlib import nullcontext

import torch


def policy_forward_context(method):
    """Build an autograd-capable eval graph without enabling RNN dropout.

    ENMuS contains a GRU and an LSTM. cuDNN's eval RNN forward does not save
    the state required for backward. ATENA keeps the whole policy in eval
    mode, so both its no-grad action forward and its gradient replay must use
    the native backend. Switching only replay risks violating exact replay;
    switching the RNNs to train mode would enable the GRU's dropout.

    The backend change is scoped to this forward, including on exceptions.
    Other methods retain their existing backend and module modes.
    """
    if method == "atena":
        return torch.backends.cudnn.flags(enabled=False)
    return nullcontext()


def forward_policy_for_replay(model, policy_inputs):
    """ATENA callback shared by reachability preflight and episode replay."""
    with policy_forward_context("atena"):
        features, _, _ = model.net(
            policy_inputs["observations"],
            policy_inputs["rnn_hidden_states"],
            policy_inputs["prev_actions"],
            policy_inputs["masks"],
            policy_inputs.get("ext_memory"),
            policy_inputs.get("ext_memory_masks"),
        )
        logits = model.action_distribution(features).logits
    return features, logits
