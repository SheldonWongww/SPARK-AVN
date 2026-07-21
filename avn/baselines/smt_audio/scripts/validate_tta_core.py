#!/usr/bin/env python3
"""Lightweight smoke test for the shared Tent/FSTTA implementation."""
from copy import deepcopy

import torch
import torch.nn as nn

from navtta_core.tta.tta_core import (
    FSTTAAdapter,
    TentAdapter,
    configure_tta_model,
)


class ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(8) for _ in range(6)])
        self.head = nn.Linear(8, 4)

    def forward(self, inputs):
        features = inputs
        for norm in self.norms:
            features = torch.tanh(norm(features))
        return self.head(features)


def test_tent_zero_lr():
    torch.manual_seed(0)
    model = ToyPolicy()
    source = deepcopy(model.state_dict())
    adapter = TentAdapter(model, lr=0.0, scope="last_k_ln", last_k=4)
    assert len(adapter.names) == 8, adapter.names
    logits = model(torch.randn(2, 8))
    adapter.adapt(logits)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, source[name]), name


def test_tent_layernorm_scopes():
    expected = {
        "first_ln": ["norms.0.weight", "norms.0.bias"],
        "last_ln": ["norms.5.weight", "norms.5.bias"],
        "last_k_ln": [
            "norms.2.weight", "norms.2.bias",
            "norms.3.weight", "norms.3.bias",
            "norms.4.weight", "norms.4.bias",
            "norms.5.weight", "norms.5.bias",
        ],
        "ln": [
            "{}.{}".format(module, parameter)
            for module in ["norms.{}".format(index) for index in range(6)]
            for parameter in ("weight", "bias")
        ],
    }
    for scope, expected_names in expected.items():
        model = ToyPolicy()
        _, names = configure_tta_model(model, scope=scope, last_k=4)
        assert names == expected_names, (scope, names)


def test_fstta_episode_schedule():
    torch.manual_seed(1)
    model = ToyPolicy()
    adapter = FSTTAAdapter(
        model,
        lr_fast=1e-4,
        lr_slow=1e-4,
        M=2,
        N=2,
        q=0.1,
        scope="last_k_ln",
        last_k=4,
    )
    for _ in range(2):
        adapter.episode_start()
        for _ in range(4):
            adapter.adapt(model(torch.randn(2, 8)))
        adapter.episode_end()
    diagnostics = adapter.diagnostics()
    assert diagnostics["episodes"] == 2, diagnostics
    assert diagnostics["updates"] == 4, diagnostics
    assert diagnostics["slow_updates"] == 1, diagnostics


if __name__ == "__main__":
    test_tent_zero_lr()
    test_tent_layernorm_scopes()
    test_fstta_episode_schedule()
    print("TTA core validation passed")
