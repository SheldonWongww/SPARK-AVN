import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class _RuntimeContext:
    def __init__(self):
        self.alpha: Optional[torch.Tensor] = None
        self.use_current_task: bool = False
        self.g_task: Optional[torch.Tensor] = None
        self.active_count: int = 0


class FuseLinear(nn.Module):
    def __init__(self, linear: nn.Linear, pool_size: int):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.pool_size = pool_size
        self.has_bias = linear.bias is not None

        self.register_buffer("base_weight", linear.weight.detach().clone())
        if self.has_bias:
            self.register_buffer("base_bias", linear.bias.detach().clone())
        else:
            self.base_bias = None

        self.register_buffer(
            "old_weights",
            torch.zeros(pool_size, self.out_features, self.in_features),
        )
        if self.has_bias:
            self.register_buffer(
                "old_biases",
                torch.zeros(pool_size, self.out_features),
            )
        else:
            self.old_biases = None

        self.current_weight = nn.Parameter(
            torch.zeros(self.out_features, self.in_features)
        )
        if self.has_bias:
            self.current_bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.current_bias = None

        self._runtime = _RuntimeContext()

    @classmethod
    def from_linear(cls, linear: nn.Linear, pool_size: int) -> "FuseLinear":
        fuse = cls(linear, pool_size)
        return fuse.to(linear.weight.device)

    def set_runtime_context(
        self,
        alpha: Optional[torch.Tensor],
        use_current_task: bool,
        g_task: Optional[torch.Tensor],
        active_count: int,
    ):
        self._runtime.alpha = alpha
        self._runtime.use_current_task = use_current_task
        self._runtime.g_task = g_task
        self._runtime.active_count = active_count

    def compose_weight_bias(
        self,
        alpha_override: Optional[torch.Tensor] = None,
        use_current_task_override: Optional[bool] = None,
        g_task_override: Optional[torch.Tensor] = None,
        active_count_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        alpha = self._runtime.alpha if alpha_override is None else alpha_override
        use_current_task = (
            self._runtime.use_current_task
            if use_current_task_override is None
            else bool(use_current_task_override)
        )
        g_task = self._runtime.g_task if g_task_override is None else g_task_override
        active_count = (
            min(self._runtime.active_count, self.pool_size)
            if active_count_override is None
            else min(int(active_count_override), self.pool_size)
        )

        g_scalar = 1.0
        if g_task is not None:
            g_scalar = (
                float(g_task.squeeze().detach())
                if not g_task.requires_grad
                else g_task.squeeze()
            )

        weight = self.base_weight
        bias = self.base_bias
        if alpha is not None:
            alpha = alpha.to(device=weight.device, dtype=weight.dtype)
            if alpha.dim() == 2:
                alpha = alpha.mean(dim=0)
            delta_w, delta_b = self._compute_old_delta(alpha, active_count)
            weight = weight + g_scalar * delta_w
            if self.has_bias and delta_b is not None:
                bias = bias + g_scalar * delta_b

        if use_current_task:
            weight = weight + self.current_weight
            if self.has_bias and self.current_bias is not None:
                bias = bias + self.current_bias
        return weight, bias

    @property
    def weight(self) -> torch.Tensor:
        weight, _ = self.compose_weight_bias()
        return weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        _, bias = self.compose_weight_bias()
        return bias

    def reset_current_vector(self):
        with torch.no_grad():
            self.current_weight.zero_()
            if self.current_bias is not None:
                self.current_bias.zero_()

    def get_current_vector(self) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "weight": self.current_weight.detach().clone(),
            "bias": None if self.current_bias is None else self.current_bias.detach().clone(),
        }

    def get_committed_vector(
        self,
        alpha_global: Optional[torch.Tensor],
        g_scalar: float,
        active_count: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        active_count = min(active_count, self.pool_size)
        with torch.no_grad():
            weight = self.current_weight.detach().clone()
            bias = None
            if self.current_bias is not None:
                bias = self.current_bias.detach().clone()

            if alpha_global is not None and active_count > 0:
                alpha = alpha_global.to(
                    device=self.base_weight.device,
                    dtype=self.base_weight.dtype,
                ).view(-1)
                delta_w, delta_b = self._compute_old_delta(alpha, active_count)
                weight = weight + float(g_scalar) * delta_w.detach()
                if bias is not None and delta_b is not None:
                    bias = bias + float(g_scalar) * delta_b.detach()

        return {"weight": weight, "bias": bias}

    def set_current_vector(self, vector: Dict[str, Optional[torch.Tensor]]):
        with torch.no_grad():
            self.current_weight.copy_(vector["weight"])
            if self.current_bias is not None and vector["bias"] is not None:
                self.current_bias.copy_(vector["bias"])

    def get_pool_vector(self, slot: int) -> Dict[str, Optional[torch.Tensor]]:
        result = {"weight": self.old_weights[slot].detach().clone(), "bias": None}
        if self.old_biases is not None:
            result["bias"] = self.old_biases[slot].detach().clone()
        return result

    def set_pool_vector(
        self,
        slot: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        with torch.no_grad():
            self.old_weights[slot].copy_(weight)
            if self.old_biases is not None:
                if bias is None:
                    self.old_biases[slot].zero_()
                else:
                    self.old_biases[slot].copy_(bias)

    def clear_pool_slot(self, slot: int):
        with torch.no_grad():
            self.old_weights[slot].zero_()
            if self.old_biases is not None:
                self.old_biases[slot].zero_()

    def _compute_old_delta(
        self, alpha: torch.Tensor, active_count: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if active_count <= 0:
            if alpha.dim() == 2:
                batch_size = alpha.size(0)
                delta_w = torch.zeros(
                    batch_size,
                    self.out_features,
                    self.in_features,
                    device=alpha.device,
                    dtype=self.base_weight.dtype,
                )
                delta_b = None
                if self.has_bias:
                    delta_b = torch.zeros(
                        batch_size,
                        self.out_features,
                        device=alpha.device,
                        dtype=self.base_weight.dtype,
                    )
                return delta_w, delta_b
            delta_w = torch.zeros_like(self.base_weight)
            delta_b = torch.zeros_like(self.base_bias) if self.has_bias else None
            return delta_w, delta_b

        weights = self.old_weights[:active_count]
        if alpha.dim() == 1:
            delta_w = torch.einsum("k,koi->oi", alpha[:active_count], weights)
            delta_b = None
            if self.has_bias:
                delta_b = torch.einsum("k,ko->o", alpha[:active_count], self.old_biases[:active_count])
            return delta_w, delta_b

        delta_w = torch.einsum("bk,koi->boi", alpha[:, :active_count], weights)
        delta_b = None
        if self.has_bias:
            delta_b = torch.einsum(
                "bk,ko->bo", alpha[:, :active_count], self.old_biases[:active_count]
            )
        return delta_w, delta_b

    def _infer_batch_dim(self, x: torch.Tensor, alpha_batch: int) -> Optional[int]:
        if x.dim() < 2:
            return None
        if x.dim() == 2:
            return 0 if x.size(0) == alpha_batch else None
        if x.dim() == 3:
            if x.size(1) == alpha_batch and x.size(0) != alpha_batch:
                return 1
            if x.size(0) == alpha_batch:
                return 0
            if x.size(1) == alpha_batch:
                return 1
            return None

        if x.size(0) == alpha_batch:
            return 0
        for dim in range(1, x.dim() - 1):
            if x.size(dim) == alpha_batch:
                return dim
        return None

    def _forward_with_batched_alpha(
        self,
        x: torch.Tensor,
        alpha: torch.Tensor,
        g_scalar,
        use_current_task: bool,
        active_count: int,
        batch_dim: int,
    ) -> torch.Tensor:
        x_perm = x.movedim(batch_dim, 0).contiguous()
        batch_size = x_perm.size(0)
        x_flat = x_perm.reshape(batch_size, -1, self.in_features)

        y_flat = F.linear(x_flat, self.base_weight, self.base_bias)
        if use_current_task:
            current_bias = self.current_bias if self.has_bias else None
            y_flat = y_flat + F.linear(x_flat, self.current_weight, current_bias)

        if active_count > 0:
            alpha = alpha[:, :active_count]
            for slot_idx in range(active_count):
                slot_bias = self.old_biases[slot_idx] if self.has_bias else None
                slot_out = F.linear(x_flat, self.old_weights[slot_idx], slot_bias)
                coeff = alpha[:, slot_idx]
                coeff = coeff * g_scalar
                y_flat = y_flat + coeff.view(batch_size, 1, 1) * slot_out

        y_perm = y_flat.reshape(*x_perm.shape[:-1], self.out_features)
        return y_perm.movedim(0, batch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self._runtime.alpha
        use_current_task = self._runtime.use_current_task
        g_task = self._runtime.g_task
        active_count = min(self._runtime.active_count, self.pool_size)

        g_scalar = 1.0
        if g_task is not None:
            g_scalar = float(g_task.squeeze().detach()) if not g_task.requires_grad else g_task.squeeze()

        if alpha is None:
            weight = self.base_weight
            bias = self.base_bias
            if use_current_task:
                weight = weight + self.current_weight
                if bias is not None and self.current_bias is not None:
                    bias = bias + self.current_bias
            return F.linear(x, weight, bias)

        alpha = alpha.to(device=x.device, dtype=self.base_weight.dtype)
        batch_dim = None
        if alpha.dim() == 2:
            batch_dim = self._infer_batch_dim(x, alpha.size(0))
            if batch_dim is None:
                alpha = alpha.mean(dim=0)

        if active_count <= 0:
            weight = self.base_weight
            bias = self.base_bias
            if use_current_task:
                weight = weight + self.current_weight
                if self.has_bias and self.current_bias is not None:
                    bias = bias + self.current_bias
            return F.linear(x, weight, bias)

        if alpha.dim() == 1:
            delta_w, delta_b = self._compute_old_delta(alpha, active_count)
            weight = self.base_weight + g_scalar * delta_w
            bias = self.base_bias
            if self.has_bias and delta_b is not None:
                bias = bias + g_scalar * delta_b
            if use_current_task:
                weight = weight + self.current_weight
                if self.has_bias:
                    bias = bias + self.current_bias
            return F.linear(x, weight, bias)
        if batch_dim is None:
            return F.linear(x, self.base_weight, self.base_bias)
        return self._forward_with_batched_alpha(
            x=x,
            alpha=alpha,
            g_scalar=g_scalar,
            use_current_task=use_current_task,
            active_count=active_count,
            batch_dim=batch_dim,
        )


class FuseConv1d(nn.Module):
    def __init__(self, conv: nn.Conv1d, pool_size: int):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.pool_size = pool_size
        self.has_bias = conv.bias is not None

        self.register_buffer("base_weight", conv.weight.detach().clone())
        if self.has_bias:
            self.register_buffer("base_bias", conv.bias.detach().clone())
        else:
            self.base_bias = None

        self.register_buffer(
            "old_weights",
            torch.zeros(pool_size, *conv.weight.shape),
        )
        if self.has_bias:
            self.register_buffer(
                "old_biases",
                torch.zeros(pool_size, self.out_channels),
            )
        else:
            self.old_biases = None

        self.current_weight = nn.Parameter(torch.zeros_like(conv.weight))
        if self.has_bias:
            self.current_bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.current_bias = None

        self._runtime = _RuntimeContext()

    @classmethod
    def from_conv1d(cls, conv: nn.Conv1d, pool_size: int) -> "FuseConv1d":
        fuse = cls(conv, pool_size)
        return fuse.to(conv.weight.device)

    def set_runtime_context(
        self,
        alpha: Optional[torch.Tensor],
        use_current_task: bool,
        g_task: Optional[torch.Tensor],
        active_count: int,
    ):
        self._runtime.alpha = alpha
        self._runtime.use_current_task = use_current_task
        self._runtime.g_task = g_task
        self._runtime.active_count = active_count

    def reset_current_vector(self):
        with torch.no_grad():
            self.current_weight.zero_()
            if self.current_bias is not None:
                self.current_bias.zero_()

    def get_current_vector(self) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "weight": self.current_weight.detach().clone(),
            "bias": None if self.current_bias is None else self.current_bias.detach().clone(),
        }

    def get_committed_vector(
        self,
        alpha_global: Optional[torch.Tensor],
        g_scalar: float,
        active_count: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        active_count = min(active_count, self.pool_size)
        with torch.no_grad():
            weight = self.current_weight.detach().clone()
            bias = None
            if self.current_bias is not None:
                bias = self.current_bias.detach().clone()

            if alpha_global is not None and active_count > 0:
                alpha = alpha_global.to(
                    device=self.base_weight.device,
                    dtype=self.base_weight.dtype,
                ).view(-1)
                delta_w, delta_b = self._compute_old_delta(alpha, active_count)
                weight = weight + float(g_scalar) * delta_w.detach()
                if bias is not None and delta_b is not None:
                    bias = bias + float(g_scalar) * delta_b.detach()

        return {"weight": weight, "bias": bias}

    def set_current_vector(self, vector: Dict[str, Optional[torch.Tensor]]):
        with torch.no_grad():
            self.current_weight.copy_(vector["weight"])
            if self.current_bias is not None and vector["bias"] is not None:
                self.current_bias.copy_(vector["bias"])

    def get_pool_vector(self, slot: int) -> Dict[str, Optional[torch.Tensor]]:
        result = {"weight": self.old_weights[slot].detach().clone(), "bias": None}
        if self.old_biases is not None:
            result["bias"] = self.old_biases[slot].detach().clone()
        return result

    def set_pool_vector(
        self,
        slot: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        with torch.no_grad():
            self.old_weights[slot].copy_(weight)
            if self.old_biases is not None:
                if bias is None:
                    self.old_biases[slot].zero_()
                else:
                    self.old_biases[slot].copy_(bias)

    def clear_pool_slot(self, slot: int):
        with torch.no_grad():
            self.old_weights[slot].zero_()
            if self.old_biases is not None:
                self.old_biases[slot].zero_()

    def _compute_old_delta(
        self, alpha: torch.Tensor, active_count: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if active_count <= 0:
            if alpha.dim() == 2:
                batch_size = alpha.size(0)
                delta_w = torch.zeros(
                    batch_size,
                    *self.base_weight.shape,
                    device=alpha.device,
                    dtype=self.base_weight.dtype,
                )
                delta_b = None
                if self.has_bias:
                    delta_b = torch.zeros(
                        batch_size,
                        self.out_channels,
                        device=alpha.device,
                        dtype=self.base_weight.dtype,
                    )
                return delta_w, delta_b
            delta_w = torch.zeros_like(self.base_weight)
            delta_b = torch.zeros_like(self.base_bias) if self.has_bias else None
            return delta_w, delta_b

        weights = self.old_weights[:active_count]
        if alpha.dim() == 1:
            delta_w = torch.einsum("k,k...->...", alpha[:active_count], weights)
            delta_b = None
            if self.has_bias:
                delta_b = torch.einsum(
                    "k,ko->o", alpha[:active_count], self.old_biases[:active_count]
                )
            return delta_w, delta_b

        delta_w = torch.einsum("bk,k...->b...", alpha[:, :active_count], weights)
        delta_b = None
        if self.has_bias:
            delta_b = torch.einsum(
                "bk,ko->bo", alpha[:, :active_count], self.old_biases[:active_count]
            )
        return delta_w, delta_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self._runtime.alpha
        use_current_task = self._runtime.use_current_task
        g_task = self._runtime.g_task
        active_count = min(self._runtime.active_count, self.pool_size)

        g_scalar = 1.0
        if g_task is not None:
            g_scalar = float(g_task.squeeze().detach()) if not g_task.requires_grad else g_task.squeeze()

        if alpha is None:
            weight = self.base_weight
            bias = self.base_bias
            if use_current_task:
                weight = weight + self.current_weight
                if self.has_bias:
                    bias = bias + self.current_bias
            return F.conv1d(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        alpha = alpha.to(device=x.device, dtype=self.base_weight.dtype)
        if alpha.dim() == 2 and alpha.size(0) != x.size(0):
            alpha = alpha.mean(dim=0)

        if active_count <= 0:
            weight = self.base_weight
            bias = self.base_bias
            if use_current_task:
                weight = weight + self.current_weight
                if self.has_bias and self.current_bias is not None:
                    bias = bias + self.current_bias
            return F.conv1d(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        delta_w, delta_b = self._compute_old_delta(alpha, active_count)

        if alpha.dim() == 1:
            weight = self.base_weight + g_scalar * delta_w
            bias = self.base_bias
            if self.has_bias and delta_b is not None:
                bias = bias + g_scalar * delta_b
            if use_current_task:
                weight = weight + self.current_weight
                if self.has_bias:
                    bias = bias + self.current_bias
            return F.conv1d(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        batch_size = x.size(0)
        weight = self.base_weight.unsqueeze(0) + g_scalar * delta_w
        if use_current_task:
            weight = weight + self.current_weight.unsqueeze(0)

        bias = None
        if self.has_bias:
            bias = self.base_bias.unsqueeze(0)
            if delta_b is not None:
                bias = bias + g_scalar * delta_b
            if use_current_task and self.current_bias is not None:
                bias = bias + self.current_bias.unsqueeze(0)
            bias = bias.reshape(batch_size * self.out_channels)

        x_grouped = x.contiguous().view(
            1,
            batch_size * self.in_channels,
            x.size(-1),
        )
        weight_grouped = weight.contiguous().view(
            batch_size * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size[0],
        )
        output = F.conv1d(
            x_grouped,
            weight_grouped,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=batch_size * self.groups,
        )
        return output.view(
            batch_size,
            self.out_channels,
            output.size(-1),
        )


class FuseLayerNorm(nn.Module):
    def __init__(self, layer_norm: nn.LayerNorm, pool_size: int):
        super().__init__()
        self.normalized_shape = tuple(layer_norm.normalized_shape)
        self.eps = layer_norm.eps
        self.pool_size = pool_size
        self.elementwise_affine = layer_norm.elementwise_affine
        self.has_bias = hasattr(layer_norm, "bias") and layer_norm.bias is not None

        if not self.elementwise_affine:
            self.base_weight = None
            self.base_bias = None
            self.old_weights = None
            self.old_biases = None
            self.current_weight = None
            self.current_bias = None
        else:
            self.register_buffer("base_weight", layer_norm.weight.detach().clone())
            if self.has_bias:
                self.register_buffer("base_bias", layer_norm.bias.detach().clone())
            else:
                self.base_bias = None

            self.register_buffer(
                "old_weights",
                torch.zeros(pool_size, *self.normalized_shape),
            )
            if self.has_bias:
                self.register_buffer(
                    "old_biases",
                    torch.zeros(pool_size, *self.normalized_shape),
                )
            else:
                self.old_biases = None

            self.current_weight = nn.Parameter(torch.zeros(*self.normalized_shape))
            if self.has_bias:
                self.current_bias = nn.Parameter(torch.zeros(*self.normalized_shape))
            else:
                self.current_bias = None

        self._runtime = _RuntimeContext()

    @classmethod
    def from_layer_norm(cls, layer_norm: nn.LayerNorm, pool_size: int) -> "FuseLayerNorm":
        fuse = cls(layer_norm, pool_size)
        device = None
        if layer_norm.elementwise_affine:
            device = layer_norm.weight.device
        return fuse if device is None else fuse.to(device)

    def set_runtime_context(
        self,
        alpha: Optional[torch.Tensor],
        use_current_task: bool,
        g_task: Optional[torch.Tensor],
        active_count: int,
    ):
        self._runtime.alpha = alpha
        self._runtime.use_current_task = use_current_task
        self._runtime.g_task = g_task
        self._runtime.active_count = active_count

    def compose_weight_bias(
        self,
        alpha_override: Optional[torch.Tensor] = None,
        use_current_task_override: Optional[bool] = None,
        g_task_override: Optional[torch.Tensor] = None,
        active_count_override: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.elementwise_affine:
            return None, None

        alpha = self._runtime.alpha if alpha_override is None else alpha_override
        use_current_task = (
            self._runtime.use_current_task
            if use_current_task_override is None
            else bool(use_current_task_override)
        )
        g_task = self._runtime.g_task if g_task_override is None else g_task_override
        active_count = (
            min(self._runtime.active_count, self.pool_size)
            if active_count_override is None
            else min(int(active_count_override), self.pool_size)
        )

        g_scalar = 1.0
        if g_task is not None:
            g_scalar = (
                float(g_task.squeeze().detach())
                if not g_task.requires_grad
                else g_task.squeeze()
            )

        weight = self.base_weight
        bias = self.base_bias
        if alpha is not None:
            alpha = alpha.to(device=weight.device, dtype=weight.dtype)
            if alpha.dim() == 2:
                alpha = alpha.mean(dim=0)
            delta_w, delta_b = self._compute_old_delta(alpha, active_count)
            weight = weight + g_scalar * delta_w
            if self.has_bias and delta_b is not None:
                bias = bias + g_scalar * delta_b

        if use_current_task:
            weight = weight + self.current_weight
            if self.has_bias and self.current_bias is not None:
                bias = bias + self.current_bias

        return weight, bias

    @property
    def weight(self) -> Optional[torch.Tensor]:
        weight, _ = self.compose_weight_bias()
        return weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        _, bias = self.compose_weight_bias()
        return bias

    def reset_current_vector(self):
        if not self.elementwise_affine:
            return
        with torch.no_grad():
            self.current_weight.zero_()
            if self.current_bias is not None:
                self.current_bias.zero_()

    def get_current_vector(self) -> Dict[str, Optional[torch.Tensor]]:
        if not self.elementwise_affine:
            return {"weight": None, "bias": None}
        return {
            "weight": self.current_weight.detach().clone(),
            "bias": None if self.current_bias is None else self.current_bias.detach().clone(),
        }

    def get_committed_vector(
        self,
        alpha_global: Optional[torch.Tensor],
        g_scalar: float,
        active_count: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if not self.elementwise_affine:
            return {"weight": None, "bias": None}

        active_count = min(active_count, self.pool_size)
        with torch.no_grad():
            weight = self.current_weight.detach().clone()
            bias = None
            if self.current_bias is not None:
                bias = self.current_bias.detach().clone()

            if alpha_global is not None and active_count > 0:
                alpha = alpha_global.to(
                    device=self.base_weight.device,
                    dtype=self.base_weight.dtype,
                ).view(-1)
                delta_w, delta_b = self._compute_old_delta(alpha, active_count)
                weight = weight + float(g_scalar) * delta_w.detach()
                if bias is not None and delta_b is not None:
                    bias = bias + float(g_scalar) * delta_b.detach()

        return {"weight": weight, "bias": bias}

    def set_current_vector(self, vector: Dict[str, Optional[torch.Tensor]]):
        if not self.elementwise_affine:
            return
        with torch.no_grad():
            if vector["weight"] is not None:
                self.current_weight.copy_(vector["weight"])
            if self.current_bias is not None and vector["bias"] is not None:
                self.current_bias.copy_(vector["bias"])

    def get_pool_vector(self, slot: int) -> Dict[str, Optional[torch.Tensor]]:
        if not self.elementwise_affine:
            return {"weight": None, "bias": None}
        result = {"weight": self.old_weights[slot].detach().clone(), "bias": None}
        if self.old_biases is not None:
            result["bias"] = self.old_biases[slot].detach().clone()
        return result

    def set_pool_vector(
        self,
        slot: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        if not self.elementwise_affine:
            return
        with torch.no_grad():
            self.old_weights[slot].copy_(weight)
            if self.old_biases is not None:
                if bias is None:
                    self.old_biases[slot].zero_()
                else:
                    self.old_biases[slot].copy_(bias)

    def clear_pool_slot(self, slot: int):
        if not self.elementwise_affine:
            return
        with torch.no_grad():
            self.old_weights[slot].zero_()
            if self.old_biases is not None:
                self.old_biases[slot].zero_()

    def _compute_old_delta(
        self, alpha: torch.Tensor, active_count: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if active_count <= 0:
            if alpha.dim() == 2:
                batch_size = alpha.size(0)
                delta_w = torch.zeros(
                    batch_size,
                    *self.base_weight.shape,
                    device=alpha.device,
                    dtype=self.base_weight.dtype,
                )
                delta_b = None
                if self.has_bias:
                    delta_b = torch.zeros(
                        batch_size,
                        *self.base_bias.shape,
                        device=alpha.device,
                        dtype=self.base_weight.dtype,
                    )
                return delta_w, delta_b
            delta_w = torch.zeros_like(self.base_weight)
            delta_b = torch.zeros_like(self.base_bias) if self.has_bias else None
            return delta_w, delta_b

        weights = self.old_weights[:active_count]
        if alpha.dim() == 1:
            delta_w = torch.einsum("k,k...->...", alpha[:active_count], weights)
            delta_b = None
            if self.has_bias:
                delta_b = torch.einsum(
                    "k,k...->...", alpha[:active_count], self.old_biases[:active_count]
                )
            return delta_w, delta_b

        delta_w = torch.einsum("bk,k...->b...", alpha[:, :active_count], weights)
        delta_b = None
        if self.has_bias:
            delta_b = torch.einsum(
                "bk,k...->b...", alpha[:, :active_count], self.old_biases[:active_count]
            )
        return delta_w, delta_b

    def _infer_batch_dim(self, x: torch.Tensor, alpha_batch: int) -> Optional[int]:
        if x.dim() < 2:
            return None
        if x.dim() == 2:
            return 0 if x.size(0) == alpha_batch else None
        if x.dim() == 3:
            if x.size(1) == alpha_batch and x.size(0) != alpha_batch:
                return 1
            if x.size(0) == alpha_batch:
                return 0
            if x.size(1) == alpha_batch:
                return 1
            return None

        if x.size(0) == alpha_batch:
            return 0
        for dim in range(1, x.dim() - len(self.normalized_shape)):
            if x.size(dim) == alpha_batch:
                return dim
        return None

    def _forward_with_batched_alpha(
        self,
        x: torch.Tensor,
        delta_w: torch.Tensor,
        delta_b: Optional[torch.Tensor],
        g_scalar,
        use_current_task: bool,
        batch_dim: int,
    ) -> torch.Tensor:
        x_perm = x.movedim(batch_dim, 0)
        batch_size = x_perm.size(0)
        norm_dims = len(self.normalized_shape)

        reduce_dims = tuple(range(x_perm.dim() - norm_dims, x_perm.dim()))
        mean = x_perm.mean(dim=reduce_dims, keepdim=True)
        var = x_perm.var(dim=reduce_dims, unbiased=False, keepdim=True)
        x_norm = (x_perm - mean) / torch.sqrt(var + self.eps)

        weight = self.base_weight.unsqueeze(0).expand(batch_size, *self.normalized_shape)
        weight = weight + g_scalar * delta_w
        if use_current_task:
            weight = weight + self.current_weight.unsqueeze(0)

        bias = None
        if self.has_bias:
            bias = self.base_bias.unsqueeze(0).expand(batch_size, *self.normalized_shape)
            if delta_b is not None:
                bias = bias + g_scalar * delta_b
            if use_current_task and self.current_bias is not None:
                bias = bias + self.current_bias.unsqueeze(0)

        expand_shape = [batch_size] + [1] * (x_perm.dim() - 1 - norm_dims) + list(self.normalized_shape)
        y_perm = x_norm * weight.view(*expand_shape)
        if bias is not None:
            y_perm = y_perm + bias.view(*expand_shape)

        return y_perm.movedim(0, batch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.elementwise_affine:
            return F.layer_norm(
                x,
                self.normalized_shape,
                None,
                None,
                self.eps,
            )

        alpha = self._runtime.alpha
        use_current_task = self._runtime.use_current_task
        g_task = self._runtime.g_task
        active_count = min(self._runtime.active_count, self.pool_size)

        g_scalar = 1.0
        if g_task is not None:
            g_scalar = float(g_task.squeeze().detach()) if not g_task.requires_grad else g_task.squeeze()

        if alpha is None:
            weight = self.base_weight
            bias = self.base_bias
            if use_current_task:
                weight = weight + self.current_weight
                if self.has_bias and self.current_bias is not None:
                    bias = bias + self.current_bias
            return F.layer_norm(
                x,
                self.normalized_shape,
                weight,
                bias,
                self.eps,
            )

        alpha = alpha.to(device=x.device, dtype=self.base_weight.dtype)
        batch_dim = None
        if alpha.dim() == 2:
            batch_dim = self._infer_batch_dim(x, alpha.size(0))
            if batch_dim is None:
                alpha = alpha.mean(dim=0)

        if active_count <= 0:
            weight = self.base_weight
            bias = self.base_bias
            if use_current_task:
                weight = weight + self.current_weight
                if self.has_bias and self.current_bias is not None:
                    bias = bias + self.current_bias
            return F.layer_norm(
                x,
                self.normalized_shape,
                weight,
                bias,
                self.eps,
            )

        delta_w, delta_b = self._compute_old_delta(alpha, active_count)

        if alpha.dim() == 1:
            weight = self.base_weight + g_scalar * delta_w
            bias = self.base_bias
            if self.has_bias and delta_b is not None:
                bias = bias + g_scalar * delta_b
            if use_current_task:
                weight = weight + self.current_weight
                if self.has_bias and self.current_bias is not None:
                    bias = bias + self.current_bias
            return F.layer_norm(
                x,
                self.normalized_shape,
                weight,
                bias,
                self.eps,
            )

        if batch_dim is None:
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.base_weight,
                self.base_bias,
                self.eps,
            )
        return self._forward_with_batched_alpha(
            x=x,
            delta_w=delta_w,
            delta_b=delta_b,
            g_scalar=g_scalar,
            use_current_task=use_current_task,
            batch_dim=batch_dim,
        )


class FuseConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, pool_size: int):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.pool_size = pool_size
        self.has_bias = conv.bias is not None

        self.register_buffer("base_weight", conv.weight.detach().clone())
        if self.has_bias:
            self.register_buffer("base_bias", conv.bias.detach().clone())
        else:
            self.base_bias = None

        self.register_buffer(
            "old_weights",
            torch.zeros(pool_size, *conv.weight.shape),
        )
        if self.has_bias:
            self.register_buffer(
                "old_biases",
                torch.zeros(pool_size, self.out_channels),
            )
        else:
            self.old_biases = None

        self.current_weight = nn.Parameter(torch.zeros_like(conv.weight))
        if self.has_bias:
            self.current_bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.current_bias = None

        self._runtime = _RuntimeContext()

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, pool_size: int) -> "FuseConv2d":
        fuse = cls(conv, pool_size)
        return fuse.to(conv.weight.device)

    def set_runtime_context(
        self,
        alpha: Optional[torch.Tensor],
        use_current_task: bool,
        g_task: Optional[torch.Tensor],
        active_count: int,
    ):
        self._runtime.alpha = alpha
        self._runtime.use_current_task = use_current_task
        self._runtime.g_task = g_task
        self._runtime.active_count = active_count

    def reset_current_vector(self):
        with torch.no_grad():
            self.current_weight.zero_()
            if self.current_bias is not None:
                self.current_bias.zero_()

    def get_current_vector(self) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "weight": self.current_weight.detach().clone(),
            "bias": None if self.current_bias is None else self.current_bias.detach().clone(),
        }

    def get_committed_vector(
        self,
        alpha_global: Optional[torch.Tensor],
        g_scalar: float,
        active_count: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        active_count = min(active_count, self.pool_size)
        with torch.no_grad():
            weight = self.current_weight.detach().clone()
            bias = None
            if self.current_bias is not None:
                bias = self.current_bias.detach().clone()

            if alpha_global is not None and active_count > 0:
                alpha = alpha_global.to(
                    device=self.base_weight.device,
                    dtype=self.base_weight.dtype,
                ).view(-1)
                delta_w, delta_b = self._compute_old_delta(alpha, active_count)
                weight = weight + float(g_scalar) * delta_w.detach()
                if bias is not None and delta_b is not None:
                    bias = bias + float(g_scalar) * delta_b.detach()

        return {"weight": weight, "bias": bias}

    def set_current_vector(self, vector: Dict[str, Optional[torch.Tensor]]):
        with torch.no_grad():
            self.current_weight.copy_(vector["weight"])
            if self.current_bias is not None and vector["bias"] is not None:
                self.current_bias.copy_(vector["bias"])

    def get_pool_vector(self, slot: int) -> Dict[str, Optional[torch.Tensor]]:
        result = {"weight": self.old_weights[slot].detach().clone(), "bias": None}
        if self.old_biases is not None:
            result["bias"] = self.old_biases[slot].detach().clone()
        return result

    def set_pool_vector(
        self,
        slot: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        with torch.no_grad():
            self.old_weights[slot].copy_(weight)
            if self.old_biases is not None:
                if bias is None:
                    self.old_biases[slot].zero_()
                else:
                    self.old_biases[slot].copy_(bias)

    def clear_pool_slot(self, slot: int):
        with torch.no_grad():
            self.old_weights[slot].zero_()
            if self.old_biases is not None:
                self.old_biases[slot].zero_()

    def _compute_old_delta(
        self, alpha: torch.Tensor, active_count: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if active_count <= 0:
            if alpha.dim() == 2:
                batch_size = alpha.size(0)
                delta_w = torch.zeros(
                    batch_size,
                    *self.base_weight.shape,
                    device=alpha.device,
                    dtype=self.base_weight.dtype,
                )
                delta_b = None
                if self.has_bias:
                    delta_b = torch.zeros(
                        batch_size,
                        self.out_channels,
                        device=alpha.device,
                        dtype=self.base_weight.dtype,
                    )
                return delta_w, delta_b
            delta_w = torch.zeros_like(self.base_weight)
            delta_b = torch.zeros_like(self.base_bias) if self.has_bias else None
            return delta_w, delta_b

        weights = self.old_weights[:active_count]
        if alpha.dim() == 1:
            delta_w = torch.einsum("k,kocij->ocij", alpha[:active_count], weights)
            delta_b = None
            if self.has_bias:
                delta_b = torch.einsum(
                    "k,ko->o", alpha[:active_count], self.old_biases[:active_count]
                )
            return delta_w, delta_b

        delta_w = torch.einsum("bk,kocij->bocij", alpha[:, :active_count], weights)
        delta_b = None
        if self.has_bias:
            delta_b = torch.einsum(
                "bk,ko->bo", alpha[:, :active_count], self.old_biases[:active_count]
            )
        return delta_w, delta_b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self._runtime.alpha
        use_current_task = self._runtime.use_current_task
        g_task = self._runtime.g_task
        active_count = min(self._runtime.active_count, self.pool_size)

        g_scalar = 1.0
        if g_task is not None:
            g_scalar = float(g_task.squeeze().detach()) if not g_task.requires_grad else g_task.squeeze()

        if alpha is None:
            weight = self.base_weight
            bias = self.base_bias
            if use_current_task:
                weight = weight + self.current_weight
                if self.has_bias:
                    bias = bias + self.current_bias
            return F.conv2d(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        alpha = alpha.to(device=x.device, dtype=self.base_weight.dtype)
        if alpha.dim() == 2 and alpha.size(0) != x.size(0):
            alpha = alpha.mean(dim=0)

        if active_count <= 0:
            weight = self.base_weight
            bias = self.base_bias
            if use_current_task:
                weight = weight + self.current_weight
                if self.has_bias and self.current_bias is not None:
                    bias = bias + self.current_bias
            return F.conv2d(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        delta_w, delta_b = self._compute_old_delta(alpha, active_count)

        if alpha.dim() == 1:
            weight = self.base_weight + g_scalar * delta_w
            bias = self.base_bias
            if self.has_bias and delta_b is not None:
                bias = bias + g_scalar * delta_b
            if use_current_task:
                weight = weight + self.current_weight
                if self.has_bias:
                    bias = bias + self.current_bias
            return F.conv2d(
                x,
                weight,
                bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )

        batch_size = x.size(0)
        weight = self.base_weight.unsqueeze(0) + g_scalar * delta_w
        if use_current_task:
            weight = weight + self.current_weight.unsqueeze(0)

        bias = None
        if self.has_bias:
            bias = self.base_bias.unsqueeze(0)
            if delta_b is not None:
                bias = bias + g_scalar * delta_b
            if use_current_task and self.current_bias is not None:
                bias = bias + self.current_bias.unsqueeze(0)
            bias = bias.reshape(batch_size * self.out_channels)

        x_grouped = x.contiguous().view(
            1,
            batch_size * self.in_channels,
            x.size(-2),
            x.size(-1),
        )
        weight_grouped = weight.contiguous().view(
            batch_size * self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        output = F.conv2d(
            x_grouped,
            weight_grouped,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=batch_size * self.groups,
        )
        return output.view(
            batch_size,
            self.out_channels,
            output.size(-2),
            output.size(-1),
        )
