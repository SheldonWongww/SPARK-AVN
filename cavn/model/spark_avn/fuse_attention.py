import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cavn.model.transformer_decoder import PoolingAttention
from cavn.model.spark_avn.fuse_layers import FuseLinear


def _slice_attn_mask_for_sample(
    attn_mask: Optional[torch.Tensor],
    sample_idx: int,
    num_heads: int,
    batch_size: int,
) -> Optional[torch.Tensor]:
    if attn_mask is None or attn_mask.dim() != 3:
        return attn_mask
    if attn_mask.size(0) == 1:
        return attn_mask
    if attn_mask.size(0) == batch_size * num_heads:
        begin = sample_idx * num_heads
        end = (sample_idx + 1) * num_heads
        return attn_mask[begin:end]
    return attn_mask


class FuseMultiheadAttention(nn.Module):
    def __init__(
        self,
        mha: nn.MultiheadAttention,
        pool_size: int,
        fuse_out_proj: bool = True,
        batched_alpha_mode: str = "loop",
    ):
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.dropout = float(mha.dropout)
        self.batch_first = mha.batch_first
        self.head_dim = mha.head_dim
        self.kdim = mha.kdim
        self.vdim = mha.vdim
        self._qkv_same_embed_dim = mha._qkv_same_embed_dim
        self.add_zero_attn = mha.add_zero_attn
        self.batched_alpha_mode = batched_alpha_mode

        if mha.bias_k is None:
            self.bias_k = None
        else:
            self.register_buffer("bias_k", mha.bias_k.detach().clone())
        if mha.bias_v is None:
            self.bias_v = None
        else:
            self.register_buffer("bias_v", mha.bias_v.detach().clone())

        in_proj = nn.Linear(
            self.embed_dim,
            3 * self.embed_dim,
            bias=mha.in_proj_bias is not None,
        )
        with torch.no_grad():
            in_proj.weight.copy_(mha.in_proj_weight.detach())
            if mha.in_proj_bias is not None:
                in_proj.bias.copy_(mha.in_proj_bias.detach())
        self.in_proj = FuseLinear.from_linear(in_proj, pool_size)

        if fuse_out_proj:
            self.out_proj = FuseLinear.from_linear(mha.out_proj, pool_size)
        else:
            self.out_proj = copy.deepcopy(mha.out_proj)
            for param in self.out_proj.parameters():
                param.requires_grad = False

    @classmethod
    def from_multihead_attention(
        cls,
        mha: nn.MultiheadAttention,
        pool_size: int,
        fuse_out_proj: bool = True,
        batched_alpha_mode: str = "loop",
    ) -> "FuseMultiheadAttention":
        module = cls(
            mha=mha,
            pool_size=pool_size,
            fuse_out_proj=fuse_out_proj,
            batched_alpha_mode=batched_alpha_mode,
        )
        return module.to(mha.out_proj.weight.device)

    @property
    def in_proj_weight(self) -> torch.Tensor:
        weight, _ = self.in_proj.compose_weight_bias()
        return weight

    @property
    def in_proj_bias(self) -> Optional[torch.Tensor]:
        _, bias = self.in_proj.compose_weight_bias()
        return bias

    def _normalize_alpha(
        self, alpha: Optional[torch.Tensor], batch_size: int
    ) -> Optional[torch.Tensor]:
        if alpha is None:
            return None
        if alpha.dim() == 1:
            return alpha
        if alpha.dim() == 2:
            if alpha.size(0) == batch_size:
                return alpha
            return alpha.mean(dim=0)
        return alpha

    def _single_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        need_weights: bool,
        attn_mask: Optional[torch.Tensor],
        average_attn_weights: bool,
        is_causal: bool,
        alpha_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        in_proj_weight, in_proj_bias = self.in_proj.compose_weight_bias(
            alpha_override=alpha_override
        )
        if isinstance(self.out_proj, FuseLinear):
            out_proj_weight, out_proj_bias = self.out_proj.compose_weight_bias(
                alpha_override=alpha_override
            )
        else:
            out_proj_weight = self.out_proj.weight
            out_proj_bias = self.out_proj.bias

        return F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=in_proj_weight,
            in_proj_bias=in_proj_bias,
            bias_k=self.bias_k,
            bias_v=self.bias_v,
            add_zero_attn=self.add_zero_attn,
            dropout_p=self.dropout,
            out_proj_weight=out_proj_weight,
            out_proj_bias=out_proj_bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        is_batched = query.dim() == 3
        transposed = False
        if self.batch_first and is_batched:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            transposed = True

        if not is_batched:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        batch_size = query.size(1)
        alpha = self._normalize_alpha(self.in_proj._runtime.alpha, batch_size)

        if alpha is not None and alpha.dim() == 2:
            if self.batched_alpha_mode != "loop":
                raise ValueError(
                    f"Unsupported batched alpha mode: {self.batched_alpha_mode}"
                )
            outputs = []
            weights = [] if need_weights else None
            for sample_idx in range(batch_size):
                q_i = query[:, sample_idx: sample_idx + 1, :]
                k_i = key[:, sample_idx: sample_idx + 1, :]
                v_i = value[:, sample_idx: sample_idx + 1, :]
                kpm_i = None
                if key_padding_mask is not None and key_padding_mask.dim() == 2:
                    kpm_i = key_padding_mask[sample_idx: sample_idx + 1]
                am_i = _slice_attn_mask_for_sample(
                    attn_mask, sample_idx, self.num_heads, batch_size
                )
                out_i, w_i = self._single_forward(
                    q_i,
                    k_i,
                    v_i,
                    key_padding_mask=kpm_i,
                    need_weights=need_weights,
                    attn_mask=am_i,
                    average_attn_weights=average_attn_weights,
                    is_causal=is_causal,
                    alpha_override=alpha[sample_idx],
                )
                outputs.append(out_i)
                if need_weights and weights is not None:
                    weights.append(w_i)
            attn_output = torch.cat(outputs, dim=1)
            attn_output_weights = None
            if need_weights and weights is not None and len(weights) > 0:
                attn_output_weights = torch.cat(weights, dim=0)
        else:
            alpha_override = alpha if alpha is not None and alpha.dim() == 1 else None
            attn_output, attn_output_weights = self._single_forward(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                alpha_override=alpha_override,
            )

        if not is_batched:
            attn_output = attn_output.squeeze(1)
        if transposed:
            attn_output = attn_output.transpose(0, 1)

        return attn_output, attn_output_weights


class FusePoolingAttention(nn.Module):
    def __init__(
        self,
        attn: PoolingAttention,
        pool_size: int,
        fuse_out_proj: bool = True,
        batched_alpha_mode: str = "loop",
    ):
        super().__init__()
        self.embed_dim = attn.embed_dim
        self.num_heads = attn.num_heads
        self.dropout = float(attn.dropout)
        self.batch_first = attn.batch_first
        self.head_dim = attn.head_dim
        self.pool_ratios = list(attn.pool_ratios)
        self.batched_alpha_mode = batched_alpha_mode

        in_proj = nn.Linear(
            self.embed_dim,
            3 * self.embed_dim,
            bias=attn.in_proj_bias is not None,
        )
        with torch.no_grad():
            in_proj.weight.copy_(attn.in_proj_weight.detach())
            if attn.in_proj_bias is not None:
                in_proj.bias.copy_(attn.in_proj_bias.detach())
        self.in_proj = FuseLinear.from_linear(in_proj, pool_size)

        if fuse_out_proj:
            self.out_proj = FuseLinear.from_linear(attn.out_proj, pool_size)
        else:
            self.out_proj = copy.deepcopy(attn.out_proj)
            for param in self.out_proj.parameters():
                param.requires_grad = False

        self.conv_construct = attn.conv_construct

    @classmethod
    def from_pooling_attention(
        cls,
        attn: PoolingAttention,
        pool_size: int,
        fuse_out_proj: bool = True,
        batched_alpha_mode: str = "loop",
    ) -> "FusePoolingAttention":
        module = cls(
            attn=attn,
            pool_size=pool_size,
            fuse_out_proj=fuse_out_proj,
            batched_alpha_mode=batched_alpha_mode,
        )
        return module.to(attn.out_proj.weight.device)

    @property
    def in_proj_weight(self) -> torch.Tensor:
        weight, _ = self.in_proj.compose_weight_bias()
        return weight

    @property
    def in_proj_bias(self) -> Optional[torch.Tensor]:
        _, bias = self.in_proj.compose_weight_bias()
        return bias

    def get_pyinput(self, key: torch.Tensor) -> torch.Tensor:
        conv_keys = self.conv_construct(key)
        return conv_keys.permute(1, 0, 2).contiguous()

    def _normalize_alpha(
        self, alpha: Optional[torch.Tensor], batch_size: int
    ) -> Optional[torch.Tensor]:
        if alpha is None:
            return None
        if alpha.dim() == 1:
            return alpha
        if alpha.dim() == 2:
            if alpha.size(0) == batch_size:
                return alpha
            return alpha.mean(dim=0)
        return alpha

    def _project_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        alpha_override: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if key.shape != value.shape:
            raise NotImplementedError(
                "PoolingAttention requires key/value with identical shapes"
            )

        in_proj_weight, in_proj_bias = self.in_proj.compose_weight_bias(
            alpha_override=alpha_override
        )
        embed_dim = query.size(-1)
        if query is key:
            proj = F.linear(query, in_proj_weight, in_proj_bias)
            return proj.chunk(3, dim=-1)

        w_q, w_kv = in_proj_weight.split([embed_dim, 2 * embed_dim], dim=0)
        if in_proj_bias is None:
            b_q = b_kv = None
        else:
            b_q, b_kv = in_proj_bias.split([embed_dim, 2 * embed_dim], dim=0)
        q = F.linear(query, w_q, b_q)
        kv = F.linear(key, w_kv, b_kv)
        k, v = kv.chunk(2, dim=-1)
        return q, k, v

    def _single_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        alpha_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if is_causal:
            raise NotImplementedError("PoolingAttention does not support causal attention")
        if key.shape != value.shape:
            raise AssertionError("PoolingAttention requires key/value with identical shapes")
        value = key
        assert key.shape[0] % self.pool_ratios[-1] == 0, (
            f"PoolingAttention only supports sequence length divisible by {self.pool_ratios[-1]}"
        )

        q, k, v = self._project_qkv(
            query=query,
            key=key,
            value=value,
            alpha_override=alpha_override,
        )

        k = k.permute(1, 2, 0)
        v = k = self.get_pyinput(k)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The size of the 2D attn_mask is {attn_mask.size()}, but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The size of the 3D attn_mask is {attn_mask.size()}, but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError("attn_mask's dimension must be 2 or 3")

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), (
                "key_padding_mask shape is not correct with size {}, but the correct shape should be {}"
            ).format(key_padding_mask.shape, (bsz, src_len))
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(
                -1, self.num_heads, -1, -1
            ).reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        if need_weights:
            _, _, head_dim = q.shape
            q_scaled = q / (head_dim ** 0.5)
            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            if self.dropout > 0.0:
                attn_output_weights = F.dropout(attn_output_weights, p=self.dropout)

            attn_output = torch.bmm(attn_output_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            if isinstance(self.out_proj, FuseLinear):
                out_w, out_b = self.out_proj.compose_weight_bias(alpha_override=alpha_override)
                attn_output = F.linear(attn_output, out_w, out_b)
            else:
                attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)
            return attn_output, attn_output_weights

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout,
            is_causal=is_causal,
        )
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        if isinstance(self.out_proj, FuseLinear):
            out_w, out_b = self.out_proj.compose_weight_bias(alpha_override=alpha_override)
            attn_output = F.linear(attn_output, out_w, out_b)
        else:
            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        return attn_output, None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        is_batched = query.dim() == 3
        if not is_batched:
            raise NotImplementedError("PoolingAttention only supports batched inputs")
        if self.batch_first:
            raise NotImplementedError("PoolingAttention only supports batch_first=False")

        alpha = self._normalize_alpha(self.in_proj._runtime.alpha, query.size(1))
        if alpha is not None and alpha.dim() == 2:
            if self.batched_alpha_mode != "loop":
                raise ValueError(
                    f"Unsupported batched alpha mode: {self.batched_alpha_mode}"
                )
            outputs = []
            weights = [] if need_weights else None
            batch_size = query.size(1)
            for sample_idx in range(batch_size):
                q_i = query[:, sample_idx: sample_idx + 1, :]
                k_i = key[:, sample_idx: sample_idx + 1, :]
                v_i = k_i
                kpm_i = None
                if key_padding_mask is not None and key_padding_mask.dim() == 2:
                    kpm_i = key_padding_mask[sample_idx: sample_idx + 1]
                am_i = _slice_attn_mask_for_sample(
                    attn_mask, sample_idx, self.num_heads, batch_size
                )
                out_i, w_i = self._single_forward(
                    query=q_i,
                    key=k_i,
                    value=v_i,
                    key_padding_mask=kpm_i,
                    need_weights=need_weights,
                    attn_mask=am_i,
                    average_attn_weights=average_attn_weights,
                    is_causal=is_causal,
                    alpha_override=alpha[sample_idx],
                )
                outputs.append(out_i)
                if need_weights and weights is not None:
                    weights.append(w_i)
            attn_output = torch.cat(outputs, dim=1)
            attn_output_weights = None
            if need_weights and weights is not None and len(weights) > 0:
                attn_output_weights = torch.cat(weights, dim=0)
            return attn_output, attn_output_weights

        alpha_override = alpha if alpha is not None and alpha.dim() == 1 else None
        return self._single_forward(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
            alpha_override=alpha_override,
        )
