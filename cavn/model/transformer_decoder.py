# cavn/model/transformer_decoder.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List
from torch.nn.modules.activation import MultiheadAttention

from cavn.common.utils import Swish, GLU, Transpose, PointwiseConv1d, DepthwiseConv1d, ConvConstruct

class PoolingAttention(MultiheadAttention):
    """Convolutional Pooling Attention Module in MSMT Transformer Decoder Layer (Branch 1)"""
    
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, 
                 add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        super(PoolingAttention, self).__init__(
            embed_dim, num_heads, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn, kdim=kdim, vdim=vdim, batch_first=batch_first, device=device, dtype=dtype
        )

        self.pool_ratios = [4, 8, 16, 32]
        self.head_dim = embed_dim // num_heads
        self.conv_construct = ConvConstruct(embed_dim, window_size=[self.pool_ratios[0], 2, 2, 2])

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3

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

        if not is_batched:
            raise NotImplementedError("PoolingAttention only supports batched inputs")
        if self.batch_first:
            raise NotImplementedError("PoolingAttention only supports batch_first=False")
        if not self._qkv_same_embed_dim:
            raise NotImplementedError("PoolingAttention only supports q, k, v with same feature dim")
         
        attn_output, attn_output_weights = self._single_forward(
            query, key, value, key_padding_mask, need_weights, attn_mask, average_attn_weights, is_causal
        )

        return attn_output, attn_output_weights
    
    def _single_forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:        
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
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
        assert key is value, "PoolingAttention only supports key is value"
        assert key.shape[0] % self.pool_ratios[-1] == 0, \
            f"PoolingAttention only supports sequence length divisible by {self.pool_ratios[-1]}"
        
        # Embed First, Pyramid Later
        q, k, v = self._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias) # [M, B, dim]
        k = k.permute(1, 2, 0) # [B, dim, M]
        v = k = self.get_pyinput(k) # [multi_scale_memory_length, B, dim]

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The size of the 2D attn_mask is {attn_mask.size()}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_2d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The size of the 3D attn_mask is {attn_mask.size()}, but should be {correct_2d_size}.")
            else:
                raise RuntimeError("attn_mask's dimension must be 2 or 3")
        
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(k.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(v.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
            "key_padding_mask shape is not correct with size {}, but the correct shape should be {}".format(key_padding_mask.shape, (bsz, src_len))
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        if need_weights:
            B, Nt, E = q.shape
            q_scaled = q / math.sqrt(E)
            if attn_mask is not None:
                attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
            else:
                attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            if self.dropout > 0.0:
                attn_output_weights = F.dropout(attn_output_weights, p=self.dropout)

            attn_output = torch.bmm(attn_output_weights, v)
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.mean(dim=1)

            return attn_output, attn_output_weights
        else:
            if attn_mask is not None:
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                    attn_mask = attn_mask.unsqueeze(0)
                else:
                    attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)
            
            q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
            k = k.view(bsz, self.num_heads, src_len, self.head_dim)
            v = v.view(bsz, self.num_heads, src_len, self.head_dim)

            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask, self.dropout, is_causal)
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

            attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

            return attn_output, None

    def _in_projection_packed(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor,
            w: Tensor,
            b: Optional[Tensor] = None,
    ) -> List[Tensor]:
        E = q.size(-1) # embed_dim
        if k is v:
            if q is k:
                # self-attention
                proj = F.linear(q, w, b)
                proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
                return proj[0], proj[1], proj[2]
            else:
                w_q, w_kv = w.split([E, E * 2])
                if b is None:
                    b_q = b_kv = None
                else:
                    b_q, b_kv = b.split([E, E * 2])
                q_proj = F.linear(q, w_q, b_q)                           # [..., E]
                kv_proj = F.linear(k, w_kv, b_kv)                        # [..., 2E]
                kv_proj = F.linear(k, w_kv, b_kv).unflatten(-1, (2, E))  # [..., 2, E]
                kv_proj = kv_proj.unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
                return (q_proj, kv_proj[0], kv_proj[1])
        else:
            raise NotImplementedError("PoolingAttention only supports k is v ")
    
    def get_pyinput(self, key):        
        conv_keys = self.conv_construct(key) # [B, multi_scale_memory_length, dim]
        return conv_keys.permute(1, 0, 2).contiguous() # [multi_scale_memory_length, B, dim]

class FeedForwardModule(nn.Module):
    """ Feed Forward module in MSMT Transformer Decoder Layer """
    def __init__(self, d_model: int = 512, expansion_factor: int = 4, dropout_p: float = 0.1):
        super(FeedForwardModule, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            Swish(),
            nn.Dropout(dropout_p),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.sequential(x)

class MSMTConvModule(nn.Module):
    """ Convolution module in MSMT Transformer Decoder Layer (Branch 2)"""
    def __init__(
            self, 
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ):
        super(MSMTConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be an odd number"
        assert expansion_factor == 2, "expansion_factor should be 2"

        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)), # [B, 1(T), 256(C)] -> [B, 256(C), 1(T)]
            PointwiseConv1d(in_channels, in_channels * expansion_factor), # [B, 512, 1]
            GLU(dim=1), # [B, 256, 1]
            # kernel_size=31, padding=15
            DepthwiseConv1d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1) // 2), # [B, 256, 1]
            nn.BatchNorm1d(in_channels), # [B, 256, 1]
            Swish(), # [B, 256, 1]
            PointwiseConv1d(in_channels, in_channels), # [B, 256, 1]
            nn.Dropout(dropout_p),
        )

    def forward(self, x):
        return self.sequential(x).transpose(1, 2)

class MSMTDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_attention_heads: int,
            feed_forward_expansion_factor: int,
            conv_expansion_factor: int,
            feed_forward_dropout_p: float,
            attention_dropout_p: float,
            conv_dropout_p: float,
            conv_kernel_size: int,
            half_step_residual: bool = True,
            norm_first: bool = True,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MSMTDecoderLayer, self).__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1.0

        self.norm_first = norm_first

        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_attention_heads,
            dropout=attention_dropout_p,
            bias = True,
            **factory_kwargs,
        )

        self.multi_head_attn = PoolingAttention(
            embed_dim=d_model,
            num_heads=num_attention_heads,
            dropout=attention_dropout_p,
            bias=True,
        )

        self.self_attn_dropout = nn.Dropout(attention_dropout_p)
        self.multi_head_attn_dropout = nn.Dropout(attention_dropout_p)

        self.linear1 = FeedForwardModule(
            d_model=d_model,
            expansion_factor=feed_forward_expansion_factor,
            dropout_p=feed_forward_dropout_p,
        )
        self.linear2 = nn.Sequential(
            nn.Linear(d_model*2, d_model*4),
            Swish(),
            nn.Dropout(feed_forward_dropout_p),
            nn.Linear(d_model*4, d_model*2),
            nn.Dropout(feed_forward_dropout_p),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.ReLU(True),
        )

        self.conv = MSMTConvModule(
            in_channels=d_model,
            kernel_size=conv_kernel_size,
            expansion_factor=conv_expansion_factor,
            dropout_p=conv_dropout_p,
        )

        self.norm1 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm4 = nn.LayerNorm(d_model, **factory_kwargs)
        self.norm5 = nn.LayerNorm(d_model, **factory_kwargs)

        if not norm_first:
            self.norm5 = nn.LayerNorm(d_model*2)

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
            tgt_is_causal: bool = False,
            memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._ff1_block(self.norm1(x)) * self.feed_forward_residual_factor
            x = x + self._sa_block(self.norm2(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x1 = x

            x = x + self._mha_block(self.norm3(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal) # cross-
            x1 = x1 + self._conv_block(self.norm4(x1)) # convolutional module
            
            x = torch.cat([x, x1], dim=-1)
            x = x + self._ff2_block(self.norm5(x)) * self.feed_forward_residual_factor

            x = self.linear3(x)
        else:
            """ Actually enter this branch """
            x = self.norm1(x + self._ff1_block(x) * self.feed_forward_residual_factor)
            x = self.norm2(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x1 = x

            x = self.norm3(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x1 = self.norm4(x1 + self._conv_block(x1))

            x = torch.cat([x, x1], dim=-1)
            x = self.norm5(x + self._ff2_block(x) * self.feed_forward_residual_factor)

            x = self.linear3(x)

        return x


    def _sa_block(
            self,
            x: Tensor,
            attn_mask: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
    ) -> Tensor:
        """ Self-attention block + dropout """
        x = self.self_attn(x, x, x, 
                           attn_mask=attn_mask, 
                           key_padding_mask=key_padding_mask, 
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.self_attn_dropout(x)

    def _mha_block(
            self,
            x: Tensor,
            memory: Tensor,
            attn_mask: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
    ) -> Tensor:
        """ Multi-head cross-attention block + dropout """
        x = self.multi_head_attn(x, memory, memory,
                                 attn_mask=attn_mask, 
                                 key_padding_mask=key_padding_mask,
                                 is_causal=is_causal,
                                 need_weights=False)[0]
        return self.multi_head_attn_dropout(x)

    def _ff1_block(self, x: Tensor) -> Tensor:
        return self.linear1(x)

    def _ff2_block(self, x: Tensor) -> Tensor:
        return self.linear2(x)

    def _conv_block(self, x: Tensor) -> Tensor:
        return self.conv(x)

class MSMTDecoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(MSMTDecoder, self).__init__()
        self.layers = nn.modules.transformer._get_clones(decoder_layer, num_layers) # 1 layer MSMTDecoderLayer
        self.num_layers = num_layers # 1
        self.norm = norm # nn.LayerNorm

    def forward(self, tgt: Tensor, memory: Tensor, 
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None, 
                tgt_key_padding_mask: Optional[Tensor] = None, 
                memory_key_padding_mask: Optional[Tensor] = None, 
                tgt_is_causal: bool = False, 
                memory_is_causal: bool = False
    ) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, 
                         memory_mask=memory_mask, 
                         tgt_key_padding_mask=tgt_key_padding_mask, 
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal = tgt_is_causal,
                         memory_is_causal=memory_is_causal
                    )

        if self.norm is not None:
            output = self.norm(output)
        
        return output
