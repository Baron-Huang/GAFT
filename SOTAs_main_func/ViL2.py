import collections
import itertools
import math

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# adapted from timm (timm/models/layers/helpers.py)
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            assert len(x) == n
            return x
        return tuple(itertools.repeat(x, n))

    return parse


# adapted from timm (timm/models/layers/helpers.py)
def to_ntuple(x, n):
    return _ntuple(n=n)(x)


# from kappamodules.functional.pos_embed import interpolate_sincos
def interpolate_sincos(embed, seqlens, mode="bicubic"):
    assert embed.ndim - 2 == len(seqlens)
    embed = F.interpolate(
        einops.rearrange(embed, "1 ... dim -> 1 dim ..."),
        size=seqlens,
        mode=mode,
    )
    embed = einops.rearrange(embed, "1 dim ... -> 1 ... dim")
    return embed


class SequenceConv2d(nn.Conv2d):
    def __init__(self, *args, seqlens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seqlens = seqlens

    def forward(self, x):
        assert x.ndim == 3
        if self.seqlens is None:
            # assuming square input
            h = math.sqrt(x.size(1))
            assert h.is_integer()
            h = int(h)
        else:
            assert len(self.seqlens) == 2
            h = self.seqlens[0]
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=h)
        x = super().forward(x)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        return x


# from kappamodules.vit import VitPatchEmbed
class VitPatchEmbed(nn.Module):
    def __init__(self, dim, num_channels, resolution, patch_size, stride=None, init_weights="xavier_uniform"):
        super().__init__()
        self.resolution = resolution
        self.init_weights = init_weights
        self.ndim = len(resolution)
        self.patch_size = to_ntuple(patch_size, n=self.ndim)
        if stride is None:
            self.stride = self.patch_size
        else:
            self.stride = to_ntuple(stride, n=self.ndim)
        for i in range(self.ndim):
            assert resolution[i] % self.patch_size[i] == 0, \
                f"resolution[{i}] % patch_size[{i}] != 0 (resolution={resolution} patch_size={patch_size})"
        self.seqlens = [resolution[i] // self.patch_size[i] for i in range(self.ndim)]
        if self.patch_size == self.stride:
            # use primitive type as np.prod gives np.int which is not compatible with all serialization/logging
            self.num_patches = int(np.prod(self.seqlens))
        else:
            if self.ndim == 1:
                conv_func = F.conv1d
            elif self.ndim == 2:
                conv_func = F.conv2d
            elif self.ndim == 3:
                conv_func = F.conv3d
            else:
                raise NotImplementedError
            self.num_patches = conv_func(
                input=torch.zeros(1, 1, *resolution),
                weight=torch.zeros(1, 1, *self.patch_size),
                stride=self.stride,
            ).numel()

        if self.ndim == 1:
            conv_ctor = nn.Conv1d
        elif self.ndim == 2:
            conv_ctor = nn.Conv2d
        elif self.ndim == 3:
            conv_ctor = nn.Conv3d
        else:
            raise NotImplementedError

        self.proj = conv_ctor(num_channels, dim, kernel_size=self.patch_size, stride=self.stride)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            # initialize as nn.Linear
            w = self.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.zeros_(self.proj.bias)
        else:
            raise NotImplementedError

    def forward(self, x):
        assert all(x.size(i + 2) % self.patch_size[i] == 0 for i in range(self.ndim)), \
            f"x.shape={x.shape} incompatible with patch_size={self.patch_size}"
        x = self.proj(x)
        x = einops.rearrange(x, "b c ... -> b ... c")
        return x


# from kappamodules.vit import VitPosEmbed2d
class VitPosEmbed2d(nn.Module):
    def __init__(self, seqlens, dim: int, allow_interpolation: bool = True):
        super().__init__()
        self.seqlens = seqlens
        self.dim = dim
        self.allow_interpolation = allow_interpolation
        self.embed = nn.Parameter(torch.zeros(1, *seqlens, dim))
        self.reset_parameters()

    @property
    def _expected_x_ndim(self):
        return len(self.seqlens) + 2

    def reset_parameters(self):
        nn.init.trunc_normal_(self.embed, std=.02)

    def forward(self, x):
        assert x.ndim == self._expected_x_ndim
        if x.shape[1:] != self.embed.shape[1:]:
            assert self.allow_interpolation
            embed = interpolate_sincos(embed=self.embed, seqlens=x.shape[1:-1])
        else:
            embed = self.embed
        return x + embed


# from kappamodules.layers import DropPath
class DropPath(nn.Sequential):
    """
    Efficiently drop paths (Stochastic Depth) per sample such that dropped samples are not processed.
    This is a subclass of nn.Sequential and can be used either as standalone Module or like nn.Sequential.
    Examples::
        >>> # use as nn.Sequential module
        >>> sequential_droppath = DropPath(nn.Linear(4, 4), drop_prob=0.2)
        >>> y = sequential_droppath(torch.randn(10, 4))

        >>> # use as standalone module
        >>> standalone_layer = nn.Linear(4, 4)
        >>> standalone_droppath = DropPath(drop_prob=0.2)
        >>> y = standalone_droppath(torch.randn(10, 4), standalone_layer)
    """

    def __init__(
            self,
            *args,
            drop_prob: float = 0.,
            scale_by_keep: bool = True,
            stochastic_drop_prob: bool = False,
            drop_prob_tolerance: float = 0.01,
    ):
        super().__init__(*args)
        assert 0. <= drop_prob < 1.
        self._drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.stochastic_drop_prob = stochastic_drop_prob
        self.drop_prob_tolerance = drop_prob_tolerance

    @property
    def drop_prob(self):
        return self._drop_prob

    @drop_prob.setter
    def drop_prob(self, value):
        assert 0. <= value < 1.
        self._drop_prob = value

    @property
    def keep_prob(self):
        return 1. - self.drop_prob

    def forward(self, x, residual_path=None, residual_path_kwargs=None):
        assert (len(self) == 0) ^ (residual_path is None)
        residual_path_kwargs = residual_path_kwargs or {}
        if self.drop_prob == 0. or not self.training:
            if residual_path is None:
                return x + super().forward(x, **residual_path_kwargs)
            else:
                return x + residual_path(x, **residual_path_kwargs)
        bs = len(x)
        # for small batchsizes its not possible to do it efficiently
        # e.g. batchsize 2 with drop_rate=0.05 would drop 1 sample and therefore increase the drop_rate to 0.5
        # resolution: fall back to inefficient version
        keep_count = max(int(bs * self.keep_prob), 1)
        # allow some level of tolerance
        actual_keep_prob = keep_count / bs
        drop_path_delta = self.keep_prob - actual_keep_prob
        # if drop_path_delta > self.drop_prob_tolerance:
        #     warnings.warn(
        #         f"efficient stochastic depth (DropPath) would change drop_path_rate by {drop_path_delta:.4f} "
        #         f"because the batchsize is too small to accurately drop {bs - keep_count} samples per forward pass"
        #         f" -> forcing stochastic_drop_prob=True drop_path_rate={self.drop_prob}"
        #     )

        # inefficient drop_path
        if self.stochastic_drop_prob or drop_path_delta > self.drop_prob_tolerance:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = x.new_empty(shape).bernoulli_(self.keep_prob)
            if self.scale_by_keep:
                random_tensor.div_(self.keep_prob)
            if residual_path is None:
                return x + super().forward(x, **residual_path_kwargs) * random_tensor
            else:
                return x + residual_path(x, **residual_path_kwargs) * random_tensor

        # generate indices to keep (propagated through transform path)
        scale = bs / keep_count
        perm = torch.randperm(bs, device=x.device)[:keep_count]

        # propagate
        if self.scale_by_keep:
            alpha = scale
        else:
            alpha = 1.
        # reduce kwargs (e.g. used for DiT block where scale/shift/gate is passed and also has to be reduced)
        residual_path_kwargs = {
            key: value[perm] if torch.is_tensor(value) else value
            for key, value in residual_path_kwargs.items()
        }
        if residual_path is None:
            residual = super().forward(x[perm], **residual_path_kwargs)
        else:
            residual = residual_path(x[perm], **residual_path_kwargs)
        return torch.index_add(
            x.flatten(start_dim=1),
            dim=0,
            index=perm,
            source=residual.to(x.dtype).flatten(start_dim=1),
            alpha=alpha,
        ).view_as(x)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'

# This file is licensed under Apache-2.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Benedikt Alkin, Maximilian Beck, Korbinian Pöppel
import math
import warnings
from enum import Enum

import einops
import torch
import torch.nn.functional as F
from torch import nn

# Utility classes are embedded above in this standalone script.


class SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """ Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py. """
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def parallel_stabilized_simple(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        lower_triangular_matrix: torch.Tensor = None,
        stabilize_rowwise: bool = True,
        eps: float = 1e-6,
) -> torch.Tensor:
    """
    This is the mLSTM cell in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        :param queries: (torch.Tensor) (B, NH, S, DH)
        :param keys: (torch.Tensor) (B, NH, S, DH)
        :param values: (torch.Tensor) (B, NH, S, DH)
        :param igate_preact: (torch.Tensor) (B, NH, S, 1)
        :param fgate_preact: (torch.Tensor) (B, NH, S, 1)
        :param lower_triangular_matrix: (torch.Tensor) (S,S). Defaults to None.
        :param stabilize_rowwise: (bool) Wether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.
        :param eps: (float) small constant to avoid division by 0. Defaults to 1e-6.

    Returns:
        torch.Tensor: (B, NH, S, DH), h_tilde_state
    """

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # forget gate matrix
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"

    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = keys / math.sqrt(DH)

    # combination matrix C
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # retrieved values
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)

    return h_tilde_state


class LinearHeadwiseExpand(nn.Module):
    """
    This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        dim_per_head = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)
        x = einops.einsum(
            x,
            self.weight,
            "... nh d, nh out_d d -> ... nh out_d",
        )
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias is not None}, "
        )


class CausalConv1d(nn.Module):
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.bias = bias
        # padding of this size assures temporal causality.
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=dim,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv requires dim first
        x = einops.rearrange(x, "b l d -> b d l")
        # causal conv1d
        x = self.conv(x)
        x = x[:, :, :-self.pad]
        # back to dim last
        x = einops.rearrange(x, "b d l -> b l d")
        return x


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False. """

    def __init__(
            self,
            ndim: int = -1,
            weight: bool = True,
            bias: bool = False,
            eps: float = 1e-5,
            residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = x.shape

        gn_in_1 = x.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )  # .to(x.dtype)
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out


class MatrixLSTMCell(nn.Module):
    def __init__(self, dim, num_heads, norm_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.igate = nn.Linear(3 * dim, num_heads)
        self.fgate = nn.Linear(3 * dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.causal_mask_cache = {}
        self.reset_parameters()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, S, _ = q.shape  # (B, S, H)

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

        # cache causal mask to avoid memory allocation in every iteration
        if S in self.causal_mask_cache:
            causal_mask = self.causal_mask_cache[(S, str(q.device))]
        else:
            causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
            self.causal_mask_cache[(S, str(q.device))] = causal_mask

        h_state = parallel_stabilized_simple(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
        )  # (B, NH, S, DH)

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        # forget gate initialization
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)


class ViLLayer(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            expansion=2,
            qkv_block_size=4,
            proj_bias=True,
            norm_bias=True,
            conv_bias=True,
            conv_kernel_size=4,
            conv_kind="2d",
            init_weights="original",
            seqlens=None,
            num_blocks=None,
    ):
        super().__init__()
        assert dim % qkv_block_size == 0
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.conv_kernel_size = conv_kernel_size
        self.conv_kind = conv_kind
        self.init_weights = init_weights
        self.num_blocks = num_blocks

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.proj_up = nn.Linear(
            in_features=dim,
            out_features=2 * inner_dim,
            bias=proj_bias,
        )
        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )

        if conv_kind == "causal1d":
            self.conv = CausalConv1d(
                dim=inner_dim,
                kernel_size=conv_kernel_size,
                bias=conv_bias,
            )
        elif conv_kind == "2d":
            assert conv_kernel_size % 2 == 1, \
                f"same output shape as input shape is required -> even kernel sizes not supported"
            self.conv = SequenceConv2d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=inner_dim,
                bias=conv_bias,
                seqlens=seqlens,
            )
        else:
            raise NotImplementedError
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=qkv_block_size,
            norm_bias=norm_bias,
        )
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))

        self.proj_down = nn.Linear(
            in_features=inner_dim,
            out_features=dim,
            bias=proj_bias,
        )
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        # alternate direction in successive layers
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        # up-projection
        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)

        # mlstm branch
        x_mlstm_conv = self.conv(x_mlstm)
        x_mlstm_conv_act = F.silu(x_mlstm_conv)
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * F.silu(z)

        # down-projection
        x = self.proj_down(h_state)

        # reverse alternating flip
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        return x

    def reset_parameters(self):
        # init inproj
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        # init outproj (original mLSTM uses num_blocks=1)
        if self.init_weights == "original":
            wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        elif self.init_weights == "original-fixed":
            wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=self.num_blocks)
        else:
            raise NotImplementedError
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()


class ViLBlock(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            drop_path=0.0,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
            num_blocks=None,
            init_weights="original",
    ):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.drop_path = drop_path
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.init_weights = init_weights

        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.layer = ViLLayer(
            dim=dim,
            direction=direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            norm_bias=norm_bias,
            proj_bias=proj_bias,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )

        self.reset_parameters()

    def _forward_path(self, x):
        x = self.norm(x)
        x = self.layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_path(x, self._forward_path)
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()


class ViLBlockPair(nn.Module):
    def __init__(
            self,
            dim,
            drop_path=0.0,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
            num_blocks=None,
            init_weights="original",
    ):
        super().__init__()
        self.rowwise_from_top_left = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )
        self.rowwise_from_bot_right = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )

    def forward(self, x):
        x = self.rowwise_from_top_left(x)
        x = self.rowwise_from_bot_right(x)
        return x


class VisionLSTM2(nn.Module):
    def __init__(
            self,
            dim=192,
            input_shape=(3, 224, 224),
            patch_size=16,
            depth=12,
            output_shape=(1000,),
            mode="classifier",
            pooling="bilateral_flatten",
            drop_path_rate=0.0,
            drop_path_decay=False,
            stride=None,
            legacy_norm=False,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            init_weights="original",
    ):
        if depth == 24 and dim < 1024:
            warnings.warn(
                "A single VisionLSTM2 block consists of two subblocks (one for each traversal direction). "
                "ViL-T, ViL-S and ViL-B therefore use depth=12 instead of depth=24, are you sure you want to use "
                "depth=24?"
            )
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        ndim = len(self.input_shape) - 1
        self.patch_size = to_ntuple(patch_size, n=ndim)
        self.dim = dim
        self.depth = depth
        self.stride = stride
        self.mode = mode
        self.pooling = pooling
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.proj_bias = proj_bias
        self.norm_bias = norm_bias
        self.init_weights = init_weights

        # initialize patch_embed
        self.patch_embed = VitPatchEmbed(
            dim=dim,
            stride=stride,
            num_channels=self.input_shape[0],
            resolution=self.input_shape[1:],
            patch_size=self.patch_size,
        )

        # pos embed
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim)

        # calculate stochastic depth per block
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        # merge two blocks into a blockpair to keep depth equal to the depth of transformers
        # useful to keep layer-wise lr decay implementations consistent with transformers
        self.blocks = nn.ModuleList(
            [
                ViLBlockPair(
                    dim=dim,
                    drop_path=dpr[i],
                    conv_kind=conv_kind,
                    seqlens=self.patch_embed.seqlens,
                    proj_bias=proj_bias,
                    norm_bias=norm_bias,
                    num_blocks=depth * 2,
                    init_weights=init_weights,
                )
                for i in range(depth)
            ],
        )
        if pooling == "bilateral_flatten" and mode == "classifier":
            head_dim = dim * 2
        else:
            head_dim = dim
        self.norm = LayerNorm(dim, bias=norm_bias, eps=1e-6)
        # LEGACY: not needed but was used during training
        if legacy_norm:
            self.legacy_norm = nn.LayerNorm(head_dim)
        else:
            self.legacy_norm = nn.Identity()

        # head
        if mode == "features":
            if self.output_shape is not None:
                warnings.warn(f"passed mode=features -> output_shape is ignored ({self.output_shape})")
            self.head = None
            if self.pooling is None:
                self.output_shape = (self.patch_embed.num_patches, dim)
            elif self.pooling == "to_image":
                self.output_shape = (dim, *self.patch_embed.seqlens)
            else:
                warnings.warn(f"passed invalid pooling -> pooling is ignored ({self.pooling})")
                self.pooling = None
        elif mode == "classifier":
            # linear classification head
            assert self.output_shape is not None and len(self.output_shape) == 1, \
                f"define number of classes via output_shape=(num_classes,) (e.g. output_shape=(1000,) for ImageNet-1K"
            self.head = nn.Linear(head_dim, self.output_shape[0])
            # following MAE https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L257
            nn.init.trunc_normal_(self.head.weight, std=2e-5)
            nn.init.zeros_(self.head.bias)
        else:
            raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        # interpolate pos_embed for different resolution (e.g. for fine-tuning on higher-resolution)
        old_pos_embed = state_dict["pos_embed.embed"]
        if old_pos_embed.shape != self.pos_embed.embed.shape:
            state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        # remove head and adapt layernorm for feature extraction
        if self.mode == "features":
            state_dict.pop("head.weight", None)
            state_dict.pop("head.bias", None)
            # legacy_norm uses head dim (is doubled for bilateral_concat) -> not usable for feature extraction
            cur_sd = self.state_dict()
            state_dict["legacy_norm.weight"] = cur_sd["legacy_norm.weight"]
            state_dict["legacy_norm.bias"] = cur_sd["legacy_norm.bias"]
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed.embed"}

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)
        # add pos_embed
        x = self.pos_embed(x)

        # flatten to 1d
        x = einops.rearrange(x, "b ... d -> b (...) d")

        # apply blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # pool
        if self.pooling is None:
            x = self.legacy_norm(x)
        elif self.pooling == "to_image":
            x = self.legacy_norm(x)
            seqlen_h, seqlen_w = self.patch_embed.seqlens
            x = einops.rearrange(
                x,
                "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
                seqlen_h=seqlen_h,
                seqlen_w=seqlen_w,
            )
        elif self.pooling == "bilateral_avg":
            # norm after pooling
            x = (x[:, 0] + x[:, -1]) / 2
            x = self.legacy_norm(x)
        elif self.pooling == "bilateral_flatten":
            # norm after pooling
            x = torch.concat([x[:, 0], x[:, -1]], dim=1)
            x = self.legacy_norm(x)
        else:
            raise NotImplementedError(f"pooling '{self.pooling}' is not implemented")

        # head
        if self.head is not None:
            x = self.head(x)

        return x
# -----------------------------------------------------------------------------
# GAFT training entry point.  The official ViL2 backbone and utilities above
# are embedded in this file; only einops is required as an external package.
# -----------------------------------------------------------------------------
import os
import random
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from torchvision.datasets import ImageFolder

_codes_root = str(Path(__file__).resolve().parent.parent)
if _codes_root not in sys.path:
    sys.path.insert(0, _codes_root)

from SIL_Utils.fit_functions import Single_out_fit, testing_funnction


class ViL2Classifier(nn.Module):
    """ViL2-Base ImageNet backbone with a GAFT_SwinT_Greece_Larynx classification head."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = VisionLSTM2(
            dim=768,
            depth=12,
            input_shape=(3, 224, 224),
            patch_size=16,
            output_shape=(1000,),
            legacy_norm=True,
            pooling='bilateral_flatten',
            conv_kind='2d',
            conv_kernel_size=3,
            norm_bias=True,
            proj_bias=True,
        )
        self.num_classes = num_classes

    def replace_classifier(self) -> None:
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, self.num_classes)
        nn.init.trunc_normal_(self.backbone.head.weight, std=2e-5)
        nn.init.zeros_(self.backbone.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_vil2_pretrained(model: ViL2Classifier, weight_path: str) -> None:
    """Load the official vil2-base ImageNet checkpoint and replace its 1000-class head."""
    path = Path(weight_path)
    if not path.is_file():
        raise FileNotFoundError(f'ViL2 pretrained weight not found: {path}')
    checkpoint = torch.load(str(path), map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)

    # This is the official ViL2 checkpoint key conversion from NX-AI hubconf.py.
    state_dict = {key.replace('.xlstm.', '.layer.'): value for key, value in state_dict.items()}
    state_dict = {key.replace('xlstm.', ''): value for key, value in state_dict.items()}
    state_dict = {key.replace('.xlstm_norm.', '.norm.'): value for key, value in state_dict.items()}
    state_dict = {key.replace('.conv1d.', '.conv.'): value for key, value in state_dict.items()}
    for index in range(24):
        if index % 2 == 0:
            state_dict = {
                key.replace(f'blocks.{index}.', f'blocks.{index // 2}.rowwise_from_top_left.'): value
                for key, value in state_dict.items()
            }
        else:
            state_dict = {
                key.replace(f'blocks.{index}.', f'blocks.{index // 2}.rowwise_from_bot_right.'): value
                for key, value in state_dict.items()
            }
    state_dict['norm.weight'] = state_dict.pop('post_blocks_norm.weight')
    state_dict['norm.bias'] = state_dict.pop('post_blocks_norm.bias')
    state_dict['legacy_norm.weight'] = state_dict.pop('head.0.weight')
    state_dict['legacy_norm.bias'] = state_dict.pop('head.0.bias')
    state_dict['head.weight'] = state_dict.pop('head.1.weight')
    state_dict['head.bias'] = state_dict.pop('head.1.bias')

    model.backbone.load_state_dict(state_dict, strict=True)
    model.replace_classifier()
    print(f'Loaded official ViL2-Base ImageNet weights from: {path}')


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(0)
    gpu_device = 0
    class_num = 3
    batch_size = 32
    epochs = 100
    image_dir = r'/root/autodl-tmp/GAFT/Datasets/Greece_Larynx'
    mode_stat = 'test'  # train, test
    num_workers = 16
    pre_weight_path = r'/root/autodl-tmp/Weights/ViL2/vil2_base16_e400_in1k.th'
    model_wegt_path = r'/root/autodl-tmp/GAFT/Results/Wegt_Results/Greece_Larynx/SOTAs/ViL2_Greece_Larynx.pth'

    test_log_path = model_wegt_path.replace('.pth', '_training_log.txt')

    print('########################## reading datas and processing datas #########################')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize(mean=0.5, std=0.5),
    ])
    train_data = ImageFolder(image_dir + '/Train_patch', transform=transform)
    val_data = ImageFolder(image_dir + '/Val_patch', transform=transform)
    test_data = ImageFolder(image_dir + '/Test_patch', transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    vil2_net = ViL2Classifier(num_classes=class_num)
    load_vil2_pretrained(vil2_net, pre_weight_path)
    summary(vil2_net, input_size=(1, 3, 224, 224), depth=4)
    print(vil2_net)
    vil2_net = vil2_net.cuda(gpu_device)

    if mode_stat == 'train':
        Single_out_fit(
            ddai_net=vil2_net,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr_fn='vit',
            epoch=epochs,
            gpu_device=gpu_device,
            weight_path=model_wegt_path,
        )
    elif mode_stat == 'test':
        checkpoint = torch.load(model_wegt_path, map_location=f'cuda:{gpu_device}')
        vil2_net.load_state_dict(checkpoint, strict=True)
        testing_funnction(
            test_model=vil2_net,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            gpu_device=gpu_device,
            out_mode='single',
            weight_path=model_wegt_path, log_path=test_log_path,
        )
    else:
        raise ValueError(f'Unsupported mode_stat: {mode_stat!r}')

