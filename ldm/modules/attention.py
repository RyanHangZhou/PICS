from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any
import numpy as np
from torch.cuda.amp import autocast
import os

from ldm.modules.diffusionmodules.util import checkpoint

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, c_mask=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        # attn_mode = "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.dim = dim
        self.context_dim = context_dim

        self.blend_mlp = nn.Sequential(
            nn.Linear(dim * 2, int(dim * 1.3)),  # First map to a hidden dimension
            nn.ReLU(),
            nn.Linear(int(dim * 1.3), dim)  # Map to final desired dimension
        )

        self.norm2g = nn.LayerNorm(dim)
        self.gate_q = nn.Linear(dim, context_dim)         # Q: x -> ctx

        self.norm_c = nn.LayerNorm(context_dim)
        self.k_proj = nn.Linear(context_dim, context_dim, bias=False)  # shared to obj0/obj1
        self.v_proj = nn.Linear(context_dim, context_dim, bias=False)

        self.tau = 0.5

        self.norm2b  = nn.LayerNorm(dim)
        self.attn_bg = attn_cls(query_dim=dim, context_dim=dim,
                                heads=n_heads, dim_head=d_head, dropout=dropout)

    def forward(self, x, context=None, c_mask=None):
        return checkpoint(self._forward, (x, context, c_mask), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, c_mask=None):
        """
        Multi-object conditioning forward pass with spatial blending.
        x: [B, N, C] - Visual tokens
        context: [B, M, D, 2] - Object-specific conditioning context
        c_mask: [B, N, 2] - Spatial masks
        """
        
        # 1. Spatial mask expansion and logic preparation
        # Expand mask to match feature dimension for element-wise multiplication
        c_mask = torch.unsqueeze(c_mask, 2)
        c_mask = c_mask.repeat(1, 1, x.shape[-1], 1)
        c_mask = c_mask.to(x.device)
        
        c_mask_0 = c_mask[..., 0] 
        c_mask_1 = c_mask[..., 1]
        context_0 = context[..., 0]
        context_1 = context[..., 1]

        # Define mutually exclusive and joint (overlapping) regions
        c_mask_0_ = c_mask_0.bool() & ~c_mask_1.bool()
        c_mask_1_ = c_mask_1.bool() & ~c_mask_0.bool()
        c_mask_joint = c_mask_0.bool() & c_mask_1.bool()

        # 2. Self-Attention (or conditioning Layer 1)
        x_orig = x.clone()
        x = self.attn1(self.norm1(x_orig), context=context_0 if self.disable_self_attn else None)
        x = x + x_orig

        # 3. Parallel Cross-Attention for individual objects (Layer 2)
        x_orig = x.clone()
        x0 = self.attn2(self.norm2(x_orig), context=context_0)
        x1 = self.attn2(self.norm2(x_orig), context=context_1)

        # 4. Gated Attention Blending for joint/overlapping areas
        with autocast(dtype=torch.bfloat16, enabled=True): 
            # Project visual queries to gate space
            qg = self.gate_q(self.norm2g(x_orig))                    # [B, N, ctx_dim]
            qg_h = qg.unsqueeze(1)                                   # [B, 1, N, ctx_dim]

            # Aggregate context features for object 0
            k0 = self.k_proj(self.norm_c(context_0)).unsqueeze(1)    # [B, 1, M, ctx_dim]
            v0 = self.v_proj(context_0).unsqueeze(1)                 # [B, 1, M, ctx_dim]
            c0_agg = F.scaled_dot_product_attention(qg_h, k0, v0).squeeze(1)

            # Aggregate context features for object 1
            k1 = self.k_proj(self.norm_c(context_1)).unsqueeze(1)    # [B, 1, M, ctx_dim]
            v1 = self.v_proj(context_1).unsqueeze(1)
            c1_agg = F.scaled_dot_product_attention(qg_h, k1, v1).squeeze(1)

        # 5. Compute blending weights (alpha) via similarity scores
        s0 = (qg * c0_agg).sum(-1) / (self.context_dim ** 0.5)
        s1 = (qg * c1_agg).sum(-1) / (self.context_dim ** 0.5)
        alpha = torch.softmax(torch.stack([s0, s1], dim=-1) / self.tau, dim=-1)[..., 0:1]

        # Blend context and compute "Pair Attention" for the overlapping region
        pair_ctx = (alpha * c0_agg + (1.0 - alpha) * c1_agg).to(context_0.dtype)
        x_pair = self.attn2(self.norm2(x_orig), context=pair_ctx)

        # Refine paired features using visual tokens as background context
        x_pair_bg = self.attn_bg(self.norm2b(x_pair), context=x_orig)

        # 6. Final Spatial Fusion based on masks
        x = x0 * c_mask_0_ + x1 * c_mask_1_ + x_pair_bg * c_mask_joint
        x = x + x_orig

        # 7. Feed-Forward Network (Layer 3)
        x_orig = x.clone()
        x = self.ff(self.norm3(x))
        x = x + x_orig

        return x

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None, c_mask=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        if not isinstance(context, list):
            c_mask = [c_mask]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x) # GroupNorm, 32 each
        
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        
        if self.use_linear:
            x = self.proj_in(x)

        for i, block in enumerate(self.transformer_blocks):
            c_mask_i = rearrange(c_mask[i], 'b h w c -> b (h w) c').contiguous()
            x = block(x, context=context[i], c_mask=c_mask_i)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in
