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
import os
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
        # print('input x shape: ', np.shape(x)) # [1, 4096, 320]

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        # print('qkv shape: ', np.shape(q), np.shape(k), np.shape(v)) # [1, 4096, 320]
        # print('mask shape: ', np.shape(mask)) # [1, 4096, 320]
        # import pdb; pdb.set_trace()


        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # print('qkv after rearrange: ', np.shape(q), np.shape(k), np.shape(v)) # [5, 4096, 64]

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # print('sim shape: ', np.shape(sim)) # [5, 4096, 4096]
        # print('mask is exist: ', exists(mask)) # False
        
        del q, k
    
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)
        # print('sim after softmax: ', np.shape(sim)) # [5, 4096, 4096]

        out = einsum('b i j, b j d -> b i d', sim, v)
        # print('out shape1: ', np.shape(out)) # [5, 4096, 64]
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # print('out shape2: ', np.shape(out)) # [1, 4096, 320]
        # zzz = self.to_out(out)
        # print('zzz shape: ', np.shape(zzz)) # [1, 4096, 320]
        # import pdb; pdb.set_trace()
        return self.to_out(out)

    # def forward(self, x, context=None, mask=None):
    #     h = self.heads
    #     # print('input x shape: ', np.shape(x)) # [1, 4096, 320]

    #     q = self.to_q(x)
    #     context = default(context, x)
    #     k = self.to_k(context)
    #     v = self.to_v(context)
    #     print('qkv shape: ', np.shape(q), np.shape(k), np.shape(v)) # [1, 4096, 320]
    #     print('mask shape: ', np.shape(mask))
    #     import pdb; pdb.set_trace()


    #     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
    #     # print('qkv after rearrange: ', np.shape(q), np.shape(k), np.shape(v)) # [5, 4096, 64]

    #     # force cast to fp32 to avoid overflowing
    #     if _ATTN_PRECISION =="fp32":
    #         with torch.autocast(enabled=False, device_type = 'cuda'):
    #             q, k = q.float(), k.float()
    #             sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    #     else:
    #         sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    #     # print('sim shape: ', np.shape(sim)) # [5, 4096, 4096]
    #     # print('mask is exist: ', exists(mask)) # False
        
    #     del q, k
    
    #     if exists(mask):
    #         mask = rearrange(mask, 'b ... -> b (...)')
    #         max_neg_value = -torch.finfo(sim.dtype).max
    #         mask = repeat(mask, 'b j -> (b h) () j', h=h)
    #         sim.masked_fill_(~mask, max_neg_value)

    #     # attention, what we cannot get enough of
    #     sim = sim.softmax(dim=-1)
    #     # print('sim after softmax: ', np.shape(sim)) # [5, 4096, 4096]

    #     out = einsum('b i j, b j d -> b i d', sim, v)
    #     # print('out shape1: ', np.shape(out)) # [5, 4096, 64]
    #     out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    #     # print('out shape2: ', np.shape(out)) # [1, 4096, 320]
    #     # zzz = self.to_out(out)
    #     # print('zzz shape: ', np.shape(zzz)) # [1, 4096, 320]
    #     # import pdb; pdb.set_trace()
    #     return self.to_out(out)


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
        # print('input x shape: ', np.shape(x)) # [1, 4096, 320]
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        # print('qkv shape: ', np.shape(q), np.shape(k), np.shape(v)) # [1, 4096, 320]
        # print('context shape: ', np.shape(context),'x shape: ', np.shape(x))

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        # print('qkv after rearrange: ', np.shape(q), np.shape(k), np.shape(v)) # [5, 4096, 64]

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        # print('out shape1: ', np.shape(out)) # [5, 4096, 64]
        # print('mask is exist: ', exists(mask)) # False

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        # print('out shape2: ', np.shape(out)) # [1, 4096, 32]
        # zzz = self.to_out(out)
        # print('zzz shape: ', np.shape(zzz)) # # [1, 4096, 32]
        # import pdb; pdb.set_trace()
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

        # self.norm2g = nn.LayerNorm(dim)
        # self.gate_q  = nn.Linear(dim, context_dim)        # 把 token 映到 context 维度
        # self.tau     = 0.5                        # 温度，可退火
        # self.norm2b  = nn.LayerNorm(dim)
        # self.attn_bg = attn_cls(query_dim=dim, context_dim=context_dim,
        #                      heads=n_heads, dim_head=d_head, dropout=dropout) 
        # self.bg_proj = nn.Linear(dim, context_dim)  # 可选；如果你已有 context_bg 则不需要

        # 用于门控打分/聚合的投影
        self.norm2g = nn.LayerNorm(dim)
        self.gate_q = nn.Linear(dim, context_dim)         # Q: x -> ctx 维

        self.norm_c = nn.LayerNorm(context_dim)
        self.k_proj = nn.Linear(context_dim, context_dim, bias=False)  # 共享给 obj0/obj1
        self.v_proj = nn.Linear(context_dim, context_dim, bias=False)

        self.tau = 0.5

        # 第二次与“背景(x_orig)”的 cross-attn（背景就是 x_orig，所以 context_dim=dim）
        self.norm2b  = nn.LayerNorm(dim)
        self.attn_bg = attn_cls(query_dim=dim, context_dim=dim,
                                heads=n_heads, dim_head=d_head, dropout=dropout)



        # self.multiview_mlp = nn.Sequential(
        #     nn.Linear(context_dim, int(context_dim * 1.3)),  # First map to a hidden dimension
        #     nn.ReLU(),
        #     nn.Linear(int(context_dim * 1.3), context_dim)  # Map to final desired dimension
        # )

    def forward(self, x, context=None, c_mask=None):
        # print('aaaaaaa')
        # import pdb; pdb.set_trace()
        return checkpoint(self._forward, (x, context, c_mask), self.parameters(), self.checkpoint)

    # def _forward(self, x, context=None, c_mask=None):

    #     c_mask = torch.unsqueeze(c_mask, 2)
    #     c_mask = c_mask.repeat(1, 1, x.shape[-1], 1)
    #     c_mask = c_mask.to(x.device)
        
    #     c_mask_0 = c_mask[..., 0] # [1, 4096, 320]
    #     context_0 = context[..., 0]
    #     c_mask_1 = c_mask[..., 1] # [1, 4096, 320]
    #     context_1 = context[..., 1]

    #     c_mask_0_ = c_mask_0.bool() & ~c_mask_1.bool()
    #     c_mask_1_ = c_mask_1.bool() & ~c_mask_0.bool()
    #     c_mask_joint = c_mask_0.bool() & c_mask_1.bool()

    #     # print(context_0.shape)
    #     batch_size, num_tokens, context_dim = context_0.shape
    #     # print(np.shape(context_0))
    #     # import pdb; pdb.set_trace()
    #     context_0 = context_0.view(batch_size, num_tokens, -1)  # Shape: [1, 257, 1024, 6]
    #     # context_0 = self.multiview_mlp(context_0)  # Shape: [1, 4096, 320]
    #     context_1 = context_1.view(batch_size, num_tokens, -1)
    #     # context_1 = self.multiview_mlp(context_1)  # Shape: [1, 4096, 320]


    #     x_orig = x.clone()
    #     x0 = self.attn1(self.norm1(x_orig), context=context_0 if self.disable_self_attn else None)
    #     x1 = self.attn1(self.norm1(x_orig), context=context_1 if self.disable_self_attn else None)
    #     x01 = torch.cat([x0, x1], dim=-1)
    #     x_blend = self.blend_mlp(x01)
    #     x = x0 * c_mask_0_ + x1 * c_mask_1_ + x_blend * c_mask_joint
    #     x = x + x_orig
    #     # x_orig = x.clone()
    #     # x = self.attn1(self.norm1(x_orig), context=context_0 if self.disable_self_attn else None)
    #     # x = x + x_orig

    #     x_orig = x.clone()
    #     x0 = self.attn2(self.norm2(x_orig), context=context_0)
    #     x1 = self.attn2(self.norm2(x_orig), context=context_1)
    #     x01 = torch.cat([x0, x1], dim=-1)
    #     x_blend = self.blend_mlp(x01)
    #     x = x0 * c_mask_0_ + x1 * c_mask_1_ + x_blend * c_mask_joint
    #     x = x + x_orig

    #     x_orig = x.clone()
    #     x = self.ff(self.norm3(x))
    #     x = x + x_orig

    #     return x

    def _forward(self, x, context=None, c_mask=None):

        c_mask = torch.unsqueeze(c_mask, 2)
        c_mask = c_mask.repeat(1, 1, x.shape[-1], 1)
        c_mask = c_mask.to(x.device)
        
        c_mask_0 = c_mask[..., 0] # [1, 4096, 320]
        context_0 = context[..., 0]
        c_mask_1 = c_mask[..., 1] # [1, 4096, 320]
        context_1 = context[..., 1]

        c_mask_0_ = c_mask_0.bool() & ~c_mask_1.bool()
        c_mask_1_ = c_mask_1.bool() & ~c_mask_0.bool()
        c_mask_joint = c_mask_0.bool() & c_mask_1.bool()

        # print(context_0.shape)
        batch_size, num_tokens, context_dim = context_0.shape
        # print(np.shape(context_0))
        # import pdb; pdb.set_trace()
        context_0 = context_0.view(batch_size, num_tokens, -1)  # Shape: [1, 257, 1024, 6]
        # context_0 = self.multiview_mlp(context_0)  # Shape: [1, 4096, 320]
        context_1 = context_1.view(batch_size, num_tokens, -1)
        # context_1 = self.multiview_mlp(context_1)  # Shape: [1, 4096, 320]

        x_orig = x.clone()
        x = self.attn1(self.norm1(x_orig), context=context_0 if self.disable_self_attn else None)
        x = x + x_orig

        x_orig = x.clone()
        x0 = self.attn2(self.norm2(x_orig), context=context_0)
        x1 = self.attn2(self.norm2(x_orig), context=context_1)

        # ---------- 1) 用 SDPA 将每个对象的上下文按 token 查询聚合到 [B,N,ctx] ----------
        with autocast(dtype=torch.bfloat16, enabled=True):  # 或者 float16，看你训练精度设定
            # Queries
            qg = self.gate_q(self.norm2g(x_orig))                            # [B,N,ctx]
            qg_h = qg.unsqueeze(1)                                           # [B,1,N,ctx]

            # obj0 keys/values
            k0 = self.k_proj(self.norm_c(context_0)).unsqueeze(1)            # [B,1,M,ctx]
            v0 = self.v_proj(context_0).unsqueeze(1)                         # [B,1,M,ctx]
            # 使用 SDPA 聚合：输出即为按 token 聚合的上下文
            c0_agg = F.scaled_dot_product_attention(qg_h, k0, v0, dropout_p=0.0).squeeze(1)  # [B,N,ctx]

            # obj1 keys/values
            k1 = self.k_proj(self.norm_c(context_1)).unsqueeze(1)            # [B,1,M,ctx]
            v1 = self.v_proj(context_1).unsqueeze(1)
            c1_agg = F.scaled_dot_product_attention(qg_h, k1, v1, dropout_p=0.0).squeeze(1)  # [B,N,ctx]

        # ---------- 2) 基于聚合后的上下文做“二选一/两者兼顾”的注意力门控 ----------
        # 用 qg 与 c0_agg/c1_agg 打分（很小的张量，不会炸显存）
        s0 = (qg * c0_agg).sum(-1) / (self.context_dim ** 0.5)   # [B,N]
        s1 = (qg * c1_agg).sum(-1) / (self.context_dim ** 0.5)   # [B,N]
        alpha = torch.softmax(torch.stack([s0, s1], dim=-1) / self.tau, dim=-1)[..., 0:1]  # [B,N,1]

        # # ===== 可视化（A-only / B-only / Joint 同图，边界着色区分）=====
        # with torch.no_grad():
        #     import os, math, numpy as np
        #     import matplotlib
        #     matplotlib.use("Agg")
        #     import matplotlib.pyplot as plt
        #     from matplotlib.lines import Line2D

        #     save_root = '/home/hang18/links/projects/rrg-vislearn/hang18/COIN/vis/wild_attn_weight'
        #     layer_tag = getattr(self, "layer_tag", self.__class__.__name__)

        #     vid = int(getattr(self, "_vis_id", 0)); self._vis_id = vid + 1
        #     save_this_time = (vid % 1 == 0)  # 每 10 次保存一次（按需改）
        #     if save_this_time:
        #         # [B,N]
        #         alpha_bn = alpha.squeeze(-1).detach()
        #         s0_bn    = s0.detach()
        #         s1_bn    = s1.detach()
        #         B, N     = alpha_bn.shape

        #         # ---- 三类区域 mask -> [B,N] bool ----
        #         def _bn_bool(m):
        #             # m 可能是 [B,N,C]，取任意通道的 or
        #             return (m.any(dim=-1) if m.dim() == 3 else m).detach().bool()

        #         A_only_bn   = _bn_bool(c_mask_0_)        # A 独占
        #         B_only_bn   = _bn_bool(c_mask_1_)        # B 独占
        #         joint_bn    = _bn_bool(c_mask_joint)     # 重叠
        #         union_bn    = (A_only_bn | B_only_bn | joint_bn)

        #         # ---- token 网格 ----
        #         if hasattr(self, "token_hw") and self.token_hw is not None:
        #             Ht, Wt = self.token_hw
        #         else:
        #             s = int(math.sqrt(N))
        #             if s * s == N: Ht, Wt = s, s
        #             else:
        #                 print(f"[VIS][SKIP] {layer_tag}: N={N} 非平方，未设置 token_hw")
        #                 Ht = Wt = None

        #         if Ht is not None:
        #             out_dir = os.path.join(save_root, layer_tag)
        #             os.makedirs(out_dir, exist_ok=True)

        #             # ---- 边界绘制工具（不同颜色区分三类区域）----
        #             def _draw_region_edges(ax, A_mask, B_mask, J_mask):
        #                 # 用等值线在 mask 边缘画线
        #                 def _contour(mask2d, color, lw=1.5):
        #                     if mask2d.any():
        #                         ax.contour(mask2d.astype(float), levels=[0.5], colors=[color], linewidths=lw)

        #                 _contour(A_mask, color="#E64B35")  # 红：A-only
        #                 _contour(B_mask, color="#4DBBD5")  # 蓝：B-only
        #                 _contour(J_mask, color="#F0E442")  # 黄：Joint

        #                 handles = [
        #                     Line2D([0],[0], color="#E64B35", lw=2, label="A-only"),
        #                     Line2D([0],[0], color="#4DBBD5", lw=2, label="B-only"),
        #                     Line2D([0],[0], color="#F0E442", lw=2, label="Joint"),
        #                 ]
        #                 ax.legend(handles=handles, loc="lower right", framealpha=0.6, fontsize=8)

        #             # ---- 保存函数：alpha（绝对刻度）、score（联合区域归一）、sdiff（对称色条）----
        #             def _save_alpha(vec_bn, name):
        #                 for bi in range(min(B, 2)):
        #                     m = vec_bn[bi].float().view(Ht, Wt).cpu().numpy()
        #                     Au = A_only_bn[bi].view(Ht, Wt).cpu().numpy()
        #                     Bu = B_only_bn[bi].view(Ht, Wt).cpu().numpy()
        #                     Ju = joint_bn[bi].view(Ht, Wt).cpu().numpy()
        #                     U  = union_bn[bi].view(Ht, Wt).cpu().numpy()

        #                     m_plot = np.full_like(m, np.nan, dtype=float)
        #                     m_plot[U] = m[U]  # 只在联合区域显示

        #                     fig, ax = plt.subplots()
        #                     im = ax.imshow(m_plot, vmin=0.0, vmax=1.0)
        #                     ax.axis('off'); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #                     _draw_region_edges(ax, Au, Bu, Ju)

        #                     fn = os.path.join(out_dir, f"{name}_vid{vid:07d}_b{bi}.png")
        #                     fig.tight_layout(); fig.savefig(fn, dpi=200); plt.close(fig)

        #             def _save_score(vec_bn, name):
        #                 for bi in range(min(B, 2)):
        #                     m = vec_bn[bi].float().view(Ht, Wt).cpu().numpy()
        #                     Au = A_only_bn[bi].view(Ht, Wt).cpu().numpy()
        #                     Bu = B_only_bn[bi].view(Ht, Wt).cpu().numpy()
        #                     Ju = joint_bn[bi].view(Ht, Wt).cpu().numpy()
        #                     U  = union_bn[bi].view(Ht, Wt).cpu().numpy()

        #                     m_plot = np.full_like(m, np.nan, dtype=float)
        #                     if U.any():
        #                         vals = m[U]
        #                         vmin, vmax = float(vals.min()), float(vals.max())
        #                         m_plot[U] = (m[U] - vmin) / (vmax - vmin + 1e-6)  # 仅联合区域归一化

        #                     fig, ax = plt.subplots()
        #                     im = ax.imshow(m_plot)
        #                     ax.axis('off'); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #                     _draw_region_edges(ax, Au, Bu, Ju)

        #                     fn = os.path.join(out_dir, f"{name}_vid{vid:07d}_b{bi}.png")
        #                     fig.tight_layout(); fig.savefig(fn, dpi=200); plt.close(fig)

        #             def _save_sdiff(vec0_bn, vec1_bn, name="sdiff"):
        #                 for bi in range(min(B, 2)):
        #                     m = (vec0_bn[bi] - vec1_bn[bi]).float().view(Ht, Wt).cpu().numpy()
        #                     Au = A_only_bn[bi].view(Ht, Wt).cpu().numpy()
        #                     Bu = B_only_bn[bi].view(Ht, Wt).cpu().numpy()
        #                     Ju = joint_bn[bi].view(Ht, Wt).cpu().numpy()
        #                     U  = union_bn[bi].view(Ht, Wt).cpu().numpy()

        #                     m_plot = np.full_like(m, np.nan, dtype=float)
        #                     if U.any():
        #                         vals = m[U]
        #                         vmax = float(np.percentile(np.abs(vals), 99.0))
        #                         vmax = max(vmax, 1e-6)
        #                         m_plot[U] = m[U]
        #                     else:
        #                         vmax = 1.0

        #                     fig, ax = plt.subplots()
        #                     im = ax.imshow(m_plot, vmin=-vmax, vmax=+vmax)
        #                     ax.axis('off'); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        #                     _draw_region_edges(ax, Au, Bu, Ju)

        #                     fn = os.path.join(out_dir, f"{name}_vid{vid:07d}_b{bi}.png")
        #                     fig.tight_layout(); fig.savefig(fn, dpi=200); plt.close(fig)

        #             # ---- 实际保存：同一张图里区分三类区域 ----
        #             _save_alpha(alpha_bn, "alpha_regions")
        #             _save_score(s0_bn,    "s0_regions")
        #             _save_score(s1_bn,    "s1_regions")
        #             _save_sdiff(s0_bn, s1_bn, "sdiff_regions")  # 可选

        #             print(f"[VIS] saved(REGIONS) {layer_tag} vid={vid} -> {out_dir}")
        # # ===== 可视化结束 =====



        pair_ctx = alpha * c0_agg + (1.0 - alpha) * c1_agg       # [B,N,ctx]
        pair_ctx = pair_ctx.to(context_0.dtype)  # 或 pair_ctx = pair_ctx.to(x_orig.dtype)

        # ---------- 3) 注入对象间关系（pair_ctx 作为 KV） ----------
        x_pair = self.attn2(self.norm2(x_orig), context=pair_ctx)            # [B,N,dim]

        # # ---------- 4) 与“背景(x_orig)”再做一次 cross-attn 对齐 ----------
        # x_pair_bg = self.attn_bg(self.norm2b(x_pair), context=x_orig)        # [B,N,dim]

        # --- 4) 与“背景(x_orig)”再做一次 cross-attn 对齐 ---
        context_bg = x_orig                                         # [B,N,dim]
        x_pair_bg  = self.attn_bg(self.norm2b(x_pair), context=context_bg)  # [B,N,dim]

        x = x0 * c_mask_0_ + x1 * c_mask_1_ + x_pair_bg * c_mask_joint
        x = x + x_orig


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
        # print('orig x shape: ', np.shape(x)) # [1, 320, 64, 64], resized image feature
        # import pdb; pdb.set_trace()
        x = self.norm(x) # GroupNorm, 32 each
        # print(self.use_linear) # True-> Use Linear
        # print('normed x shape: ', np.shape(x)) # [1, 320, 64, 64]
        
        if not self.use_linear: # not implement
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        # print('rearranged x shape: ', np.shape(x)) # [1, 4096, 320]
        
        if self.use_linear: # implement
            x = self.proj_in(x)
            # print('projected x shape: ', np.shape(x)) # [1, 4096, 320]
            # import pdb; pdb.set_trace()

        for i, block in enumerate(self.transformer_blocks):
            # print(np.shape(x), np.shape(context[i]), np.shape(c_mask[i])) # [1, 4096, 320], [1, 257, 1024, 2], [1, 64, 64, 2]
            # print('x shape: ', np.shape(x))
            # print('context[i] shape: ', np.shape(context[i])) # [8, 257, 1024, 6, 2] -> [12, 257, 1024, 6, 2]? 
            # print('c_mask shape: ', np.shape(c_mask))
            # print('c_mask[i] shape: ', np.shape(c_mask[i]))
            # print(len(x), len(context[i]), len(c_mask[i]))
            # import pdb; pdb.set_trace()
            c_mask_i = rearrange(c_mask[i], 'b h w c -> b (h w) c').contiguous()
            # print('mask unique: ', torch.unique(c_mask_i)) [0, 1]
            # print('c_mask_i shape: ', np.shape(c_mask_i)) # [1, 4096, 2]
            # import pdb; pdb.set_trace()
            x = block(x, context=context[i], c_mask=c_mask_i)
            # print('i', 'transformer x shape: ', np.shape(x)) # [1, 4096, 320]
        if self.use_linear:
            x = self.proj_out(x)
        # print('projected back: ', np.shape(x)) # [1, 4096, 320]
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        # print('after rearrange:', np.shape(x)) # [1, 320, 64, 64]
        if not self.use_linear:
            x = self.proj_out(x)
        # import pdb; pdb.set_trace()
        return x + x_in

