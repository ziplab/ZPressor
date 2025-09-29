import torch
from einops import rearrange
from torch import nn

from itertools import repeat
import collections.abc
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

        self.norm = nn.LayerNorm(in_features)

    def forward(self, input):
        x = self.norm(input)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x + input

class Attention(nn.Module):
    def __init__(
        self, dim, heads=8, dim_head=128, dropout=0.0, ffn=True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.ffn = ffn
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        self.norm = nn.LayerNorm(dim)

        if ffn:
            self.mlp = Mlp(in_features=dim, hidden_features=dim * 4, out_features=dim, act_layer=nn.GELU, drop=0.0)

    def forward(self, qkv, implementation="flash_attn"):
        B, N, C = qkv.shape
        x = self.norm(qkv)
        x = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), x)

        if implementation == "default":
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            attn = dots.softmax(dim=-1)

            x = torch.matmul(attn, v)
            x = rearrange(x, "b h n d -> b n (h d)")
            x = self.to_out(x)

        elif implementation == "flash_attn":
            with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                dtype = k.dtype
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x = scaled_dot_product_attention(q, k, v, scale=self.scale)
                if dtype == torch.float32:  # if input was FP32, cast back to FP32
                    x = x.to(torch.float32)
                x = rearrange(x, "b h n d -> b n (h d)")
                x = self.to_out(x)

        if self.ffn:
            return self.mlp(x + qkv)
        else:
            return x + qkv

class CrossAttention(nn.Module):
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., ffn=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = ffn
        if ffn:
            self.mlp = Mlp(in_features=dim, hidden_features=dim * 4, out_features=dim, act_layer=nn.GELU, drop=0.0)
        
    def forward(self, query, context, implementation="flash_attn"):
        B, Nq, C = query.shape
        Nk = context.shape[1]
        Nv = context.shape[1]

        qu = self.norm1(query)
        co = self.norm2(context)
        
        q = self.projq(qu).reshape(B,Nq,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        k, v = self.proj_kv(co).chunk(2, dim=-1)
        k = k.reshape(B,Nk,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B,Nv,self.num_heads, C// self.num_heads).permute(0, 2, 1, 3)
            
        if implementation == "default":
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
            x = self.proj(x)
            x = self.proj_drop(x)
        elif implementation == "flash_attn":
            with torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                q = q.to(torch.bfloat16)
                k = k.to(torch.bfloat16)
                v = v.to(torch.bfloat16)
                x = scaled_dot_product_attention(q, k, v, scale=self.scale)
                x = x.to(torch.float32)
                x = x.transpose(1, 2).reshape(B, Nq, C)
                x = self.proj(x)
                x = self.proj_drop(x)

        if self.ffn:
            return self.mlp(x + query)
        else:
            return x + query
    