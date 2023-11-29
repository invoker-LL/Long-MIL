import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.rotary import apply_rotary_position_embeddings, Rotary2D
from torch import autograd
try:
    from xformers.ops import memory_efficient_attention
except:
    print('please install xformer')
    pass

def exists(val):
    return val is not None

def precompute_freqs_cis(dim: int, end: int, pos_idx, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # t = pos_idx.cpu()
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # pdb.set_trace()
    return freqs_cis[pos_idx.long()]

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,freqs_cis: torch.Tensor,):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.05, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis=None, alibi=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4) #
        q, k, v = qkv[0], qkv[1], qkv[2]
        q,k = q.view(1,N,1,-1), k.view(1,N,1,-1)
        if exists(freqs_cis):
            q,k = apply_rotary_position_embeddings(freqs_cis, q, k)
        q, k = q.view(1, N, self.num_heads, -1), k.view(1, N, self.num_heads, -1)
        if exists(alibi):
            try:
                x = memory_efficient_attention(q, k, v, alibi,p=self.attn_drop).reshape(B, N, C)
            except:
                pdb.set_trace()
        else:
            x = memory_efficient_attention(q,k,v,p=self.attn_drop).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512,num_heads=4):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(dim=dim,num_heads=num_heads)

    def forward(self, x, rope, alibi):
        x = self.norm(x)
        x, attn = self.attn(x, rope, alibi)
        return x, attn

import math
def get_slopes(n_heads: int):
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))

    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m

class LongMIL(nn.Module):
    def __init__(self, n_classes, input_size=384):
        super(LongMIL, self).__init__()
        self.n_heads = 1
        self.input_size = input_size
        # feat_size = 384
        feat_size = input_size # depends on using _fc1
        self.feat_size = feat_size
        self._fc1 = nn.Sequential(nn.Linear(input_size, feat_size), nn.ReLU())
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=feat_size,num_heads=self.n_heads)
        self.layer2 = TransLayer(dim=feat_size,num_heads=self.n_heads)
        self.layer3 = TransLayer(dim=feat_size, num_heads=self.n_heads)
        self.layer4 = TransLayer(dim=feat_size, num_heads=self.n_heads)
        self.norm = nn.LayerNorm(feat_size)
        self._fc2 = nn.Linear(feat_size, self.n_classes)
        self.rotary = Rotary2D(dim=feat_size)
        self.alibi = torch.load('./alibi_tensor_core.pt').cuda()
        self.slope_m = get_slopes(self.n_heads)

    def positional_embedding(self, x, use_alibi=False, use_rope=False):
        scale = 1  # for 20x 224 with 112 overlap (or 40x 224)
        scale = 2  # for 20x 224 with 0 overlap
        freqs_cis = None
        alibi_bias = None
        if use_rope or use_alibi:
            abs_pos = x[::, -2:]
            x_pos, y_pos = abs_pos[:, 0], abs_pos[:, 1]
            x_pos = torch.round((x_pos - x_pos.min()) / (112 * scale) / 4)
            y_pos = torch.round((y_pos - y_pos.min()) / (112 * scale) / 4)
            # x_pos = torch.round((x_pos - x_pos.min()) / (128 * scale) / 4) # for shape 256*256
            # y_pos = torch.round((y_pos - y_pos.min()) / (128 * scale) / 4)
            H, W = 600 // scale, 600 // scale
            selected_idx = (x_pos * W + y_pos).to(torch.int)
            if use_rope:
                pos_cached = self.rotary.forward(torch.tensor([H, W]))
                freqs_cis = pos_cached[selected_idx].cuda()
            if use_alibi:
                alibi_bias = self.alibi[selected_idx, :][:, selected_idx]
                alibi_bias = alibi_bias[:, :, None] * self.slope_m[None, None, :].cuda()
                alibi_bias = alibi_bias.permute(2, 0, 1).unsqueeze(0).float()

                shape3 = alibi_bias.shape[3]
                pad_num = 8 - shape3 % 8 # to tackle FlashAttention problems
                padding_bias = torch.zeros(1, alibi_bias.shape[1], alibi_bias.shape[2], pad_num).cuda()
                alibi_bias = torch.cat([alibi_bias, padding_bias], dim=-1)
                alibi_bias = autograd.Variable(alibi_bias.contiguous())[:, :, :, :shape3]

        return freqs_cis, alibi_bias


    def forward(self, x):
        h = x[::, :self.input_size].unsqueeze(0)  # [B, n, feat_dim]

        freqs_cis, alibi_bias = self.positional_embedding(x)

        h, attn = self.layer1(h, freqs_cis, alibi_bias)
        h, attn2 = self.layer2(h, freqs_cis, alibi_bias)  # [B, N, 512]

        h = self.norm(h.mean(1))

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=-1)
        Y_prob = F.softmax(logits, dim=-1)

        return logits, Y_hat, Y_prob, attn

if __name__ == "__main__":
    data = torch.randn((12800, 384+2)).cuda() # 384 is feature, 2 is coordinates
    model = LongMIL(n_classes=2).cuda()
    print(model.eval())
    results_dict = model(data)
    # pdb.set_trace()
    print(results_dict)