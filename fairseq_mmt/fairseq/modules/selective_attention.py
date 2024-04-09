# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.incremental_decoding_utils import with_incremental_state
from torch import nn

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

@with_incremental_state
class SelectiveAttention(nn.Module):
    def __init__(self, qdim, kdim, vdim, attn_dim, intermediate_dim, output_dim, num_heads=1, qkv_bias=True, attn_drop=0.):
        super().__init__()
        # print(f"qdim: {qdim}, kdim: {kdim}, vdim: {vdim}, attn_dim: {attn_dim}, intermediate_dim: {intermediate_dim}, output_dim: {output_dim}")
        # c4c: (12, 512), qdim: 128, kdim: 512, vdim: 512, attn_dim: 128, intermediate_dim: 128, output_dim: 128
        self.num_heads = num_heads
        self.qdim = qdim
        self.kdim = kdim
        self.vdim = vdim
        self.output_dim = output_dim
        self.intermediate_dim = intermediate_dim

        self.qkhead_dim = attn_dim // num_heads # 128
        self.vhead_dim = intermediate_dim // num_heads               
        self.scale = self.qkhead_dim ** -0.5

        self.q_proj = Linear(qdim, attn_dim, bias=qkv_bias)
        self.k_proj = Linear(kdim, attn_dim, bias=qkv_bias)
        self.v_proj = Linear(vdim, intermediate_dim, bias=qkv_bias)   
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(intermediate_dim, output_dim)

    def forward(self, query, key, value, key_padding_mask=None):
        Tq, Bq, Cq = query.shape
        Tk, Bk, Ck = key.shape
        Tv, Bv, Cv = value.shape
        # print(f"query: {query.shape}, key: {key.shape}, value: {value.shape}")
        # query: torch.Size([6, 496, 128]), key: torch.Size([12, 496, 512]), value: torch.Size([12, 496, 512])
        assert Bq == Bk == Bv # batch size
        assert Tk == Tv         # num_frames 12
        assert Cq == self.qdim # text feature size 128
        assert Ck == self.kdim # image feature size 512
        assert Cv == self.vdim # image feature size 512
        bsz = Bq
        
        q = self.q_proj(query)  # TxBx128 -> TxBx128 
        k = self.k_proj(key) # 12xBx512 -> 12xBx128
        v = self.v_proj(value) # 12xBx512 -> 12xBx128
       
        q *= self.scale
        
        q = q.contiguous().view(Tq, bsz * self.num_heads, self.qkhead_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.qkhead_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.vhead_dim).transpose(0, 1)
        # B*H, T, C//H

        attn = (q @ k.transpose(-2, -1)) 
        if key_padding_mask is not None:
            attn = attn.view(bsz, self.num_heads, Tq, Tk)
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf"))
            attn = attn.view(bsz * self.num_heads, Tq, Tk)

        attn = attn.softmax(dim=-1)
        attn_after_drop = self.attn_drop(attn)

        x = (attn_after_drop @ v)
        assert list(x.size()) == [bsz * self.num_heads, Tq, self.vhead_dim]
        x = x.transpose(0, 1).contiguous().view(Tq, bsz, self.intermediate_dim)
        x = self.proj(x)
        return x, attn
