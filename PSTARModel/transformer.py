"""PSTAR backbone: enhanced PerceiverAR (2D sinusoidal PE) that emits route-direction logits."""
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from config import config
from sinusoidal_2dpe import Sinusoidal2DPE

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_tensor, **kwargs):
        input_tensor = self.norm(input_tensor)
        return self.fn(input_tensor, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout):
        super().__init__()
        hidden_dim = dim * mult
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim))

    def forward(self, input_tensor):
        return self.net(input_tensor)

class PerceiverARAttention(nn.Module):
    def __init__(self, dim, heads, causal, sequence_len, dropout, layer_num):
        # REMOVED: `pos_emb` parameter — was accepted but silently discarded
        super().__init__()
        self.sequence_len = sequence_len
        self.context_len = config["context_matrix_length"]
        assert (config["context_matrix_length"] + config["foundroute_matrix_length"] == sequence_len), \
            'context_length plus latent should be equal to sequence length'
        self.layer_num = layer_num
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.heads = heads
        self.causal = causal
        self.norm = nn.LayerNorm(self.dim_head)

        self.attn_dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, input_tensor):
        b, n, d = input_tensor.shape
        h = self.heads
        device = input_tensor.device

        qkv = (self.to_q(input_tensor), self.to_kv(input_tensor))
        q, kv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), qkv)

        q_latent  = q[:, self.context_len:, :]   # [B*H, L, d]
        q_context = q[:, :self.context_len, :]   # [B*H, C, d]
        # kv_context = kv[:, :self.context_len, :] # [B*H, C, d]

        q_latent  = q_latent  * self.scale
        q_context = q_context * self.scale

        lkv     = self.norm(kv)            # [B*H, C+L, d]
        lkv_ctx = lkv[:, :self.context_len, :]   # [B*H, C, d]

        attn1 = einsum('b i d, b j d -> b i j', q_context, lkv_ctx)  # [B*H, C, C]
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_dropout(attn1)

        attn2 = einsum('b i d, b j d -> b i j', q_latent, lkv)  # [B*H, L, C+L]
        if self.causal:
            i, j = attn2.shape[-2], attn2.shape[-1]
            start_point_index = self.context_len
            causal_mask = torch.ones(i, j, device=device, dtype=torch.bool)
            causal_mask[:, :self.context_len] = False
            for row in range(i):
                causal_mask[row, start_point_index : start_point_index + row + 1] = False
            attn2 = attn2.masked_fill(causal_mask, float('-inf'))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_dropout(attn2)

        out_context = einsum('b i j, b j d -> b i d', attn1, lkv_ctx)
        out_latent  = einsum('b i j, b j d -> b i d', attn2, lkv)
        out_latent  = rearrange(out_latent,  '(b h) n d -> b n (h d)', h=h)
        out_context = rearrange(out_context, '(b h) n d -> b n (h d)', h=h)

        concatenate_out = torch.cat([out_context, out_latent], dim=1)
        return self.to_out(concatenate_out)


class PerceiverARTransformer(nn.Module):
    def __init__(self, num_unique_tokens, dim, num_layers, heads, sequence_len,
                 causal, ff_mult, ff_dropout, attn_dropout):
        super().__init__()
        self.sequence_len = sequence_len
        self.dim = dim
        self.token_emb = nn.Linear(config["feature_size"], dim, bias=False)
        self.num_layers = num_layers
        self.heads = heads
        self.dim_head = dim // heads

        self.layers = nn.ModuleList([])
        self.PE = Sinusoidal2DPE(dim=dim, grid_size=config["grid_size"])

        for i in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, PerceiverARAttention(
                    dim=dim,
                    heads=heads,
                    sequence_len=sequence_len,
                    causal=causal,
                    layer_num=i,
                    dropout=attn_dropout
                )),
                PreNorm(dim, FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout))
            ]))

        self.to_logits = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_unique_tokens))

        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p, gain=0.1)

    def forward(self, input_tensor):
        token_tensor = self.token_emb(input_tensor)
        token_tensor = token_tensor + self.PE(input_tensor)

        for attn, ff in self.layers:
            token_tensor = attn(token_tensor) + token_tensor  # REMOVED: mask=None arg
            token_tensor = ff(token_tensor) + token_tensor

        transformerPerceiverAR_output = self.to_logits(
            token_tensor[:, -config["foundroute_matrix_length"]:, :]
        )
        return transformerPerceiverAR_output

