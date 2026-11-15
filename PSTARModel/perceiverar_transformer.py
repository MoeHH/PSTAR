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


def _causal_latent_mask(n_latent, n_keys, n_context, device):
    mask = torch.zeros(n_latent, n_keys, device=device, dtype=torch.bool)
    latent_block = torch.ones(n_latent, n_keys - n_context, device=device, dtype=torch.bool)
    mask[:, n_context:] = torch.triu(latent_block, diagonal=1)
    return mask


class PerceiverARAttention(nn.Module):
    def __init__(self, dim, heads, causal, sequence_len, dropout, layer_num):
        super().__init__()
        self.sequence_len = sequence_len
        self.context_len = config["context_matrix_length"]
        assert (config["context_matrix_length"] + config["foundroute_matrix_length"] == sequence_len), \
            'context_length plus latent should be equal to sequence length'
        self.layer_num = layer_num
        self.cross_attend = (layer_num == 0)
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
        c = self.context_len

        latent_in = input_tensor[:, c:, :]                       # [B, L, D]
        kv_in = input_tensor if self.cross_attend else latent_in # [B, C+L, D] or [B, L, D]
        n_context_keys = c if self.cross_attend else 0

        q = rearrange(self.to_q(latent_in), 'b n (h d) -> (b h) n d', h=h)   # [B*H, L, d]
        kv = rearrange(self.to_kv(kv_in), 'b n (h d) -> (b h) n d', h=h)     # [B*H, K, d]

        q = q * self.scale
        lkv = self.norm(kv)                                      # per-head LayerNorm, as in PSTAR

        attn = einsum('b i d, b j d -> b i j', q, lkv)           # [B*H, L, K]
        if self.causal:
            mask = _causal_latent_mask(attn.shape[-2], attn.shape[-1], n_context_keys, device)
            attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out_latent = einsum('b i j, b j d -> b i d', attn, lkv)  # [B*H, L, d]
        out_latent = rearrange(out_latent, '(b h) n d -> b n (h d)', h=h)
        out_latent = self.to_out(out_latent)                     # [B, L, D]

        out_context = torch.zeros(b, c, d, device=device, dtype=out_latent.dtype)
        return torch.cat([out_context, out_latent], dim=1)       # [B, C+L, D]


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
        self.context_len = config["context_matrix_length"]

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
        self.register_buffer("architecture_marker", torch.tensor(1, dtype=torch.long))

    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p, gain=0.1)

    def forward(self, input_tensor):
        token_tensor = self.token_emb(input_tensor)
        token_tensor = token_tensor + self.PE(input_tensor)
        c = self.context_len

        for attn, ff in self.layers:
            token_tensor = attn(token_tensor) + token_tensor
            latent = token_tensor[:, c:, :]
            latent = ff(latent) + latent
            token_tensor = torch.cat([token_tensor[:, :c, :], latent], dim=1)

        transformerPerceiverAR_output = self.to_logits(token_tensor[:, -config["foundroute_matrix_length"]:, :])
        
        return transformerPerceiverAR_output
