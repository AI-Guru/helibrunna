import torch
import torch.nn.functional as F
from torch import nn
from dataclasses import dataclass
import math

@dataclass
class AethonConfig:
    dim: int
    n_layers: int
    n_heads: int
    n_key_value_heads: int
    fc_scale: int
    vocab_size: int
    context_length: int

# Helper function for rotary embedding
def apply_rotary_pos_emb(q, k, sinusoidal_pos):
    cos_pos = sinusoidal_pos[..., 0::2].repeat_interleave(2, dim=-1)
    sin_pos = sinusoidal_pos[..., 1::2].repeat_interleave(2, dim=-1)

    q_rotated = (q * cos_pos) + (rotate_half(q) * sin_pos)
    k_rotated = (k * cos_pos) + (rotate_half(k) * sin_pos)

    return q_rotated, k_rotated

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, config: AethonConfig):
        super(MultiHeadAttention, self).__init__()
        assert config.dim % config.n_heads == 0, "Dimension must be divisible by number of heads."
        
        self.n_heads = config.n_heads
        self.n_key_value_heads = config.n_key_value_heads
        self.n_key_value_groups = self.n_heads // self.n_key_value_heads
        self.head_dim = config.dim // config.n_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(config.dim, config.dim)
        self.key = nn.Linear(config.dim, self.n_key_value_heads * self.head_dim)
        self.value = nn.Linear(config.dim, self.n_key_value_heads * self.head_dim)
        self.out = nn.Linear(config.dim, config.dim)

    def forward(self, x, mask=None, sinusoidal_pos=None):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings to queries and keys
        if sinusoidal_pos is not None:
            q, k = apply_rotary_pos_emb(q, k, sinusoidal_pos)

        k = k.repeat(1, self.n_key_value_groups, 1, 1)
        v = v.repeat(1, self.n_key_value_groups, 1, 1)

        scores = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)



class FeedForward(nn.Module):
    def __init__(self, config: AethonConfig):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(config.dim, config.fc_scale * config.dim)
        self.fc2 = nn.Linear(config.fc_scale * config.dim, config.dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class AethonLayer(nn.Module):
    def __init__(self, config: AethonConfig):
        super(AethonLayer, self).__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.dim)
        self.ffn = FeedForward(config)

    def forward(self, x, mask=None, sinusoidal_pos=None):
        
        # GPT Order.
        #x = x + self.attn(self.ln1(x), mask, sinusoidal_pos)
        #x = x + self.ffn(self.ln2(x))
        
        # Transformer-XL Order.
        x = self.ln1(x + self.attn(x, mask, sinusoidal_pos))
        x = self.ln2(x + self.ffn(x))
        
        return x


class Aethon(nn.Module):
    def __init__(self, config: AethonConfig):
        super(Aethon, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([AethonLayer(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.context_length = config.context_length

        # Generate rotary sinusoidal embeddings
        self.sinusoidal_pos = self.create_sinusoidal_embedding(config.context_length, config.dim // config.n_heads)

    def create_sinusoidal_embedding(self, context_length, dim):
        position = torch.arange(context_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pos_encoding = torch.zeros(context_length, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, context_length, dim)

    def forward(self, x):
        B, T = x.shape

        # Create the causal mask (size T x T)
        causal_mask = torch.tril(torch.ones(T, T)).to(x.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, T, T)

        # Embed tokens
        x = self.embedding(x)

        # Create sinusoidal positional embeddings for rotary embedding
        sinusoidal_pos = self.sinusoidal_pos[:, :, :T, :].to(x.device)

        # Pass through layers with rotary embedding applied to attention
        for layer in self.layers:
            x = layer(x, mask=causal_mask, sinusoidal_pos=sinusoidal_pos)
        x = self.ln_f(x)
        return self.head(x)
