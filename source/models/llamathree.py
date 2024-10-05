import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class LlamaThreeConfig:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    hidden_dim: int
    n_kv_groups: int
    rope_base: int
    rope_freq: Optional[dict]
    dtype: str
    hidden_activation: str

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False)
        self.fc2 = nn.Linear(config.emb_dim, config.hidden_dim, dtype=config.dtype, bias=False)
        self.fc3 = nn.Linear(config.hidden_dim, config.emb_dim, dtype=config.dtype, bias=False)
        if config.hidden_activation == "silu":
            self.activation = SiLU()
        elif config.hidden_activation == "gelu":
            self.activation = GELU()
        else:
            raise ValueError(f"Unsupported activation: {config.hidden_activation}")

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.activation(x_fc1) * x_fc2
        return self.fc3(x)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight).to(dtype=x.dtype)
        

class GroupedQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, context_length, num_heads,
            num_kv_groups,       # NEW
            rope_base=10_000,    # NEW
            rope_config=None,    # NEW
            dtype=None
        ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        ############################# NEW  #############################
        # self.W_key = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        # self.W_value = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        ################################################################

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        cos, sin = precompute_rope_params(
            head_dim=self.head_dim,
            theta_base=rope_base,      # NEW
            freq_config=rope_config,   # NEW
            context_length=8192
        )
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)  # Shape: (b, num_tokens, d_out)
        keys = self.W_key(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)  # Shape: (b, num_tokens, num_kv_groups * head_dim)

        # Reshape queries, keys, and values
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        ##################### NEW  #####################
        # keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        # values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        ################################################

        # Transpose keys, values, and queries
        keys = keys.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.transpose(1, 2)  # Shape: (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # Shape: (b, num_query_groups, num_tokens, head_dim)

        # Apply RoPE
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        ##################### NEW  #####################
        # Expand keys and values to match the number of heads
        # Shape: (b, num_heads, num_tokens, head_dim)

        keys = keys.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        values = values.repeat_interleave(self.group_size, dim=1)  # Shape: (b, num_heads, num_tokens, head_dim)
        # For example, before repeat_interleave along dim=1 (query groups):
        #   [K1, K2]
        # After repeat_interleave (each query group is repeated group_size times):
        #   [K1, K1, K2, K2]
        # If we used regular repeat instead of repeat_interleave, we'd get:
        #   [K1, K2, K1, K2]
        ################################################

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # Shape: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec
    

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att =  GroupedQueryAttention(  # MultiHeadAttention(
            d_in=config.emb_dim,
            d_out=config.emb_dim,
            context_length=config.context_length,
            num_heads=config.n_heads,
            num_kv_groups=config.n_kv_groups,  # NEW
            rope_base=config.rope_base,        # NEW
            rope_config=config.rope_freq,      # NEW
            dtype=config.dtype
        )
        self.ff = FeedForward(config)
        self.norm1 = RMSNorm(config.emb_dim, eps=1e-5)
        self.norm2 = RMSNorm(config.emb_dim, eps=1e-5)
        self.dtype = config.dtype

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x.to(self.dtype))   # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x.to(self.dtype))
        x = x + shortcut  # Add the original input back

        return x


class LlamaThree(nn.Module):

    def __init__(self, config):
        super().__init__()

        if config.dtype == "float32":
            config.dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {config.dtype}")
        
        self.dtype = config.dtype

        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim, dtype=config.dtype)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)])

        self.final_norm = RMSNorm(config.emb_dim, eps=1e-5)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False, dtype=config.dtype)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.dtype))
        return logits

 
def precompute_rope_params(head_dim, theta_base=10000, context_length=4096, freq_config=None):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim // 2) / (head_dim // 2)))

    ################################ NEW ###############################################
    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama
    ####################################################################################


    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def precompute_rope_params(head_dim, theta_base=10000, context_length=4096, freq_config=None):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim // 2) / (head_dim // 2)))

    ################################ NEW ###############################################
    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama
    ####################################################################################


    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)

