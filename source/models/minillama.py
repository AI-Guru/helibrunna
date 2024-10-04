import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from dataclasses import dataclass

@dataclass
class MiniLlamaConfig:
    vocab_size: int
    hidden_size: int
    hidden_act: str
    context_length: int
    n_layer: int
    n_head: int
    n_local_heads: int
    n_local_kv_heads: int
    num_key_value_heads: int
    attention_dropout: float
    fc_scale: int
    rope_theta: int
    eps: float = 1e-6
    rms_norm_eps: float = 1e-6
    weight_tying: bool = False
    padding_token_id: int = 0

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        https://arxiv.org/pdf/1910.07467.pdf
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.intermediate_size = int(config.fc_scale * self.config.hidden_size * (2 / 3))  # https://arxiv.org/pdf/2302.13971.pdf PAGE 3
        self.c_fc = nn.Linear(self.config.hidden_size, self.intermediate_size, bias=False)
        self.v_proj = nn.Linear(self.config.hidden_size, self.intermediate_size, bias=False)
        self.c_proj = nn.Linear(self.intermediate_size, self.config.hidden_size, bias=False)
        if config.hidden_act == "gelu":
            self.act = F.gelu
        elif config.hidden_act == "silu":
            self.act = F.silu
        else:
            raise ValueError(f"Activation function {config.hidden_act} not supported")

    def forward(self, x):
        x = self.act(self.c_fc(x)) * self.v_proj(x)
        return self.c_proj(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        freqs = (self.inv_freq[:, None].float().expand(-1, position_ids.shape[0]) @ (position_ids.float())).t()
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head attention module."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.n_rep = self.config.n_local_heads // self.config.n_local_kv_heads
        self.num_heads = config.n_head
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = self.config.attention_dropout

        self.wq = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)

        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.context_length,
            base=config.rope_theta
        )

    def forward(self, x, mask=None, training=False):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (hidden_size)

        position_ids = torch.arange(T, device=x.device).repeat([B, 1]).long()

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshaping for multi-head attention.
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Assertions to check dimensions before rotary embeddings
        assert q.shape[-1] == self.head_dim, f"q head_dim mismatch: {q.shape[-1]} != {self.head_dim}"
        assert k.shape[-1] == self.head_dim, f"k head_dim mismatch: {k.shape[-1]} != {self.head_dim}"
        assert v.shape[-1] == self.head_dim, f"v head_dim mismatch: {v.shape[-1]} != {self.head_dim}"

        # Rotary embeddings.
        cos, sin = self.rotary_emb(v, position_ids, seq_len=None)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, None)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        matmul_qk = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            matmul_qk += (mask * -1e9)

        attn_scores = F.softmax(matmul_qk, dim=-1)
        attn_scores = F.dropout(attn_scores, p=self.attention_dropout, training=self.training)
        y = torch.matmul(attn_scores, v)  # Weighted sum

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)


class DecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.hidden_size, eps=config.eps)
        self.attn = Attention(config)
        self.ln_2 = RMSNorm(config.hidden_size, eps=config.eps)
        self.mlp = FeedForward(config)

    def forward(self, x, mask):
        x = self.ln_1(x)
        x = x + self.attn(x, mask=mask)
        x = self.ln_2(x)
        x = x + self.mlp(x)
        return x


class MiniLlama(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_length is not None
        self.config = config

        self.vocab_size = config.vocab_size

        self.padding_token_id = config.padding_token_id

        self.transformer = nn.ModuleDict(dict(
            embedding_layer=nn.Embedding(self.vocab_size, config.hidden_size),
            h=nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            layer_norm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        ))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.weight_tying:
            self.transformer.embedding_layer.weight = self.lm_head.weight
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        taken this method from Andrej karpathy minGPT
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.embedding_layer.weight.numel()
        return n_params

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()

        # Create the causal mask.
        mask = create_masks(idx, self.padding_token_id, device)  # Creating mask to handle left to right attention and mask

        # Create the causal mask. NOTE: This is an replacement of the above because the above is not working with ONNX.
        #mask = 1 - torch.tril(torch.ones(t, t)).to(device) # 1 - was used to make the lower triangle 0
        #mask = mask.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, T, T)

        x = self.transformer.embedding_layer(idx)  # token embeddings of shape (b, t, embd)

        for decoder_block in self.transformer.h:
            x = decoder_block(x, mask)
        x = self.transformer.layer_norm(x)
        assert x.shape[-1] == self.config.hidden_size, f"x shape: {x.shape} hidden_size: {self.config.hidden_size}"
        assert x.shape[1] == t, f"x shape: {x.shape} t: {t}"
        assert x.shape[0] == b, f"x shape: {x.shape} b: {b}"

        logits = self.lm_head(x)  # note: using list [-1] to preserve the time dim

        return logits



    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

def create_masks(inp, padding_token_id, device=None):
    """
    Create both padding mask and attention mask for the input sequence using PyTorch.

    Args:
    inp: Input sequence tensor (PyTorch).

    Returns:
    mask: Combined mask tensor (PyTorch).
    """

    # Padding mask: 1 where inp is 0 (padding), 0 elsewhere
    padding_mask = (inp == padding_token_id).unsqueeze(1).unsqueeze(2).float()

    # Attention mask: 1 for positions that shouldn't be attended to
    seq_len = inp.size(1)
    att_mask = (1 - torch.tril(torch.ones(seq_len, seq_len, device=inp.device)))

    # Combine padding and attention masks
    mask = torch.max(padding_mask, att_mask)

    return mask.to(device)
