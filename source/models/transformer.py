# Helibrunna - A HuggingFace compatible xLSTM trainer.
# Copyright (c) 2024 Dr. Tristan Behrens
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TransformerConfig:
    dim: int
    n_layers: int
    n_heads: int
    fc_scale: int
    vocab_size: int
    context_length: int


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(MultiHeadAttention, self).__init__()
        assert config.dim % config.n_heads == 0, "Dimension must be divisible by number of heads."
        
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(config.dim, config.dim)
        self.key = nn.Linear(config.dim, config.dim)
        self.value = nn.Linear(config.dim, config.dim)
        self.out = nn.Linear(config.dim, config.dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(config.dim, config.fc_scale * config.dim)
        self.fc2 = nn.Linear(config.fc_scale * config.dim, config.dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerLayer, self).__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.dim)
        self.ffn = FeedForward(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        
        # Initialize positional encodings
        self.positional_encoding = self.create_positional_encoding(config.context_length, config.dim)
        
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.context_length = config.context_length

    def create_positional_encoding(self, context_length, dim):
        position = torch.arange(context_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(torch.log(torch.tensor(10000.0)) / dim))
        pos_encoding = torch.zeros(context_length, dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)  # Shape: (1, context_length, dim)

    def forward(self, x):
        B, T = x.shape

        # Create the causal mask (size T x T)
        causal_mask = torch.tril(torch.ones(T, T)).to(x.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, T, T)

        # Embed tokens and add positional encodings
        x = self.embedding(x) + self.positional_encoding[:, :T, :].to(x.device)
        
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
        x = self.ln_f(x)
        return self.head(x)
