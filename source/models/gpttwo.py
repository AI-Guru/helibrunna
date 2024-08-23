import math
from dataclasses import dataclass, field
from typing import Sequence, Union, Literal, Optional
from copy import deepcopy

import torch
from torch import nn
from transformers import GPT2Config, GPT2Model

from xlstm.utils import WeightDecayOptimGroupMixin
# from .components.init import small_init_init_

# NOTE: This is untested code. It is meant to be used as a reference for the actual implementation.

@dataclass
class GPT2LMModelConfig(GPT2Config):
    tie_weights: bool = False
    weight_decay_on_embedding: bool = False
    add_embedding_dropout: bool = False
    num_blocks: int = 1
    num_heads: int = 4
    embedding_dim: int = 128
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    layer_norm_epsilon: float = 1e-5
    n_inner: Optional[int] = None
    vocab_size: int = 50257
    max_position_embeddings: int = 1024  # Add max position embeddings


@dataclass
class GPT2BlockStackConfig:
    num_blocks: int = 1
    embedding_dim: int = 128
    add_post_blocks_norm: bool = True
    dropout: float = 0.0
    gpt2_config: Optional[GPT2Config] = None  # Adding GPT2Config here


class GPT2LMModel(WeightDecayOptimGroupMixin, nn.Module):
    config_class = GPT2LMModelConfig

    def __init__(self, config: GPT2LMModelConfig, **kwargs):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim)
        self.positional_encoding = PositionalEncoding(config.embedding_dim, config.max_position_embeddings)
        self.emb_dropout = nn.Dropout(config.resid_pdrop) if config.add_embedding_dropout else nn.Identity()

        block_stack_config = GPT2BlockStackConfig(
            num_blocks=config.num_blocks,
            embedding_dim=config.embedding_dim,
            add_post_blocks_norm=True,
            dropout=config.resid_pdrop,
            gpt2_config=config  # Pass the GPT2LMModelConfig to GPT2BlockStackConfig
        )
        self.gpt2_block_stack = GPT2BlockStack(config=block_stack_config)

        self.lm_head = nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.vocab_size,
            bias=False,
        )
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

    def reset_parameters(self):
        self.gpt2_block_stack.reset_parameters()

        # Uncomment the small_init_init_ if using this initialization
        # small_init_init_(self.token_embedding.weight, dim=self.config.n_embd)

        if not self.config.tie_weights:
            # small_init_init_(self.lm_head.weight, dim=self.config.n_embd)
            pass

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(idx)
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.emb_dropout(x)
        x = self.gpt2_block_stack(x)
        logits = self.lm_head(x)
        return logits

    def step(
        self, idx: torch.Tensor, state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None, **kwargs
    ) -> tuple[torch.Tensor, dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        x = self.token_embedding(idx)
        x = self.positional_encoding(x)  # Add positional encoding
        x = self.emb_dropout(x)
        x, state = self.gpt2_block_stack.step(x, state=state, **kwargs)
        logits = self.lm_head(x)
        return logits, state

    def _create_weight_decay_optim_groups(self, **kwargs) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = super()._create_weight_decay_optim_groups(**kwargs)
        # remove token embedding and add it to the correct group, according to the config
        weight_decay = list(weight_decay)
        removed = 0
        for idx in range(len(weight_decay)):
            if weight_decay[idx - removed] is self.token_embedding.weight:
                weight_decay.pop(idx - removed)
                removed += 1
        weight_decay = tuple(weight_decay)
        if self.config.weight_decay_on_embedding:
            weight_decay += (self.token_embedding.weight,)
        else:
            no_weight_decay += (self.token_embedding.weight,)

        return weight_decay, no_weight_decay


class GPT2BlockStack(nn.Module):
    config_class = GPT2BlockStackConfig

    def __init__(self, config: GPT2BlockStackConfig):
        super().__init__()
        self.config = config

        self.blocks = self._create_blocks(config=config)
        if config.add_post_blocks_norm:
            self.post_blocks_norm = nn.LayerNorm(config.embedding_dim)
        else:
            self.post_blocks_norm = nn.Identity()

    def _create_blocks(self, config: GPT2BlockStackConfig):
        blocks = []
        for _ in range(config.num_blocks):
            gpt2_block_config = deepcopy(config.gpt2_config)  # Use gpt2_config from GPT2BlockStackConfig
            blocks.append(GPT2Block(config=gpt2_block_config))

        return nn.ModuleList(blocks)

    def reset_parameters(self) -> None:
        for block in self.blocks:
            block.reset_parameters()
        if not isinstance(self.post_blocks_norm, nn.Identity):
            self.post_blocks_norm.reset_parameters()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, **kwargs)

        x = self.post_blocks_norm(x)

        return x

    def step(
        self, x: torch.Tensor, state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None
    ) -> tuple[torch.Tensor, dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        if state is None:
            state = {}

        for block_idx, block in enumerate(self.blocks):
            x, state[f"block_{block_idx}"] = block.step(x, **state.get(f"block_{block_idx}", {}))

        x = self.post_blocks_norm(x)

        return x, state


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        # Self-Attention Layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.attn_pdrop,
            batch_first=True
        )
        
        # Layer Normalization
        self.ln_1 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)
        
        # Feed Forward Network
        self.mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.n_inner if config.n_inner is not None else 4 * config.embedding_dim),
            nn.GELU(),
            nn.Linear(config.n_inner if config.n_inner is not None else 4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.resid_pdrop)
        )
        
        # Another Layer Normalization
        self.ln_2 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_epsilon)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, use_cache: bool = False, **kwargs) -> torch.Tensor:
        # Self-Attention with Mask
        attn_output, _ = self.self_attn(query=x, key=x, value=x, attn_mask=attn_mask, need_weights=False)
        attn_output = nn.functional.dropout(attn_output, p=self.config.resid_pdrop, training=self.training)
        x = self.ln_1(x + attn_output)
        
        # Feed Forward Network
        mlp_output = self.mlp(x)
        x = self.ln_2(x + mlp_output)
        
        return x

    def step(
        self, x: torch.Tensor, state: Optional[dict[str, tuple[torch.Tensor, ...]]] = None, attn_mask: Optional[torch.Tensor] = None, **kwargs
    ) -> tuple[torch.Tensor, dict[str, tuple[torch.Tensor, ...]]]:
        # The step function processes one step at a time in autoregressive generation.
        output = self.forward(x, attn_mask=attn_mask)
        return output, {}  # State is not needed in the same way as LSTMs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return x
