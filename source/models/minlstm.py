import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class MinLSTMLMConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    ff_mult: float = 4.0
    min_lstm_expansion: float = 1.5
    enable_conv: bool = False
    conv_kernel_size: int = 3

def exists(v):
    return v is not None

class MinLSTM(nn.Module):
    def __init__(self, hidden_size, expansion_factor = 1.):
        super().__init__()
        
        self.hidden_size_inner = int(hidden_size * expansion_factor)
        self.to_gates_and_hidden = nn.Linear(hidden_size, self.hidden_size_inner * 3, bias = False)
        self.to_out = nn.Linear(self.hidden_size_inner, hidden_size, bias = False) if expansion_factor != 1. else nn.Identity()

    def forward(self, x, prev_hidden = None, prev_cell = None, return_next_state = False):
        batch_size, seq_len, _ = x.shape
        
        if prev_hidden is None:
            prev_hidden = torch.zeros(batch_size, self.hidden_size_inner, device=x.device)
        if prev_cell is None:
            prev_cell = torch.zeros(batch_size, self.hidden_size_inner, device=x.device)

        gates_and_hidden = self.to_gates_and_hidden(x)
        f_gate, i_gate, tilde_h = gates_and_hidden.chunk(3, dim=-1)

        f_gate = torch.sigmoid(f_gate)
        i_gate = torch.sigmoid(i_gate)

        if seq_len == 1:
            # Handle sequential processing
            sum_f_i = f_gate + i_gate
            f_prime = f_gate / sum_f_i
            i_prime = i_gate / sum_f_i

            next_cell = f_prime * prev_cell + i_prime * tilde_h
            next_hidden = next_cell

        else:
            # Parallel processing using associative scan
            log_f = torch.log(f_gate + 1e-8)
            log_i = torch.log(i_gate + 1e-8)
            
            log_coeffs = torch.stack([log_f, log_i], dim=-1)
            log_coeffs = torch.logsumexp(log_coeffs, dim=-1, keepdim=True) - log_coeffs

            log_f_prime, log_i_prime = log_coeffs.unbind(dim=-1)

            prev_state = torch.stack([prev_cell, prev_hidden], dim=1)
            log_prev_state = torch.log(prev_state + 1e-8)

            log_state = torch.stack([
                log_f_prime + log_prev_state[:, 0:1].expand(-1, seq_len, -1),
                log_i_prime + torch.log(tilde_h + 1e-8)
            ], dim=-1)
            log_state = torch.logsumexp(log_state, dim=-1)

            cumsum_log_f_prime = torch.cumsum(log_f_prime, dim=1)
            log_cell = cumsum_log_f_prime + log_state

            next_cell = torch.exp(log_cell)
            next_hidden = next_cell

        out = self.to_out(next_hidden)

        if not return_next_state:
            return out

        return out, (next_hidden[:, -1:], next_cell[:, -1:])

class MinLSTMLM(nn.Module):

    def __init__(self, config: MinLSTMLMConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([])

        for _ in range(config.num_hidden_layers):
            self.layers.append(nn.ModuleList([
                nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=config.conv_kernel_size, groups=config.hidden_size, padding=config.conv_kernel_size-1) if config.enable_conv else None,
                nn.LayerNorm(config.hidden_size),
                MinLSTM(config.hidden_size, expansion_factor=config.min_lstm_expansion),
                nn.LayerNorm(config.hidden_size),
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.ff_mult),
                    nn.GELU(),
                    nn.Linear(config.hidden_size * config.ff_mult, config.hidden_size)
                )
            ]))

        self.norm = nn.LayerNorm(config.hidden_size)
        self.to_logits = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        x,
        return_loss = False,
        return_prev_states = False,
        prev_states = None
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        if exists(prev_states):
            x = x[:, -1:]

        next_prev_states = []
        prev_states = iter(prev_states) if exists(prev_states) else None

        for conv, norm, MinLSTM, ff_norm, ff in self.layers:
            if exists(conv):
                assert not exists(prev_states), 'caching not supported for conv version'
                x = conv(x.transpose(1, 2)).transpose(1, 2) + x

            prev_state = next(prev_states, None) if exists(prev_states) else None

            MinLSTM_out, next_prev_state = MinLSTM(
                norm(x),
                *prev_state if exists(prev_state) else (None, None),
                return_next_state = True
            )

            x = MinLSTM_out + x
            next_prev_states.append(next_prev_state)

            x = ff(ff_norm(x)) + x

        embed = self.norm(x)
        logits = self.to_logits(embed)

        if not return_loss:
            if not return_prev_states:
                return logits

            return logits, next_prev_states

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels
        )

        return loss