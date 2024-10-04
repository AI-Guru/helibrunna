# https://arxiv.org/abs/2410.01201v1

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Module, ModuleList


@dataclass
class MinGRULMConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    ff_mult: int = 4

def exists(v):
    return v is not None

# appendix B
# https://github.com/glassroom/heinsen_sequence

def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

# appendix B.3

def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())

def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

# log-space version of MinGRU - B.3.1
# they enforce the hidden states to be positive

class MinGRU(Module):
    def __init__(self, dim):
        super().__init__()
        self.to_hidden_and_gate = Linear(dim, dim * 2, bias = False)

    def forward(self, x, prev_hidden = None):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim = -1)

        # handle sequential

        if seq_len == 1:
            hidden = g(hidden)
            gate = gate.sigmoid()
            return torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)

        # parallel

        log_coeffs = -F.softplus(gate)

        log_z = -F.softplus(-gate)
        log_tilde_h = log_g(hidden)
        log_values = log_z + log_tilde_h

        if exists(prev_hidden):
            log_values = torch.cat((log_g(prev_hidden), log_values), dim = 1)
            log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

        out = heinsen_associative_scan_log(log_coeffs, log_values)
        return out[:, -seq_len:]



# classes

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# main class

class MinGRULM(Module):
    def __init__(
        self,
        config: MinGRULMConfig
        #*,
        #vocab_size,
        #hidden_size,
        #num_hidden_layers,
        #ff_mult = 4
    ):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = ModuleList([])

        for _ in range(config.num_hidden_layers):
            self.layers.append(ModuleList([
                RMSNorm(config.hidden_size),
                MinGRU(config.hidden_size),
                RMSNorm(config.hidden_size),
                FeedForward(config.hidden_size, mult = config.ff_mult)
            ]))

        self.norm = RMSNorm(config.hidden_size)
        self.to_logits = nn.Linear(config.hidden_size, config.vocab_size, bias = False)

    def forward(
        self,
        x,
        return_loss = False
    ):

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        for norm, MinGRU, ff_norm, ff in self.layers:

            x = MinGRU(norm(x)) + x

            x = ff(ff_norm(x)) + x

        embed = self.norm(x)
        logits = self.to_logits(embed)

        if not return_loss:
            return logits

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels
        )

        return loss
