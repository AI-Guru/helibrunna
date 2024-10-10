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
    min_gru_expansion: float = 1.5
    enable_conv: bool = False
    conv_kernel_size: int = 3

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# appendix B
# https://github.com/glassroom/heinsen_sequence
def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
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
    def __init__(self, dim, expansion_factor=1.):
        super().__init__()
        dim_inner = int(dim * expansion_factor)
        self.to_hidden_and_gate = Linear(dim, dim_inner * 2, bias=False)
        self.to_out = Linear(dim_inner, dim, bias=False) if expansion_factor != 1. else nn.Identity()

    def forward(self, x, prev_hidden=None, return_next_prev_hidden=False):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim=-1)

        if seq_len == 1:
            # handle sequential
            hidden = g(hidden)
            gate = gate.sigmoid()
            out = torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)
        else:
            # parallel
            log_coeffs = -F.softplus(gate)
            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((log_g(prev_hidden), log_values), dim=1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]
        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out

        return out, next_prev_hidden

# classes
class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)

def FeedForward(dim, mult=4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

class CausalDepthWiseConv1d(Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim),
            nn.Conv1d(dim, dim, kernel_size=1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value=0.)
        x = self.net(x)
        return x.transpose(1, 2)  # b d n -> b n d


class MinGRULM(Module):
    def __init__(self, config: MinGRULMConfig):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = ModuleList([])

        for _ in range(config.num_hidden_layers):
            self.layers.append(ModuleList([
                CausalDepthWiseConv1d(config.hidden_size, config.conv_kernel_size) if config.enable_conv else None,
                RMSNorm(config.hidden_size),
                MinGRU(config.hidden_size, expansion_factor=config.min_gru_expansion),
                RMSNorm(config.hidden_size),
                FeedForward(config.hidden_size, mult=config.ff_mult)
            ]))

        self.norm = RMSNorm(config.hidden_size)
        self.to_logits = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        x,
        return_loss=False,
        return_prev_hiddens=False,
        prev_hiddens=None
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        # handle previous hiddens, for recurrent decoding
        if exists(prev_hiddens):
            x = x[:, -1:]

        next_prev_hiddens = []
        prev_hiddens = iter(default(prev_hiddens, []))

        for conv, norm, mingru, ff_norm, ff in self.layers:
            # conv
            if exists(conv):
                assert not exists(prev_hiddens), 'caching not supported for conv version'
                x = conv(x) + x

            # min gru
            prev_hidden = next(prev_hiddens, None)
            min_gru_out, next_prev_hidden = mingru(
                norm(x),
                prev_hidden,
                return_next_prev_hidden=True
            )
            x = min_gru_out + x
            next_prev_hiddens.append(next_prev_hidden)

            # feedforward
            x = ff(ff_norm(x)) + x

        embed = self.norm(x)
        logits = self.to_logits(embed)

        if not return_loss:
            if not return_prev_hiddens:
                return logits
            return logits, next_prev_hiddens

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels
        )

        return loss