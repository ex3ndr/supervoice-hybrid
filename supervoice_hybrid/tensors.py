import torch
import torch.nn.functional as F
from einops import rearrange
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class RMSNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1, eps = 1e-05) * self.scale * self.gamma
    
class AdaptiveRMSNorm(torch.nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.scale = dim ** 0.5
        self.to_gamma = torch.nn.Linear(dim, dim)
        self.to_beta = torch.nn.Linear(dim, dim)

        # Identity initialization
        torch.nn.init.zeros_(self.to_gamma.weight)
        torch.nn.init.ones_(self.to_gamma.bias)
        torch.nn.init.zeros_(self.to_beta.weight)
        torch.nn.init.zeros_(self.to_beta.bias)

    def forward(self, x, *, condition):
        normed = F.normalize(x, dim = -1, eps = 1e-05) * self.scale
        gamma, beta = self.to_gamma(condition), self.to_beta(condition)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (gamma, beta))

        return normed * gamma + beta

class ConvPositionEmbed(torch.nn.Module):
    def __init__(self, n_dim, kernel_size):
        super().__init__()
        self.dw_conv1d = torch.nn.Sequential(nn.Conv1d(n_dim, n_dim, kernel_size, groups = n_dim, padding = kernel_size // 2), nn.GELU())

    def forward(self, x, mask = None):

        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.)

        x = rearrange(x, 'b n c -> b c n')
        x = self.dw_conv1d(x)
        out = rearrange(x, 'b c n -> b n c')

        if mask is not None:
            out = out.masked_fill(~mask, 0.)

        return out

class LearnedSinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        half_dim = dim // 2
        self.weights = torch.nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        return fouriered

def probability_binary_mask(shape, true_prob, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1) < true_prob


def debug_if_invalid(x):
    if torch.isnan(x).any() or torch.isinf(x).any():
        print('Invalid tensor')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def drop_using_mask(source, replacement, mask):
    while mask.dim() < source.dim():
        mask = mask.unsqueeze(-1)
    return torch.where(mask, torch.full(source.shape, replacement, dtype = source.dtype, device = source.device), source)

def merge_mask(source, replacement, mask):
    while mask.dim() < source.dim():
        mask = mask.unsqueeze(-1)
    return torch.where(mask, replacement, source)

def random_interval_masking(batch_size, length, *, min_size, min_count, max_count, device):
    tensor = torch.full((batch_size, length), False, device=device, dtype=torch.bool)
    for i in range(batch_size):

        # Expected sum of all intervals
        expected_length = random.randint(min_count, max_count)

        # Number of intervals
        num_intervals = random.randint(1, expected_length // min_size)

        # Generate interval lengths
        lengths = [min_size] * num_intervals
        for _ in range(expected_length - num_intervals * min_size):
            lengths[random.randint(0, num_intervals - 1)] += 1

        # Generate start points
        placements = []
        offset = 0
        remaining = expected_length
        for l in lengths:
            start_position = random.uniform(offset, remaining - l)
            placements.append(start_position)
            offset = start_position + l
            remaining -= l

        # Write to tensor
        for l, p in zip(lengths, placements):
            tensor[i, int(p):int(p + l)] = True

    return tensor

def sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

def list_to_tensors(tensors):

    # Calculate lengths
    l = list(map(len, tensors))

    # Padded tensors
    padded = pad_sequence(tensors, batch_first=True, padding_value=0)

    # Mask
    mask = torch.zeros((padded.shape[0], padded.shape[1]), device=padded.device)
    for i in range(len(l)):
        mask[i, l[i]:] = -10000.0
    
    return padded, mask

def join_tensors(tensors, sep = None):

    # No separator
    if sep is None:
        return torch.cat(tensors)

    # Initial
    res = tensors[0]

    # Join rest
    for t in tensors[1:]:
        res = torch.cat([res, sep, t])
    
    return res
    
def get_slopes_power_of_2(n_heads):
    start = (2**(-2**-(math.log2(n_heads)-3)))
    ratio = start
    return torch.tensor([start*ratio**i for i in range(n_heads)], dtype=torch.float32)