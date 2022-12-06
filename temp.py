import pytest
import torch

from pyro.distributions import Normal
from stochproc import distributions as dists, timeseries as ts

def mean_scale(x_, alpha, sigma):
    return alpha * x_.value, sigma * torch.eye(2, device=x_.value.device)


def initial_kernel(alpha, sigma):
    return Normal(loc=0.0, scale=1.0).expand(torch.Size([2])).to_event(1)


params = [
        1.0,
        0.05
    ]


increment_dist = Normal(loc=0.0, scale=1.0).expand(torch.Size([2])).to_event(1)
t = ts.LowerCholeskyAffineProcess(mean_scale, increment_dist, params, initial_kernel)

x = t.sample_states(10)