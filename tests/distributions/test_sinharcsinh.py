from math import pi, sqrt

import pytest
import torch
from torch.distributions import (
    AffineTransform,
    TransformedDistribution,
    Normal,
)

from stochproc.distributions import SinhArcsinhTransform


@pytest.fixture
def grid():
    return torch.linspace(-5.0, 5.0, steps=1_000)


def _sinh_transform(z, skew, kurt):
    inner = (torch.asinh(z) + skew) * kurt

    return torch.sinh(inner)


def normal_arcsinh_log_prob(x, loc, scale, skew, tail):
    """
    Implements the analytical density for a standard Normal distribution undergoing an Sinh-Archsinh transform:
        https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1740-9713.2019.01245.x
    """

    first = (tail / scale).log()

    y = (x - loc) / scale
    s2 = _sinh_transform(y, -skew / tail, 1 / tail) ** 2
    second = (1 + s2).log() - (2 * pi * (1 + y ** 2)).log()
    third = s2

    return first + 0.5 * (second - third)


EPS = sqrt(torch.finfo(torch.float32).eps)


class TestSinhArcsinh(object):
    def test_sin_arcsinh_no_transform(self, grid):
        sinarc = TransformedDistribution(Normal(0.0, 1.0), SinhArcsinhTransform(0.0, 1.0))
        normal = Normal(0.0, 1.0)

        assert (sinarc.log_prob(grid) - normal.log_prob(grid)).abs().max() <= EPS

    def test_sin_arcsinh_affine_transform(self, grid):
        skew, tail = torch.tensor(1.0), torch.tensor(1.0)
        sinarc = TransformedDistribution(Normal(0.0, 1.0), SinhArcsinhTransform(skew, tail))

        mean, scale = torch.tensor(1.0), torch.tensor(0.5)
        transformed = TransformedDistribution(sinarc, AffineTransform(mean, scale))

        manual = normal_arcsinh_log_prob(grid, mean, scale, skew, tail)

        assert (manual - transformed.log_prob(grid)).abs().max() <= EPS

