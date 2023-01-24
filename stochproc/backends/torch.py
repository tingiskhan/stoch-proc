import torch
from pyro.distributions import Distribution, TransformedDistribution, transforms as t

from .backend import Backend


_DTYPE_MAP = {    
}


class Torch(Backend[torch.Tensor, Distribution]):
    """
    Implements torch as backend.
    """

    def coerce_arrays(self, *x):
        tensors = tuple(p if isinstance(p, torch.Tensor) else torch.tensor(p) for p in x)

        assert all(t.device == tensors[0].device for t in tensors), "All tensors do not have the same device!"

        return tensors

    def affine_transform(self, base, loc, scale, n_dim):
        return TransformedDistribution(base, t.AffineTransform(loc, scale, event_dim=n_dim))

    def broadcast_arrays(self, *x):
        return torch.broadcast_tensors(*x)
