import torch
from pyro.distributions import Delta, Normal
from torch.distributions.utils import broadcast_all

from ...typing import ParameterType
from ..linear import LinearModel


def _initial_kernel(loc):
    return Delta(loc)


class RandomWalk(LinearModel):
    r"""
    Defines a one-dimensional Gaussian random walk process, i.e. in which the dynamics are given by
        .. math::
            X_{t+1} \sim \mathcal{N}(X_t, \sigma), \newline
            X_0 \sim \mathcal{N}(x_0, \sigma_0),

    where :math:`x_0` is the initial mean, defaulting to zero.
    """

    def __init__(self, scale: ParameterType, initial_mean: ParameterType = 0.0):
        r"""
        Internal initializer for :class:`RandomWalk`.

        Args:
            scale: :math:`\sigma` in class doc.
            initial_mean: parameter specifying the mean of the initial distribution. Defaults to 0.
            kwargs: see base.
        """

        scale, initial_mean = broadcast_all(scale, initial_mean)

        increment_distribution = Normal(torch.tensor(0.0, device=scale.device), torch.tensor(1.0, device=scale.device))
        a = torch.tensor(1.0, device=scale.device, dtype=scale.dtype)

        super().__init__(
            (a, torch.zeros_like(scale), scale),
            increment_distribution=increment_distribution,
            initial_kernel=_initial_kernel,
            initial_parameters=(initial_mean,),
        )

    def expand(self, batch_shape):
        new_parameters = self._expand_parameters(batch_shape)
        new = self._get_checked_instance(RandomWalk)
        super(RandomWalk, new).__init__(
            new_parameters["parameters"],
            self.increment_distribution,
            self._initial_kernel,
            new_parameters["initial_parameters"],
        )

        return new
