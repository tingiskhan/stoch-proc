from typing import Tuple

import torch
from pyro.distributions import Distribution, TransformedDistribution
from pyro.distributions.transforms import AffineTransform

from .state import TimeseriesState
from .stochastic_process import StructuralStochasticProcess
from .typing import MeanScaleFun


class AffineProcess(StructuralStochasticProcess):
    r"""
    Class for defining stochastic processes of affine nature, i.e. where we can express the next state :math:`X_{t+1}`
    given the previous state :math:`X_t` as:
        .. math::
            X_{t+1} = f(X_t, \theta) + g(X_t, \theta) \cdot W_{t+1},

    where :math:`X \in \mathbb{R}^n`, :math:`\theta \in \Theta \subset \mathbb{R}^m`,
    :math:`f, g : \: \mathbb{R}^n \times \Theta \rightarrow \mathbb{R}^n`, and :math:`W_t` denotes a random variable
    with arbitrary density (from which we can sample).

    Example:
        One example of an affine stochastic process is the AR(1) process. We define it by:
            >>> from stochproc import timeseries as ts, distributions as dists
            >>> from torch.distributions import Normal, TransformedDistribution, AffineTransform
            >>>
            >>> def mean_scale(x, alpha, beta, sigma):
            >>>     return alpha + beta * x.value, sigma
            >>>
            >>> def initial_builder(alpha, beta, sigma):
            >>>     return Normal(loc=alpha, scale=sigma / (1 - beta ** 2).sqrt())
            >>>
            >>> alpha = 0.0
            >>> beta = 0.99
            >>> sigma = 0.05
            >>>
            >>> increment_dist = Normal(loc=0.0, scale=1.0)
            >>>
            >>> parameters = (alpha, beta, sigma)
            >>> ar_1 = ts.AffineProcess(mean_scale, increment_dist, parameters, initial_dist)
            >>>
            >>> samples = ar_1.sample_path(1_000)
    """

    def __init__(
        self,
        mean_scale: MeanScaleFun,
        parameters,
        increment_distribution: Distribution,
        initial_kernel,
        initial_parameters=None,
    ):
        """
        Internal initializer for :class:`AffineProcess`.

        Args:
            mean_scale: function constructing the mean and scale.
            parameters: see base.
            initial_dist: see base.
            increment_dist: distribution that we location-scale transform.
        """

        super().__init__(
            self._mean_scale_kernel,
            parameters=parameters,
            initial_kernel=initial_kernel,
            initial_parameters=initial_parameters,
        )

        self.mean_scale_fun = mean_scale
        self.increment_distribution = increment_distribution

    def _mean_scale_kernel(self, x, *_):
        loc, scale = self.mean_scale(x)

        return TransformedDistribution(self.increment_distribution, AffineTransform(loc, scale, event_dim=self.n_dim))

    def mean_scale(self, x: TimeseriesState) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and scale of the process evaluated at ``x`` and :meth:`functional_parameters` or ``parameters``.

        Args:
            x: previous state of the process.

        Returns:
            Returns the tuple ``(mean, scale)``.
        """

        mean, scale = self.mean_scale_fun(x, *self.parameters)

        return torch.broadcast_tensors(mean, scale)

    def add_sub_process(self, sub_process: "AffineProcess") -> "AffineProcess":
        """
        Adds a sub process to ``self`` and returns a :class:`stochproc.timeseries.AffineHierarchalProcess`.

        Args:
            sub_process: sub/child process to add.

        Returns:
            Returns an instance of :class:`stochproc.timeseries.AffineHierarchalProcess`.
        """

        from . import AffineHierarchicalProcess

        return AffineHierarchicalProcess(sub_process=sub_process, main_process=self)

    def expand(self, batch_shape):
        new_parameters = self._expand_parameters(batch_shape)
        new = self._get_checked_instance(AffineProcess)
        new.__init__(self.mean_scale_fun, new_parameters["parameters"], self.increment_distribution, self._initial_kernel, new_parameters["initial_parameters"])

        return new
