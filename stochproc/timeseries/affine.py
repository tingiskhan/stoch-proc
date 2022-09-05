from typing import Tuple

import torch
from pyro.distributions import Distribution, TransformedDistribution
from pyro.distributions.transforms import AffineTransform

from .state import TimeseriesState
from .stochastic_process import StructuralStochasticProcess
from .typing import MeanScaleFun
from ..distributions import DistributionModule


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
            >>>     return alpha + beta * x.values, sigma
            >>>
            >>> def initial_builder(alpha, beta, sigma):
            >>>     return Normal(loc=alpha, scale=sigma / (1 - beta ** 2).sqrt())
            >>>
            >>> alpha = 0.0
            >>> beta = 0.99
            >>> sigma = 0.05
            >>>
            >>> initial_dist = dists.DistributionModule(initial_builder, alpha=alpha, beta=beta, sigma=sigma)
            >>> increment_dist = dists.DistributionModule(Normal, loc=0.0, scale=1.0)
            >>>
            >>> parameters = (alpha, beta, sigma)
            >>> ar_1 = ts.AffineProcess(mean_scale, parameters, initial_dist, increment_dist)
            >>>
            >>> samples = ar_1.sample_path(1_000)
    """

    def __init__(
        self, mean_scale: MeanScaleFun, parameters, initial_dist, increment_dist: DistributionModule, **kwargs
    ):
        """
        Initializes the :class:`AffineProcess` class.

        Args:
            mean_scale: function constructing the mean and scale.
            parameters: see base.
            initial_dist: see base.
            increment_dist: distribution that we location-scale transform.
        """

        super().__init__(parameters=parameters, initial_dist=initial_dist, **kwargs)

        self.mean_scale_fun = mean_scale
        self.increment_dist = increment_dist

    def _define_transdist(self, x: TimeseriesState):
        """
        Helper method for defining an affine transition density given the location and scale.

        Args:
            x: state of the process.

        Returns:
            The resulting affine transformed distribution.
        """

        loc, scale = self.mean_scale(x)

        return TransformedDistribution(self.increment_dist(), AffineTransform(loc, scale, event_dim=self.n_dim))

    def build_density(self, x):
        return self._define_transdist(x)

    def mean_scale(self, x: TimeseriesState, parameters=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and scale of the process evaluated at ``x`` and :meth:`functional_parameters` or ``parameters``.

        Args:
            x: previous state of the process.
            parameters: whether to override the current parameters of the model, otherwise uses
                :meth:`stochproc.timeseries.StructuralStochasticProcess.functional_parameters`.

        Returns:
            Returns the tuple ``(mean, scale)``.
        """

        mean, scale = self.mean_scale_fun(x, *(parameters or self.functional_parameters()))

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
