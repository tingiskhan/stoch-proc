from torch.distributions import Distribution, AffineTransform, TransformedDistribution
import torch
from typing import Tuple
from .stochastic_process import StructuralStochasticProcess
from ..distributions import DistributionModule
from .typing import MeanScaleFun
from .state import TimeseriesState


def _define_transdist(
    loc: torch.Tensor, scale: torch.Tensor, n_dim: int, dist: Distribution
) -> TransformedDistribution:
    """
    Helper method for defining an affine transition density given the location and scale.

    Args:
        loc: location of the distribution.
        scale: scale of the distribution.
        n_dim: dimension of space of the distribution.
        dist: base distribution to apply the location-scale transform..

    Returns:
        The resulting affine transformed distribution.
    """

    batch_shape = loc.shape[:loc.dim() - n_dim]

    return TransformedDistribution(
        dist.expand(batch_shape), AffineTransform(loc, scale, event_dim=n_dim), validate_args=False
    )


class AffineProcess(StructuralStochasticProcess):
    """
    Class for defining stochastic processes of affine nature, i.e. where we can express the next state :math:`X_{t+1}`
    given the previous state :math:`X_t` as:
        .. math::
            X_{t+1} = f(X_t, \\theta) + g(X_t, \\theta) \\cdot W_{t+1},

    where :math:`\\theta` denotes the parameter set governing the functions :math:`f` and :math:`g`, and :math:`W_t`
    denotes random variable with arbitrary density (from which we can sample).

    Example:
        One example of an affine stochastic process is the AR(1) process. We define it by:
            >>> from stochproc import timeseries as ts, distributions as dists, NamedParameter
            >>> from torch.distributions import Normal, TransformedDistribution, AffineTransform
            >>>
            >>> def mean_scale(x, alpha, beta, sigma):
            >>>     return alpha + beta * x.values, sigma
            >>>
            >>> def initial_builder(alpha, beta, sigma):
            >>>     return Normal(loc=alpha, scale=sigma / (1 - beta ** 2).sqrt())
            >>>
            >>> alpha = NamedParameter("alpha", 0.0)
            >>> beta = NamedParameter("beta", 0.99)
            >>> sigma = NamedParameter("sigma", 0.05)
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
        self,
        mean_scale: MeanScaleFun,
        parameters,
        initial_dist,
        increment_dist: DistributionModule,
        **kwargs
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

    def build_density(self, x):
        loc, scale = self.mean_scale(x)

        return _define_transdist(loc, scale, self.n_dim, self.increment_dist())

    def mean_scale(self, x: TimeseriesState, parameters=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the mean and scale of the process evaluated at ``x`` and ``.functional_parameters()`` or ``parameters``.

        Args:
            x: previous state of the process.
            parameters: whether to override the current parameters of the model, otherwise uses
                :meth:`stochproc.timeseries.StructuralStochasticProcess.functional_parameters`.

        Returns:
            Returns the tuple ``(mean, scale)``.
        """

        mean, scale = self.mean_scale_fun(x, *(parameters or self.functional_parameters()))

        return torch.broadcast_tensors(mean, scale)

    def propagate_conditional(
            self, x: TimeseriesState, u: torch.Tensor, parameters=None, time_increment=1.0
    ) -> TimeseriesState:
        super(AffineProcess, self).propagate_conditional(x, u, parameters, time_increment)

        for _ in range(self.num_steps):
            loc, scale = self.mean_scale(x, parameters=parameters)
            x = x.propagate_from(values=loc + scale * u, time_increment=time_increment)

        return x
