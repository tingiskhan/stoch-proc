import math
from abc import ABC

import torch
from pyro.distributions import Delta, Normal

from ..typing import ParameterType
from .affine import AffineProcess, MeanScaleFun
from .stochastic_process import StructuralStochasticProcess
from .typing import Drift

_info = torch.finfo(torch.get_default_dtype())
EPS = math.sqrt(_info.eps)


class StochasticDifferentialEquation(StructuralStochasticProcess, ABC):
    r"""
    Abstract base class for stochastic differential equations, i.e. stochastic processes given by
        .. math::
            dX_{t+1} = h(X_t, \theta, dt),

    where :math:`\theta` is the parameter set, and :math:`h` the dynamics of the SDE.
    """

    def __init__(self, *args, dt: float, **kwargs):
        """
        Internal initializer for :class:`StochasticDifferentialEquation`.

        Args:
            parameters: see base.
            initial_dist: see base.
            dt: time discretization to use.
            kwargs: see base.
        """

        super().__init__(*args, **kwargs)
        self.dt = dt

    def propagate(self, x, time_increment=1):
        res = super().propagate(x, time_increment=time_increment)
        res["dt"] = self.dt

        return res


class DiscretizedStochasticDifferentialEquation(StochasticDifferentialEquation):
    r"""
    Defines a discretized stochastic differential equation, in which the next state :math:`X_{t+\Delta t}` is given by
        .. math::
            X_{t + \Delta t} = h(X_t, \theta, \Delta t).

    This e.g. encompasses the Euler-Maruyama and Milstein schemes.
    """

    def build_density(self, x):
        return self._kernel(x, *(self.parameters + (self.dt,)))


class AffineEulerMaruyama(AffineProcess):
    r"""
    Defines the Euler-Maruyama scheme for an SDE of affine nature, i.e. in which the dynamics are given by the
    functional pair of the drift and diffusion, such that we have
        .. math::
            X_{t + \Delta t} = X_t + f_\theta(X_t) \Delta t + g_\theta(X_t) \Delta W_t,

    where :math:`W_t` is an arbitrary random variable from which we can sample.
    """

    def __init__(
        self, dynamics: MeanScaleFun, parameters, increment_distribution, dt, initial_kernel, initial_parameters=None
    ):
        """
        Internal initializer for :class:`AffineEulerMaruyama`.

        Args:
            dynamics: callable returning the drift and diffusion.
            parameters: see base.
            initial_dist: see base.
            increment_dist: see base.
            dt: see base.
            kwargs: see base.
        """

        super().__init__(
            dynamics, parameters, increment_distribution, initial_kernel, initial_parameters=initial_parameters
        )

        # TODO: Code duplication...
        self.dt = dt

    def propagate(self, x, time_increment=1):
        res = super().propagate(x, time_increment=time_increment)
        res["dt"] = self.dt

        return res

    def mean_scale(self, x, parameters=None):
        drift, diffusion = super().mean_scale(x, parameters)
        return x.value + drift * self.dt, diffusion


class Euler(AffineEulerMaruyama):
    r"""
    Implements the standard Euler scheme for an ODE by reframing the model into a stochastic process using low variance
    Normal distributed noise in the state process [1]. That is, given an ODE of the form
        .. math::
            \frac{dx}{dt} = f_\theta(x_t),

    we recast the model into
        .. math::
            X_{t + \Delta t} = X_t + f_\theta(X_t) \Delta t + W_t,
            _
    where :math:`W_t` is a zero mean Gaussian distribution with a tunable standard deviation :math:`\sigma_{tune}`. In
    the limit that :math:`\sigma_{tune} \rightarrow 0` we get the standard Euler scheme for ODEs. Thus, the higher the
    value of :math:`\sigma_{tune}`, the more we allow to deviate from the original model.

    References:
        [1]: https://arxiv.org/abs/2011.09718?context=stat
    """

    def __init__(
        self,
        dynamics: Drift,
        parameters,
        initial_values: ParameterType,
        dt,
        event_dim: int,
        tuning_std: float = False,
    ):
        """
        Internal initializer for :class:`Euler`.

        Args:
            dynamics: the function :math:`f` in the main docs.
            parameters: see base.
            initial_values: initial value(s) of the ODE. Can be a prior as well.
            dt: see base.
            event_dim: event dimension.
            tuning_std: tuning standard deviation of the Gaussian distribution.
            kwargs: see base.
        """

        # TODO: Consider making methods instead of bound functions...
        def initial_kernel(loc, scale):
            if tuning_std:
                return Normal(loc, scale).to_event(event_dim)

            return Delta(v=initial_values, event_dim=event_dim)

        def mean_scale(x, *args, **kwargs):
            return dynamics(x, *args, **kwargs), tuning_std * torch.ones_like(x.value)

        if not tuning_std:
            dist = Delta(v=torch.zeros_like(initial_values), event_dim=event_dim)
        else:
            dist = Normal(
                loc=torch.zeros_like(initial_values), scale=math.sqrt(dt) * torch.ones_like(initial_values)
            ).to_event(event_dim)

        super().__init__(
            dynamics=mean_scale,
            increment_distribution=dist,
            dt=dt,
            parameters=parameters,
            initial_kernel=initial_kernel,
            initial_parameters=(initial_values, EPS),
        )
        self.f = dynamics


class RungeKutta(Euler):
    """
    Same as :class:`Euler`, but instead of utilizing the Euler scheme, we use the `RK4 method`_.

    .. _`RK4 method`: https://en.wikipedia.org/wiki/Runge–Kutta_methods
    """

    def mean_scale(self, x, parameters=None):
        params = parameters or self.parameters

        k1, g = self.mean_scale_fun(x, *params)
        k2 = self.f(x.propagate_from(time_increment=self.dt / 2, values=x.value + self.dt * k1 / 2), *params)
        k3 = self.f(x.propagate_from(time_increment=self.dt / 2, values=x.value + self.dt * k2 / 2), *params)
        k4 = self.f(x.propagate_from(time_increment=self.dt, values=x.value + self.dt * k3), *params)

        return x.value + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), g
