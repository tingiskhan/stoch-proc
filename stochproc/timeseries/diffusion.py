import math
from abc import ABC

import torch
from pyro.distributions import Normal, Delta

from .affine import AffineProcess, MeanScaleFun
from .stochastic_process import StructuralStochasticProcess
from .typing import DiffusionFunction, Drift
from ..distributions import DistributionModule
from ..typing import ParameterType

_info = torch.finfo(torch.get_default_dtype())
EPS = math.sqrt(_info.eps)


class OneStepEulerMaruyma(AffineProcess):
    r"""
    Implements a one-step Euler-Maruyama model, similar to PyMC3. I.e. where we perform one iteration of the
    following recursion:
        .. math::
            X_{t+1} = X_t + a(X_t) \Delta t + b(X_t) \cdot \Delta W_t
    """

    def __init__(self, mean_scale, parameters, initial_dist, increment_dist, dt: float, **kwargs):
        r"""
        Initializes the :class:`OneStepEulerMaruyma` class.

        Args:
            mean_scale: see base.
            parameters: see base.
            initial_dist: see base.
            increment_dist: see base. However, do not that you need to include the :math:`\Delta t` term yourself in
               the ``DistributionModule`` class.
            dt: The time delta to use.
        """

        super().__init__(mean_scale, parameters, initial_dist, increment_dist, **kwargs)
        self.dt = torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt

    def mean_scale(self, x, parameters=None):
        drift, diffusion = super(OneStepEulerMaruyma, self).mean_scale(x, parameters=parameters)

        return x.values + drift * self.dt, diffusion

    def forward(self, x, time_increment=1.0):
        # TODO: Can we inherit from Discretized instead...?
        return super(OneStepEulerMaruyma, self).forward(x, time_increment=self.dt)


class StochasticDifferentialEquation(StructuralStochasticProcess, ABC):
    r"""
    Abstract base class for stochastic differential equations, i.e. stochastic processes given by
        .. math::
            dX_{t+1} = h(X_t, \theta, dt),

    where :math:`\theta` is the parameter set, and :math:`h` the dynamics of the SDE.
    """

    def __init__(self, parameters, initial_dist: DistributionModule, dt: float, **kwargs):
        """
        Initializes the :class:`StochasticDifferentialEquation`.

        Args:
            parameters: see base.
            initial_dist: see base.
            dt: time discretization to use.
            kwargs: see base.
        """

        super().__init__(parameters=parameters, initial_dist=initial_dist, **kwargs)
        self.register_buffer("dt", torch.tensor(dt) if not isinstance(dt, torch.Tensor) else dt)

    def forward(self, x, time_increment=1.0):
        res = super(StochasticDifferentialEquation, self).forward(x)
        res["dt"] = self.dt

        return res


class DiscretizedStochasticDifferentialEquation(StochasticDifferentialEquation):
    r"""
    Defines a discretized stochastic differential equation, in which the next state :math:`X_{t+\Delta t}` is given by
        .. math::
            X_{t + \Delta t} = h(X_t, \theta, \Delta t).

    This e.g. encompasses the Euler-Maruyama and Milstein schemes.
    """

    def __init__(self, prop_state: DiffusionFunction, parameters, initial_dist: DistributionModule, dt, **kwargs):
        """
        Initializes the :class:`DiscretizedStochasticDifferentialEquation` class.

        Args:
            prop_state: corresponds to the function :math:`h`.
            parameters: see base.
            initial_dist: see base.
            dt: see base.
            kwargs: see base.
        """

        super().__init__(parameters, initial_dist, dt, **kwargs)
        self._propagator = prop_state

    def build_density(self, x):
        return self._propagator(x, self.dt, *self.functional_parameters())


class AffineEulerMaruyama(AffineProcess, StochasticDifferentialEquation):
    r"""
    Defines the Euler-Maruyama scheme for an SDE of affine nature, i.e. in which the dynamics are given by the
    functional pair of the drift and diffusion, such that we have
        .. math::
            X_{t + \Delta t} = X_t + f_\theta(X_t) \Delta t + g_\theta(X_t) \Delta W_t,

    where :math:`W_t` is an arbitrary random variable from which we can sample.
    """

    def __init__(
        self, dynamics: MeanScaleFun, parameters, initial_dist, increment_dist: DistributionModule, dt, **kwargs
    ):
        """
        Initializes the :class:`AffineEulerMaruyama` class.

        Args:
            dynamics: callable returning the drift and diffusion.
            parameters: see base.
            initial_dist: see base.
            increment_dist: see base.
            dt: see base.
            kwargs: see base.
        """

        super(AffineEulerMaruyama, self).__init__(
            dynamics, parameters, initial_dist, dt=dt, increment_dist=increment_dist, **kwargs
        )

    def mean_scale(self, x, parameters=None):
        drift, diffusion = self.mean_scale_fun(x, *(parameters or self.functional_parameters()))
        return x.values + drift * self.dt, diffusion


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
        **kwargs
    ):
        """
        Initializes the :class:`Euler` class.

        Args:
            dynamics: the function :math:`f` in the main docs.
            parameters: see base.
            initial_values: initial value(s) of the ODE. Can be a prior as well.
            dt: see base.
            event_dim: event dimension.
            tuning_std: tuning standard deviation of the Gaussian distribution.
            kwargs: see base.
        """

        if not tuning_std:
            iv = DistributionModule(Delta, v=initial_values, event_dim=event_dim)
            dist = DistributionModule(Delta, v=torch.zeros(initial_values.shape), event_dim=event_dim)
        else:
            iv = DistributionModule(lambda **u: Normal(**u).to_event(event_dim), loc=initial_values, scale=EPS)
            dist = DistributionModule(
                lambda **u: Normal(**u).expand(initial_values.shape).to_event(event_dim), loc=0.0, scale=math.sqrt(dt)
            )

        def _mean_scale(x, *params):
            return dynamics(x, *params), torch.ones_like(x.values) * tuning_std

        super().__init__(_mean_scale, parameters, iv, dist, dt, **kwargs)
        self.f = dynamics


class RungeKutta(Euler):
    """
    Same as :class:`Euler`, but instead of utilizing the Euler scheme, we use the `RK4 method`_.

    .. _`RK4 method`: https://en.wikipedia.org/wiki/Runge–Kutta_methods
    """

    def mean_scale(self, x, parameters=None):
        params = parameters or self.functional_parameters()

        k1, g = self.mean_scale_fun(x, *params)
        k2 = self.f(x.propagate_from(time_increment=self.dt / 2, values=x.values + self.dt * k1 / 2), *params)
        k3 = self.f(x.propagate_from(time_increment=self.dt / 2, values=x.values + self.dt * k2 / 2), *params)
        k4 = self.f(x.propagate_from(time_increment=self.dt, values=x.values + self.dt * k3), *params)

        return x.values + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), g
