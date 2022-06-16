from abc import ABC
from copy import deepcopy
from typing import TypeVar, Callable, Union, Tuple, Sequence, Iterable

import pyro
import torch
from pyro.distributions import Normal, Distribution
from torch.nn import Module, Parameter, ParameterDict
import warnings

from .state import TimeseriesState
from .result import TimeseriesPath
from ..container import BufferIterable, BufferDict
from ..distributions import DistributionModule
from ..typing import ParameterType

T = TypeVar("T")


class StochasticProcess(Module, ABC):
    r"""
    Abstract base class for stochastic processes. By "stochastic process" we mean a sequence of random variables,
    :math:`\{X_t\}_{t \in T}`, defined on a common probability space
    :math:`\{ \Omega, \mathcal{F}, \{ \mathcal{F}_t \}`, with joint distribution
        .. math::
            p(x_0, ..., x_t) = p(x_0) \prod^t_{k=1} p(x_k \mid x_{1:k-1})

    Derived classes should override the ``.build_distribution(...)`` method, which builds the distribution of
    :math:`X_{t+1}` given :math:`\{ X_j \}_{j \leq t}`.
    """

    _EXOGENOUS = "exogenous"

    def __init__(
        self, initial_dist: Union[None, DistributionModule], exogenous: Sequence[torch.Tensor] = None,
    ):
        """
        Initializes the ``StochasticProcess`` class.

        Args:
            initial_dist: initial distribution of the process. Corresponds to a
                ``stochproc.distributions.DistributionModule`` rather than a ``pytorch`` distribution as we require
                being able to move the distribution between devices.
            exogenous: parameter specifying exogenous data to include.
        """

        super().__init__()
        self._initial_dist = initial_dist

        self._tensor_tuples = BufferIterable(**{self._EXOGENOUS: exogenous})
        self._event_shape = None if initial_dist is None else self.initial_dist.event_shape

    @property
    def exogenous(self) -> torch.Tensor:
        """
        The exogenous variables.
        """

        return self._tensor_tuples.get_as_tensor(self._EXOGENOUS)

    @property
    def event_shape(self) -> torch.Size:
        """
        Returns the event shape of the process.
        """

        return self._event_shape

    @property
    def n_dim(self) -> int:
        """
        Returns the dimension of the process. If it's univariate it returns a 0, 1 for a vector etc, just like
        ``torch``.
        """

        return len(self.event_shape)

    @property
    def num_vars(self) -> int:
        """
        Returns the number of variables of the stochastic process. E.g. if it's a univariate process it returns 1, and
        if it's a multivariate process it returns the number of elements in the vector or matrix.
        """

        return self.event_shape.numel()

    @property
    def initial_dist(self) -> Distribution:
        """
        Returns the initial distribution and any re-parameterization given by ``._init_transform``.
        """

        return self._initial_dist()

    def initial_sample(self, shape: torch.Size = torch.Size([])) -> TimeseriesState:
        """
        Samples a state from the initial distribution.

        Args:
            shape: the batch shape to use.

        Returns:
            Returns an initial sample of the process wrapped in a ``NewState`` object.
        """

        dist = self.initial_dist
        if shape:
            dist = dist.expand(shape)

        return TimeseriesState(0, dist.sample, event_dim=dist.event_shape)

    def build_density(self, x: TimeseriesState) -> Distribution:
        r"""
        Method to be overridden by derived classes. Defines how to construct the transition density to :math:`X_{t+1}`
        given the state at :math:`t`, i.e. this method corresponds to building the density:
            .. math::
                x_{t+1} \sim p \right ( \cdot \mid \{ x_j \}_{j \leq t} \left ).

        Args:
            x: previous state of the process.

        Returns:
            Returns the density of the state at :math:`t+1`.
        """

        raise NotImplementedError()

    def _add_exog_to_state(self, x: TimeseriesState):
        if self._EXOGENOUS in self._tensor_tuples:
            x.add_exog(self.exog[x.time_index.int()])

    def forward(self, x: TimeseriesState, time_increment: int = 1) -> TimeseriesState:
        self._add_exog_to_state(x)

        density = self.build_density(x)
        return x.propagate_from(values=density.sample, time_increment=time_increment)

    def propagate(self, x: TimeseriesState, time_increment=1.0) -> TimeseriesState:
        """
        Propagates the process from a previous state to a new state. Wraps around the ``__call__`` method of
        ``pytorch.nn.Module`` to allow registering forward hooks etc.

        Args:
            x: previous state of the process.
            time_increment: the amount of time steps to increment the time index with.

        Returns:
            The new state of the process.
        """

        return self.__call__(x, time_increment=time_increment)

    def sample_states(
        self, steps: int, samples: torch.Size = torch.Size([]), x_0: TimeseriesState = None
    ) -> TimeseriesPath:
        r"""
        Samples a trajectory from the stochastic process, i.e. samples the collection :math:`\{ X_j \}^T_{j = 0}`,
        where :math:`T` corresponds to ``steps``.

        Args:
            steps: number of steps to sample.
            samples: parameter corresponding to the batch size to sample.
            x_0: whether to use a pre-defined initial state or sample a new one. If ``None`` samples
                an initial state, else uses ``x_s``.

        Returns:
            Returns a tensor of shape ``(steps, [samples], [.n_dim])``.
        """

        x_0 = self.initial_sample(samples) if x_0 is None else x_0

        res = (x_0,)
        for i in range(1, steps + 1):
            res += (self.propagate(res[-1]),)

        return TimeseriesPath(*res[1:])

    def copy(self) -> "StochasticProcess":
        """
        Returns a deep copy of the object.
        """

        return deepcopy(self)

    def append_exog(self, exogenous: torch.Tensor):
        """
        Appends and exogenous variable.

        Args:
            exogenous: new exogenous variable to add.
        """

        self._tensor_tuples[self._EXOGENOUS] += (exogenous,)

    def _pyro_params_only(self, pyro_lib: pyro, obs: torch.Tensor):
        time_index = torch.arange(1, obs.shape[0])

        with torch.no_grad():
            init_state = self.initial_sample()
            batched_state = init_state.propagate_from(values=obs[:-1], time_increment=time_index)

        log_prob = self.build_density(batched_state).log_prob(obs[1:]).sum() + self.initial_dist.log_prob(obs[0])
        pyro_lib.factor("model_prob", log_prob)

        return obs

    def _pyro_full(self, pyro_lib: pyro, t_final: int, obs: torch.Tensor = None):
        """
        Implements the full joint distribution of the states. Not that this one is very slow.

        Args:
            pyro_lib: pyro library.
            t_final: length of the timeseries.
            obs: optional observations.

        Returns:
            Returns the sampled path together with the log likelihood.
        """

        with torch.no_grad():
            state = self.initial_sample()

        x = pyro_lib.sample("x_0", self.initial_dist, obs=obs[0] if obs is not None else None)
        latent = torch.empty((t_final, *state.event_dim))

        latent[0] = x
        for t in pyro_lib.markov(range(1, latent.shape[0])):
            state = state.propagate_from(values=x)

            obs_t = obs[t] if obs is not None else None
            x = pyro.sample(f"x_{t}", self.build_density(state), obs=obs_t)

            latent[t] = x

        pyro.deterministic("x", latent)

        return latent

    def _pyro_approximate(self, pyro_lib: pyro, t_final: int, obs: torch.Tensor = None):
        """
        Implements a vectorized inference model utilizing an auxiliary model comprising a random walk of the same
        dimension as the timeseries dimension.

        Args:
            pyro_lib: pyro library.
            t_final: length of the timeseries.
            obs: optional observations.

        Returns:
            Returns the sampled path.
        """

        event_shape = self.initial_dist.event_shape

        loc = torch.zeros(event_shape)
        scale = torch.ones_like(loc)

        re_interpreted_dims = len(event_shape)

        with torch.no_grad():
            initial_state = self.initial_sample()

        with pyro_lib.plate("time", t_final, dim=-1) as t:
            # NB: This is a purely heuristic approach and I don't really know if you actually can do this...
            rw_dist = Normal(loc=loc, scale=scale).mask(False)
            auxiliary = pyro_lib.sample("_auxiliary", rw_dist.to_event(re_interpreted_dims)).cumsum(dim=0)

            if re_interpreted_dims == 0:
                auxiliary.squeeze_(-1)

            pyro.deterministic("auxiliary", auxiliary)

        state = initial_state.propagate_from(values=auxiliary[:-1], time_increment=t[1:])

        y_eval = auxiliary.clone()
        if obs is not None:
            y_eval = obs

        tot_dist = self.build_density(state)
        log_prob = tot_dist.log_prob(y_eval[1:]).sum(dim=0) + self.initial_dist.log_prob(y_eval[0])

        pyro.factor("x_log_prob", log_prob)

        return auxiliary

    @staticmethod
    def _check_obs_and_t(t_final: int, obs: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Helper method for coalescing `t_final` and `obs`.

        Args:
            t_final: the final time index.
            obs: the (potentially) observed data.

        """

        if (t_final is None) and (obs is None):
            raise Exception("Both 't_final' and 'obs' cannot be None!")
        elif t_final is None:
            t_final = obs.shape[0]
        elif (obs is not None) and (t_final != obs.shape[0]):
            raise Exception(f"'t_final' != 'obs.shape[0]': {t_final:d} != {obs.shape[0]:d}")

        return t_final, obs

    def do_sample_pyro(
        self, pyro_lib: pyro, t_final: int = None, obs: torch.Tensor = None, mode: str = "parameters_only"
    ) -> torch.Tensor:
        """
        Samples pyro primitives for inferring the parameters of the model.

        Args:
            pyro_lib: the pyro library.
            t_final: length to sample.
            obs: the data to generate for.
            mode: the mode of sampling, can be:
                - "parameters_only" - just infers the parameters, not available if ``obs`` is not none
                - "approximate" - uses an approximate scheme by sampling from a RW enabling batched inference
                - "full" - samples from the full joint distribution

        Returns:
            Returns the latent state.

        References:
            https://forum.pyro.ai/t/using-pyro-markov-for-time-series-variational-inference/1960/2
            http://pyro.ai/examples/sir_hmc.html
        """

        t_final, obs = self._check_obs_and_t(t_final, obs)

        if mode == "full":
            latent = self._pyro_full(pyro_lib, t_final, obs)
        elif mode == "approximate":
            latent = self._pyro_approximate(pyro_lib, t_final, obs)
        elif mode == "parameters_only":
            if obs is None:
                raise Exception(f"'{mode}' only works with observable data and `num_steps` == 1")

            latent = self._pyro_params_only(pyro_lib, obs)
        else:
            raise Exception(f"No such mode exists: '{mode}'!")

        return latent


_Parameters = Iterable[ParameterType]


class StructuralStochasticProcess(StochasticProcess, ABC):
    r"""
    Similar to ``StochasticProcess``, but where we assume that the conditional distribution
    :math:`p(x_k \mid  x_{1:k-1})` is further parameterized by a collection of parameters :math:`\theta`, s.t.
        .. math::
            p_{\theta}(x_k \mid x_{1:k-1}) = p(x_k \mid x_{1:k-1}, \theta).

    """

    def __init__(self, parameters: _Parameters, initial_dist, **kwargs):
        """
        Initializes the :class:`StructuralStochasticProcess` class.

        Args:
            parameters: parameters governing the dynamics of the process.
            kwargs: see :class:`StochasticProcess`.
        """

        super().__init__(initial_dist=initial_dist, **kwargs)
        self._functional_parameters = list()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.parameter_dict = ParameterDict()
            self.buffer_dict = BufferDict()

        for i, p in enumerate(parameters):
            if isinstance(p, torch.nn.Parameter):
                self.parameter_dict[str(i)] = v = p
            else:
                self.buffer_dict[str(i)] = v = p if isinstance(p, torch.Tensor) else torch.tensor(p)

            self._functional_parameters.append(v)

    def functional_parameters(self, f: Callable[[torch.Tensor], torch.Tensor] = None) -> Tuple[Parameter, ...]:
        """
        Returns the functional parameters of the process.

        Args:
            f: Optional parameter, whether to apply some sort of transformation to the parameters prior to returning.
        """

        res = self._functional_parameters
        if f is not None:
            res = (f(v) for v in res)

        return tuple(res)

    def _apply(self, fn):
        super(StructuralStochasticProcess, self)._apply(fn)

        res = dict()
        res.update(self.parameter_dict)
        res.update(self.buffer_dict)

        vals = sorted(res.items(), key=lambda u: u[0])
        self._functional_parameters = [v for _, v in vals]

        return self
