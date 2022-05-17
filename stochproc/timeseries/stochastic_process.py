from abc import ABC
from copy import deepcopy
from typing import TypeVar, Callable, Union, Tuple, Sequence, Iterable

import pyro
import torch
from pyro.distributions import Distribution
from torch.nn import Module, Parameter

from .state import TimeseriesState
from ..container import BufferIterable
from ..distributions import DistributionModule, _HasPriorsModule
from ..typing import ParameterType

T = TypeVar("T")


class StochasticProcess(Module, ABC):
    r"""
    Abstract base class for stochastic processes. By "stochastic process" we mean a sequence of random variables,
    :math:`\{X_t\}_{t \in T}`, defined on a common probability space
    :math:`\{ \Omega, \mathcal{F}, \{\mathcal{F\}_t \}`, with joint distribution
        .. math::
            p(x_1, ..., x_t) = p(x_1) \prod^t_{k=2} p(x_k \mid x_{1:k-1})

    Derived classes should override the ``.build_distribution(...)`` method, which builds the distribution of
    :math:`X_{t+1}` given :math:`\{ X_j \}_{j \leq t}`.
    """

    _EXOGENOUS = "exogenous"

    def __init__(
        self,
        initial_dist: DistributionModule,
        initial_transform: Union[Callable[["StochasticProcess", Distribution], Distribution], None] = None,
        num_steps: int = 1,
        exogenous: Sequence[torch.Tensor] = None,
    ):
        """
        Initializes the ``StochasticProcess`` class.

        Args:
            initial_dist: initial distribution of the process. Corresponds to a
                ``stochproc.distributions.DistributionModule`` rather than a ``pytorch`` distribution as we require
                being able to move the distribution between devices.
            initial_transform: parameter allowing for re-parameterizing the initial distribution with
                parameters of the ``StochasticProcess`` object. One example is the Ornstein-Uhlenbeck process, where
                the initial distribution is usually defined as the stationary process of the distribution, which in turn
                is defined by the three parameters governing the process.
            num_steps: parameter allowing to skip time steps when sampling. E.g. if we set ``num_steps`` to 5,
                we only return every fifth sample when propagating the process.
            exogenous: parameter specifying exogenous data to include.
        """

        super().__init__()
        self._initial_dist = initial_dist
        self._init_transform = initial_transform
        self.num_steps = num_steps

        self._tensor_tuples = BufferIterable(**{
            self._EXOGENOUS: tuple(exogenous) if isinstance(exogenous, torch.Tensor) else (exogenous or ())
        })

    @property
    def exogenous(self) -> torch.Tensor:
        """
        The exogenous variables.
        """

        return self._tensor_tuples.get_as_tensor(self._EXOGENOUS)

    @property
    def n_dim(self) -> int:
        """
        Returns the dimension of the process. If it's univariate it returns a 0, 1 for a vector etc, just like
        ``torch``.
        """

        return len(self.initial_dist.event_shape)

    @property
    def num_vars(self) -> int:
        """
        Returns the number of variables of the stochastic process. E.g. if it's a univariate process it returns 1, and
        if it's a multivariate process it returns the number of elements in the vector or matrix.
        """

        return self.initial_dist.event_shape.numel()

    @property
    def initial_dist(self) -> Distribution:
        """
        Returns the initial distribution and any re-parameterization given by ``._init_transform``.
        """

        dist = self._initial_dist()
        if self._init_transform is not None:
            dist = self._init_transform(self, dist)

        return dist

    def initial_sample(self, shape: torch.Size = torch.Size([])) -> TimeseriesState:
        """
        Samples a state from the initial distribution.

        Args:
            shape: the batch shape to use.

        Returns:
            Returns an initial sample of the process wrapped in a ``NewState`` object.
        """

        return TimeseriesState(0.0, self.initial_dist.expand(shape).sample, event_dim=self.initial_dist.event_shape)

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
        if any(self._tensor_tuples[self._EXOGENOUS]):
            x.add_exog(self.exog[x.time_index.int()])

    def forward(self, x: TimeseriesState, time_increment=1.0) -> TimeseriesState:
        self._add_exog_to_state(x)

        for _ in range(self.num_steps):
            density = self.build_density(x)
            x = x.propagate_from(values=density.sample, time_increment=time_increment)

        return x

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

    def sample_path(self, steps: int, samples: torch.Size = torch.Size([]), x_s: TimeseriesState = None) -> torch.Tensor:
        r"""
        Same as ``.sample_states(...)`` but combines the values of the states into a single tensor instead, does not
        include the zeroth sample.

        Returns:
            Returns a tensor of shape ``(steps, [samples], [.n_dim])``.
        """

        states = self.sample_states(steps, samples, x_s)
        return torch.stack(tuple(r.values for r in states[1:]), dim=0)

    def sample_states(self, steps: int, samples: torch.Size = torch.Size([]), x_s: TimeseriesState = None) -> Tuple[TimeseriesState, ...]:
        r"""
        Samples a trajectory from the stochastic process, i.e. samples the collection :math:`\{ X_j \}^T_{j = 0}`,
        where :math:`T` corresponds to ``steps``.

        Args:
            steps: number of steps to sample.
            samples: parameter corresponding to the batch size to sample.
            x_s: whether to use a pre-defined initial state or sample a new one. If ``None`` samples
                an initial state, else uses ``x_s``.

        Returns:
            Returns a tensor of shape ``(steps, [samples], [.n_dim])``.
        """

        x_s = self.initial_sample(samples) if x_s is None else x_s

        res = (x_s,)
        for i in range(1, steps + 1):
            res += (self.propagate(res[-1]),)

        return res

    def copy(self) -> "StochasticProcess":
        """
        Returns a deep copy of the object.
        """

        return deepcopy(self)

    def propagate_conditional(
            self, x: TimeseriesState, u: torch.Tensor, parameters=None, time_increment=1.0
    ) -> TimeseriesState:
        r"""
        Propagate the process conditional on both state and draws from an incremental distribution. This method assumes
        that we may perform the following parameterization:
            .. math::
                X_{t+1} = H(t, \{ X_j \}, W_t},

        where :math:`H: \: T \times \mathcal{X}^t \times \mathcal{W} \rightarrow \mathcal{X}`, where :math:`W_t`
        are samples drawn from the incremental distribution.

        Args:
            x: See ``.propagate(...)``
            u: The samples from the incremental distribution.
            parameters: when performing the re-parameterization we sometimes require the parameters
                of self to be of another dimension. This parameter allows that.
            time_increment: See ``.propagate(...)``.
        """

        self._add_exog_to_state(x)

        return

    def append_exog(self, exogenous: torch.Tensor):
        """
        Appends and exogenous variable.

        Args:
            exogenous: new exogenous variable to add.
        """

        self._tensor_tuples[self._EXOGENOUS] += (exogenous,)

    def do_sample_pyro(self, pyro_lib: pyro, obs: torch.Tensor, n_plates=1):
        """
        Samples pyro primitives for inferring the parameters of the model.

        Args:
            pyro_lib: The pyro library
            obs: The data to generate for.
            n_plates: The number of data plates.

        References:
            https://forum.pyro.ai/t/using-pyro-markov-for-time-series-variational-inference/1960/2
        """

        if self.num_steps != 1:
            raise NotImplementedError(f"Currently do not support {self.num_steps} != 1")

        time_index = torch.arange(1, obs.shape[0])

        init_state = self.initial_sample()
        batched_state = init_state.propagate_from(values=obs[:-1], time_increment=time_index)

        with pyro_lib.plate("data", n_plates):
            dist = pyro_lib.sample("x", self.build_density(batched_state).to_event(1), obs=obs[1:])

        return dist


_Parameters = Iterable[ParameterType]


class StructuralStochasticProcess(StochasticProcess, _HasPriorsModule, ABC):
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

        for i, p in enumerate(parameters):
            self._register_parameter_or_prior(f"parameter_{i}", p)

    def functional_parameters(self, f: Callable[[torch.Tensor], torch.Tensor] = None) -> Tuple[Parameter, ...]:
        """
        Returns the functional parameters of the process.

        Args:
            f: Optional parameter, whether to apply some sort of transformation to the parameters prior to returning.
        """

        res = self.parameters_and_buffers().values()
        if f is not None:
            res = (f(v) for v in res)

        return tuple(res)
