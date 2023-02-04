from abc import ABC
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from typing import Callable, Sequence, Tuple, TypeVar, Dict

import pyro
import torch
from pyro.distributions import Distribution, Normal

from ..typing import ParameterType
from .result import TimeseriesPath
from .state import TimeseriesState
from .utils import coerce_tensors

T = TypeVar("T")

Kernel = Callable[[TimeseriesState, Sequence[ParameterType]], Distribution]
InitialKernel = Callable[[Sequence[ParameterType]], Distribution]


class StructuralStochasticProcess(ABC):
    r"""
    Abstract base class for stochastic processes. By "stochastic process" we mean a sequence of random variables,
    :math:`\{X_t\}_{t \in T}`, defined on a common probability space
    :math:`\{ \Omega, \mathcal{F}, \{ \mathcal{F}_t \}`, with joint distribution
        .. math::
            p(x_0, ..., x_t) = p(x_0) \prod^t_{k=1} p(x_k \mid x_{1:k-1})

    Derived classes should override the :meth:`build_density` method, which builds the distribution of
    :math:`X_{t+1}` given :math:`\{ X_j \}_{j \leq t}`.
    """

    def __init__(
        self,
        kernel: Kernel,
        parameters: Sequence[ParameterType],
        initial_kernel: InitialKernel,
        initial_parameters: Sequence[ParameterType] = None,
    ):
        """
        Internal initializer for :class:`StructuralStochasticProcess`.

        Args:
            kernel: kernel that propagates the process from :math:`t` to :math:`t + 1`.
            parameters: parameters that govern the kernel.
            initial_kernel: kernel used to initialize the stochastic process.
            initial_parameters: optional, parameters that govern the initial kernel. If `None`, uses `parameters`.
        """

        self._initial_kernel = initial_kernel
        self._kernel = kernel

        # TODO: Consider using a custom container instead
        self.parameters = coerce_tensors(*parameters)
        self.initial_parameters = coerce_tensors(*initial_parameters) if initial_parameters else self.parameters

        self._event_shape = None if initial_kernel is None else self.initial_distribution.event_shape

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
    def initial_distribution(self) -> Distribution:
        """
        Returns the initial distribution.
        """

        return self._initial_kernel(*self.initial_parameters)

    def initial_sample(self, shape: torch.Size = torch.Size([])) -> TimeseriesState:
        """
        Samples a state from the initial distribution.

        Args:
            shape: the batch shape to use.

        Returns:
            Returns an initial sample of the process wrapped in a ``NewState`` object.
        """

        dist = self.initial_distribution
        if shape:
            dist = dist.expand(shape)

        return TimeseriesState(0, dist.sample, event_shape=dist.event_shape)

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

        return self._kernel(x, *self.parameters)

    def propagate(self, x: TimeseriesState, time_increment=1.0) -> TimeseriesState:
        """
        Propagates the process from :math:`t` to :math:`t + 1`.

        Args:
            x: previous state of the process.
            time_increment: the amount of time steps to increment the time index with.

        Returns:
            The new state of the process.
        """

        density = self.build_density(x)
        return x.propagate_from(values=density.sample, time_increment=time_increment)

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
        for _ in range(1, steps + 1):
            res += (self.propagate(res[-1]),)

        return TimeseriesPath(*res[1:])

    def copy(self) -> "StructuralStochasticProcess":
        """
        Returns a deep copy of the object.
        """

        return deepcopy(self)

    def _pyro_params_only(self, pyro_lib: pyro, obs: torch.Tensor):
        time_index = torch.arange(1, obs.shape[0])

        with torch.no_grad():
            init_state = self.initial_sample()
            batched_state = init_state.propagate_from(values=obs[:-1], time_increment=time_index)

        log_prob = self.build_density(batched_state).log_prob(obs[1:]).sum() + self.initial_distribution.log_prob(
            obs[0]
        )
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

        x = pyro_lib.sample("x_0", self.initial_distribution, obs=obs[0] if obs is not None else None)
        latent = torch.empty((t_final, *state.event_shape))

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

        event_shape = self.initial_distribution.event_shape

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
        log_prob = tot_dist.log_prob(y_eval[1:]).sum(dim=0) + self.initial_distribution.log_prob(y_eval[0])

        pyro.factor("x_log_prob", log_prob)

        return auxiliary

    @staticmethod
    def _check_obs_and_t(t_final: int, obs: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Helper method for coalescing ``t_final`` and ``obs``.

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

    def yield_parameters(self, filt: Callable[[torch.Tensor], bool] = None) -> Dict[str, Sequence[torch.Tensor]]:
        """
        Yields parameters of models.

        Args:
            filt: filter function.

        Returns:
            Sequence[torch.Tensor]: yields the parameters of the model.
        """

        if filt is None:

            def filt(_):
                return True

        return {
            "initial_parameters": tuple(p for p in self.initial_parameters if filt(p)),
            "parameters": tuple(p for p in self.parameters if filt(p)),
        }

    @contextmanager
    def override_parameters(self, parameters: Sequence[ParameterType]):
        """
        Manually overrides current parameter set with :attr:`parameters`.

        Args:
            parameters (Sequence[ParameterType]): parameters to override with.
        """

        old_parameters = self.parameters

        try:
            param_len = len(old_parameters)
            new_param_len = len(parameters)

            msg = f"Number of parameters is not congruent, you're trying to override {param_len:d} with {new_param_len:d}!"

            assert len(parameters) == len(old_parameters), msg
            self.parameters = parameters

            yield self
        finally:
            self.parameters = old_parameters
    
    def _get_checked_instance(self: T, cls, _instance=None) -> T:
        """
        Basically copies the method of same name in :class:`torch.distributions.Distribution` .

        Returns:
            StructuralStochasticProcess: new instance.
        """

        if _instance is None and type(self).__init__ != cls.__init__:
            raise NotImplementedError("Subclass {} of {} that defines a custom __init__ method "
                                      "must also define a custom .expand() method.".
                                      format(self.__class__.__name__, cls.__name__))

        return self.__new__(type(self)) if _instance is None else _instance

    def _apply_parameters(self, f: Callable[[torch.Tensor], torch.Tensor]) -> Dict[str, Sequence[ParameterType]]:
        """
        Applies `f` to each tensor and returns a dictionary.

        Args:
            batch_shape (torch.Size): batch shape to use.

        Returns:
            Dict[str, Tuple[ParameterType, ...]]: dictionary index by `parameters` and `initial_parameters`.
        """

        new_parameters = OrderedDict([])
        for key, parameters in self.yield_parameters().items():
            new_parameters[key] = tuple(f(p) for p in parameters)
        
        return new_parameters

    def _expand_parameters(self, batch_shape: torch.Size):
        return self._apply_parameters(lambda u: u.expand(batch_shape + u.shape))

    def expand(self: T, batch_shape: torch.Size) -> T:
        """
        Expands parameters of self by `batch_size`.

        Args:
            batch_shape (torch.Size): batch shape to use.
        """

        new_parameters = self._expand_parameters(batch_shape)
        return StructuralStochasticProcess(self._kernel, new_parameters["parameters"], self._initial_kernel, new_parameters["initial_parameters"])
