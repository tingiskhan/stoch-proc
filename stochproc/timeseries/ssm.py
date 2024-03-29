import pyro
import torch
from pyro.distributions import Distribution, Normal

from .linear import LinearModel
from .result import StateSpacePath
from .state import StateSpaceModelState, TimeseriesState
from .stochastic_process import StructuralStochasticProcess
from .utils import coerce_tensors

_NAN = float("nan")


class StateSpaceModel(StructuralStochasticProcess):
    r"""
    Class representing a state space model, i.e. a dynamical system given by the pair stochastic processes
    :math:`\{ X_t \}` and :math:`\{ Y_t \}`, where :math:`X_t` is independent from :math:`Y_t`, and :math:`Y_t`
    conditionally independent given :math:`X_t`. See more `here`_.

    .. _`here`: https://en.wikipedia.org/wiki/State-space_representation
    """

    def __init__(self, hidden: StructuralStochasticProcess, kernel, parameters, observe_every_step=1):
        """
        Internal initializer for :class:`StateSpaceModel`.

        Args:
            hidden: hidden process.
            kernel: see :class:`StructuralStochasticProcess`.
            parameters: see :class:`StructuralStochasticProcess`.
            observe_every_step: parameter for specifying the frequency at which we observe the data.
            event_shape: manual override for the event shape.
        """

        StructuralStochasticProcess.__init__(self, kernel=kernel, parameters=parameters, initial_kernel=None)

        self.hidden = hidden
        self.observe_every_step = observe_every_step
        self._infer_event_shape()

    def _infer_event_shape(self):
        """
        Method for inferring and setting the property :prop:`_event_shape`.
        """

        self._event_shape = self.initial_sample()["y"].event_shape

    def _build_initial_state(self, x: TimeseriesState) -> StateSpaceModelState:
        density = self.build_density(x)
        empty = torch.empty(density.batch_shape + density.event_shape, device=x.value.device).fill_(_NAN)

        return StateSpaceModelState(x=x, y=TimeseriesState(x.time_index, values=empty, event_shape=density.event_shape))

    def initial_sample(self, shape: torch.Size = torch.Size([])) -> StateSpaceModelState:
        x_state = self.hidden.initial_sample(shape)
        return self._build_initial_state(x_state)

    @property
    def initial_distribution(self) -> Distribution:
        raise NotImplementedError("Cannot sample from initial distribution of SSM directly!")

    def propagate(self, x: StateSpaceModelState, time_increment=1.0) -> TimeseriesState:
        x_state = x["x"]
        hidden_state = self.hidden.propagate(x_state)

        if (hidden_state.time_index - 1) % self.observe_every_step == 0:
            vals = self.build_density(hidden_state).sample
        else:
            vals = torch.ones_like(x["y"].value).fill_(_NAN)

        return StateSpaceModelState(x=hidden_state, y=x["y"].propagate_from(values=vals))

    def sample_states(
        self, steps: int, samples: torch.Size = torch.Size([]), x_0: TimeseriesState = None
    ) -> StateSpacePath:
        if (x_0 is not None) and isinstance(x_0, TimeseriesState):
            x_0 = self._build_initial_state(x_0)

        path = super().sample_states(steps, samples, x_0)

        return StateSpacePath(*path.path)

    def do_sample_pyro(
        self, pyro_lib: pyro, t_final: int = None, obs: torch.Tensor = None, mode: str = "approximate"
    ) -> torch.Tensor:

        assert mode != "parameters_only", f"Mode cannot be '{mode}'!"

        if obs is not None:
            t_final = obs.shape[0] * self.observe_every_step

        latent = self.hidden.do_sample_pyro(pyro_lib, t_final + 1, mode=mode)
        time = torch.arange(1, latent.shape[0] + 1, device=latent.device)

        x = latent[1 :: self.observe_every_step]
        state = self.hidden.initial_sample().propagate_from(values=x, time_increment=time)
        obs_dist = self.build_density(state)

        if obs is None:
            pyro_lib.sample("y", obs_dist)
        else:
            pyro_lib.factor("y_log_prob", obs_dist.log_prob(obs).sum(dim=0))

        return latent

    def expand(self, batch_shape):
        new_parameters = self._expand_parameters(batch_shape)["parameters"]

        return StateSpaceModel(self.hidden.expand(batch_shape), self._kernel, new_parameters, self.observe_every_step)


class LinearStateSpaceModel(StateSpaceModel, LinearModel):
    """
    Implements a linear Gaussian state space model, similar to :class:`LinearModel` but where the left hand side
    is replaced with :math:`Y_t`.
    """

    def __init__(
        self,
        hidden: StructuralStochasticProcess,
        parameters,
        event_shape: torch.Size,
        observe_every_step=1,
    ):
        """
        Internal initializer for :class:`LinearStateSpaceModel`.

        Args:
            hidden: see :class:`StateSpaceModel`.
            parameters: see :class:`LinearModel`.
            event_shape: see :class:`StateSpaceModel`.
            observe_every_step: see :class:`StateSpaceModel`.
        """

        coerced_parameters = coerce_tensors(*parameters)

        device = coerced_parameters[0].device
        increment_distribution = Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))

        if event_shape:
            increment_distribution = increment_distribution.expand(event_shape).to_event(1)

        LinearModel.__init__(self, coerced_parameters, increment_distribution, None, None)
        StateSpaceModel.__init__(self, hidden, self._kernel, self.parameters, observe_every_step=observe_every_step)

        self._event_shape = event_shape

    def _infer_event_shape(self) -> torch.Size:
        pass

    def add_sub_process(self, sub_process):
        raise NotImplementedError(f"Cannot register a sub process to '{self.__class__}'!")

    def expand(self, batch_shape):
        new_parameters = self._expand_parameters(batch_shape)["parameters"]

        return LinearStateSpaceModel(
            self.hidden.expand(batch_shape),
            new_parameters,
            self.event_shape,
            self.observe_every_step,
        )
