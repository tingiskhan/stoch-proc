
import pyro
import torch
from pyro.distributions import Distribution

from .result import StateSpacePath
from .state import StateSpaceModelState, TimeseriesState
from .stochastic_process import StructuralStochasticProcess

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
        Initializes the :class:`StateSpaceModel` class.

        Args:
            hidden: hidden process.
            kernel: see :class:`StructuralStochasticProcess`.
            parameters: see :class:`StructuralStochasticProcess`.
            observe_every_step: parameter for specifying the frequency at which we observe the data.
        """

        super().__init__(kernel=kernel, parameters=parameters, initial_kernel=None)

        self.hidden = hidden
        self.observe_every_step = observe_every_step
        self._event_shape = self.initial_sample()["y"].event_shape

    @property
    def initial_distribution(self) -> Distribution:
        raise NotImplementedError("Cannot sample from initial distribution of SSM directly!")

    def _build_initial_state(self, x: TimeseriesState) -> StateSpaceModelState:
        density = self.build_density(x)
        empty = float("nan") * torch.ones(density.batch_shape + density.event_shape, device=x.value.device)

        return StateSpaceModelState(x=x, y=TimeseriesState(x.time_index, values=empty, event_shape=density.event_shape))

    def initial_sample(self, shape: torch.Size = torch.Size([])) -> StateSpaceModelState:
        x_state = self.hidden.initial_sample(shape)
        return self._build_initial_state(x_state)

    def propagate(self, x: StateSpaceModelState, time_increment=1.0) -> TimeseriesState:
        x_state = x["x"]
        hidden_state = self.hidden.propagate(x_state)

        if (hidden_state.time_index - 1) % self.observe_every_step == 0:
            vals = self.build_density(hidden_state).sample
        else:
            vals = _NAN * torch.ones_like(x["y"].value)

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
