from typing import Tuple, Callable

import pyro
import torch
from pyro.distributions import Distribution

from .stochastic_process import StructuralStochasticProcess
from .state import TimeseriesState, StateSpaceModelState
from .result import StateSpacePath


DistBuilder = Callable[[TimeseriesState, Tuple[torch.Tensor, ...]], Distribution]


class StateSpaceModel(StructuralStochasticProcess):
    r"""
    Class representing a state space model, i.e. a dynamical system given by the pair stochastic processes
    :math:`\{ X_t \}` and :math:`\{ Y_t \}`, where :math:`X_t` is independent from :math:`Y_t`, and :math:`Y_t`
    conditionally independent given :math:`X_t`. See more `here`_.

    .. _`here`: https://en.wikipedia.org/wiki/State-space_representation
    """

    def __init__(self, hidden: StructuralStochasticProcess, f: DistBuilder, parameters, **kwargs):
        """
        Initializes the :class:`StateSpaceModel` class.

        Args:
            hidden: hidden process.
        """

        super().__init__(parameters=parameters, initial_dist=None, **kwargs)
        self.hidden = hidden
        self._dist_builder = f

    def initial_dist(self) -> Distribution:
        raise NotImplementedError("Cannot sample from initial distribution of SSM directly!")

    def _build_initial_state(self, x: TimeseriesState) -> StateSpaceModelState:
        init_dist = self.build_density(x)

        return StateSpaceModelState(
            x=x, y=TimeseriesState(0.0, values=torch.tensor([]), event_dim=init_dist.event_shape)
        )

    def initial_sample(self, shape: torch.Size = torch.Size([])) -> StateSpaceModelState:
        x_state = self.hidden.initial_sample(shape)
        return self._build_initial_state(x_state)

    def build_density(self, x):
        return self._dist_builder(x, *self.functional_parameters())

    def _add_exog_to_state(self, x: TimeseriesState):
        if self._EXOGENOUS in self._tensor_tuples:
            x.add_exog(self.exog[(x.time_index - 1.0).int()])

    def forward(self, x: StateSpaceModelState, time_increment=1.0) -> TimeseriesState:
        x_state = x["x"]
        hidden_state = self.hidden.propagate(x_state)

        self._add_exog_to_state(hidden_state)
        dist = self.build_density(hidden_state)

        return StateSpaceModelState(x=hidden_state, y=x["y"].propagate_from(values=dist.sample))

    def sample_states(
        self, steps: int, samples: torch.Size = torch.Size([]), x_0: TimeseriesState = None
    ) -> StateSpacePath:
        if (x_0 is not None) and isinstance(x_0, TimeseriesState):
            x_0 = self._build_initial_state(x_0)

        path = super(StateSpaceModel, self).sample_states(steps, samples, x_0)

        return StateSpacePath(*path.path)

    def do_sample_pyro(
            self, pyro_lib: pyro, t_final: int = None, obs: torch.Tensor = None, mode: str = "approximate"
    ) -> torch.Tensor:

        assert mode != "parameters_only", f"Mode cannot be '{mode}'!"

        t_final, obs = self._check_obs_and_t(t_final, obs)

        latent = self.hidden.do_sample_pyro(pyro_lib, t_final + 1, mode=mode)
        time = torch.arange(1, latent.shape[0] + 1)

        x = latent[self.hidden.num_steps :: self.hidden.num_steps]
        state = self.hidden.initial_sample().propagate_from(values=x, time_increment=time)
        obs_dist = self.build_density(state)

        if obs is None:
            pyro_lib.sample("y", obs_dist)
        else:
            pyro_lib.factor("y_log_prob", obs_dist.log_prob(obs).sum(dim=0))

        return latent
