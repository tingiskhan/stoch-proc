from copy import deepcopy
from typing import Tuple

import pyro
import torch
from torch.nn import Module

from .stochastic_process import StochasticProcess
from ..distributions.prior_module import _SampleParameterMixin


class StateSpaceModel(Module, _SampleParameterMixin):
    r"""
    Class representing a state space model, i.e. a dynamical system given by the pair stochastic processes
    :math:`\{ X_t \}` and :math:`\{ Y_t \}`, where :math:`X_t` is independent from :math:`Y_t`, and :math:`Y_t`
    conditionally independent given :math:`X_t`. See more `here`_.

    .. _`here`: https://en.wikipedia.org/wiki/State-space_representation
    """

    def __init__(self, hidden: StochasticProcess, observable: StochasticProcess):
        """
        Initializes the :class:`StateSpaceModel` class.

        Args:
            hidden: hidden process.
            observable: observable process.
        """

        super().__init__()
        self.hidden = hidden
        self.observable = observable

    def sample_path(self, steps, samples=torch.Size([]), x_s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x_s if x_s is not None else self.hidden.initial_sample(shape=samples)

        hidden = tuple()
        obs = tuple()

        for t in range(1, steps + 1):
            x = self.hidden.propagate(x)
            obs_state = self.observable.propagate(x)

            obs += (obs_state,)
            hidden += (x,)

        return torch.stack([t.values for t in hidden]), torch.stack([t.values for t in obs])

    def copy(self) -> "StateSpaceModel":
        """
        Creates a deep copy of self.
        """

        return deepcopy(self)

    def do_sample_pyro(self, pyro_lib: pyro, obs: torch.Tensor, mode: str = "approximate") -> torch.Tensor:
        """
        Samples the state space model utilizing pyro.

        Args:
            pyro_lib: pyro library.
            obs: the observed data.
            mode: the mode to use for the latent process, see :meth:`StochasticProcess.do_sample_pyro`.

        Returns:
            Returns the latent process.
        """

        latent = self.hidden.do_sample_pyro(pyro_lib, obs.shape[0] + 1, mode=mode)

        time = torch.arange(1, latent.shape[0] + 1)

        x = latent[self.hidden.num_steps :: self.hidden.num_steps]
        state = self.hidden.initial_sample().propagate_from(values=x, time_increment=time)
        obs_dist = self.observable.build_density(state)

        pyro_lib.factor("y_log_prob", obs_dist.log_prob(obs).sum(dim=0))

        return latent

    def sample_params_(self, shape: torch.Size = torch.Size([])):
        self.hidden.sample_params_(shape)
        self.observable.sample_params_(shape)
