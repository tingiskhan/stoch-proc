from copy import deepcopy
from functools import wraps
from typing import Tuple

import pyro
import torch
from torch.nn import Module

from .stochastic_process import StochasticProcess
from ..distributions.prior_module import UpdateParametersMixin, _HasPriorsModule


def _check_has_priors_wrapper(f):
    @wraps(f)
    def _wrapper(ssm: "StateSpaceModel", *args, **kwargs):
        if not ssm._has_priors:
            raise Exception(f"No module is subclassed by {_HasPriorsModule.__name__}")

        return f(ssm, *args, **kwargs)

    return _wrapper


# TODO: Add methods ``concat_parameters`` and ``update_parameters_from_tensor``
class StateSpaceModel(Module, UpdateParametersMixin):
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

        self._has_priors = issubclass(self.hidden.__class__, _HasPriorsModule) or issubclass(self.observable.__class__, _HasPriorsModule)

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

    def do_sample_pyro(self, pyro_lib: pyro, obs: torch.Tensor) -> torch.Tensor:
        """
        Samples the state space model utilizing pyro.

        Args:
            pyro_lib: pyro library.
            obs: the observed data.

        Returns:
            Returns the latent process.
        """

        latent = self.hidden.do_sample_pyro(pyro_lib, obs.shape[0] + 1, use_full=True)

        time = torch.arange(1, latent.shape[0] + 1)
        state = self.hidden.initial_sample().propagate_from(values=latent[1::self.hidden.num_steps], time_increment=time)
        obs_dist = self.observable.build_density(state)

        pyro_lib.factor("obs_prob", obs_dist.log_prob(obs).sum())

        return latent
