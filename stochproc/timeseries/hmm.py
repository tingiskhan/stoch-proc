from typing import Callable, Sequence
from pyro.distributions import Multinomial
import torch

from ..typing import ParameterType
from .stochastic_process import StructuralStochasticProcess


def _parameter_transform(*parameters):
    return parameters[0]


def _initial_kernel(init_probabilities):
    return Multinomial(probs=init_probabilities)


class HiddenMarkovModel(StructuralStochasticProcess):
    r"""
    Implements a discrete `Hidden Markov Model`_.

    .. _`Hidden Markov Model`: https://en.wikipedia.org/wiki/Hidden_Markov_model
    """

    def __init__(
        self,
        transition_matrix: ParameterType,
        initial_probabilities: ParameterType,
        parameter_transform: Callable[[Sequence[torch.Tensor]], Sequence[torch.Tensor]] = _parameter_transform,
    ):
        """
        Internal initializer for :class:`HiddenMarkovModel`.

        Args:
            transition_matrix (ParameterType): transition probabilities.
            initial_probabilities (ParameterType): initial probabilities.
            parameter_transform (ParameterType): function for transforming parameters.
        """

        if transition_matrix.shape[-1] != transition_matrix.shape[-2]:
            raise Exception("Transition matrix is not rectangular!")

        if transition_matrix.shape[-1] != initial_probabilities.shape[-1]:
            raise Exception("Initial probabilities is not congruent with transition!")

        super().__init__(
            self._hmm_kernel, (transition_matrix,), _initial_kernel, initial_parameters=(initial_probabilities,)
        )
        self._parameter_transform = parameter_transform

    def expand(self, batch_shape):
        new_parameters = self._expand_parameters(batch_shape)
        return HiddenMarkovModel(*new_parameters["parameters"], *new_parameters["initial_parameters"])

    def _hmm_kernel(self, x, *parameters: torch.Tensor):
        probs = self._parameter_transform(*parameters)
        state = x.value.argmax(dim=-1)

        p = probs.take_along_dim(state.reshape(state.shape + torch.Size([1, 1])), dim=-2).squeeze(-2)
        return Multinomial(probs=p)
