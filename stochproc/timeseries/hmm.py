from typing import Callable, Sequence
from pyro.distributions import Multinomial
import torch

from ..typing import ParameterType
from .stochastic_process import StructuralStochasticProcess


def _parameter_transform(*parameters):
    return parameters[0]
    

def _find_initial_probabilities(probs: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(probs.shape[:-1], device=probs.device).unsqueeze(-2)
    eye = torch.eye(probs.shape[-1], device=probs.device)

    p = torch.cat((eye - probs.transpose(-2, -1), ones), dim=-2)

    transpose = p.transpose(-2, -1)
    inverse = torch.linalg.pinv(transpose @ p)

    return (inverse @ transpose)[..., -1]


class HiddenMarkovModel(StructuralStochasticProcess):
    """
    Implements a discrete `Hidden Markov Model`_. More resources from `Oxford`.

    .. _`Hidden Markov Model`: https://en.wikipedia.org/wiki/Hidden_Markov_model
    .. _`Oxford`: https://oxfordre.com/economics/display/10.1093/acrefore/9780190625979.001.0001/acrefore-9780190625979-e-174;jsessionid=01A52B578F4523C66A4DCCA27ACB1DE7
    """

    def __init__(
        self,
        parameters: Sequence[ParameterType],
        parameter_transform: Callable[[Sequence[torch.Tensor]], Sequence[torch.Tensor]] = _parameter_transform,
    ):
        """
        Internal initializer for :class:`HiddenMarkovModel`.

        Args:
            transition_matrix (ParameterType): transition probabilities.
            initial_probabilities (ParameterType): initial probabilities.
            parameter_transform (Callable[[Sequence[torch.Tensor]], Sequence[torch.Tensor]]): function for transforming parameters. Defaults to returning the first parameter as is.
        """

        transition_matrix = _parameter_transform(*parameters)

        if transition_matrix.shape[-1] != transition_matrix.shape[-2]:
            raise Exception("Transition matrix is not rectangular!")

        if (transition_matrix.sum(dim=-1) != 1.0).all():
            raise Exception("Not a proper transition matrix!")

        self._parameter_transform = parameter_transform

        super().__init__(
            self._hmm_kernel, parameters, self._initial_hmm_kernel, initial_parameters=()
        )
        
    @property
    def initial_probabilities(self) -> torch.Tensor:
        """
        Returns the initial probabilities.
        """

        return _find_initial_probabilities(self._parameter_transform(*self.parameters))

    def expand(self, batch_shape):
        new_parameters = self._expand_parameters(batch_shape)
        return HiddenMarkovModel(new_parameters["parameters"])

    def _initial_hmm_kernel(self, *_):
        return Multinomial(probs=self.initial_probabilities)

    def _hmm_kernel(self, x, *parameters: torch.Tensor):
        probs = self._parameter_transform(*parameters)
        state = x.value.argmax(dim=-1)

        p = probs.take_along_dim(state.reshape(state.shape + torch.Size([1, 1])), dim=-2).squeeze(-2)
        return Multinomial(probs=p)
