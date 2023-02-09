import torch
from pyro.distributions import Categorical

from ..typing import ParameterType
from .stochastic_process import StructuralStochasticProcess


def _find_initial_probabilities(probs: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(probs.shape[:-1], device=probs.device).unsqueeze(-2)
    eye = torch.eye(probs.shape[-1], device=probs.device)

    p = torch.cat((eye - probs.transpose(-2, -1), ones), dim=-2)

    transpose = p.transpose(-2, -1)
    inverse = torch.linalg.pinv(transpose @ p)

    return (inverse @ transpose)[..., -1]


def _initial_kernel(probs):
    return Categorical(probs=probs)


def _transition_kernel(x, probs):
    state = x.value.reshape(x.value.shape + torch.Size([1, 1]))

    p = probs.broadcast_to(state.shape[:-2] + probs.shape[-2:]).take_along_dim(state, dim=-2).squeeze(-2)
    return Categorical(probs=p)


class HiddenMarkovModel(StructuralStochasticProcess):
    """
    Implements a discrete `Hidden Markov Model`_. More resources from `Oxford`.

    .. _`Hidden Markov Model`: https://en.wikipedia.org/wiki/Hidden_Markov_model
    .. _`Oxford`: https://oxfordre.com/economics/display/10.1093/acrefore/9780190625979.001.0001/acrefore-9780190625979-e-174;jsessionid=01A52B578F4523C66A4DCCA27ACB1DE7
    """

    def __init__(
        self,
        transition_matrix: ParameterType,
    ):
        """
        Internal initializer for :class:`HiddenMarkovModel`.

        Args:
            parameters (ParameterType): transition probabilities.
        """

        if transition_matrix.shape[-1] != transition_matrix.shape[-2]:
            raise Exception("Transition matrix is not rectangular!")

        if (transition_matrix.sum(dim=-1) != 1.0).all():
            raise Exception("Not a proper transition matrix!")

        initial_probabilities = _find_initial_probabilities(transition_matrix)

        super().__init__(
            _transition_kernel,
            (transition_matrix,),
            _initial_kernel,
            initial_parameters=(initial_probabilities,),
        )

    def expand(self, batch_shape):
        new_parameters = self._expand_parameters(batch_shape)
        new = self._get_checked_instance(HiddenMarkovModel)
        super(HiddenMarkovModel, new).__init__(
            _transition_kernel, new_parameters["parameters"], _initial_kernel, new_parameters["initial_parameters"]
        )

        return new
