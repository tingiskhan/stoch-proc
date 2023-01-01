import torch
from typing import Sequence, Tuple, Callable

from .affine import AffineProcess


# TODO: Skip if-else and do this on instantiation instead
def default_transform(*args):
    if len(args) == 2:
        a, s = args
        return a, torch.zeros_like(s), s

    if len(args) == 3:
        return args

    raise Exception("You sent more parameters than expected, please provide a custom transform for your parameters!")


ParameterTransformer = Callable[[Sequence[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


class LinearModel(AffineProcess):
    r"""
    Implements a linear process, i.e. in which the distribution at :math:`t + 1` is given by a linear combination of
    the states at :math:`t`, i.e.
        .. math::
            X_{t+1} = b + A \cdot X_t + \sigma \epsilon_{t+1}, \newline
            X_0 \sim p(x_0)

    where :math:`X_t, b, \sigma \in \mathbb{R}^n`, :math:`\{ \epsilon_t \}` some distribution, and
    :math:`A \in \mathbb{R}^{n \times n}`.
    """

    def __init__(
        self,
        parameters,
        increment_distribution,
        initial_kernel,
        initial_parameters=None,
        parameter_transform: ParameterTransformer = default_transform,
    ):
        """
        Initializes the :class:`LinearModel` class.

        Args:
            a: ``A`` matrix in the class docs.
            sigma: ``sigma`` vector in the class docs.
            b: ``b`` vector in the class docs.
            parameter_transform: function for transforming parameters into expected. Defaults to assuming that the
                order of the parameters are ``a, b, s``.
        """

        super().__init__(
            self._mean_scale,
            parameters=parameters,
            increment_distribution=increment_distribution,
            initial_kernel=initial_kernel,
            initial_parameters=initial_parameters,
        )

        self._parameter_transform = parameter_transform

        assert (
            len(self._parameter_transform(*self.parameters)) == 3
        ), "Your parameter transform does not return a triple!"

    def transformed_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Returns the triple :math:`\{ \alpha, \beta, \sigma }`.
        """

        return tuple(self._parameter_transform(*self.parameters))

    def _mean_scale(self, x, *args):
        a, b, s = self._parameter_transform(*args)

        if self.n_dim > 0:
            res = (b.unsqueeze(-1) + a @ x.value.unsqueeze(-1)).squeeze(-1)
        else:
            res = b + a * x.value

        return res, s
