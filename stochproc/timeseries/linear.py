import torch

from .affine import AffineProcess
from ..typing import ParameterType


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

    def __init__(self, a: ParameterType, sigma: ParameterType, b: ParameterType = None, **kwargs):
        """
        Initializes the ``LinearModel`` class.

        Args:
            a: ``A`` matrix in the class docs.
            sigma: ``sigma`` vector in the class docs.
            b: ``b`` vector in the class docs.
        """

        if b is None:
            b = torch.tensor(0.0)

        super(LinearModel, self).__init__(self._mean_scale, parameters=(a, b, sigma), **kwargs)

    def _mean_scale(self, x, a, b, s):
        if x.event_shape.numel() > 1:
            res = b + a.matmul(x.values.unsqueeze(-1)).squeeze(-1)
        else:
            res = b + a * x.values

        return res, s
