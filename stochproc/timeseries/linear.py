import torch
from .affine import AffineProcess
from ..typing import ParameterType


def _f0d(x, a, b):
    return b + a * x.values


def _f1d(x, a, b):
    return b + a.matmul(x.values.unsqueeze(-1)).squeeze(-1)


_mapping = {0: _f0d, 1: _f1d}


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
        self, a: ParameterType, sigma: ParameterType, increment_dist, b: ParameterType = None, **kwargs
    ):
        """
        Initializes the ``LinearModel`` class.

        Args:
            a: ``A`` matrix in the class docs.
            sigma: ``sigma`` vector in the class docs.
            b: ``b`` vector in the class docs.
        """

        initial_dist = kwargs.pop("initial_dist")
        dimension = len(initial_dist.shape)

        if b is None:
            b = torch.zeros(dimension) if dimension > 0 else 0.0

        def _mean_scale(x, a_, b_, s_):
            return _mapping[dimension](x, a_, b_), s_

        super(LinearModel, self).__init__(
            _mean_scale, parameters=(a, b, sigma), initial_dist=initial_dist, increment_dist=increment_dist,
            **kwargs
        )
