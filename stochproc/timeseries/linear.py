import torch
from .affine import AffineProcess
from ..typing import ParameterType
from ..utils import broadcast_all


def _f0d(x, a, b):
    return b + a * x.values


def _f1d(x, a, b):
    return _f2d(x, a, b)


def _f2d(x, a, b):
    return b + a.matmul(x.values.unsqueeze(-1)).squeeze(-1)


_mapping = {0: _f0d, 1: _f1d, 2: _f2d}


class LinearModel(AffineProcess):
    """
    Implements a linear process, i.e. in which the distribution at :math:`t + 1` is given by a linear combination of
    the states at :math:`t`, i.e.
        .. math::
            X_{t+1} = b + A \\cdot X_t + \\sigma \\epsilon_{t+1}, \n
            X_0 \\sim p(x_0)

    where :math:`X_t, b, \\sigma \\in \\mathbb{R}^n`, :math:`\\{\\epsilon_t\\}` some distribution, and
    :math:`A \\in \\mathbb{R}^{n \\times n}`.
    """

    def __init__(self, a: ParameterType, sigma: ParameterType, increment_dist, b: ParameterType = None, **kwargs):
        """
        Initializes the ``LinearModel`` class.

        Args:
            a: The ``A`` matrix in the class docs.
            sigma: The ``sigma`` vector in the class docs.
            b: The ``b`` vector in the class docs.
        """

        a = broadcast_all(a)[0]
        sigma = broadcast_all(sigma)[0]

        if b is None:
            b = torch.zeros(sigma.shape)

        dimension = len(a.shape)
        params = (a, b, sigma)

        initial_dist = kwargs.pop("initial_dist", None)

        def _mean_scale(x, a_, b_, s_):
            return _mapping[dimension](x, a_, b_), s_

        super(LinearModel, self).__init__(
            _mean_scale, parameters=params, initial_dist=initial_dist or increment_dist, increment_dist=increment_dist,
            **kwargs
        )
