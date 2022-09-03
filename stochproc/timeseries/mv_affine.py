from pyro.distributions import transforms as t, TransformedDistribution

from .affine import AffineProcess


class MultivariateAffineProcess(AffineProcess):
    r"""
    Defines a process similar to :class:`AffineProcess` but in which the scale parameter is a lower triangular matrix.
    """

    def _define_transdist(self, x):
        loc, scale = self.mean_scale(x)

        return TransformedDistribution(self.increment_dist(), t.LowerCholeskyAffine(loc, scale))

    def mean_scale(self, x, parameters=None):
        return self.mean_scale_fun(x, *(parameters or self.functional_parameters()))
