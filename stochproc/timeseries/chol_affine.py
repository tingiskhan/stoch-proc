from pyro.distributions import transforms as t, TransformedDistribution

from .affine import AffineProcess


# TODO: Not too sure that this is should be a subclass of `AffineProcess`...
class LowerCholeskyAffineProcess(AffineProcess):
    r"""
    Defines a process similar to :class:`AffineProcess` but in which the scale parameter is a lower triangular matrix.

    In more detail we have a similar functional expression given by
        .. math::
            X_{t+1} = f(X_t, \theta) + g(X_t, \theta) \cdot W_{t+1},

    where :math:`X \in \mathbb{R}^n` and :math:`n \geq 2`, :math:`\theta \in \Theta \subset \mathbb{R}^m`,
    :math:`f : \: \mathbb{R}^n \times \Theta \rightarrow \mathbb{R}^n`,
    :math:`g : \: \mathbb{R}^n \times \Theta \rightarrow \mathbb{R}^{n \times n}` and is a lower triangular matrix,
    and :math:`W_t` denotes a random variable with arbitrary density (from which we can sample).
    """

    def __init__(self, mean_scale, parameters, initial_dist, increment_dist, **kwargs):
        """
        Initializes the :class:`MultivariateAffineProcess`.

        Args:
            mean_scale: see base.
            parameters. see base.
            initial_dist: see base.
            increment_dist: see base.
            kwargs: see base.
        """

        super().__init__(mean_scale, parameters, initial_dist, increment_dist, **kwargs)
        assert self.n_dim >= 1, "This process only covers multi-dimensional processes!"

    def build_density(self, x):
        loc, scale = self.mean_scale(x)

        return TransformedDistribution(self.increment_dist(), t.LowerCholeskyAffine(loc, scale))

    def mean_scale(self, x, parameters=None):
        return self.mean_scale_fun(x, *(parameters or self.functional_parameters()))

    def add_sub_process(self, sub_process):
        raise NotImplementedError("Currently does not support adding sub processes!")
