from .affine import AffineProcess


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
    ):
        """
        Internal initializer for :class:`LinearModel`.

        Args:
            parameters: see :class:`AffineProcess`.
            increment_distribution: see :class:`AffineProcess`.
            initial_kernel: see :class:`AffineProcess`.
            initial_parameters: see :class:`AffineProcess`.            
        """

        assert len(parameters) == 3, "Must pass three parameters!"

        super().__init__(
            self._mean_scale_0d if increment_distribution.event_shape.numel() == 1 else self._mean_scale_md,
            parameters=parameters,
            increment_distribution=increment_distribution,
            initial_kernel=initial_kernel,
            initial_parameters=initial_parameters,
        )

    def _mean_scale_0d(self, x, a, b, s):
        return b + a * x.value, s

    def _mean_scale_md(self, x, a, b, s):
        values = x.value.unsqueeze(-1)

        # This is a non-SOLID "hack" to accomodate for event shapes differing between model and state
        if not x.event_shape:
            values = values.unsqueeze(-1)

        res = (b.unsqueeze(-1) + a @ values).squeeze(-1)

        return res, s

    def expand(self, batch_shape):
        new_parameters = self._expand_parameters(batch_shape)
        return LinearModel(
            new_parameters["parameters"],
            self.increment_distribution,
            self._initial_kernel,
            new_parameters["initial_parameters"],
        )
