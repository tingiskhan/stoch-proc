from typing import Any, Optional, Sequence, Tuple, Union

import torch
from torch.distributions import Distribution


class JointDistribution(Distribution):
    r"""
    Defines an object for combining multiple distributions by assuming independence, i.e. we define:
        .. math::
            p(x_1, x_2, ..., x_n) = p(x_1) \cdot p(x_2) ... \cdot p(x_n)

    Example:
        A basic example can be seen below, where we combine a normal and and exponential distribution:

            >>> from torch.distributions import Normal, Exponential
            >>> import torch
            >>>
            >>> distribution = JointDistribution(Normal(0.0, 1.0), Exponential(1.0))
            >>> y = distribution.sample((1000,))    # should be 1000 x 2
            >>>
            >>> log_prob = distribution.log_prob(y)

    """

    arg_constraints = {}

    def __init__(self, *distributions: Distribution, indices: Sequence[Union[int, slice]] = None, **kwargs):
        """
        Internal initializer for :class:`JointDistribution`.

        Args:
            distributions: Iterable of :class:`pytorch.distributions.Distribution` objects.
            indices: which distribution corresponds to which column in input tensors, inferred if ``None``.
            kwargs: Key-worded arguments passed to base class.
        """

        if any(len(d.event_shape) > 1 for d in distributions):
            raise NotImplementedError("Currently cannot handle matrix valued distributions!")
        
        event_shape = torch.Size([sum(d.event_shape.numel() for d in distributions)])

        batch_shapes = [(d.batch_shape.numel(), d.batch_shape) for d in distributions]
        single_batch_shape = sorted(batch_shapes, key=lambda u: u[0])[-1][1]
        distributions = [d.expand(single_batch_shape) for d in distributions]

        super().__init__(event_shape=event_shape, batch_shape=single_batch_shape, **kwargs)        

        self.distributions = distributions
        self.indices = indices if indices is not None else self.infer_indices(*distributions)

    def expand(self, batch_shape, _instance=None):
        return JointDistribution(*(d.expand(batch_shape) for d in self.distributions), indices=self.indices)

    @property
    def support(self) -> Optional[Any]:
        raise NotImplementedError()

    @property
    def mean(self):
        raise NotImplementedError()

    @property
    def variance(self):
        raise NotImplementedError()

    def cdf(self, value):
        res = 0.0
        for d, m in zip(self.distributions, self.indices):
            res *= d.cdf(value[..., m])

        return res

    def icdf(self, value):
        raise NotImplementedError()

    def enumerate_support(self, expand=True):
        raise NotImplementedError()

    def entropy(self):
        return sum(d.entropy() for d in self.distributions)

    @staticmethod
    def infer_indices(*distributions: Distribution) -> Tuple[Union[int, slice]]:
        """
        Given a sequence of class:`pytorch.distributions.Distribution` objects, this method infers the indices at which
        to slice an input tensor.

        Args:
            distributions: sequence of class:`pytorch.distributions.Distribution` objects.

        Returns:
            A tuple containing indices and/or slices.

        Example:
            >>> from torch.distributions import Normal, Exponential
            >>> import torch
            >>> from stochproc.distributions import JointDistribution
            >>>
            >>> distributions = Normal(0.0, 1.0), Exponential(1.0)
            >>> y = torch.stack([d.sample((1000,)) for d in distributions], dim=-1)
            >>>
            >>> slices = JointDistribution.infer_indices(*distributions)
            >>> log_probs = [d.log_prob(y[..., s]) for d, s in zip(distributions, slices)]

        """

        res = tuple()

        left = 0
        for sub_dist in distributions:
            numel = sub_dist.event_shape.numel()

            if numel > 1:
                indexes = slice(left, left + numel)
            else:
                indexes = left

            res += (indexes,)
            left += numel

        return res

    def log_prob(self, value):
        # TODO: Add check for wrong dimensions
        return sum(d.log_prob(value[..., m]) for d, m in zip(self.distributions, self.indices))

    def rsample(self, sample_shape=torch.Size()):
        res = tuple(
            d.rsample(sample_shape) if len(d.event_shape) > 0 else d.rsample(sample_shape).unsqueeze(-1)
            for d in self.distributions
        )

        return torch.cat(res, dim=-1)
