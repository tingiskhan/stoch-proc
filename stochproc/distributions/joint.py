from functools import partial
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from torch.distributions import Distribution


def _unsqueezer(x: torch.Tensor, do_unsqueeze: bool) -> torch.Tensor:
    if not do_unsqueeze:
        return x

    return x.unsqueeze(-1)


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
        single_batch_shape = max(d.batch_shape for d in distributions)
        
        super().__init__(event_shape=event_shape, batch_shape=single_batch_shape, **kwargs)

        self.distributions = [d.expand(single_batch_shape) for d in distributions]
        self.indices = indices if indices is not None else self.infer_indices(*self.distributions)
        self.has_rsample = all(d.has_rsample for d in self.distributions)

        self._unsqueezers = [partial(_unsqueezer, do_unsqueeze=d.event_shape.numel() == 1) for d in self.distributions]

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(JointDistribution)
        super(JointDistribution, new).__init__(batch_shape, self.event_shape, self._validate_args)
        
        new.distributions = [d.expand(batch_shape) for d in self.distributions]
        new.indices = self.indices
        new.has_rsample = self.has_rsample
        new._unsqueezers = self._unsqueezers

        return new

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

    # TODO: Fix s.t. we use wrapper for unsqueezer
    def rsample(self, sample_shape=torch.Size()):
        res = tuple(
            unsq(d.rsample(sample_shape)) for d, unsq in zip(self.distributions, self._unsqueezers)
        )

        return torch.cat(res, dim=-1)

    def sample(self, sample_shape=torch.Size()):
        res = tuple(
            unsq(d.sample(sample_shape)) for d, unsq in zip(self.distributions, self._unsqueezers)
        )

        return torch.cat(res, dim=-1)
