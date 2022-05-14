from .base import _DistributionModule
from stochproc.distributions.prior_module import _HasPriorsModule
from .typing import DistributionOrBuilder
from ..typing import ParameterType, NamedParameter


class DistributionModule(_DistributionModule, _HasPriorsModule):
    """
    Implements a wrapper around ``pytorch.distributions.Distribution`` objects. It inherits from ``pytorch.nn.Module``
    in order to utilize all of the associated methods and attributes. One such is e.g. moving tensors between different
    devices.

    Example:
        >>> from torch.distributions import Normal
        >>> from stochproc.distributions import DistributionModule
        >>>
        >>> wrapped_normal_cpu = DistributionModule(Normal, loc=0.0, scale=1.0)
        >>> wrapped_normal_cuda = wrapped_normal_cpu.cuda()
        >>>
        >>> cpu_samples = wrapped_normal_cpu.build_distribution().sample((1000,)) # device cpu
        >>> cuda_samples = wrapped_normal_cuda.build_distribution().sample((1000,)) # device cuda
    """

    def __init__(
        self,
        base_dist: DistributionOrBuilder,
        reinterpreted_batch_ndims=None,
        **parameters: ParameterType
    ):
        """
        Initializes the ``DistributionModule`` class.

        Args:
            base_dist: See the ``distribution`` of ``pyfilter.distributions.Prior``.
            parameters: See ``parameters`` of ``pyfilter.distributions.Prior``. With the addition that we can pass
                ``pyfilter.distributions.Prior`` objects as parameters.

        Example:
            In this example we'll construct a distribution wrapper around a normal distribution where the location is a
            prior:
                >>> from torch.distributions import Normal
                >>> import torch
                >>> from stochproc.distributions import DistributionModule, Prior
                >>>
                >>> loc_prior = Prior(Normal, loc=0.0, scale=1.0)
                >>> wrapped_normal_with_prior = DistributionModule(Normal, loc=loc_prior, scale=1.0)
                >>>
                >>> size = torch.Size([1000])
                >>> wrapped_normal_with_prior.sample_params_(size)
                >>> samples = wrapped_normal_with_prior.build_distribution().sample(size) # should be 1000 x 1000
        """

        super(DistributionModule, self).__init__(
            base_dist=base_dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims
        )

        for k, v in parameters.items():
            if isinstance(v, NamedParameter) and (v.name != k):
                raise Exception(f"Key != name of parameter: {k} != {v.name}")

            self._register_parameter_or_prior(k, v)

    def _get_parameters(self):
        return self.parameters_and_buffers()
