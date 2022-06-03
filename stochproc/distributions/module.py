import warnings
from collections import OrderedDict

import torch.nn
from torch.nn import ParameterDict

from .base import _DistributionModule
from .typing import DistributionOrBuilder
from ..container import BufferDict
from ..typing import ParameterType


class DistributionModule(_DistributionModule):
    """
    See :class:`_DistributionModule`.

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

    def __init__(self, base_dist: DistributionOrBuilder, reinterpreted_batch_ndims=None, **parameters: ParameterType):
        """
        Initializes the :class:`DistributionModule` class.

        Args:
            base_dist: See the ``distribution`` of :class:`stochproc.distributions.Prior`.
            parameters: See ``parameters`` of :class:`stochproc.distributions.Prior`. With the addition that we can pass
                :class:`stochproc.distributions.Prior` objects as parameters.

        """

        super(DistributionModule, self).__init__(
            base_dist=base_dist, reinterpreted_batch_ndims=reinterpreted_batch_ndims
        )

        # TODO: This is duplicate code, could perhaps move to one and the same
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            self.parameter_dict = ParameterDict()
            self.buffer_dict = BufferDict()

        self._helper_parameters = OrderedDict([])

        for k, p in parameters.items():
            if isinstance(p, torch.nn.Parameter):
                self.parameter_dict[k] = v = p
            else:
                self.buffer_dict[k] = v = p if isinstance(p, torch.Tensor) else torch.tensor(p)

            self._helper_parameters[k] = v

    def _apply(self, fn):
        super(DistributionModule, self)._apply(fn)

        res = OrderedDict([])
        res.update(self.parameter_dict)
        res.update(self.buffer_dict)

        self._helper_parameters = res

        return self

    def _get_parameters(self):
        return self._helper_parameters
