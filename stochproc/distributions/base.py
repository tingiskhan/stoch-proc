from pyro.distributions import Distribution, TorchDistribution
import torch
from typing import Dict, OrderedDict as tOrderedDict, Any
from collections import OrderedDict
from .typing import DistributionOrBuilder
from abc import ABC


class _DistributionModule(torch.nn.Module, ABC):
    """
    Abstract base class for wrapping :class:`torch.distributions.Distribution` objects in a :class:`torch.nn.Module`.
    """

    def __init__(self, base_dist: DistributionOrBuilder):
        """
        Initializes the :class:`_DistributionModule` class.

        Args:
            base_dist: The base distribution, or distribution builder.
        """

        super(_DistributionModule, self).__init__()

        self.base_dist = base_dist
        self._funs: tOrderedDict[str, Any] = OrderedDict([])

    def build_distribution(self) -> Distribution:
        """
        Constructs the distribution.
        """

        return self.__call__()

    def forward(self):
        dist = self.base_dist(**self._get_parameters())

        for f, v in self._funs.items():
            dist = getattr(dist, f)(v)

        return dist

    def _get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Gets the parameters to initialize the distribution with, overridden by derived classes.
        """

        raise NotImplementedError()

    @property
    def shape(self) -> torch.Size:
        """
        Returns the event shape of the distribution.
        """

        return self.build_distribution().event_shape

    def expand(self, batch_shape) -> "_DistributionModule":
        self._funs["expand"] = batch_shape
        return self

    def to_event(self, reinterpreted_batch_ndims=None) -> "_DistributionModule":
        self._funs["to_event"] = reinterpreted_batch_ndims
        return self
