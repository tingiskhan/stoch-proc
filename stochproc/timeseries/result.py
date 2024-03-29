from typing import Tuple

import torch

from .state import TimeseriesState


class TimeseriesPath(object):
    """
    Implements a container for storing the path of sampled timeseries states.
    """

    def __init__(self, *states: TimeseriesState):
        """
        Internal initializer for :class:`TimeseriesPath` object.

        Args:
            *states: iterable of timeseries states.
        """

        self.path = sorted(states, key=lambda u: u.time_index)
        self.time_indexes = torch.stack([s.time_index for s in self.path], dim=0)

    def get_path(self) -> torch.Tensor:
        """
        Returns the collection of samples as :class:`torch.Tensor`.
        """

        return torch.stack([s.value for s in self.path], dim=0)

    def __iter__(self):
        return (s for s in self.path)


class StateSpacePath(TimeseriesPath):
    """
    Implements a container for storing the path of sampled state space model.
    """

    def get_path(self):
        raise NotImplementedError(f"Not available on '{self.__class__.__name__}'")

    def get_paths(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the collection of samples for both the latent and observable process.
        """

        x = torch.stack([s["x"].value for s in self.path], dim=0)
        y = torch.stack([s["y"].value for s in self.path], dim=0)

        return x, y
