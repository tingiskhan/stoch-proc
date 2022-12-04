from dataclasses import dataclass
from typing import Generic, Tuple

from .typing import TArray
from .state import _TimeseriesState


@dataclass(frozen=True)
class ArrayPath(Generic[TArray]):
    time_indexes: TArray
    path: TArray


class _StochasticProcessPath(Generic[TArray]):
    """
    Base container object for storing sampled paths from a stochastic process.
    """

    def __init__(self, states: _TimeseriesState[TArray]):
        """
        Internal initializer for :class:`ProcessPath`.

        Args:
            states (TimeseriesState): timeseries states to combine.
        """

        self.path = tuple(sorted(states, key=lambda u: u.time_index))
    
    def get_path(self) -> ArrayPath[TArray]:
        """
        Returns the time indexes together with the sampled path.

        Returns:
            CombinedPath: time indexes together with sampled path.
        """

        raise NotImplementedError()
