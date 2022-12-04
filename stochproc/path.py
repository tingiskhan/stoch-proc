from dataclasses import dataclass
from typing import Generic, Tuple

from .typing import TArray
from .state import TimeseriesState


@dataclass(frozen=True)
class CombinedPath(Generic[TArray]):
    time_indexes: TArray
    path: TArray


class StochasticProcessPath(Generic[TArray]):
    """
    Base container object for storing sampled paths from a stochastic process.
    """

    def __init__(self, states: TimeseriesState[TArray]):
        """
        Internal initializer for :class:`ProcessPath`.

        Args:
            states (TimeseriesState): timeseries states to combine.
        """

        self.path = sorted(states, key=lambda u: u.time_index)
    
    def get_path(self) -> CombinedPath:
        """
        Returns the time indexes together with the sampled path.

        Returns:
            CombinedPath: time indexes together with sampled path.
        """

        raise NotImplementedError()

    