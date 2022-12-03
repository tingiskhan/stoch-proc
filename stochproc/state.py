from typing import TypeVar, Generic

from .typing import ShapeLike
from .utils import lazy_property


TArray = TypeVar("TArray")


class TimeseriesState(dict, Generic[TArray]):
    """
    State object for timeseries objects.
    """

    def __init__(
        self,
        time_index: TArray,
        values: TArray,
        event_shape: ShapeLike,
    ):
        """
        Initializes the :class:`TimeseriesState` class.

        Args:
            time_index: time index of the state.
            values: values of the state. Can be a lazy evaluated tensor as well.
            event_shape: event dimension.
        """

        super().__init__()

        self["_time_index"] = time_index
        self["_values"] = values
        self["_event_shape"] = event_shape

    @property
    def time_index(self) -> TArray:
        return self["_time_index"]

    # TODO: Fix lazy evaluation
    @lazy_property("_values")
    def values(self) -> TArray:
        return self["_values"]
    
    @property
    def event_shape(self) -> ShapeLike:
        return self["_event_shape"]

    def __repr__(self):
        return f"TimeseriesState at t={self.time_index} containing: {self.values.__repr__()}"
    
    def propagate_from(self, values: TArray, time_increment: int = 1) -> "TimeseriesState[TArray]":
        """
        Returns a new instance of :class:`TimeseriesState` with `values`` and ``time_index`` given by
        ``.time_index + time_increment``.

        Args:
            values: see ``__init__``.
            time_increment: how much to increase ``.time_index`` with for new state.
        """
        

        return TimeseriesState(time_index=self.time_index + time_increment, values=values, event_shape=self.event_shape)
