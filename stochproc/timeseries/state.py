import torch
from typing import Union, Callable


LazyTensor = Union[torch.Tensor, Callable[[], torch.Tensor]]


class TimeseriesState(dict):
    """
    State object for ``StochasticProcess``.
    """

    def __init__(
        self, time_index: Union[float, torch.Tensor], values: LazyTensor, exogenous: torch.Tensor = None
    ):
        """
        Initializes the ``TimeseriesState`` class.

        Args:
            time_index: The time index of the state.
            values: The values of the state. Can be a lazy evaluated tensor as well.
        """

        super().__init__()

        self.time_index: torch.Tensor = time_index if isinstance(time_index, torch.Tensor) else torch.tensor(time_index)
        self.exogenous: torch.Tensor = exogenous

        self._values = values

    @property
    def values(self) -> torch.Tensor:
        """
        The values of the state.
        """

        if callable(self._values):
            self._values = self._values()

        return self._values

    @values.setter
    def values(self, x):
        self._values = x

    def copy(self, values: torch.Tensor) -> "TimeseriesState":
        """
        Copies ``self`` with specified ``distribution`` and ``values`` but with ``time_index`` of current instance.

        Args:
            values: See ``__init__``.
        """

        res = self.propagate_from(values, time_increment=0.0)
        res.add_exog(self.exogenous)

        return res

    def propagate_from(self, values: torch.Tensor, time_increment=1.0):
        """
        Returns a new instance of ``NewState`` with ``distribution`` and ``values``, and ``time_index`` given by
        ``.time_index + time_increment``.

        Args:
            values: See ``__init__``.
            time_increment: Optional, specifies how much to increase ``.time_index`` with for new state.
        """

        return TimeseriesState(self.time_index + time_increment, values)

    def add_exog(self, x: torch.Tensor):
        """
        Adds an exogenous variable to the state.

        Args:
            x: The exogenous variable.
        """

        self.exogenous = x

    def __repr__(self):
        return f"TimeseriesState at t={self.time_index} containing: {self.values.__repr__()}"
