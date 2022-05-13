import torch
from typing import Union
from torch.distributions import Distribution


class TimeseriesState(dict):
    """
    State object for ``StochasticProcess``.
    """

    def __init__(
        self, time_index: Union[float, torch.Tensor], distribution: Distribution = None, values: torch.Tensor = None,
        exogenous: torch.Tensor = None
    ):
        """
        Initializes the ``TimeseriesState`` class.

        Args:
            time_index: The time index of the state.
            distribution: Optional parameter, the distribution of the state at ``time_index``.
            values: Optional parameter, the values of the state at ``time_index``. If ``None`` and passing
                ``distribution`` values will be sampled from ``distribution`` when accessing ``.values`` attribute.
        """

        super().__init__()

        self.distribution: Distribution = distribution
        self.time_index: torch.Tensor = time_index if isinstance(time_index, torch.Tensor) else torch.tensor(time_index)
        self.exogenous: torch.Tensor = exogenous

        self._values: torch.Tensor = values

    @property
    def values(self) -> torch.Tensor:
        """
        The values of the state.
        """

        if self._values is not None:
            return self._values

        self._values = self.distribution.sample()
        return self._values

    @values.setter
    def values(self, x):
        self._values = x

    def copy(self, distribution: Distribution = None, values: torch.Tensor = None) -> "TimeseriesState":
        """
        Copies ``self`` with specified ``distribution`` and ``values`` but with ``time_index`` of current instance.

        Args:
            distribution: See ``__init__``.
            values: See ``__init__``.
        """

        res = self.propagate_from(distribution, values, time_increment=0.0)
        res.add_exog(self.exogenous)

        return res

    def propagate_from(self, distribution: Distribution = None, values: torch.Tensor = None, time_increment=1.0):
        """
        Returns a new instance of ``NewState`` with ``distribution`` and ``values``, and ``time_index`` given by
        ``.time_index + time_increment``.

        Args:
            distribution: See ``__init__``.
            values: See ``__init__``.
            time_increment: Optional, specifies how much to increase ``.time_index`` with for new state.
        """

        return TimeseriesState(self.time_index + time_increment, distribution, values)

    def add_exog(self, x: torch.Tensor):
        """
        Adds an exogenous variable to the state.

        Args:
            x: The exogenous variable.
        """

        self.exogenous = x

    def __repr__(self):
        return f"TimeseriesState at t={self.time_index} containing: {self.values}"
