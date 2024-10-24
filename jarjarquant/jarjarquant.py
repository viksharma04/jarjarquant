import pandas as pd
from .data_gatherer import DataGatherer
from .feature_engineer import FeatureEngineer
from .labeller import Labeller


class Jarjarquant:
    """
    Jarjarquant integrates data gathering, labeling, and feature engineering for financial time series.
    """

    def __init__(self, series=None):
        self.data_gatherer = DataGatherer()
        if series is None:
            self._series = self.data_gatherer.generate_random_normal()
        else:
            self._series = series
        self.labeller = Labeller(self._series)
        self.feature_engineer = FeatureEngineer(pd.DataFrame(self._series))

    @classmethod
    def from_random_normal(cls, loc: float = 0.005, volatility: float = 0.05, periods: int = 252, **kwargs):
        """
        Create a random price series using returns from a normal distribution.

        Args:
            loc (float): Mean return. Defaults to 0.005.
            volatility (float): Period volatility. Defaults to 0.05.
            periods (int): Number of data points. Defaults to 252.

        Returns:
            Jarjarquant: Instance of Jarjarquant with generated series.
        """
        data_gatherer = DataGatherer()
        series = data_gatherer.generate_random_normal(
            loc=loc, volatility=volatility, periods=periods, **kwargs)
        return cls(series)

    @classmethod
    def from_yf_ticker(cls, ticker: str = "SPY", **kwargs):
        """
        Initialize from a Yahoo Finance ticker.

        Args:
            ticker (str): Ticker symbol. Defaults to "SPY".

        Returns:
            Jarjarquant: Instance of Jarjarquant with data from the ticker.
        """
        data_gatherer = DataGatherer()  # or access via composition
        series = data_gatherer.get_yf_ticker(ticker, **kwargs)
        return cls(series)

    @property
    def series(self):
        return self._series

    @series.setter
    def series(self, series):
        if not isinstance(series, pd.Series):
            raise ValueError("Must be a pandas Series!")
        self._series = series

    @series.deleter
    def series(self):
        del self._series

    # Add any additional methods or functionality here
