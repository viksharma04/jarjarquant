"""Main module for jarjarquant package."""

from typing import Dict

import pandas as pd

from jarjarquant.data_gatherer import DataGatherer
from jarjarquant.indicators import (
    IndicatorType,
    get_indicator_class,
    get_indicator_parameters,
    list_available_indicators,
)

from .data_analyst import DataAnalyst
from .feature_engineer import FeatureEngineer
from .feature_evaluator import FeatureEvaluator
from .labeller import Labeller


class Jarjarquant(Labeller):
    """
    Jarjarquant integrates data gathering, labeling, and feature engineering for financial time series.
    """

    def __init__(self):
        """_summary_

        Args:
            ohlcv_df (pd.DataFrame): _description_
            data_source (Optional[str]): _description_

        Raises:
            ValueError: _description_
        """
        self._df = pd.DataFrame()

        # Super init
        super().__init__(self._df)
        self.data_gatherer = DataGatherer()
        self.feature_engineer = FeatureEngineer()
        self.data_analyst = DataAnalyst()
        self.feature_evaluator = FeatureEvaluator()

    @classmethod
    def from_random_normal(
        cls, loc: float = 0.005, volatility: float = 0.05, periods: int = 252, **kwargs
    ):
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
            loc=loc, volatility=volatility, periods=periods, **kwargs
        )
        return cls(series, data_source=None)

    @classmethod
    def from_random_sample(
        cls, num_tickers_to_sample: int = 1, years_in_sample: int = 10, **kwargs
    ):
        """
        Create a random price series using a random sample of tickers.

        Args:
            num_tickers_to_sample (int): Number of tickers to sample. Defaults to 1.

        Returns:
            Jarjarquant: Instance of Jarjarquant with generated series.
        """
        data_gatherer = DataGatherer()
        samples = data_gatherer.get_random_price_samples_tws(
            num_tickers_to_sample=num_tickers_to_sample,
            years_in_sample=years_in_sample,
            **kwargs,
        )
        if not samples:
            raise ValueError(
                "No price samples were returned. Please check the data source."
            )
        return cls(samples[0], data_source=None)

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
        try:
            series = data_gatherer.get_yf_ticker(ticker, **kwargs)
        except Exception as e:
            raise ValueError(
                f"Failed to fetch data for ticker '{ticker}'. Error: {e}"
            ) from e
        return cls(series, data_source=None)

    @property
    def df(self):
        """
        Returns the DataFrame stored in the instance.

        Returns:
            pandas.DataFrame: The DataFrame stored in the instance.
        """
        return self._df

    @df.setter
    def df(self, df):
        """
        Sets the DataFrame for the instance.

        Parameters:
        df (pd.DataFrame): The DataFrame to be set. Must be an instance of pandas DataFrame.

        Raises:
        ValueError: If the provided df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Must be a pandas DataFrame!")
        self._df = df

    @df.deleter
    def df(self):
        """
        Deletes the attribute `_df` from the instance.

        This method removes the `_df` attribute from the instance, effectively
        deleting any data stored in it. Use this method with caution as it will
        permanently remove the attribute and its contents.

        Raises:
            AttributeError: If the `_df` attribute does not exist.
        """
        del self._df

    def list_indicators(
        self, verbose: bool = False
    ) -> list[IndicatorType] | Dict[IndicatorType, dict]:
        """
        Discover all classes in jarjarquant.indicators and
        return their __init__ signature parameters (excluding self & ohlcv_df)
        as a JSON string.
        """

        return list_available_indicators(verbose=verbose)

    def get_indicator_details(self, indicator: IndicatorType) -> dict[str, dict]:
        return get_indicator_parameters(indicator_type=indicator)

    # Indicator methods
    def add_indicator(
        self, indicator_type: IndicatorType, column_name: str, *args, **kwargs
    ):
        """
        Generic method to add an indicator column to the DataFrame using the indicator registry.

        Parameters:
            indicator_type (IndicatorType): The type of indicator to add from the registry.
            column_name (str): The name of the column to add.
            *args: Positional arguments to pass to the indicator constructor.
            **kwargs: Keyword arguments to pass to the indicator constructor.

        Returns:
            None: The method modifies the DataFrame in place by adding the new column.

        Raises:
            ValueError: If the DataFrame is empty or lacks required columns.
            KeyError: If the indicator type is not registered.

        Example usage:
            # Add RSI indicator
            jq.add_indicator(IndicatorType.RSI, "rsi_14", period=14)
            # Add Stochastic indicator
            jq.add_indicator(IndicatorType.STOCHASTIC, "stoch_14", period=14, n_smooth=3)
            # Add MACD indicator
            jq.add_indicator(IndicatorType.MACD, "macd", short_period=12, long_period=26)
        """
        if self._df.empty:
            raise ValueError(
                "DataFrame is empty. Cannot add indicators to empty DataFrame."
            )

        # Get the indicator class from the registry
        indicator_class = get_indicator_class(indicator_type)

        # Create the indicator instance and calculate values
        indicator_instance = indicator_class(self._df, *args, **kwargs)
        indicator_values = indicator_instance.calculate()

        # Add the indicator values as a new column
        self._df = self._df.assign(**{column_name: indicator_values})
