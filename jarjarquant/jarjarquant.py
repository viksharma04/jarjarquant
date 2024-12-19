import pandas as pd
from .data_gatherer import DataGatherer
from .feature_engineer import FeatureEngineer
from .labeller import Labeller

from .indicator import Indicator, RSI, CMMA


class Jarjarquant:
    """
    Jarjarquant integrates data gathering, labeling, and feature engineering for financial time series.
    """

    def __init__(self, ohlcv_df=None):
        self.data_gatherer = DataGatherer()
        if ohlcv_df is None:
            samples = self.data_gatherer.get_random_price_samples(
                num_tickers_to_sample=1)
            if not samples:
                raise ValueError(
                    "No price samples were returned. Please check the data source.")
            self._df = samples[0]
        else:
            self._df = ohlcv_df
        self.labeller = Labeller(self._df)
        self.feature_engineer = FeatureEngineer()

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
        try:
            series = data_gatherer.get_yf_ticker(ticker, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to fetch data for ticker '{
                             ticker}'. Error: {e}") from e
        return cls(series)

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Must be a pandas DataFrame!")
        self._df = df

    @df.deleter
    def df(self):
        del self._df

    # Add any additional methods or functionality here
    @staticmethod
    def rsi(ohlcv_df, period: int = 14):
        _df = ohlcv_df.copy()
        if 'Close' not in _df.columns:
            raise ValueError(
                "The input DataFrame must contain a 'Close' column for RSI calculation.")
        rsi_indicator = RSI(_df, period)
        return rsi_indicator

    def add_rsi(self, period: int = 14):
        self._df = self._df.assign(RSI=self.rsi(
            ohlcv_df=self._df, period=period))

    @staticmethod
    def cmma(ohlcv_df: pd.DataFrame, lookback: int = 21, atr_length: int = 21):
        _df = ohlcv_df.copy()
        if 'Close' not in _df.columns:
            raise ValueError(
                "The input dataframe must contain a 'Close' column for CMMA calculation")
        cmma_indicator = CMMA(_df, lookback, atr_length)

        return cmma_indicator

    def add_cmma(self, lookback: int = 21, atr_length: int = 21):
        self._df = self._df.assign(CMMA=self.cmma(
            ohlcv_df=self._df, lookback=lookback, atr_length=atr_length))
