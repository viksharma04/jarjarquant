import numpy as np
import pandas as pd
from scipy.stats import norm

from jarjarquant.indicators.base import Indicator


class MovingAverageDifference(Indicator):
    """
    A class to calculate the Moving Average Difference (MAD) indicator.
    The MAD indicator is a normalized difference between a short-term and a long-term moving average,
    adjusted by the Average True Range (ATR) to account for volatility.
    Attributes:
        ohlcv_df (pd.DataFrame): DataFrame containing OHLCV (Open, High, Low, Close, Volume) data.
        short_period (int): The period for the short-term moving average. Default is 5.
        long_period (int): The period for the long-term moving average. Default is 20.
        indicator_type (str): Type of the indicator, set to 'continuous'.
    Methods:
        calculate() -> np.ndarray:
            Calculates the Moving Average Difference (MAD) indicator.
            Returns:
                np.ndarray: The calculated MAD values.
    """

    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        short_period: int = 5,
        long_period: int = 20,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.short_period = short_period
        self.long_period = long_period
        self.indicator_type = "continuous"
        self.transform = transform

    def calculate(self) -> np.ndarray:
        close = self.df["Close"].values
        # Calculate the short-term MA
        short_ma = pd.Series(close).rolling(window=self.short_period).mean().values

        # Calculate the long-term MA
        long_ma = pd.Series(close).rolling(window=self.long_period).mean()
        # Lag long_ma by short_period
        long_ma = long_ma.shift(self.short_period).values

        # Ensure both arrays are NumPy arrays for arithmetic operations
        short_ma = np.asarray(short_ma)
        long_ma = np.asarray(long_ma)

        # See pg 116 eq. 4.7 and 4.8 of Statistically Sound Indicators
        atr_values = self.data_analyst.atr(
            self.short_period + self.long_period,
            self.df["High"],
            self.df["Low"],
            self.df["Close"],
        ).values

        denom = atr_values * np.sqrt(
            (0.5 * (self.long_period - 1) + self.short_period)
            - (0.5 * (self.short_period - 1))
        )

        norm_diff = (short_ma - long_ma) / denom
        COMPRESSION_FACTOR = 1.5
        mad = 100 * norm.cdf(COMPRESSION_FACTOR * norm_diff) - 50

        # Replace nan values with 0
        mad = np.where(np.isnan(mad), 0, mad)

        if self.transform is not None:
            mad = self.feature_engineer.transform(pd.Series(mad), self.transform)
            mad = np.asarray(mad)

        return mad
