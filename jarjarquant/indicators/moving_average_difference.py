import numpy as np
import polars as pl
import pandas as pd
from scipy.stats import norm

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType
from jarjarquant.volatility import atr_volatility


@register_indicator(IndicatorType.MOVING_AVERAGE_DIFFERENCE)
class MovingAverageDifference(Indicator):
    """Moving Average Difference (MAD) indicator."""

    def __init__(
        self,
        ohlcv_df: pl.DataFrame,
        short_period: int = 5,
        long_period: int = 20,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.short_period = short_period
        self.long_period = long_period
        self._transform = transform

    def _compute(self) -> np.ndarray:
        close = self._df["Close"].to_numpy()
        short_ma = pd.Series(close).rolling(window=self.short_period).mean().values
        long_ma = pd.Series(close).rolling(window=self.long_period).mean()
        long_ma = long_ma.shift(self.short_period).values

        short_ma = np.asarray(short_ma)
        long_ma = np.asarray(long_ma)

        atr_values = atr_volatility(
            self._df["High"].to_numpy(),
            self._df["Low"].to_numpy(),
            self._df["Close"].to_numpy(),
            self.short_period + self.long_period,
        )

        denom = atr_values * np.sqrt(
            (0.5 * (self.long_period - 1) + self.short_period)
            - (0.5 * (self.short_period - 1))
        )

        norm_diff = (short_ma - long_ma) / denom
        COMPRESSION_FACTOR = 1.5
        mad = 100 * norm.cdf(COMPRESSION_FACTOR * norm_diff) - 50

        mad = np.where(np.isnan(mad), 0, mad)

        return mad
