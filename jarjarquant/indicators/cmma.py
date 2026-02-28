import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import IndicatorType, register_indicator
from jarjarquant.volatility import atr_volatility


@register_indicator(IndicatorType.CMMA)
class CMMA(Indicator):
    """Close Minus Moving Average (CMMA) Indicator"""

    def __init__(
        self,
        ohlcv_df: pl.DataFrame,
        lookback: int = 21,
        atr_length: int = 21,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.atr_length = atr_length
        self._transform = transform

    def _compute(self) -> np.ndarray:
        Close = self._df["Close"].to_numpy()
        Low = self._df["Low"].to_numpy()
        High = self._df["High"].to_numpy()

        log_close = np.log(Close)
        rolling_mean = pd.Series(log_close).ewm(span=self.lookback).mean()

        denom = atr_volatility(
            High, Low, Close, self.atr_length, use_ema=True
        ) * np.sqrt(self.lookback + 1)

        normalized_output = np.where(denom > 0, (log_close - rolling_mean) / denom, 0)
        output = 100 * norm.cdf(normalized_output) - 50

        return output
