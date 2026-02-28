import numpy as np
import polars as pl
from scipy.stats import norm

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import IndicatorType, register_indicator
from jarjarquant.indicators._math_utils import ewm
from jarjarquant.volatility import atr_volatility


@register_indicator(IndicatorType.MACD)
class MACD(Indicator):
    def __init__(
        self,
        ohlcv_df: pl.DataFrame,
        short_period: int = 5,
        long_period: int = 20,
        smoothing_factor: int = 2,
        return_raw_macd: bool = False,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.short_period = short_period
        self.long_period = long_period
        self.smoothing_factor = smoothing_factor
        self.return_raw_macd = return_raw_macd
        self._transform = transform

    def _compute(self) -> np.ndarray:
        close = self._df["Close"].to_numpy()
        short_ema = ewm(close, span=self.short_period, adjust=False).astype(np.float64)
        long_ema = ewm(close, span=self.long_period, adjust=False).astype(np.float64)

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

        norm_diff = (short_ema - long_ema) / denom
        COMPRESSION_FACTOR = 1.0
        macd = 100 * norm.cdf(COMPRESSION_FACTOR * norm_diff) - 50

        macd = np.where(np.isnan(macd), 0, macd)

        if self.return_raw_macd:
            return macd
        else:
            signal_line = ewm(macd, span=self.smoothing_factor, adjust=False)
            return macd - signal_line
