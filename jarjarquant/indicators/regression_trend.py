import numpy as np
import polars as pl
from scipy.stats import norm

from jarjarquant._cython.indicators import compute_trend_indicator
from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType
from jarjarquant.indicators._math_utils import compute_normalized_legendre_coefficients
from jarjarquant.volatility import atr_volatility


@register_indicator(IndicatorType.REGRESSION_TREND)
class RegressionTrend(Indicator):
    def __init__(
        self,
        ohlcv_df: pl.DataFrame,
        lookback: int = 21,
        atr_length_mult: int = 3,
        degree: int = 1,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.degree = degree
        self.atr_length = atr_length_mult * lookback
        self._transform = transform

    def _compute(self) -> np.ndarray:
        close = self._df["Close"].to_numpy()
        n = len(close)

        if self.lookback > n:
            raise ValueError(
                "Lookback period is greater than the number of data points!"
            )

        lgdre = compute_normalized_legendre_coefficients(self.lookback, self.degree)

        if self.atr_length < 1:
            self.atr_length = self.lookback

        atr_values = atr_volatility(
            self._df["High"].to_numpy(),
            self._df["Low"].to_numpy(),
            self._df["Close"].to_numpy(),
            self.atr_length,
        )

        COMPRESSION_FACTOR = 1.5

        output = compute_trend_indicator(
            close, lgdre, self.lookback, self.atr_length, atr_values
        )

        output = 100 * norm.cdf(COMPRESSION_FACTOR * output) - 50

        return output
