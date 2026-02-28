import numpy as np
import polars as pl
from numba import jit

from jarjarquant.volatility import atr_volatility
from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import IndicatorType, register_indicator


@jit(nopython=True, cache=True, nogil=True)
def _compute_gap_size(
    open_prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_values: np.ndarray,
) -> np.ndarray:
    """Numba-optimized computation of the gap size indicator."""
    n = len(open_prices)
    output = np.full(n, 0.0)

    for i in range(1, n):
        prev_range = high[i - 1] - low[i - 1]

        if prev_range <= 0.0 or np.isnan(atr_values[i]) or atr_values[i] <= 0.0:
            continue

        gap_abs = abs(open_prices[i] - close[i - 1])
        raw = gap_abs / prev_range

        prev_direction = close[i - 1] - open_prices[i - 1]
        gap_direction = open_prices[i] - close[i - 1]

        if gap_direction == 0.0:
            continue

        if prev_direction * gap_direction > 0.0:
            sign = 1.0
        elif prev_direction * gap_direction < 0.0:
            sign = -1.0
        else:
            sign = 1.0

        output[i] = sign * raw / atr_values[i]

    return output


@register_indicator(IndicatorType.GAP_SIZE)
class GapSize(Indicator):
    """Gap Size indicator measuring opening gap magnitude and direction."""

    def __init__(
        self,
        ohlcv_df: pl.DataFrame,
        atr_period: int = 14,
    ):
        super().__init__(ohlcv_df)
        self.atr_period = atr_period

    def _compute(self) -> np.ndarray:
        open_prices = self._df["Open"].to_numpy().astype(np.float64)
        high = self._df["High"].to_numpy().astype(np.float64)
        low = self._df["Low"].to_numpy().astype(np.float64)
        close = self._df["Close"].to_numpy().astype(np.float64)

        atr_values = atr_volatility(high, low, close, self.atr_period)

        return _compute_gap_size(open_prices, high, low, close, atr_values)
