import numpy as np
import polars as pl
from numba import jit

from ..core.volatility_calculations import atr_volatility
from ..indicators.base import Indicator
from ..indicators.registry import IndicatorType, register_indicator


@jit(nopython=True, cache=True, nogil=True)
def _compute_gap_size(
    open_prices: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_values: np.ndarray,
) -> np.ndarray:
    """
    Numba-optimized computation of the gap size indicator.

    For each bar i (starting from i=1):
      1. gap_abs = |open[i] - close[i-1]|
      2. prev_range = high[i-1] - low[i-1]
      3. raw = gap_abs / prev_range
      4. Sign: positive if the gap extends the previous day's direction,
         negative if it reverses.
         - prev_direction = close[i-1] - open[i-1]  (positive = bullish, negative = bearish)
         - gap_direction  = open[i] - close[i-1]
         - If both have the same sign, the gap extends -> positive sign
         - If opposite signs, the gap reverses -> negative sign
      5. Scale by ATR: result = signed_raw / atr[i]

    Args:
        open_prices: Array of open prices
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        atr_values: Array of ATR values

    Returns:
        np.ndarray: Gap size indicator values
    """
    n = len(open_prices)
    output = np.full(n, 0.0)

    for i in range(1, n):
        prev_range = high[i - 1] - low[i - 1]

        # Skip if previous range is zero (no price movement) or ATR is NaN/zero
        if prev_range <= 0.0 or np.isnan(atr_values[i]) or atr_values[i] <= 0.0:
            continue

        gap_abs = abs(open_prices[i] - close[i - 1])
        raw = gap_abs / prev_range

        # Determine sign based on whether gap extends or reverses previous direction
        prev_direction = close[i - 1] - open_prices[i - 1]
        gap_direction = open_prices[i] - close[i - 1]

        # If both directions are zero, the gap size is zero anyway
        if gap_direction == 0.0:
            continue

        # Extends: same sign -> positive; Reverses: opposite sign -> negative
        if prev_direction * gap_direction > 0.0:
            sign = 1.0
        elif prev_direction * gap_direction < 0.0:
            sign = -1.0
        else:
            # prev_direction is zero (doji) - treat gap as extending
            sign = 1.0

        output[i] = sign * raw / atr_values[i]

    return output


@register_indicator(IndicatorType.GAP_SIZE)
class GapSize(Indicator):
    """
    Gap Size indicator measuring the magnitude and direction of opening gaps
    relative to the previous day's range, scaled by ATR.

    The indicator captures:
      - How large the gap is relative to the previous day's high-low range
      - Whether the gap extends or reverses the previous day's direction
      - Normalization by ATR for cross-asset comparability

    A positive value means the gap extends the prior day's trend direction.
    A negative value means the gap reverses it.

    Args:
        ohlcv_df: OHLCV DataFrame with Open, High, Low, Close columns
        atr_period: Lookback period for ATR calculation (default: 14)
    """

    def __init__(
        self,
        ohlcv_df: pl.DataFrame,
        atr_period: int = 14,
    ):
        super().__init__(ohlcv_df)
        self.atr_period = atr_period
        self.indicator_type = "continuous"

    def calculate(self) -> np.ndarray:
        """
        Calculate the gap size indicator.

        Returns:
            np.ndarray: Array of gap size values. Values before the ATR
                        warmup period will be 0.0.
        """
        open_prices = self.df["Open"].to_numpy().astype(np.float64)
        high = self.df["High"].to_numpy().astype(np.float64)
        low = self.df["Low"].to_numpy().astype(np.float64)
        close = self.df["Close"].to_numpy().astype(np.float64)

        # Calculate ATR using core volatility calculations
        atr_values = atr_volatility(high, low, close, self.atr_period)

        return _compute_gap_size(open_prices, high, low, close, atr_values)
