import numpy as np
import polars as pl

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType
from jarjarquant.indicators._math_utils import directional_change_pivots
from jarjarquant.volatility import atr_volatility


@register_indicator(IndicatorType.ANCHORED_VWAP)
class AnchoredVWAP(Indicator):
    """Anchored VWAP indicator using directional change pivots."""

    def __init__(
        self,
        ohlcv_df: pl.DataFrame,
        threshold_value: float = 0.02,
        atr_period: int = 14,
        price_formula: str = "ohlc4",
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.threshold_value = threshold_value
        self.atr_period = atr_period
        self.price_formula = price_formula
        self._transform = transform

    def _compute(self) -> np.ndarray:
        n = len(self._df)
        close_prices = self._df["Close"].to_numpy()

        pivots = directional_change_pivots(
            series=close_prices, threshold_value=self.threshold_value
        )

        if self.price_formula == "ohlc4":
            typical_price = (
                self._df["Open"] + self._df["High"] + self._df["Low"] + self._df["Close"]
            ) / 4
        elif self.price_formula == "hlc3":
            typical_price = (self._df["High"] + self._df["Low"] + self._df["Close"]) / 3
        else:
            raise ValueError("price_formula must be 'ohlc4' or 'hlc3'")

        volume = self._df["Volume"].to_numpy()
        typical_price_values = typical_price.to_numpy()

        atr_values = atr_volatility(
            self._df["High"].to_numpy(),
            self._df["Low"].to_numpy(),
            self._df["Close"].to_numpy(),
            self.atr_period,
        )

        output = np.full(n, np.nan)

        for i, pivot_idx in enumerate(pivots):
            start_idx = pivot_idx + 1
            if start_idx >= n:
                continue

            cumulative_pv = 0.0
            cumulative_volume = 0.0

            for j in range(start_idx, n):
                cumulative_pv += typical_price_values[j] * volume[j]
                cumulative_volume += volume[j]

                if cumulative_volume > 0:
                    vwap = cumulative_pv / cumulative_volume
                    price_diff = abs(close_prices[j] - vwap)

                    if atr_values[j] > 0:
                        normalized_distance = price_diff / atr_values[j]
                    else:
                        normalized_distance = 0.0

                    output[j] = normalized_distance

        output = np.nan_to_num(output, nan=0.0)

        return output
