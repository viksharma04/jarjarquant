import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

from jarjarquant.core.schemas import VolatilityMeasure
from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import IndicatorType, register_indicator


@register_indicator(IndicatorType.PRICE_CHANGE_OSCILLATOR)
class PriceChangeOscillator(Indicator):
    def __init__(
        self,
        ohlcv_df: pl.DataFrame,
        short_lookback: int = 5,
        long_lookback_multiplier: int = 5,
        indicator_volatilty_measure: VolatilityMeasure = VolatilityMeasure.ATR,
        scaling_volatility_measure: VolatilityMeasure = VolatilityMeasure.ATR,
    ):
        super().__init__(ohlcv_df)
        self.short_lookback = short_lookback
        self.long_lookback_multiplier = long_lookback_multiplier
        self.indicator_type = "continuous"

    def calculate(self) -> np.ndarray:
        close = self.df["Close"].to_numpy()
        prices = np.log(close)
        n = len(close)

        output = np.full(n, 0.0)
        long_lookback = self.long_lookback_multiplier * self.short_lookback

        # Calculate ATR over the long lookback period
        from jarjarquant.data_analyst import atr

        atr_values = atr(
            long_lookback,
            pd.Series(self.df["High"].to_numpy()),
            pd.Series(self.df["Low"].to_numpy()),
            pd.Series(self.df["Close"].to_numpy()),
        ).values

        for i in range(long_lookback + self.short_lookback, n):
            # Calculate the short-term mean (most recent short_lookback returns)
            short_ma = np.mean(
                prices[i - self.short_lookback + 1 : i + 1]
                - prices[i - self.short_lookback : i]
            )
            # Calculate the long-term mean (lagged by short_lookback to avoid overlap)
            long_start = i - self.short_lookback - long_lookback
            long_end = i - self.short_lookback
            long_ma = np.mean(
                prices[long_start + 1 : long_end + 1] - prices[long_start:long_end]
            )

            const = (
                0.36
                + (1 / self.short_lookback)
                + 0.7 * np.log(0.5 * self.long_lookback_multiplier) / 1.609
            )
            denom = atr_values[i] * const
            denom = np.maximum(denom, 1e-8)

            raw = (short_ma - long_ma) / denom
            output[i] = 100 * norm.cdf(4 * raw) - 50

        # Replace nan and inf values with 0
        output = np.where(np.isnan(output), 0, output)

        return output
