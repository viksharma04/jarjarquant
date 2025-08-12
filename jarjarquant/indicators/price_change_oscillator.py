import numpy as np
import polars as pl
import pandas as pd
from scipy.stats import norm

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType


@register_indicator(IndicatorType.PRICE_CHANGE_OSCILLATOR)
class PriceChangeOscillator(Indicator):
    def __init__(
        self,
        ohlcv_df: pl.DataFrame,
        short_lookback: int = 5,
        long_lookback_multiplier: int = 5,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.short_lookback = short_lookback
        self.long_lookback_multiplier = long_lookback_multiplier
        self.indicator_type = "continuous"
        self.transform = transform

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
            pd.Series(self.df["Close"].to_numpy())
        ).values

        for i in range(long_lookback, n):
            # Calculate the short-term and long-term mean
            short_ma = np.mean(
                prices[i - self.short_lookback + 1 : i]
                - prices[i - self.short_lookback : i - 1]
            )
            long_ma = np.mean(
                prices[i - long_lookback + 1 : i] - prices[i - long_lookback : i - 1]
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

        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)
            output = np.asarray(output)

        return output
