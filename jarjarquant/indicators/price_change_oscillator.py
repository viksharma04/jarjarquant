import numpy as np
import pandas as pd
from scipy.stats import norm

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType


@register_indicator(IndicatorType.PRICE_CHANGE_OSCILLATOR)
class PriceChangeOscillator(Indicator):
    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
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
        close = np.asarray(self.df["Close"].values)
        prices = np.log(close)
        n = len(close)

        output = np.full(n, 0.0)
        long_lookback = self.long_lookback_multiplier * self.short_lookback

        # Calculate ATR over the long lookback period
        atr = self.data_analyst.atr(
            long_lookback, self.df["High"], self.df["Low"], self.df["Close"]
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
            denom = atr[i] * const
            denom = np.maximum(denom, 1e-8)

            raw = (short_ma - long_ma) / denom
            output[i] = 100 * norm.cdf(4 * raw) - 50

        # Replace nan and inf values with 0
        output = np.where(np.isnan(output), 0, output)

        if self.transform is not None:
            output = self.feature_engineer.transform(pd.Series(output), self.transform)
            output = np.asarray(output)

        return output
