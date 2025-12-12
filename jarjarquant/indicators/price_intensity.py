import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import IndicatorType, register_indicator


@register_indicator(IndicatorType.PRICE_INTENSITY)
class PriceIntensity(Indicator):
    def __init__(
        self, ohlcv_df: pl.DataFrame, smoothing_factor: int = 2, transform=None
    ):
        super().__init__(ohlcv_df)
        self.smoothing_factor = smoothing_factor
        self.indicator_type = "continuous"
        self.transform = transform

    def calculate(self) -> np.ndarray:
        close = self.df["Close"].to_numpy()
        high = self.df["High"].to_numpy()
        low = self.df["Low"].to_numpy()
        _open = self.df["Open"].to_numpy()

        n = len(close)
        output = np.full(n, 0.0)

        # Special case for the first value
        range_0 = high[0] - low[0]
        output[0] = (close[0] - _open[0]) / range_0 if range_0 != 0 else 0.0

        # Calculate Raw Price Intensity
        for i in range(1, n):
            denom = np.maximum.reduce(
                [high[i] - low[i], high[i] - close[i - 1], close[i - 1] - low[i]]
            )
            output[i] = (close[i] - _open[i]) / denom if denom != 0 else 0.0

        # Smooth the Price Intensity values
        output = (
            pd.Series(output)
            .ewm(span=self.smoothing_factor, adjust=False)
            .mean()
            .values
        )

        # Normalize the Price Intensity values
        output = 100 * norm.cdf(0.8 * np.sqrt(self.smoothing_factor) * output) - 50

        # Replace nan and inf values with 0
        output = np.where(np.isnan(output), 0, output)

        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)
            output = np.asarray(output)

        return output
