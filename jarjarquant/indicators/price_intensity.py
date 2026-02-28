import numpy as np
import polars as pl
from scipy.stats import norm

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import IndicatorType, register_indicator
from jarjarquant.indicators._math_utils import ewm


@register_indicator(IndicatorType.PRICE_INTENSITY)
class PriceIntensity(Indicator):
    def __init__(
        self, ohlcv_df: pl.DataFrame, smoothing_factor: int = 2, transform=None
    ):
        super().__init__(ohlcv_df)
        self.smoothing_factor = smoothing_factor
        self._transform = transform

    def _compute(self) -> np.ndarray:
        close = self._df["Close"].to_numpy()
        high = self._df["High"].to_numpy()
        low = self._df["Low"].to_numpy()
        _open = self._df["Open"].to_numpy()

        n = len(close)
        output = np.full(n, 0.0)

        range_0 = high[0] - low[0]
        output[0] = (close[0] - _open[0]) / range_0 if range_0 != 0 else 0.0

        for i in range(1, n):
            denom = np.maximum.reduce(
                [high[i] - low[i], high[i] - close[i - 1], close[i - 1] - low[i]]
            )
            output[i] = (close[i] - _open[i]) / denom if denom != 0 else 0.0

        output = ewm(output, span=self.smoothing_factor, adjust=False)

        output = 100 * norm.cdf(0.8 * np.sqrt(self.smoothing_factor) * output) - 50
        output = np.where(np.isnan(output), 0, output)

        return output
