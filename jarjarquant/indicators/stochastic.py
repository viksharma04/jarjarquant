import numpy as np
import polars as pl

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType
from jarjarquant.indicators._math_utils import rolling_max, rolling_min


@register_indicator(IndicatorType.STOCHASTIC)
class Stochastic(Indicator):
    """Class to calculate the stochastic oscillator"""

    def __init__(
        self,
        ohlcv_df: pl.DataFrame,
        lookback: int = 14,
        n_smooth: int = 2,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.n_smooth = n_smooth
        self._transform = transform

    def _compute(self) -> np.ndarray:
        close = self._df["Close"].to_numpy()
        n = len(close)
        output = np.full(n, 50.0)

        high_max_arr = rolling_max(close, self.lookback)
        low_min_arr = rolling_min(close, self.lookback)

        for i in range(self.lookback, n):
            if high_max_arr[i] == low_min_arr[i]:
                output[i] = 50.0
            else:
                sto_0 = 100 * (close[i] - low_min_arr[i]) / (high_max_arr[i] - low_min_arr[i])
                if self.n_smooth == 0:
                    output[i] = sto_0
                elif self.n_smooth == 1:
                    if i == self.lookback:
                        output[i] = sto_0
                    else:
                        output[i] = 0.33333 * sto_0 + 0.66667 * output[i - 1]
                else:
                    if i < self.lookback + 1:
                        output[i] = sto_0
                    elif i == self.lookback + 1:
                        output[i] = 0.33333 * sto_0 + 0.66667 * output[i - 1]
                    else:
                        sto_1 = 0.33333 * sto_0 + 0.66667 * output[i - 1]
                        output[i] = 0.33333 * sto_1 + 0.66667 * output[i - 2]

        return output
