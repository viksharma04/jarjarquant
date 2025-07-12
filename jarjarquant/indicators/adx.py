import numpy as np
import pandas as pd

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType


@register_indicator(IndicatorType.ADX)
class ADX(Indicator):
    def __init__(self, ohlcv_df: pd.DataFrame, lookback: int = 14, transform=None):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.indicator_type = "continuous"
        self.transform = transform

    def calculate(self) -> np.ndarray:
        close = self.df["Close"].values
        high = self.df["High"].values
        low = self.df["Low"].values
        _open = self.df["Open"].values

        n = len(close)
        output = np.full(n, 0.0)

        dms_plus = 0
        dms_minus = 0
        atr_sum = 0

        # Initialize the high and low movement variables using a SMA over the lookback period
        for i in range(1, self.lookback):
            dm_plus = high[i] - high[i - 1]
            dm_minus = low[i - 1] - low[i]

            if dm_plus >= dm_minus:
                dm_minus = 0
            else:
                dm_plus = 0

            dm_plus = 0 if dm_plus < 0 else dm_plus
            dm_minus = 0 if dm_minus < 0 else dm_minus

            dms_plus += dm_plus
            dms_minus += dm_minus

            # Calculate and cumulate the ATR
            atr = np.maximum.reduce(
                [high[i] - low[i], high[i] - close[i - 1], close[i - 1] - low[i]]
            )
            atr_sum += atr

            di_plus = dms_plus / atr_sum if atr_sum != 0 else 0
            di_minus = dms_minus / atr_sum if atr_sum != 0 else 0

            adx = (
                np.abs(di_plus - di_minus) / (di_plus + di_minus)
                if di_plus + di_minus != 0
                else 0
            )

            output[i] = 100 * adx

        adx_sum = 0
        # Secondary initialization to generate ADX values to begin exp smoothing
        for i in range(self.lookback, self.lookback * 2):
            dm_plus = high[i] - high[i - 1]
            dm_minus = low[i - 1] - low[i]

            if dm_plus >= dm_minus:
                dm_minus = 0
            else:
                dm_plus = 0

            dm_plus = 0 if dm_plus < 0 else dm_plus
            dm_minus = 0 if dm_minus < 0 else dm_minus

            # Begin using exp smoothing instead of SMA
            dms_plus = (self.lookback - 1) / self.lookback * dms_plus + dm_plus
            dms_minus = (self.lookback - 1) / self.lookback * dms_minus + dm_minus

            # Calculate and cumulate the ATR
            atr = np.maximum.reduce(
                [high[i] - low[i], high[i] - close[i - 1], close[i - 1] - low[i]]
            )
            atr_sum = (self.lookback - 1) / self.lookback * atr_sum + atr

            di_plus = dms_plus / atr_sum if atr_sum != 0 else 0
            di_minus = dms_minus / atr_sum if atr_sum != 0 else 0

            adx = (
                np.abs(di_plus - di_minus) / (di_plus + di_minus)
                if di_plus + di_minus != 0
                else 0
            )

            adx_sum += adx

            output[i] = 100 * adx

        # Secondary initialization complete - use adx/lookback as the first value
        adx_sum /= self.lookback

        # Final loop to calculate rest of the values
        for i in range(self.lookback * 2, n):
            dm_plus = high[i] - high[i - 1]
            dm_minus = low[i - 1] - low[i]

            if dm_plus >= dm_minus:
                dm_minus = 0
            else:
                dm_plus = 0

            dm_plus = 0 if dm_plus < 0 else dm_plus
            dm_minus = 0 if dm_minus < 0 else dm_minus

            # Begin using exp smoothing instead of SMA
            dms_plus = (self.lookback - 1) / self.lookback * dms_plus + dm_plus
            dms_minus = (self.lookback - 1) / self.lookback * dms_minus + dm_minus

            # Calculate and cumulate the ATR
            atr = np.maximum.reduce(
                [high[i] - low[i], high[i] - close[i - 1], close[i - 1] - low[i]]
            )
            atr_sum = (self.lookback - 1) / self.lookback * atr_sum + atr

            di_plus = dms_plus / atr_sum if atr_sum != 0 else 0
            di_minus = dms_minus / atr_sum if atr_sum != 0 else 0

            adx = (
                np.abs(di_plus - di_minus) / (di_plus + di_minus)
                if di_plus + di_minus != 0
                else 0
            )

            adx_sum = (self.lookback - 1) / self.lookback * adx_sum + adx

            output[i] = 100 * adx_sum

        # Replace any nan values with 0
        output = np.where(np.isnan(output), 0, output)

        if self.transform is not None:
            output = self.feature_engineer.transform(pd.Series(output), self.transform)
            output = np.asarray(output)

        return output
