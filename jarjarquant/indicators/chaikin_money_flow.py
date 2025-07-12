import numpy as np
import pandas as pd

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType


@register_indicator(IndicatorType.CHAIKIN_MONEY_FLOW)
class ChaikinMoneyFlow(Indicator):
    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        smoothing_lookback: int = 21,
        volume_lookback: int = 21,
        return_cmf: bool = False,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.smoothing_lookback = smoothing_lookback
        self.volume_lookback = volume_lookback
        self.return_cmf = return_cmf
        self.transform = transform
        self.indicator_type = "continuous"

    def calculate(self) -> np.ndarray:
        Close = self.df["Close"].values
        High = self.df["High"].values
        Low = self.df["Low"].values
        Volume = self.df["Volume"].values

        output = np.full(len(Close), 0.0)

        # Look for first bar with non-zero volume
        first_non_zero_vol = np.argmax(np.asarray(Volume) > 0)

        # Calculate the intraday intensity of each bar after the first non-zero volume bar
        # Handle case if high and low are equal
        for i in range(first_non_zero_vol, len(Close)):
            if High[i] == Low[i]:
                output[i] = 0
            else:
                output[i] = (
                    100
                    * ((2 * Close[i] - High[i] - Low[i]) / (High[i] - Low[i]))
                    * Volume[i]
                )

        # Calculate the SMA of output using the smoothing lookback period
        output = np.asarray(
            pd.Series(output).rolling(window=self.smoothing_lookback).mean().values
        )

        if self.return_cmf:
            # Calculate the SMA of Volume values using the volume lookback period
            sma_volume = (
                pd.Series(Volume).rolling(window=self.volume_lookback).mean().values
            )

            # Calculate the Chaikin Money Flow by dividing the SMA of output by the SMA of Volume and handling division by zero
            output = np.where(sma_volume != 0, output / sma_volume, 0)

        else:
            # Finally smooth the indicator values by dividing the output by the EMA of volume calculated using n_smooth
            ema_volume = pd.Series(Volume).ewm(span=self.volume_lookback).mean().values

            # Normalized output
            output = np.where(ema_volume != 0, output / ema_volume, 0)

        # Replace nan and inf values with 0
        output = np.where(np.isnan(output), 0, output)

        if self.transform is not None:
            output = self.feature_engineer.transform(pd.Series(output), self.transform)
            output = np.asarray(output)

        return output
