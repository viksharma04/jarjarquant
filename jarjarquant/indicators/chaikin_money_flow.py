import numpy as np
import polars as pl

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import IndicatorType, register_indicator
from jarjarquant.indicators._math_utils import ewm, rolling_mean


@register_indicator(IndicatorType.CHAIKIN_MONEY_FLOW)
class ChaikinMoneyFlow(Indicator):
    def __init__(
        self,
        ohlcv_df: pl.DataFrame,
        smoothing_lookback: int = 21,
        volume_lookback: int = 21,
        return_cmf: bool = False,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.smoothing_lookback = smoothing_lookback
        self.volume_lookback = volume_lookback
        self.return_cmf = return_cmf
        self._transform = transform

    def _compute(self) -> np.ndarray:
        Close = self._df["Close"].to_numpy()
        High = self._df["High"].to_numpy()
        Low = self._df["Low"].to_numpy()
        Volume = self._df["Volume"].to_numpy()

        output = np.full(len(Close), 0.0)

        first_non_zero_vol = np.argmax(np.asarray(Volume) > 0)

        for i in range(first_non_zero_vol, len(Close)):
            if High[i] == Low[i]:
                output[i] = 0
            else:
                output[i] = (
                    100
                    * ((2 * Close[i] - High[i] - Low[i]) / (High[i] - Low[i]))
                    * Volume[i]
                )

        output = rolling_mean(output, self.smoothing_lookback)

        if self.return_cmf:
            sma_volume = rolling_mean(Volume, self.volume_lookback)
            output = np.where(sma_volume != 0, output / sma_volume, 0)
        else:
            ema_volume = ewm(Volume, span=self.volume_lookback)
            output = np.where(ema_volume != 0, output / ema_volume, 0)

        output = np.where(np.isnan(output), 0, output)

        return output
