import numpy as np
import polars as pl

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import IndicatorType, register_indicator

from .rsi import RSI
from .stochastic import Stochastic


@register_indicator(IndicatorType.STOCHASTIC_RSI)
class StochasticRSI(Indicator):
    """Class to calculate the Stochastic RSI indicator"""

    def __init__(
        self,
        ohlcv_df: pl.DataFrame,
        rsi_period: int = 14,
        stochastic_period: int = 14,
        n_smooth: int = 2,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.rsi_period = rsi_period
        self.stochastic_period = stochastic_period
        self.n_smooth = n_smooth
        self._transform = transform

    def _compute(self) -> np.ndarray:
        rsi = RSI(ohlcv_df=self._df, period=self.rsi_period).calculate()
        rsi_df = pl.DataFrame({"Close": rsi})
        output = Stochastic(rsi_df, self.stochastic_period, self.n_smooth).calculate()
        return output
