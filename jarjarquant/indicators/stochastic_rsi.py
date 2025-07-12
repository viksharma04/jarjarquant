import numpy as np
import pandas as pd

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType

from .rsi import RSI
from .stochastic import Stochastic


@register_indicator(IndicatorType.STOCHASTIC_RSI)
class StochasticRSI(Indicator):
    """Class to calculate the Stochastic RSI indicator"""

    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        rsi_period: int = 14,
        stochastic_period: int = 14,
        n_smooth: int = 2,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.rsi_period = rsi_period
        self.stochastic_period = stochastic_period
        self.n_smooth = n_smooth
        self.indicator_type = "continuous"
        self.transform = transform

    def calculate(self) -> np.ndarray:
        rsi = RSI(self.df, self.rsi_period).calculate()
        # Store RSI values in a DataFrame and rename the column to 'Close'
        rsi_df = pd.DataFrame(rsi, columns=["Close"])
        output = Stochastic(rsi_df, self.stochastic_period, self.n_smooth).calculate()

        if self.transform is not None:
            output = self.feature_engineer.transform(pd.Series(output), self.transform)
            output = np.asarray(output)

        return output
