import numpy as np
import pandas as pd
from scipy.stats import norm

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType


@register_indicator(IndicatorType.MACD)
class MACD(Indicator):
    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        short_period: int = 5,
        long_period: int = 20,
        smoothing_factor: int = 2,
        return_raw_macd: bool = False,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.short_period = short_period
        self.long_period = long_period
        self.smoothing_factor = smoothing_factor
        self.indicator_type = "continuous"
        self.return_raw_macd = return_raw_macd
        self.transform = transform

    def calculate(self) -> np.ndarray:
        close = self.df["Close"].values
        short_ema = np.asarray(
            pd.Series(close).ewm(span=self.short_period, adjust=False).mean().values,
            dtype=np.float64,
        )
        long_ema = np.asarray(
            pd.Series(close).ewm(span=self.long_period, adjust=False).mean().values,
            dtype=np.float64,
        )

        atr_values = self.data_analyst.atr(
            self.short_period + self.long_period,
            self.df["High"],
            self.df["Low"],
            self.df["Close"],
        ).values

        denom = atr_values * np.sqrt(
            (0.5 * (self.long_period - 1) + self.short_period)
            - (0.5 * (self.short_period - 1))
        )

        norm_diff = (short_ema - long_ema) / denom
        COMPRESSION_FACTOR = 1.0
        macd = 100 * norm.cdf(COMPRESSION_FACTOR * norm_diff) - 50

        # Replace nan values with 0
        macd = np.where(np.isnan(macd), 0, macd)

        if self.transform is not None:
            macd = self.feature_engineer.transform(pd.Series(macd), self.transform)
            macd = np.asarray(macd)

        if self.return_raw_macd:
            return macd
        else:
            signal_line = (
                pd.Series(macd)
                .ewm(span=self.smoothing_factor, adjust=False)
                .mean()
                .values
            )
            return macd - signal_line
