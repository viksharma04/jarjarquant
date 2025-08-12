import numpy as np
import polars as pl
import pandas as pd
from scipy.stats import norm

from jarjarquant.cython_utils.indicators import compute_trend_indicator
from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType


@register_indicator(IndicatorType.REGRESSION_TREND)
class RegressionTrend(Indicator):
    def __init__(
        self,
        ohlcv_df: pl.DataFrame,
        lookback: int = 21,
        atr_length_mult: int = 3,
        degree: int = 1,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.degree = degree
        self.atr_length = atr_length_mult * lookback
        self.indicator_type = "continuous"
        self.transform = transform

    def calculate(self) -> np.ndarray:
        close = self.df["Close"].to_numpy()
        n = len(close)
        output = np.full(n, 0.0)

        if self.lookback > n:
            raise ValueError(
                "Lookback period is greater than the number of data points!"
            )

        # Calculate the Legendre polynomials
        from jarjarquant.data_analyst import compute_normalized_legendre_coefficients
        lgdre = compute_normalized_legendre_coefficients(
            self.lookback, self.degree
        )
        if self.atr_length < 1:
            self.atr_length = self.lookback
            expanding_atr = True
        else:
            expanding_atr = False
        from jarjarquant.data_analyst import atr
        atr_values = atr(
            self.atr_length,
            pd.Series(self.df["High"].to_numpy()),
            pd.Series(self.df["Low"].to_numpy()),
            pd.Series(self.df["Close"].to_numpy()),
            expanding=expanding_atr,
        ).values
        COMPRESSION_FACTOR = 1.5

        output = compute_trend_indicator(
            close, lgdre, self.lookback, self.atr_length, atr_values
        )

        output = 100 * norm.cdf(COMPRESSION_FACTOR * output) - 50

        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)
            output = np.asarray(output)

        return output
