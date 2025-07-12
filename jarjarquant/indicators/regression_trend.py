import numpy as np
import pandas as pd
from scipy.stats import norm

from jarjarquant.cython_utils.indicators import compute_trend_indicator
from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType


@register_indicator(IndicatorType.REGRESSION_TREND)
class RegressionTrend(Indicator):
    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
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
        close = self.df["Close"].values
        n = len(close)
        output = np.full(n, 0.0)

        if self.lookback > n:
            raise ValueError(
                "Lookback period is greater than the number of data points!"
            )

        # Calculate the Legendre polynomials
        lgdre = self.data_analyst.compute_normalized_legendre_coefficients(
            self.lookback, self.degree
        )
        if self.atr_length < 1:
            self.atr_length = self.lookback
            expanding_atr = True
        else:
            expanding_atr = False
        atr = self.data_analyst.atr(
            self.atr_length,
            self.df["High"],
            self.df["Low"],
            self.df["Close"],
            expanding=expanding_atr,
        ).values
        COMPRESSION_FACTOR = 1.5

        output = compute_trend_indicator(
            close, lgdre, self.lookback, self.atr_length, atr
        )

        output = 100 * norm.cdf(COMPRESSION_FACTOR * output) - 50

        if self.transform is not None:
            output = self.feature_engineer.transform(pd.Series(output), self.transform)
            output = np.asarray(output)

        return output
