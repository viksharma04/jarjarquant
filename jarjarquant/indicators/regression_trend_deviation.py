import numpy as np
import pandas as pd
from scipy.stats import norm

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType


@register_indicator(IndicatorType.REGRESSION_TREND_DEVIATION)
class RegressionTrendDeviation(Indicator):
    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        lookback: int = 14,
        fit_degree: int = 1,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.fit_degree = fit_degree
        self.indicator_type = "continuous"
        self.transform = transform

    def calculate(self) -> np.ndarray:
        if self.lookback < self.fit_degree:
            self.lookback = self.fit_degree + 2
            print(
                f"Warning: Lookback period adjusted to {
                    self.lookback
                } to accommodate fit_degree of {self.fit_degree}."
            )

        close = self.df["Close"].values
        n = len(close)

        output = np.full(n, 0.0)

        # Calculate the Legendre polynomials for 1, 2, and 3 degrees
        lgdre_1 = self.data_analyst.compute_legendre_coefficients(self.lookback, 1)
        lgdre_2 = self.data_analyst.compute_legendre_coefficients(self.lookback, 2)
        lgdre_3 = self.data_analyst.compute_legendre_coefficients(self.lookback, 3)

        # Loop over data starting from lookback-1
        for i in range(self.lookback - 1, n):
            prices = np.log(np.asarray(close[i - self.lookback + 1 : i + 1]))

            reg_coeff_1 = self.data_analyst.calculate_regression_coefficient(
                prices, lgdre_1
            )
            reg_coeff_2 = self.data_analyst.calculate_regression_coefficient(
                prices, lgdre_2
            )
            reg_coeff_3 = self.data_analyst.calculate_regression_coefficient(
                prices, lgdre_3
            )

            intercept = sum(prices) / self.lookback

            if self.fit_degree == 1:
                _predictions = reg_coeff_1 * lgdre_1 + intercept
                rms = np.sqrt(np.sum((prices - _predictions) ** 2) / self.lookback)
                error_contribution = (prices[-1] - _predictions[-1]) / rms
                output[i] = 100 * norm.cdf(0.6 * error_contribution) - 50
            elif self.fit_degree == 2:
                _predictions = reg_coeff_1 * lgdre_1 + reg_coeff_2 * lgdre_2 + intercept
                rms = np.sqrt(np.sum((prices - _predictions) ** 2) / self.lookback)
                error_contribution = (prices[-1] - _predictions[-1]) / rms
                output[i] = 100 * norm.cdf(0.6 * error_contribution) - 50
            elif self.fit_degree == 3:
                _predictions = (
                    reg_coeff_1 * lgdre_1
                    + reg_coeff_2 * lgdre_2
                    + reg_coeff_3 * lgdre_3
                    + intercept
                )
                rms = np.sqrt(np.sum((prices - _predictions) ** 2) / self.lookback)
                error_contribution = (prices[-1] - _predictions[-1]) / rms
                output[i] = 100 * norm.cdf(0.6 * error_contribution) - 50

        if self.transform is not None:
            output = self.feature_engineer.transform(pd.Series(output), self.transform)
            output = np.asarray(output)

        return output
