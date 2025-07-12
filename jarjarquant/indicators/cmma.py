import numpy as np
import pandas as pd
from scipy.stats import norm

from jarjarquant.indicators.base import Indicator
from jarjarquant.indicators.registry import register_indicator, IndicatorType


@register_indicator(IndicatorType.CMMA)
class CMMA(Indicator):
    """Close Minus Moving Average (CMMA) Indicator
    This class calculates the CMMA indicator, which is a normalized measure of the
    logarithmic difference between the closing prices and their rolling mean, adjusted
    by the Average True Range (ATR).
    Attributes:
        ohlcv_df (pd.DataFrame): A pandas DataFrame containing 'Close', 'Open', 'Low',
                                 and 'High' columns.
        lookback (int): The number of periods for calculating the rolling mean. Default is 21.
        atr_length (int): The length of the rolling window for the ATR calculation. Default is 21.
        indicator_type (str): The type of the indicator, set to 'continuous'.
    Methods:
        calculate() -> np.ndarray:
            Calculate the CMMA values.
                np.ndarray: A numpy array with the CMMA values."""

    def __init__(
        self,
        ohlcv_df: pd.DataFrame,
        lookback: int = 21,
        atr_length: int = 21,
        transform=None,
    ):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.atr_length = atr_length
        self.indicator_type = "continuous"
        self.transform = transform

    def calculate(self) -> np.ndarray:
        """
        Calculate the Cumulative Moving Mean Average (CMMA) indicator.

        Parameters:
        lookback: The number of periods for calculating the rolling mean.
        atr_length: The length of the rolling window for the ATR calculation.
        df: A pandas DataFrame containing 'Close', 'Open', 'Low', and 'High' columns.

        Returns:
        A pandas Series with the CMMA values.
        """

        # Extract the relevant columns from the DataFrame
        Close = self.df["Close"]
        Low = self.df["Low"]
        High = self.df["High"]

        # Compute the natural logarithm of the close prices
        log_close = np.log(Close)

        # Calculate the rolling mean of the log of close prices over the lookback period
        rolling_mean = pd.Series(log_close).ewm(span=self.lookback).mean()

        # Calculate the denominator using the ATR function and adjust by the sqrt of (lookback + 1)
        denom = self.data_analyst.atr(
            self.atr_length, High, Low, Close, ema=True
        ) * np.sqrt(self.lookback + 1)

        # Normalize the output by dividing the difference between log_close and rolling_mean by denom
        # If denom is 0 or negative, set the normalized output to 0
        normalized_output = np.where(denom > 0, (log_close - rolling_mean) / denom, 0)

        # Transform the normalized output using the cumulative distribution function (CDF) of the normal distribution
        output = 100 * norm.cdf(normalized_output) - 50

        # Return the final CMMA values as a pandas Series with the same index as the input DataFrame
        if self.transform is not None:
            output = self.feature_engineer.transform(pd.Series(output), self.transform)
            output = np.asarray(output)

        return output
