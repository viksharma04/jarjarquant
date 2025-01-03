"""Indicator class - each indicator in jarjarquant is an instance of the Indicator class.
Each indicator is a specific calculation on market data with an underlying market hypothesis"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
from .data_analyst import DataAnalyst
from .feature_engineer import FeatureEngineer


class Indicator:
    """Base class to implement indicators"""

    def __init__(self, ohlcv_df: pd.DataFrame):
        if ohlcv_df is None or ohlcv_df.empty:
            raise ValueError("Please provide a valid OHLCV DataFrame!")

        self.df = ohlcv_df.copy()
        self.indicator_type = None
        self.data_analyst = DataAnalyst()
        self.feature_engineer = FeatureEngineer()

    def calculate(self):
        """Placeholder - to be implemented in derived classes

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "Derived classes must implement the calculate method.")

    def indicator_evaluation_report(self, transform: Optional[str] = None, n_bins_to_discretize: Optional[int] = 10, **kwargs):
        """Runs a set of statistical tests to examine various properties of the 
        indicator series, such as stationarity, normality, entropy, mutual
        information, etc.

        Args:
            transform (str, optional): Acceptable values: 'log', 'root', 'tanh'. Transformation to apply to the indicator values.
            n_bins_to_discretize (int, optional): Number of bins to use if indicator
            is continuous. Used for mutual information calculation. Defaults to 10.
        """
        values = self.calculate()
        if transform is not None:
            values = self.feature_engineer.transform(
                values, transform, **kwargs)

        if self.indicator_type == 'continuous':
            d_values = self.data_analyst.discretize_array(
                values, n_bins_to_discretize)
        else:
            d_values = values

        self.data_analyst.visual_stationary_test(values)
        print("/n")
        self.data_analyst.adf_test(values)
        print("/n")
        self.data_analyst.jb_normality_test(values, plot_dist=True)
        print("/n")
        print(f"Relative entropy = {
              self.data_analyst.relative_entropy(values, True)}")
        print(f"Range IQR Ratio = {
              self.data_analyst.range_iqr_ratio(values, True)}")
        print(f"Mutual information at lag 1 = {
              self.data_analyst.mutual_information(d_values, 1)}")


class RSI(Indicator):
    """Class to calculate the Relative Strength Index (RSI)"""

    def __init__(self, ohlcv_df: pd.DataFrame, period: int = 14):
        super().__init__(ohlcv_df)
        self.period = period
        self.indicator_type = 'continuous'  # continuous or discrete

    def calculate(self) -> np.ndarray:
        close = self.df['Close'].values
        n = len(close)
        front_bad = self.period
        output = np.full(n, 50.0)  # Default RSI of 50.0 for undefined values

        # Calculate initial sums for up and down movements
        deltas = np.diff(close)
        ups = np.where(deltas > 0, deltas, 0)
        downs = np.where(deltas < 0, -deltas, 0)

        # Initialize the up and down sums
        upsum = np.sum(ups[:self.period - 1]) / \
            (self.period - 1) + np.finfo(float).eps
        dnsum = np.sum(downs[:self.period - 1]) / \
            (self.period - 1) + np.finfo(float).eps

        # Compute RSI values after initial self.period period

        for i in range(front_bad, n):
            diff = deltas[i - 1]
            if diff > 0:
                upsum = ((self.period - 1) * upsum + diff) / self.period
                dnsum *= (self.period - 1) / self.period
            else:
                dnsum = ((self.period - 1) * dnsum - diff) / self.period
                upsum *= (self.period - 1) / self.period

            # RSI calculation
            if upsum + dnsum == 0:
                output[i] = 50.0  # Default RSI value when both sums are zero
            else:
                output[i] = 100.0 * upsum / (upsum + dnsum)

        return output


class DetrendedRSI(Indicator):
    """Class to calculate the Detrended RSI"""

    def __init__(self, ohlcv_df: pd.DataFrame, short_period: int = 2, long_period: int = 21, regression_length: int = 120):
        super().__init__(ohlcv_df)
        self.short_period = short_period
        self.long_period = long_period
        self.regression_length = regression_length
        self.indicator_type = 'continuous'

    def calculate(self) -> np.ndarray:
        close = self.df['Close'].values
        n = len(close)
        output = np.full(n, 0.0)

        short_rsi = RSI(self.df, self.short_period).calculate()
        # Apply inverse logistic transformation to the short RSI values
        # Look at pg 103 of Statistically Sound Indicators
        short_rsi = -10*np.log((2/(1+0.00999*(2*short_rsi-100))) - 1)

        # Long RSI
        long_rsi = RSI(self.df, self.long_period).calculate()

        for i in range(self.regression_length+self.long_period-1, n):
            x = long_rsi[i-self.regression_length:i]
            y = short_rsi[i-self.regression_length:i]

            x_mean = np.mean(x)
            y_mean = np.mean(y)

            x_diff = x - x_mean
            y_diff = y - y_mean

            coef = np.dot(x_diff, y_diff) / (np.dot(x_diff, x_diff) + 1e-10)

            output[i] = (y[-1] - y_mean) - coef * (x[-1] - x_mean)

        return output


class Stochastic(Indicator):
    """Class to calculate the stochastic oscillator"""

    def __init__(self, ohlcv_df, lookback: int = 14, n_smooth: int = 2):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.n_smooth = n_smooth
        self.indicator_type = 'continuous'

    def calculate(self) -> np.ndarray:
        close = self.df['Close'].values
        n = len(close)
        output = np.full(n, 50.0)

        # Calculate rolling max and min for Close values
        high_max = pd.Series(close).rolling(window=self.lookback).max().values
        low_min = pd.Series(close).rolling(window=self.lookback).min().values

        for i in range(self.lookback, n):
            if high_max[i] == low_min[i]:
                output[i] = 50.0
            else:
                sto_0 = 100 * (close[i] - low_min[i]) / \
                    (high_max[i] - low_min[i])
                if self.n_smooth == 0:
                    output[i] = sto_0
                elif self.n_smooth == 1:
                    if i == self.lookback:
                        output[i] = sto_0
                    else:
                        output[i] = 0.33333 * sto_0 + 0.66667 * output[i - 1]
                else:
                    if i < self.lookback + 1:
                        output[i] = sto_0
                    elif i == self.lookback + 1:
                        output[i] = 0.33333 * sto_0 + 0.66667 * output[i - 1]
                    else:
                        sto_1 = 0.33333 * sto_0 + 0.66667 * output[i - 1]
                        output[i] = 0.33333 * sto_1 + 0.66667 * output[i - 2]

        return output


class StochasticRSI(Indicator):
    """Class to calculate the Stochastic RSI indicator"""

    def __init__(self, ohlcv_df: pd.DataFrame, rsi_period: int = 14, stochastic_period: int = 14, n_smooth: int = 2):
        super().__init__(ohlcv_df)
        self.rsi_period = rsi_period
        self.stochastic_period = stochastic_period
        self.n_smooth = n_smooth
        self.indicator_type = 'continuous'

    def calculate(self) -> np.ndarray:
        rsi = RSI(self.df, self.rsi_period).calculate()
        # Store RSI values in a DataFrame and rename the column to 'Close'
        rsi_df = pd.DataFrame(rsi, columns=['Close'])
        sto_rsi = Stochastic(rsi_df, self.stochastic_period,
                             self.n_smooth).calculate()

        return sto_rsi


class MovingAverageDifference(Indicator):
    """
    A class to calculate the Moving Average Difference (MAD) indicator.
    The MAD indicator is a normalized difference between a short-term and a long-term moving average,
    adjusted by the Average True Range (ATR) to account for volatility.
    Attributes:
        ohlcv_df (pd.DataFrame): DataFrame containing OHLCV (Open, High, Low, Close, Volume) data.
        short_period (int): The period for the short-term moving average. Default is 5.
        long_period (int): The period for the long-term moving average. Default is 20.
        indicator_type (str): Type of the indicator, set to 'continuous'.
    Methods:
        calculate() -> np.ndarray:
            Calculates the Moving Average Difference (MAD) indicator.
            Returns:
                np.ndarray: The calculated MAD values.
    """

    def __init__(self, ohlcv_df: pd.DataFrame, short_period: int = 5, long_period: int = 20):
        super().__init__(ohlcv_df)
        self.short_period = short_period
        self.long_period = long_period
        self.indicator_type = 'continuous'

    def calculate(self) -> np.ndarray:
        close = self.df['Close'].values
        # Calculate the short-term MA
        short_ma = pd.Series(close).rolling(
            window=self.short_period).mean().values

        # Calculate the long-term MA
        long_ma = pd.Series(close).rolling(window=self.long_period).mean()
        # Lag long_ma by short_period
        long_ma = long_ma.shift(self.short_period).values

        # See pg 116 eq. 4.7 and 4.8 of Statistically Sound Indicators
        atr_values = self.data_analyst.atr(
            self.short_period+self.long_period, self.df['Open'], self.df['High'], self.df['Low'], self.df['Close']).values

        denom = atr_values * \
            np.sqrt((0.5*(self.long_period-1)+self.short_period) -
                    (0.5*(self.short_period-1)))

        norm_diff = (short_ma - long_ma) / denom
        COMPRESSION_FACTOR = 1.5
        mad = 100 * norm.cdf(COMPRESSION_FACTOR*norm_diff) - 50

        # Replace nan values with 0
        mad = np.where(np.isnan(mad), 0, mad)

        return mad


class MACD(Indicator):

    def __init__(self, ohlcv_df: pd.DataFrame, short_period: int = 5, long_period: int = 20, smoothing_factor: int = 2, return_raw_macd: bool = False):
        super().__init__(ohlcv_df)
        self.short_period = short_period
        self.long_period = long_period
        self.smoothing_factor = smoothing_factor
        self.indicator_type = 'continuous'
        self.return_raw_macd = return_raw_macd

    def calculate(self) -> np.ndarray:
        close = self.df['Close'].values
        short_ema = pd.Series(close).ewm(
            span=self.short_period, adjust=False).mean().values
        long_ema = pd.Series(close).ewm(
            span=self.long_period, adjust=False).mean().values

        atr_values = self.data_analyst.atr(
            self.short_period+self.long_period, self.df['Open'], self.df['High'], self.df['Low'], self.df['Close']).values

        denom = atr_values * \
            np.sqrt((0.5*(self.long_period-1)+self.short_period) -
                    (0.5*(self.short_period-1)))

        norm_diff = (short_ema - long_ema) / denom
        COMPRESSION_FACTOR = 1.0
        macd = 100 * norm.cdf(COMPRESSION_FACTOR*norm_diff) - 50

        # Replace nan values with 0
        macd = np.where(np.isnan(macd), 0, macd)

        if self.return_raw_macd:
            return macd
        else:
            signal_line = pd.Series(macd).ewm(
                span=self.smoothing_factor, adjust=False).mean().values
            return macd - signal_line


class CMMA(Indicator):
    """Cumulative Moving Mean Average (CMMA) Indicator
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

    def __init__(self, ohlcv_df: pd.DataFrame, lookback: int = 21, atr_length: int = 21):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.atr_length = atr_length
        self.indicator_type = 'continuous'

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
        Close = self.df['Close']
        Open = self.df['Open']
        Low = self.df['Low']
        High = self.df['High']

        # Compute the natural logarithm of the close prices
        log_close = np.log(Close)

        # Calculate the rolling mean of the log of close prices over the lookback period
        rolling_mean = log_close.shift(1).ewm(span=self.lookback).mean()

        # Calculate the denominator using the ATR function and adjust by the sqrt of (lookback + 1)
        denom = self.data_analyst.atr(self.atr_length, Open, High, Low, Close) * \
            np.sqrt(self.lookback + 1)

        # Normalize the output by dividing the difference between log_close and rolling_mean by denom
        # If denom is 0 or negative, set the normalized output to 0
        normalized_output = np.where(
            denom > 0,
            (log_close - rolling_mean) / denom,
            0
        )

        # Transform the normalized output using the cumulative distribution function (CDF) of the normal distribution
        output = 100 * norm.cdf(normalized_output) - 50

        # Return the final CMMA values as a pandas Series with the same index as the input DataFrame
        return output


class RegressionTrend(Indicator):

    def __init__(self, ohlcv_df: pd.DataFrame, lookback: int = 21, atr_length_mult: int = 3, degree: int = 1):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.degree = degree
        self.atr_length = atr_length_mult * lookback
        self.indicator_type = 'continuous'

    def calculate(self) -> np.ndarray:
        close = self.df['Close'].values
        n = len(close)
        output = np.full(n, 0.0)

        if self.lookback > n:
            raise ValueError(
                "Lookback period is greater than the number of data points!")

        # Calculate the Legendre polynomials
        lgdre = self.data_analyst.compute_legendre_coefficients(
            self.lookback, self.degree)
        atr = self.data_analyst.atr(
            self.atr_length, self.df['Open'], self.df['High'], self.df['Low'], self.df['Close']).values
        COMPRESSION_FACTOR = 1.5

        for i in range(self.atr_length+1, n):
            prices = np.log(close[i-self.lookback:i])

            # Calculate the regression coefficient
            reg_coeff = self.data_analyst.calculate_regression_coefficient(
                prices, lgdre)

            # Contextualize the regression coefficient using long term atr
            reg_coeff = (2*reg_coeff)/(atr[i]*(self.lookback-1))

            # Calculate the R^2 value over the lookback period
            y_diff = prices - np.mean(prices)
            y_diff_sq = np.sum(y_diff**2)
            predictions = reg_coeff * lgdre
            residuals = y_diff - predictions
            residuals_sq = np.sum(residuals**2)

            r_squared = 1 - residuals_sq/y_diff_sq

            # Multiply the regression coefficient by the R^2 value to devalue weak trends
            norm_coeff = reg_coeff * r_squared
            output[i] = 100 * norm.cdf(COMPRESSION_FACTOR*norm_coeff) - 50

        # Replace nan and inf values with 0
        output = np.where(np.isnan(output), 0, output)

        return output


class PriceIntensity(Indicator):

    def __init__(self, ohlcv_df: pd.DataFrame, smoothing_factor: int = 2):
        super().__init__(ohlcv_df)
        self.smoothing_factor = smoothing_factor
        self.indicator_type = 'continuous'

    def calculate(self) -> np.ndarray:
        close = self.df['Close'].values
        high = self.df['High'].values
        low = self.df['Low'].values
        _open = self.df['Open'].values

        n = len(close)
        output = np.full(n, 0.0)

        # Special case for the first value
        output[0] = (close[0] - _open[0]) / (high[0] - low[0])

        # Calculate Raw Price Intensity
        for i in range(1, n):
            denom = np.max(high[i]-low[i], high[i] -
                           close[i-1], close[i-1]-low[i])
            output[i] = (close[i] - _open[i]) / denom

        # Smooth the Price Intensity values
        output = pd.Series(output).ewm(
            span=self.smoothing_factor).mean().values

        # Normalize the Price Intensity values
        output = 100 * norm.cdf(0.8*np.sqrt(self.smoothing_factor)*output) - 50

        # Replace nan and inf values with 0
        output = np.where(np.isnan(output), 0, output)

        return output
