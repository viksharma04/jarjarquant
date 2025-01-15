"""Indicator class - each indicator in jarjarquant is an instance of the Indicator class.
Each indicator is a specific calculation on market data with an underlying market hypothesis"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
from .data_analyst import DataAnalyst
from .feature_engineer import FeatureEngineer
from .feature_evaluator import FeatureEvaluator


class Indicator:
    """Base class to implement indicators"""

    def __init__(self, ohlcv_df: pd.DataFrame):
        if ohlcv_df is None or ohlcv_df.empty:
            raise ValueError("Please provide a valid OHLCV DataFrame!")

        self.df = ohlcv_df.copy()
        self.indicator_type = None
        self.data_analyst = DataAnalyst()
        self.feature_engineer = FeatureEngineer()
        self.feature_evaluator = FeatureEvaluator()

        self.relative_entropy = None
        self.mutual_information = None
        self.range_iqr_ratio = None
        self.adf_test = None
        self.jb_normality_test = None

    def calculate(self):
        """Placeholder - to be implemented in derived classes

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "Derived classes must implement the calculate method.")

    def indicator_evaluation_report(self, verbose: bool = False, transform: Optional[str] = None, n_bins_to_discretize: Optional[int] = 10, **kwargs):
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

        self.data_analyst.visual_stationary_test(values)
        self.adf_test = self.data_analyst.adf_test(values, verbose)
        self.jb_normality_test = self.data_analyst.jb_normality_test(
            values, verbose)
        self.relative_entropy = self.data_analyst.relative_entropy(
            values, verbose)
        self.range_iqr_ratio = self.data_analyst.range_iqr_ratio(
            values, verbose)
        if self.indicator_type == 'continuous':
            self.mutual_information = self.data_analyst.mutual_information(
                array=values, lag=10, n_bins=n_bins_to_discretize, is_discrete=False, verbose=verbose)
        else:
            self.mutual_information = self.data_analyst.mutual_information(
                array=values, lag=10, n_bins=None, is_discrete=True, verbose=verbose)

        for i in range(1, 11):
            print(f"NMI @ lag {i} = {self.mutual_information[i-1]}")

    # TODO: Implement a robust indicator evaluation report method which creates multiple instances of the indicator and averages the results


class RSI(Indicator):
    """Class to calculate the Relative Strength Index (RSI)"""

    def __init__(self, ohlcv_df: pd.DataFrame, period: int = 14, transform=None):
        super().__init__(ohlcv_df)
        self.period = period
        self.indicator_type = 'continuous'  # continuous or discrete
        self.transform = transform

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

        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)
        return output


class DetrendedRSI(Indicator):
    """Class to calculate the Detrended RSI"""

    def __init__(self, ohlcv_df: pd.DataFrame, short_period: int = 2, long_period: int = 21, regression_length: int = 120, transform=None):
        super().__init__(ohlcv_df)
        self.short_period = short_period
        self.long_period = long_period
        self.regression_length = regression_length
        self.indicator_type = 'continuous'
        self.transform = transform

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

        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)
        return output


class Stochastic(Indicator):
    """Class to calculate the stochastic oscillator"""

    def __init__(self, ohlcv_df, lookback: int = 14, n_smooth: int = 2, transform=None):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.n_smooth = n_smooth
        self.indicator_type = 'continuous'
        self.transform = transform

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

        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)
        return output


class StochasticRSI(Indicator):
    """Class to calculate the Stochastic RSI indicator"""

    def __init__(self, ohlcv_df: pd.DataFrame, rsi_period: int = 14, stochastic_period: int = 14, n_smooth: int = 2, transform=None):
        super().__init__(ohlcv_df)
        self.rsi_period = rsi_period
        self.stochastic_period = stochastic_period
        self.n_smooth = n_smooth
        self.indicator_type = 'continuous'
        self.transform = transform

    def calculate(self) -> np.ndarray:
        rsi = RSI(self.df, self.rsi_period).calculate()
        # Store RSI values in a DataFrame and rename the column to 'Close'
        rsi_df = pd.DataFrame(rsi, columns=['Close'])
        sto_rsi = Stochastic(rsi_df, self.stochastic_period,
                             self.n_smooth).calculate()

        if self.transform is not None:
            sto_rsi = self.feature_engineer.transform(sto_rsi, self.transform)
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

    def __init__(self, ohlcv_df: pd.DataFrame, short_period: int = 5, long_period: int = 20, transform=None):
        super().__init__(ohlcv_df)
        self.short_period = short_period
        self.long_period = long_period
        self.indicator_type = 'continuous'
        self.transform = transform

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
            self.short_period+self.long_period, self.df['High'], self.df['Low'], self.df['Close']).values

        denom = atr_values * \
            np.sqrt((0.5*(self.long_period-1)+self.short_period) -
                    (0.5*(self.short_period-1)))

        norm_diff = (short_ma - long_ma) / denom
        COMPRESSION_FACTOR = 1.5
        mad = 100 * norm.cdf(COMPRESSION_FACTOR*norm_diff) - 50

        # Replace nan values with 0
        mad = np.where(np.isnan(mad), 0, mad)

        if self.transform is not None:
            mad = self.feature_engineer.transform(mad, self.transform)
        return mad


class MACD(Indicator):

    def __init__(self, ohlcv_df: pd.DataFrame, short_period: int = 5, long_period: int = 20, smoothing_factor: int = 2, return_raw_macd: bool = False, transform=None):
        super().__init__(ohlcv_df)
        self.short_period = short_period
        self.long_period = long_period
        self.smoothing_factor = smoothing_factor
        self.indicator_type = 'continuous'
        self.return_raw_macd = return_raw_macd
        self.transform = transform

    def calculate(self) -> np.ndarray:
        close = self.df['Close'].values
        short_ema = pd.Series(close).ewm(
            span=self.short_period, adjust=False).mean().values
        long_ema = pd.Series(close).ewm(
            span=self.long_period, adjust=False).mean().values

        atr_values = self.data_analyst.atr(
            self.short_period+self.long_period, self.df['High'], self.df['Low'], self.df['Close']).values

        denom = atr_values * \
            np.sqrt((0.5*(self.long_period-1)+self.short_period) -
                    (0.5*(self.short_period-1)))

        norm_diff = (short_ema - long_ema) / denom
        COMPRESSION_FACTOR = 1.0
        macd = 100 * norm.cdf(COMPRESSION_FACTOR*norm_diff) - 50

        # Replace nan values with 0
        macd = np.where(np.isnan(macd), 0, macd)

        if self.transform is not None:
            macd = self.feature_engineer.transform(macd, self.transform)

        if self.return_raw_macd:
            return macd
        else:
            signal_line = pd.Series(macd).ewm(
                span=self.smoothing_factor, adjust=False).mean().values
            return macd - signal_line


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

    def __init__(self, ohlcv_df: pd.DataFrame, lookback: int = 21, atr_length: int = 21, transform=None):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.atr_length = atr_length
        self.indicator_type = 'continuous'
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
        Close = self.df['Close']
        Open = self.df['Open']
        Low = self.df['Low']
        High = self.df['High']

        # Compute the natural logarithm of the close prices
        log_close = np.log(Close)

        # Calculate the rolling mean of the log of close prices over the lookback period
        rolling_mean = log_close.ewm(span=self.lookback).mean()

        # Calculate the denominator using the ATR function and adjust by the sqrt of (lookback + 1)
        denom = self.data_analyst.atr(self.atr_length, High, Low, Close, ema=True) * \
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
        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)

        return output


class RegressionTrend(Indicator):

    def __init__(self, ohlcv_df: pd.DataFrame, lookback: int = 21, atr_length_mult: int = 3, degree: int = 1, transform=None):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.degree = degree
        self.atr_length = atr_length_mult * lookback
        self.indicator_type = 'continuous'
        self.transform = transform

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
            self.atr_length, self.df['High'], self.df['Low'], self.df['Close']).values
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

        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)
        return output


class PriceIntensity(Indicator):

    def __init__(self, ohlcv_df: pd.DataFrame, smoothing_factor: int = 2, transform=None):
        super().__init__(ohlcv_df)
        self.smoothing_factor = smoothing_factor
        self.indicator_type = 'continuous'
        self.transform = transform

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
            denom = np.maximum.reduce([high[i]-low[i], high[i] -
                                       close[i-1], close[i-1]-low[i]])
            output[i] = (close[i] - _open[i]) / denom

        # Smooth the Price Intensity values
        output = pd.Series(output).ewm(
            span=self.smoothing_factor, adjust=False).mean().values

        # Normalize the Price Intensity values
        output = 100 * norm.cdf(0.8*np.sqrt(self.smoothing_factor)*output) - 50

        # Replace nan and inf values with 0
        output = np.where(np.isnan(output), 0, output)

        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)
        return output


class ADX(Indicator):

    def __init__(self, ohlcv_df: pd.DataFrame, lookback: int = 14, transform=None):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.indicator_type = 'continuous'
        self.transform = transform

    def calculate(self) -> np.ndarray:
        close = self.df['Close'].values
        high = self.df['High'].values
        low = self.df['Low'].values
        _open = self.df['Open'].values

        n = len(close)
        output = np.full(n, 0.0)

        dms_plus = 0
        dms_minus = 0
        atr_sum = 0

        # Initialize the high and low movement variables using a SMA over the lookback period
        for i in range(1, self.lookback):
            dm_plus = high[i] - high[i-1]
            dm_minus = low[i-1] - low[i]

            if dm_plus >= dm_minus:
                dm_minus = 0
            else:
                dm_plus = 0

            dm_plus = 0 if dm_plus < 0 else dm_plus
            dm_minus = 0 if dm_minus < 0 else dm_minus

            dms_plus += dm_plus
            dms_minus += dm_minus

            # Calculate and cumulate the ATR
            atr = np.maximum.reduce([high[i] - low[i], high[i] - close[i-1],
                                     close[i-1] - low[i]])
            atr_sum += atr

            di_plus = dms_plus / atr_sum if atr_sum != 0 else 0
            di_minus = dms_minus / atr_sum if atr_sum != 0 else 0

            adx = np.abs(di_plus - di_minus) / (di_plus +
                                                di_minus) if di_plus + di_minus != 0 else 0

            output[i] = 100*adx

        adx_sum = 0
        # Secondary initialization to generate ADX values to begin exp smoothing
        for i in range(self.lookback, self.lookback*2):
            dm_plus = high[i] - high[i-1]
            dm_minus = low[i-1] - low[i]

            if dm_plus >= dm_minus:
                dm_minus = 0
            else:
                dm_plus = 0

            dm_plus = 0 if dm_plus < 0 else dm_plus
            dm_minus = 0 if dm_minus < 0 else dm_minus

            # Begin using exp smoothing instead of SMA
            dms_plus = (self.lookback-1)/self.lookback * dms_plus + dm_plus
            dms_minus = (self.lookback-1)/self.lookback * dms_minus + dm_minus

            # Calculate and cumulate the ATR
            atr = np.maximum.reduce([high[i] - low[i], high[i] - close[i-1],
                                     close[i-1] - low[i]])
            atr_sum = (self.lookback-1)/self.lookback * atr_sum + atr

            di_plus = dms_plus / atr_sum if atr_sum != 0 else 0
            di_minus = dms_minus / atr_sum if atr_sum != 0 else 0

            adx = np.abs(di_plus - di_minus) / (di_plus +
                                                di_minus) if di_plus + di_minus != 0 else 0

            adx_sum += adx

            output[i] = 100*adx

        # Secondary initialization complete - use adx/lookback as the first value
        adx_sum /= self.lookback

        # Final loop to calculate rest of the values
        for i in range(self.lookback*2, n):
            dm_plus = high[i] - high[i-1]
            dm_minus = low[i-1] - low[i]

            if dm_plus >= dm_minus:
                dm_minus = 0
            else:
                dm_plus = 0

            dm_plus = 0 if dm_plus < 0 else dm_plus
            dm_minus = 0 if dm_minus < 0 else dm_minus

            # Begin using exp smoothing instead of SMA
            dms_plus = (self.lookback-1)/self.lookback * dms_plus + dm_plus
            dms_minus = (self.lookback-1)/self.lookback * dms_minus + dm_minus

            # Calculate and cumulate the ATR
            atr = np.maximum.reduce([high[i] - low[i], high[i] - close[i-1],
                                     close[i-1] - low[i]])
            atr_sum = (self.lookback-1)/self.lookback * atr_sum + atr

            di_plus = dms_plus / atr_sum if atr_sum != 0 else 0
            di_minus = dms_minus / atr_sum if atr_sum != 0 else 0

            adx = np.abs(di_plus - di_minus) / (di_plus +
                                                di_minus) if di_plus + di_minus != 0 else 0

            adx_sum = (self.lookback-1)/self.lookback * adx_sum + adx

            output[i] = 100*adx_sum

        # Replace any nan values with 0
        output = np.where(np.isnan(output), 0, output)

        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)
        return output


class Aroon(Indicator):
    """Class to calculate the Aroon indicator"""

    def __init__(self, ohlcv_df: pd.DataFrame, lookback: int = 25, transform=None):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.indicator_type = 'continuous'
        self.transform = transform

    def calculate(self) -> np.ndarray:
        high = self.df['High'].values
        low = self.df['Low'].values
        n = len(high)
        output = np.full(n, 0.0)

        for i in range(self.lookback, n):
            high_max = np.argmax(high[i-self.lookback:i])
            low_min = np.argmin(low[i-self.lookback:i])

            aroon_up = 100 * (self.lookback - (i - high_max)) / self.lookback
            aroon_down = 100 * (self.lookback - (i - low_min)) / self.lookback

            output[i] = aroon_up - aroon_down

        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)
        return output


class RegressionTrendDeviation(Indicator):

    def __init__(self, ohlcv_df: pd.DataFrame, lookback: int = 14, fit_degree: int = 1, transform=None):
        super().__init__(ohlcv_df)
        self.lookback = lookback
        self.fit_degree = fit_degree
        self.indicator_type = 'continuous'
        self.transform = transform

    def calculate(self) -> np.ndarray:

        if self.lookback < self.fit_degree:
            self.lookback = self.fit_degree + 2
            print(f"Warning: Lookback period adjusted to {
                  self.lookback} to accommodate fit_degree of {self.fit_degree}.")

        close = self.df['Close'].values
        n = len(close)

        output = np.full(n, 0.0)

        # Calculate the Legendre polynomials for 1, 2, and 3 degrees
        lgdre_1 = self.data_analyst.compute_legendre_coefficients(
            self.lookback, 1)
        lgdre_2 = self.data_analyst.compute_legendre_coefficients(
            self.lookback, 2)
        lgdre_3 = self.data_analyst.compute_legendre_coefficients(
            self.lookback, 3)

        # Loop over data starting from lookback-1
        for i in range(self.lookback-1, n):
            prices = np.log(close[i-self.lookback+1:i+1])

            reg_coeff_1 = self.data_analyst.calculate_regression_coefficient(
                prices, lgdre_1)
            reg_coeff_2 = self.data_analyst.calculate_regression_coefficient(
                prices, lgdre_2)
            reg_coeff_3 = self.data_analyst.calculate_regression_coefficient(
                prices, lgdre_3)

            intercept = sum(prices)/self.lookback

            if self.fit_degree == 1:
                _predictions = reg_coeff_1 * lgdre_1 + intercept
                rms = np.sqrt(np.sum((prices - _predictions)**2)/self.lookback)
                error_contribution = (prices[-1] - _predictions[-1])/rms
                output[i] = 100 * norm.cdf(0.6 * error_contribution) - 50
            elif self.fit_degree == 2:
                _predictions = reg_coeff_1*lgdre_1 + reg_coeff_2*lgdre_2 + intercept
                rms = np.sqrt(np.sum((prices - _predictions)**2)/self.lookback)
                error_contribution = (prices[-1] - _predictions[-1])/rms
                output[i] = 100 * norm.cdf(0.6 * error_contribution) - 50
            elif self.fit_degree == 3:
                _predictions = reg_coeff_1*lgdre_1 + reg_coeff_2 * \
                    lgdre_2 + reg_coeff_3*lgdre_3 + intercept
                rms = np.sqrt(np.sum((prices - _predictions)**2)/self.lookback)
                error_contribution = (prices[-1] - _predictions[-1])/rms
                output[i] = 100 * norm.cdf(0.6 * error_contribution) - 50

        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)
        return output


class PriceChangeOscillator(Indicator):

    def __init__(self, ohlcv_df: pd.DataFrame, short_lookback: int = 5, long_lookback_multiplier: int = 5, transform=None):
        super().__init__(ohlcv_df)
        self.short_lookback = short_lookback
        self.long_lookback_multiplier = long_lookback_multiplier
        self.indicator_type = 'continuous'
        self.transform = transform

    def calculate(self) -> np.ndarray:

        close = self.df['Close'].values
        prices = np.log(close)
        n = len(close)

        output = np.full(n, 0.0)
        long_lookback = self.long_lookback_multiplier * self.short_lookback

        # Calculate ATR over the long lookback period
        atr = self.data_analyst.atr(
            long_lookback, self.df['High'], self.df['Low'], self.df['Close']).values

        for i in range(long_lookback, n):
            # Calculate the short-term and long-term mean
            short_ma = np.mean(
                prices[i-self.short_lookback+1:i] - prices[i-self.short_lookback:i-1])
            long_ma = np.mean(
                prices[i-long_lookback+1:i] - prices[i-long_lookback:i-1])

            const = 0.36 + (1/self.short_lookback) + 0.7 * \
                np.log(0.5*self.long_lookback_multiplier) / 1.609
            denom = atr[i] * const
            denom = np.maximum(denom, 1e-8)

            raw = (short_ma - long_ma) / denom
            output[i] = 100 * norm.cdf(4 * raw) - 50

        # Replace nan and inf values with 0
        output = np.where(np.isnan(output), 0, output)

        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)
        return output


class ChaikinMoneyFlow(Indicator):

    def __init__(self, ohlcv_df: pd.DataFrame, smoothing_lookback: int = 21, volume_lookback: int = 21, return_cmf: bool = False, transform=None):
        super().__init__(ohlcv_df)
        self.smoothing_lookback = smoothing_lookback
        self.volume_lookback = volume_lookback
        self.return_cmf = return_cmf
        self.transform = transform
        self.indicator_type = 'continuous'

    def calculate(self) -> np.ndarray:

        Close = self.df['Close'].values
        High = self.df['High'].values
        Low = self.df['Low'].values
        Volume = self.df['Volume'].values

        output = np.full(len(Close), 0.0)

        # Look for first bar with non-zero volume
        first_non_zero_vol = np.argmax(Volume > 0)

        # Calculate the intraday intensity of each bar after the first non-zero volume bar
        # Handle case if high and low are equal
        for i in range(first_non_zero_vol, len(Close)):
            if High[i] == Low[i]:
                output[i] = 0
            else:
                output[i] = 100 * ((2 * Close[i] - High[i] -
                                   Low[i]) / (High[i] - Low[i])) * Volume[i]

        # Calculate the SMA of output using the smoothing lookback period
        output = pd.Series(output).rolling(
            window=self.smoothing_lookback).mean().values

        if self.return_cmf:
            # Calculate the SMA of Volume values using the volume lookback period
            sma_volume = pd.Series(Volume).rolling(
                window=self.volume_lookback).mean().values

            # Calculate the Chaikin Money Flow by dividing the SMA of output by the SMA of Volume and handling division by zero
            output = np.where(sma_volume != 0, output / sma_volume, 0)

        else:
            # Finally smooth the indicator values by dividing the output by the EMA of volume calculated using n_smooth
            ema_volume = pd.Series(Volume).ewm(
                span=self.volume_lookback).mean().values

            # Normalized output
            output = np.where(ema_volume != 0, output / ema_volume, 0)

        # Replace nan and inf values with 0
        output = np.where(np.isnan(output), 0, output)

        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)

        return output
