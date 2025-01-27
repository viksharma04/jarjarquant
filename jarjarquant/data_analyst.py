"""The data analyst specializes in performing and contextualizing common statistical tests on one or many data series"""
import pandas as pd
import numpy as np
from scipy.special import legendre
from statsmodels.tsa.stattools import adfuller
from scipy.stats import jarque_bera, norm
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt


class DataAnalyst:

    def __init__(self) -> None:
        pass

    @staticmethod
    def visual_stationary_test(values: np.ndarray):
        """Plots a line graph of calculated values for any class that extends indicator. Draws a top and bottom horizontal threshold above/below which 15% of observations lie

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        plt.plot(values, label='Indicator Value', color='blue')

        # Calculate thresholds
        upper_threshold = np.quantile(values, 0.85)
        lower_threshold = np.quantile(values, 0.15)

        # Plot thresholds
        plt.axhline(y=upper_threshold, color='red',
                    linestyle='--', label='85th Percentile')
        plt.axhline(y=lower_threshold, color='green',
                    linestyle='--', label='15th Percentile')

        plt.title('Visual Stationary Test')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def adf_test(values: np.ndarray, verbose: bool = False):
        """
        Performs the Augmented Dickey-Fuller test to check if the time series is stationary.

        Returns:
            None
        """
        result = adfuller(values)
        test_statistic, p_value, _, _, critical_values, _ = result

        # Print test results
        if verbose:
            print("Augmented Dickey-Fuller Test Results:")
            print(f"Test Statistic: {test_statistic}")
            print(f"P-Value: {p_value}")
            print("Critical Values:")
            for key, value in critical_values.items():
                print(f"\t{key}: {value}")

        # Interpretation
        if p_value < 0.05:
            print("STATIONARY @ 5% CONFIDENCE LEVEL")
        else:
            print("NON-STATIONARY @ 5% CONFIDENCE LEVEL")

        if test_statistic < critical_values['1%']:
            print("STRONG EVIDENCE OF STATIONARITY")
        else:
            print("WEAK EVIDENCE OF STATIONARITY")

        print("----------------------------------------")

        if p_value < 0.05:
            return "passed"
        else:
            return "failed"

    @staticmethod
    def jb_normality_test(values: np.ndarray, verbose=False):
        """
        Performs the Jarque-Bera test to check if the time series is normally distributed.

        Args:
            plot_dist (bool): If True, plots a histogram of the indicator values with a standard normal density curve.

        Returns:
            None
        """
        jb_statistic, p_value = jarque_bera(values)

        # Print test results
        if verbose:
            print("Jarque-Bera Test Results:")
            print(f"JB Statistic: {jb_statistic}")
            print(f"P-Value: {p_value}")

        # Interpretation
        if p_value < 0.05:
            print("NOT NORMAL @ 5% CONFIDENCE LEVEL")
        else:
            print("NORMAL @ 5% CONFIDENCE LEVEL")

        # Plot distribution if requested
        if verbose:
            plt.figure(figsize=(10, 6))
            plt.hist(values, bins=30, density=True, color='blue',
                     alpha=0.6, label='Indicator Value Distribution')

            # Plot standard normal density curve
            x = np.linspace(values.min(), values.max(), 1000)
            plt.plot(x, norm.pdf(x, values.mean(), values.std()),
                     color='red', linestyle='--', label='Normal Distribution')

            plt.title('Distribution of Indicator Values with Normal Curve')
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            plt.show()
            print("\n")

        print("----------------------------------------")

        if p_value < 0.05:
            return "failed"
        else:
            return "passed"

    @staticmethod
    def relative_entropy(values: np.ndarray, verbose: bool = False) -> np.float64:
        # Convert values to numpy array for efficient computation
        values = values
        n = len(values)

        # Determine the number of bins based on sample size
        if n >= 10000:
            nbins = 20
        elif n >= 1000:
            nbins = 10
        elif n >= 100:
            nbins = 5
        else:
            nbins = 3

        # Compute histogram (bin counts and bin edges)
        counts, _ = np.histogram(values, bins=nbins, density=False)

        # Calculate relative entropy
        probabilities = counts / n  # Normalize counts to probabilities
        # Exclude zero probabilities
        nonzero_probs = probabilities[probabilities > 0]
        entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))

        # Normalize entropy by log of number of bins
        normalized_entropy = entropy / np.log(nbins)

        if verbose:
            print("The underlying idea for relative entropy is that valuable discriminatory information is nearly as likely to lie within clumps as it is in broad, thin regions. If the indicator's values lie within a few concentrated regions surrounded by broad swaths of emptiness, most models will focus on the wide spreads of the range while putting little effort into studying what's going on inside the clumps. \n The entropy of an indicator is an upper limit on the amount of infornmation it can carry. Anything above 0.8 is plenty, an even somewhat lower is usually fine. Anything below 0.5 is concerning and below 0.2 is very concerning. \n ---------------------------------------- \n")

        print(f"Relative Entropy: {normalized_entropy}")
        if normalized_entropy < 0.2:
            print("VERY CONCERNING")
        elif normalized_entropy < 0.5:
            print("CONCERNING")
        elif normalized_entropy < 0.8:
            print("FINE")
        elif normalized_entropy >= 0.8:
            print("EXCELLENT")

        print("----------------------------------------")

        return normalized_entropy

    @staticmethod
    def range_iqr_ratio(values: np.ndarray, verbose: bool = False) -> np.float64:
        """Returns the range/interquartile range ratio for an indicator

        Args:
            values (np.ndarray): Indicator time series

        Returns:
            np.float64: range iqr ratios
        """
        range_iqr_ratio = (np.max(values) - np.min(values)) / \
            (np.quantile(values, 0.75) - np.quantile(values, 0.25))

        print(range_iqr_ratio)
        if verbose:
            print("Presence of outliers can greatly reduce the performance of an algorithm. The most obvious reason is that the presence of an outlier reduces entropy, causing the 'normal' observations to form a compact cluster and hence reducing the information carrying capacity of the indicator. \n Ratios of 2 and 3 are reasonable and upto 5 is usually not excessive. But iof the indicator has a range IQR ratio of more than 5, the tails should be tamed. \n ----------------------------- \n")
        else:
            if range_iqr_ratio <= 3:
                print("GREAT DISTRIBUTION - MINIMAL OUTLIERS")
            elif range_iqr_ratio <= 5:
                print("PASSABLE DISTRIBUTION - SOME OUTLIERS - INSPECT VISUALLY")
            elif range_iqr_ratio > 5:
                print("CONCERNING AMOUNT OF OUTLIER - CONSIDER TRANSFORMATIONS")

        print("----------------------------------------")

        return range_iqr_ratio

    @staticmethod
    def mutual_information(array: np.ndarray, lag: int, n_bins: int = None, is_discrete: bool = False, verbose: bool = False) -> np.ndarray:
        """
        Calculates the mutual information of the array with a specified time lag.

        Args:
            array (np.ndarray): A numpy array representing the time series data.
            lag (int): NMI to calculated upto lag.

        Returns:
            float: The mutual information score.
        """
        nmi_scores = np.full(lag, np.nan)\

        if not is_discrete:
            n = len(array)
            if n_bins is None:
                #  Determine the number of bins based on sample size
                if n >= 10000:
                    nbins = 20
                elif n >= 1000:
                    nbins = 10
                elif n >= 100:
                    nbins = 5
                else:
                    nbins = 3
            array = DataAnalyst.discretize_array(array, n_bins=n_bins)

        for i in range(1, lag + 1):
            # Create lagged array
            if i >= len(array):
                raise ValueError(
                    "Lag is greater than or equal to the length of the array.")

            array_lagged = array[:-i]
            array_future = array[i:]

            # Calculate mutual information
            nmi_scores[i -
                       1] = normalized_mutual_info_score(array_lagged, array_future)

        return nmi_scores

    @staticmethod
    def discretize_array(array: np.ndarray, n_bins: int) -> np.ndarray:
        """
        Discretizes the array into n_bins using quantile-based binning.

        Args:
            array (np.ndarray): A numpy array representing the time series data.
            n_bins (int): Number of bins to discretize into.

        Returns:
            np.ndarray: An array of discretized values.
        """
        # Discretize the array into n_bins using np.percentile for quantile-based binning
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(array, percentiles)
        discretized = np.digitize(array, bins, right=True) - 1

        # Ensure the discretized values are within the range [0, n_bins - 1]
        discretized[discretized == n_bins] = n_bins - 1

        return discretized

    @staticmethod
    def mutual_information_with_time():
        pass

    @staticmethod
    def atr(atr_length: int, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, ema: bool = False) -> pd.Series:
        """
            Parameters:
            atr_length (int): The period over which to calculate the ATR.
            open_series (pd.Series): Series of open prices.
            high_series (pd.Series): Series of high prices.
            low_series (pd.Series): Series of low prices.
            close_series (pd.Series): Series of close prices.

            Returns:
            pd.Series: The ATR values for the given period.
        """
        # Calculate the true range for each row in the data
        true_ranges = np.maximum(
            high_series - low_series,
            np.maximum(
                np.abs(high_series - close_series.shift(1)),
                np.abs(low_series - close_series.shift(1))
            )
        )
        # Compute the rolling mean of the true ranges over the specified atr_length
        if ema:
            return true_ranges.ewm(span=atr_length, adjust=False).mean()
        else:
            return true_ranges.rolling(window=atr_length).mean()

    @staticmethod
    def compute_legendre_coefficients(lookback, degree):
        """
        Compute the Legendre polynomial coefficients over a given lookback window.

        Parameters:
        - lookback (int): The number of points in the window.
        - degree (int): The degree of the Legendre polynomial (1, 2, or 3).

        Returns:
        - numpy.ndarray: The coefficients of the Legendre polynomial for the given degree.
        """
        if degree not in [1, 2, 3]:
            raise ValueError("Only degrees 1, 2, or 3 are supported.")

        # Normalize x-values to the range [-1, 1]
        x = np.linspace(-1, 1, lookback)

        # Generate the Legendre polynomial of the specified degree
        legendre_poly = legendre(degree)

        # Evaluate the polynomial at the normalized x-values
        coefficients = legendre_poly(x)

        return coefficients

    @staticmethod
    def calculate_regression_coefficient(prices: np.ndarray, legendre_coeffs: np.ndarray) -> float:
        """
        Calculate the linear regression coefficient using the dot product method.

        Parameters:
        - prices (numpy.ndarray): The series of prices (e.g., log prices).
        - legendre_coeffs (numpy.ndarray): The precomputed Legendre coefficients.

        Returns:
        - float: The slope of the least squares line.
        """
        norm_coeffs_squared = np.sum(legendre_coeffs ** 2)
        slope = np.dot(legendre_coeffs, prices) / norm_coeffs_squared
        return slope

    @staticmethod
    def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
        # Find the index of each previous day in the close series
        df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
        # If a day does not fit in the series, exclude it
        df0 = df0[df0 > 0]
        # Create a series which stores the previous date for each date in the close series
        df0 = pd.Series(
            close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:])
        # Calculate the return by dividing current close by previous close -1
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
        # Calculate the EMA(100) of the standard deviation of the return series
        df0 = df0.ewm(span=span).std()
        # Reindex to match the original close series index and fill missing values
        df0 = df0.reindex(close.index)
        df0.fillna(0.01, inplace=True)

        return df0
