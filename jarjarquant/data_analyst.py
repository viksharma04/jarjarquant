"""The data analyst specializes in performing and contextualizing common statistical tests on one or many data series"""
import pandas as pd
import numpy as np
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
    def adf_test(values: np.ndarray):
        """
        Performs the Augmented Dickey-Fuller test to check if the time series is stationary.

        Returns:
            None
        """
        result = adfuller(values)
        test_statistic, p_value, _, _, critical_values, _ = result

        # Print test results
        print("Augmented Dickey-Fuller Test Results:")
        print(f"Test Statistic: {test_statistic}")
        print(f"P-Value: {p_value}")
        print("Critical Values:")
        for key, value in critical_values.items():
            print(f"\t{key}: {value}")

        # Interpretation
        print("\nInterpretation:")
        if p_value < 0.05:
            print("The p-value is less than 0.05, indicating that we can reject the null hypothesis.\n"
                  "The time series is likely stationary.")
        else:
            print("The p-value is greater than or equal to 0.05, indicating that we cannot reject the null hypothesis.\n"
                  "The time series is likely non-stationary.")

        if test_statistic < min(critical_values.values()):
            print("The test statistic is less than the critical value at all levels, indicating strong evidence of stationarity.")
        else:
            print("The test statistic is not less than the critical value at all levels, suggesting weaker evidence of stationarity.")

    @staticmethod
    def jb_normality_test(values: np.ndarray, plot_dist=False):
        """
        Performs the Jarque-Bera test to check if the time series is normally distributed.

        Args:
            plot_dist (bool): If True, plots a histogram of the indicator values with a standard normal density curve.

        Returns:
            None
        """
        jb_statistic, p_value = jarque_bera(values)

        # Print test results
        print("Jarque-Bera Test Results:")
        print(f"JB Statistic: {jb_statistic}")
        print(f"P-Value: {p_value}")

        # Interpretation
        print("\nInterpretation:")
        if p_value < 0.05:
            print("The p-value is less than 0.05, indicating that we can reject the null hypothesis.\n"
                  "The time series is likely not normally distributed.")
        else:
            print("The p-value is greater than or equal to 0.05, indicating that we cannot reject the null hypothesis.\n"
                  "The time series is likely normally distributed.")

        # Plot distribution if requested
        if plot_dist:
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

    @staticmethod
    def relative_entropy(values: np.ndarray, print_desc: bool = False) -> np.float64:
        # Convert values to numpy array for efficient computation
        values = np.array(values)
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

        if print_desc:
            print("The underlying idea for relative entropy is that valuable discriminatory information is nearly as likely to lie within clumps as it is in broad, thin regions. If the indicator's values lie within a few concentrated regions surrounded by broad swaths of emptiness, most models will focus on the wide spreads of the range while putting little effort into studying what's going on inside the clumps. \n The entropy of an indicator is an upper limit on the amount of infornmation it can carry. Anything above 0.8 is plenty, an even somewhat lower is usually fine. Anything below 0.5 is concerning and below 0.2 is very concerning. \n ---------------------------------------- \n")

        return normalized_entropy

    @staticmethod
    def range_iqr_ratio(values: np.ndarray, print_desc: bool = True) -> np.float64:
        """Returns the range/interquartile range ratio for an indicator

        Args:
            values (np.ndarray): Indicator time series

        Returns:
            np.float64: range iqr ratios
        """
        range_iqr_ratio = (np.max(values) - np.min(values)) / \
            (np.quantile(values, 0.75) - np.quantile(values, 0.25))
        if print_desc:
            print("Presence of outliers can greatly reduce the performance of an algorithm. The most obvious reason is that the presence of an outlier reduces entropy, causing the 'normal' observations to form a compact cluster and hence reducing the information carrying capacity of the indicator. \n Ratios of 2 and 3 are reasonable and upto 5 is usually not excessive. But iof the indicator has a range IQR ratio of more than 5, the tails should be tamed. \n ----------------------------- \n")
        return range_iqr_ratio

    @staticmethod
    def mutual_information(array: np.ndarray, lag: int, n_bins: int = 10, is_discrete: bool = False) -> float:
        """
        Calculates the mutual information of the array with a specified time lag.

        Args:
            array (np.ndarray): A numpy array representing the time series data.
            lag (int): Time lag to use for calculating mutual information.

        Returns:
            float: The mutual information score.
        """
        if not is_discrete:
            array = DataAnalyst.discretize_array(array, n_bins=n_bins)

        # Create lagged array
        if lag >= len(array):
            raise ValueError(
                "Lag is greater than or equal to the length of the array.")

        array_lagged = array[:-lag]
        array_future = array[lag:]

        # Calculate mutual information
        mi = normalized_mutual_info_score(array_lagged, array_future)
        return mi

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
    def atr(atr_length: int, open_series: pd.Series, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series):
        """
        Calculate the Average True Range (ATR) for a given period.
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
        return true_ranges.shift(1).rolling(window=atr_length).mean()
