"""The data analyst specializes in performing and contextualizing common statistical tests on one or many data series"""

import logging
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numba import jit
from scipy.special import legendre
from scipy.stats import jarque_bera, norm, normaltest, spearmanr
from sklearn.metrics import normalized_mutual_info_score
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


@dataclass
class ADFTestResult:
    """Result of Augmented Dickey-Fuller test"""

    statistic: float
    pvalue: float
    lags: int
    nobs: int
    critical_values: Dict[str, float]
    decision: str
    is_stationary: bool


@dataclass
class NormalityTestResult:
    """Result of normality test"""

    statistic: float
    pvalue: float
    method: str
    decision: str
    is_normal: bool


@dataclass
class EntropyResult:
    """Result of relative entropy calculation"""

    entropy: float
    normalized_entropy: float
    n_bins: int
    n_observations: int
    quality_assessment: str
    is_concerning: bool


@dataclass
class RangeIQRResult:
    """Result of range to IQR ratio calculation"""

    ratio: float
    range_value: float
    iqr_value: float
    q25: float
    q75: float
    min_value: float
    max_value: float
    n_observations: int
    quality_assessment: str
    has_excessive_outliers: bool


def visual_stationary_test(
    values: np.ndarray,
    upper_percentile: Optional[float] = None,
    lower_percentile: Optional[float] = None,
):
    """Plots a line graph of calculated values for any class that extends indicator. Draws a top and bottom horizontal threshold above/below which 15% of observations lie

    Returns:
        None
    """
    _, ax = plt.subplots(figsize=(10, 6))
    ax.plot(values, label="Indicator Value", color="blue")

    # Calculate thresholds
    upper_threshold = np.nanquantile(values, upper_percentile or 0.85)
    lower_threshold = np.nanquantile(values, lower_percentile or 0.15)

    # Plot thresholds
    ax.axhline(
        y=float(upper_threshold),
        color="red",
        linestyle="--",
        label=f"{upper_threshold} Percentile",
    )
    ax.axhline(
        y=float(lower_threshold),
        color="green",
        linestyle="--",
        label=f"{lower_threshold} Percentile",
    )

    ax.set_title("Visual Stationary Test")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    return ax


def adf_test(
    values: np.ndarray,
    maxlag: Optional[int] = None,
    regression: str = "c",
    autolag: Optional[str] = "AIC",
    alpha: float = 0.05,
    verbose: bool = False,
) -> ADFTestResult:
    """
    Performs the Augmented Dickey-Fuller test to check if the time series is stationary.

    Args:
        values: Time series data
        maxlag: Maximum number of lags to include in test
        regression: Constant and trend order to include in regression
        autolag: Method to use when automatically determining the lag length
        alpha: Significance level for hypothesis test
        verbose: If True, log test results

    Returns:
        ADFTestResult: Structured result with test statistics and decision

    Raises:
        ValueError: If series is too short after dropping NaNs
    """
    # Drop NaNs and validate
    clean_values = values[~np.isnan(values)] if np.any(np.isnan(values)) else values

    if len(clean_values) < 3:
        raise ValueError("Series too short for ADF test after dropping NaNs")

    try:
        result = adfuller(
            clean_values, maxlag=maxlag, regression=regression, autolag=autolag
        )
        test_statistic, p_value, lags, nobs, critical_values, _ = result

        # Make decision based on p-value and alpha
        is_stationary = bool(p_value < alpha)

        # Determine strength of evidence
        if test_statistic < critical_values.get("1%", float("-inf")):
            decision = "strong_evidence_stationary"
        elif is_stationary:
            decision = "stationary"
        else:
            decision = "non_stationary"

        test_result = ADFTestResult(
            statistic=test_statistic,
            pvalue=p_value,
            lags=lags,
            nobs=nobs,
            critical_values=critical_values,
            decision=decision,
            is_stationary=is_stationary,
        )

        if verbose:
            logger.info(
                f"ADF Test: statistic={test_statistic:.4f}, p-value={p_value:.4f}, decision={decision}"
            )

        return test_result

    except ValueError as e:
        raise ValueError(f"ADF test failed: {str(e)}")


def jb_normality_test(
    values: np.ndarray,
    method: Literal["jb", "dagostino"] = "jb",
    alpha: float = 0.05,
    plot: bool = False,
    verbose: bool = False,
) -> NormalityTestResult:
    """
    Performs normality test to check if the time series is normally distributed.

    Args:
        values: Time series data
        method: Test method - 'jb' for Jarque-Bera, 'dagostino' for D'Agostino-Pearson
        alpha: Significance level for hypothesis test
        plot: If True, plots distribution histogram with normal curve
        verbose: If True, log test results

    Returns:
        NormalityTestResult: Structured result with test statistics and decision

    Raises:
        ValueError: If series is too short or constant
    """
    # Drop NaNs and validate
    clean_values = values[~np.isnan(values)] if np.any(np.isnan(values)) else values

    if len(clean_values) < 2:
        raise ValueError("Series too short for normality test after dropping NaNs")

    # Precompute statistics once
    mean_val = np.mean(clean_values)
    std_val = np.std(clean_values, ddof=1)  # Sample standard deviation
    min_val = np.min(clean_values)
    max_val = np.max(clean_values)

    # Handle constant series
    if std_val == 0:
        raise ValueError("Cannot perform normality test on constant series")

    # Perform test based on method
    if method == "jb":
        statistic, p_value = jarque_bera(clean_values)
    elif method == "dagostino":
        statistic, p_value = normaltest(clean_values)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'jb' or 'dagostino'")

    # Make decision
    is_normal = bool(p_value >= alpha)
    decision = "normal" if is_normal else "not_normal"

    test_result = NormalityTestResult(
        statistic=statistic,
        pvalue=p_value,
        method=method,
        decision=decision,
        is_normal=is_normal,
    )

    if verbose:
        logger.info(
            f"{method.upper()} Test: statistic={statistic:.4f}, p-value={p_value:.4f}, decision={decision}"
        )

    # Plot if requested
    if plot:
        plt.figure(figsize=(10, 6))

        # Use adaptive binning
        n_bins = min(30, max(10, len(clean_values) // 20))
        plt.hist(
            clean_values,
            bins=n_bins,
            density=True,
            color="blue",
            alpha=0.6,
            label="Data Distribution",
        )

        # Create efficient grid for normal curve (don't use 1000 points for short series)
        n_points = min(1000, max(100, len(clean_values) * 2))
        x = np.linspace(min_val, max_val, n_points)
        plt.plot(
            x,
            norm.pdf(x, mean_val, std_val),
            color="red",
            linestyle="--",
            label="Normal Distribution",
        )

        plt.title(f"Distribution with {method.upper()} Test (p={p_value:.4f})")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()

    return test_result


def relative_entropy(
    values: np.ndarray, verbose: bool = False, explain: bool = False
) -> EntropyResult:
    """
    Calculate the relative entropy of a time series to assess its information content.

    Args:
        values: Time series data
        verbose: If True, print entropy value and assessment
        explain: If True, print detailed explanation of entropy concept

    Returns:
        EntropyResult: Structured result with entropy metrics and quality assessment

    Raises:
        ValueError: If series is empty or contains only NaN values
    """
    # Convert values to numpy array for efficient computation
    values = np.asarray(values)

    # Drop NaNs and validate
    clean_values = values[~np.isnan(values)] if np.any(np.isnan(values)) else values

    if len(clean_values) == 0:
        raise ValueError("Cannot calculate entropy on empty series after dropping NaNs")

    n = len(clean_values)

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
    counts, _ = np.histogram(clean_values, bins=nbins, density=False)

    # Calculate relative entropy
    probabilities = counts / n  # Normalize counts to probabilities
    # Exclude zero probabilities
    nonzero_probs = probabilities[probabilities > 0]
    entropy_val = -np.sum(nonzero_probs * np.log(nonzero_probs))

    # Normalize entropy by log of number of bins
    normalized_entropy = entropy_val / np.log(nbins)

    # Determine quality assessment
    if normalized_entropy < 0.2:
        quality_assessment = "VERY CONCERNING"
        is_concerning = True
    elif normalized_entropy < 0.5:
        quality_assessment = "CONCERNING"
        is_concerning = True
    elif normalized_entropy < 0.8:
        quality_assessment = "FINE"
        is_concerning = False
    else:
        quality_assessment = "EXCELLENT"
        is_concerning = False

    # Create result object
    result = EntropyResult(
        entropy=entropy_val,
        normalized_entropy=normalized_entropy,
        n_bins=nbins,
        n_observations=n,
        quality_assessment=quality_assessment,
        is_concerning=is_concerning,
    )

    if explain:
        print(
            "The underlying idea for relative entropy is that valuable discriminatory information is nearly as likely to lie within clumps as it is in broad, thin regions. If the indicator's values lie within a few concentrated regions surrounded by broad swaths of emptiness, most models will focus on the wide spreads of the range while putting little effort into studying what's going on inside the clumps. \n The entropy of an indicator is an upper limit on the amount of information it can carry. Anything above 0.8 is plenty, an even somewhat lower is usually fine. Anything below 0.5 is concerning and below 0.2 is very concerning. \n ---------------------------------------- \n"
        )

    if verbose:
        print(f"Relative Entropy: {normalized_entropy:.4f}")
        print(quality_assessment)
        print("----------------------------------------")

    return result


def range_iqr_ratio(
    values: np.ndarray, verbose: bool = False, explain: bool = False
) -> RangeIQRResult:
    """
    Calculate the range to IQR ratio to assess the presence of outliers in a time series.

    Args:
        values: Time series data
        verbose: If True, print ratio value and assessment
        explain: If True, print detailed explanation of outlier impact

    Returns:
        RangeIQRResult: Structured result with ratio metrics and quality assessment

    Raises:
        ValueError: If series is empty, contains only NaN values, or has constant values (IQR=0)
    """
    # Convert values to numpy array for efficient computation
    values = np.asarray(values)

    # Drop NaNs and validate
    clean_values = values[~np.isnan(values)] if np.any(np.isnan(values)) else values

    if len(clean_values) == 0:
        raise ValueError(
            "Cannot calculate range/IQR ratio on empty series after dropping NaNs"
        )

    if len(clean_values) < 2:
        raise ValueError(
            "Cannot calculate range/IQR ratio with less than 2 observations"
        )

    n = len(clean_values)

    # Calculate statistics using nanquantile for robustness
    min_value = np.nanmin(clean_values)
    max_value = np.nanmax(clean_values)
    q25 = np.nanquantile(clean_values, 0.25)
    q75 = np.nanquantile(clean_values, 0.75)

    range_value = max_value - min_value
    iqr_value = q75 - q25

    # Guard against IQR = 0 (constant or near-constant series)
    if iqr_value == 0:
        raise ValueError(
            "Cannot calculate range/IQR ratio: IQR is zero (constant series)"
        )

    ratio = range_value / iqr_value

    # Determine quality assessment
    if ratio <= 3:
        quality_assessment = "GREAT DISTRIBUTION - MINIMAL OUTLIERS"
        has_excessive_outliers = False
    elif ratio <= 5:
        quality_assessment = "PASSABLE DISTRIBUTION - SOME OUTLIERS - INSPECT VISUALLY"
        has_excessive_outliers = False
    else:
        quality_assessment = "CONCERNING AMOUNT OF OUTLIERS - CONSIDER TRANSFORMATIONS"
        has_excessive_outliers = True

    # Create result object
    result = RangeIQRResult(
        ratio=ratio,
        range_value=range_value,
        iqr_value=iqr_value,
        q25=q25,
        q75=q75,
        min_value=min_value,
        max_value=max_value,
        n_observations=n,
        quality_assessment=quality_assessment,
        has_excessive_outliers=has_excessive_outliers,
    )

    if explain:
        print(
            "Presence of outliers can greatly reduce the performance of an algorithm. The most obvious reason is that the presence of an outlier reduces entropy, causing the 'normal' observations to form a compact cluster and hence reducing the information carrying capacity of the indicator. \n Ratios of 2 and 3 are reasonable and up to 5 is usually not excessive. But if the indicator has a range IQR ratio of more than 5, the tails should be tamed. \n ----------------------------- \n"
        )

    if verbose:
        print(f"Range/IQR Ratio: {ratio:.4f}")
        print(quality_assessment)
        print("----------------------------------------")

    return result


def mutual_information(
    array: np.ndarray,
    lag: int,
    n_bins: Optional[int] = None,
    is_discrete: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Calculates the mutual information of the array with a specified time lag.

    Args:
        array (np.ndarray): A numpy array representing the time series data.
        lag (int): NMI to calculated upto lag.

    Returns:
        float: The mutual information score.
    """
    nmi_scores = np.full(lag, np.nan)
    if not is_discrete:
        n = len(array)
        if n_bins is None:
            #  Determine the number of bins based on sample size
            if n >= 10000:
                n_bins = 20
            elif n >= 1000:
                n_bins = 10
            elif n >= 100:
                n_bins = 5
            else:
                n_bins = 3
        array = discretize_array(array, n_bins=n_bins)

    for i in range(1, lag + 1):
        # Create lagged array
        if i >= len(array):
            raise ValueError("Lag is greater than or equal to the length of the array.")

        array_lagged = array[:-i]
        array_future = array[i:]

        # Calculate mutual information
        nmi_scores[i - 1] = normalized_mutual_info_score(array_lagged, array_future)

    return nmi_scores


def discretize_array(array: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Discretizes the array into n_bins using quantile-based binning.

    Args:
        array (np.ndarray): A numpy array representing the time series data.
        n_bins (int): Number of bins to discretize into using quantile-based binning.

    Returns:
        np.ndarray: An array of discretized values.
    """
    # Discretize the array into n_bins using np.percentile for quantile-based binning
    bins = np.percentile(array, np.linspace(0, 100, n_bins + 1))
    discretized = np.digitize(array, bins, right=False) - 1

    # Ensure the discretized values are within the range [0, n_bins - 1]
    discretized[discretized >= n_bins] = n_bins - 1

    return discretized


def mutual_information_with_time():
    pass


@jit(nopython=True)
def _compute_true_ranges(
    high: np.ndarray, low: np.ndarray, prev_close: np.ndarray
) -> np.ndarray:
    """
    Numba-optimized computation of true ranges.

    Args:
        high: Array of high prices
        low: Array of low prices
        prev_close: Array of previous close prices

    Returns:
        np.ndarray: True range values
    """
    n = len(high)
    true_ranges = np.empty(n)

    for i in range(n):
        if np.isnan(prev_close[i]):
            # If previous close is NaN, use high-low range
            true_ranges[i] = high[i] - low[i]
        else:
            # True range = max(high-low, |high-prev_close|, |low-prev_close|)
            hl = high[i] - low[i]
            hc = abs(high[i] - prev_close[i])
            lc = abs(low[i] - prev_close[i])
            true_ranges[i] = max(hl, hc, lc)

    return true_ranges


def atr(
    atr_length: int,
    high_series: pd.Series,
    low_series: pd.Series,
    close_series: pd.Series,
    ema: bool = False,
    expanding: bool = False,
) -> pd.Series:
    """
    Parameters:
    atr_length (int): The period over which to calculate the ATR.
    high_series (pd.Series): Series of high prices.
    low_series (pd.Series): Series of low prices.
    close_series (pd.Series): Series of close prices.
    ema (bool): If True, use exponential moving average instead of simple moving average.
    expanding (bool): If True, use expanding window instead of rolling window, calculating ATR using all available data up to each point.

    Returns:
    pd.Series: The ATR values for the given period.
    """
    # Validate inputs
    if not all(
        isinstance(s, pd.Series) for s in [high_series, low_series, close_series]
    ):
        raise ValueError("All price inputs must be pandas Series")

    if not (
        high_series.index.equals(low_series.index)
        and high_series.index.equals(close_series.index)
    ):
        raise ValueError("All series must have the same index")

    if len(high_series) < atr_length:
        raise ValueError(f"Insufficient data: need at least {atr_length} observations")

    # Store original index for return
    original_index = high_series.index

    # Forward-fill close prices to handle NaNs before shifting
    close_filled = close_series.ffill()

    # Precompute previous close
    prev_close = close_filled.shift(1)

    # Use numba-optimized true range calculation
    true_ranges_array = _compute_true_ranges(
        np.asarray(high_series.values),
        np.asarray(low_series.values),
        np.asarray(prev_close.values),
    )

    # Create Series with original index
    true_ranges = pd.Series(true_ranges_array, index=original_index)

    # Compute the moving average of the true ranges
    if ema:
        atr_result = true_ranges.ewm(span=atr_length, adjust=False).mean()
    else:
        if expanding:
            # Use expanding window (all data up to current point)
            # Set min_periods to atr_length to match the behavior of rolling window
            atr_result = true_ranges.expanding(min_periods=atr_length).mean()
        else:
            # Use fixed-length rolling window
            atr_result = true_ranges.rolling(window=atr_length).mean()

    return atr_result


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


def compute_normalized_legendre_coefficients(n, degree):
    """
    Compute three normalized arrays c1, c2, and c3 of length n.

    The arrays are computed as follows:
    - c1: Linearly spaced values in [-1, 1] normalized to unit length.
    - c2: Square of c1, centered by subtracting the mean, then normalized.
    - c3: Cube of c1, centered by subtracting the mean, normalized,
            and then made orthogonal to c1 by removing its projection onto c1,
            with a final normalization.

    Parameters:
        n (int): Number of elements in each output array.

    Returns:
        tuple: (c1, c2, c3) as NumPy arrays.
    """
    # Compute c1: linearly spaced from -1 to 1 and normalized to unit length.
    c1 = np.linspace(-1.0, 1.0, n)
    c1 /= np.linalg.norm(c1)

    # Compute c2: square of c1, then center and normalize.
    c2 = c1**2
    mean_c2 = np.mean(c2)
    c2 -= mean_c2
    c2 /= np.linalg.norm(c2)

    # Compute c3: cube of c1, then center and normalize.
    c3 = c1**3
    mean_c3 = np.mean(c3)
    c3 -= mean_c3
    c3 /= np.linalg.norm(c3)

    # Remove the projection of c1 from c3.
    proj = np.dot(c1, c3)
    c3 = c3 - proj * c1
    c3 /= np.linalg.norm(c3)

    if degree == 1:
        return c1
    elif degree == 2:
        return c2
    elif degree == 3:
        return c3


def calculate_regression_coefficient(
    prices: np.ndarray, legendre_coeffs: np.ndarray
) -> float:
    """
    Calculate the linear regression coefficient using the dot product method.

    Parameters:
    - prices (numpy.ndarray): The series of prices (e.g., log prices).
    - legendre_coeffs (numpy.ndarray): The precomputed Legendre coefficients.

    Returns:
    - float: The slope of the least squares line.
    """
    # norm_coeffs_squared = np.sum(legendre_coeffs ** 2)
    slope = np.dot(legendre_coeffs, prices)  # / norm_coeffs_squared
    return slope


def get_daily_vol(close: pd.Series, span: int = 100) -> pd.Series:
    # Find the index of each previous day in the close series
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    # If a day does not fit in the series, exclude it
    df0 = df0[df0 > 0]
    # Create a series which stores the previous date for each date in the close series
    df0 = pd.Series(
        close.index[df0 - 1], index=close.index[close.shape[0] - df0.shape[0] :]
    )
    # Calculate the return by dividing current close by previous close -1
    df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    # Calculate the EMA(100) of the standard deviation of the return series
    df0 = df0.ewm(span=span).std()
    # Reindex to match the original close series index and fill missing values
    df0 = df0.reindex(close.index)
    df0.fillna(0.01, inplace=True)

    return df0


def get_spearman_correlation(series1, series2):
    # Convert to numpy arrays if pd.Series
    if isinstance(series1, pd.Series):
        series1 = series1.values
    if isinstance(series2, pd.Series):
        series2 = series2.values

    # Calculate Spearman correlation
    spearman_corr, _ = spearmanr(series1, series2)

    # Co sort the series and find spearmann correlation by quantile
    sorted_indices = np.argsort(series1)
    sorted_series1 = series1[sorted_indices]
    sorted_series2 = series2[sorted_indices]

    n = len(sorted_series1)
    # Define bin edges for 4 equal-sized bins
    bin_edges = np.linspace(0, n, 5, dtype=int)

    correlations = []
    for i in range(4):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        s1_bin = sorted_series1[start:end]
        s2_bin = sorted_series2[start:end]

        # Compute Spearman correlation if there are at least 2 data points
        if len(s1_bin) > 1:
            corr, _ = spearmanr(s1_bin, s2_bin)
        else:
            corr = np.nan
        correlations.append(corr)

    return {"spearman_corr": spearman_corr, "spearman_corr_quartile": correlations}


def plot_loess(
    x: np.ndarray,
    y: np.ndarray,
    smoothing_factor: Optional[int] = 3,
    x_label: Optional[str] = "x",
    y_label: Optional[str] = "y",
    title: Optional[str] = "LOESS Fit",
    annotation: Optional[float] = None,
):
    # Fit LOESS (LOWESS in statsmodels)
    frac = float(smoothing_factor / 10)
    lowess = sm.nonparametric.lowess(y, x, frac=frac)  # frac controls smoothing

    # Extract smoothed values
    x_smooth, y_smooth = lowess[:, 0], lowess[:, 1]

    # Plot
    plt.scatter(x, y, alpha=0.5, label="Data")
    plt.plot(x_smooth, y_smooth, color="red", linewidth=2, label="LOESS Fit")
    # Annotate the correlation coefficient on the plot
    if annotation is not None:
        plt.text(
            0.05,
            0.95,
            f"Spearman r = {annotation:.2f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
        )

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


@jit(nopython=True)
def _determine_initial_direction(series: np.ndarray, start_idx: int, thresholds: np.ndarray) -> int:
    """
    Look ahead to determine the initial direction based on first significant move.
    
    Args:
        series: Price series
        start_idx: Starting index
        thresholds: Threshold values
    
    Returns:
        1 for up, -1 for down, 0 if no significant move found
    """
    n = len(series)
    if start_idx >= n - 1:
        return 1  # Default to up if no data to analyze
    
    initial_price = series[start_idx]
    if abs(initial_price) < 1e-10:
        return 1  # Default to up if price is zero
    
    # Look ahead for first significant directional change
    for i in range(start_idx + 1, n):
        value = series[i]
        
        # Skip invalid values
        if value == 0.0 or np.isnan(value):
            continue
        
        threshold = thresholds[i]
        change = (value - initial_price) / abs(initial_price)
        
        if abs(change) >= threshold:
            return 1 if change > 0 else -1
    
    return 1  # Default to up if no significant move found


@jit(nopython=True)
def _directional_change_core(series: np.ndarray, thresholds: np.ndarray) -> tuple:
    """
    Numba-optimized core directional change detection.
    
    Args:
        series: Price series
        thresholds: Threshold values (same length as series)
    
    Returns:
        Tuple of (recognition_indices, extreme_indices)
    """
    n = len(series)
    if n == 0:
        empty_array = np.empty(0, dtype=np.int64)
        return (empty_array, empty_array)
    
    # Find first valid (non-zero) value for initial detection
    start_idx = 0
    for i in range(n):
        if series[i] != 0.0 and not np.isnan(series[i]):
            start_idx = i
            break
    
    if start_idx == n - 1:  # Only one valid value
        result = np.empty(1, dtype=np.int64)
        result[0] = start_idx
        return (result, result)
    
    # Determine initial direction by looking ahead
    initial_direction = _determine_initial_direction(series, start_idx, thresholds)
    up = initial_direction > 0
    
    # Pre-allocate arrays with maximum possible size
    max_pivots = n // 2 + 1
    recognition_indices_temp = np.empty(max_pivots, dtype=np.int64)
    extreme_indices_temp = np.empty(max_pivots, dtype=np.int64)
    
    pivot_count = 0
    current_pivot = series[start_idx]
    current_pivot_idx = start_idx
    
    # Add initial pivot
    recognition_indices_temp[pivot_count] = start_idx
    extreme_indices_temp[pivot_count] = start_idx
    pivot_count += 1
    
    for i in range(start_idx + 1, n):
        value = series[i]
        
        # Skip invalid values
        if value == 0.0 or np.isnan(value):
            continue
            
        threshold = thresholds[i]
        
        # Guard against division by zero
        if abs(current_pivot) < 1e-10:
            current_pivot = value
            current_pivot_idx = i
            continue
        
        if up:
            # Looking for higher highs
            if value > current_pivot:
                current_pivot = value
                current_pivot_idx = i
            else:
                # Check for downward directional change
                change = (value - current_pivot) / abs(current_pivot)
                if change < -threshold:
                    up = False
                    recognition_indices_temp[pivot_count] = i  # Recognition index
                    extreme_indices_temp[pivot_count] = current_pivot_idx  # Extreme index
                    pivot_count += 1
                    current_pivot = value
                    current_pivot_idx = i
        else:
            # Looking for lower lows
            if value < current_pivot:
                current_pivot = value
                current_pivot_idx = i
            else:
                # Check for upward directional change
                change = (value - current_pivot) / abs(current_pivot)
                if change > threshold:
                    up = True
                    recognition_indices_temp[pivot_count] = i  # Recognition index
                    extreme_indices_temp[pivot_count] = current_pivot_idx  # Extreme index
                    pivot_count += 1
                    current_pivot = value
                    current_pivot_idx = i
    
    # Trim arrays to actual size
    recognition_indices = recognition_indices_temp[:pivot_count]
    extreme_indices = extreme_indices_temp[:pivot_count]
    
    return (recognition_indices, extreme_indices)


def directional_change_pivots(
    series: np.ndarray,
    threshold_type: str = "static",
    threshold_value: Optional[float] = 0.01,
    atr_window: int = 14,
    high_series: Optional[np.ndarray] = None,
    low_series: Optional[np.ndarray] = None,
    close_series: Optional[np.ndarray] = None,
    return_extremes: bool = False,
) -> tuple:
    """
    Detect directional change pivots using static or volatility-based thresholds.
    
    Args:
        series: Price series (typically close prices)
        threshold_type: "static" for fixed threshold, "volatility" for ATR-based
        threshold_value: Static threshold value (percentage as decimal, e.g., 0.01 for 1%)
        atr_window: Window size for ATR calculation when using volatility threshold
        high_series: High prices (required for volatility threshold)
        low_series: Low prices (required for volatility threshold)
        close_series: Close prices (required for volatility threshold, defaults to series)
        return_extremes: If True, return tuple of (recognition_indices, extreme_indices).
                        If False, return only recognition_indices for backward compatibility.
    
    Returns:
        If return_extremes=True: tuple of (recognition_indices, extreme_indices)
        If return_extremes=False: recognition_indices only (for backward compatibility)
    
    Raises:
        ValueError: If inputs are invalid or missing required data for volatility threshold
    """
    if len(series) == 0:
        empty_array = np.array([], dtype=np.int64)
        return (empty_array, empty_array) if return_extremes else empty_array
    
    series = np.asarray(series, dtype=np.float64)
    
    if threshold_type == "static":
        if threshold_value is None:
            raise ValueError("threshold_value must be provided for static threshold type")
        
        # Create constant threshold array
        thresholds = np.full(len(series), abs(threshold_value), dtype=np.float64)
        
    elif threshold_type == "volatility":
        if any(x is None for x in [high_series, low_series]):
            raise ValueError("high_series and low_series must be provided for volatility threshold")
        
        high_series = np.asarray(high_series, dtype=np.float64)
        low_series = np.asarray(low_series, dtype=np.float64)
        close_series = np.asarray(close_series if close_series is not None else series, dtype=np.float64)
        
        if not (len(series) == len(high_series) == len(low_series) == len(close_series)):
            raise ValueError("All price series must have the same length")
        
        # Convert to pandas for ATR calculation
        import pandas as pd
        high_pd = pd.Series(high_series)
        low_pd = pd.Series(low_series)
        close_pd = pd.Series(close_series)
        
        # Calculate ATR
        atr_series = atr(atr_window, high_pd, low_pd, close_pd)
        atr_values = atr_series.values
        
        # Use ATR as percentage of current price for threshold
        # Guard against division by zero
        thresholds = np.where(
            np.abs(series) > 1e-10,
            atr_values / np.abs(series),
            np.full(len(series), 0.01)  # Default 1% if division by zero
        )
        
        # Handle NaN values in ATR (fill with default)
        thresholds = np.where(np.isnan(thresholds), 0.01, thresholds)
        
    else:
        raise ValueError(f"Invalid threshold_type: {threshold_type}. Must be 'static' or 'volatility'")
    
    # Call optimized core function
    recognition_indices, extreme_indices = _directional_change_core(series, thresholds)
    
    if return_extremes:
        return (recognition_indices, extreme_indices)
    else:
        return recognition_indices
