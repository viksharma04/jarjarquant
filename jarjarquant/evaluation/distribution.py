"""Statistical tests and distribution analysis for indicator evaluation.

Merged from data_analyst.py (statistical tests) and feature_evaluator.py
(distribution study orchestration).
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Literal

import numpy as np
from scipy.stats import jarque_bera, normaltest
from statsmodels.tsa.stattools import adfuller

from jarjarquant.schemas import (
    ADFTestResult,
    NormalityTestResult,
    EntropyResult,
    RangeIQRResult,
)

logger = logging.getLogger(__name__)


def adf_test(
    values: np.ndarray,
    alpha: float = 0.05,
    fast: bool = True,
) -> ADFTestResult:
    """Perform the Augmented Dickey-Fuller test for stationarity.

    Args:
        values: Time series data.
        alpha: Significance level for hypothesis test.
        fast: If True, use optimized fast mode with limited maxlag.

    Returns:
        ADFTestResult with test statistics and decision.
    """
    clean_values = values[~np.isnan(values)] if np.any(np.isnan(values)) else values

    if len(clean_values) < 3:
        raise ValueError("Series too short for ADF test after dropping NaNs")

    if fast:
        autolag = None
        n = len(clean_values)
        if n < 100:
            maxlag = min(3, n // 4)
        elif n < 500:
            maxlag = min(8, n // 10)
        else:
            maxlag = min(12, n // 20)
    else:
        autolag = "AIC"
        maxlag = None

    result = adfuller(clean_values, maxlag=maxlag, autolag=autolag)
    test_statistic = result[0]
    p_value = result[1]
    lags = result[2]
    nobs = result[3]
    critical_values = result[4]

    is_stationary = bool(p_value < alpha)

    if test_statistic < critical_values.get("1%", float("-inf")):
        decision = "strong_evidence_stationary"
    elif is_stationary:
        decision = "stationary"
    else:
        decision = "non_stationary"

    return ADFTestResult(
        statistic=test_statistic,
        p_value=p_value,
        lags=lags,
        nobs=nobs,
        critical_values=critical_values,
        decision=decision,
        is_stationary=is_stationary,
    )


def adf_test_ultra_fast(
    values: np.ndarray,
    alpha: float = 0.05,
) -> ADFTestResult:
    """Ultra-fast ADF test using minimal lags.

    Sacrifices some statistical rigor for ~10x speed improvement.

    Args:
        values: Time series data.
        alpha: Significance level.

    Returns:
        ADFTestResult with test statistics and decision.
    """
    clean_values = values[~np.isnan(values)] if np.any(np.isnan(values)) else values

    if len(clean_values) < 3:
        raise ValueError("Series too short for ADF test after dropping NaNs")

    maxlag = min(2, len(clean_values) // 10)

    result = adfuller(clean_values, maxlag=maxlag, autolag=None)
    test_statistic = result[0]
    p_value = result[1]
    lags = result[2]
    nobs = result[3]
    critical_values = result[4]

    is_stationary = bool(p_value < alpha)

    if test_statistic < critical_values.get("1%", float("-inf")):
        decision = "strong_evidence_stationary"
    elif is_stationary:
        decision = "stationary"
    else:
        decision = "non_stationary"

    return ADFTestResult(
        statistic=test_statistic,
        p_value=p_value,
        lags=lags,
        nobs=nobs,
        critical_values=critical_values,
        decision=decision,
        is_stationary=is_stationary,
    )


def jb_normality_test(
    values: np.ndarray,
    method: Literal["jb", "dagostino"] = "jb",
    alpha: float = 0.05,
) -> NormalityTestResult:
    """Perform normality test on a time series.

    Args:
        values: Time series data.
        method: 'jb' for Jarque-Bera, 'dagostino' for D'Agostino-Pearson.
        alpha: Significance level.

    Returns:
        NormalityTestResult with test statistics and decision.
    """
    clean_values = values[~np.isnan(values)] if np.any(np.isnan(values)) else values

    if len(clean_values) < 2:
        raise ValueError("Series too short for normality test after dropping NaNs")

    std_val = np.std(clean_values, ddof=1)
    if std_val == 0:
        raise ValueError("Cannot perform normality test on constant series")

    if method == "jb":
        statistic, p_value = jarque_bera(clean_values)
    elif method == "dagostino":
        statistic, p_value = normaltest(clean_values)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'jb' or 'dagostino'")

    is_normal = bool(p_value >= alpha)
    decision = "normal" if is_normal else "not_normal"

    return NormalityTestResult(
        statistic=float(statistic),
        p_value=float(p_value),
        method=method,
        decision=decision,
        is_normal=is_normal,
    )


def relative_entropy(values: np.ndarray) -> EntropyResult:
    """Calculate relative entropy to assess information content.

    Args:
        values: Time series data.

    Returns:
        EntropyResult with entropy metrics and quality assessment.
    """
    values = np.asarray(values)
    clean_values = values[~np.isnan(values)] if np.any(np.isnan(values)) else values

    if len(clean_values) == 0:
        raise ValueError("Cannot calculate entropy on empty series after dropping NaNs")

    n = len(clean_values)

    if n >= 10000:
        nbins = 20
    elif n >= 1000:
        nbins = 10
    elif n >= 100:
        nbins = 5
    else:
        nbins = 3

    counts, _ = np.histogram(clean_values, bins=nbins, density=False)
    probabilities = counts / n
    nonzero_probs = probabilities[probabilities > 0]
    entropy_val = -np.sum(nonzero_probs * np.log(nonzero_probs))
    normalized_entropy = entropy_val / np.log(nbins)

    if normalized_entropy < 0.2:
        quality = "VERY CONCERNING"
        is_concerning = True
    elif normalized_entropy < 0.5:
        quality = "CONCERNING"
        is_concerning = True
    elif normalized_entropy < 0.8:
        quality = "FINE"
        is_concerning = False
    else:
        quality = "EXCELLENT"
        is_concerning = False

    return EntropyResult(
        entropy=float(entropy_val),
        normalized_entropy=float(normalized_entropy),
        n_bins=nbins,
        quality=quality,
        is_concerning=is_concerning,
    )


def range_iqr_ratio(values: np.ndarray) -> RangeIQRResult:
    """Calculate range-to-IQR ratio to assess outlier presence.

    Args:
        values: Time series data.

    Returns:
        RangeIQRResult with ratio metrics and quality assessment.
    """
    values = np.asarray(values)
    clean_values = values[~np.isnan(values)] if np.any(np.isnan(values)) else values

    if len(clean_values) == 0:
        raise ValueError(
            "Cannot calculate range/IQR ratio on empty series after dropping NaNs"
        )
    if len(clean_values) < 2:
        raise ValueError(
            "Cannot calculate range/IQR ratio with less than 2 observations"
        )

    q25 = float(np.nanquantile(clean_values, 0.25))
    q75 = float(np.nanquantile(clean_values, 0.75))
    min_value = float(np.nanmin(clean_values))
    max_value = float(np.nanmax(clean_values))

    range_value = max_value - min_value
    iqr_value = q75 - q25

    if iqr_value == 0:
        raise ValueError(
            "Cannot calculate range/IQR ratio: IQR is zero (constant series)"
        )

    ratio = range_value / iqr_value

    if ratio <= 3:
        assessment = "GREAT DISTRIBUTION - MINIMAL OUTLIERS"
    elif ratio <= 5:
        assessment = "PASSABLE DISTRIBUTION - SOME OUTLIERS"
    else:
        assessment = "CONCERNING AMOUNT OF OUTLIERS - CONSIDER TRANSFORMATIONS"

    return RangeIQRResult(
        ratio=float(ratio),
        range_val=float(range_value),
        iqr=float(iqr_value),
        q1=q25,
        q3=q75,
        assessment=assessment,
    )


def indicator_design_eval(values: np.ndarray) -> dict:
    """Run all statistical tests on indicator values in parallel.

    Args:
        values: Indicator output array.

    Returns:
        Dictionary with keys: adf, normality, entropy, range_iqr.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        adf_future = executor.submit(adf_test_ultra_fast, values)
        normality_future = executor.submit(jb_normality_test, values)
        entropy_future = executor.submit(relative_entropy, values)
        r_iqr_future = executor.submit(range_iqr_ratio, values)

        return {
            "adf": adf_future.result(),
            "normality": normality_future.result(),
            "entropy": entropy_future.result(),
            "range_iqr": r_iqr_future.result(),
        }


def indicator_distribution_study(inputs: dict) -> dict:
    """Run distribution analysis on a single ticker's indicator values.

    Args:
        inputs: Dict with keys: indicator_spec, ohlcv_df, ticker (optional).

    Returns:
        Dict with ticker and flattened evaluation results.
    """
    indicator_spec = inputs["indicator_spec"]
    ohlcv_df = inputs["ohlcv_df"]

    indicator_instance = indicator_spec.create_indicator(ohlcv_df)
    eval_results = indicator_design_eval(indicator_instance.calculate())

    result = {"ticker": inputs.get("ticker", "")}

    # Flatten dataclass results into the dict
    for key, value in eval_results.items():
        if hasattr(value, "__dataclass_fields__"):
            for field_name in value.__dataclass_fields__:
                result[f"{key}_{field_name}"] = getattr(value, field_name)
        else:
            result[key] = value

    return result
