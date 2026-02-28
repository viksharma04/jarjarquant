"""Pure transformation functions for feature engineering."""

import numpy as np


def root_transform(data: np.ndarray, degree: int = 2) -> np.ndarray:
    """Apply a sign-preserving root transformation.

    Args:
        data: Input array.
        degree: Root degree (e.g., 2 for square root).

    Returns:
        Transformed array with original signs preserved.
    """
    signs = np.sign(data)
    roots = np.abs(data) ** (1 / degree)
    return signs * roots


def log_transform(data: np.ndarray) -> np.ndarray:
    """Apply a logarithmic transformation, shifting if necessary for non-positive values.

    Args:
        data: Input array.

    Returns:
        Log-transformed array.
    """
    min_val = np.min(data)
    if min_val <= 0:
        shifted = data + abs(min_val) + 1
    else:
        shifted = data
    return np.log(shifted)


def sigmoid_transform(data: np.ndarray, center: bool = True) -> np.ndarray:
    """Apply a tanh sigmoid transformation bounded to [-1, 1].

    Args:
        data: Input array.
        center: Whether to center data around expanding median.

    Returns:
        Transformed array with values in [-1, 1].
    """
    series = data.copy().astype(np.float64)

    if center:
        # Center around expanding median
        medians = np.array([np.median(series[: i + 1]) for i in range(len(series))])
        series -= medians

    # Expanding 15th and 85th percentiles for scaling
    p15 = np.array(
        [np.percentile(series[: i + 1], 15) for i in range(len(series))]
    )
    p85 = np.array(
        [np.percentile(series[: i + 1], 85) for i in range(len(series))]
    )

    spread = (p85 - p15) * 1.5
    # Avoid division by zero
    spread = np.where(spread == 0, 1.0, spread)
    series /= spread

    transformed = np.tanh(series)
    return np.where(np.isnan(transformed), 0.0, transformed)


def exp_smoothing(data: np.ndarray, span: int = 2) -> np.ndarray:
    """Apply exponential smoothing.

    Args:
        data: Input array.
        span: Smoothing span (alpha = 2 / (span + 1)).

    Returns:
        Smoothed array.
    """
    alpha = 2 / (span + 1)
    smoothed = np.empty(len(data))
    smoothed[0] = data[0]

    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]

    return smoothed


def apply_transform(data: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """Dispatch to a named transform function.

    Args:
        data: Input array.
        method: Transform name — "root", "log", or "sigmoid".
        **kwargs: Passed to the transform function.

    Returns:
        Transformed array.

    Raises:
        ValueError: If method is not recognized.
    """
    transforms = {
        "root": root_transform,
        "log": log_transform,
        "sigmoid": sigmoid_transform,
        "tanh": sigmoid_transform,  # alias for backward compat
    }
    if method not in transforms:
        raise ValueError(
            f"Unknown transform: '{method}'. Available: {list(transforms.keys())}"
        )
    return transforms[method](data, **kwargs)
