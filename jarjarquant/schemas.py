"""Consolidated data contracts for jarjarquant."""

from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Dict

import polars as pl


class VolatilityMeasure(str, Enum):
    """Volatility calculation methods."""

    MAX_RANGE = "max_range"
    ATR = "atr"
    ANNUALIZED_STD_DEV = "annualized_std_dev"
    HIGH_LOW = "high_low"
    HIGH_LOW_CLOSE = "high_low_close"
    YANG_ZHANG = "yang_zhang"
    AVG_LOG_RETURNS = "avg_log_returns"
    EWM_STD = "ewm_std"


class BarSize(StrEnum):
    """Bar size for price data."""

    ONE_SECOND = "1 sec"
    FIVE_SECONDS = "5 secs"
    TEN_SECONDS = "10 secs"
    FIFTEEN_SECONDS = "15 secs"
    THIRTY_SECONDS = "30 secs"
    ONE_MINUTE = "1 min"
    TWO_MINUTES = "2 mins"
    THREE_MINUTES = "3 mins"
    FIVE_MINUTES = "5 mins"
    TEN_MINUTES = "10 mins"
    FIFTEEN_MINUTES = "15 mins"
    TWENTY_MINUTES = "20 mins"
    THIRTY_MINUTES = "30 mins"
    ONE_HOUR = "1 hour"
    TWO_HOURS = "2 hours"
    THREE_HOURS = "3 hours"
    FOUR_HOURS = "4 hours"
    EIGHT_HOURS = "8 hours"
    ONE_DAY = "1 day"
    ONE_WEEK = "1 week"
    ONE_MONTH = "1 month"


@dataclass(slots=True)
class SampleRequest:
    """Request for a random sample of price data."""

    start_date: str
    end_date: str
    bar_size: BarSize = BarSize.ONE_DAY
    n_samples: int = 10


@dataclass(slots=True)
class Sample:
    """A sample of price data."""

    start_date: str
    end_date: str
    data: pl.DataFrame
    bar_size: BarSize


@dataclass
class ADFTestResult:
    """Result of Augmented Dickey-Fuller test."""

    statistic: float
    p_value: float
    lags: int
    nobs: int
    critical_values: Dict[str, float]
    decision: str
    is_stationary: bool


@dataclass
class NormalityTestResult:
    """Result of normality test."""

    statistic: float
    p_value: float
    method: str
    decision: str
    is_normal: bool


@dataclass
class EntropyResult:
    """Result of relative entropy calculation."""

    entropy: float
    normalized_entropy: float
    n_bins: int
    quality: str
    is_concerning: bool


@dataclass
class RangeIQRResult:
    """Result of range to IQR ratio calculation."""

    ratio: float
    range_val: float
    iqr: float
    q1: float
    q3: float
    assessment: str
