from enum import Enum


class VolatilityMeasure(str, Enum):
    MAX_RANGE = "max_range"
    ATR = "atr"
    ANNUNALIZED_STD_DEV = "annualized_std_dev"
    HIGH_LOW = "high_low"
    HIGH_LOW_CLOSE = "high_low_close"
    YANG_ZHANG = "yang_zhang"
    AVG_LOG_RETURNS = "avg_log_returns"
    EWM_STD = "ewm_std"
