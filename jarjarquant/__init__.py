"""Jarjarquant — composable financial ML toolkit."""

__version__ = "1.0.0"

# Schemas and types
from jarjarquant.schemas import (
    VolatilityMeasure as VolatilityMeasure,
    BarSize as BarSize,
    SampleRequest as SampleRequest,
    Sample as Sample,
    ADFTestResult as ADFTestResult,
    NormalityTestResult as NormalityTestResult,
    EntropyResult as EntropyResult,
    RangeIQRResult as RangeIQRResult,
)

# Indicators
from jarjarquant.indicators import (
    Indicator as Indicator,
    IndicatorSpec as IndicatorSpec,
    IndicatorType as IndicatorType,
    list_available_indicators as list_available_indicators,
    get_indicator_parameters as get_indicator_parameters,
)

# Transforms
from jarjarquant.transforms import (
    log_transform as log_transform,
    sigmoid_transform as sigmoid_transform,
    root_transform as root_transform,
    exp_smoothing as exp_smoothing,
    apply_transform as apply_transform,
)

# Fractional differentiation
from jarjarquant.fractional_diff import (
    frac_diff as frac_diff,
    frac_diff_ffd as frac_diff_ffd,
)

# Volatility
from jarjarquant.volatility import calculate_volatility as calculate_volatility

# Labelling
from jarjarquant.labelling import (
    triple_barrier_labels as triple_barrier_labels,
    event_sampling as event_sampling,
    inverse_cumsum_filter as inverse_cumsum_filter,
    get_sample_weights as get_sample_weights,
    one_period_with_sl as one_period_with_sl,
    n_period_with_sl as n_period_with_sl,
)

# Evaluation
from jarjarquant.evaluation import (
    PurgedKFold as PurgedKFold,
    cv_score as cv_score,
    feature_importance_mdi as feature_importance_mdi,
    feature_importance_mda as feature_importance_mda,
    adf_test as adf_test,
    indicator_distribution_study as indicator_distribution_study,
    optimize_threshold as optimize_threshold,
)

# Storage
from jarjarquant.storage import (
    DataRepository as DataRepository,
    DuckDBRepository as DuckDBRepository,
)
