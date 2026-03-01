"""Jarjarquant — composable financial ML toolkit."""

__version__ = "1.0.0"

# Schemas and types
from jarjarquant.schemas import (
    VolatilityMeasure,
    BarSize,
    SampleRequest,
    Sample,
    ADFTestResult,
    NormalityTestResult,
    EntropyResult,
    RangeIQRResult,
)

# Indicators
from jarjarquant.indicators import (
    Indicator,
    IndicatorSpec,
    IndicatorType,
    list_available_indicators,
    get_indicator_parameters,
)

# Transforms
from jarjarquant.transforms import (
    log_transform,
    sigmoid_transform,
    root_transform,
    exp_smoothing,
    apply_transform,
)

# Fractional differentiation
from jarjarquant.fractional_diff import frac_diff, frac_diff_ffd

# Volatility
from jarjarquant.volatility import calculate_volatility

# Labelling
from jarjarquant.labelling import (
    triple_barrier_labels,
    event_sampling,
    inverse_cumsum_filter,
    get_sample_weights,
    one_period_with_sl,
    n_period_with_sl,
)

# Evaluation
from jarjarquant.evaluation import (
    PurgedKFold,
    cv_score,
    feature_importance_mdi,
    feature_importance_mda,
    adf_test,
    indicator_distribution_study,
    optimize_threshold,
)

# Storage
from jarjarquant.storage import DataRepository, DuckDBRepository
