"""Evaluation package — cross-validation, feature importance, distribution analysis, threshold optimization."""

from .cross_validation import PurgedKFold as PurgedKFold, cv_score as cv_score
from .importance import (
    feature_importance_mdi as feature_importance_mdi,
    feature_importance_mda as feature_importance_mda,
    feature_importance_sfi as feature_importance_sfi,
)
from .distribution import (
    adf_test as adf_test,
    adf_test_ultra_fast as adf_test_ultra_fast,
    jb_normality_test as jb_normality_test,
    relative_entropy as relative_entropy,
    range_iqr_ratio as range_iqr_ratio,
    indicator_design_eval as indicator_design_eval,
    indicator_distribution_study as indicator_distribution_study,
)
from .threshold import (
    optimize_threshold as optimize_threshold,
    threshold_search as threshold_search,
)
