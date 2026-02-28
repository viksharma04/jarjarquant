"""Evaluation package — cross-validation, feature importance, distribution analysis, threshold optimization."""

from .cross_validation import PurgedKFold, cv_score
from .importance import feature_importance_mdi, feature_importance_mda, feature_importance_sfi
from .distribution import (
    adf_test,
    adf_test_ultra_fast,
    jb_normality_test,
    relative_entropy,
    range_iqr_ratio,
    indicator_design_eval,
    indicator_distribution_study,
)
from .threshold import optimize_threshold, threshold_search
