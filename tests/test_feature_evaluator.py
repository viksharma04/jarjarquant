import pytest
import pandas as pd
import numpy as np
from jarjarquant.feature_evaluator import FeatureEvaluator


def test_indicator_threshold_search_linear():
    # Create sample data
    indicator_values = pd.Series(np.linspace(0, 100, 100))
    associated_returns = pd.Series(np.random.randn(100))

    # Call the function with linear threshold option
    result = FeatureEvaluator.indicator_threshold_search(
        indicator_values=indicator_values,
        associated_returns=associated_returns,
        n_thresholds=10,
        threshold_option='linear'
    )

    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check if the DataFrame has the expected columns
    expected_columns = [
        'Threshold', '% values > threshold', 'Mean return above threshold', 'Std dev return above threshold',
        'Median return above threshold', 'Q25 return above threshold', 'Q75 return above threshold',
        'PF Long above threshold', 'PF Short above threshold', '% values < threshold',
        'Mean return below threshold', 'Std dev return below threshold', 'Median return below threshold',
        'Q25 return below threshold', 'Q75 return below threshold', 'PF Long below threshold', 'PF Short below threshold'
    ]
    assert list(result.columns) == expected_columns


def test_indicator_threshold_search_percentile():
    # Create sample data
    indicator_values = pd.Series(np.linspace(0, 100, 100))
    associated_returns = pd.Series(np.random.randn(100))

    # Call the function with percentile threshold option
    result = FeatureEvaluator.indicator_threshold_search(
        indicator_values=indicator_values,
        associated_returns=associated_returns,
        n_thresholds=10,
        threshold_option='percentile'
    )

    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check if the DataFrame has the expected columns
    expected_columns = [
        'Threshold', '% values > threshold', 'Mean return above threshold', 'Std dev return above threshold',
        'Median return above threshold', 'Q25 return above threshold', 'Q75 return above threshold',
        'PF Long above threshold', 'PF Short above threshold', '% values < threshold',
        'Mean return below threshold', 'Std dev return below threshold', 'Median return below threshold',
        'Q25 return below threshold', 'Q75 return below threshold', 'PF Long below threshold', 'PF Short below threshold'
    ]
    assert list(result.columns) == expected_columns


def test_indicator_threshold_search_predefined_thresholds():
    # Create sample data
    indicator_values = pd.Series(np.linspace(0, 100, 100))
    associated_returns = pd.Series(np.random.randn(100))
    predefined_thresholds = [20, 40, 60, 80]

    # Call the function with predefined thresholds
    result = FeatureEvaluator.indicator_threshold_search(
        indicator_values=indicator_values,
        associated_returns=associated_returns,
        thresholds=predefined_thresholds
    )

    # Check if the result is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check if the DataFrame has the expected columns
    expected_columns = [
        'Threshold', '% values > threshold', 'Mean return above threshold', 'Std dev return above threshold',
        'Median return above threshold', 'Q25 return above threshold', 'Q75 return above threshold',
        'PF Long above threshold', 'PF Short above threshold', '% values < threshold',
        'Mean return below threshold', 'Std dev return below threshold', 'Median return below threshold',
        'Q25 return below threshold', 'Q75 return below threshold', 'PF Long below threshold', 'PF Short below threshold'
    ]
    assert list(result.columns) == expected_columns


def test_indicator_threshold_search_invalid_threshold_option():
    # Create sample data
    indicator_values = pd.Series(np.linspace(0, 100, 100))
    associated_returns = pd.Series(np.random.randn(100))

    # Call the function with an invalid threshold option
    with pytest.raises(ValueError, match="threshold_option must be 'linear' or 'percentile'."):
        FeatureEvaluator.indicator_threshold_search(
            indicator_values=indicator_values,
            associated_returns=associated_returns,
            n_thresholds=10,
            threshold_option='invalid_option'
        )


def test_indicator_threshold_search_no_thresholds_provided():
    # Create sample data
    indicator_values = pd.Series(np.linspace(0, 100, 100))
    associated_returns = pd.Series(np.random.randn(100))

    # Call the function without providing thresholds or n_thresholds
    with pytest.raises(ValueError, match="Either n_thresholds or thresholds must be provided."):
        FeatureEvaluator.indicator_threshold_search(
            indicator_values=indicator_values,
            associated_returns=associated_returns,
            thresholds=None,
            n_thresholds=None
        )
