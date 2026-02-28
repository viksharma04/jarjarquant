"""Tests for evaluation/distribution module."""

import numpy as np
import pytest
from jarjarquant.evaluation.distribution import (
    adf_test,
    jb_normality_test,
    relative_entropy,
    range_iqr_ratio,
    indicator_design_eval,
)
from jarjarquant.schemas import ADFTestResult, NormalityTestResult, EntropyResult, RangeIQRResult


def test_adf_test_returns_result():
    data = np.random.randn(200)
    result = adf_test(data)
    assert isinstance(result, ADFTestResult)
    assert isinstance(result.is_stationary, bool)


def test_jb_normality_test_returns_result():
    data = np.random.randn(200)
    result = jb_normality_test(data)
    assert isinstance(result, NormalityTestResult)


def test_relative_entropy_returns_result():
    data = np.random.randn(200)
    result = relative_entropy(data)
    assert isinstance(result, EntropyResult)


def test_range_iqr_ratio_returns_result():
    data = np.random.randn(200)
    result = range_iqr_ratio(data)
    assert isinstance(result, RangeIQRResult)


def test_indicator_design_eval_returns_dict():
    data = np.random.randn(300)
    result = indicator_design_eval(data)
    assert "adf" in result
    assert "normality" in result
    assert "entropy" in result
    assert "range_iqr" in result
