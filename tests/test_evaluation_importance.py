"""Tests for evaluation/importance module."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from jarjarquant.evaluation.importance import (
    feature_importance_mdi,
    feature_importance_mda,
)


def test_feature_importance_mdi_returns_dataframe():
    n = 200
    X = np.random.randn(n, 5)
    y = (X[:, 0] > 0).astype(int)
    result = feature_importance_mdi(X, y, feature_names=[f"f{i}" for i in range(5)], n_estimators=10)
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 5  # 5 features


def test_feature_importance_mda_returns_dataframe():
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    X = pd.DataFrame(np.random.randn(n, 3), index=idx, columns=["a", "b", "c"])
    y = pd.Series((np.random.randn(n) > 0).astype(int), index=idx)
    sw = pd.Series(np.ones(n), index=idx)
    t1 = pd.Series(idx + pd.Timedelta(days=1), index=idx)
    result, mean_score = feature_importance_mda(
        X=X, y=y, sample_weight=sw, t1=t1, cv=3, n_estimators=10
    )
    assert isinstance(result, pl.DataFrame)
    assert isinstance(mean_score, float)
