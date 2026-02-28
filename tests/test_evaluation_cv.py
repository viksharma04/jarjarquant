"""Tests for evaluation/cross_validation module."""

import numpy as np
import pandas as pd
import pytest
from jarjarquant.evaluation.cross_validation import PurgedKFold, cv_score


def test_purged_kfold_produces_correct_n_splits():
    n = 100
    t1 = pd.Series(
        pd.date_range("2020-01-01", periods=n, freq="D") + pd.Timedelta(days=2),
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    kf = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
    X = pd.DataFrame(np.random.randn(n, 3), index=t1.index)
    splits = list(kf.split(X))
    assert len(splits) == 5


def test_purged_kfold_no_overlap():
    n = 100
    t1 = pd.Series(
        pd.date_range("2020-01-01", periods=n, freq="D") + pd.Timedelta(days=1),
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    kf = PurgedKFold(n_splits=3, t1=t1)
    X = pd.DataFrame(np.random.randn(n, 3), index=t1.index)
    splits = list(kf.split(X))
    for train_idx, test_idx in splits:
        assert len(set(train_idx) & set(test_idx)) == 0


def test_cv_score_returns_array():
    n = 100
    X = pd.DataFrame(
        np.random.randn(n, 3),
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    y = pd.Series(
        (np.random.randn(n) > 0).astype(int),
        index=X.index,
    )
    sw = pd.Series(np.ones(n), index=X.index)
    t1 = pd.Series(
        X.index + pd.Timedelta(days=1),
        index=X.index,
    )
    scores = cv_score(
        clf=None,
        X=X,
        y=y,
        sample_weight=sw,
        scoring="accuracy",
        t1=t1,
        cv=3,
        pct_embargo=0.01,
    )
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 3
