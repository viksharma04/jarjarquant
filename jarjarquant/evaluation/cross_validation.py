"""Purged cross-validation for financial time series.

Extracted from FeatureEvaluator. Implements PurgedKFold and cv_score
from Advances in Financial Machine Learning.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection._split import _BaseKFold


class PurgedKFold(_BaseKFold):
    """Custom KFold with purging and embargo for time-series data.

    Ensures that the training set does not contain observations that overlap
    with the test label intervals, and applies an embargo period after each
    test interval to prevent look-ahead bias.

    Args:
        n_splits: Number of folds for cross-validation.
        t1: Series of end times for each observation's label.
        pct_embargo: Fraction of observations to embargo after each test interval.
    """

    def __init__(self, n_splits: int = 3, t1: pd.Series | None = None, pct_embargo: float = 0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("Label through-dates must be a pandas Series.")

        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        """Generate indices for training and test splits with purge and embargo.

        Args:
            X: Input data with an index matching t1.
            y: Target values (not used in splitting).
            groups: Not used, only for compatibility.

        Yields:
            Tuple of (train_indices, test_indices) for each fold.
        """
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and t1 must have the same index.")

        indices = np.arange(X.shape[0])
        embargo_size = int(X.shape[0] * self.pct_embargo)

        test_intervals = [
            (i[0], i[-1] + 1)
            for i in np.array_split(np.arange(X.shape[0]), self.n_splits)
        ]

        for start_idx, end_idx in test_intervals:
            t0 = self.t1.index[start_idx]
            test_indices = indices[start_idx:end_idx]
            max_t1_idx = self.t1.index.searchsorted(self.t1.iloc[test_indices].max())

            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)

            if max_t1_idx < X.shape[0]:
                train_indices = np.concatenate(
                    (train_indices, indices[max_t1_idx + embargo_size:])
                )

            yield train_indices, test_indices


def cv_score(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    scoring: str = "neg_log_loss",
    t1: pd.Series | None = None,
    cv: int | None = None,
    cv_gen: PurgedKFold | None = None,
    pct_embargo: float | None = None,
) -> np.ndarray:
    """Calculate cross-validation scores using purged k-fold splits.

    Args:
        clf: Classifier with fit, predict, and optionally predict_proba methods.
            If None, a RandomForestClassifier is used.
        X: Feature matrix.
        y: Target labels.
        sample_weight: Sample weights for each observation.
        scoring: Scoring method, either 'neg_log_loss' or 'accuracy'.
        t1: End times of each observation's label.
        cv: Number of cross-validation folds.
        cv_gen: Custom cross-validation generator.
        pct_embargo: Fraction of samples to embargo after each test interval.

    Returns:
        Array of scores for each cross-validation fold.
    """
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise ValueError("Scoring method must be 'neg_log_loss' or 'accuracy'.")

    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)

    if clf is None:
        clf = RandomForestClassifier(
            n_estimators=100, max_features="sqrt", random_state=42
        )

    scores = []

    for train_indices, test_indices in cv_gen.split(X=X):
        model = clf.fit(
            X=X.iloc[train_indices, :],
            y=y.iloc[train_indices],
            sample_weight=sample_weight.iloc[train_indices].values,
        )

        if scoring == "neg_log_loss":
            probabilities = model.predict_proba(X.iloc[test_indices, :])
            score = -log_loss(
                y.iloc[test_indices],
                probabilities,
                sample_weight=sample_weight.iloc[test_indices].values,
                labels=clf.classes_,
            )
        else:
            predictions = model.predict(X.iloc[test_indices, :])
            score = accuracy_score(
                y.iloc[test_indices],
                predictions,
                sample_weight=sample_weight.iloc[test_indices].values,
            )

        scores.append(score)

    return np.array(scores)
