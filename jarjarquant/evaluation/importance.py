"""Feature importance methods for financial ML.

Implements MDI, MDA, and SFI from Advances in Financial Machine Learning.
Extracted from FeatureEvaluator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss

from .cross_validation import PurgedKFold, cv_score


def feature_importance_mdi(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    n_estimators: int = 100,
) -> pl.DataFrame:
    """Calculate feature importance using Mean Decrease Impurity (MDI).

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target labels.
        feature_names: Names for features. If None, uses f0, f1, ...
        n_estimators: Number of trees in the random forest.

    Returns:
        Polars DataFrame with columns: feature, mean, std.
    """
    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n_features)]

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_features="sqrt", random_state=42
    )
    clf.fit(X, y)

    # Collect feature importances from each tree
    importance_dict = {
        i: tree.feature_importances_ for i, tree in enumerate(clf.estimators_)
    }

    importance_df = pd.DataFrame.from_dict(importance_dict, orient="index")
    importance_df.columns = feature_names

    # Replace zeros with NaN to avoid distortions
    importance_df = importance_df.replace(0, np.nan)

    means = importance_df.mean().values
    stds = (importance_df.std() * (importance_df.shape[0] ** -0.5)).values

    # Normalize means to sum to 1
    total = np.nansum(means)
    if total > 0:
        means = means / total

    return pl.DataFrame({
        "feature": feature_names,
        "mean": means,
        "std": stds,
    })


def feature_importance_mda(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
    t1: pd.Series,
    cv: int = 4,
    clf=None,
    pct_embargo: float = 0.04,
    scoring: str = "neg_log_loss",
    n_estimators: int = 100,
) -> tuple[pl.DataFrame, float]:
    """Calculate feature importance using Mean Decrease Accuracy (MDA).

    Permutes each feature to measure the resulting drop in model performance.

    Args:
        X: Feature matrix as pandas DataFrame.
        y: Target labels as pandas Series.
        sample_weight: Sample weights.
        t1: End times of each observation's label.
        cv: Number of cross-validation folds.
        clf: Classifier. If None, uses RandomForestClassifier.
        pct_embargo: Fraction of samples to embargo.
        scoring: 'neg_log_loss' or 'accuracy'.
        n_estimators: Number of trees if using default classifier.

    Returns:
        Tuple of (importance_df, mean_base_score).
    """
    if scoring not in ["neg_log_loss", "accuracy"]:
        raise ValueError("Scoring method must be 'neg_log_loss' or 'accuracy'.")

    if clf is None:
        clf = RandomForestClassifier(n_estimators=n_estimators, max_features="sqrt")

    cv_generator = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)
    base_scores = pd.Series(dtype=float)
    permuted_scores = pd.DataFrame(columns=X.columns)

    for fold_index, (train_indices, test_indices) in enumerate(
        cv_generator.split(X=X)
    ):
        X_train, y_train, w_train = (
            X.iloc[train_indices, :],
            y.iloc[train_indices],
            sample_weight.iloc[train_indices],
        )
        X_test, y_test, w_test = (
            X.iloc[test_indices, :],
            y.iloc[test_indices],
            sample_weight.iloc[test_indices],
        )

        model = clf.fit(X=X_train, y=y_train, sample_weight=w_train.values)

        if scoring == "neg_log_loss":
            probabilities = model.predict_proba(X_test)
            base_scores.loc[fold_index] = -log_loss(
                y_test,
                probabilities,
                sample_weight=w_test.values,
                labels=clf.classes_,
            )
        else:
            predictions = model.predict(X_test)
            base_scores.loc[fold_index] = accuracy_score(
                y_test, predictions, sample_weight=w_test.values
            )

        for feature in X.columns:
            X_test_permuted = X_test.copy(deep=True)
            np.random.shuffle(X_test_permuted[feature].values)

            if scoring == "neg_log_loss":
                probabilities = model.predict_proba(X_test_permuted)
                permuted_scores.loc[fold_index, feature] = -log_loss(
                    y_test,
                    probabilities,
                    sample_weight=w_test.values,
                    labels=clf.classes_,
                )
            else:
                predictions = model.predict(X_test_permuted)
                permuted_scores.loc[fold_index, feature] = accuracy_score(
                    y_test, predictions, sample_weight=w_test.values
                )

    importance_scores = (-permuted_scores).add(base_scores, axis=0)
    if scoring == "neg_log_loss":
        importance_scores = importance_scores / -permuted_scores
    else:
        importance_scores = importance_scores / (1.0 - permuted_scores)

    importance_summary = pd.concat(
        {
            "mean": importance_scores.mean(),
            "std": importance_scores.std() * importance_scores.shape[0] ** -0.5,
        },
        axis=1,
    )

    result_df = pl.DataFrame({
        "feature": list(importance_summary.index),
        "mean": importance_summary["mean"].values,
        "std": importance_summary["std"].values,
    })

    return result_df, float(base_scores.mean())


def feature_importance_sfi(
    feature: pl.Series,
    labels: pl.DataFrame,
    cv: int = 4,
    pct_embargo: float = 0.01,
    clf=None,
    scoring: str = "accuracy",
    use_sample_weights: bool = True,
) -> dict[str, float]:
    """Calculate Single Feature Importance (SFI) using purged cross-validation.

    Args:
        feature: Polars Series of feature values.
        labels: Polars DataFrame with columns: date, exit_date, label, returns.
        cv: Number of cross-validation folds.
        pct_embargo: Fraction of observations to embargo.
        clf: Classifier. If None, uses RandomForestClassifier.
        scoring: 'neg_log_loss' or 'accuracy'.
        use_sample_weights: If True, compute sample weights via average uniqueness.

    Returns:
        Dictionary with mean and std of cross-validation scores.
    """
    from jarjarquant.labelling import num_co_events, average_uniqueness

    dates_pd = labels["date"].to_pandas()
    exit_dates_pd = labels["exit_date"].to_pandas()
    t1 = pd.Series(exit_dates_pd.values, index=dates_pd.values)

    if use_sample_weights:
        all_dates = np.sort(
            np.unique(np.concatenate([dates_pd.values, exit_dates_pd.values]))
        )
        date_array = all_dates[~pd.isna(all_dates)]

        event_dates = dates_pd.values
        exit_dates = exit_dates_pd.values

        co_events = num_co_events(date_array, event_dates, exit_dates)
        sw_values = average_uniqueness(date_array, event_dates, exit_dates, co_events)
        sw = pd.Series(sw_values, index=dates_pd.values)
    else:
        sw = pd.Series(1.0, index=dates_pd.values)

    feature_name = feature.name or "feature"
    X = pd.DataFrame(
        {feature_name: feature.to_pandas().values}, index=dates_pd.values
    )
    y = pd.Series(labels["label"].to_numpy(), index=dates_pd.values)

    scores = cv_score(
        clf,
        X=X,
        y=y,
        sample_weight=sw,
        t1=t1,
        cv=cv,
        scoring=scoring,
        pct_embargo=pct_embargo,
    )

    return {
        "mean": float(scores.mean()),
        "std": float(scores.std() * scores.shape[0] ** -0.5),
    }
