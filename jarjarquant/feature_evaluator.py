"""The feature evaluator specializes in calculating the efficacy of one or many indicators given a matrix of features X and a target label/series y"""

import concurrent.futures
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pandas as pd
import polars as pl
from indicators.base import IndicatorEvalResult
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection._split import _BaseKFold

from jarjarquant.core.utils import _flatten_dataclass
from jarjarquant.cython_utils.opt_threshold import optimize_threshold_cython

from .data_analyst import get_spearman_correlation, plot_loess
from .data_gatherer import DataGatherer
from .data_service import DataService, SampleRequest

if TYPE_CHECKING:
    # Imported only for type checking to avoid circular import at runtime
    from .indicators.base import IndicatorSpec


@dataclass
class EvalResult:
    """Dataclass to hold evaluation results for indicators"""

    indicator_spec: "IndicatorSpec"
    sample_request: SampleRequest
    results_dict: dict[str, IndicatorEvalResult | None]


def _format_ind_dist_outputs(basic_outputs_list: list) -> list:
    adf_test = [output[0] for output in basic_outputs_list]
    jb_test = [output[1] for output in basic_outputs_list]
    relative_entropy = [output[2] for output in basic_outputs_list]
    range_iqr_ratio = [output[3] for output in basic_outputs_list]

    final_results = [
        np.mean(adf_test),
        np.mean(jb_test),
        np.mean([x for x in relative_entropy if not (np.isnan(x) or np.isinf(x))]),
        np.mean([x for x in range_iqr_ratio if not (np.isnan(x) or np.isinf(x))]),
    ]
    final_results = [round(value, 2) for value in final_results]

    return final_results


### PurgedKFold Class as implemented in advances in financial machine learning ###
# Define the PurgedKFold class for feature importance scores
class PurgedKFold(_BaseKFold):
    """
    Custom KFold class to handle overlapping label intervals in time-series data.

    This class extends the basic KFold functionality to ensure that:
    - The training set does not contain observations that overlap with the test label intervals.
    - An embargo period is applied after each test interval to prevent look-ahead bias.

    Attributes:
        t1 (pd.Series): Series of end times for each observation's label.
        pct_embargo (float): Fraction of observations to embargo after the test interval.
    """

    def __init__(self, n_splits=3, t1=None, pct_embargo=0.0):
        """
        Initialize PurgedKFold with the number of splits, through-dates, and embargo percentage.

        Args:
            n_splits (int): Number of folds for cross-validation.
            t1 (pd.Series): Series of end times for each observation.
            pct_embargo (float): Fraction of observations to embargo after each test interval.
        """
        if not isinstance(t1, pd.Series):
            raise ValueError("Label through-dates must be a pandas Series.")

        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        """
        Generate indices for training and test splits with purge and embargo applied.

        Args:
            X (pd.DataFrame): Input data with an index matching `t1`.
            y (pd.Series, optional): Target values (not used in splitting).
            groups (None): Not used, only for compatibility.

        Yields:
            train_indices (np.array): Indices for the training set in the current fold.
            test_indices (np.array): Indices for the test set in the current fold.
        """
        if (X.index == self.t1.index).sum() != len(self.t1):
            raise ValueError("X and t1 must have the same index.")

        indices = np.arange(X.shape[0])
        embargo_size = int(X.shape[0] * self.pct_embargo)

        # Define test intervals for each fold
        test_intervals = [
            (i[0], i[-1] + 1)
            for i in np.array_split(np.arange(X.shape[0]), self.n_splits)
        ]

        for start_idx, end_idx in test_intervals:
            # Identify test indices and corresponding max label end time
            t0 = self.t1.index[start_idx]
            test_indices = indices[start_idx:end_idx]
            max_t1_idx = self.t1.index.searchsorted(self.t1.iloc[test_indices].max())

            # Train indices: observations ending before the test set starts
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)

            # Include embargo if max_t1_idx is within bounds
            if max_t1_idx < X.shape[0]:
                train_indices = np.concatenate(
                    (train_indices, indices[max_t1_idx + embargo_size :])
                )

            yield train_indices, test_indices


### FeatureEvaluator Class ###
class FeatureEvaluator:
    """Class to implement common feature evaluation and indicator testing methods"""

    ### Class Methods ###
    def __init__(self, X=None, y=None, sw=None):
        self.X = X
        self.y = y
        self.sw = sw
        self._ds = None  # Lazy initialization of DataService

    @property
    def ds(self):
        """Lazily initialize DataService when first accessed."""
        if self._ds is None:
            self._ds = DataService()
        return self._ds

    # Feature Importance Methods from Advances in Financial Machine Learning ###
    @staticmethod
    def cv_score(
        clf,
        X,
        y,
        sample_weight,
        scoring="neg_log_loss",
        t1=None,
        cv=None,
        cv_gen=None,
        pct_embargo=None,
    ):
        """
        Calculate cross-validation scores using a classifier with purged k-fold splits and optional embargo.

        This function implements purged cross-validation with embargo handling, useful for time-series data
        where test and train sets should be separated by a buffer to prevent leakage.

        Args:
            clf (object): Classifier with `fit`, `predict`, and optionally `predict_proba` methods.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target labels.
            sample_weight (pd.Series): Sample weights for each observation.
            scoring (str): Scoring method, either 'neg_log_loss' or 'accuracy'.
            t1 (pd.Series, optional): End times of each observation's label.
            cv (int, optional): Number of cross-validation folds.
            cv_gen (PurgedKFold, optional): Custom cross-validation generator.
            pct_embargo (float, optional): Fraction of samples to embargo after each test interval.

        Returns:
            np.array: Array of scores for each cross-validation fold.
        """
        if scoring not in ["neg_log_loss", "accuracy"]:
            raise ValueError("Scoring method must be 'neg_log_loss' or 'accuracy'.")

        # Initialize purged cross-validation generator if not provided
        if cv_gen is None:
            cv_gen = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)

        # Use a RandomForestClasifier if clf is None
        if clf is None:
            clf = RandomForestClassifier(
                n_estimators=100, max_features="sqrt", random_state=42
            )

        scores = []  # List to store scores for each fold

        # Cross-validation loop
        for train_indices, test_indices in cv_gen.split(X=X):
            # Train the classifier on the training set
            model = clf.fit(
                X=X.iloc[train_indices, :],
                y=y.iloc[train_indices],
                sample_weight=sample_weight.iloc[train_indices].values,
            )

            # Evaluate the model on the test set
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

    def feature_importance_MDI(self, fit, feature_names):
        """
        Calculates feature importance based on Mean Decrease Impurity (MDI) for tree-based models.

        Parameters:
        -----------
        fit : sklearn.tree._forest.ForestClassifier or ForestRegressor
            A fitted ensemble model (e.g., RandomForestClassifier) with an 'estimators_' attribute
            containing individual decision trees.

        featureNames : list of str
            A list of feature names to label the output DataFrame columns.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the mean and standard deviation of feature importances,
            normalized so that the mean importances sum to 1.
        """

        # Collect feature importances from each tree in the ensemble
        importance_dict = {
            i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)
        }

        # Convert the dictionary to a DataFrame, with rows as trees and columns as features
        importance_df = pd.DataFrame.from_dict(importance_dict, orient="index")
        importance_df.columns = feature_names

        # Replace zeros with NaN to handle cases where max_features=1, preventing distortions in the mean calculation
        importance_df = importance_df.replace(0, np.nan)

        # Calculate mean and standard deviation of feature importances across trees
        importance_stats = pd.concat(
            {
                "mean": importance_df.mean(),
                "std": importance_df.std() * (importance_df.shape[0] ** -0.5),
            },
            axis=1,
        )

        # Normalize the mean importances to sum to 1
        importance_stats["mean"] /= importance_stats["mean"].sum()

        return importance_stats

    def feature_importance_MDA(
        self,
        X,
        y,
        sample_weight,
        t1,
        cv: int = 4,
        clf=None,
        pct_embargo=0.04,
        scoring="neg_log_loss",
    ):
        """
        Calculate feature importance using Mean Decrease Accuracy (MDA) with purged cross-validation.

        This method permutes each feature to measure the resulting drop in model performance,
        indicating how essential each feature is for the model's predictions.

        Args:
            clf (object): Classifier implementing `fit` and `predict` methods.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target labels.
            cv (int): Number of cross-validation folds.
            sample_weight (pd.Series): Sample weights for each observation.
            t1 (pd.Series): End times of each observation's label.
            pct_embargo (float): Fraction of samples to embargo after each test interval.
            scoring (str): Scoring method, either 'neg_log_loss' or 'accuracy'.

        Returns:
            pd.DataFrame: Mean and standard deviation of feature importance scores.
            float: Mean score of the original, unpermuted model.
        """
        if scoring not in ["neg_log_loss", "accuracy"]:
            raise ValueError("Scoring method must be 'neg_log_loss' or 'accuracy'.")

        # Use a RandomForestClasifier if clf is None
        if clf is None:
            clf = RandomForestClassifier(n_estimators=100, max_features="sqrt")

        # Initialize purged cross-validation generator
        cv_generator = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)
        base_scores = pd.Series()  # scores for the original, unpermuted model
        # scores for each permuted feature
        permuted_scores = pd.DataFrame(columns=X.columns)

        # Cross-validation loop
        for fold_index, (train_indices, test_indices) in enumerate(
            cv_generator.split(X=X)
        ):
            # Split data into train and test sets
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

            # Fit classifier on the training set
            model = clf.fit(X=X_train, y=y_train, sample_weight=w_train.values)

            # Score the model on the test set
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

            # Permute each feature and calculate the impact on model performance
            for feature in X.columns:
                X_test_permuted = X_test.copy(deep=True)
                # Permute one feature at a time
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

        # Calculate feature importance as the relative decrease in accuracy
        importance_scores = (-permuted_scores).add(base_scores, axis=0)
        if scoring == "neg_log_loss":
            importance_scores = importance_scores / -permuted_scores
        else:
            importance_scores = importance_scores / (1.0 - permuted_scores)

        # Aggregate mean and standard error for each feature's importance across folds
        importance_summary = pd.concat(
            {
                "mean": importance_scores.mean(),
                "std": importance_scores.std() * importance_scores.shape[0] ** -0.5,
            },
            axis=1,
        )

        return importance_summary, base_scores.mean()

    def feature_importance_SFI(
        self,
        feature_names,
        X,
        y,
        sw,
        t1,
        cv: int = 4,
        pct_embargo: float = 0.04,
        clf=None,
        cv_gen=None,
        scoring="accuracy",
    ):
        """
        Calculate Single Feature Importance (SFI) scores for each feature using cross-validation.

        This function evaluates the importance of each feature independently by training and scoring
        a model on only that feature in a cross-validation loop, providing a measure of each feature's
        contribution to the model.

        Args:
            feature_names (list): List of feature names to evaluate.
            clf (object): Classifier with `fit`, `predict`, and/or `predict_proba` methods.
            transformed_X (pd.DataFrame): Feature matrix.
            cont (pd.DataFrame): DataFrame containing target values ('bin') and sample weights ('w').
            scoring (str): Scoring method to use, either 'neg_log_loss' or 'accuracy'.
            cv_gen (PurgedKFold): Cross-validation generator with purged k-fold splits.

        Returns:
            pd.DataFrame: DataFrame with mean and standard deviation of SFI scores for each feature.
        """
        importance_scores = pd.DataFrame(columns=["mean", "std"])

        # Loop through each feature and calculate its importance using cross-validation
        for feature_name in feature_names:
            # Calculate cross-validation scores using only the current feature
            feature_scores = self.cv_score(
                clf,
                X=X[[feature_name]],  # Single feature DataFrame
                y=y,
                sample_weight=sw,
                t1=t1,
                cv=cv,
                scoring=scoring,
                cv_gen=cv_gen,
                pct_embargo=pct_embargo,
            )

            # Record mean and standard deviation of scores for the feature
            importance_scores.loc[feature_name, "mean"] = feature_scores.mean()
            importance_scores.loc[feature_name, "std"] = (
                feature_scores.std() * feature_scores.shape[0] ** -0.5
            )

        return importance_scores

    ### Indicator Distribution (Statistical) Methods - Indicator Design Analysis ###
    @staticmethod
    def co_distribution_analysis(
        indicator_values: np.ndarray, associated_returns: np.ndarray
    ):
        # Use standalone function instead of DataAnalyst class
        indicator_values = np.asarray(indicator_values)
        associated_returns = np.asarray(associated_returns)

        spearman_results = get_spearman_correlation(
            indicator_values, associated_returns
        )

        # 1. Plot a LOESS scatter plot of the indicator values and associated returns
        plot_loess(
            x=indicator_values,
            y=associated_returns,
            x_label="Indicator Values",
            y_label="Returns",
            title="Overall LOESS Scatter Plot",
            annotation=spearman_results["spearman_corr"],
        )

        # 2. Co sort the returns and indicator values and then create 4 LOESS plots for each quartile of indicator values
        sorted_indices = np.argsort(indicator_values)
        sorted_indicator = indicator_values[sorted_indices]
        sorted_returns = associated_returns[sorted_indices]

        n = len(sorted_indicator)
        # Define bin edges for 4 equal-sized bins
        bin_edges = np.linspace(0, n, 5, dtype=int)

        for i in range(4):
            start = bin_edges[i]
            end = bin_edges[i + 1]
            s1_bin = sorted_indicator[start:end]
            s2_bin = sorted_returns[start:end]

            plot_loess(
                x=s1_bin,
                y=s2_bin,
                x_label="Indicator Values",
                y_label="Returns",
                title=f"Quartile {i + 1} LOESS Scatter Plot",
                annotation=spearman_results["spearman_corr_quartile"][i],
            )

    @staticmethod
    def indicator_distribution_study(inputs: dict) -> dict:
        indicator_spec = inputs["indicator_spec"]
        ohlcv_df = inputs["ohlcv_df"]
        include_detailed_data = inputs.get("include_detailed_data", False)

        # Timer: Indicator creation
        indicator_instance = indicator_spec.create_indicator(ohlcv_df)

        # Timer: Indicator evaluation report
        indicator_instance.indicator_evaluation_report()

        # Basic outputs for backward compatibility
        basic_outputs = [
            indicator_instance.eval_result.adf_test.is_stationary,
            indicator_instance.eval_result.jb_normality_test.is_normal,
            indicator_instance.eval_result.relative_entropy.entropy,
            indicator_instance.eval_result.range_iqr_ratio.ratio,
        ]

        if include_detailed_data:
            # Return detailed data for database saving
            return {
                "basic_outputs": basic_outputs,
                "ticker": inputs.get("ticker", ""),
                "indicator_spec": indicator_spec,
                "eval_result": indicator_instance.eval_result,
            }
        else:
            # Return just basic outputs for backward compatibility
            return {"basic_outputs": basic_outputs}

    def parallel_indicator_distribution_study(
        self,
        indicator_spec: "IndicatorSpec",
        sample_request: Optional[SampleRequest],
        save_run: bool = True,
    ):
        start_time = time.time()
        logger = logging.getLogger(__name__)

        # Timer: Sample request setup
        sample_setup_start = time.time()
        if sample_request is None:
            sample_request = SampleRequest(
                sample_type="equities",
                start_date=(pd.Timestamp.now() - pd.Timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
                end_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
            )
        sample_setup_time = time.time() - sample_setup_start
        logger.info(f"Sample request setup time: {sample_setup_time:.4f}s")

        # Check if results already exist in database
        logger.info("Checking for existing results in database...")
        existing_results = self._check_existing_results(sample_request, indicator_spec)

        if existing_results is not None:
            total_time = time.time() - start_time
            logger.info(
                f"Found existing results in database! Total time: {total_time:.4f}s"
            )
            return existing_results

        logger.info("No existing results found. Running new study...")

        # Timer: Data gathering
        data_gather_start = time.time()
        sample = self.ds.get_sample(sample_request)
        df = sample.data
        data_gather_time = time.time() - data_gather_start
        logger.info(f"Data gathering time: {data_gather_time:.4f}s")

        # Timer: Data preparation
        data_prep_start = time.time()
        grouped_data = list(df.group_by("ticker", maintain_order=True))
        inputs_list = []

        for ticker, ohlcv_df in grouped_data:
            inputs = {
                "indicator_spec": indicator_spec,
                "ohlcv_df": ohlcv_df,
                "ticker": ticker[0],
                "include_detailed_data": save_run,
            }
            inputs_list.append(inputs)
        data_prep_time = time.time() - data_prep_start
        logger.info(f"Data preparation time: {data_prep_time:.4f}s")
        logger.info(f"Processing {len(inputs_list)} ticker datasets")

        # Timer: Parallel processing
        parallel_start = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(
                executor.map(FeatureEvaluator.indicator_distribution_study, inputs_list)
            )
        parallel_time = time.time() - parallel_start
        logger.info(f"Parallel processing time: {parallel_time:.4f}s")

        # Extract outputs and save detailed data if requested
        if save_run:
            # Save detailed data to database
            self._save_detailed_results(results, sample_request, indicator_spec)

            # Extract basic outputs for aggregation
            basic_outputs_list = [result["basic_outputs"] for result in results]
        else:
            # Handle backward compatibility - results are basic outputs
            basic_outputs_list = [
                result["basic_outputs"] if isinstance(result, dict) else result
                for result in results
            ]

        final_results = _format_ind_dist_outputs(basic_outputs_list)

        total_time = time.time() - start_time
        logger.info(
            f"Total parallel_indicator_distribution_study time: {total_time:.4f}s"
        )

        return {
            "ADF Test": final_results[0],
            "Jarque-Bera Test": final_results[1],
            "Relative Entropy": final_results[2],
            "Range-IQR Ratio": final_results[3],
        }

    ### Indicator Performance Analysis Methods - Study joint properties with returns ###
    @staticmethod
    def indicator_threshold_search(
        indicator_values: pd.Series,
        associated_returns: pd.Series,
        thresholds: Optional[list] = None,
        n_thresholds: Optional[int] = 10,
        threshold_option: Optional[str] = "percentile",
    ):
        """
        Evaluate profit factors for different thresholds of an indicator.

        Parameters:
        indicator_values (pd.Series): Series of indicator values.
        associated_returns (pd.Series): Series of returns associated with the indicator values.
        n_thresholds (int, optional): Number of thresholds to evaluate.
        thresholds (list, optional): List of predefined threshold values.
        threshold_option (str, optional): Method to calculate thresholds ('linear' or 'percentile').

        Returns:
        pd.DataFrame: DataFrame with thresholds and profit factors for long/short positions above/below the thresholds.
        """
        # Ensure the inputs are of the same length
        if len(indicator_values) != len(associated_returns):
            raise ValueError(
                "indicator_values and associated_returns must have the same length."
            )

        if thresholds is None:
            if n_thresholds is None:
                raise ValueError("Either n_thresholds or thresholds must be provided.")
            if threshold_option == "linear":
                # Calculate threshold values
                min_val, max_val = indicator_values.min(), indicator_values.max()
                # Exclude min and max values
                thresholds = np.linspace(min_val, max_val, n_thresholds + 2)[1:-1]
            elif threshold_option == "percentile":
                # Calculate threshold values using percentiles
                percentiles = np.linspace(0, 100, n_thresholds + 2)[1:-1]
                thresholds = np.percentile(indicator_values, percentiles)
            else:
                raise ValueError("threshold_option must be 'linear' or 'percentile'.")

        results = []

        for threshold in thresholds:
            # Calculate profit factors for long/short positions above/below the threshold
            above_threshold = indicator_values > threshold
            below_threshold = indicator_values < threshold

            # ABove threshold
            pf_long_above = (
                associated_returns[above_threshold & (associated_returns > 0)].sum()
                / -associated_returns[above_threshold & (associated_returns < 0)].sum()
            )
            pf_short_above = (
                -associated_returns[above_threshold & (associated_returns < 0)].sum()
                / associated_returns[above_threshold & (associated_returns > 0)].sum()
            )

            # Calculate mean and median return, standard deviation of return, 25th percentile and 75th percentile return above each threshold
            mean_return_above = associated_returns[above_threshold.values].mean()
            std_return_above = associated_returns[above_threshold.values].std()
            median_return_above = associated_returns[above_threshold.values].median()
            q25_return_above = associated_returns[above_threshold.values].quantile(0.25)
            q75_return_above = associated_returns[above_threshold.values].quantile(0.75)

            # Calculate spearman rank correlation above threshold
            spearman_corr_above, _ = spearmanr(
                indicator_values[above_threshold.values],
                associated_returns[above_threshold.values],
            )

            # Below threshold
            pf_long_below = (
                associated_returns[below_threshold & (associated_returns > 0)].sum()
                / -associated_returns[below_threshold & (associated_returns < 0)].sum()
            )
            pf_short_below = (
                -associated_returns[below_threshold & (associated_returns < 0)].sum()
                / associated_returns[below_threshold & (associated_returns > 0)].sum()
            )

            # Calculate mean and median return, standard deviation of return, 25th percentile and 75th percentile return below each threshold
            mean_return_below = associated_returns[below_threshold.values].mean()
            std_return_below = associated_returns[below_threshold.values].std()
            median_return_below = associated_returns[below_threshold.values].median()
            q25_return_below = associated_returns[below_threshold.values].quantile(0.25)
            q75_return_below = associated_returns[below_threshold.values].quantile(0.75)

            # Calculate spearman rank correlation below threshold
            spearman_corr_below, _ = spearmanr(
                indicator_values[below_threshold.values],
                associated_returns[below_threshold.values],
            )

            results.append(
                {
                    "Threshold": threshold,
                    "% values > threshold": above_threshold.mean() * 100,
                    "Spearman correlation above threshold": spearman_corr_above,
                    "Mean return above threshold": mean_return_above,
                    "Std dev return above threshold": std_return_above,
                    "Median return above threshold": median_return_above,
                    "Q25 return above threshold": q25_return_above,
                    "Q75 return above threshold": q75_return_above,
                    "PF Long above threshold": np.nan_to_num(
                        pf_long_above, nan=0.0, posinf=0.0, neginf=0.0
                    ),
                    "PF Short above threshold": np.nan_to_num(
                        pf_short_above, nan=0.0, posinf=0.0, neginf=0.0
                    ),
                    "% values < threshold": below_threshold.mean() * 100,
                    "Spearman correlation below threshold": spearman_corr_below,
                    "Mean return below threshold": mean_return_below,
                    "Std dev return below threshold": std_return_below,
                    "Median return below threshold": median_return_below,
                    "Q25 return below threshold": q25_return_below,
                    "Q75 return below threshold": q75_return_below,
                    "PF Long below threshold": np.nan_to_num(
                        pf_long_below, nan=0.0, posinf=0.0, neginf=0.0
                    ),
                    "PF Short below threshold": np.nan_to_num(
                        pf_short_below, nan=0.0, posinf=0.0, neginf=0.0
                    ),
                }
            )

        return pd.DataFrame(results)

    @staticmethod
    def single_indicator_threshold_search(inputs: dict):
        ohlcv_df = inputs["ohlcv_df"]
        indicator_values = inputs["indicator_values"]
        thresholds = inputs["thresholds"]

        ohlcv_df["returns"] = ohlcv_df["Open"].pct_change().shift(-1)
        ohlcv_df["ind"] = indicator_values
        ohlcv_df["ind"] = ohlcv_df["ind"].shift(1)

        results = FeatureEvaluator.indicator_threshold_search(
            indicator_values=ohlcv_df["ind"].dropna(),
            associated_returns=ohlcv_df["returns"].dropna(),
            thresholds=thresholds,
        )

        return results

    @staticmethod
    def parallel_indicator_threshold_search(
        indicator_func: Callable, n_runs: int = 10, n_thresholds: int = 10, **kwargs
    ):
        inputs_list = []
        indicator_values_list = []

        # Generate dataframes and calculate indicator values in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            data_gatherer = DataGatherer()
            futures = [
                executor.submit(
                    data_gatherer.get_random_price_samples_tws, num_tickers_to_sample=1
                )
                for _ in range(n_runs)
            ]
            for future in concurrent.futures.as_completed(futures):
                ohlcv_df = future.result()[0]
                indicator_values = (
                    indicator_func(ohlcv_df, **kwargs).calculate()
                    if kwargs
                    else indicator_func(ohlcv_df).calculate()
                )
                indicator_values_list.append(indicator_values)
                inputs_list.append(
                    {"ohlcv_df": ohlcv_df, "indicator_values": indicator_values}
                )

        # Determine thresholds based on the entire range of indicator values across all runs
        all_indicator_values = np.concatenate(indicator_values_list)
        # thresholds = np.linspace(all_indicator_values.min(
        # ), all_indicator_values.max(), n_thresholds + 2)[1:-1]

        # Implement percentile thresholds
        percentiles = np.linspace(0, 100, n_thresholds + 2)[1:-1]
        thresholds = np.percentile(all_indicator_values, percentiles)

        # Update inputs with calculated thresholds
        for inputs in inputs_list:
            inputs["thresholds"] = thresholds

        # Run threshold search in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(
                executor.map(
                    FeatureEvaluator.single_indicator_threshold_search, inputs_list
                )
            )

        # Concatenate results and average across runs
        results = pd.concat(results).groupby("Threshold").mean().reset_index()

        return results

    @staticmethod
    def optimize_threshold(
        indicator_values,
        return_values,
        min_kept: float = 0.1,
        flip_sign: bool = False,
        return_pval: bool = True,
    ):
        """
        Optimize the threshold for a given indicator to maximize the performance factor (PF).
        Parameters:
        indicator_values (array-like): Array of indicator values.
        return_values (array-like): Array of return values corresponding to the indicator values.
        min_kept (float, optional): Minimum fraction of data points to keep. Default is 0.1.
        flip_sign (bool, optional): Whether to flip the sign of the indicator values. Default is False.
        Returns:
        dict: A dictionary containing the following keys:
            - 'spearman_corr': Spearman rank correlation between the indicator and returns.
            - 'optimal_long_thresh': Optimal threshold for long positions.
            - 'optimal_long_pf': Performance factor for the optimal long threshold.
            - 'optimal_short_thresh': Optimal threshold for short positions.
            - 'optimal_short_pf': Performance factor for the optimal short threshold.
            - 'best_bf': Best performance factor between long and short positions.
            - 'best_pf_pval': P-value of the best performance factor.
        Raises:
        ValueError: If the input arrays have less than one element.
        """

        # Ensure the inputs are numpy arrays.
        indicator_values = np.asarray(indicator_values)
        return_values = np.asarray(return_values)

        n = len(indicator_values)
        if n == 0:
            raise ValueError("Input arrays must have at least one element.")

        # Enforce that min_kept is at least 1.
        min_kept = max(int(n * min_kept), 1)

        # Calculate the spearman rank correlation between the indicator and returns.
        spearman_result = spearmanr(indicator_values, return_values)
        spearman_corr = (
            spearman_result[0]
            if isinstance(spearman_result, tuple)
            else spearman_result
        )
        if spearman_corr < 0.0:
            indicator_sign = -1.0
        else:
            indicator_sign = 1.0

        # Copy signals and returns into work arrays.
        # Optionally flip the sign of indicator values.
        if flip_sign:
            work_signal = -indicator_sign * indicator_values.copy()
        else:
            work_signal = indicator_sign * indicator_values.copy()
        work_return = return_values.copy()

        # Find the indices of NaN values in either array and drop them from both arrays
        nan_indices = np.isnan(work_signal) | np.isnan(work_return)
        work_signal = work_signal[~nan_indices]
        work_return = work_return[~nan_indices]

        n = len(work_signal)

        # Sort the work arrays based on work_signal.
        sort_index = np.argsort(work_signal)
        work_signal = work_signal[sort_index]
        work_return = work_return[sort_index]

        (
            best_high_index,
            best_low_index,
            best_high_pf,
            best_low_pf,
            best_high_acc,
            best_low_acc,
        ) = optimize_threshold_cython(work_signal, work_return, int(min_kept))

        # The best thresholds are the signal values at the recorded indices.
        high_thresh = work_signal[best_high_index]
        low_thresh = work_signal[best_low_index]
        pf_high = best_high_pf
        pf_low = best_low_pf
        best_overall_pf = max(pf_high, pf_low)

        # Calculate the p-value for the best performance factor.
        if return_pval:
            i = 0

            for _ in range(1000):
                permuted_returns = np.random.choice(
                    work_return, size=len(work_return), replace=True
                )
                _, _, high_pf, low_pf, _, _ = optimize_threshold_cython(
                    work_signal, permuted_returns, int(min_kept)
                )
                permuted_pf = max(high_pf, low_pf)
                if permuted_pf >= best_overall_pf:
                    i += 1

            best_pf_pval = i / 1000

        else:
            best_pf_pval = None

        return {
            "spearman_corr": spearman_corr,
            "optimal_long_thresh": high_thresh,
            "optimal_long_pf": pf_high,
            "optimal_long_acc": best_high_acc,
            "optimal_short_thresh": low_thresh,
            "optimal_short_pf": pf_low,
            "optimal_short_acc": best_low_acc,
            "best_pf": best_overall_pf,
            "best_pf_pval": best_pf_pval,
        }

    @staticmethod
    def threshold_optimization_study(inputs: dict) -> dict:
        indicator_spec = inputs["indicator_spec"]
        ohlcv_df = inputs["ohlcv_df"]
        ticker = inputs.get("ticker", "")

        # Timer: Indicator creation
        indicator_instance = indicator_spec.create_indicator(ohlcv_df)

        # Calculate indicator values
        indicator_values = indicator_instance.calculate()

        # Prepare returns
        returns = ohlcv_df["Open"].pct_change().shift(-1)

        # Optimize threshold
        optimization_results = FeatureEvaluator.optimize_threshold(
            indicator_values=indicator_values,
            return_values=returns,
            min_kept=0.1,
            flip_sign=False,
            return_pval=True,
        )

        # Return detailed data for database saving
        return {
            "ticker": ticker,
            "indicator_spec": indicator_spec,
            "optimization_results": optimization_results,
        }

    def parallel_threshold_optimization_study(
        self,
        indicator_spec: "IndicatorSpec",
        sample_request: Optional[SampleRequest],
        save_run: bool = False,
    ):
        start_time = time.time()
        logger = logging.getLogger(__name__)

        # Timer: Sample request setup
        sample_setup_start = time.time()
        if sample_request is None:
            sample_request = SampleRequest(
                sample_type="equities",
                start_date=(pd.Timestamp.now() - pd.Timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
                end_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
            )
        sample_setup_time = time.time() - sample_setup_start
        logger.info(f"Sample request setup time: {sample_setup_time:.4f}s")

        # Timer: Data gathering
        data_gather_start = time.time()
        sample = self.ds.get_sample(sample_request)
        df = sample.data
        data_gather_time = time.time() - data_gather_start
        logger.info(f"Data gathering time: {data_gather_time:.4f}s")

        # Timer: Data preparation
        data_prep_start = time.time()
        grouped_data = list(df.group_by("ticker", maintain_order=True))
        inputs_list = []

        for ticker, ohlcv_df in grouped_data:
            inputs = {
                "indicator_spec": indicator_spec,
                "ohlcv_df": ohlcv_df,
                "ticker": ticker[0],
            }
            inputs_list.append(inputs)
        data_prep_time = time.time() - data_prep_start
        logger.info(f"Data preparation time: {data_prep_time:.4f}s")
        logger.info(f"Processing {len(inputs_list)} ticker datasets")

        # Timer: Parallel processing
        parallel_start = time.time()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(
                executor.map(FeatureEvaluator.threshold_optimization_study, inputs_list)
            )
        parallel_time = time.time() - parallel_start
        logger.info(f"Parallel processing time: {parallel_time:.4f}s")

        # # Extract outputs and save detailed data if requested
        # if save_run:
        #     # Save detailed data to database
        #     self._save_optimization_results(results, sample_request, indicator_spec)

        total_time = time.time() - start_time
        logger.info(
            f"Total parallel_threshold_optimization_study time: {total_time:.4f}s"
        )

        return results

    ### Helper Methods ###
    def _save_detailed_results(
        self,
        db_name: str,
        results: EvalResult,
    ) -> None:
        """
        Save detailed results to the database.

        Args:
            db_name: The name of the database to save results to
            results: EvalResult dataclass containing indicator_spec, sample_request, and results_dict
        """
        rows = []

        for ticker, eval_result in results.results_dict.items():
            if eval_result is None:
                continue

            # Create a row dictionary
            row = {}
            row.update(_flatten_dataclass(results.indicator_spec))
            row.update(_flatten_dataclass(results.sample_request))
            row["ticker"] = ticker
            row["timestamp"] = datetime.now().isoformat()
            row.update(_flatten_dataclass(eval_result))
            rows.append(row)

        if rows:
            # Convert to Polars DataFrame and save
            df = pl.DataFrame(rows)
            self.ds.save_to_database(df, db_name, append=True)
            logging.getLogger(__name__).info(
                f"Saved {len(rows)} rows to {db_name} table"
            )
        else:
            raise ValueError(
                "Error saving evaluation results: No valid rows were extracted."
            )

    def _check_existing_results(
        self, sample_request: SampleRequest, indicator_spec: "IndicatorSpec"
    ) -> Optional[dict]:
        """
        Check if results already exist in the database for the given sample request and indicator spec.

        Args:
            sample_request: The sample request parameters
            indicator_spec: The indicator specification

        Returns:
            Dictionary with aggregated results if found, None otherwise
        """

        import duckdb

        logger = logging.getLogger(__name__)
        db_file_path = self.ds.db_path / "ind_dist_studies_db.duckdb"

        if not db_file_path.exists():
            logger.debug("Database file does not exist")
            return None

        try:
            # Connect to the database
            conn = duckdb.connect(str(db_file_path))

            try:
                # Check if table exists
                result = conn.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'ind_dist_studies_db'"
                ).fetchone()

                if result is None or result[0] == 0:
                    logger.debug("Table ind_dist_studies_db does not exist")
                    return None

                # Build query parameters
                indicator_name = (
                    indicator_spec.indicator_type.value
                    if hasattr(indicator_spec.indicator_type, "value")
                    else str(indicator_spec.indicator_type)
                )

                bar_size = (
                    sample_request.bar_size.value
                    if hasattr(sample_request.bar_size, "value")
                    else str(sample_request.bar_size)
                )

                indicator_params_json = json.dumps(indicator_spec.parameters)

                # Query for existing results
                query = """
                SELECT 
                    adf_is_stationary,
                    jb_is_normal,
                    entropy,
                    range_iqr_ratio
                FROM ind_dist_studies_db 
                WHERE indicator = ? 
                    AND bar_size = ? 
                    AND start_date = ? 
                    AND end_date = ? 
                    AND indicator_params = ?
                """

                result_df = conn.execute(
                    query,
                    [
                        indicator_name,
                        bar_size,
                        sample_request.start_date,
                        sample_request.end_date,
                        indicator_params_json,
                    ],
                ).pl()

                if result_df.height == 0:
                    logger.debug("No existing results found for the given parameters")
                    return None

                # Calculate aggregated results
                adf_tests = result_df["adf_is_stationary"].to_list()
                jb_tests = result_df["jb_is_normal"].to_list()
                relative_entropies = result_df["entropy"].to_list()
                range_iqr_ratios = result_df["range_iqr_ratio"].to_list()

                # Filter out NaN and inf values for entropy and range_iqr_ratio
                filtered_entropies = [
                    x for x in relative_entropies if not (np.isnan(x) or np.isinf(x))
                ]
                filtered_ratios = [
                    x for x in range_iqr_ratios if not (np.isnan(x) or np.isinf(x))
                ]

                final_results = [
                    np.mean(adf_tests),
                    np.mean(jb_tests),
                    np.mean(filtered_entropies) if filtered_entropies else 0.0,
                    np.mean(filtered_ratios) if filtered_ratios else 0.0,
                ]
                final_results = [round(value, 2) for value in final_results]

                logger.info(f"Found {result_df.height} existing results in database")

                return {
                    "ADF Test": final_results[0],
                    "Jarque-Bera Test": final_results[1],
                    "Relative Entropy": final_results[2],
                    "Range-IQR Ratio": final_results[3],
                }

            finally:
                conn.close()

        except Exception as e:
            logger.error(f"Error checking existing results: {e}")
            return None
