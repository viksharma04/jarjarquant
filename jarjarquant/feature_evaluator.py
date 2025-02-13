"""The feature evaluator specializes in calculating the efficacy of one or many indicators given a matrix of features X and a target label/series y"""
import concurrent.futures
from typing import Callable, Optional

from jarjarquant.cython_utils.opt_threshold import optimize_threshold_cython

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection._split import _BaseKFold

from .data_gatherer import DataGatherer

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
            (i[0], i[-1] + 1) for i in np.array_split(np.arange(X.shape[0]), self.n_splits)
        ]

        for start_idx, end_idx in test_intervals:
            # Identify test indices and corresponding max label end time
            t0 = self.t1.index[start_idx]
            test_indices = indices[start_idx:end_idx]
            max_t1_idx = self.t1.index.searchsorted(
                self.t1.iloc[test_indices].max())

            # Train indices: observations ending before the test set starts
            train_indices = self.t1.index.searchsorted(
                self.t1[self.t1 <= t0].index)

            # Include embargo if max_t1_idx is within bounds
            if max_t1_idx < X.shape[0]:
                train_indices = np.concatenate(
                    (train_indices, indices[max_t1_idx + embargo_size:])
                )

            yield train_indices, test_indices


class FeatureEvaluator:
    """Class to implement common feature evaluation and indicator testing methods"""

    def __init__(self, X=None, y=None, sw=None):

        self.X = X
        self.y = y
        self.sw = sw

    @staticmethod
    def cv_score(clf, X, y, sample_weight, scoring='neg_log_loss', t1=None, cv=None, cv_gen=None, pct_embargo=None):
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
        if scoring not in ['neg_log_loss', 'accuracy']:
            raise ValueError(
                "Scoring method must be 'neg_log_loss' or 'accuracy'.")

        # Initialize purged cross-validation generator if not provided
        if cv_gen is None:
            cv_gen = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)

        # Use a RandomForestClasifier if clf is None
        if clf is None:
            clf = RandomForestClassifier(
                n_estimators=100, max_features='sqrt', random_state=42)

        scores = []  # List to store scores for each fold

        # Cross-validation loop
        for train_indices, test_indices in cv_gen.split(X=X):
            # Train the classifier on the training set
            model = clf.fit(
                X=X.iloc[train_indices, :],
                y=y.iloc[train_indices],
                sample_weight=sample_weight.iloc[train_indices].values
            )

            # Evaluate the model on the test set
            if scoring == 'neg_log_loss':
                probabilities = model.predict_proba(X.iloc[test_indices, :])
                score = -log_loss(
                    y.iloc[test_indices],
                    probabilities,
                    sample_weight=sample_weight.iloc[test_indices].values,
                    labels=clf.classes_
                )
            else:
                predictions = model.predict(X.iloc[test_indices, :])
                score = accuracy_score(
                    y.iloc[test_indices],
                    predictions,
                    sample_weight=sample_weight.iloc[test_indices].values
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
        importance_dict = {i: tree.feature_importances_ for i,
                           tree in enumerate(fit.estimators_)}

        # Convert the dictionary to a DataFrame, with rows as trees and columns as features
        importance_df = pd.DataFrame.from_dict(importance_dict, orient='index')
        importance_df.columns = feature_names

        # Replace zeros with NaN to handle cases where max_features=1, preventing distortions in the mean calculation
        importance_df = importance_df.replace(0, np.nan)

        # Calculate mean and standard deviation of feature importances across trees
        importance_stats = pd.concat({
            'mean': importance_df.mean(),
            'std': importance_df.std() * (importance_df.shape[0] ** -0.5)
        }, axis=1)

        # Normalize the mean importances to sum to 1
        importance_stats['mean'] /= importance_stats['mean'].sum()

        return importance_stats

    def feature_importance_MDA(self, X, y, sample_weight, t1, cv: int = 4,  clf=None,  pct_embargo=0.04, scoring='neg_log_loss'):
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
        if scoring not in ['neg_log_loss', 'accuracy']:
            raise ValueError(
                "Scoring method must be 'neg_log_loss' or 'accuracy'.")

        # Use a RandomForestClasifier if clf is None
        if clf is None:
            clf = RandomForestClassifier(n_estimators=100, max_features='sqrt')

        # Initialize purged cross-validation generator
        cv_generator = PurgedKFold(
            n_splits=cv, t1=t1, pct_embargo=pct_embargo)
        base_scores = pd.Series()  # scores for the original, unpermuted model
        # scores for each permuted feature
        permuted_scores = pd.DataFrame(columns=X.columns)

        # Cross-validation loop
        for fold_index, (train_indices, test_indices) in enumerate(cv_generator.split(X=X)):
            # Split data into train and test sets
            X_train, y_train, w_train = X.iloc[train_indices,
                                               :], y.iloc[train_indices], sample_weight.iloc[train_indices]
            X_test, y_test, w_test = X.iloc[test_indices,
                                            :], y.iloc[test_indices], sample_weight.iloc[test_indices]

            # Fit classifier on the training set
            model = clf.fit(X=X_train, y=y_train, sample_weight=w_train.values)

            # Score the model on the test set
            if scoring == 'neg_log_loss':
                probabilities = model.predict_proba(X_test)
                base_scores.loc[fold_index] = -log_loss(
                    y_test, probabilities, sample_weight=w_test.values, labels=clf.classes_)
            else:
                predictions = model.predict(X_test)
                base_scores.loc[fold_index] = accuracy_score(
                    y_test, predictions, sample_weight=w_test.values)

            # Permute each feature and calculate the impact on model performance
            for feature in X.columns:
                X_test_permuted = X_test.copy(deep=True)
                # Permute one feature at a time
                np.random.shuffle(X_test_permuted[feature].values)

                if scoring == 'neg_log_loss':
                    probabilities = model.predict_proba(X_test_permuted)
                    permuted_scores.loc[fold_index, feature] = -log_loss(
                        y_test, probabilities, sample_weight=w_test.values, labels=clf.classes_)
                else:
                    predictions = model.predict(X_test_permuted)
                    permuted_scores.loc[fold_index, feature] = accuracy_score(
                        y_test, predictions, sample_weight=w_test.values)

        # Calculate feature importance as the relative decrease in accuracy
        importance_scores = (-permuted_scores).add(base_scores, axis=0)
        if scoring == 'neg_log_loss':
            importance_scores = importance_scores / -permuted_scores
        else:
            importance_scores = importance_scores / (1.0 - permuted_scores)

        # Aggregate mean and standard error for each feature's importance across folds
        importance_summary = pd.concat({
            'mean': importance_scores.mean(),
            'std': importance_scores.std() * importance_scores.shape[0]**-0.5
        }, axis=1)

        return importance_summary, base_scores.mean()

    def feature_importance_SFI(self, feature_names, X, y, sw, t1, cv: int = 4, pct_embargo: float = 0.04, clf=None, cv_gen=None, scoring='accuracy'):
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
        importance_scores = pd.DataFrame(columns=['mean', 'std'])

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
                pct_embargo=pct_embargo
            )

            # Record mean and standard deviation of scores for the feature
            importance_scores.loc[feature_name, 'mean'] = feature_scores.mean()
            importance_scores.loc[feature_name, 'std'] = feature_scores.std(
            ) * feature_scores.shape[0]**-0.5

        return importance_scores

    @staticmethod
    def indicator_threshold_search(indicator_values: pd.Series, associated_returns: pd.Series, n_thresholds: int = None, thresholds: list = None):
        """
        Evaluate profit factors for different thresholds of an indicator.

        Parameters:
        indicator_values (pd.Series): Series of indicator values.
        associated_returns (pd.Series): Series of returns associated with the indicator values.
        n_thresholds (int, optional): Number of thresholds to evaluate.
        thresholds (list, optional): List of predefined threshold values.

        Returns:
        pd.DataFrame: DataFrame with thresholds and profit factors for long/short positions above/below the thresholds.
        """
        if thresholds is None:
            if n_thresholds is None:
                raise ValueError(
                    "Either n_thresholds or thresholds must be provided.")
            # Calculate threshold values
            min_val, max_val = indicator_values.min(), indicator_values.max()
            # Exclude min and max values
            thresholds = np.linspace(min_val, max_val, n_thresholds + 2)[1:-1]

        results = []

        for threshold in thresholds:
            # Calculate profit factors for long/short positions above/below the threshold
            above_threshold = indicator_values > threshold
            below_threshold = indicator_values < threshold

            pf_long_above = associated_returns[above_threshold & (associated_returns > 0)].sum() / \
                -associated_returns[above_threshold &
                                    (associated_returns < 0)].sum()
            pf_short_above = -associated_returns[above_threshold & (associated_returns < 0)].sum() / \
                associated_returns[above_threshold &
                                   (associated_returns > 0)].sum()

            pf_long_below = associated_returns[below_threshold & (associated_returns > 0)].sum() / \
                -associated_returns[below_threshold &
                                    (associated_returns < 0)].sum()
            pf_short_below = -associated_returns[below_threshold & (associated_returns < 0)].sum() / \
                associated_returns[below_threshold &
                                   (associated_returns > 0)].sum()

            results.append({
                'Threshold': threshold,
                '% values > threshold': above_threshold.mean()*100,
                'PF Long above threshold': np.nan_to_num(pf_long_above, nan=0.0, posinf=0.0, neginf=0.0),
                'PF Short above threshold': np.nan_to_num(pf_short_above, nan=0.0, posinf=0.0, neginf=0.0),
                '% values < threshold': below_threshold.mean()*100,
                'PF Long below threshold': np.nan_to_num(pf_long_below, nan=0.0, posinf=0.0, neginf=0.0),
                'PF Short below threshold': np.nan_to_num(pf_short_below, nan=0.0, posinf=0.0, neginf=0.0)
            })

        return pd.DataFrame(results)

    @staticmethod
    def single_indicator_threshold_search(inputs: dict):

        ohlcv_df = inputs["ohlcv_df"]
        indicator_values = inputs["indicator_values"]
        thresholds = inputs["thresholds"]

        ohlcv_df['returns'] = ohlcv_df['Open'].pct_change().shift(-1)
        ohlcv_df['ind'] = indicator_values
        ohlcv_df['ind'] = ohlcv_df['ind'].shift(1)

        results = FeatureEvaluator.indicator_threshold_search(indicator_values=ohlcv_df['ind'].dropna(
        ), associated_returns=ohlcv_df['returns'].dropna(), thresholds=thresholds)

        return results

    @staticmethod
    def parallel_indicator_threshold_search(indicator_func: Callable, n_runs: int = 10, n_thresholds: int = 10, **kwargs):

        inputs_list = []
        indicator_values_list = []

        # Generate dataframes and calculate indicator values in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            data_gatherer = DataGatherer()
            futures = [executor.submit(
                data_gatherer.get_random_price_samples, num_tickers_to_sample=1) for _ in range(n_runs)]
            for future in concurrent.futures.as_completed(futures):
                ohlcv_df = future.result()[0]
                indicator_values = indicator_func(
                    ohlcv_df, **kwargs).calculate() if kwargs else indicator_func(ohlcv_df).calculate()
                indicator_values_list.append(indicator_values)
                inputs_list.append({
                    "ohlcv_df": ohlcv_df,
                    "indicator_values": indicator_values
                })

        # Determine thresholds based on the entire range of indicator values across all runs
        all_indicator_values = np.concatenate(indicator_values_list)
        thresholds = np.linspace(all_indicator_values.min(
        ), all_indicator_values.max(), n_thresholds + 2)[1:-1]

        # Update inputs with calculated thresholds
        for inputs in inputs_list:
            inputs["thresholds"] = thresholds

        # Run threshold search in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(
                FeatureEvaluator.single_indicator_threshold_search, inputs_list))

        # Concatenate results and average across runs
        results = pd.concat(results).groupby('Threshold').mean().reset_index()

        return results

    @staticmethod
    def single_indicator_evaluation(inputs: dict):

        outputs = []

        indicator_func = inputs["indicator_func"]
        kwargs = inputs["kwargs"]

        data_gatherer = DataGatherer()
        ohlcv_df = data_gatherer.get_random_price_samples(
            num_tickers_to_sample=1)[0]
        indicator_instance = indicator_func(ohlcv_df, **kwargs)

        indicator_instance.indicator_evaluation_report()

        outputs.append(
            True if indicator_instance.adf_test == "passed" else False)
        outputs.append(True if indicator_instance.jb_normality_test ==
                       "passed" else False)
        outputs.append(indicator_instance.relative_entropy)
        outputs.append(indicator_instance.range_iqr_ratio)

        return outputs

    @staticmethod
    def parallel_indicator_evaluation(indicator_func: Callable, n_runs: int = 10, **kwargs):

        inputs_list = []

        # Create multiple instances of the indicator with a different data sample each time
        for i in range(n_runs):
            inputs = {
                "indicator_func": indicator_func,
                "kwargs": kwargs
            }
            inputs_list.append(inputs)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(
                FeatureEvaluator.single_indicator_evaluation, inputs_list))

        # results is a list of lists - average across the lists to get the final results
        results = np.mean(results, axis=0)

        return {'ADF Test': results[0], 'Jarque-Bera Test': results[1], 'Relative Entropy': results[2], 'Range-IQR Ratio': results[3]}

    @staticmethod
    def optimize_threshold(indicator_values, return_values, min_kept=1, flip_sign=0):
        """
        Find optimal thresholds that maximize profit factors for "long" and "short" trades.

        Parameters:
        indicator_values : numpy array of indicator values.
        return_values    : numpy array of associated returns.
        min_kept         : minimum number of cases that must remain in a group (default 1).
        flip_sign      : if nonzero, the indicator values are negated (default 0).

        Returns:
        pf_all    : profit factor for the entire dataset.
        high_thresh : optimal threshold for long trades.
        pf_high   : profit factor for the "above threshold" (long) set.
        low_thresh  : optimal threshold for short trades.
        pf_low    : profit factor for the "below threshold" (short) set.
        """

        # Ensure the inputs are numpy arrays.
        indicator_values = np.asarray(indicator_values)
        return_values = np.asarray(return_values)

        n = len(indicator_values)
        if n == 0:
            raise ValueError("Input arrays must have at least one element.")

        # Enforce that min_kept is at least 1.
        if min_kept < 1:
            min_kept = 1

        # Copy signals and returns into work arrays.
        # Optionally flip the sign of indicator values.
        if flip_sign:
            work_signal = -indicator_values.copy()
        else:
            work_signal = indicator_values.copy()
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

        best_high_index, best_low_index, best_high_pf, best_low_pf = optimize_threshold_cython(
            work_signal, work_return, min_kept)

        # Initialize accumulators for wins and losses.
        win_above = 0.0
        lose_above = 0.0

        # Compute total wins and losses for the complete dataset.
        # For a "long" position, positive returns are wins and negative returns are losses.
        for i in range(n):
            r = work_return[i]
            if r > 0.0:
                win_above += r
            else:
                lose_above -= r  # subtracting a negative adds its magnitude

        # Compute profit factor for the entire dataset.
        pf_all = win_above / (lose_above + 1.e-30)

        # # Initialize best profit factor for the high group with the overall pf.
        # best_high_pf = pf_all
        # # A threshold below the smallest value implies using all cases.
        # best_high_index = 0

        # # Initialize best profit factor for the low group.
        # best_low_pf = -1.0
        # # Default (should be updated if any legitimate threshold is found)
        # best_low_index = n - 1

        # # Loop over possible thresholds. Each candidate threshold is at index i+1.
        # for i in range(n - 1):
        #     r = work_return[i]

        #     # Remove this case from the "above" (long) set.
        #     if r > 0.0:
        #         win_above -= r
        #     else:
        #         lose_above += r  # r is negative, so this reduces lose_above.

        #     # Add this case to the "below" (short) set.
        #     if r > 0.0:
        #         lose_below += r
        #     else:
        #         win_below -= r

        #     # Only consider a new threshold if the next signal value differs.
        #     if work_signal[i + 1] == work_signal[i]:
        #         continue

        #     # Check if the "above" set (i+1 to n-1) has at least min_kept cases.
        #     if (n - i - 1) >= min_kept:
        #         current_high_pf = win_above / (lose_above + 1.e-30)
        #         if current_high_pf > best_high_pf:
        #             best_high_pf = current_high_pf
        #             best_high_index = i + 1

        #     # Check if the "below" set (0 to i) has at least min_kept cases.
        #     if (i + 1) >= min_kept:
        #         current_low_pf = win_below / (lose_below + 1.e-30)
        #         if current_low_pf > best_low_pf:
        #             best_low_pf = current_low_pf
        #             best_low_index = i + 1

        # The best thresholds are the signal values at the recorded indices.
        high_thresh = work_signal[best_high_index]
        low_thresh = work_signal[best_low_index]
        pf_high = best_high_pf
        pf_low = best_low_pf

        return {'pf_all': pf_all, 'optimal_long_thresh': high_thresh, 'optimal_long_pf': pf_high, 'optimal_short_thresh': low_thresh, 'optimal_short_pf': pf_low}
