"""Feature engineering class to manipulate features df"""
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection._split import _BaseKFold


class purged_k_fold(_BaseKFold):
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


class FeatureEngineer:
    """Class to implement common featrue engineering transformations"""

    def __init__(self, features_df):
        """Initialize Featureengineering

        Args:
            features_df (pd.DataFrame): 
                a pandas dataframe containg the features timeseries with pd.DateTime index
        """
        self.features_df = features_df

    @staticmethod
    def get_weights(d, size):
        """
        Calculate the weights for fractional differentiation.

        Parameters:
        -----------
        d : float
            The order of differentiation. Determines the depth of the fractional differentiation.
        size : int
            The size of the output array (typically corresponds to the number of observations 
            or time periods).

        Returns:
        --------
        numpy.ndarray
            A 2D numpy array (size x 1) of weights, where each weight corresponds to the fractional 
            differentiation weight for that time period.
        """
        w = [1.]
        for k in range(1, size):
            w_ = -w[-1]/k*(d-k+1)
            w.append(w_)
        w = np.array(w[::-1]).reshape(-1, 1)
        return w

    def frac_diff(self, features_df: pd.DataFrame = None, d: float = 0.7, thres=.01):
        """
        Apply fractional differentiation to time series data, preserving memory of long-term dependencies.

        Parameters:
        -----------
        d : float
            The order of fractional differentiation.
        thres : float, optional (default = 0.01)
            The threshold for determining how many weights to skip based on cumulative sum of the weights.
            Helps to avoid applying very small weights.

        Returns:
        --------
        pandas.DataFrame
            A DataFrame with fractionally differentiated series for each column in `features_df`.
            The output retains long-term memory by applying the computed weights to each time series.
        """
        if features_df is None:
            features_df = self.features_df

        w = self.get_weights(d, features_df.shape[0])
        w_ = np.cumsum(abs(w))
        w_ /= w_[-1]
        skip = w_[w_ > thres].shape[0]
        df = {}
        for name in features_df.columns:
            seriesF, df_ = features_df[[name]].ffill().dropna(
            ), pd.Series(index=features_df.index, dtype=float)
            for iloc in range(skip, seriesF.shape[0]):
                loc = seriesF.index[iloc]
                if not np.isfinite(features_df.loc[loc, name]):
                    continue
                df_[loc] = np.dot(w[-(iloc+1):, :].T, seriesF.loc[:loc])[0, 0]
            df[name] = df_.copy(deep=True)
        df = pd.concat(df, axis=1)

        features_df = df

        return features_df

    @staticmethod
    def getWeights_FFD(d, thres):
        """
        Compute the weights for Fractional Differencing using the Fixed-Width Window method.

        This function calculates the weights required for fractional differencing based on the 
        parameter `d`. The weights decrease in magnitude as the function iterates and stops 
        when the absolute value of the next weight falls below a specified threshold (`thres`).

        Args:
        - d (float): The fractional differencing parameter. Determines the strength of differencing.
        - thres (float): Threshold for the smallest weight to include. Iteration stops when the 
        next weight is smaller than this threshold.

        Returns:
        - np.ndarray: Array of fractional differencing weights, arranged in reverse order.
        """
        w, k = [1.], 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres:
                break
            w.append(w_)
            k += 1
        return np.array(w[::-1]).reshape(-1, 1)

    def fracDiff_FFD(self, features_df: pd.DataFrame = None, d: float = 0.7, thres=1e-5):
        """
        Apply Fractional Differencing with Fixed-Width Window on a time series to make it stationary.

        Fractional differencing aims to remove long-term dependence from time series data while 
        preserving memory in the data. This method uses a fixed-width window and calculates the 
        fractional differences using the precomputed weights.

        Args:
        - series (pd.DataFrame): Time series data (columns represent individual time series) to be fractionally differenced.
        - d (float): The fractional differencing parameter. Determines the strength of differencing.
        - thres (float, optional): Threshold for the smallest weight to include. Default is 1e-5.

        Returns:
        - pd.DataFrame: A DataFrame with the same structure as `series`, containing the fractionally differenced series.
        """
        # Get fractional differencing weights using the provided threshold and d value
        w, df = self.getWeights_FFD(d, thres), {}
        width = len(w) - 1

        if features_df is None:
            features_df = self.features_df

        # Apply fractional differencing to each column in the DataFrame
        for name in features_df.columns:
            # Forward fill to handle missing data and create a new series for results
            seriesF = features_df[[name]].ffill().dropna()
            df_ = pd.Series(index=features_df.index, dtype=float)

            # Loop through the series and apply the fractional differencing weights
            for iloc in range(width, seriesF.shape[0]):
                loc0, loc1 = seriesF.index[iloc - width], seriesF.index[iloc]
                if not np.isfinite(features_df.loc[loc1, name]):
                    continue
                df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0, 0]

            # Store the fractionally differenced column in the result dictionary
            df[name] = df_.copy(deep=True)

        # Combine all columns into a DataFrame
        df = pd.concat(df, axis=1)

        features_df = df

        return features_df

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
            cv_gen = purged_k_fold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)

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

    def feature_importance_MDA(self, clf, X, y, cv, sample_weight, t1, pct_embargo, scoring='neg_log_loss'):
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

        # Initialize purged cross-validation generator
        cv_generator = purged_k_fold(
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

    def feature_importance_SFI(self, feature_names, clf, X, y, sw, t1, cv, pct_embargo, cv_gen=None, scoring='accuracy'):
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
