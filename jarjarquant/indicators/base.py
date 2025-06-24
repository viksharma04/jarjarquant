from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from jarjarquant.data_analyst import DataAnalyst
from jarjarquant.feature_engineer import FeatureEngineer
from jarjarquant.feature_evaluator import FeatureEvaluator


@dataclass
class IndicatorEvalResult:
    adf_test: Optional[str] = None
    jb_normality_test: Optional[str] = None
    relative_entropy: Optional[np.float64] = None
    range_iqr_ratio: Optional[np.float64] = None
    mutual_information: Optional[np.ndarray] = None


class Indicator:
    """Base class to implement indicators"""

    def __init__(self, ohlcv_df: pd.DataFrame):
        if ohlcv_df is None or ohlcv_df.empty:
            raise ValueError("Please provide a valid OHLCV DataFrame!")

        self.df = ohlcv_df.copy()
        self.indicator_type = None
        self.data_analyst = DataAnalyst()
        self.feature_engineer = FeatureEngineer()
        self.feature_evaluator = FeatureEvaluator()

        self.eval_result = None

    def calculate(self):
        """Implemented in derived classes

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "Derived classes must implement the calculate method."
        )

    def indicator_evaluation_report(
        self,
        verbose: bool = False,
        transform: Optional[str] = None,
        n_bins_to_discretize: Optional[int] = None,
        **kwargs,
    ):
        """Runs a set of statistical tests to examine various properties of the
        indicator series, such as stationarity, normality, entropy, mutual
        information, etc.

        Args:
            transform (str, optional): Acceptable values: 'log', 'root', 'tanh'. Transformation to apply to the indicator values.
            n_bins_to_discretize (int, optional): Number of bins to use if indicator
            is continuous. Used for mutual information calculation. Defaults to 10.
        """
        eval_result = IndicatorEvalResult()
        values = self.calculate()
        if transform is not None:
            values = self.feature_engineer.transform(values, transform, **kwargs)
            if not isinstance(values, np.ndarray):
                values = np.asarray(values)

        self.data_analyst.visual_stationary_test(values)
        eval_result.adf_test = self.data_analyst.adf_test(values, verbose)
        eval_result.jb_normality_test = self.data_analyst.jb_normality_test(
            values, verbose
        )
        eval_result.relative_entropy = self.data_analyst.relative_entropy(
            values, verbose
        )
        eval_result.range_iqr_ratio = self.data_analyst.range_iqr_ratio(values, verbose)

        if self.indicator_type == "continuous":
            n_bins_to_discretize = (
                n_bins_to_discretize if n_bins_to_discretize is not None else 10
            )
            eval_result.mutual_information = self.data_analyst.mutual_information(
                array=values,
                lag=10,
                n_bins=n_bins_to_discretize,
                is_discrete=False,
                verbose=verbose,
            )
        else:
            eval_result.mutual_information = self.data_analyst.mutual_information(
                array=values,
                lag=10,
                n_bins=None,
                is_discrete=True,
                verbose=verbose,
            )

        for i in range(1, 11):
            print(f"NMI @ lag {i} = {eval_result.mutual_information[i - 1]}")

    # TODO: Implement a robust indicator evaluation report method which creates multiple instances of the indicator and averages the results
