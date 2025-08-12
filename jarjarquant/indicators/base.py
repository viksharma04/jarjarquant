from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import polars as pl

from jarjarquant.data_analyst import (
    adf_test,
    jb_normality_test,
    mutual_information,
    range_iqr_ratio,
    relative_entropy,
    visual_stationary_test,
)
from jarjarquant.feature_engineer import FeatureEngineer
from jarjarquant.feature_evaluator import FeatureEvaluator

from .registry import IndicatorType


@dataclass
class IndicatorSpec:
    """
    Specification for creating an indicator instance with type-safe parameters.

    This dataclass provides a convenient way to specify an indicator along with
    its parameters, allowing for easy configuration and instantiation of indicators.

    Attributes:
        indicator_type: The type of indicator to create (from IndicatorType enum)
        parameters: Dictionary of parameters to pass to the indicator constructor
                   (excluding the required ohlcv_df parameter)

    Example:
        spec = IndicatorSpec(
            indicator_type=IndicatorType.RSI,
            parameters={'period': 21, 'transform': 'log'}
        )
    """

    indicator_type: "IndicatorType"  # Forward reference to avoid circular imports
    parameters: Dict[str, Any] = field(default_factory=dict)

    def create_indicator(self, ohlcv_df: pl.DataFrame) -> "Indicator":
        """
        Create an indicator instance using this specification.

        Args:
            ohlcv_df: The OHLCV DataFrame to pass to the indicator constructor

        Returns:
            Configured indicator instance

        Raises:
            KeyError: If the indicator type is not registered
            TypeError: If invalid parameters are provided
        """
        from jarjarquant.indicators.registry import get_indicator_class

        indicator_class = get_indicator_class(self.indicator_type)
        return indicator_class(ohlcv_df, **self.parameters)


@dataclass
class IndicatorEvalResult:
    adf_test: Optional[str] = None
    jb_normality_test: Optional[str] = None
    relative_entropy: Optional[np.float64] = None
    range_iqr_ratio: Optional[np.float64] = None
    mutual_information: Optional[np.ndarray] = None


class Indicator:
    """Base class to implement indicators"""

    def __init__(self, ohlcv_df: pl.DataFrame):
        if ohlcv_df is None or ohlcv_df.height == 0:
            raise ValueError("Please provide a valid OHLCV DataFrame!")

        self.df = ohlcv_df
        self.indicator_type = None
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
        self.eval_result = IndicatorEvalResult()
        values = self.calculate()
        if transform is not None:
            values = self.feature_engineer.transform(values, transform, **kwargs)
            if not isinstance(values, np.ndarray):
                values = np.asarray(values)

        visual_stationary_test(values)
        self.eval_result.adf_test = adf_test(values, verbose=verbose).decision
        self.eval_result.jb_normality_test = jb_normality_test(
            values, verbose=verbose
        ).decision
        self.eval_result.relative_entropy = relative_entropy(
            values, verbose=verbose
        ).normalized_entropy
        self.eval_result.range_iqr_ratio = range_iqr_ratio(
            values, verbose=verbose
        ).ratio

        if self.indicator_type == "continuous":
            n_bins_to_discretize = (
                n_bins_to_discretize if n_bins_to_discretize is not None else 10
            )
            self.eval_result.mutual_information = mutual_information(
                array=values,
                lag=10,
                n_bins=n_bins_to_discretize,
                is_discrete=False,
                verbose=verbose,
            )
        else:
            self.eval_result.mutual_information = mutual_information(
                array=values,
                lag=10,
                n_bins=None,
                is_discrete=True,
                verbose=verbose,
            )

        for i in range(1, 11):
            print(f"NMI @ lag {i} = {self.eval_result.mutual_information[i - 1]}")
