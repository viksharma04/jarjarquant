from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import polars as pl

from jarjarquant.data_analyst import (
    ADFTestResult,
    EntropyResult,
    NormalityTestResult,
    RangeIQRResult,
    adf_test_ultra_fast,
    jb_normality_test,
    range_iqr_ratio,
    relative_entropy,
    visual_stationary_test,
)
from jarjarquant.feature_engineer import FeatureEngineer
from jarjarquant.feature_evaluator import FeatureEvaluator

from .registry import IndicatorType

FEATURE_ENGINEER = FeatureEngineer()
FEATURE_EVALUATOR = FeatureEvaluator()


@dataclass
class IndicatorSpec:
    """
    Specification for creating an indicator instance with type-safe parameters.

    This dataclass provides a convenient way to specify an indicator along with
    its parameters, allowing for easy configuration and instantiation of indicators.
    Default parameters are automatically populated from the indicator class, and user
    parameters override defaults with validation.

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

    def __post_init__(self):
        """
        Post-initialization hook to populate default parameters and validate user inputs.
        """
        from jarjarquant.indicators.registry import get_indicator_parameters

        # Get the default parameters for this indicator type
        default_params = get_indicator_parameters(self.indicator_type)

        # Remove 'ohlcv_df' from default parameters as it's handled separately
        if "ohlcv_df" in default_params:
            del default_params["ohlcv_df"]

        # Validate that user-provided parameters are valid for this indicator
        invalid_params = set(self.parameters.keys()) - set(default_params.keys())
        if invalid_params:
            valid_params = list(default_params.keys())
            raise ValueError(
                f"Invalid parameters for {self.indicator_type.value}: {list(invalid_params)}. "
                f"Valid parameters are: {valid_params}"
            )

        # Create final parameters by merging defaults with user overrides
        final_params = {}
        for param_name, param_info in default_params.items():
            if param_name in self.parameters:
                # User provided this parameter - use their value
                final_params[param_name] = self.parameters[param_name]
            elif not param_info["required"]:
                # Parameter has a default value - use it
                final_params[param_name] = param_info["default"]
            # Required parameters without defaults will be caught during indicator instantiation

        # Update the parameters dict with the final merged parameters
        self.parameters = final_params

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
    adf_test: ADFTestResult
    jb_normality_test: NormalityTestResult
    relative_entropy: EntropyResult
    range_iqr_ratio: RangeIQRResult


class Indicator:
    """Base class to implement indicators"""

    def __init__(self, ohlcv_df: pl.DataFrame):
        if ohlcv_df is None or ohlcv_df.height == 0:
            raise ValueError("Please provide a valid OHLCV DataFrame!")

        self.df = ohlcv_df
        self.indicator_type = None
        self.feature_engineer = FEATURE_ENGINEER
        self.feature_evaluator = FEATURE_EVALUATOR

        self.eval_result = None

    def calculate(self) -> np.ndarray:
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
        visual_test: Optional[bool] = False,
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
        values = self.calculate()
        if transform is not None:
            values = self.feature_engineer.transform(values, transform, **kwargs)
            if not isinstance(values, np.ndarray):
                values = np.asarray(values)

        if visual_test:
            visual_stationary_test(values)
        import concurrent.futures

        # Run statistical tests in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tests to the executor
            adf_future = executor.submit(adf_test_ultra_fast, values, verbose=verbose)
            normality_future = executor.submit(
                jb_normality_test, values, verbose=verbose
            )
            entropy_future = executor.submit(relative_entropy, values, verbose=verbose)
            r_iqr_future = executor.submit(range_iqr_ratio, values, verbose=verbose)

            # Wait for all results
            adf_test_result = adf_future.result()
            normality_test_result = normality_future.result()
            entropy_result = entropy_future.result()
            r_iqr_result = r_iqr_future.result()

        self.eval_result = IndicatorEvalResult(
            adf_test_result, normality_test_result, entropy_result, r_iqr_result
        )
