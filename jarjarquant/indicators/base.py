from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import polars as pl

from .registry import IndicatorType


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


class Indicator:
    """Base class to implement indicators"""

    def __init__(self, ohlcv_df: pl.DataFrame):
        if ohlcv_df is None or ohlcv_df.height == 0:
            raise ValueError("Please provide a valid OHLCV DataFrame!")

        self.df = ohlcv_df
        self.indicator_type = None

    def calculate(self) -> np.ndarray:
        """Implemented in derived classes

        Raises:
            NotImplementedError
        """
        raise NotImplementedError(
            "Derived classes must implement the calculate method."
        )
