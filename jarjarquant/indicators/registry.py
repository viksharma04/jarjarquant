"""
Indicator registry system for managing and discovering available indicators.

This module provides a centralized registry for all indicators, allowing for
type-safe indicator discovery and automatic registration. It includes functionality
to introspect indicator classes and retrieve their initialization parameters.

Key functions:
- list_available_indicators(): List indicators with optional detailed parameter info
- get_indicator_parameters(): Get initialization parameters for a specific indicator
- register_indicator(): Decorator to register indicator classes
- get_indicator_class(): Retrieve registered indicator class by type
"""

import inspect
from enum import Enum
from typing import TYPE_CHECKING, Dict, Type

if TYPE_CHECKING:
    from jarjarquant.indicators.base import Indicator


class IndicatorType(Enum):
    """Enumeration of all available indicators for type-safe usage."""

    RSI = "rsi"
    MACD = "macd"
    ADX = "adx"
    AROON = "aroon"
    STOCHASTIC = "stochastic"
    STOCHASTIC_RSI = "stochastic_rsi"
    CHAIKIN_MONEY_FLOW = "chaikin_money_flow"
    DETRENDED_RSI = "detrended_rsi"
    CMMA = "cmma"  # Cumulative Moving Median Average
    MOVING_AVERAGE_DIFFERENCE = "moving_average_difference"
    PRICE_CHANGE_OSCILLATOR = "price_change_oscillator"
    PRICE_INTENSITY = "price_intensity"
    REGRESSION_TREND = "regression_trend"
    REGRESSION_TREND_DEVIATION = "regression_trend_deviation"
    ANCHORED_VWAP = "anchored_vwap"


# Global registry mapping indicator types to their classes
INDICATOR_REGISTRY: Dict[IndicatorType, Type["Indicator"]] = {}


def register_indicator(indicator_type: IndicatorType):
    """
    Decorator to register an indicator class with the registry.

    Args:
        indicator_type: The IndicatorType enum value for this indicator

    Returns:
        The decorator function that registers the class

    Example:
        @register_indicator(IndicatorType.RSI)
        class RSI(Indicator):
            pass
    """

    def decorator(cls: Type["Indicator"]) -> Type["Indicator"]:
        INDICATOR_REGISTRY[indicator_type] = cls
        return cls

    return decorator


def get_indicator_class(indicator_type: IndicatorType) -> Type["Indicator"]:
    """
    Get the indicator class for a given indicator type.

    Args:
        indicator_type: The type of indicator to retrieve

    Returns:
        The indicator class

    Raises:
        KeyError: If the indicator type is not registered
    """
    if indicator_type not in INDICATOR_REGISTRY:
        registered_types = list(INDICATOR_REGISTRY.keys())
        raise KeyError(
            f"Indicator type {indicator_type} not found in registry. "
            f"Available types: {registered_types}"
        )
    return INDICATOR_REGISTRY[indicator_type]


def list_available_indicators(verbose: bool = False):
    """
    List all available registered indicators.

    Args:
        verbose: If True, returns detailed parameter information for each indicator.
                If False, returns just the list of indicator types.

    Returns:
        If verbose=False: List of all registered indicator types
        If verbose=True: Dictionary mapping indicator types to their parameter details
    """
    if not verbose:
        return list(INDICATOR_REGISTRY.keys())

    detailed_info = {}
    for indicator_type in INDICATOR_REGISTRY.keys():
        detailed_info[indicator_type] = get_indicator_parameters(indicator_type)

    return detailed_info


def is_indicator_registered(indicator_type: IndicatorType) -> bool:
    """
    Check if an indicator type is registered.

    Args:
        indicator_type: The indicator type to check

    Returns:
        True if registered, False otherwise
    """
    return indicator_type in INDICATOR_REGISTRY


def get_indicator_parameters(indicator_type: IndicatorType) -> dict[str, dict]:
    """
    Get the initialization parameters for a specific indicator type.

    Args:
        indicator_type: The IndicatorType to get parameters for

    Returns:
        Dictionary containing parameter information with the following structure:
        {
            'parameter_name': {
                'type': 'parameter_type',
                'default': default_value,
                'required': bool
            }
        }

    Raises:
        KeyError: If the indicator type is not registered
    """
    if indicator_type not in INDICATOR_REGISTRY:
        registered_types = list(INDICATOR_REGISTRY.keys())
        raise KeyError(
            f"Indicator type {indicator_type} not found in registry. "
            f"Available types: {registered_types}"
        )

    indicator_class = INDICATOR_REGISTRY[indicator_type]
    signature = inspect.signature(indicator_class.__init__)

    parameters = {}
    for param_name, param in signature.parameters.items():
        # Skip 'self' parameter
        if param_name == "self":
            continue

        param_info = {
            "type": str(param.annotation)
            if param.annotation != inspect.Parameter.empty
            else "Any",
            "required": param.default == inspect.Parameter.empty,
            "default": param.default
            if param.default != inspect.Parameter.empty
            else None,
        }

        parameters[param_name] = param_info

    return parameters
