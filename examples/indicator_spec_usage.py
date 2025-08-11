#!/usr/bin/env python3
"""
Example usage of IndicatorSpec for easy indicator configuration and creation.

This example shows how to use the IndicatorSpec dataclass to:
1. Define indicator configurations declaratively
2. Create multiple indicator instances efficiently
3. Build indicator analysis pipelines
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jarjarquant.indicators.base import IndicatorSpec
from jarjarquant.indicators.registry import IndicatorType


def load_sample_data() -> pd.DataFrame:
    """Load or create sample OHLCV data."""
    # In a real application, you would load actual market data
    # For this example, we'll create synthetic data
    np.random.seed(123)
    n_periods = 252  # One year of trading days
    
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n_periods)  # Daily returns
    
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])
    
    # Create realistic OHLCV data
    noise_factor = 0.02
    high = prices * (1 + np.random.uniform(0, noise_factor, n_periods))
    low = prices * (1 - np.random.uniform(0, noise_factor, n_periods))
    open_prices = np.roll(prices, 1)
    open_prices[0] = base_price
    volume = np.random.randint(50000, 200000, n_periods)
    
    # Create date index
    dates = pd.date_range('2023-01-01', periods=n_periods, freq='D')
    
    return pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': prices,
        'Volume': volume
    }).set_index('Date')


def create_indicator_portfolio() -> list[IndicatorSpec]:
    """
    Create a portfolio of indicator specifications for analysis.
    
    Returns:
        List of IndicatorSpec objects representing different analysis components
    """
    return [
        # Momentum indicators with different periods
        IndicatorSpec(
            indicator_type=IndicatorType.RSI,
            parameters={'period': 14}
        ),
        IndicatorSpec(
            indicator_type=IndicatorType.RSI,
            parameters={'period': 21}
        ),
        
        # Trend indicators
        IndicatorSpec(
            indicator_type=IndicatorType.MACD,
            parameters={
                'short_period': 5,
                'long_period': 20,
                'smoothing_factor': 3
            }
        ),
        IndicatorSpec(
            indicator_type=IndicatorType.MACD,
            parameters={
                'short_period': 8,
                'long_period': 21,
                'return_raw_macd': True
            }
        ),
        
        # Transformed indicators for different analysis perspectives
        IndicatorSpec(
            indicator_type=IndicatorType.RSI,
            parameters={
                'period': 14,
                'transform': 'tanh'  # Bounded transformation
            }
        ),
        IndicatorSpec(
            indicator_type=IndicatorType.MACD,
            parameters={
                'short_period': 5,
                'long_period': 15,
                'transform': 'log'
            }
        )
    ]


def analyze_indicators(ohlcv_data: pd.DataFrame, specs: list[IndicatorSpec]):
    """
    Analyze a portfolio of indicators and generate insights.
    
    Args:
        ohlcv_data: OHLCV market data
        specs: List of indicator specifications to analyze
    """
    print("=" * 80)
    print("INDICATOR PORTFOLIO ANALYSIS")
    print("=" * 80)
    
    results = []
    
    for i, spec in enumerate(specs, 1):
        print(f"\n--- Indicator {i}: {spec.indicator_type.value.upper()} ---")
        print(f"Parameters: {spec.parameters}")
        
        try:
            # Create and calculate indicator
            indicator = spec.create_indicator(ohlcv_data)
            values = indicator.calculate()
            
            # Calculate statistics
            stats = {
                'name': f"{spec.indicator_type.value}_{i}",
                'type': spec.indicator_type.value,
                'parameters': spec.parameters,
                'count': len(values),
                'mean': np.nanmean(values),
                'std': np.nanstd(values),
                'min': np.nanmin(values),
                'max': np.nanmax(values),
                'skewness': calculate_skewness(values),
                'nan_count': np.sum(np.isnan(values))
            }
            
            results.append(stats)
            
            # Display key metrics
            print(f"  Values calculated: {stats['count']}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Skewness: {stats['skewness']:.4f}")
            if stats['nan_count'] > 0:
                print(f"  NaN values: {stats['nan_count']}")
            
            print("  [SUCCESS]")
            
        except Exception as e:
            print(f"  [ERROR]: {str(e)}")
            results.append({
                'name': f"{spec.indicator_type.value}_{i}",
                'type': spec.indicator_type.value,
                'parameters': spec.parameters,
                'error': str(e)
            })
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("PORTFOLIO SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"Indicators processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nStatistics across successful indicators:")
        means = [r['mean'] for r in successful]
        stds = [r['std'] for r in successful]
        print(f"  Average mean: {np.mean(means):.4f}")
        print(f"  Average std: {np.mean(stds):.4f}")
        print(f"  Mean range: [{min(means):.4f}, {max(means):.4f}]")
        print(f"  Std range: [{min(stds):.4f}, {max(stds):.4f}]")
    
    if failed:
        print(f"\nFailed indicators:")
        for result in failed:
            print(f"  - {result['name']}: {result['error']}")
    
    return results


def calculate_skewness(values: np.ndarray) -> float:
    """Calculate skewness of a series."""
    values_clean = values[~np.isnan(values)]
    if len(values_clean) < 3:
        return np.nan
    
    mean_val = np.mean(values_clean)
    std_val = np.std(values_clean, ddof=1)
    
    if std_val == 0:
        return 0.0
    
    # Calculate skewness
    skew = np.mean(((values_clean - mean_val) / std_val) ** 3)
    return skew


def demonstrate_dynamic_specs():
    """Demonstrate creating IndicatorSpec objects dynamically."""
    print("\n" + "=" * 80)
    print("DYNAMIC INDICATOR SPECIFICATION")
    print("=" * 80)
    
    # Configuration-driven indicator creation
    config = {
        'momentum_analysis': {
            'rsi_periods': [10, 14, 21, 30],
            'transforms': [None, 'tanh']
        },
        'trend_analysis': {
            'macd_configs': [
                {'short': 5, 'long': 15},
                {'short': 8, 'long': 21},
                {'short': 12, 'long': 26}
            ]
        }
    }
    
    dynamic_specs = []
    
    # Generate RSI specs
    for period in config['momentum_analysis']['rsi_periods']:
        for transform in config['momentum_analysis']['transforms']:
            params = {'period': period}
            if transform:
                params['transform'] = transform
            
            spec = IndicatorSpec(
                indicator_type=IndicatorType.RSI,
                parameters=params
            )
            dynamic_specs.append(spec)
    
    # Generate MACD specs
    for macd_config in config['trend_analysis']['macd_configs']:
        spec = IndicatorSpec(
            indicator_type=IndicatorType.MACD,
            parameters={
                'short_period': macd_config['short'],
                'long_period': macd_config['long']
            }
        )
        dynamic_specs.append(spec)
    
    print(f"Generated {len(dynamic_specs)} indicator specifications dynamically:")
    for i, spec in enumerate(dynamic_specs, 1):
        print(f"  {i}. {spec.indicator_type.value}: {spec.parameters}")
    
    return dynamic_specs


def main():
    """Main demonstration function."""
    print("IndicatorSpec Usage Examples")
    print("=" * 80)
    
    # Load sample data
    ohlcv_data = load_sample_data()
    print(f"Loaded {len(ohlcv_data)} periods of market data")
    print(f"Date range: {ohlcv_data.index.min()} to {ohlcv_data.index.max()}")
    print(f"Price range: ${ohlcv_data['Close'].min():.2f} - ${ohlcv_data['Close'].max():.2f}")
    
    # Example 1: Predefined indicator portfolio
    indicator_specs = create_indicator_portfolio()
    results = analyze_indicators(ohlcv_data, indicator_specs)
    
    # Example 2: Dynamic specification generation
    dynamic_specs = demonstrate_dynamic_specs()
    
    print(f"\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print(f"Total indicators analyzed: {len(results)}")
    print(f"Dynamic specs generated: {len(dynamic_specs)}")
    print("\nIndicatorSpec provides a clean, flexible way to:")
    print("  - Configure indicators declaratively")
    print("  - Create multiple indicator instances efficiently")
    print("  - Build scalable analysis pipelines")
    print("  - Generate configurations programmatically")


if __name__ == "__main__":
    main()