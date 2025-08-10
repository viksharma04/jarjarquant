"""
Mock data generators for testing CustomDataSource and other data-related functionality.
"""

from typing import Dict

import pandas as pd
import polars as pl


def create_mock_daily_equity_data(
    ticker: str = "AAPL",
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    base_price: float = 150.0,
) -> pl.DataFrame:
    """
    Create mock daily equity price data for testing.
    
    Args:
        ticker: Stock symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format  
        base_price: Starting price for the stock
        
    Returns:
        Polars DataFrame with daily OHLCV data
    """
    import numpy as np
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    # Generate realistic price movements
    np.random.seed(42)  # For reproducible tests
    returns = np.random.normal(0.001, 0.02, n_days)  # ~0.1% daily return, 2% volatility
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLC data from prices
    opens, highs, lows, closes = [], [], [], []
    volumes, averages, bar_counts = [], [], []
    
    for i, price in enumerate(prices):
        daily_volatility = price * 0.01  # 1% daily range
        high = price + np.random.uniform(0, daily_volatility)
        low = price - np.random.uniform(0, daily_volatility)
        open_price = price + np.random.uniform(-daily_volatility/2, daily_volatility/2)
        close_price = price
        
        volume = int(np.random.uniform(1000000, 10000000))  # 1M to 10M shares
        average = (high + low + open_price + close_price) / 4
        bar_count = int(np.random.uniform(1000, 5000))
        
        opens.append(round(open_price, 2))
        highs.append(round(high, 2))
        lows.append(round(low, 2))
        closes.append(round(close_price, 2))
        volumes.append(volume)
        averages.append(round(average, 2))
        bar_counts.append(bar_count)
    
    return pl.DataFrame({
        'date': [d.date() for d in dates],
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes,
        'average': averages,
        'barCount': bar_counts
    })


def create_mock_intraday_equity_data(
    ticker: str = "AAPL",
    date: str = "2023-12-29",
    bar_size: str = "1hour",
    base_price: float = 193.0,
) -> pl.DataFrame:
    """
    Create mock intraday equity price data for testing.
    
    Args:
        ticker: Stock symbol
        date: Date in YYYY-MM-DD format
        bar_size: "1hour" or "1min"
        base_price: Starting price for the day
        
    Returns:
        Polars DataFrame with intraday OHLCV data
    """
    import numpy as np
    
    np.random.seed(42)  # For reproducible tests
    
    if bar_size == "1hour":
        # Market hours: 9:30 AM to 4:00 PM (6.5 hours)
        times = pd.date_range(
            start=f"{date} 09:30:00",
            end=f"{date} 16:00:00",
            freq='h',
            tz='America/New_York'
        )
    elif bar_size == "1min":
        # Generate a smaller subset for testing (first 2 hours)
        times = pd.date_range(
            start=f"{date} 09:30:00",
            end=f"{date} 11:30:00",
            freq='T',
            tz='America/New_York'
        )
    else:
        raise ValueError(f"Unsupported bar_size: {bar_size}")
    
    opens, highs, lows, closes, volumes = [], [], [], [], []
    current_price = base_price
    
    for _ in times:
        # Small price movements for intraday
        price_change = np.random.normal(0, 0.001)  # 0.1% volatility per bar
        current_price *= (1 + price_change)
        
        spread = current_price * 0.002  # 0.2% spread
        high = current_price + np.random.uniform(0, spread)
        low = current_price - np.random.uniform(0, spread)
        open_price = current_price + np.random.uniform(-spread/2, spread/2)
        close_price = current_price
        
        volume = int(np.random.uniform(100000, 1000000))
        
        opens.append(round(open_price, 2))
        highs.append(round(high, 2))
        lows.append(round(low, 2))
        closes.append(round(close_price, 2))
        volumes.append(volume)
    
    return pl.DataFrame({
        'datetime': times,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes
    })


def create_mock_forex_data(
    pair: str = "EURUSD",
    start_date: str = "2023-12-01",
    end_date: str = "2023-12-31",
    base_rate: float = 1.0950,
    bar_size: str = "1d"
) -> pl.DataFrame:
    """
    Create mock forex rate data for testing.
    
    Args:
        pair: Currency pair (e.g., "EURUSD")
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        base_rate: Starting exchange rate
        bar_size: "1d" or "1hour"
        
    Returns:
        Polars DataFrame with forex OHLC data
    """
    import numpy as np
    
    np.random.seed(42)  # For reproducible tests
    
    if bar_size == "1d":
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        date_column = "date"
    elif bar_size == "1hour":
        dates = pd.date_range(
            start=f"{start_date} 00:00:00",
            end=f"{end_date} 23:00:00",
            freq='h',
            tz='UTC'
        )
        date_column = "datetime"
    else:
        raise ValueError(f"Unsupported bar_size: {bar_size}")
    
    opens, highs, lows, closes = [], [], [], []
    current_rate = base_rate
    
    for _ in dates:
        # Forex typically has smaller movements
        rate_change = np.random.normal(0, 0.002)  # 0.2% volatility
        current_rate *= (1 + rate_change)
        
        spread = current_rate * 0.0005  # 5 pips spread
        high = current_rate + np.random.uniform(0, spread)
        low = current_rate - np.random.uniform(0, spread)
        open_rate = current_rate + np.random.uniform(-spread/2, spread/2)
        close_rate = current_rate
        
        opens.append(round(open_rate, 5))
        highs.append(round(high, 5))
        lows.append(round(low, 5))
        closes.append(round(close_rate, 5))
    
    date_values = [d.date() for d in dates] if date_column == "date" else list(dates)
    return pl.DataFrame({
        date_column: date_values,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes
    })


def create_mock_iv_data(
    ticker: str = "AAPL",
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    base_iv: float = 0.25,
) -> pl.DataFrame:
    """
    Create mock implied volatility data for testing.
    
    Args:
        ticker: Stock symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        base_iv: Base implied volatility (0.25 = 25%)
        
    Returns:
        Polars DataFrame with IV data
    """
    import numpy as np
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)  # For reproducible tests
    
    opens, highs, lows, closes = [], [], [], []
    volumes, averages, bar_counts = [], [], []
    current_iv = base_iv
    
    for _ in dates:
        # IV changes more slowly than prices
        iv_change = np.random.normal(0, 0.01)  # 1% daily volatility of volatility
        current_iv = max(0.05, current_iv * (1 + iv_change))  # Min 5% IV
        
        spread = current_iv * 0.02
        high = current_iv + np.random.uniform(0, spread)
        low = current_iv - np.random.uniform(0, spread)
        open_iv = current_iv + np.random.uniform(-spread/2, spread/2)
        close_iv = current_iv
        
        volume = int(np.random.uniform(50000, 500000))
        average = (high + low + open_iv + close_iv) / 4
        bar_count = int(np.random.uniform(500, 2000))
        
        opens.append(round(open_iv, 4))
        highs.append(round(high, 4))
        lows.append(round(low, 4))
        closes.append(round(close_iv, 4))
        volumes.append(volume)
        averages.append(round(average, 4))
        bar_counts.append(bar_count)
    
    return pl.DataFrame({
        'date': [d.date() for d in dates],
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes,
        'average': averages,
        'barCount': bar_counts
    })


# Pre-generated mock data for common test cases
MOCK_DATA_CACHE: Dict[str, pl.DataFrame] = {}


def get_mock_data(data_type: str, **kwargs) -> pl.DataFrame:
    """
    Get cached mock data or generate it if not cached.
    
    Args:
        data_type: Type of mock data ("daily_equity", "hourly_equity", "minute_equity", 
                  "daily_forex", "hourly_forex", "iv")
        **kwargs: Additional parameters for data generation
        
    Returns:
        Polars DataFrame with mock data
    """
    cache_key = f"{data_type}_{hash(frozenset(kwargs.items()))}"
    
    if cache_key not in MOCK_DATA_CACHE:
        if data_type == "daily_equity":
            MOCK_DATA_CACHE[cache_key] = create_mock_daily_equity_data(**kwargs)
        elif data_type == "hourly_equity":
            kwargs.setdefault("bar_size", "1hour")
            MOCK_DATA_CACHE[cache_key] = create_mock_intraday_equity_data(**kwargs)
        elif data_type == "minute_equity":
            kwargs.setdefault("bar_size", "1min")
            MOCK_DATA_CACHE[cache_key] = create_mock_intraday_equity_data(**kwargs)
        elif data_type == "daily_forex":
            kwargs.setdefault("bar_size", "1d")
            MOCK_DATA_CACHE[cache_key] = create_mock_forex_data(**kwargs)
        elif data_type == "hourly_forex":
            kwargs.setdefault("bar_size", "1hour")
            MOCK_DATA_CACHE[cache_key] = create_mock_forex_data(**kwargs)
        elif data_type == "iv":
            MOCK_DATA_CACHE[cache_key] = create_mock_iv_data(**kwargs)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
    
    return MOCK_DATA_CACHE[cache_key].clone()  # Use clone() instead of copy() for Polars