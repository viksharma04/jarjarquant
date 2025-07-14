"""Example usage of the DataService for accessing financial data."""

from jarjarquant.data_service import DataService


def main():
    """Demonstrate various DataService capabilities."""

    # Initialize the service
    with DataService("jarjarquant/sample_data/data/") as ds:
        print("=== DataService Example Usage ===\n")

        # 1. List available tickers
        print("1. Available tickers (first 10):")
        tickers = ds.list_available_tickers()
        print(f"   Total tickers: {len(tickers)}")
        print(f"   Sample: {tickers[:10]}")
        print()

        # 2. Get price data for a single ticker
        print("2. Price data for AAPL (last 5 days):")
        aapl_data = ds.get_price_data("AAPL")
        print(aapl_data.tail())
        print()

        # 3. Get price data for multiple tickers
        print("3. Price data for multiple tickers (AAPL, MSFT, GOOGL):")
        multi_data = ds.get_price_data(
            ["AAPL", "MSFT", "GOOGL"], start_date="2024-01-01", end_date="2024-01-31"
        )
        if not multi_data.empty:
            print(f"   Shape: {multi_data.shape}")
            print(
                f"   Date range: {multi_data.index.get_level_values('date').min()} to {multi_data.index.get_level_values('date').max()}"
            )
        print()

        # 4. Get latest prices
        print("4. Latest prices for tech giants:")
        latest = ds.get_latest_prices(["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
        print(latest[["date", "Close", "Volume"]])
        print()

        # 5. Get metadata
        print("5. Metadata for selected tickers:")
        metadata = ds.get_metadata(["AAPL", "MSFT", "GOOGL"])
        if not metadata.empty:
            print(
                metadata[
                    ["Description", "Sector", "Market capitalization", "Analyst Rating"]
                ]
            )
        print()

        # 6. Get sectors
        print("6. Available sectors:")
        sectors = ds.get_sectors()
        print(f"   Total sectors: {len(sectors)}")
        print(f"   Sectors: {', '.join(sectors[:5])}...")
        print()

        # 7. Get sample by criteria
        print("7. Sample large-cap tech stocks:")
        tech_stocks = ds.get_sample_by_criteria(
            n_samples=5,
            sector="Technology services",
            min_market_cap=5e11,  # 500 billion
            random_seed=42,
        )
        print(f"   Selected: {tech_stocks}")

        # Get their metadata
        if tech_stocks:
            tech_metadata = ds.get_metadata(tech_stocks)
            if not tech_metadata.empty:
                print(tech_metadata[["Description", "Market capitalization"]])
        print()

        # 8. Get date range for a ticker
        print("8. Date range for AAPL:")
        start, end = ds.get_date_range("AAPL")
        print(f"   Available from {start.date()} to {end.date()}")
        print(f"   Total days: {(end - start).days}")
        print()

        # 9. Get specific columns
        print("9. Get only Close and Volume for AAPL (last 5 days):")
        close_vol = ds.get_price_data("AAPL", columns=["Close", "Volume"])
        print(close_vol.tail())
        print()

        # 10. Filter by analyst rating
        print("10. Strong buy recommendations:")
        strong_buys = ds.get_metadata(
            filters={"Analyst Rating": "Strong buy", "Sector": "Technology services"}
        )
        if not strong_buys.empty:
            print(f"   Total strong buys: {len(strong_buys)}")
            print("   Top 5 by market cap:")
            top_5 = strong_buys.nlargest(5, "Market capitalization")
            print(top_5[["Description", "Market capitalization", "Sector"]])


if __name__ == "__main__":
    main()
