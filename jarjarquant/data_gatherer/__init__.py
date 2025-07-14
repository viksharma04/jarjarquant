import asyncio
import csv
import importlib
import os
import pkgutil
import random

from jarjarquant.config import DATA_SOURCE_CONFIG

from .base import DataSource, get_all_data_sources, list_data_sources


class DataGatherer:
    def __init__(self):
        self._sources: dict[str, DataSource] = {}
        for name, cls in get_all_data_sources().items():
            cfg = DATA_SOURCE_CONFIG.get(name, {})
            self._sources[name] = cls(**cfg) if cfg else cls()

    @classmethod
    def available_sources(cls) -> list[str]:
        """What you can pass as `source=`."""
        return list_data_sources()

    def get_random_tickers(self, num_tickers: int = 30):
        # Construct the path relative to this file's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(
            base_dir,
            "..",
            "sample_data",
            "data",
            "prices",
            "equities",
            "equities_metadata.csv",
        )
        csv_path = os.path.normpath(csv_path)

        tickers = []
        with open(csv_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ticker = row.get("Symbol")
                if ticker:
                    tickers.append(ticker)

        return random.sample(tickers, min(num_tickers, len(tickers)))

    async def get(self, ticker: str, source: str = "eodhd", **kwargs):
        """
        Fetch from `source`.  E.g.
          await dg.get("AAPL", source="eodhd", start="2025-07-01", end="2025-07-06")
        """
        if source not in self._sources:
            raise ValueError(
                f"Unknown source '{source}'. Available: {self.available_sources()}"
            )
        if ticker.lower() == "random":
            ticker = self.get_random_tickers(1)[0]
        return await self._sources[source].fetch(ticker, **kwargs)

    def get_sync(self, ticker: str, source: str = "eodhd", **kwargs):
        """Sync wrapper around `get`."""
        return asyncio.run(self.get(ticker, source, **kwargs))

    async def generate_synthetic(self, **kwargs):
        """
        Generate synthetic OHLCV data using various statistical distributions.

        This is a convenience method for accessing the synthetic data source.

        Args:
            **kwargs: Parameters passed to SyntheticDataSource.fetch()

        Returns:
            DataFrame with synthetic OHLCV data

        Example:
            # Generate 252 days of normally distributed returns
            data = await dg.generate_synthetic(
                distribution_type="normal",
                periods=252,
                return_mean=0.001,
                return_std=0.02
            )

            # Generate jump diffusion model
            data = await dg.generate_synthetic(
                distribution_type="jump_diffusion",
                periods=1000,
                jump_intensity=0.1,
                jump_size_mean=0.02
            )
        """
        return await self.get("", source="synthetic", **kwargs)

    def generate_synthetic_sync(self, **kwargs):
        """Sync wrapper around generate_synthetic."""
        return asyncio.run(self.generate_synthetic(**kwargs))


# Auto-import all modules in this package's directory
for _, modname, ispkg in pkgutil.iter_modules(__path__):
    if not ispkg:
        importlib.import_module(f"{__name__}.{modname}")
