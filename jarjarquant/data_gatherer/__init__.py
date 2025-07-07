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
        return await self._sources[source].fetch(ticker, **kwargs)

    def get_sync(self, ticker: str, source: str = "eodhd", **kwargs):
        """Sync wrapper around `get`."""
        return asyncio.run(self.get(ticker, source, **kwargs))


# Auto-import all modules in this package's directory
for _, modname, ispkg in pkgutil.iter_modules(__path__):
    if not ispkg:
        importlib.import_module(f"{__name__}.{modname}")
