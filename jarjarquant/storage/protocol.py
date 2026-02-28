"""DataRepository protocol — defines the storage interface for jarjarquant."""

from typing import Protocol

import polars as pl

from jarjarquant.schemas import BarSize, SampleRequest, Sample


class DataRepository(Protocol):
    """Protocol for data storage backends."""

    def get_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        bar_size: BarSize = BarSize.ONE_DAY,
    ) -> pl.DataFrame: ...

    def get_sample(self, request: SampleRequest) -> Sample: ...

    def list_tickers(self) -> list[str]: ...

    def save(self, table_name: str, data: pl.DataFrame) -> None: ...

    def load(self, table_name: str) -> pl.DataFrame: ...
