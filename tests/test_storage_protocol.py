"""Tests for storage protocol."""

import polars as pl
import pytest
from jarjarquant.storage.protocol import DataRepository
from jarjarquant.schemas import BarSize, SampleRequest, Sample


class MockRepository:
    """Verify that any class with the right methods satisfies the protocol."""

    def get_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        bar_size: BarSize = BarSize.ONE_DAY,
    ) -> pl.DataFrame:
        return pl.DataFrame()

    def get_sample(self, request: SampleRequest) -> Sample:
        return Sample(
            start_date="", end_date="", data=pl.DataFrame(), bar_size=BarSize.ONE_DAY
        )

    def list_tickers(self) -> list[str]:
        return []

    def save(self, table_name: str, data: pl.DataFrame) -> None:
        pass

    def load(self, table_name: str) -> pl.DataFrame:
        return pl.DataFrame()


def test_mock_satisfies_protocol():
    repo: DataRepository = MockRepository()
    assert repo.list_tickers() == []
