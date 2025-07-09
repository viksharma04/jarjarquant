import os
from datetime import datetime
from typing import Optional

import duckdb
from dateutil.relativedelta import relativedelta

from .base import DataSource, register_data_source
from .utils import DURATION_TO_DAYS_MAP, BarSize, Duration


@register_data_source("custom")
class CustomDataSource(DataSource):
    async def fetch(
        self,
        ticker: str = "MSFT",
        bar_size: BarSize = BarSize.ONE_DAY,
        duration: Duration = Duration.ONE_MONTH,
        end_date: Optional[str] = None,
        security_type: str = "STK",
        database_folder: str = "sample_data/",
    ):
        security_map = {"STK": "equities"}
        sec_folder = security_map.get(security_type, security_type.lower())
        # Map bar_size to folder name
        bar_size_map = {
            BarSize.ONE_MINUTE: "1min",
            BarSize.ONE_HOUR: "1hour",
            BarSize.ONE_DAY: "1d",
            # Add more mappings as needed
        }
        bar_folder = bar_size_map.get(bar_size, str(bar_size).lower())

        # Build path to parquet file
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        parquet_path = os.path.join(
            parent_dir,
            database_folder,
            "data",
            "prices",
            sec_folder,
            bar_folder,
            f"{ticker}.parquet",
        )
        parquet_path = os.path.normpath(parquet_path)

        con = duckdb.connect(database=":memory:")
        query = f"SELECT * FROM '{parquet_path}'"

        if end_date:
            duration_days = DURATION_TO_DAYS_MAP[duration]
            parsed_end_date = datetime.strptime(end_date, "%Y-%m-%d")
            start_date = parsed_end_date - relativedelta(days=duration_days)
            query += (
                f" WHERE datetime > '{start_date.strftime('%Y-%m-%d')}'"
                f" AND datetime <= '{parsed_end_date.strftime('%Y-%m-%d')}'"
            )

        df = con.execute(query).fetch_df()
        con.close()

        return df
