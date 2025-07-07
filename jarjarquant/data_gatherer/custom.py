import os

import pandas as pd

from .base import DataSource, register_data_source


@register_data_source("custom")
class CustomDataSource(DataSource):
    async def fetch(self, sample_name: str):
        """Reads csv files in a given folder and returns a list of DataFrames

        Args:
            sample_name (str): Path to the folder with the custom sample files
        """
        folder_path = os.path.join(
            os.path.dirname(__file__), "ticker_samples", "data", sample_name
        )
        data = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)

                df = pd.read_csv(
                    file_path,
                    index_col=0,
                    parse_dates=["date"],
                    date_format="%Y-%m-%d %H:%M:%S%z",
                )
                df.set_index("date", inplace=True)
                df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
                data.append(df)

        return data
