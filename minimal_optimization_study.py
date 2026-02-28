import asyncio
import logging

import pandas as pd

from jarjarquant import (
    IndicatorSpec,
    IndicatorType,
    Jarjarquant,
)
from jarjarquant.data_gatherer.synthetic import DistributionType, SyntheticDataSource

# Setup logging
logging.basicConfig(level=logging.INFO)


async def get_synthetic_data():
    source = SyntheticDataSource()
    return await source.fetch(
        ticker="SYNTHETIC",
        distribution_type=DistributionType.NORMAL,
        periods=10000,
        return_mean=0.0005,
        return_std=0.01,
    )


def main():
    # Initialize the main class
    jjq = Jarjarquant()

    # Define the indicator: Price Intensity with smoothing_factor=3
    pi_3 = IndicatorSpec(
        IndicatorType.PRICE_INTENSITY, parameters={"smoothing_factor": 4}
    )

    # Generate synthetic data
    print("Generating synthetic data...")
    ohlcv_df = asyncio.run(get_synthetic_data())
    print(f"Generated {len(ohlcv_df)} rows of synthetic data.")

    # Prepare inputs for threshold_optimization_study
    inputs = {
        "indicator_spec": pi_3,
        "ohlcv_df": ohlcv_df,
        "ticker": "SYNTHETIC_1",
    }

    # Run the study
    print("Starting threshold optimization study...")
    result = jjq.feature_evaluator.threshold_optimization_study(inputs)

    print("Study completed.")
    print("Result:")
    print(pd.Series(result))


if __name__ == "__main__":
    main()
