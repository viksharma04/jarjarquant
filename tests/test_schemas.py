import polars as pl
import pytest
from jarjarquant.schemas import (
    VolatilityMeasure,
    BarSize,
    SampleRequest,
    Sample,
    ADFTestResult,
    NormalityTestResult,
    EntropyResult,
    RangeIQRResult,
)


def test_volatility_measure_has_all_variants():
    assert len(VolatilityMeasure) == 8
    assert VolatilityMeasure.ATR.value == "atr"


def test_bar_size_is_str_enum():
    assert BarSize.ONE_DAY == "1 day"
    assert isinstance(BarSize.ONE_DAY, str)


def test_sample_request_defaults():
    req = SampleRequest(start_date="2020-01-01", end_date="2021-01-01")
    assert req.bar_size == BarSize.ONE_DAY
    assert req.n_samples == 10


def test_sample_holds_polars_dataframe():
    df = pl.DataFrame({"close": [1.0, 2.0]})
    sample = Sample(start_date="2020-01-01", end_date="2021-01-01", data=df, bar_size=BarSize.ONE_DAY)
    assert isinstance(sample.data, pl.DataFrame)


def test_adf_test_result_fields():
    r = ADFTestResult(
        statistic=-3.5, p_value=0.01, lags=2, nobs=100,
        critical_values={"1%": -3.4}, decision="stationary", is_stationary=True,
    )
    assert r.is_stationary is True


def test_normality_test_result_fields():
    r = NormalityTestResult(statistic=1.0, p_value=0.5, method="jarque_bera", decision="normal", is_normal=True)
    assert r.method == "jarque_bera"


def test_entropy_result_fields():
    r = EntropyResult(entropy=2.0, normalized_entropy=0.8, n_bins=10, quality="good", is_concerning=False)
    assert r.is_concerning is False


def test_range_iqr_result_fields():
    r = RangeIQRResult(ratio=4.0, range_val=10.0, iqr=2.5, q1=2.0, q3=4.5, assessment="normal")
    assert r.ratio == 4.0
