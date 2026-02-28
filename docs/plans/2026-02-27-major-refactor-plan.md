# Jarjarquant Major Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor jarjarquant from a monolithic orchestrator-driven design into a composable, function-first library with clean separation of concerns, Polars-only data flow, and minimal dependencies.

**Architecture:** Hybrid flat/nested package — flat modules for simple concerns (transforms, volatility, labelling), subdirectories for multi-file domains (indicators/, evaluation/, storage/). No orchestrator class. BYOD (Bring Your Own Data) — data fetching archived entirely.

**Tech Stack:** Python 3.12+, Polars, NumPy, Numba, Cython, DuckDB, scikit-learn, scipy, statsmodels

**Design Doc:** `docs/plans/2026-02-27-major-refactor-design.md`

---

## Phase 1: Foundation — Schemas, Transforms, Archive

> Set up the new module structure and migrate the simplest, most self-contained pieces first. Archive dead code. This phase has zero risk — nothing existing breaks because new modules are additive.

### Task 1.1: Archive old modules and create directory structure

**Files:**
- Move: `jarjarquant/jarjarquant.py` → `jarjarquant/_archive/jarjarquant.py`
- Move: `jarjarquant/data_gatherer/` → `jarjarquant/_archive/data_gatherer/`
- Move: `jarjarquant/data_analyst.py` → `jarjarquant/_archive/data_analyst.py`
- Create: `jarjarquant/evaluation/__init__.py` (empty)
- Create: `jarjarquant/storage/__init__.py` (empty)
- Rename: `jarjarquant/cython_utils/` → `jarjarquant/_cython/`

**Step 1: Create _archive directory and move files**

```bash
cd jarjarquant
mkdir -p _archive
git mv jarjarquant.py _archive/jarjarquant.py
git mv data_gatherer _archive/data_gatherer
git mv data_analyst.py _archive/data_analyst.py
```

**Step 2: Rename cython_utils to _cython**

```bash
git mv cython_utils _cython
```

**Step 3: Create new subdirectory stubs**

```bash
mkdir -p evaluation storage
touch evaluation/__init__.py storage/__init__.py
```

**Step 4: Update setup.py Cython paths**

Change `jarjarquant/cython_utils/*.pyx` → `jarjarquant/_cython/*.pyx` in all 3 Extension definitions.

**Step 5: Verify Cython build still works**

```bash
cd ../..
source .venv/Scripts/activate
python setup.py build_ext --inplace
```

Expected: 3 extensions compile successfully.

**Step 6: Commit**

```bash
git add -A
git commit -m "refactor: archive old modules, rename cython_utils to _cython, create new directory structure"
```

---

### Task 1.2: Create schemas.py — consolidated data contracts

**Files:**
- Create: `jarjarquant/schemas.py`
- Source from: `jarjarquant/core/schemas.py` (VolatilityMeasure), `jarjarquant/data_gatherer/utils.py` (BarSize), `jarjarquant/data_service.py` (SampleRequest, Sample), `jarjarquant/_archive/data_analyst.py` (ADFTestResult, NormalityTestResult, EntropyResult, RangeIQRResult)
- Test: `tests/test_schemas.py`

**Step 1: Write the failing test**

```python
# tests/test_schemas.py
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
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'jarjarquant.schemas'`

**Step 3: Write schemas.py**

Consolidate from these sources:
- `VolatilityMeasure` from `jarjarquant/core/schemas.py` (lines 4-12)
- `BarSize` from `jarjarquant/data_gatherer/utils.py` (lines 10-34) — copy enum values, keep as StrEnum
- `SampleRequest` from `jarjarquant/data_service.py` (lines 47-63) — simplified: remove BaseParams/EquityParams/ForexParams, remove `__post_init__`
- `Sample` from `jarjarquant/data_service.py` (lines 69-76) — simplified: remove Generic[TParams], remove sample_type
- `ADFTestResult` from `jarjarquant/_archive/data_analyst.py` (line 21)
- `NormalityTestResult` from `jarjarquant/_archive/data_analyst.py` (line 34)
- `EntropyResult` from `jarjarquant/_archive/data_analyst.py` (line 45)
- `RangeIQRResult` from `jarjarquant/_archive/data_analyst.py` (line 57)

Key changes:
- `SampleRequest`: Remove `params` field and `__post_init__`. Just `start_date`, `end_date`, `bar_size`, `n_samples`.
- `Sample`: Remove `sample_type` and generic parameter. Just `start_date`, `end_date`, `data: pl.DataFrame`, `bar_size`.
- `RangeIQRResult`: Rename `range` field to `range_val` (avoid shadowing builtin).

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_schemas.py -v
```

Expected: All 8 tests PASS.

**Step 5: Commit**

```bash
git add jarjarquant/schemas.py tests/test_schemas.py
git commit -m "feat: add consolidated schemas module with all data contracts"
```

---

### Task 1.3: Create transforms.py — pure transformation functions

**Files:**
- Create: `jarjarquant/transforms.py`
- Source from: `jarjarquant/feature_engineer.py` (lines 260-370 — simple_root, log_transform, sigmoid_transform, transform, exp_smoothing)
- Test: `tests/test_transforms.py`

**Step 1: Write the failing test**

```python
# tests/test_transforms.py
import numpy as np
import pytest
from jarjarquant.transforms import (
    root_transform,
    log_transform,
    sigmoid_transform,
    exp_smoothing,
    apply_transform,
)


def test_root_transform_preserves_sign():
    data = np.array([-4.0, 0.0, 9.0])
    result = root_transform(data, degree=2)
    assert result[0] < 0  # negative preserved
    assert result[1] == 0.0
    assert result[2] > 0


def test_log_transform_handles_negatives():
    data = np.array([-1.0, 0.0, 1.0, 10.0])
    result = log_transform(data)
    assert not np.any(np.isnan(result))


def test_sigmoid_transform_bounds():
    data = np.array([-100.0, 0.0, 100.0])
    result = sigmoid_transform(data)
    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)


def test_exp_smoothing():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = exp_smoothing(data, span=3)
    assert len(result) == len(data)
    assert result[-1] != data[-1]  # smoothed, not identity


def test_apply_transform_dispatches():
    data = np.array([1.0, 4.0, 9.0])
    result = apply_transform(data, "root")
    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0]))


def test_apply_transform_unknown_raises():
    with pytest.raises(ValueError, match="Unknown transform"):
        apply_transform(np.array([1.0]), "nonexistent")
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_transforms.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write transforms.py**

Extract from `feature_engineer.py`:
- `simple_root` (line 260) → rename to `root_transform`
- `log_transform` (line 277) → keep name
- `sigmoid_transform` (line 301) → keep name
- `exp_smoothing` (line 357) → keep name
- `transform` (line 330) → rename to `apply_transform`, convert from method to function

All become module-level pure functions. Remove `self` parameter. Remove `FeatureEngineer` dependency. Input: `np.ndarray`. Output: `np.ndarray`.

The `apply_transform` function dispatches by string name:
```python
def apply_transform(data: np.ndarray, method: str, **kwargs) -> np.ndarray:
    transforms = {
        "root": root_transform,
        "log": log_transform,
        "sigmoid": sigmoid_transform,
    }
    if method not in transforms:
        raise ValueError(f"Unknown transform: '{method}'. Available: {list(transforms.keys())}")
    return transforms[method](data, **kwargs)
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_transforms.py -v
```

Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add jarjarquant/transforms.py tests/test_transforms.py
git commit -m "feat: add transforms module with pure transformation functions"
```

---

### Task 1.4: Create fractional_diff.py — pure differentiation functions

**Files:**
- Create: `jarjarquant/fractional_diff.py`
- Source from: `jarjarquant/feature_engineer.py` (lines 27-165 — get_weights, frac_diff, getWeights_FFD, fracDiff_FFD)
- Test: `tests/test_fractional_diff.py`

**Step 1: Write the failing test**

```python
# tests/test_fractional_diff.py
import numpy as np
import polars as pl
import pytest
from jarjarquant.fractional_diff import get_weights, frac_diff, get_weights_ffd, frac_diff_ffd


def test_get_weights_length():
    weights = get_weights(d=0.5, size=10)
    assert len(weights) == 10


def test_get_weights_first_is_one():
    weights = get_weights(d=0.5, size=5)
    assert weights[0] == 1.0


def test_frac_diff_returns_series():
    series = pl.Series("price", np.cumsum(np.random.randn(100)) + 100)
    result = frac_diff(series, d=0.5)
    assert isinstance(result, pl.Series)
    assert len(result) == len(series)


def test_frac_diff_ffd_returns_series():
    series = pl.Series("price", np.cumsum(np.random.randn(100)) + 100)
    result = frac_diff_ffd(series, d=0.5, threshold=1e-4)
    assert isinstance(result, pl.Series)
    assert len(result) == len(series)


def test_frac_diff_d_zero_is_identity():
    values = np.cumsum(np.random.randn(50)) + 100
    series = pl.Series("price", values)
    result = frac_diff(series, d=0.0)
    # d=0 means no differentiation — should be close to original
    non_null = result.drop_nulls()
    assert len(non_null) > 0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_fractional_diff.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write fractional_diff.py**

Extract from `feature_engineer.py`:
- `get_weights` (line 27) → module-level function, remove `self`
- `frac_diff` (line 52) → module-level function, change input from `pd.Series` to `pl.Series`, remove `self`
- `getWeights_FFD` (line 96) → rename to `get_weights_ffd`, module-level function
- `fracDiff_FFD` (line 121) → rename to `frac_diff_ffd`, change input from `pd.Series` to `pl.Series`

Key changes:
- All pandas operations converted to Polars equivalents
- No `self` parameter anywhere
- No class instantiation needed
- Snake_case naming throughout

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_fractional_diff.py -v
```

Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add jarjarquant/fractional_diff.py tests/test_fractional_diff.py
git commit -m "feat: add fractional_diff module with pure differentiation functions"
```

---

### Task 1.5: Create permutation.py — bar and price permutation

**Files:**
- Create: `jarjarquant/permutation.py`
- Source from: `jarjarquant/feature_engineer.py` (lines 379-578 — BarPermute, PricePermute)
- Test: `tests/test_permutation.py`

**Step 1: Write the failing test**

```python
# tests/test_permutation.py
import numpy as np
import polars as pl
import pytest
from jarjarquant.permutation import BarPermute, PricePermute


def test_bar_permute_init():
    n = 50
    df = pl.DataFrame({
        "Open": np.random.randn(n).cumsum() + 100,
        "High": np.random.randn(n).cumsum() + 102,
        "Low": np.random.randn(n).cumsum() + 98,
        "Close": np.random.randn(n).cumsum() + 100,
    })
    bp = BarPermute(df)
    assert bp.basis_prices is not None
    assert bp.relative_prices is not None


def test_bar_permute_output_shape():
    n = 50
    close = np.cumsum(np.random.randn(n)) + 100
    df = pl.DataFrame({
        "Open": close + np.random.randn(n) * 0.1,
        "High": close + abs(np.random.randn(n)),
        "Low": close - abs(np.random.randn(n)),
        "Close": close,
    })
    bp = BarPermute(df)
    result = bp.permute()
    assert isinstance(result, pl.DataFrame)
    assert "Open" in result.columns
    assert "Close" in result.columns
    assert len(result) == n


def test_price_permute_init():
    prices = pl.Series("price", np.cumsum(np.random.randn(50)) + 100)
    pp = PricePermute(prices)
    assert pp.basis_prices is not None


def test_price_permute_output_length():
    prices = pl.Series("price", np.cumsum(np.random.randn(50)) + 100)
    pp = PricePermute(prices)
    result = pp.permute()
    assert isinstance(result, pl.Series)
    assert len(result) == 50
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_permutation.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write permutation.py**

Extract from `feature_engineer.py`:
- `BarPermute` class (line 379) — convert internal pandas to Polars. Keep Cython import: `from jarjarquant._cython.bar_permute import permute_cython, permute_cython_single`
- `PricePermute` class (line 485) — convert internal pandas to Polars

Key changes:
- Input: `pl.DataFrame` (BarPermute) or `pl.Series` (PricePermute) instead of pandas
- Output: `pl.DataFrame` or `pl.Series` instead of pandas
- Internal numpy operations stay the same (Cython interface unchanged)
- Update Cython import path from `cython_utils` to `_cython`

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_permutation.py -v
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add jarjarquant/permutation.py tests/test_permutation.py
git commit -m "feat: add permutation module with BarPermute and PricePermute"
```

---

### Task 1.6: Move volatility.py to top level

**Files:**
- Create: `jarjarquant/volatility.py`
- Source from: `jarjarquant/core/volatility_calculations.py` (entire file, 391 lines)
- Test: `tests/test_volatility.py`

**Step 1: Write the failing test**

```python
# tests/test_volatility.py
import numpy as np
import pytest
from jarjarquant.volatility import calculate_volatility
from jarjarquant.schemas import VolatilityMeasure


def test_atr_volatility_output_shape():
    n = 100
    high = np.random.randn(n).cumsum() + 102
    low = high - abs(np.random.randn(n)) - 0.5
    close = (high + low) / 2
    result = calculate_volatility(high, low, close, VolatilityMeasure.ATR, window=14)
    assert len(result) == n


def test_ewm_std_volatility():
    close = np.cumsum(np.random.randn(100)) + 100
    high = close + 1
    low = close - 1
    result = calculate_volatility(high, low, close, VolatilityMeasure.EWM_STD, window=20)
    assert len(result) == len(close)
    assert not np.all(np.isnan(result))


def test_invalid_measure_raises():
    with pytest.raises((ValueError, KeyError)):
        calculate_volatility(
            np.array([1.0]), np.array([1.0]), np.array([1.0]),
            "invalid", window=10,
        )
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_volatility.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Create volatility.py**

Copy `jarjarquant/core/volatility_calculations.py` to `jarjarquant/volatility.py`. Change import:
- From: `from .schemas import VolatilityMeasure`
- To: `from jarjarquant.schemas import VolatilityMeasure`

No other changes needed — the file is already well-structured with pure functions.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_volatility.py -v
```

Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add jarjarquant/volatility.py tests/test_volatility.py
git commit -m "feat: add top-level volatility module (moved from core/)"
```

---

## Phase 2: Indicator System Refactor

> Fix the broken transform pattern, refactor base class to use _compute(), update all 16 indicators.

### Task 2.1: Refactor Indicator base class with transform support

**Files:**
- Modify: `jarjarquant/indicators/base.py`
- Test: `tests/test_indicator_base.py`

**Step 1: Write the failing test**

```python
# tests/test_indicator_base.py
import numpy as np
import polars as pl
import pytest
from jarjarquant.indicators.base import Indicator, IndicatorSpec
from jarjarquant.indicators.registry import IndicatorType


class MockIndicator(Indicator):
    """Test indicator that returns a fixed array."""

    def _compute(self) -> np.ndarray:
        return np.array([1.0, 4.0, 9.0, 16.0])


def test_indicator_calculate_without_transform():
    df = pl.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100.0]})
    ind = MockIndicator(df)
    result = ind.calculate()
    np.testing.assert_array_equal(result, np.array([1.0, 4.0, 9.0, 16.0]))


def test_indicator_calculate_with_root_transform():
    df = pl.DataFrame({"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [100.0]})
    ind = MockIndicator(df)
    ind._transform = "root"
    result = ind.calculate()
    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0, 4.0]))


def test_indicator_spec_creates_indicator():
    spec = IndicatorSpec(IndicatorType.RSI, {"period": 14})
    df = pl.DataFrame({
        "Open": np.random.randn(50).cumsum() + 100,
        "High": np.random.randn(50).cumsum() + 102,
        "Low": np.random.randn(50).cumsum() + 98,
        "Close": np.random.randn(50).cumsum() + 100,
        "Volume": np.random.rand(50) * 1000,
    })
    indicator = spec.create_indicator(df)
    result = indicator.calculate()
    assert isinstance(result, np.ndarray)


def test_indicator_spec_with_transform():
    spec = IndicatorSpec(IndicatorType.RSI, {"period": 14, "transform": "log"})
    assert spec.parameters.get("transform") == "log"
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_indicator_base.py -v
```

Expected: FAIL — MockIndicator has no `_compute` method (old base class doesn't define it)

**Step 3: Rewrite indicators/base.py**

Replace the `Indicator` class (currently lines 91-109 in `indicators/base.py`):

```python
from jarjarquant.transforms import apply_transform

class Indicator:
    """Base class for all technical indicators."""

    def __init__(self, ohlcv_df: pl.DataFrame):
        if ohlcv_df is None or ohlcv_df.is_empty():
            raise ValueError("ohlcv_df must be a non-empty Polars DataFrame")
        self._df = ohlcv_df
        self._transform: str | None = None

    def calculate(self) -> np.ndarray:
        """Calculate indicator values, applying transform if set."""
        raw = self._compute()
        if self._transform is not None:
            return apply_transform(raw, self._transform)
        return raw

    def _compute(self) -> np.ndarray:
        """Override in subclasses — raw indicator computation."""
        raise NotImplementedError
```

Keep `IndicatorSpec` as-is (it already works). The `transform` parameter flows through `parameters` dict into the indicator constructor, which sets `self._transform`.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_indicator_base.py -v
```

Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add jarjarquant/indicators/base.py tests/test_indicator_base.py
git commit -m "refactor: update Indicator base class with _compute() and built-in transform support"
```

---

### Task 2.2: Update all indicator implementations to use _compute()

**Files:**
- Modify: All 16 files in `jarjarquant/indicators/` (rsi.py, macd.py, adx.py, aroon.py, stochastic.py, stochastic_rsi.py, chaikin_money_flow.py, detrended_rsi.py, cmma.py, moving_average_difference.py, price_change_oscillator.py, price_intensity.py, regression_trend.py, regression_trend_deviation.py, anchored_vwap.py, gap_size.py)
- Test: `tests/test_indicators.py`

**Step 1: Write the failing test**

```python
# tests/test_indicators.py
import numpy as np
import polars as pl
import pytest
from jarjarquant.indicators.registry import IndicatorType, list_available_indicators, get_indicator_class


def _make_ohlcv(n: int = 200) -> pl.DataFrame:
    close = np.cumsum(np.random.randn(n)) + 100
    return pl.DataFrame({
        "Open": close + np.random.randn(n) * 0.5,
        "High": close + abs(np.random.randn(n)),
        "Low": close - abs(np.random.randn(n)),
        "Close": close,
        "Volume": (np.random.rand(n) * 1e6).astype(np.float64),
    })


@pytest.mark.parametrize("indicator_type", list(IndicatorType))
def test_every_indicator_calculates(indicator_type):
    """Every registered indicator must produce an ndarray without crashing."""
    cls = get_indicator_class(indicator_type)
    df = _make_ohlcv(200)
    indicator = cls(df)
    result = indicator.calculate()
    assert isinstance(result, np.ndarray)
    assert len(result) == 200


@pytest.mark.parametrize("indicator_type", list(IndicatorType))
def test_every_indicator_with_transform(indicator_type):
    """Every indicator must work with a transform applied."""
    cls = get_indicator_class(indicator_type)
    df = _make_ohlcv(200)
    indicator = cls(df)
    indicator._transform = "root"
    result = indicator.calculate()
    assert isinstance(result, np.ndarray)
    assert len(result) == 200
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_indicators.py -v
```

Expected: FAIL — indicators still use `calculate()` directly, not `_compute()`

**Step 3: Update each indicator**

For each of the 16 indicators, apply this pattern:

1. Rename `calculate` method → `_compute`
2. Remove any `self.feature_engineer.transform(...)` calls (the base class handles this now)
3. Remove `self.transform` assignment from `__init__` — instead, if `transform` kwarg is passed, set `self._transform = transform`
4. Remove `from jarjarquant.feature_engineer import FeatureEngineer` imports
5. Remove any `self.feature_engineer = FeatureEngineer()` instantiation (in chaikin_money_flow.py and cmma.py)
6. Update Cython import paths from `cython_utils` to `_cython` where applicable

Pattern for each indicator:
```python
# Before:
class RSI(Indicator):
    def __init__(self, ohlcv_df, period=14, transform=None):
        super().__init__(ohlcv_df)
        self.period = period
        self.transform = transform

    def calculate(self):
        # ... compute raw ...
        if self.transform is not None:
            output = self.feature_engineer.transform(output, self.transform)
        return output

# After:
class RSI(Indicator):
    def __init__(self, ohlcv_df, period=14, transform=None):
        super().__init__(ohlcv_df)
        self.period = period
        self._transform = transform

    def _compute(self):
        # ... compute raw (same logic, no transform) ...
        return output
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_indicators.py -v
```

Expected: All 32 tests PASS (16 indicators × 2 tests each).

**Step 5: Run existing indicator tests too**

```bash
uv run pytest tests/ -v --maxfail=5
```

Expected: No regressions in other test files.

**Step 6: Commit**

```bash
git add jarjarquant/indicators/
git commit -m "refactor: update all indicators to use _compute() pattern with base class transform support"
```

---

### Task 2.3: Update indicators/__init__.py imports

**Files:**
- Modify: `jarjarquant/indicators/__init__.py`

**Step 1: Simplify the __init__.py**

The current file (48 lines) imports every indicator class individually. Keep this pattern but ensure it exports cleanly:

```python
from .base import Indicator, IndicatorSpec
from .registry import (
    IndicatorType,
    get_indicator_class,
    get_indicator_parameters,
    list_available_indicators,
    is_indicator_registered,
)
# Individual indicators imported to trigger registration
from .rsi import RSI
from .macd import MACD
# ... (all others)
```

**Step 2: Verify all indicators are registered**

```bash
uv run python -c "from jarjarquant.indicators import list_available_indicators; list_available_indicators(verbose=True)"
```

Expected: All 16 indicators listed.

**Step 3: Commit**

```bash
git add jarjarquant/indicators/__init__.py
git commit -m "refactor: clean up indicators __init__.py exports"
```

---

## Phase 3: Labelling Refactor

> Convert Labeller class to pure functions. Remove pandas dependency. Remove old triple_barrier_method.

### Task 3.1: Create labelling.py with pure functions

**Files:**
- Create: `jarjarquant/labelling.py`
- Source from: `jarjarquant/labeller.py` (799 lines)
- Test: `tests/test_labelling.py`

**Step 1: Write the failing test**

```python
# tests/test_labelling.py
import numpy as np
import polars as pl
import pytest
from jarjarquant.labelling import (
    inverse_cumsum_filter,
    event_sampling,
    triple_barrier_labels,
    one_period_with_sl,
    n_period_with_sl,
    get_sample_weights,
    get_vertical_barrier,
)


def _make_prices(n: int = 500) -> np.ndarray:
    return np.cumsum(np.random.randn(n)) + 100


def test_inverse_cumsum_filter():
    close = _make_prices(500)
    result = inverse_cumsum_filter(close, threshold=0.02)
    assert isinstance(result, np.ndarray)
    assert result.dtype == bool or result.dtype == np.int64  # indices or mask


def test_triple_barrier_labels_returns_polars():
    n = 200
    close = _make_prices(n)
    vol = np.full(n, 0.02)
    dates = pl.Series("date", pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 1) + pl.duration(days=n - 1), eager=True))
    result = triple_barrier_labels(
        close=close,
        dates=dates.to_numpy(),
        volatility=vol,
        vertical_barrier_days=5,
        upper_barrier_mult=2.0,
        lower_barrier_mult=2.0,
    )
    assert isinstance(result, pl.DataFrame)
    assert "label" in result.columns


def test_get_sample_weights():
    # Minimal test that it runs
    labels_df = pl.DataFrame({
        "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 10), eager=True),
        "exit_date": pl.date_range(pl.date(2020, 1, 3), pl.date(2020, 1, 12), eager=True),
        "label": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        "returns": np.random.randn(10) * 0.01,
    })
    close = _make_prices(20)
    result = get_sample_weights(labels_df, close)
    assert isinstance(result, np.ndarray)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_labelling.py -v
```

Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write labelling.py**

Extract from `jarjarquant/labeller.py`:

**Keep (as module-level functions):**
- Numba JIT functions: `_find_barrier_exits` (line 17), `_one_period_sl_labels` (line 90), `_n_period_sl_labels` (line 143) — these are already module-level
- `inverse_cumsum_filter` — convert from method to function. Input: `np.ndarray` close prices and threshold. Remove `self._df` usage.
- `event_sampling` — convert from method to function. Input: close array, volatility array, method string. Remove `self._df` usage.
- `get_vertical_barrier` — already static, make module-level
- `triple_barrier_labels` — already static, make module-level. Keep numba version only.
- `one_period_with_sl` — already static, make module-level
- `n_period_with_sl` — already static, make module-level
- `num_co_events` — already static, make module-level
- `average_uniqueness` — already static, make module-level
- `get_sample_weights` — already static, make module-level

**Remove:**
- `Labeller` class entirely
- `triple_barrier_method` (old pandas version, lines 344-410)
- `add_labels` method (instance state mutation)
- `add_sample_weights` method (instance state mutation)
- `plot_with_flags` (visualization)
- `find_min_column` (utility only used by removed triple_barrier_method)

**Convert pandas → Polars:**
- Any remaining pandas Series/DataFrame outputs → Polars
- Internally, most compute is already numpy. Just change the output wrapping.

**Import changes:**
- `from jarjarquant.volatility import ewm_std_volatility` (instead of from core)
- Remove `from .data_analyst import get_daily_vol` — inline or extract the simple volatility calculation

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_labelling.py -v
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add jarjarquant/labelling.py tests/test_labelling.py
git commit -m "feat: add labelling module with pure functions (replaces Labeller class)"
```

---

## Phase 4: Evaluation Decomposition

> Split the 1354-line FeatureEvaluator into 4 focused modules.

### Task 4.1: Create evaluation/cross_validation.py

**Files:**
- Create: `jarjarquant/evaluation/cross_validation.py`
- Source from: `jarjarquant/feature_evaluator.py` (PurgedKFold lines 81-150, cv_score lines 173-247)
- Test: `tests/test_evaluation_cv.py`

**Step 1: Write the failing test**

```python
# tests/test_evaluation_cv.py
import numpy as np
import pytest
from jarjarquant.evaluation.cross_validation import PurgedKFold, cv_score


def test_purged_kfold_produces_correct_n_splits():
    n = 100
    kf = PurgedKFold(n_splits=5, embargo_pct=0.01)
    X = np.random.randn(n, 3)
    labels = np.array([0, 1] * (n // 2))
    splits = list(kf.split(X, labels))
    assert len(splits) == 5


def test_purged_kfold_no_overlap():
    n = 100
    kf = PurgedKFold(n_splits=3)
    X = np.random.randn(n, 3)
    splits = list(kf.split(X))
    for train_idx, test_idx in splits:
        assert len(set(train_idx) & set(test_idx)) == 0
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_evaluation_cv.py -v
```

**Step 3: Write cross_validation.py**

Extract `PurgedKFold` class and `cv_score` function from `feature_evaluator.py`. Convert `cv_score` from static method to module-level function. Remove `self` references. Keep sklearn imports.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_evaluation_cv.py -v
```

**Step 5: Commit**

```bash
git add jarjarquant/evaluation/cross_validation.py tests/test_evaluation_cv.py
git commit -m "feat: add evaluation/cross_validation with PurgedKFold and cv_score"
```

---

### Task 4.2: Create evaluation/importance.py

**Files:**
- Create: `jarjarquant/evaluation/importance.py`
- Source from: `jarjarquant/feature_evaluator.py` (feature_importance_MDI lines 249-294, feature_importance_MDA lines 295-410, feature_importance_SFI lines 411-487)
- Test: `tests/test_evaluation_importance.py`

**Step 1: Write the failing test**

```python
# tests/test_evaluation_importance.py
import numpy as np
import polars as pl
import pytest
from jarjarquant.evaluation.importance import feature_importance_mdi, feature_importance_mda


def test_feature_importance_mdi_returns_dataframe():
    X = np.random.randn(200, 5)
    y = (X[:, 0] > 0).astype(int)
    result = feature_importance_mdi(X, y, n_estimators=10)
    assert isinstance(result, pl.DataFrame)
    assert len(result) == 5  # 5 features


def test_feature_importance_mda_returns_dataframe():
    X = np.random.randn(200, 3)
    y = (X[:, 0] > 0).astype(int)
    sw = np.ones(200)
    result = feature_importance_mda(X, y, sample_weight=sw, n_estimators=10, n_splits=3)
    assert isinstance(result, pl.DataFrame)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_evaluation_importance.py -v
```

**Step 3: Write importance.py**

Extract MDI, MDA, SFI from `feature_evaluator.py`. Convert from methods to functions. Change output from pandas to Polars. Import `PurgedKFold` and `cv_score` from `.cross_validation`.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_evaluation_importance.py -v
```

**Step 5: Commit**

```bash
git add jarjarquant/evaluation/importance.py tests/test_evaluation_importance.py
git commit -m "feat: add evaluation/importance with MDI, MDA, SFI feature importance"
```

---

### Task 4.3: Create evaluation/distribution.py

**Files:**
- Create: `jarjarquant/evaluation/distribution.py`
- Source from: `jarjarquant/_archive/data_analyst.py` (adf_test, jb_normality_test, relative_entropy, range_iqr_ratio) + `jarjarquant/feature_evaluator.py` (indicator_design_eval, indicator_distribution_study, parallel_indicator_distribution_study)
- Test: `tests/test_evaluation_distribution.py`

**Step 1: Write the failing test**

```python
# tests/test_evaluation_distribution.py
import numpy as np
import pytest
from jarjarquant.evaluation.distribution import (
    adf_test,
    jb_normality_test,
    relative_entropy,
    range_iqr_ratio,
    indicator_design_eval,
)
from jarjarquant.schemas import ADFTestResult, NormalityTestResult, EntropyResult, RangeIQRResult


def test_adf_test_returns_result():
    data = np.random.randn(200)
    result = adf_test(data)
    assert isinstance(result, ADFTestResult)
    assert isinstance(result.is_stationary, bool)


def test_jb_normality_test_returns_result():
    data = np.random.randn(200)
    result = jb_normality_test(data)
    assert isinstance(result, NormalityTestResult)


def test_relative_entropy_returns_result():
    data = np.random.randn(200)
    result = relative_entropy(data)
    assert isinstance(result, EntropyResult)


def test_range_iqr_ratio_returns_result():
    data = np.random.randn(200)
    result = range_iqr_ratio(data)
    assert isinstance(result, RangeIQRResult)


def test_indicator_design_eval_returns_dict():
    data = np.random.randn(300)
    result = indicator_design_eval(data)
    assert "adf" in result
    assert "normality" in result
    assert "entropy" in result
    assert "range_iqr" in result
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_evaluation_distribution.py -v
```

**Step 3: Write distribution.py**

Merge statistical test functions from `data_analyst.py` with distribution study orchestration from `feature_evaluator.py`:
- `adf_test`, `adf_test_ultra_fast` from data_analyst.py
- `jb_normality_test` from data_analyst.py
- `relative_entropy` from data_analyst.py
- `range_iqr_ratio` from data_analyst.py
- `indicator_design_eval` from feature_evaluator.py (line 534) — calls the above tests
- `indicator_distribution_study` from feature_evaluator.py (line 558) — single ticker
- `parallel_indicator_distribution_study` from feature_evaluator.py (line 574) — parallel across tickers

Remove all database caching logic (`_save_detailed_results`, `_check_existing_results`). Remove visualization functions (`plot_loess`). Import result dataclasses from `jarjarquant.schemas`.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_evaluation_distribution.py -v
```

**Step 5: Commit**

```bash
git add jarjarquant/evaluation/distribution.py tests/test_evaluation_distribution.py
git commit -m "feat: add evaluation/distribution with statistical tests and distribution studies"
```

---

### Task 4.4: Create evaluation/threshold.py

**Files:**
- Create: `jarjarquant/evaluation/threshold.py`
- Source from: `jarjarquant/feature_evaluator.py` (indicator_threshold_search line 665, optimize_threshold line 939, etc.)
- Test: `tests/test_evaluation_threshold.py`

**Step 1: Write the failing test**

```python
# tests/test_evaluation_threshold.py
import numpy as np
import pytest
from jarjarquant.evaluation.threshold import optimize_threshold, threshold_search


def test_optimize_threshold_returns_tuple():
    signal = np.sort(np.random.randn(200))
    returns = np.random.randn(200) * 0.01
    result = optimize_threshold(signal, returns, min_kept=0.1)
    assert isinstance(result, tuple)
    assert len(result) == 6  # best_high_idx, best_low_idx, high_pf, low_pf, high_acc, low_acc
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_evaluation_threshold.py -v
```

**Step 3: Write threshold.py**

Extract from `feature_evaluator.py`:
- `optimize_threshold` — wraps Cython `optimize_threshold_cython`. Update import to `from jarjarquant._cython.opt_threshold import optimize_threshold_cython`.
- `indicator_threshold_search`, `single_indicator_threshold_search`, `parallel_indicator_threshold_search` — convert from static methods to functions.
- `threshold_optimization_study`, `parallel_threshold_optimization_study` — convert to functions.

Remove database caching. Remove `_flatten_dataclass` usage (no more result persistence).

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_evaluation_threshold.py -v
```

**Step 5: Commit**

```bash
git add jarjarquant/evaluation/threshold.py tests/test_evaluation_threshold.py
git commit -m "feat: add evaluation/threshold with threshold search and optimization"
```

---

### Task 4.5: Wire up evaluation/__init__.py

**Files:**
- Modify: `jarjarquant/evaluation/__init__.py`

**Step 1: Write the public re-exports**

```python
# jarjarquant/evaluation/__init__.py
from .cross_validation import PurgedKFold, cv_score
from .importance import feature_importance_mdi, feature_importance_mda, feature_importance_sfi
from .distribution import (
    adf_test,
    jb_normality_test,
    relative_entropy,
    range_iqr_ratio,
    indicator_design_eval,
    indicator_distribution_study,
    parallel_indicator_distribution_study,
)
from .threshold import optimize_threshold, threshold_search
```

**Step 2: Verify imports work**

```bash
uv run python -c "from jarjarquant.evaluation import cv_score, feature_importance_mdi, adf_test, optimize_threshold; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add jarjarquant/evaluation/__init__.py
git commit -m "feat: wire up evaluation package re-exports"
```

---

## Phase 5: Storage Layer

> Replace DataService with repository protocol + DuckDB implementation.

### Task 5.1: Create storage/protocol.py

**Files:**
- Create: `jarjarquant/storage/protocol.py`
- Test: `tests/test_storage_protocol.py`

**Step 1: Write the failing test**

```python
# tests/test_storage_protocol.py
import polars as pl
import pytest
from jarjarquant.storage.protocol import DataRepository
from jarjarquant.schemas import BarSize, SampleRequest, Sample


class MockRepository:
    """Verify that any class with the right methods satisfies the protocol."""

    def get_prices(self, ticker: str, start_date: str, end_date: str,
                   bar_size: BarSize = BarSize.ONE_DAY) -> pl.DataFrame:
        return pl.DataFrame()

    def get_sample(self, request: SampleRequest) -> Sample:
        return Sample(start_date="", end_date="", data=pl.DataFrame(), bar_size=BarSize.ONE_DAY)

    def list_tickers(self) -> list[str]:
        return []

    def save(self, table_name: str, data: pl.DataFrame) -> None:
        pass

    def load(self, table_name: str) -> pl.DataFrame:
        return pl.DataFrame()


def test_mock_satisfies_protocol():
    repo: DataRepository = MockRepository()
    assert repo.list_tickers() == []
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_storage_protocol.py -v
```

**Step 3: Write protocol.py**

```python
from typing import Protocol
import polars as pl
from jarjarquant.schemas import BarSize, SampleRequest, Sample


class DataRepository(Protocol):
    def get_prices(self, ticker: str, start_date: str, end_date: str,
                   bar_size: BarSize = BarSize.ONE_DAY) -> pl.DataFrame: ...
    def get_sample(self, request: SampleRequest) -> Sample: ...
    def list_tickers(self) -> list[str]: ...
    def save(self, table_name: str, data: pl.DataFrame) -> None: ...
    def load(self, table_name: str) -> pl.DataFrame: ...
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_storage_protocol.py -v
```

**Step 5: Commit**

```bash
git add jarjarquant/storage/protocol.py tests/test_storage_protocol.py
git commit -m "feat: add DataRepository protocol in storage/"
```

---

### Task 5.2: Create storage/duckdb.py

**Files:**
- Create: `jarjarquant/storage/duckdb.py`
- Source from: `jarjarquant/data_service.py` (885 lines — extract core query logic, simplify)
- Test: `tests/test_storage_duckdb.py`

**Step 1: Write the failing test**

```python
# tests/test_storage_duckdb.py
import os
import tempfile
import polars as pl
import pytest
from jarjarquant.storage.duckdb import DuckDBRepository
from jarjarquant.schemas import BarSize


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a minimal Parquet file structure for testing."""
    prices_dir = tmp_path / "prices" / "equities" / "1d"
    prices_dir.mkdir(parents=True)
    df = pl.DataFrame({
        "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), eager=True),
        "Open": [100.0] * 366,
        "High": [105.0] * 366,
        "Low": [95.0] * 366,
        "Close": [102.0] * 366,
        "Volume": [1000.0] * 366,
    })
    df.write_parquet(str(prices_dir / "TEST.parquet"))
    return tmp_path


def test_duckdb_repo_list_tickers(tmp_data_dir):
    repo = DuckDBRepository(str(tmp_data_dir))
    tickers = repo.list_tickers()
    assert "TEST" in tickers


def test_duckdb_repo_get_prices(tmp_data_dir):
    repo = DuckDBRepository(str(tmp_data_dir))
    df = repo.get_prices("TEST", "2020-01-01", "2020-06-30")
    assert isinstance(df, pl.DataFrame)
    assert len(df) > 0
    assert "Close" in df.columns


def test_duckdb_repo_save_and_load(tmp_data_dir):
    repo = DuckDBRepository(str(tmp_data_dir))
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    repo.save("test_table", df)
    loaded = repo.load("test_table")
    assert loaded.shape == (3, 2)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_storage_duckdb.py -v
```

**Step 3: Write duckdb.py**

Refactor from `data_service.py`. Keep:
- DuckDB connection management (context manager)
- `get_prices` — Parquet file querying with date filtering
- `list_tickers` — discover tickers from file structure (simplified from `list_available_tickers`)
- `save` / `load` — DuckDB table persistence
- `get_sample` — simplified random sampling (no EquityParams/ForexParams filtering)

Remove:
- All `EquityParams`, `ForexParams`, `BaseParams` related code
- `get_metadata`, `get_sample_by_criteria`, `get_sectors`, `get_analyst_ratings` — these are equity-specific, not needed in BYOD
- `_register_data_views` — metadata CSV registration
- `get_latest_prices` — niche method

Import `BarSize`, `SampleRequest`, `Sample` from `jarjarquant.schemas` instead of defining them locally.

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_storage_duckdb.py -v
```

**Step 5: Commit**

```bash
git add jarjarquant/storage/duckdb.py tests/test_storage_duckdb.py
git commit -m "feat: add DuckDBRepository implementing DataRepository protocol"
```

---

### Task 5.3: Wire up storage/__init__.py

**Files:**
- Modify: `jarjarquant/storage/__init__.py`

```python
from .protocol import DataRepository
from .duckdb import DuckDBRepository
```

**Commit:**

```bash
git add jarjarquant/storage/__init__.py
git commit -m "feat: wire up storage package re-exports"
```

---

## Phase 6: Cleanup — Dependencies, Public API, Tests

> Remove old code paths, update pyproject.toml, rewrite __init__.py, update tests.

### Task 6.1: Archive remaining old modules

**Files:**
- Move: `jarjarquant/feature_engineer.py` → `jarjarquant/_archive/feature_engineer.py`
- Move: `jarjarquant/feature_evaluator.py` → `jarjarquant/_archive/feature_evaluator.py`
- Move: `jarjarquant/data_service.py` → `jarjarquant/_archive/data_service.py`
- Move: `jarjarquant/labeller.py` → `jarjarquant/_archive/labeller.py`
- Remove: `jarjarquant/core/` directory (schemas moved to schemas.py, volatility moved to volatility.py, utils inlined)

**Step 1: Move files**

```bash
cd jarjarquant
git mv feature_engineer.py _archive/feature_engineer.py
git mv feature_evaluator.py _archive/feature_evaluator.py
git mv data_service.py _archive/data_service.py
git mv labeller.py _archive/labeller.py
git rm -r core/
```

**Step 2: Verify no import breaks**

```bash
uv run python -c "import jarjarquant"
```

This will likely fail because `__init__.py` still imports old paths. That's expected — Task 6.2 fixes it.

**Step 3: Commit**

```bash
git add -A
git commit -m "refactor: archive old modules (feature_engineer, feature_evaluator, data_service, labeller, core/)"
```

---

### Task 6.2: Rewrite __init__.py — new public API

**Files:**
- Modify: `jarjarquant/__init__.py`

**Step 1: Write the new __init__.py**

```python
"""Jarjarquant — composable financial ML toolkit."""

# Schemas and types
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

# Indicators
from jarjarquant.indicators import (
    Indicator,
    IndicatorSpec,
    IndicatorType,
    list_available_indicators,
    get_indicator_parameters,
)

# Transforms
from jarjarquant.transforms import (
    log_transform,
    sigmoid_transform,
    root_transform,
    exp_smoothing,
    apply_transform,
)

# Fractional differentiation
from jarjarquant.fractional_diff import frac_diff, frac_diff_ffd

# Volatility
from jarjarquant.volatility import calculate_volatility

# Labelling
from jarjarquant.labelling import (
    triple_barrier_labels,
    event_sampling,
    inverse_cumsum_filter,
    get_sample_weights,
    one_period_with_sl,
    n_period_with_sl,
)

# Evaluation
from jarjarquant.evaluation import (
    PurgedKFold,
    cv_score,
    feature_importance_mdi,
    feature_importance_mda,
    adf_test,
    indicator_distribution_study,
    optimize_threshold,
)

# Storage
from jarjarquant.storage import DataRepository, DuckDBRepository
```

**Step 2: Verify import works**

```bash
uv run python -c "import jarjarquant; print(dir(jarjarquant))"
```

Expected: All exported names listed.

**Step 3: Commit**

```bash
git add jarjarquant/__init__.py
git commit -m "refactor: rewrite public API surface for v1.0"
```

---

### Task 6.3: Update pyproject.toml — dependency cleanup

**Files:**
- Modify: `pyproject.toml`

**Step 1: Replace dependencies section**

```toml
dependencies = [
    "polars>=1.31.0",
    "numpy>=2.1.1",
    "numba>=0.61.2",
    "scipy>=1.14.1",
    "scikit-learn>=1.5.2",
    "statsmodels>=0.14.4",
    "duckdb>=1.3.2",
    "pyarrow>=20.0.0",
    "python-dotenv>=1.1.1",
    "tqdm>=4.67.1",
    "joblib>=1.4.2",
]

[dependency-groups]
dev = [
    "pytest>=8.3.4",
    "pytest-asyncio>=0.25.3",
    "ruff>=0.12.0",
    "cython>=3.0.12",
    "setuptools>=75.1.0",
    "wheel",
]
```

**Step 2: Update setup.py Cython extension paths**

Ensure all 3 extensions reference `jarjarquant/_cython/` not `jarjarquant/cython_utils/`.

**Step 3: Sync environment**

```bash
uv sync
```

**Step 4: Verify build**

```bash
python setup.py build_ext --inplace
uv run python -c "import jarjarquant; print('OK')"
```

**Step 5: Commit**

```bash
git add pyproject.toml setup.py
git commit -m "refactor: clean up dependencies (87 → 11 runtime deps)"
```

---

### Task 6.4: Update and consolidate tests

**Files:**
- Remove: `tests/test_jarjarquant.py` (orchestrator tests — class removed)
- Remove: `tests/test_data_gatherer.py` (data sources archived)
- Remove: `tests/test_data_service.py` (replaced by test_storage_duckdb.py)
- Remove: `tests/test_synthetic.py` (synthetic data source archived)
- Update: `tests/test_feature_engineer.py` → update imports to new modules
- Update: `tests/test_labeller.py` → update imports to new labelling.py
- Update: `tests/test_data_analyst.py` → update imports to evaluation/distribution.py
- Keep: All new test files from earlier tasks

**Step 1: Remove obsolete test files**

```bash
git rm tests/test_jarjarquant.py tests/test_data_gatherer.py tests/test_data_service.py tests/test_synthetic.py
```

**Step 2: Update remaining test imports**

- `test_feature_engineer.py`: Change `from jarjarquant.feature_engineer import FeatureEngineer` → import from `jarjarquant.transforms`, `jarjarquant.fractional_diff`, `jarjarquant.permutation`
- `test_labeller.py`: Change `from jarjarquant.labeller import Labeller` → `from jarjarquant.labelling import ...`
- `test_data_analyst.py`: Change imports to `from jarjarquant.evaluation.distribution import ...`

**Step 3: Run full test suite**

```bash
uv run pytest tests/ -v --maxfail=5
```

Expected: All tests PASS.

**Step 4: Commit**

```bash
git add -A
git commit -m "refactor: update test suite for new module structure"
```

---

### Task 6.5: Update config.py — remove data source config

**Files:**
- Modify: `jarjarquant/config.py`

**Step 1: Simplify config.py**

Remove `EODHD_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `DATA_SOURCE_CONFIG`. Keep only `LOCAL_DB_PATH`.

```python
import os
from dotenv import load_dotenv

load_dotenv()

LOCAL_DB_PATH = os.getenv("LOCAL_DB_PATH", "db/sample_data")
```

**Step 2: Commit**

```bash
git add jarjarquant/config.py
git commit -m "refactor: simplify config.py, remove data source API keys"
```

---

### Task 6.6: Final verification — full test + lint + build

**Step 1: Run linter**

```bash
uv run ruff check jarjarquant/
uv run ruff format jarjarquant/
```

Fix any issues.

**Step 2: Build Cython extensions**

```bash
python setup.py build_ext --inplace
```

**Step 3: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: All tests PASS.

**Step 4: Verify clean import**

```bash
uv run python -c "
from jarjarquant import (
    VolatilityMeasure, BarSize, SampleRequest, Sample,
    IndicatorSpec, IndicatorType, list_available_indicators,
    log_transform, sigmoid_transform, root_transform,
    frac_diff, frac_diff_ffd,
    calculate_volatility,
    triple_barrier_labels, event_sampling, inverse_cumsum_filter,
    get_sample_weights, one_period_with_sl, n_period_with_sl,
    PurgedKFold, cv_score, feature_importance_mdi,
    adf_test, optimize_threshold,
    DataRepository, DuckDBRepository,
)
print('All imports successful')
"
```

**Step 5: Commit any final fixes**

```bash
git add -A
git commit -m "chore: final cleanup — lint fixes, verified all imports and tests pass"
```

---

## Phase Summary

| Phase | Tasks | What Changes |
|-------|-------|-------------|
| 1: Foundation | 1.1–1.6 | Archive old code, create schemas/transforms/fractional_diff/permutation/volatility |
| 2: Indicators | 2.1–2.3 | Refactor base class, update all 16 indicators to _compute() pattern |
| 3: Labelling | 3.1 | Convert Labeller class to pure functions, remove pandas |
| 4: Evaluation | 4.1–4.5 | Decompose 1354-line FeatureEvaluator into 4 modules |
| 5: Storage | 5.1–5.3 | DataRepository protocol + DuckDB implementation |
| 6: Cleanup | 6.1–6.6 | Remove old code, update deps/tests/config, final verification |

**Total tasks:** 20
**Estimated commits:** ~20

## Risk Mitigation

- **Phase 1 is additive** — new modules alongside old ones. Nothing breaks until Phase 6 removes old imports.
- **Phase 6 is the big bang** — archiving old modules and rewriting __init__.py happens together. Run full test suite immediately after.
- **Cython rename (cython_utils → _cython)** happens in Task 1.1. Verify build immediately.
- **Each task has a test** — if a test fails, the issue is scoped to that task.
