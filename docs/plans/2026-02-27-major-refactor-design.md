# Jarjarquant Major Refactor Design

**Date:** 2026-02-27
**Status:** Approved
**Scope:** Complete architectural refactor of jarjarquant

---

## Goals

1. Clean separation of concerns across modules
2. Minimal bloat — functions over classes, no unnecessary abstractions
3. Self-explanatory code and usage — composable, explicit API
4. BYOD (Bring Your Own Data) — no built-in data fetching
5. Polars-only DataFrame standard throughout
6. Aggressive dependency cleanup (87 → ~12 runtime deps)

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| DataFrame library | Polars only | Already dominant in newer code, faster, better API |
| Orchestrator class | Remove entirely | Users compose components directly |
| FeatureEngineer | Decompose into pure function modules | God class with 4 unrelated concerns |
| Backward compatibility | Clean break (1.0) | No shims, no deprecation warnings |
| Performance strategy | Keep both Numba + Cython | Each fits different use cases |
| Data access | Repository protocol + DuckDB impl | Swappable backends, hidden storage details |
| Visualization | Remove from core | Return data, users plot themselves |
| Data sources | Archive entirely (BYOD) | Users bring their own data |
| Indicator transforms | Baked into base class | Part of indicator identity for storage/comparison |
| Evaluation caching | Remove built-in persistence | Users persist their own results |

## Package Structure

```
jarjarquant/
├── __init__.py                 # Public API surface
├── schemas.py                  # All dataclasses, enums, type definitions
├── transforms.py               # Pure functions: log, sigmoid, root, exp_smoothing
├── fractional_diff.py          # Pure functions: frac_diff, frac_diff_ffd, get_weights
├── volatility.py               # Numba-optimized volatility calculations
├── labelling.py                # Pure functions: triple barrier, sample weights, event sampling
├── permutation.py              # BarPermute, PricePermute classes (stateful: basis prices)
│
├── indicators/                 # Registry + implementations
│   ├── __init__.py
│   ├── base.py                 # Indicator ABC with transform support, IndicatorSpec
│   ├── registry.py             # IndicatorType enum, registry, decorator
│   └── *.py                    # Individual indicator implementations
│
├── evaluation/                 # Decomposed FeatureEvaluator
│   ├── __init__.py             # Re-exports key functions
│   ├── cross_validation.py     # PurgedKFold, cv_score
│   ├── importance.py           # MDI, MDA, SFI feature importance
│   ├── distribution.py         # Statistical tests, indicator distribution study
│   └── threshold.py            # Threshold search and optimization
│
├── storage/                    # Repository pattern for data access
│   ├── __init__.py
│   ├── protocol.py             # DataRepository Protocol definition
│   └── duckdb.py               # DuckDB/Parquet implementation
│
├── _cython/                    # Renamed from cython_utils/
│   ├── __init__.py
│   ├── bar_permute.pyx
│   ├── indicators.pyx
│   └── opt_threshold.pyx
│
└── _archive/                   # Preserved but excluded from package
    ├── data_gatherer/          # All data source code
    ├── jarjarquant.py          # Old orchestrator
    ├── data_analyst.py         # Viz functions removed, stat functions → evaluation/
    ├── feature_engineer.py     # Decomposed into transforms/fractional_diff/permutation
    └── data_service.py         # Replaced by storage/
```

## Data Contracts (schemas.py)

```python
from enum import Enum
from dataclasses import dataclass
from typing import Any

class VolatilityMeasure(Enum):
    MAX_RANGE = "max_range"
    ATR = "atr"
    ANNUALIZED_STD_DEV = "annualized_std_dev"
    HIGH_LOW = "high_low"
    HIGH_LOW_CLOSE = "high_low_close"
    YANG_ZHANG = "yang_zhang"
    AVG_LOG_RETURNS = "avg_log_returns"
    EWM_STD = "ewm_std"

class BarSize(Enum):
    ONE_DAY = "1 day"
    ONE_HOUR = "1 hour"
    ONE_MINUTE = "1 min"
    # ... (keep existing values)

@dataclass
class SampleRequest:
    start_date: str
    end_date: str
    bar_size: BarSize = BarSize.ONE_DAY
    n_samples: int = 10

@dataclass
class Sample:
    start_date: str
    end_date: str
    data: pl.DataFrame
    bar_size: BarSize

# Statistical test result dataclasses
@dataclass
class ADFTestResult: ...
@dataclass
class NormalityTestResult: ...
@dataclass
class EntropyResult: ...
@dataclass
class RangeIQRResult: ...
```

## Indicator System

### Base class with built-in transform support

```python
class Indicator:
    def __init__(self, ohlcv_df: pl.DataFrame):
        self._df = ohlcv_df
        self._transform: str | None = None

    def calculate(self) -> np.ndarray:
        raw = self._compute()
        if self._transform is not None:
            return apply_transform(raw, self._transform)
        return raw

    def _compute(self) -> np.ndarray:
        raise NotImplementedError
```

- Subclasses implement `_compute()` for raw indicator logic
- Base class handles transform dispatch via `apply_transform()` from `transforms.py`
- Subclasses can override `calculate()` entirely for custom transform behavior
- No FeatureEngineer instance needed anywhere

### IndicatorSpec (stays in indicators/base.py)

```python
@dataclass
class IndicatorSpec:
    indicator_type: IndicatorType
    parameters: dict[str, Any]  # includes 'transform' if set

    def create_indicator(self, ohlcv_df: pl.DataFrame) -> Indicator: ...
```

- Two specs with different transforms are distinct (different identity)
- Enables storing and comparing indicator configurations

## Storage Protocol

```python
# storage/protocol.py
from typing import Protocol

class DataRepository(Protocol):
    def get_prices(self, ticker: str, start_date: str, end_date: str,
                   bar_size: BarSize = BarSize.ONE_DAY) -> pl.DataFrame: ...
    def get_sample(self, request: SampleRequest) -> Sample: ...
    def list_tickers(self) -> list[str]: ...
    def save(self, table_name: str, data: pl.DataFrame) -> None: ...
    def load(self, table_name: str) -> pl.DataFrame: ...
```

DuckDB implementation in `storage/duckdb.py` refactored from `data_service.py`.

## Labelling

All functions become pure — no instance state:

```python
# labelling.py — all take explicit inputs, return outputs
def inverse_cumsum_filter(close: np.ndarray, threshold: float) -> np.ndarray: ...
def event_sampling(close: np.ndarray, volatility: np.ndarray, method: str) -> np.ndarray: ...
def triple_barrier_labels(prices: np.ndarray, ...) -> pl.DataFrame: ...
def one_period_with_sl(prices: np.ndarray, ...) -> pl.DataFrame: ...
def n_period_with_sl(prices: np.ndarray, ...) -> pl.DataFrame: ...
def get_sample_weights(labels: pl.DataFrame, ...) -> np.ndarray: ...
```

The old pandas `triple_barrier_method()` is removed — numba version supersedes it.

## Dependencies

### Runtime (~12 direct)

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
```

### Dev only

```toml
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

### Removed entirely

- Data source deps: yfinance, httpx, requests, beautifulsoup4, lxml, ib-async
- Jupyter deps: ipython, ipykernel, jupyter-*, ipywidgets
- Visualization: matplotlib, altair
- Legacy: six, pandas, debugpy, peewee
- Build-as-runtime: cython, setuptools, pytest

## Public API Surface

```python
# jarjarquant/__init__.py
from jarjarquant.schemas import VolatilityMeasure, BarSize, SampleRequest, Sample
from jarjarquant.indicators import IndicatorType, IndicatorSpec, list_available_indicators
from jarjarquant.transforms import log_transform, sigmoid_transform, root_transform
from jarjarquant.fractional_diff import frac_diff, frac_diff_ffd
from jarjarquant.volatility import calculate_volatility
from jarjarquant.labelling import (
    triple_barrier_labels, event_sampling, inverse_cumsum_filter,
    get_sample_weights, one_period_with_sl, n_period_with_sl
)
from jarjarquant.evaluation import (
    cv_score, feature_importance_mdi, feature_importance_mda,
    indicator_distribution_study, threshold_search
)
from jarjarquant.storage import DataRepository, DuckDBRepository
```

## Example Usage

```python
from jarjarquant import (
    IndicatorSpec, IndicatorType, triple_barrier_labels,
    calculate_volatility, VolatilityMeasure, DuckDBRepository
)

# BYOD: user provides data via repository
repo = DuckDBRepository("path/to/data")
sample = repo.get_sample(request)

# Compute indicator
spec = IndicatorSpec(IndicatorType.RSI, {"period": 14, "transform": "log"})
indicator = spec.create_indicator(sample.data)
values = indicator.calculate()  # transform applied internally

# Generate labels
vol = calculate_volatility(prices, VolatilityMeasure.ATR, window=20)
labels = triple_barrier_labels(prices, vol, ...)

# Evaluate
from jarjarquant.evaluation import indicator_distribution_study
results = indicator_distribution_study(values)
```

## What Gets Archived

| Current Location | Destination | Reason |
|---|---|---|
| `jarjarquant/jarjarquant.py` | `_archive/` | Orchestrator removed |
| `jarjarquant/data_gatherer/` | `_archive/` | BYOD — no data fetching |
| `jarjarquant/data_analyst.py` | `_archive/` | Stat functions → evaluation/, viz removed |
| `jarjarquant/feature_engineer.py` | `_archive/` | Split into transforms/fractional_diff/permutation |
| `jarjarquant/feature_evaluator.py` | `_archive/` | Split into evaluation/ subpackage |
| `jarjarquant/data_service.py` | `_archive/` | Replaced by storage/ |
| `jarjarquant/core/` | dissolved | volatility → volatility.py, schemas → schemas.py |
