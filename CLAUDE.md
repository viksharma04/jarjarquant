# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Build
```bash
# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies and sync environment
uv sync

# Install in development mode
uv pip install -e .
```

### Testing
```bash
# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Run all tests
uv run pytest tests/

# Run tests with CI configuration
uv run pytest tests/ --maxfail=3 --disable-warnings

# Run specific test file
uv run pytest tests/test_specific_module.py
```

### Linting and Code Quality
```bash
# Activate virtual environment (Windows)
.\venv\Scripts\activate

# Run linting (ruff is configured as dev dependency)
uv run ruff check jarjarquant/

# Run formatting
uv run ruff format jarjarquant/
```

## High-Level Architecture

### Core Design Pattern
Jarjarquant follows a **composition-based architecture** where the main `Jarjarquant` class orchestrates specialized components:

```python
Jarjarquant (inherits from Labeller)
├── DataGatherer (registry pattern for data sources)
├── FeatureEngineer (fractional differentiation, transformations)  
├── FeatureEvaluator (cross-validation, feature importance)
└── DataAnalyst (statistical tests, visualization)
```

### Data Source Architecture
Uses a **registry pattern** with the `@register_data_source` decorator:
- Base `DataSource` class with async `fetch` method
- Sources: Yahoo Finance, Interactive Brokers, EODHD, Alpha Vantage, Custom (DuckDB/Parquet)
- Configuration via environment variables (EODHD_API_KEY, ALPHA_VANTAGE_API_KEY)

### Data Flow Pipeline
```
Raw Data Sources → DataGatherer → Feature Engineering → 
Triple-Barrier Labeling → Feature Evaluation → Statistical Analysis
```

### Performance-Critical Components
Three Cython extensions handle computational bottlenecks:
- `bar_permute.pyx`: Permutation operations for cross-validation
- `indicators.pyx`: Fast technical indicator calculations
- `opt_threshold.pyx`: Optimization routines for thresholds

### Indicator System
Modular indicator architecture with base `Indicator` class:
- 15+ indicators (RSI, MACD, ADX, Aroon, Stochastic, etc.)
- Built-in evaluation: ADF stationarity test, Jarque-Bera normality test, entropy analysis
- Consistent interface across all indicators

## Key Development Notes

### Python Version Requirement
Project requires Python >=3.12. The CI uses Python 3.12 specifically.

### Testing Strategy
- pytest-based testing framework
- Some tests in `test_jarjarquant.py` are currently commented out
- Tests include mocking for external data dependencies
- CI runs on Ubuntu with system dependencies: build-essential, python3-dev

### Package Manager
Uses `uv` as the modern Python package manager. All commands should be prefixed with `uv run` for proper environment isolation. **Important**: Always activate the virtual environment with `.\venv\Scripts\activate` before running any uv commands on Windows.

### Sample Data Structure
Large sample dataset organized as:
```
sample_data/
├── equities/daily/1d/{ticker}.parquet
├── metadata/{source}_tickers.csv
└── alpha_vantage/{ticker}.json
```

### Configuration
API keys and settings managed through:
- `config.py` for centralized configuration
- Environment variables for sensitive data
- No hardcoded credentials in codebase

## Development Principles

### Code Quality Standards
- **SOLID Principles**: Always follow SOLID principles when adding to the codebase, refactoring existing code when necessary
- **Type Hints**: Use comprehensive type annotations for all functions, methods, and class attributes
- **Docstrings**: Follow Google-style docstrings for all public classes, methods, and functions
- **Error Handling**: Use specific exception types and provide meaningful error messages
- **Immutability**: Prefer immutable data structures where possible; avoid side effects in pure functions
- **Dependency Injection**: Use dependency injection for external services and data sources
- **Interface Segregation**: Keep interfaces focused and cohesive; avoid large, monolithic classes

### Design Patterns
- **Factory Pattern**: Use for creating data source instances and indicators
- **Registry Pattern**: Continue using for extensible component registration

### Performance Considerations
- **Lazy Loading**: Load data and compute features only when needed
- **Caching**: Cache expensive computations and data fetches appropriately
- **Vectorization**: Prefer vectorized operations over loops
- **Memory Management**: Be mindful of memory usage with large datasets
- **Low level code**: Use Cython code to leverage C for extensive loops that need to be called frequently for example, indicator calculations

### Testing Excellence
- **Test Coverage**: Maintain >80% test coverage for new code
- **Test Isolation**: Each test should be independent and deterministic
- **Mock External Dependencies**: Mock all external APIs and data sources
- **Property-Based Testing**: Consider using hypothesis for complex financial calculations
- **Integration Tests**: Include tests that verify component interactions

### Code Organization
- **Single Responsibility**: Each class/function should have one clear purpose
- **DRY Principle**: Don't repeat yourself; extract common functionality
- **Composition Over Inheritance**: Favor composition for flexibility
- **Clear Naming**: Use descriptive names that express intent, not implementation
- **Package Structure**: Organize code logically by feature/domain, not by technical layer