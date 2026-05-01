# Quantum Portfolio Optimization Engine

Quantum Portfolio Optimization Engine is a Python research project for exploring
portfolio construction methods that combine classical optimization baselines with
future quantum optimization experiments.

The project is currently in its initial research-foundation phase. Optimization
routines and research experiments have not been implemented yet.

## Folder Structure

```text
backend/
  __init__.py
  data/
    __init__.py
  classical/
    __init__.py
  analysis/
    __init__.py
tests/
  __init__.py
notebooks/
```

- `backend/data`: Data loading, preprocessing, and dataset utilities.
- `backend/classical`: Classical optimization baselines and comparison methods.
- `backend/analysis`: Experiment analysis, metrics, and reporting helpers.
- `tests`: Test suite for project modules.
- `notebooks`: Research notebooks and exploratory analysis.

## Phase 1 Goal

Phase 1 focuses on establishing a clean research foundation: project structure,
dependency hygiene, and space for classical portfolio optimization experiments
before introducing quantum optimization components.

### Market Data Loading

Phase 1 includes an initial market data loader at `backend/data/market_data.py`.
It uses yfinance to fetch historical adjusted close prices for one or more
tickers and returns a clean pandas DataFrame indexed by date.

### Return Calculations

Phase 1 also includes return utilities at `backend/data/returns.py` for
computing daily percentage returns from price data and annualized mean returns
from daily returns.
