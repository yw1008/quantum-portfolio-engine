# Quantum Portfolio Optimization Engine

Quantum Portfolio Optimization Engine is a Python research project for exploring
portfolio construction methods that combine classical optimization baselines
with future quantum optimization experiments.

## Current Status

The project is in Phase 1: building the research foundation. Current
capabilities include historical market data loading and basic return
calculations.

## Roadmap

- Phase 1: Data loading, return calculations, covariance estimation, and
  classical portfolio optimization foundations.
- Phase 2: Portfolio objective modeling and constraint handling.
- Phase 3: Quantum optimization formulation and solver experiments.
- Phase 4: Comparative analysis, reporting, and research notebooks.

## Tech Stack

- Python
- pandas
- yfinance
- pytest
- Jupyter notebooks

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Documentation

Detailed project notes live in `docs/`.

- Architecture decisions: `docs/architecture.md`
- Development log: `docs/development-log.md`
- Learning notes: `docs/learning/`
