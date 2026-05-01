# Quantum Portfolio Optimization Engine

Quantum Portfolio Optimization Engine is a Python research project for exploring
portfolio construction methods that combine classical optimization baselines with
future quantum optimization experiments.

The project is currently in its initial scaffold phase. No research logic,
optimization routines, or data pipelines have been implemented yet.

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
