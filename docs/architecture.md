# Architecture

This document records architecture decisions for the Quantum Portfolio
Optimization Engine.

## Current Structure

- `backend/data`: Market data loading and return calculation utilities.
- `backend/classical`: Classical portfolio optimization methods.
- `backend/analysis`: Analysis, metrics, and reporting utilities.
- `notebooks`: Exploratory research notebooks.
- `tests`: Automated tests for project modules.

## Decisions

- Keep data ingestion and return calculations separate from optimization logic.
- Prefer small, typed, documented functions with focused tests.
- Keep README content concise; move deeper explanations and design notes into
  `docs/`.
