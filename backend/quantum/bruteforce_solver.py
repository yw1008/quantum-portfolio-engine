"""Brute-force solver for small QUBO problems."""

from __future__ import annotations

from itertools import product

import pandas as pd


def evaluate_qubo(
    selection: pd.Series,
    Q: pd.DataFrame,
) -> float:
    """Evaluate a binary selection against a QUBO matrix.

    Args:
        selection: Binary selection indexed by ticker.
        Q: Square QUBO matrix indexed and columned by ticker.

    Returns:
        Scalar objective value computed as ``x.T @ Q @ x``.

    Raises:
        ValueError: If ``selection`` is not binary, labels do not match, or
            ``Q`` is empty or not square.
    """
    _validate_qubo_matrix(Q)
    _validate_selection(selection)

    if set(selection.index) != set(Q.index):
        raise ValueError("Tickers in selection and Q must match.")

    aligned_selection = selection.loc[Q.index].astype(float)
    return float(aligned_selection.T @ Q @ aligned_selection)


def solve_qubo_bruteforce(Q: pd.DataFrame) -> tuple[pd.Series, float]:
    """Solve a QUBO exactly by enumerating every binary selection.

    Args:
        Q: Square QUBO matrix indexed and columned by ticker.

    Returns:
        Tuple containing the best binary selection and its objective value.

    Raises:
        ValueError: If ``Q`` is empty or not square.
    """
    _validate_qubo_matrix(Q)

    best_selection: pd.Series | None = None
    best_objective: float | None = None

    for values in product([0, 1], repeat=len(Q.index)):
        selection = pd.Series(values, index=Q.index)
        objective = evaluate_qubo(selection, Q)

        if best_objective is None or objective < best_objective:
            best_selection = selection
            best_objective = objective

    if best_selection is None or best_objective is None:
        raise ValueError("Q must contain at least one binary variable.")

    return best_selection, float(best_objective)


def _validate_qubo_matrix(Q: pd.DataFrame) -> None:
    """Validate that a QUBO matrix is non-empty and square."""
    if Q.empty:
        raise ValueError("Q must not be empty.")
    if len(Q.index) != len(Q.columns):
        raise ValueError("Q must be square.")
    if list(Q.index) != list(Q.columns):
        raise ValueError("Q must have matching row and column labels.")


def _validate_selection(selection: pd.Series) -> None:
    """Validate that a selection Series is binary and non-empty."""
    if selection.empty:
        raise ValueError("selection must not be empty.")

    invalid_values = ~selection.isin([0, 1])
    if invalid_values.any():
        raise ValueError("selection values must be only 0 or 1.")
