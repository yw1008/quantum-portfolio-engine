"""QUBO to Ising conversion utilities."""

from __future__ import annotations

import pandas as pd


def qubo_to_ising(Q: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, float]:
    """Convert a QUBO objective into Ising coefficients.

    Converts ``x.T @ Q @ x`` with ``x_i`` in ``{0, 1}`` into:

    ``sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + offset``

    using ``x_i = (1 - z_i) / 2``.

    Args:
        Q: Square QUBO matrix indexed and columned by ticker.

    Returns:
        Linear coefficients ``h``, pairwise couplings ``J``, and constant offset.

    Raises:
        ValueError: If ``Q`` is empty or not a labeled square matrix.
    """
    _validate_qubo_matrix(Q)

    tickers = Q.index
    h = pd.Series(0.0, index=tickers, name="h")
    J = pd.DataFrame(0.0, index=tickers, columns=tickers)
    offset = 0.0

    for ticker in tickers:
        diagonal_coefficient = float(Q.loc[ticker, ticker])
        h.loc[ticker] -= diagonal_coefficient / 2.0
        offset += diagonal_coefficient / 2.0

    for row_position, row_ticker in enumerate(tickers):
        for column_ticker in tickers[row_position + 1 :]:
            pair_coefficient = float(
                Q.loc[row_ticker, column_ticker] + Q.loc[column_ticker, row_ticker]
            )
            coupling = pair_coefficient / 4.0

            J.loc[row_ticker, column_ticker] = coupling
            J.loc[column_ticker, row_ticker] = coupling
            h.loc[row_ticker] -= coupling
            h.loc[column_ticker] -= coupling
            offset += coupling

    return h, J, float(offset)


def evaluate_ising(
    spins: pd.Series,
    h: pd.Series,
    J: pd.DataFrame,
    offset: float,
) -> float:
    """Evaluate an Ising energy for a spin assignment.

    Args:
        spins: Spin values indexed by ticker, where each value is -1 or +1.
        h: Linear Ising coefficients indexed by ticker.
        J: Pairwise Ising couplings indexed and columned by ticker.
        offset: Constant energy offset.

    Returns:
        Scalar Ising energy.

    Raises:
        ValueError: If spins are not -1/+1 or labels do not match.
    """
    _validate_ising_inputs(spins, h, J)

    aligned_spins = spins.loc[h.index].astype(float)
    energy = float(h.dot(aligned_spins)) + float(offset)

    for row_position, row_ticker in enumerate(h.index):
        for column_ticker in h.index[row_position + 1 :]:
            energy += float(
                J.loc[row_ticker, column_ticker]
                * aligned_spins.loc[row_ticker]
                * aligned_spins.loc[column_ticker]
            )

    return float(energy)


def _validate_qubo_matrix(Q: pd.DataFrame) -> None:
    """Validate that a QUBO matrix is non-empty and square."""
    if Q.empty:
        raise ValueError("Q must not be empty.")
    if len(Q.index) != len(Q.columns):
        raise ValueError("Q must be square.")
    if list(Q.index) != list(Q.columns):
        raise ValueError("Q must have matching row and column labels.")


def _validate_ising_inputs(
    spins: pd.Series,
    h: pd.Series,
    J: pd.DataFrame,
) -> None:
    """Validate Ising energy inputs."""
    if spins.empty:
        raise ValueError("spins must not be empty.")
    if h.empty:
        raise ValueError("h must not be empty.")
    if J.empty:
        raise ValueError("J must not be empty.")

    invalid_spins = ~spins.isin([-1, 1])
    if invalid_spins.any():
        raise ValueError("spins values must be only -1 or +1.")

    if len(J.index) != len(J.columns):
        raise ValueError("J must be square.")
    if list(J.index) != list(J.columns):
        raise ValueError("J must have matching row and column labels.")
    if set(spins.index) != set(h.index):
        raise ValueError("Tickers in spins and h must match.")
    if set(h.index) != set(J.index):
        raise ValueError("Tickers in h and J must match.")
