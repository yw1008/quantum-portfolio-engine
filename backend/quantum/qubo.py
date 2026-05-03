"""QUBO compiler for binary portfolio selection."""

from __future__ import annotations

import pandas as pd


def build_portfolio_qubo(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    num_assets_to_select: int,
    risk_aversion: float = 1.0,
    penalty_strength: float = 10.0,
) -> pd.DataFrame:
    """Build a QUBO matrix for equal-weight binary portfolio selection.

    The returned matrix represents the minimization objective ``x.T @ Q @ x``:

    ``risk term - return term + penalty_strength * (sum(x) - K)^2``

    The constant part of the penalty expansion is omitted because it does not
    change which binary selection minimizes the objective.

    Args:
        expected_returns: Expected return for each ticker.
        covariance_matrix: Covariance matrix indexed and columned by ticker.
        num_assets_to_select: Exact number of assets to select.
        risk_aversion: Positive multiplier controlling the risk penalty.
        penalty_strength: Positive multiplier enforcing the cardinality target.

    Returns:
        Symmetric QUBO matrix indexed and columned by ticker.

    Raises:
        ValueError: If inputs are empty, labels do not match, the requested
            number of selected assets is invalid, or multipliers are not positive.
    """
    _validate_inputs(
        expected_returns,
        covariance_matrix,
        num_assets_to_select,
        risk_aversion,
        penalty_strength,
    )

    tickers = expected_returns.index
    selected_count = float(num_assets_to_select)
    aligned_covariance = covariance_matrix.loc[tickers, tickers]

    risk_scale = risk_aversion / (selected_count**2)
    return_scale = 1.0 / selected_count
    qubo = risk_scale * aligned_covariance.astype(float)

    for ticker in tickers:
        qubo.loc[ticker, ticker] += (
            penalty_strength * (1 - 2 * num_assets_to_select)
            - return_scale * float(expected_returns.loc[ticker])
        )

    for row_position, row_ticker in enumerate(tickers):
        for column_ticker in tickers[row_position + 1 :]:
            qubo.loc[row_ticker, column_ticker] += penalty_strength
            qubo.loc[column_ticker, row_ticker] += penalty_strength

    return qubo


def _validate_inputs(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    num_assets_to_select: int,
    risk_aversion: float,
    penalty_strength: float,
) -> None:
    """Validate QUBO compiler inputs."""
    if expected_returns.empty:
        raise ValueError("expected_returns must not be empty.")
    if covariance_matrix.empty:
        raise ValueError("covariance_matrix must not be empty.")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive.")
    if penalty_strength <= 0:
        raise ValueError("penalty_strength must be positive.")

    asset_count = len(expected_returns)
    if num_assets_to_select < 1 or num_assets_to_select > asset_count:
        raise ValueError(
            "num_assets_to_select must be between 1 and the number of assets."
        )

    if list(covariance_matrix.index) != list(covariance_matrix.columns):
        raise ValueError("covariance_matrix must have matching row and column labels.")
    if set(expected_returns.index) != set(covariance_matrix.index):
        raise ValueError(
            "Tickers in expected_returns and covariance_matrix must match."
        )
