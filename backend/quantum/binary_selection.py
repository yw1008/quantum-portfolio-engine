"""Binary portfolio selection model."""

from __future__ import annotations

import pandas as pd


def normalize_binary_selection(selection: pd.Series) -> pd.Series:
    """Convert a binary asset selection into equal portfolio weights.

    Args:
        selection: Binary selection indexed by ticker, where 1 means selected
            and 0 means excluded.

    Returns:
        Equal weights across selected assets and zero for unselected assets.

    Raises:
        ValueError: If ``selection`` is empty, contains values other than 0 or 1,
            or selects no assets.
    """
    _validate_selection(selection)

    selected_count = int(selection.sum())
    if selected_count == 0:
        raise ValueError("selection must include at least one selected asset.")

    weights = selection.astype(float) / selected_count
    weights.name = "weight"

    return weights


def compute_binary_portfolio_score(
    selection: pd.Series,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_aversion: float = 1.0,
) -> float:
    """Compute the mean-variance score for a binary asset selection.

    Selected assets are normalized to equal weights before scoring:

    ``expected_return - risk_aversion * portfolio_variance``

    Args:
        selection: Binary selection indexed by ticker.
        expected_returns: Expected return for each ticker.
        covariance_matrix: Covariance matrix indexed and columned by ticker.
        risk_aversion: Positive multiplier controlling the penalty for risk.

    Returns:
        Portfolio score for the selected asset set.

    Raises:
        ValueError: If inputs are empty, labels do not match, selection is not
            binary, no assets are selected, or risk aversion is not positive.
    """
    _validate_inputs(selection, expected_returns, covariance_matrix, risk_aversion)

    weights = normalize_binary_selection(selection)
    aligned_returns = expected_returns.loc[weights.index]
    aligned_covariance = covariance_matrix.loc[weights.index, weights.index]

    expected_return = float(weights.dot(aligned_returns))
    portfolio_variance = float(weights.T @ aligned_covariance @ weights)

    return expected_return - risk_aversion * portfolio_variance


def _validate_inputs(
    selection: pd.Series,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_aversion: float,
) -> None:
    """Validate binary portfolio score inputs."""
    _validate_selection(selection)
    _validate_series(expected_returns, "expected_returns")

    if covariance_matrix.empty:
        raise ValueError("covariance_matrix must not be empty.")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive.")

    if list(covariance_matrix.index) != list(covariance_matrix.columns):
        raise ValueError("covariance_matrix must have matching row and column labels.")
    if set(selection.index) != set(expected_returns.index):
        raise ValueError("Tickers in selection and expected_returns must match.")
    if set(selection.index) != set(covariance_matrix.index):
        raise ValueError("Tickers in selection and covariance_matrix must match.")


def _validate_selection(selection: pd.Series) -> None:
    """Validate that a selection Series is non-empty and binary."""
    _validate_series(selection, "selection")

    invalid_values = ~selection.isin([0, 1])
    if invalid_values.any():
        raise ValueError("selection values must be only 0 or 1.")


def _validate_series(data: pd.Series, name: str) -> None:
    """Validate that a Series contains data."""
    if data.empty:
        raise ValueError(f"{name} must not be empty.")
