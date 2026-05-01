"""Portfolio performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_portfolio_return(
    weights: pd.Series,
    expected_returns: pd.Series,
) -> float:
    """Compute expected portfolio return from weights and asset returns.

    Args:
        weights: Portfolio weights indexed by ticker.
        expected_returns: Expected returns indexed by ticker.

    Returns:
        The weighted expected portfolio return.

    Raises:
        ValueError: If inputs are empty or ticker labels do not match.
    """
    _validate_series(weights, "weights")
    _validate_series(expected_returns, "expected_returns")
    _validate_matching_tickers(weights.index, expected_returns.index)

    aligned_returns = expected_returns.loc[weights.index]
    return float(weights.dot(aligned_returns))


def compute_portfolio_volatility(
    weights: pd.Series,
    covariance_matrix: pd.DataFrame,
) -> float:
    """Compute portfolio volatility from weights and a covariance matrix.

    Args:
        weights: Portfolio weights indexed by ticker.
        covariance_matrix: Covariance matrix indexed and columned by ticker.

    Returns:
        Portfolio volatility, computed as the square root of portfolio variance.

    Raises:
        ValueError: If inputs are empty or ticker labels do not match.
    """
    _validate_series(weights, "weights")
    if covariance_matrix.empty:
        raise ValueError("covariance_matrix must not be empty.")
    if len(covariance_matrix.columns) == 0:
        raise ValueError("covariance_matrix must contain at least one asset column.")
    if list(covariance_matrix.index) != list(covariance_matrix.columns):
        raise ValueError("covariance_matrix must have matching row and column labels.")
    _validate_matching_tickers(weights.index, covariance_matrix.index)

    aligned_covariance = covariance_matrix.loc[weights.index, weights.index]
    portfolio_variance = float(weights.T @ aligned_covariance @ weights)

    return float(np.sqrt(portfolio_variance))


def compute_sharpe_ratio(
    portfolio_return: float,
    portfolio_volatility: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute the Sharpe ratio for a portfolio.

    Args:
        portfolio_return: Expected or realized portfolio return.
        portfolio_volatility: Portfolio volatility over the same period.
        risk_free_rate: Risk-free return over the same period.

    Returns:
        The Sharpe ratio, defined as excess return divided by volatility.

    Raises:
        ValueError: If ``portfolio_volatility`` is not positive.
    """
    if portfolio_volatility <= 0:
        raise ValueError("portfolio_volatility must be positive.")

    return float((portfolio_return - risk_free_rate) / portfolio_volatility)


def _validate_series(data: pd.Series, name: str) -> None:
    """Validate that a Series contains data."""
    if data.empty:
        raise ValueError(f"{name} must not be empty.")


def _validate_matching_tickers(
    left_tickers: pd.Index,
    right_tickers: pd.Index,
) -> None:
    """Validate that two ticker indexes contain the same labels."""
    if set(left_tickers) != set(right_tickers):
        raise ValueError("Ticker labels must match.")
