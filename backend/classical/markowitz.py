"""Classical Markowitz mean-variance portfolio optimization."""

from __future__ import annotations

import cvxpy as cp
import pandas as pd


def optimize_portfolio(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_aversion: float = 1.0,
) -> pd.Series:
    """Optimize long-only portfolio weights using mean-variance optimization.

    The objective maximizes expected portfolio return minus a covariance-based
    risk penalty:

    ``expected_returns @ weights - risk_aversion * weights.T @ covariance @ weights``

    Args:
        expected_returns: Expected return for each ticker.
        covariance_matrix: Covariance matrix indexed and columned by ticker.
        risk_aversion: Positive multiplier controlling the penalty for risk.

    Returns:
        Optimized portfolio weights as a Series indexed by ticker.

    Raises:
        ValueError: If inputs are empty, labels do not match, risk aversion is
            not positive, or the optimization problem cannot be solved.
    """
    _validate_inputs(expected_returns, covariance_matrix, risk_aversion)

    tickers = expected_returns.index
    returns = expected_returns.to_numpy(dtype=float)
    covariance = covariance_matrix.loc[tickers, tickers].to_numpy(dtype=float)

    weights = cp.Variable(len(tickers))
    expected_return = returns @ weights
    risk_penalty = cp.quad_form(weights, covariance)
    objective = cp.Maximize(expected_return - risk_aversion * risk_penalty)
    constraints = [weights >= 0, cp.sum(weights) == 1]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
        raise ValueError(f"Portfolio optimization failed with status: {problem.status}.")

    optimized_weights = pd.Series(weights.value, index=tickers, name="weight")
    optimized_weights[optimized_weights.abs() < 1e-12] = 0.0

    return optimized_weights


def _validate_inputs(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_aversion: float,
) -> None:
    """Validate optimizer inputs."""
    if expected_returns.empty:
        raise ValueError("expected_returns must not be empty.")
    if covariance_matrix.empty:
        raise ValueError("covariance_matrix must not be empty.")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive.")

    expected_tickers = list(expected_returns.index)
    covariance_index = list(covariance_matrix.index)
    covariance_columns = list(covariance_matrix.columns)

    if covariance_index != covariance_columns:
        raise ValueError("covariance_matrix must have matching row and column labels.")
    if set(expected_tickers) != set(covariance_index):
        raise ValueError(
            "Tickers in expected_returns and covariance_matrix must match."
        )
