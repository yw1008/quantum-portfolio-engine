"""Efficient frontier generation for classical portfolios."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from backend.analysis.metrics import (
    compute_portfolio_return,
    compute_portfolio_volatility,
    compute_sharpe_ratio,
)


def generate_efficient_frontier(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    num_points: int = 50,
) -> pd.DataFrame:
    """Generate optimal long-only portfolios across target returns.

    Args:
        expected_returns: Expected return for each ticker.
        covariance_matrix: Covariance matrix indexed and columned by ticker.
        num_points: Number of target return points to evaluate.

    Returns:
        A DataFrame with target return, expected return, volatility, Sharpe
        ratio, and optimized weights for each feasible frontier point.

    Raises:
        ValueError: If inputs are empty, tickers do not match, ``num_points`` is
            not positive, or no feasible frontier points can be generated.
    """
    _validate_inputs(expected_returns, covariance_matrix, num_points)

    tickers = expected_returns.index
    returns = expected_returns.to_numpy(dtype=float)
    covariance = covariance_matrix.loc[tickers, tickers].to_numpy(dtype=float)
    target_returns = pd.Series(expected_returns).sort_values().pipe(
        lambda sorted_returns: _build_target_returns(sorted_returns, num_points)
    )

    rows: list[dict[str, object]] = []

    for target_return in target_returns:
        weights = _optimize_for_target_return(
            tickers=tickers,
            returns=returns,
            covariance=covariance,
            target_return=float(target_return),
        )
        if weights is None:
            continue

        expected_return = compute_portfolio_return(weights, expected_returns)
        volatility = compute_portfolio_volatility(weights, covariance_matrix)
        sharpe_ratio = compute_sharpe_ratio(expected_return, volatility)

        rows.append(
            {
                "target_return": float(target_return),
                "expected_return": expected_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "weights": weights,
            }
        )

    if not rows:
        raise ValueError("No feasible efficient frontier points could be generated.")

    return pd.DataFrame(
        rows,
        columns=[
            "target_return",
            "expected_return",
            "volatility",
            "sharpe_ratio",
            "weights",
        ],
    )


def _build_target_returns(
    sorted_returns: pd.Series,
    num_points: int,
) -> list[float]:
    """Build evenly spaced target returns from minimum to maximum asset return."""
    min_return = float(sorted_returns.iloc[0])
    max_return = float(sorted_returns.iloc[-1])

    if num_points == 1:
        return [min_return]

    step = (max_return - min_return) / (num_points - 1)
    return [min_return + step * point for point in range(num_points)]


def _optimize_for_target_return(
    tickers: pd.Index,
    returns: np.ndarray,
    covariance: np.ndarray,
    target_return: float,
) -> pd.Series | None:
    """Optimize minimum variance weights for a target return."""
    weights = cp.Variable(len(tickers))
    objective = cp.Minimize(cp.quad_form(weights, covariance))
    constraints = [
        weights >= 0,
        cp.sum(weights) == 1,
        returns @ weights >= target_return,
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or weights.value is None:
        return None

    optimized_weights = pd.Series(weights.value, index=tickers, name="weight")
    optimized_weights[optimized_weights.abs() < 1e-12] = 0.0
    optimized_weights = optimized_weights.clip(lower=0.0)

    weight_sum = optimized_weights.sum()
    if weight_sum <= 0:
        return None

    return optimized_weights / weight_sum


def _validate_inputs(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    num_points: int,
) -> None:
    """Validate efficient frontier inputs."""
    if expected_returns.empty:
        raise ValueError("expected_returns must not be empty.")
    if covariance_matrix.empty:
        raise ValueError("covariance_matrix must not be empty.")
    if num_points <= 0:
        raise ValueError("num_points must be positive.")

    expected_tickers = set(expected_returns.index)
    covariance_index = set(covariance_matrix.index)
    covariance_columns = set(covariance_matrix.columns)

    if covariance_index != covariance_columns:
        raise ValueError("covariance_matrix must have matching row and column labels.")
    if expected_tickers != covariance_index:
        raise ValueError(
            "Tickers in expected_returns and covariance_matrix must match."
        )
