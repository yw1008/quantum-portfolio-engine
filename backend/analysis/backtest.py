"""Portfolio backtesting utilities."""

from __future__ import annotations

import pandas as pd


def compute_portfolio_returns(
    daily_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Compute realized daily portfolio returns from asset returns and weights.

    Args:
        daily_returns: Daily asset returns with dates as rows and tickers as columns.
        weights: Portfolio weights indexed by ticker.

    Returns:
        Daily realized portfolio returns.

    Raises:
        ValueError: If inputs are empty or ticker labels do not match.
    """
    _validate_daily_returns(daily_returns)
    _validate_series(weights, "weights")
    _validate_matching_tickers(daily_returns.columns, weights.index)

    aligned_weights = weights.loc[daily_returns.columns]
    return daily_returns.dot(aligned_weights)


def compute_cumulative_returns(
    portfolio_returns: pd.Series,
) -> pd.Series:
    """Compute cumulative returns from periodic portfolio returns.

    Args:
        portfolio_returns: Portfolio returns for each period.

    Returns:
        Cumulative portfolio returns over time.

    Raises:
        ValueError: If ``portfolio_returns`` is empty.
    """
    _validate_series(portfolio_returns, "portfolio_returns")

    return (1.0 + portfolio_returns).cumprod() - 1.0


def compute_drawdown(
    cumulative_returns: pd.Series,
) -> pd.Series:
    """Compute drawdown from cumulative returns.

    Args:
        cumulative_returns: Cumulative portfolio returns over time.

    Returns:
        Drawdown series, expressed as negative percentages from prior peaks.

    Raises:
        ValueError: If ``cumulative_returns`` is empty.
    """
    _validate_series(cumulative_returns, "cumulative_returns")

    wealth_index = 1.0 + cumulative_returns
    running_peak = wealth_index.cummax()

    return (wealth_index / running_peak) - 1.0


def compute_max_drawdown(
    drawdown: pd.Series,
) -> float:
    """Compute the maximum drawdown from a drawdown series.

    Args:
        drawdown: Drawdown series over time.

    Returns:
        The most negative drawdown value.

    Raises:
        ValueError: If ``drawdown`` is empty.
    """
    _validate_series(drawdown, "drawdown")

    return float(drawdown.min())


def _validate_daily_returns(daily_returns: pd.DataFrame) -> None:
    """Validate that a daily returns DataFrame contains asset returns."""
    if daily_returns.empty:
        raise ValueError("daily_returns must not be empty.")
    if len(daily_returns.columns) == 0:
        raise ValueError("daily_returns must contain at least one ticker column.")


def _validate_series(data: pd.Series, name: str) -> None:
    """Validate that a Series contains data."""
    if data.empty:
        raise ValueError(f"{name} must not be empty.")


def _validate_matching_tickers(
    daily_return_tickers: pd.Index,
    weight_tickers: pd.Index,
) -> None:
    """Validate that return columns and weight indexes contain the same tickers."""
    if set(daily_return_tickers) != set(weight_tickers):
        raise ValueError("daily_returns and weights tickers must match.")
