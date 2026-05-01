"""Covariance calculation utilities."""

from __future__ import annotations

import pandas as pd


def compute_covariance_matrix(
    daily_returns: pd.DataFrame, trading_days: int = 252
) -> pd.DataFrame:
    """Compute an annualized covariance matrix from daily asset returns.

    Args:
        daily_returns: Daily percentage returns with one column per ticker.
        trading_days: Number of trading days used for annualization.

    Returns:
        A covariance matrix DataFrame with ticker labels preserved in both rows
        and columns.

    Raises:
        ValueError: If ``daily_returns`` is empty, contains no asset columns, or
            ``trading_days`` is not positive.
    """
    if daily_returns.empty:
        raise ValueError("daily_returns must not be empty.")
    if len(daily_returns.columns) == 0:
        raise ValueError("daily_returns must contain at least one asset column.")
    if trading_days <= 0:
        raise ValueError("trading_days must be a positive integer.")

    covariance_matrix = daily_returns.cov() * trading_days
    covariance_matrix.index.name = daily_returns.columns.name
    covariance_matrix.columns.name = daily_returns.columns.name

    return covariance_matrix
