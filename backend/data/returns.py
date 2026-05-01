"""Return calculation utilities."""

from __future__ import annotations

import pandas as pd


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage daily returns from asset price data.

    Args:
        prices: Price DataFrame indexed by date with one column per ticker.

    Returns:
        A DataFrame of daily percentage returns with the same ticker columns as
        the input prices. Rows where every return is missing are dropped.

    Raises:
        ValueError: If ``prices`` is empty or contains no usable price columns.
    """
    _validate_dataframe(prices, "prices")

    daily_returns = prices.pct_change(fill_method=None).dropna(how="all")

    if daily_returns.empty:
        raise ValueError("No daily returns could be computed from the price data.")

    daily_returns.columns.name = prices.columns.name
    return daily_returns


def compute_annualized_returns(
    daily_returns: pd.DataFrame, trading_days: int = 252
) -> pd.Series:
    """Compute annualized mean returns from daily returns.

    Args:
        daily_returns: Daily percentage returns with one column per ticker.
        trading_days: Number of trading days used for annualization.

    Returns:
        A Series of annualized returns indexed by ticker.

    Raises:
        ValueError: If ``daily_returns`` is empty, contains no usable return
            columns, or ``trading_days`` is not positive.
    """
    _validate_dataframe(daily_returns, "daily_returns")
    if trading_days <= 0:
        raise ValueError("trading_days must be a positive integer.")

    annualized_returns = daily_returns.mean(skipna=True) * trading_days

    if annualized_returns.dropna().empty:
        raise ValueError("No annualized returns could be computed from daily returns.")

    return annualized_returns


def _validate_dataframe(data: pd.DataFrame, name: str) -> None:
    """Validate that a DataFrame contains data and at least one column."""
    if data.empty:
        raise ValueError(f"{name} must not be empty.")
    if len(data.columns) == 0:
        raise ValueError(f"{name} must contain at least one asset column.")
