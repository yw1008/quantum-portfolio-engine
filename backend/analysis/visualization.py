"""Visualization utilities for portfolio analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame) -> Figure:
    """Plot a heatmap of asset correlations.

    Args:
        correlation_matrix: Correlation matrix indexed and columned by ticker.

    Returns:
        A matplotlib Figure containing the correlation heatmap.

    Raises:
        ValueError: If ``correlation_matrix`` is empty.
    """
    if correlation_matrix.empty:
        raise ValueError("correlation_matrix must not be empty.")

    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_title("Asset Correlation Heatmap")
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_yticklabels(correlation_matrix.index)
    fig.colorbar(heatmap, ax=ax, label="Correlation")
    fig.tight_layout()

    return fig


def plot_allocation_bar(weights: pd.Series) -> Figure:
    """Plot portfolio allocation weights as a bar chart.

    Args:
        weights: Portfolio weights indexed by ticker.

    Returns:
        A matplotlib Figure containing the allocation bar chart.

    Raises:
        ValueError: If ``weights`` is empty.
    """
    if weights.empty:
        raise ValueError("weights must not be empty.")

    fig, ax = plt.subplots(figsize=(8, 5))
    weights.plot(kind="bar", ax=ax, color="#4C78A8")

    ax.set_title("Portfolio Allocation")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Weight")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    return fig


def plot_price_history(prices: pd.DataFrame) -> Figure:
    """Plot historical asset prices.

    Args:
        prices: Price DataFrame indexed by date with one column per ticker.

    Returns:
        A matplotlib Figure containing price history lines.

    Raises:
        ValueError: If ``prices`` is empty.
    """
    if prices.empty:
        raise ValueError("prices must not be empty.")

    fig, ax = plt.subplots(figsize=(10, 6))
    prices.plot(ax=ax)

    ax.set_title("Historical Price Data")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(title="Ticker")
    fig.tight_layout()

    return fig


def plot_efficient_frontier(frontier: pd.DataFrame) -> Figure:
    """Plot efficient frontier portfolios by volatility and expected return.

    Args:
        frontier: DataFrame containing ``expected_return``, ``volatility``, and
            ``sharpe_ratio`` columns.

    Returns:
        A matplotlib Figure containing the efficient frontier scatter plot.

    Raises:
        ValueError: If ``frontier`` is empty or missing required columns.
    """
    if frontier.empty:
        raise ValueError("frontier must not be empty.")

    required_columns = {"expected_return", "volatility", "sharpe_ratio"}
    missing_columns = required_columns.difference(frontier.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"frontier is missing required columns: {missing}.")

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        frontier["volatility"],
        frontier["expected_return"],
        c=frontier["sharpe_ratio"],
        cmap="viridis",
        s=45,
    )

    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Expected Return")
    fig.colorbar(scatter, ax=ax, label="Sharpe Ratio")
    fig.tight_layout()

    return fig
