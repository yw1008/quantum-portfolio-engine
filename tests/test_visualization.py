from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from backend.analysis.visualization import (
    plot_allocation_bar,
    plot_correlation_heatmap,
    plot_efficient_frontier,
    plot_price_history,
)


def test_plot_correlation_heatmap_returns_figure():
    correlation_matrix = pd.DataFrame(
        [[1.0, 0.25], [0.25, 1.0]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    fig = plot_correlation_heatmap(correlation_matrix)

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Asset Correlation Heatmap"
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_allocation_bar_returns_figure():
    weights = pd.Series({"AAPL": 0.60, "MSFT": 0.40})

    fig = plot_allocation_bar(weights)

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Portfolio Allocation"
    assert len(fig.axes[0].patches) == 2
    plt.close(fig)


def test_plot_price_history_returns_figure():
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 101.0, 103.0],
            "MSFT": [200.0, 198.0, 202.0],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    fig = plot_price_history(prices)

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Historical Price Data"
    assert len(fig.axes[0].lines) == 2
    plt.close(fig)


def test_plot_efficient_frontier_returns_figure():
    frontier = pd.DataFrame(
        {
            "target_return": [0.05, 0.08],
            "expected_return": [0.05, 0.08],
            "volatility": [0.10, 0.15],
            "sharpe_ratio": [0.50, 0.53],
            "weights": [pd.Series({"AAPL": 1.0}), pd.Series({"MSFT": 1.0})],
        }
    )

    fig = plot_efficient_frontier(frontier)

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Efficient Frontier"
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_correlation_heatmap_raises_value_error_for_empty_input():
    with pytest.raises(ValueError, match="correlation_matrix must not be empty"):
        plot_correlation_heatmap(pd.DataFrame())


def test_plot_allocation_bar_raises_value_error_for_empty_input():
    with pytest.raises(ValueError, match="weights must not be empty"):
        plot_allocation_bar(pd.Series(dtype=float))


def test_plot_price_history_raises_value_error_for_empty_input():
    with pytest.raises(ValueError, match="prices must not be empty"):
        plot_price_history(pd.DataFrame())


def test_plot_efficient_frontier_raises_value_error_for_missing_columns():
    frontier = pd.DataFrame({"expected_return": [0.05], "volatility": [0.10]})

    with pytest.raises(ValueError, match="frontier is missing required columns"):
        plot_efficient_frontier(frontier)
