from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from backend.quantum.qubo_visualization import (
    plot_qubo_heatmap,
    plot_qubo_solution_landscape,
)


def test_plot_qubo_heatmap_returns_figure_with_ticker_labels():
    Q = pd.DataFrame(
        [[-1.0, 0.25], [0.25, -0.50]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    fig = plot_qubo_heatmap(Q)

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "QUBO Matrix Heatmap"
    assert [label.get_text() for label in fig.axes[0].get_xticklabels()] == [
        "AAPL",
        "MSFT",
    ]
    assert [label.get_text() for label in fig.axes[0].get_yticklabels()] == [
        "AAPL",
        "MSFT",
    ]
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_qubo_solution_landscape_returns_figure():
    solutions = pd.DataFrame(
        {
            "bitstring": ["00", "01", "10", "11"],
            "objective_value": [0.0, -0.5, -1.0, 0.25],
            "num_selected": [0, 1, 1, 2],
        }
    )

    fig = plot_qubo_solution_landscape(solutions)

    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "QUBO Solution Landscape"
    assert fig.axes[0].get_xlabel() == "Candidate Solution"
    assert fig.axes[0].get_ylabel() == "Objective Value"
    assert len(fig.axes[0].collections) == 1
    assert len(fig.axes[0].lines) == 1
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_qubo_heatmap_raises_value_error_for_empty_input():
    with pytest.raises(ValueError, match="Q must not be empty"):
        plot_qubo_heatmap(pd.DataFrame())


def test_plot_qubo_heatmap_raises_value_error_for_non_square_input():
    Q = pd.DataFrame([[1.0, 0.0]], index=["AAPL"], columns=["AAPL", "MSFT"])

    with pytest.raises(ValueError, match="Q must be square"):
        plot_qubo_heatmap(Q)


def test_plot_qubo_heatmap_raises_value_error_for_mismatched_labels():
    Q = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["AAPL", "MSFT"],
        columns=["MSFT", "AAPL"],
    )

    with pytest.raises(ValueError, match="Q must have matching row and column labels"):
        plot_qubo_heatmap(Q)


def test_plot_qubo_solution_landscape_raises_value_error_for_empty_input():
    with pytest.raises(ValueError, match="solutions must not be empty"):
        plot_qubo_solution_landscape(pd.DataFrame())


def test_plot_qubo_solution_landscape_raises_value_error_for_missing_columns():
    solutions = pd.DataFrame({"bitstring": ["0"], "objective_value": [0.0]})

    with pytest.raises(ValueError, match="solutions is missing required columns"):
        plot_qubo_solution_landscape(solutions)
