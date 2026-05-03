"""Visualization utilities for QUBO analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_qubo_heatmap(Q: pd.DataFrame) -> Figure:
    """Plot a heatmap of a QUBO matrix.

    Args:
        Q: Square QUBO matrix indexed and columned by ticker.

    Returns:
        A matplotlib Figure containing the QUBO heatmap.

    Raises:
        ValueError: If ``Q`` is empty or not a labeled square matrix.
    """
    _validate_qubo_matrix(Q)

    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap = ax.imshow(Q, cmap="coolwarm")

    ax.set_title("QUBO Matrix Heatmap")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Ticker")
    ax.set_xticks(range(len(Q.columns)))
    ax.set_xticklabels(Q.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(Q.index)))
    ax.set_yticklabels(Q.index)
    fig.colorbar(heatmap, ax=ax, label="QUBO Coefficient")
    fig.tight_layout()

    return fig


def plot_qubo_solution_landscape(solutions: pd.DataFrame) -> Figure:
    """Plot QUBO objective values across candidate binary solutions.

    Args:
        solutions: DataFrame containing ``bitstring``, ``objective_value``, and
            ``num_selected`` columns.

    Returns:
        A matplotlib Figure containing the solution landscape.

    Raises:
        ValueError: If ``solutions`` is empty or missing required columns.
    """
    _validate_solutions(solutions)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = range(len(solutions))
    scatter = ax.scatter(
        x_positions,
        solutions["objective_value"],
        c=solutions["num_selected"],
        cmap="viridis",
        s=45,
    )

    ax.plot(x_positions, solutions["objective_value"], color="#4C78A8", linewidth=1.0)
    ax.set_title("QUBO Solution Landscape")
    ax.set_xlabel("Candidate Solution")
    ax.set_ylabel("Objective Value")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(solutions["bitstring"], rotation=45, ha="right")
    fig.colorbar(scatter, ax=ax, label="Number Selected")
    fig.tight_layout()

    return fig


def _validate_qubo_matrix(Q: pd.DataFrame) -> None:
    """Validate that a QUBO matrix is non-empty and square."""
    if Q.empty:
        raise ValueError("Q must not be empty.")
    if len(Q.index) != len(Q.columns):
        raise ValueError("Q must be square.")
    if list(Q.index) != list(Q.columns):
        raise ValueError("Q must have matching row and column labels.")


def _validate_solutions(solutions: pd.DataFrame) -> None:
    """Validate solution landscape input."""
    if solutions.empty:
        raise ValueError("solutions must not be empty.")

    required_columns = {"bitstring", "objective_value", "num_selected"}
    missing_columns = required_columns.difference(solutions.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"solutions is missing required columns: {missing}.")
