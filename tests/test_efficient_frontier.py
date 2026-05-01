from __future__ import annotations

import pandas as pd
import pytest

from backend.classical import efficient_frontier
from backend.classical.efficient_frontier import generate_efficient_frontier


def test_generate_efficient_frontier_returns_expected_columns():
    expected_returns = pd.Series({"AAPL": 0.12, "MSFT": 0.08, "BND": 0.03})
    covariance_matrix = pd.DataFrame(
        [
            [0.040, 0.010, 0.002],
            [0.010, 0.030, 0.001],
            [0.002, 0.001, 0.005],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    frontier = generate_efficient_frontier(
        expected_returns, covariance_matrix, num_points=5
    )

    assert list(frontier.columns) == [
        "target_return",
        "expected_return",
        "volatility",
        "sharpe_ratio",
        "weights",
    ]
    assert len(frontier) == 5


def test_generate_efficient_frontier_returns_valid_long_only_weights():
    expected_returns = pd.Series({"AAPL": 0.12, "MSFT": 0.08, "BND": 0.03})
    covariance_matrix = pd.DataFrame(
        [
            [0.040, 0.010, 0.002],
            [0.010, 0.030, 0.001],
            [0.002, 0.001, 0.005],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    frontier = generate_efficient_frontier(
        expected_returns, covariance_matrix, num_points=4
    )

    for weights in frontier["weights"]:
        assert list(weights.index) == ["AAPL", "MSFT", "BND"]
        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= -1e-8).all()

    assert (frontier["expected_return"] >= frontier["target_return"] - 1e-8).all()
    assert (frontier["volatility"] > 0).all()


def test_generate_efficient_frontier_aligns_covariance_by_ticker_label():
    expected_returns = pd.Series({"AAPL": 0.12, "MSFT": 0.08})
    covariance_matrix = pd.DataFrame(
        [[0.03, 0.01], [0.01, 0.04]],
        index=["MSFT", "AAPL"],
        columns=["MSFT", "AAPL"],
    )

    frontier = generate_efficient_frontier(
        expected_returns, covariance_matrix, num_points=3
    )

    assert len(frontier) == 3
    for weights in frontier["weights"]:
        assert list(weights.index) == ["AAPL", "MSFT"]


def test_generate_efficient_frontier_skips_infeasible_points(monkeypatch):
    expected_returns = pd.Series({"AAPL": 0.10, "MSFT": 0.05})
    covariance_matrix = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.03]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    monkeypatch.setattr(
        efficient_frontier,
        "_build_target_returns",
        lambda sorted_returns, num_points: [0.05, 0.50],
    )

    frontier = generate_efficient_frontier(
        expected_returns, covariance_matrix, num_points=2
    )

    assert len(frontier) == 1
    assert frontier.loc[0, "target_return"] == pytest.approx(0.05)


def test_generate_efficient_frontier_raises_value_error_for_empty_expected_returns():
    covariance_matrix = pd.DataFrame([[0.01]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="expected_returns must not be empty"):
        generate_efficient_frontier(pd.Series(dtype=float), covariance_matrix)


def test_generate_efficient_frontier_raises_value_error_for_empty_covariance_matrix():
    expected_returns = pd.Series({"AAPL": 0.10})

    with pytest.raises(ValueError, match="covariance_matrix must not be empty"):
        generate_efficient_frontier(expected_returns, pd.DataFrame())


def test_generate_efficient_frontier_raises_value_error_for_mismatched_tickers():
    expected_returns = pd.Series({"AAPL": 0.10, "MSFT": 0.08})
    covariance_matrix = pd.DataFrame(
        [[0.01, 0.00], [0.00, 0.02]],
        index=["AAPL", "GOOGL"],
        columns=["AAPL", "GOOGL"],
    )

    with pytest.raises(
        ValueError, match="Tickers in expected_returns and covariance_matrix must match"
    ):
        generate_efficient_frontier(expected_returns, covariance_matrix)


def test_generate_efficient_frontier_raises_value_error_for_invalid_num_points():
    expected_returns = pd.Series({"AAPL": 0.10})
    covariance_matrix = pd.DataFrame([[0.01]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="num_points must be positive"):
        generate_efficient_frontier(expected_returns, covariance_matrix, num_points=0)
