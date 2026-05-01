from __future__ import annotations

import pandas as pd
import pytest

from backend.classical.markowitz import optimize_portfolio


def test_optimize_portfolio_returns_long_only_weights_that_sum_to_one():
    expected_returns = pd.Series({"AAPL": 0.12, "MSFT": 0.10, "BND": 0.03})
    covariance_matrix = pd.DataFrame(
        [
            [0.040, 0.018, 0.002],
            [0.018, 0.030, 0.001],
            [0.002, 0.001, 0.005],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    weights = optimize_portfolio(expected_returns, covariance_matrix)

    assert list(weights.index) == ["AAPL", "MSFT", "BND"]
    assert weights.name == "weight"
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= -1e-8).all()


def test_optimize_portfolio_uses_covariance_labels_for_alignment():
    expected_returns = pd.Series({"AAPL": 0.12, "MSFT": 0.08})
    covariance_matrix = pd.DataFrame(
        [[0.02, 0.01], [0.01, 0.03]],
        index=["MSFT", "AAPL"],
        columns=["MSFT", "AAPL"],
    )

    weights = optimize_portfolio(expected_returns, covariance_matrix)

    assert list(weights.index) == ["AAPL", "MSFT"]
    assert weights.sum() == pytest.approx(1.0)


def test_optimize_portfolio_risk_aversion_changes_solution():
    expected_returns = pd.Series({"HIGH": 0.20, "LOW": 0.05})
    covariance_matrix = pd.DataFrame(
        [[0.20, 0.00], [0.00, 0.01]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    low_risk_aversion = optimize_portfolio(
        expected_returns, covariance_matrix, risk_aversion=0.1
    )
    high_risk_aversion = optimize_portfolio(
        expected_returns, covariance_matrix, risk_aversion=10.0
    )

    assert low_risk_aversion["HIGH"] > high_risk_aversion["HIGH"]


def test_optimize_portfolio_raises_value_error_for_empty_expected_returns():
    covariance_matrix = pd.DataFrame([[0.01]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="expected_returns must not be empty"):
        optimize_portfolio(pd.Series(dtype=float), covariance_matrix)


def test_optimize_portfolio_raises_value_error_for_empty_covariance_matrix():
    expected_returns = pd.Series({"AAPL": 0.10})

    with pytest.raises(ValueError, match="covariance_matrix must not be empty"):
        optimize_portfolio(expected_returns, pd.DataFrame())


def test_optimize_portfolio_raises_value_error_for_mismatched_tickers():
    expected_returns = pd.Series({"AAPL": 0.10, "MSFT": 0.08})
    covariance_matrix = pd.DataFrame(
        [[0.01, 0.00], [0.00, 0.02]],
        index=["AAPL", "GOOGL"],
        columns=["AAPL", "GOOGL"],
    )

    with pytest.raises(
        ValueError, match="Tickers in expected_returns and covariance_matrix must match"
    ):
        optimize_portfolio(expected_returns, covariance_matrix)


def test_optimize_portfolio_raises_value_error_for_invalid_risk_aversion():
    expected_returns = pd.Series({"AAPL": 0.10})
    covariance_matrix = pd.DataFrame([[0.01]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="risk_aversion must be positive"):
        optimize_portfolio(expected_returns, covariance_matrix, risk_aversion=0.0)
