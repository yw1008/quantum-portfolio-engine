from __future__ import annotations

import pandas as pd
import pytest

from backend.quantum.qubo import build_portfolio_qubo


def test_build_portfolio_qubo_returns_labeled_symmetric_matrix():
    expected_returns = pd.Series({"AAPL": 0.12, "MSFT": 0.08, "BND": 0.04})
    covariance_matrix = pd.DataFrame(
        [
            [0.040, 0.010, 0.002],
            [0.010, 0.030, 0.001],
            [0.002, 0.001, 0.010],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    qubo = build_portfolio_qubo(
        expected_returns,
        covariance_matrix,
        num_assets_to_select=2,
        risk_aversion=2.0,
        penalty_strength=5.0,
    )

    assert list(qubo.index) == ["AAPL", "MSFT", "BND"]
    assert list(qubo.columns) == ["AAPL", "MSFT", "BND"]
    pd.testing.assert_frame_equal(qubo, qubo.T)


def test_build_portfolio_qubo_uses_risk_return_and_penalty_coefficients():
    expected_returns = pd.Series({"AAPL": 0.12, "MSFT": 0.08, "BND": 0.04})
    covariance_matrix = pd.DataFrame(
        [
            [0.040, 0.010, 0.002],
            [0.010, 0.030, 0.001],
            [0.002, 0.001, 0.010],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    qubo = build_portfolio_qubo(
        expected_returns,
        covariance_matrix,
        num_assets_to_select=2,
        risk_aversion=2.0,
        penalty_strength=5.0,
    )

    expected_qubo = pd.DataFrame(
        [
            [-15.04, 5.005, 5.001],
            [5.005, -15.025, 5.0005],
            [5.001, 5.0005, -15.015],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    pd.testing.assert_frame_equal(qubo, expected_qubo)


def test_build_portfolio_qubo_aligns_covariance_by_ticker_label():
    expected_returns = pd.Series({"AAPL": 0.12, "MSFT": 0.08})
    covariance_matrix = pd.DataFrame(
        [[0.030, 0.010], [0.010, 0.040]],
        index=["MSFT", "AAPL"],
        columns=["MSFT", "AAPL"],
    )

    qubo = build_portfolio_qubo(
        expected_returns,
        covariance_matrix,
        num_assets_to_select=1,
    )

    assert list(qubo.index) == ["AAPL", "MSFT"]
    assert qubo.loc["AAPL", "AAPL"] == pytest.approx(0.040 - 0.120 - 10.0)
    assert qubo.loc["MSFT", "MSFT"] == pytest.approx(0.030 - 0.080 - 10.0)
    assert qubo.loc["AAPL", "MSFT"] == pytest.approx(0.010 + 10.0)


def test_qubo_objective_matches_minimized_binary_selection_objective_without_constant():
    expected_returns = pd.Series({"AAPL": 0.12, "MSFT": 0.08, "BND": 0.04})
    covariance_matrix = pd.DataFrame(
        [
            [0.040, 0.010, 0.002],
            [0.010, 0.030, 0.001],
            [0.002, 0.001, 0.010],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    selected_assets = pd.Series({"AAPL": 1.0, "MSFT": 0.0, "BND": 1.0})
    selected_count = 2
    risk_aversion = 2.0
    penalty_strength = 5.0

    qubo = build_portfolio_qubo(
        expected_returns,
        covariance_matrix,
        num_assets_to_select=selected_count,
        risk_aversion=risk_aversion,
        penalty_strength=penalty_strength,
    )

    x = selected_assets.loc[qubo.index]
    qubo_objective = float(x.T @ qubo @ x)
    expected_return = float(x.dot(expected_returns.loc[x.index]) / selected_count)
    portfolio_variance = float(
        x.T @ covariance_matrix.loc[x.index, x.index] @ x / selected_count**2
    )
    penalty = penalty_strength * (x.sum() - selected_count) ** 2
    expected_objective = (
        risk_aversion * portfolio_variance
        - expected_return
        + penalty
        - penalty_strength * selected_count**2
    )

    assert qubo_objective == pytest.approx(expected_objective)


def test_build_portfolio_qubo_raises_value_error_for_empty_expected_returns():
    covariance_matrix = pd.DataFrame([[0.01]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="expected_returns must not be empty"):
        build_portfolio_qubo(
            pd.Series(dtype=float),
            covariance_matrix,
            num_assets_to_select=1,
        )


def test_build_portfolio_qubo_raises_value_error_for_empty_covariance():
    expected_returns = pd.Series({"AAPL": 0.10})

    with pytest.raises(ValueError, match="covariance_matrix must not be empty"):
        build_portfolio_qubo(expected_returns, pd.DataFrame(), num_assets_to_select=1)


def test_build_portfolio_qubo_raises_value_error_for_ticker_mismatch():
    expected_returns = pd.Series({"AAPL": 0.10, "MSFT": 0.08})
    covariance_matrix = pd.DataFrame(
        [[0.01, 0.00], [0.00, 0.02]],
        index=["AAPL", "GOOGL"],
        columns=["AAPL", "GOOGL"],
    )

    with pytest.raises(
        ValueError,
        match="Tickers in expected_returns and covariance_matrix must match",
    ):
        build_portfolio_qubo(expected_returns, covariance_matrix, 1)


def test_build_portfolio_qubo_raises_value_error_for_invalid_covariance_labels():
    expected_returns = pd.Series({"AAPL": 0.10, "MSFT": 0.08})
    covariance_matrix = pd.DataFrame(
        [[0.01, 0.00], [0.00, 0.02]],
        index=["AAPL", "MSFT"],
        columns=["MSFT", "AAPL"],
    )

    with pytest.raises(
        ValueError,
        match="covariance_matrix must have matching row and column labels",
    ):
        build_portfolio_qubo(expected_returns, covariance_matrix, 1)


@pytest.mark.parametrize("selected_count", [0, 3])
def test_build_portfolio_qubo_raises_value_error_for_invalid_selected_count(
    selected_count: int,
):
    expected_returns = pd.Series({"AAPL": 0.10, "MSFT": 0.08})
    covariance_matrix = pd.DataFrame(
        [[0.01, 0.00], [0.00, 0.02]],
        index=expected_returns.index,
        columns=expected_returns.index,
    )

    with pytest.raises(
        ValueError,
        match="num_assets_to_select must be between 1 and the number of assets",
    ):
        build_portfolio_qubo(expected_returns, covariance_matrix, selected_count)


def test_build_portfolio_qubo_raises_value_error_for_invalid_risk_aversion():
    expected_returns = pd.Series({"AAPL": 0.10})
    covariance_matrix = pd.DataFrame([[0.01]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="risk_aversion must be positive"):
        build_portfolio_qubo(
            expected_returns,
            covariance_matrix,
            num_assets_to_select=1,
            risk_aversion=0.0,
        )


def test_build_portfolio_qubo_raises_value_error_for_invalid_penalty_strength():
    expected_returns = pd.Series({"AAPL": 0.10})
    covariance_matrix = pd.DataFrame([[0.01]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="penalty_strength must be positive"):
        build_portfolio_qubo(
            expected_returns,
            covariance_matrix,
            num_assets_to_select=1,
            penalty_strength=0.0,
        )
