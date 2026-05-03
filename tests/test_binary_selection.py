from __future__ import annotations

import pandas as pd
import pytest

from backend.quantum.binary_selection import (
    compute_binary_portfolio_score,
    normalize_binary_selection,
)


def test_normalize_binary_selection_equal_weights_selected_assets():
    selection = pd.Series({"AAPL": 1, "MSFT": 0, "BND": 1})

    weights = normalize_binary_selection(selection)

    expected_weights = pd.Series(
        {"AAPL": 0.50, "MSFT": 0.00, "BND": 0.50},
        name="weight",
    )
    pd.testing.assert_series_equal(weights, expected_weights)


def test_compute_binary_portfolio_score_uses_equal_weight_selection():
    selection = pd.Series({"AAPL": 1, "MSFT": 0, "BND": 1})
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

    score = compute_binary_portfolio_score(
        selection,
        expected_returns,
        covariance_matrix,
        risk_aversion=2.0,
    )

    expected_return = 0.50 * 0.12 + 0.50 * 0.04
    expected_variance = 0.50**2 * 0.040 + 2 * 0.50 * 0.50 * 0.002 + 0.50**2 * 0.010
    expected_score = expected_return - 2.0 * expected_variance
    assert score == pytest.approx(expected_score)


def test_compute_binary_portfolio_score_aligns_by_ticker_label():
    selection = pd.Series({"AAPL": 1, "MSFT": 0, "BND": 1})
    expected_returns = pd.Series({"BND": 0.04, "MSFT": 0.08, "AAPL": 0.12})
    covariance_matrix = pd.DataFrame(
        [
            [0.010, 0.001, 0.002],
            [0.001, 0.030, 0.010],
            [0.002, 0.010, 0.040],
        ],
        index=["BND", "MSFT", "AAPL"],
        columns=["BND", "MSFT", "AAPL"],
    )

    score = compute_binary_portfolio_score(selection, expected_returns, covariance_matrix)

    expected_return = 0.50 * 0.12 + 0.50 * 0.04
    expected_variance = 0.50**2 * 0.040 + 2 * 0.50 * 0.50 * 0.002 + 0.50**2 * 0.010
    expected_score = expected_return - expected_variance
    assert score == pytest.approx(expected_score)


def test_normalize_binary_selection_raises_value_error_for_empty_selection():
    with pytest.raises(ValueError, match="selection must not be empty"):
        normalize_binary_selection(pd.Series(dtype=int))


def test_normalize_binary_selection_raises_value_error_for_non_binary_values():
    selection = pd.Series({"AAPL": 1, "MSFT": 2})

    with pytest.raises(ValueError, match="selection values must be only 0 or 1"):
        normalize_binary_selection(selection)


def test_normalize_binary_selection_raises_value_error_for_no_selected_assets():
    selection = pd.Series({"AAPL": 0, "MSFT": 0})

    with pytest.raises(ValueError, match="selection must include at least one"):
        normalize_binary_selection(selection)


def test_compute_binary_portfolio_score_raises_value_error_for_empty_expected_returns():
    selection = pd.Series({"AAPL": 1})
    covariance_matrix = pd.DataFrame([[0.01]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="expected_returns must not be empty"):
        compute_binary_portfolio_score(
            selection,
            pd.Series(dtype=float),
            covariance_matrix,
        )


def test_compute_binary_portfolio_score_raises_value_error_for_empty_covariance():
    selection = pd.Series({"AAPL": 1})
    expected_returns = pd.Series({"AAPL": 0.10})

    with pytest.raises(ValueError, match="covariance_matrix must not be empty"):
        compute_binary_portfolio_score(selection, expected_returns, pd.DataFrame())


def test_compute_binary_portfolio_score_raises_value_error_for_invalid_risk_aversion():
    selection = pd.Series({"AAPL": 1})
    expected_returns = pd.Series({"AAPL": 0.10})
    covariance_matrix = pd.DataFrame([[0.01]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="risk_aversion must be positive"):
        compute_binary_portfolio_score(
            selection,
            expected_returns,
            covariance_matrix,
            risk_aversion=0.0,
        )


def test_compute_binary_portfolio_score_raises_value_error_for_mismatched_returns():
    selection = pd.Series({"AAPL": 1, "MSFT": 1})
    expected_returns = pd.Series({"AAPL": 0.10, "GOOGL": 0.08})
    covariance_matrix = pd.DataFrame(
        [[0.01, 0.00], [0.00, 0.02]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    with pytest.raises(
        ValueError,
        match="Tickers in selection and expected_returns must match",
    ):
        compute_binary_portfolio_score(selection, expected_returns, covariance_matrix)


def test_compute_binary_portfolio_score_raises_value_error_for_mismatched_covariance():
    selection = pd.Series({"AAPL": 1, "MSFT": 1})
    expected_returns = pd.Series({"AAPL": 0.10, "MSFT": 0.08})
    covariance_matrix = pd.DataFrame(
        [[0.01, 0.00], [0.00, 0.02]],
        index=["AAPL", "GOOGL"],
        columns=["AAPL", "GOOGL"],
    )

    with pytest.raises(
        ValueError,
        match="Tickers in selection and covariance_matrix must match",
    ):
        compute_binary_portfolio_score(selection, expected_returns, covariance_matrix)


def test_compute_binary_portfolio_score_raises_value_error_for_invalid_covariance_labels():
    selection = pd.Series({"AAPL": 1, "MSFT": 1})
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
        compute_binary_portfolio_score(selection, expected_returns, covariance_matrix)
