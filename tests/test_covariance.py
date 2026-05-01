from __future__ import annotations

import pandas as pd
import pytest

from backend.data.covariance import compute_covariance_matrix


def test_compute_covariance_matrix_annualizes_daily_covariance():
    daily_returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, 0.03],
            "MSFT": [0.03, 0.02, 0.01],
        }
    )

    covariance_matrix = compute_covariance_matrix(daily_returns, trading_days=10)

    expected = daily_returns.cov() * 10
    pd.testing.assert_frame_equal(covariance_matrix, expected)


def test_compute_covariance_matrix_preserves_ticker_labels():
    daily_returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, 0.03],
            "MSFT": [0.02, 0.01, 0.04],
            "GOOGL": [-0.01, 0.00, 0.02],
        }
    )
    daily_returns.columns.name = "Ticker"

    covariance_matrix = compute_covariance_matrix(daily_returns)

    assert list(covariance_matrix.index) == ["AAPL", "MSFT", "GOOGL"]
    assert list(covariance_matrix.columns) == ["AAPL", "MSFT", "GOOGL"]
    assert covariance_matrix.index.name == "Ticker"
    assert covariance_matrix.columns.name == "Ticker"


def test_compute_covariance_matrix_raises_value_error_for_empty_input():
    with pytest.raises(ValueError, match="daily_returns must not be empty"):
        compute_covariance_matrix(pd.DataFrame())


def test_compute_covariance_matrix_raises_value_error_for_invalid_trading_days():
    daily_returns = pd.DataFrame({"AAPL": [0.01, 0.02]})

    with pytest.raises(ValueError, match="trading_days must be a positive integer"):
        compute_covariance_matrix(daily_returns, trading_days=0)
