from __future__ import annotations

import pandas as pd
import pytest

from backend.analysis.backtest import (
    compute_cumulative_returns,
    compute_drawdown,
    compute_max_drawdown,
    compute_portfolio_returns,
)


def test_compute_portfolio_returns_uses_weighted_daily_returns():
    daily_returns = pd.DataFrame(
        {
            "AAPL": [0.01, -0.02, 0.03],
            "MSFT": [0.02, 0.01, -0.01],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    weights = pd.Series({"AAPL": 0.60, "MSFT": 0.40})

    portfolio_returns = compute_portfolio_returns(daily_returns, weights)

    expected_returns = pd.Series(
        [0.014, -0.008, 0.014],
        index=daily_returns.index,
    )
    pd.testing.assert_series_equal(portfolio_returns, expected_returns)


def test_compute_portfolio_returns_aligns_weights_by_ticker_label():
    daily_returns = pd.DataFrame(
        {
            "AAPL": [0.01, -0.02],
            "MSFT": [0.02, 0.01],
        }
    )
    weights = pd.Series({"MSFT": 0.40, "AAPL": 0.60})

    portfolio_returns = compute_portfolio_returns(daily_returns, weights)

    expected_returns = pd.Series([0.014, -0.008])
    pd.testing.assert_series_equal(portfolio_returns, expected_returns)


def test_compute_cumulative_returns_compounds_returns():
    portfolio_returns = pd.Series([0.10, -0.05, 0.02])

    cumulative_returns = compute_cumulative_returns(portfolio_returns)

    expected_returns = pd.Series([0.10, 0.045, 0.0659])
    pd.testing.assert_series_equal(cumulative_returns, expected_returns)


def test_compute_drawdown_uses_running_peak():
    cumulative_returns = pd.Series([0.10, 0.20, 0.08, 0.32, 0.188])

    drawdown = compute_drawdown(cumulative_returns)

    expected_drawdown = pd.Series([0.0, 0.0, -0.10, 0.0, -0.10])
    pd.testing.assert_series_equal(drawdown, expected_drawdown)


def test_compute_max_drawdown_returns_most_negative_drawdown():
    drawdown = pd.Series([0.0, -0.05, -0.25, -0.10])

    max_drawdown = compute_max_drawdown(drawdown)

    assert max_drawdown == pytest.approx(-0.25)


def test_compute_portfolio_returns_raises_value_error_for_empty_daily_returns():
    weights = pd.Series({"AAPL": 1.0})

    with pytest.raises(ValueError, match="daily_returns must not be empty"):
        compute_portfolio_returns(pd.DataFrame(), weights)


def test_compute_portfolio_returns_raises_value_error_for_empty_weights():
    daily_returns = pd.DataFrame({"AAPL": [0.01]})

    with pytest.raises(ValueError, match="weights must not be empty"):
        compute_portfolio_returns(daily_returns, pd.Series(dtype=float))


def test_compute_portfolio_returns_raises_value_error_for_mismatched_tickers():
    daily_returns = pd.DataFrame({"AAPL": [0.01], "MSFT": [0.02]})
    weights = pd.Series({"AAPL": 0.50, "GOOGL": 0.50})

    with pytest.raises(
        ValueError,
        match="daily_returns and weights tickers must match",
    ):
        compute_portfolio_returns(daily_returns, weights)


def test_compute_cumulative_returns_raises_value_error_for_empty_input():
    with pytest.raises(ValueError, match="portfolio_returns must not be empty"):
        compute_cumulative_returns(pd.Series(dtype=float))


def test_compute_drawdown_raises_value_error_for_empty_input():
    with pytest.raises(ValueError, match="cumulative_returns must not be empty"):
        compute_drawdown(pd.Series(dtype=float))


def test_compute_max_drawdown_raises_value_error_for_empty_input():
    with pytest.raises(ValueError, match="drawdown must not be empty"):
        compute_max_drawdown(pd.Series(dtype=float))
