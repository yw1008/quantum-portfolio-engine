from __future__ import annotations

import pandas as pd
import pytest

from backend.data.returns import compute_annualized_returns, compute_daily_returns


def test_compute_daily_returns_preserves_ticker_columns():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 121.0],
            "MSFT": [200.0, 210.0, 189.0],
        },
        index=pd.Index(dates, name="Date"),
    )

    daily_returns = compute_daily_returns(prices)

    expected = pd.DataFrame(
        {
            "AAPL": [0.10, 0.10],
            "MSFT": [0.05, -0.10],
        },
        index=pd.Index(dates[1:], name="Date"),
    )
    pd.testing.assert_frame_equal(daily_returns, expected)


def test_compute_daily_returns_drops_rows_where_all_returns_are_missing():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    prices = pd.DataFrame(
        {
            "AAPL": [None, 100.0, 105.0],
            "MSFT": [None, 200.0, 210.0],
        },
        index=dates,
    )

    daily_returns = compute_daily_returns(prices)

    expected = pd.DataFrame(
        {
            "AAPL": [0.05],
            "MSFT": [0.05],
        },
        index=pd.Index([dates[2]]),
    )
    pd.testing.assert_frame_equal(daily_returns, expected)


def test_compute_annualized_returns_uses_mean_daily_return():
    daily_returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.03],
            "MSFT": [0.02, -0.01],
        }
    )

    annualized_returns = compute_annualized_returns(daily_returns, trading_days=10)

    expected = pd.Series({"AAPL": 0.20, "MSFT": 0.05})
    pd.testing.assert_series_equal(annualized_returns, expected)


def test_compute_daily_returns_raises_value_error_for_empty_prices():
    with pytest.raises(ValueError, match="prices must not be empty"):
        compute_daily_returns(pd.DataFrame())


def test_compute_daily_returns_raises_value_error_when_no_returns_can_be_computed():
    prices = pd.DataFrame({"AAPL": [100.0]})

    with pytest.raises(ValueError, match="No daily returns"):
        compute_daily_returns(prices)


def test_compute_annualized_returns_raises_value_error_for_empty_returns():
    with pytest.raises(ValueError, match="daily_returns must not be empty"):
        compute_annualized_returns(pd.DataFrame())


def test_compute_annualized_returns_raises_value_error_for_invalid_trading_days():
    daily_returns = pd.DataFrame({"AAPL": [0.01]})

    with pytest.raises(ValueError, match="trading_days must be a positive integer"):
        compute_annualized_returns(daily_returns, trading_days=0)
