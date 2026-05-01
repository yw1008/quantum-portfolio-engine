from __future__ import annotations

import pandas as pd
import pytest

from backend.data import market_data


def test_fetch_price_data_returns_adjusted_close_for_multiple_tickers(monkeypatch):
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    columns = pd.MultiIndex.from_product(
        [["Adj Close", "Close"], ["AAPL", "MSFT"]],
        names=["Price", "Ticker"],
    )
    raw_data = pd.DataFrame(
        [
            [100.0, 200.0, 99.0, 199.0],
            [None, 201.0, None, 198.0],
            [102.0, None, 101.0, None],
        ],
        index=dates,
        columns=columns,
    )

    def fake_download(**kwargs):
        assert kwargs["tickers"] == ["AAPL", "MSFT"]
        assert kwargs["period"] == "1y"
        assert kwargs["auto_adjust"] is False
        assert kwargs["progress"] is False
        return raw_data

    monkeypatch.setattr(market_data.yf, "download", fake_download)

    prices = market_data.fetch_price_data(["aapl", "msft"], period="1y")

    expected = pd.DataFrame(
        {
            "AAPL": [100.0, 100.0, 102.0],
            "MSFT": [200.0, 201.0, 201.0],
        },
        index=pd.Index(dates, name="Date"),
    )
    pd.testing.assert_frame_equal(prices, expected)


def test_fetch_price_data_falls_back_to_close_for_single_ticker(monkeypatch):
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    raw_data = pd.DataFrame({"Close": [50.0, 51.0]}, index=dates)

    monkeypatch.setattr(market_data.yf, "download", lambda **_: raw_data)

    prices = market_data.fetch_price_data(["SPY"])

    expected = pd.DataFrame(
        {"SPY": [50.0, 51.0]},
        index=pd.Index(dates, name="Date"),
    )
    pd.testing.assert_frame_equal(prices, expected)


def test_fetch_price_data_drops_rows_where_all_prices_are_missing(monkeypatch):
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    raw_data = pd.DataFrame(
        {
            "Adj Close": [None, 10.0, 11.0],
            "Close": [None, 9.5, 10.5],
        },
        index=dates,
    )

    monkeypatch.setattr(market_data.yf, "download", lambda **_: raw_data)

    prices = market_data.fetch_price_data(["QQQ"])

    assert list(prices.index) == list(dates[1:])
    assert list(prices["QQQ"]) == [10.0, 11.0]


def test_fetch_price_data_raises_value_error_when_no_data(monkeypatch):
    monkeypatch.setattr(market_data.yf, "download", lambda **_: pd.DataFrame())

    with pytest.raises(ValueError, match="No market data returned"):
        market_data.fetch_price_data(["MISSING"])


def test_fetch_price_data_raises_value_error_without_tickers():
    with pytest.raises(ValueError, match="At least one ticker"):
        market_data.fetch_price_data([])
