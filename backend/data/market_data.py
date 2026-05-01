"""Market data loading utilities."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_price_data(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Fetch historical adjusted close prices for one or more tickers.

    Args:
        tickers: Ticker symbols to download.
        period: yfinance period string, such as ``"1y"``, ``"5y"``, or
            ``"max"``.

    Returns:
        A DataFrame of prices indexed by date with one column per ticker.
        Rows where every ticker is missing are dropped, and small gaps are
        forward-filled with a limited fill window.

    Raises:
        ValueError: If no tickers are provided or yfinance returns no usable
            price data.
    """
    cleaned_tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    if not cleaned_tickers:
        raise ValueError("At least one ticker symbol is required.")

    raw_data = yf.download(
        tickers=cleaned_tickers,
        period=period,
        auto_adjust=False,
        progress=False,
    )

    if raw_data.empty:
        raise ValueError(
            f"No market data returned for tickers: {', '.join(cleaned_tickers)}."
        )

    prices = _extract_close_prices(raw_data, cleaned_tickers)
    prices = prices.dropna(how="all").ffill(limit=5)

    if prices.empty or prices.dropna(how="all").empty:
        raise ValueError(
            f"No usable price data returned for tickers: {', '.join(cleaned_tickers)}."
        )

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    prices.columns.name = None

    return prices.sort_index()


def _extract_close_prices(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract adjusted close prices from yfinance output."""
    if isinstance(data.columns, pd.MultiIndex):
        return _extract_multi_index_close_prices(data, tickers)

    price_column = "Adj Close" if "Adj Close" in data.columns else "Close"
    if price_column not in data.columns:
        raise ValueError("No adjusted close or close price data found.")

    column_name = tickers[0] if len(tickers) == 1 else price_column
    return data[[price_column]].rename(columns={price_column: column_name})


def _extract_multi_index_close_prices(
    data: pd.DataFrame, tickers: list[str]
) -> pd.DataFrame:
    """Extract close prices from MultiIndex yfinance output."""
    if "Adj Close" in data.columns.get_level_values(0):
        prices = data["Adj Close"]
    elif "Close" in data.columns.get_level_values(0):
        prices = data["Close"]
    elif "Adj Close" in data.columns.get_level_values(1):
        prices = data.xs("Adj Close", axis=1, level=1)
    elif "Close" in data.columns.get_level_values(1):
        prices = data.xs("Close", axis=1, level=1)
    else:
        raise ValueError("No adjusted close or close price data found.")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    return prices.reindex(columns=tickers)
