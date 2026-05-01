# Market Data

Market data is the starting point for portfolio research. In this project, the
Phase 1 loader retrieves historical price data for one or more tickers and
returns a clean pandas DataFrame indexed by date.

## Current Implementation

- Module: `backend/data/market_data.py`
- Function: `fetch_price_data`
- Source: yfinance
- Preferred price field: adjusted close, with close prices used as a fallback

## Notes

Adjusted close prices are preferred because they account for corporate actions
such as dividends and stock splits. This makes return calculations more suitable
for historical portfolio analysis.
