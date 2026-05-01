# End-to-End Classical Portfolio Demo

The Phase 1 demo notebook connects the classical portfolio pipeline from raw
market data through optimized portfolio evaluation.

## Pipeline Overview

The notebook uses five ETFs:

- `SPY`: broad U.S. equity exposure
- `QQQ`: Nasdaq-100 growth-oriented equity exposure
- `VIG`: dividend growth equity exposure
- `TLT`: long-term U.S. Treasury bond exposure
- `GLD`: gold exposure

These assets provide a compact universe for observing how return, volatility,
and diversification interact.

## 1. Market Data

The pipeline starts with `fetch_price_data`, which downloads historical adjusted
close prices and returns a clean DataFrame indexed by date.

Prices are the raw input for the rest of the workflow. No optimization happens
at this stage.

## 2. Returns

`compute_daily_returns` converts prices into daily percentage returns.

`compute_annualized_returns` estimates expected annual returns by averaging
daily returns and scaling by the number of trading days in a year.

Expected returns become the reward estimate used by the optimizer.

## 3. Covariance

`compute_covariance_matrix` estimates how asset returns vary together.

The covariance matrix is annualized and becomes the risk estimate used by both
the Markowitz optimizer and efficient frontier generator.

## 4. Markowitz Optimization

`optimize_portfolio` chooses long-only weights that sum to one.

The objective rewards expected return and penalizes covariance-based risk. The
result is a portfolio allocation indexed by ticker.

## 5. Portfolio Metrics

The optimized weights are evaluated with:

- expected portfolio return
- portfolio volatility
- Sharpe ratio

These metrics summarize the optimizer's result in portfolio-level terms.

## 6. Efficient Frontier

`generate_efficient_frontier` solves a sequence of minimum-variance optimization
problems across different target returns.

The resulting frontier shows how expected return and volatility trade off across
many feasible portfolios. Plotting the frontier makes it easier to compare the
single optimized portfolio against the broader opportunity set.

## What the Demo Does Not Include

The Phase 1 demo is classical only. It does not include QUBO formulation,
quantum optimization, or quantum solver logic.
