# Efficient Frontier

The efficient frontier is the set of portfolios that offer the lowest risk for
a given target return, or equivalently the highest return for a given level of
risk.

It is a way to visualize the tradeoff that sits at the heart of portfolio
optimization: more expected return usually requires accepting more uncertainty.

## Intuitive View

Imagine generating many portfolios from the same assets. Some portfolios are
clearly worse than others: they may have lower expected return and higher risk.
Those portfolios are inefficient.

The efficient frontier keeps only the portfolios that are not obviously
dominated. Each point on the curve represents a portfolio where, for that level
of expected return, the optimizer has found the minimum estimated volatility.

## Target Return Optimization

One way to generate the frontier is to choose a sequence of target returns and
solve a separate optimization problem for each target.

For each target return, the optimizer minimizes portfolio variance:

```text
minimize w^T * Sigma * w
```

Subject to:

```text
w^T * mu >= target return
sum(w) = 1
w_i >= 0 for every asset
```

Here:

- `w` is the vector of portfolio weights
- `mu` is the vector of expected returns
- `Sigma` is the covariance matrix

The result is a portfolio that reaches at least the target return while taking
as little covariance-estimated risk as possible.

## Risk-Return Tradeoff

The frontier makes the risk-return tradeoff visible.

Lower target returns often allow the optimizer to choose lower-volatility
assets, producing portfolios with less risk. Higher target returns may require
larger allocations to higher-return assets, which can increase volatility.

This tradeoff is not always linear because covariance matters. A high-return
asset may be useful if it diversifies well with the rest of the portfolio, while
a moderate-return asset may add more risk if it moves closely with existing
holdings.

## Why the Curve Matters

The efficient frontier helps compare portfolio choices.

Instead of looking at one optimized portfolio in isolation, the frontier shows a
range of reasonable portfolios across different return goals. This helps answer
questions such as:

- How much additional volatility is required to seek a higher return?
- Which target return produces the best Sharpe ratio?
- Where does increasing risk stop adding enough expected return?

In portfolio selection, the final choice depends on investor preferences. A more
risk-tolerant investor may choose a point farther up the frontier. A more
risk-sensitive investor may choose a lower-volatility point.
