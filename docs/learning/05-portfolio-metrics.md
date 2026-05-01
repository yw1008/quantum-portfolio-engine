# Portfolio Metrics

Portfolio metrics summarize the behavior of a set of asset weights. After an
optimizer chooses weights, metrics help evaluate whether the resulting portfolio
has an attractive balance of return and risk.

## Expected Portfolio Return

Expected portfolio return is the weighted average of the assets' expected
returns.

If `w` is the vector of portfolio weights and `mu` is the vector of expected
asset returns:

```text
expected portfolio return = w^T * mu
```

Each asset contributes according to both its expected return and its allocation.
An asset with a high expected return has little effect if its weight is small,
while a large allocation can strongly influence total expected return.

## Portfolio Volatility

Portfolio volatility measures the estimated uncertainty of portfolio returns. It
is the square root of portfolio variance.

Using weights `w` and covariance matrix `Sigma`:

```text
portfolio variance = w^T * Sigma * w
portfolio volatility = sqrt(w^T * Sigma * w)
```

The covariance matrix matters because total portfolio risk depends on how assets
move together, not only on each asset's standalone volatility.

## Sharpe Ratio

The Sharpe ratio measures excess return per unit of risk:

```text
Sharpe ratio = (portfolio return - risk-free rate) / portfolio volatility
```

The risk-free rate represents the return that could theoretically be earned
without taking market risk. Subtracting it focuses the metric on compensation
for risk.

A higher Sharpe ratio means the portfolio is producing more excess return for
each unit of volatility. A lower Sharpe ratio means the portfolio is taking risk
without as much return compensation.

## Why These Metrics Matter

Portfolio optimization produces weights, but weights alone do not explain the
quality of a portfolio. Metrics translate those weights into interpretable
portfolio-level quantities.

Expected return answers: how much return does the portfolio seek?

Volatility answers: how much uncertainty does the portfolio carry?

Sharpe ratio answers: how much excess return does the portfolio seek per unit of
risk?

Together, these metrics make it easier to compare portfolios, tune optimizer
settings such as risk aversion, and understand the tradeoff between return and
risk.
