# Covariance

Covariance describes how two assets tend to move together. If two assets often
rise and fall at the same time, their covariance is positive. If one tends to
rise when the other falls, their covariance is negative. If their movements do
not have a consistent relationship, their covariance is close to zero.

## Intuitive View

Portfolio risk is not only about how risky each asset is on its own. It also
depends on how the assets behave relative to each other.

For example, two volatile assets can still reduce portfolio risk if they do not
move together. On the other hand, two assets that look different but usually move
in the same direction may provide less diversification than expected.

Covariance helps measure this joint movement.

## Mathematical View

For two assets \(X\) and \(Y\), covariance is:

```text
cov(X, Y) = E[(X - E[X]) * (Y - E[Y])]
```

Using return observations, this is commonly estimated as:

```text
cov(X, Y) = sum((x_i - mean(X)) * (y_i - mean(Y))) / (n - 1)
```

Where:

- `x_i` is one observed return for asset `X`
- `y_i` is one observed return for asset `Y`
- `mean(X)` and `mean(Y)` are average returns
- `n` is the number of paired return observations

A covariance matrix applies this calculation to every pair of assets. The
diagonal contains each asset's variance, because each asset is being compared
with itself.

## Annualized Covariance

Daily covariance is measured using daily returns. Portfolio analysis often works
with annualized return and risk estimates, so daily covariance is scaled to an
annual basis.

Assuming daily returns are measured across approximately 252 trading days in a
year:

```text
annualized covariance = daily covariance * 252
```

This works because variance scales linearly with time under the common
simplifying assumption that daily returns are independent and identically
distributed. Covariance follows the same scaling relationship because it is a
joint measure of return variation.

## Covariance vs Correlation

Covariance and correlation both describe how assets move together, but they are
not the same.

Covariance keeps the original scale of the returns. Its magnitude depends on the
volatility of the assets being compared, so it can be harder to interpret
directly.

Correlation standardizes covariance into a unitless value between `-1` and `1`:

```text
correlation(X, Y) = covariance(X, Y) / (std(X) * std(Y))
```

Correlation is easier to compare across asset pairs. Covariance is more directly
useful in portfolio optimization because it preserves the scale of risk.

## Why Covariance Matters for Portfolio Optimization

Portfolio optimization needs to estimate both expected return and risk. The
covariance matrix is the core input for estimating portfolio risk.

For a portfolio with weights `w` and covariance matrix `Sigma`, portfolio
variance is:

```text
portfolio variance = w^T * Sigma * w
```

This equation combines individual asset risks with the relationships between
assets. Because of that, covariance is what lets optimization models reason
about diversification instead of treating each asset in isolation.
