# Markowitz Mean-Variance Optimization

Markowitz mean-variance optimization is a classical portfolio construction
method. It chooses asset weights by balancing expected return against estimated
risk.

The central idea is simple: a portfolio should not be judged only by how much
return it might earn. It should also be judged by how much uncertainty it takes
on to seek that return.

## Intuitive View

Each asset receives a portfolio weight. A weight says what fraction of total
capital is allocated to that asset. The optimizer searches for the set of
weights that provides the best tradeoff between reward and risk under the chosen
constraints.

Higher-return assets are attractive, but they may also be volatile or move
closely with other risky assets. Lower-return assets can still be useful if they
reduce total portfolio risk.

## Expected Return Term

The expected return term rewards portfolios with higher weighted average return.
If `mu` is the vector of expected asset returns and `w` is the vector of
portfolio weights, expected portfolio return is:

```text
expected portfolio return = mu^T * w
```

This is the sum of each asset's expected return multiplied by its portfolio
weight.

## Covariance Risk Term

The risk term uses the covariance matrix to estimate portfolio variance. If
`Sigma` is the covariance matrix, portfolio variance is:

```text
portfolio variance = w^T * Sigma * w
```

This term captures both individual asset volatility and how assets move
together. That is what allows the optimizer to account for diversification.

## Objective Function

This project uses a mean-variance objective:

```text
maximize expected return - risk_aversion * portfolio variance
```

In vector form:

```text
maximize mu^T * w - lambda * w^T * Sigma * w
```

Where `lambda` is the risk aversion parameter.

## Long-Only Constraint

A long-only portfolio requires:

```text
w_i >= 0 for every asset
```

This means the optimizer cannot short sell. Every asset weight must be zero or
positive. Long-only portfolios are easier to interpret and are common in
traditional investment settings.

## Why Weights Sum to 1

The constraint:

```text
sum(w) = 1
```

means all capital is allocated across the available assets. A weight of `0.25`
means 25% of the portfolio is invested in that asset. If all weights sum to 1,
the portfolio is fully invested with no leverage and no unallocated cash.

## Risk Aversion

Risk aversion controls how strongly the optimizer penalizes risk.

A lower risk aversion value makes expected return more important, so the
optimizer may allocate more heavily to high-return assets. A higher risk
aversion value makes portfolio variance more important, so the optimizer tends
to prefer lower-risk or more diversifying allocations.

## How cvxpy Solves the Problem

cvxpy is a convex optimization modeling library. Instead of manually deriving
the solution, the project declares:

- decision variables: portfolio weights
- objective: maximize return minus risk penalty
- constraints: long-only weights and weights summing to 1

cvxpy converts this mathematical model into a form that numerical solvers can
handle. For the Markowitz objective, the problem is a quadratic optimization
problem: the expected return term is linear, the covariance risk term is
quadratic, and the constraints are linear.

When the covariance matrix is suitable for optimization, this is a convex
problem. Convexity is important because it means a local optimum is also a
global optimum.
