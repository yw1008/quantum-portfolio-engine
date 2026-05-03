# Backtesting

Backtesting means applying a portfolio strategy to historical return data to see
how it would have behaved over time. In this project, the first backtesting step
is intentionally simple: combine daily asset returns with a fixed vector of
portfolio weights, then inspect the resulting portfolio return path.

This does not prove a strategy will work in the future. It gives a disciplined
way to test assumptions, compare approaches, and find obvious weaknesses before
adding more complex optimization methods.

## Portfolio Returns

A portfolio return is the weighted sum of the returns of the assets inside the
portfolio.

For example, if a portfolio is 60% in one asset and 40% in another, the daily
portfolio return is:

```text
(0.60 * asset_a_return) + (0.40 * asset_b_return)
```

The weights must line up with the same tickers used in the return data. Aligning
by ticker label matters because column order can change, especially as data moves
between notebooks, CSV files, and analysis functions.

## Cumulative Returns

Cumulative returns show how the portfolio grows or shrinks over time after
compounding each daily return.

If a portfolio gains 10% and then loses 5%, the total return is not 5%. The
second return applies to the already-changed portfolio value:

```text
(1 + 0.10) * (1 - 0.05) - 1 = 0.045
```

That means the cumulative return is 4.5%.

## Drawdown

Drawdown measures how far the portfolio has fallen from its previous peak.

If a portfolio grows to a cumulative value of 1.20 and later falls to 1.08, the
drawdown is:

```text
1.08 / 1.20 - 1 = -0.10
```

That is a 10% decline from the previous high. Drawdown is useful because two
strategies can have similar total returns but very different paths. A smoother
strategy may be easier to hold through volatile periods.

## Max Drawdown

Max drawdown is the worst drawdown observed during the backtest. It captures the
largest peak-to-trough loss in the period being tested.

For risk analysis, max drawdown is often more intuitive than volatility. It
answers a practical question: how painful was the worst historical decline?

## Why Backtesting Matters Before QUBO/QAOA Benchmarking

QUBO and QAOA methods should eventually be compared against classical portfolio
approaches using clear, consistent metrics. Backtesting provides that foundation.

Before introducing quantum optimization, the project needs a classical baseline
that can answer:

- what returns did the portfolio realize?
- how did those returns compound over time?
- how deep were the losses from prior peaks?
- what was the worst historical decline?

Without these utilities, QUBO or QAOA results would be hard to judge. A quantum
optimizer may produce a different set of weights, but those weights still need to
be evaluated through the same realized return, cumulative return, drawdown, and
max drawdown workflow.
