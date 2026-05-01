# Visualization

Visualization helps turn portfolio data into patterns that are easier to
inspect. Tables are precise, but plots make relationships, outliers, and
tradeoffs visible quickly.

## Price History Plot

A price history plot shows how each asset's price has moved over time.

This is useful for checking:

- whether data loaded correctly
- whether an asset has unusual jumps or missing regions
- how assets behaved during different market regimes

Price history is not enough to optimize a portfolio, but it is a helpful first
sanity check before computing returns.

## Correlation Heatmap

A correlation heatmap shows how strongly assets move together on a standardized
scale from `-1` to `1`.

This is useful for understanding diversification. Assets with high positive
correlation tend to move in the same direction, while assets with low or
negative correlation can provide more diversification benefit.

Correlation is easier to read than covariance because it is unitless and bounded.
The heatmap makes relationships across many asset pairs visible at once.

## Allocation Bar Chart

An allocation bar chart shows the portfolio weight assigned to each asset.

This is useful because optimizer output can otherwise be abstract. The chart
makes it clear which assets dominate the portfolio, which assets receive small
allocations, and whether the resulting portfolio is concentrated or diversified.

## Efficient Frontier Plot

The efficient frontier plot shows portfolios by expected return and volatility.

This is useful for visualizing the risk-return tradeoff. Each point represents a
portfolio generated under a target return. Coloring the points by Sharpe ratio
adds another layer: it highlights which portfolios offer more excess return per
unit of risk.

The curve helps compare a single optimized portfolio against a broader set of
feasible choices.

## Why Visualization Matters

Portfolio analysis combines many inputs: prices, returns, covariance,
optimization weights, and performance metrics. Visualizations help connect those
pieces into a coherent picture.

Good plots do not replace the calculations. They make the calculations easier to
question, explain, and compare.
