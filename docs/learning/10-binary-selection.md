# Binary Selection

Binary portfolio selection turns portfolio construction into a choose-or-skip
problem. Each asset receives a binary decision:

```text
1 = include the asset
0 = exclude the asset
```

This is different from continuous portfolio optimization, where the optimizer can
assign any allowed weight, such as 12.5% to one asset and 37.5% to another.

## Continuous vs Binary Optimization

In continuous optimization, the decision variables are portfolio weights. The
optimizer directly decides how much capital to allocate to each asset, usually
subject to constraints such as long-only weights and total weight equal to 1.

In binary optimization, the decision variables are selections. The optimizer
decides which assets belong in the portfolio. A separate rule then converts the
selected assets into weights.

This project starts with a simple weighting rule so the binary model stays easy
to inspect before introducing QUBO or QAOA.

## Equal Weighting Among Selected Assets

After the binary selection is made, selected assets are normalized into equal
weights.

For example:

```text
AAPL = 1
MSFT = 0
BND  = 1
```

This becomes:

```text
AAPL = 0.50
MSFT = 0.00
BND  = 0.50
```

Equal weighting keeps the first binary model focused on asset inclusion. It also
makes results easier to compare because the score depends on which assets were
chosen, not on a second weighting optimizer.

## Portfolio Score

The binary selection score follows the same mean-variance idea used in classical
portfolio optimization:

```text
expected_return - risk_aversion * portfolio_variance
```

Higher expected return improves the score. Higher variance lowers the score. The
`risk_aversion` value controls how strongly risk is penalized.

## Why Binary Encoding Is Quantum-Friendly

Binary variables map naturally to quantum computation because qubits are measured
as binary outcomes. A candidate portfolio can be represented as a bitstring where
each bit corresponds to one asset.

For three assets, the bitstring `101` means include the first and third assets
and exclude the second asset. This makes binary portfolio selection a natural
bridge between classical portfolio analysis and quantum optimization methods.

## Connection to Combinatorial Optimization

Binary portfolio selection is a combinatorial optimization problem. With each
asset either included or excluded, the number of possible portfolios grows
quickly as more assets are added.

For `n` assets, there are:

```text
2^n
```

possible selections before constraints are applied.

This combinatorial structure is why binary portfolio selection is a useful
stepping stone toward QUBO and QAOA. QUBO will later express the objective as a
quadratic function of binary variables, and QAOA will search for strong binary
solutions. At this step, the project only defines and tests the binary scoring
model.
