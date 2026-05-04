# QAOA Simulation

Local QAOA simulation runs a QAOA circuit on a classical simulator instead of
real quantum hardware. This lets the project inspect measured bitstrings and
probabilities before adding parameter optimization or IBM hardware execution.

## Shot-Based Simulation

A single circuit measurement produces one bitstring. A shot-based simulation runs
the same circuit many times.

For example, with `1024` shots, the simulator samples the circuit 1024 times and
counts how often each bitstring appears.

The result is not just one answer. It is a distribution over candidate portfolio
selections.

## Bitstrings

A bitstring is a string of measured 0 and 1 values.

With one qubit per asset, each bit maps to one ticker in ticker order:

```text
1 = selected
0 = not selected
```

For tickers:

```text
SPY, QQQ, VIG
```

the bitstring:

```text
101
```

means select `SPY`, do not select `QQQ`, and select `VIG`.

## Probabilities

The probability for a bitstring is:

```text
count / shots
```

If a bitstring appears 256 times in 1024 shots, its estimated probability is
25%.

Higher-probability bitstrings are the selections the current QAOA parameters are
more likely to produce. Before optimization, those probabilities may not yet
favor the best portfolio.

## Mapping Measurements to Portfolio Selections

Measured bitstrings are converted into `pandas.Series` selections indexed by
ticker. This preserves the same portfolio representation used by the QUBO,
brute-force, and backtesting utilities.

The local simulation step therefore connects quantum circuit measurements back
to portfolio analysis:

- build a QAOA circuit from Ising coefficients
- simulate many measurement shots
- collect bitstring counts and probabilities
- convert promising bitstrings into binary asset selections
- evaluate or backtest those selections with existing project tools

This step does not optimize QAOA parameters and does not run real IBM hardware.
