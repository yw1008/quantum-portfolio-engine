# QUBO Formulation

QUBO stands for Quadratic Unconstrained Binary Optimization. It is a way to write
an optimization problem using binary variables and a quadratic objective.

In matrix form, a QUBO is usually written as:

```text
minimize x.T Q x
```

where `x` is a vector of binary variables and `Q` is a matrix of coefficients.

## Why Binary Variables Matter

Quantum optimization methods often work naturally with binary decisions. A qubit
is measured as one of two outcomes, and a collection of qubits can represent a
bitstring.

For portfolio selection, each bit can represent whether an asset is included:

```text
x_i = 1 means select asset i
x_i = 0 means exclude asset i
```

This makes binary portfolio selection a useful bridge between portfolio
optimization and quantum algorithms.

## Equal-Weight Selection

At this stage, selected assets are weighted equally. If exactly `K` assets are
selected, each selected asset receives weight:

```text
1 / K
```

The binary vector decides which assets are included. The equal-weight rule turns
that selection into portfolio weights.

## Risk Term

The portfolio variance for equal-weight selected assets is:

```text
(1 / K^2) * x.T covariance_matrix x
```

The QUBO is a minimization problem, so risk is added to the objective:

```text
risk_aversion * (1 / K^2) * x.T covariance_matrix x
```

Higher covariance values increase the objective and make a selection less
attractive.

## Return Term

Expected return should improve the portfolio, so it is subtracted in the
minimization objective.

For equal-weight selected assets:

```text
(1 / K) * expected_returns.T x
```

The minimized return term is:

```text
-(1 / K) * expected_returns.T x
```

Because binary variables satisfy `x_i^2 = x_i`, this linear return term can be
stored on the diagonal of the QUBO matrix.

## Penalty Term

The model requires exactly `K` selected assets. The constraint is:

```text
sum_i x_i = K
```

QUBO objectives are unconstrained, so the constraint is moved into the objective
as a penalty:

```text
penalty_strength * (sum_i x_i - K)^2
```

This value is zero when exactly `K` assets are selected and positive otherwise.

Expanding the penalty:

```text
penalty_strength * (sum_i x_i - K)^2
= penalty_strength * ((sum_i x_i)^2 - 2K * sum_i x_i + K^2)
```

For binary variables:

```text
(sum_i x_i)^2 = sum_i x_i + 2 * sum_{i<j} x_i x_j
```

So the penalty becomes:

```text
penalty_strength * (
    (1 - 2K) * sum_i x_i
    + 2 * sum_{i<j} x_i x_j
    + K^2
)
```

The constant `penalty_strength * K^2` does not change which portfolio minimizes
the objective, so the QUBO matrix can omit it.

## How Q Represents the Objective

The matrix `Q` stores the coefficients of the binary objective.

Diagonal entries represent single-variable terms such as:

```text
x_i
```

Off-diagonal entries represent pairwise terms such as:

```text
x_i * x_j
```

This project returns a symmetric `Q` matrix evaluated as:

```text
x.T Q x
```

Because symmetric matrix multiplication counts each off-diagonal pair twice, each
off-diagonal entry stores half of the full pair coefficient. The result is a
labeled matrix that can be inspected with the same ticker labels used by the
portfolio data.
