# QUBO Pipeline Demo

The Phase 2 QUBO pipeline connects the classical portfolio data workflow to the
binary optimization tools needed before quantum algorithms are introduced.

## Full Phase 2 Pipeline

The pipeline starts with the same market inputs used in the classical portfolio
work:

- fetch historical prices
- compute daily returns
- compute annualized expected returns
- compute the annualized covariance matrix

Those inputs define the binary portfolio selection problem. Each asset receives a
binary variable:

```text
1 = selected
0 = not selected
```

The QUBO compiler then combines three pieces:

- a risk term from covariance
- a return term from expected returns
- a penalty term that enforces selecting exactly `K` assets

The brute-force solver checks every binary selection for small asset universes.
This gives a known best solution and objective value, which is useful for
validating the QUBO formulation.

Visualization adds two sanity checks. The QUBO heatmap shows whether matrix
coefficients landed in sensible diagonal and off-diagonal locations. The solution
landscape shows how objective values vary across candidate bitstrings.

## QUBO to Ising

After the QUBO is built, it can be converted to Ising form using:

```text
x_i = (1 - z_i) / 2
```

The QUBO objective:

```text
x.T Q x
```

becomes an Ising energy:

```text
sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + offset
```

The `h` terms are individual spin coefficients, the `J` terms are pairwise spin
couplings, and the offset preserves exact energy equality.

## Connection to Later QAOA

QAOA has not been implemented yet, but this pipeline prepares the inputs it will
eventually need.

QAOA works with a Hamiltonian, which can be built from the Ising coefficients.
Before building that quantum layer, the project needs confidence that:

- the market data inputs are correct
- the QUBO matrix represents the intended portfolio objective
- brute force finds the expected small-system optimum
- QUBO and Ising energies match for the same selection

The demo notebook verifies those steps in order. That makes the future QAOA work
an extension of a tested pipeline rather than a separate experiment.
