# QUBO Visualization

QUBO matrices can be hard to inspect as raw numbers. Visualization helps reveal
whether the matrix structure matches the intended optimization problem.

## QUBO Heatmaps

A QUBO heatmap shows the matrix coefficients as colors. Large positive and
negative values become easier to see than they are in a table.

This is useful for checking:

- whether penalty coefficients dominate the matrix as intended
- whether covariance terms appear in the expected locations
- whether ticker labels stayed aligned across rows and columns
- whether the matrix has unexpected asymmetry or missing structure

## Diagonal vs Off-Diagonal Terms

The diagonal entries of a QUBO matrix represent single-variable terms. In the
portfolio model, these include each asset's return contribution and part of the
cardinality penalty.

Off-diagonal entries represent pairwise interactions. In the portfolio model,
these include covariance effects between two selected assets and the pairwise
part of the selection-count penalty.

Reading the diagonal and off-diagonal regions separately helps debug whether the
return, risk, and constraint pieces were compiled into the right places.

## Solution Landscapes

A solution landscape plots candidate binary solutions against their objective
values.

For small QUBO problems, every bitstring can be evaluated. The landscape then
shows which selections are low-cost, which are high-cost, and how sharply the
objective separates good candidates from poor ones.

The `num_selected` value adds another useful layer. It helps reveal whether the
penalty term is pushing invalid selections away from the best part of the
landscape.

## Why Visualization Helps Debug QUBO Formulations

QUBO formulation bugs often come from coefficient placement, scaling, or missing
penalty terms. These problems can be subtle in code because the final output is
just a matrix.

Heatmaps and solution landscapes provide quick sanity checks before moving to
more advanced solvers. If the heatmap structure looks wrong or the best
solutions violate the intended selection count, the issue should be fixed before
introducing Ising conversion or QAOA.
