# Brute-Force QUBO Solving

Brute-force QUBO solving means checking every possible binary selection and
choosing the one with the lowest objective value.

For a QUBO written as:

```text
minimize x.T Q x
```

the brute-force solver tries every possible binary vector `x`, evaluates the
objective, and keeps the best result.

## Exhaustive Search

Exhaustive search is the most direct way to solve a binary optimization problem.
For each variable, there are two choices:

```text
0 or 1
```

For `n` binary variables, the solver enumerates:

```text
2^n
```

possible selections.

## Why It Guarantees the Global Optimum

Because brute force evaluates every possible solution, it cannot miss the best
one. The lowest objective value found across all combinations is the global
minimum for that QUBO.

This makes brute force a useful reference method. It is not clever, but it is
complete.

## Why It Scales Poorly

The weakness is that the number of combinations grows exponentially.

For example:

```text
10 assets -> 1,024 selections
20 assets -> 1,048,576 selections
30 assets -> 1,073,741,824 selections
```

This is the `2^n` explosion. Each added asset doubles the search space, so brute
force quickly becomes too slow for larger portfolios.

## Why It Helps Validate QUBO Compilers

Even though brute force does not scale, it is very useful while building the
project.

A QUBO compiler converts a portfolio objective into matrix form. A small mistake
in a diagonal coefficient, off-diagonal coefficient, or penalty term can change
which portfolio appears optimal.

For small examples, brute force can verify that:

- the QUBO objective evaluates as expected
- the best binary selection satisfies the intended constraints
- the penalty term discourages invalid selections
- later approximate solvers can be compared against the known optimum

This makes brute force a testing tool before introducing QAOA or other quantum
optimization methods.
