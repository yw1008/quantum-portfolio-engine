# Ising Conversion

QUBO and Ising formulations are two closely related ways to describe binary
optimization problems.

## QUBO vs Ising

A QUBO uses binary variables:

```text
x_i in {0, 1}
```

and an objective such as:

```text
x.T Q x
```

An Ising model uses spin variables:

```text
z_i in {-1, +1}
```

and an energy function:

```text
sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + offset
```

Both forms can represent the same optimization problem. The difference is the
variable encoding and coefficient layout.

## Why QAOA Uses Hamiltonians

QAOA works with quantum Hamiltonians. A Hamiltonian acts like an energy function
for a quantum system. The optimization goal is to find a low-energy state that
corresponds to a strong solution of the original problem.

The Ising form is close to this Hamiltonian view because spin variables map
naturally onto Pauli-Z measurements on qubits. That makes Ising conversion an
important bridge between a QUBO matrix and a later QAOA circuit.

## The Variable Mapping

The conversion uses:

```text
x_i = (1 - z_i) / 2
```

This maps spin values to binary values:

```text
z_i = +1 -> x_i = 0
z_i = -1 -> x_i = 1
```

So a selected asset in the QUBO representation corresponds to a `-1` spin in the
Ising representation.

## h, J, and Offset

The Ising model separates energy into three parts.

The `h` coefficients are linear terms. They describe how much each individual
spin contributes to the energy.

The `J` coefficients are pairwise couplings. They describe how pairs of spins
interact.

The `offset` is a constant energy shift. It does not change which spin assignment
minimizes the energy, but it is needed when comparing exact QUBO and Ising energy
values.

## Verifying Energy Equivalence

To verify a conversion, evaluate both models on matching assignments.

First choose a binary vector:

```text
x = [1, 0, 1]
```

Then convert it to spins using:

```text
z = 1 - 2x
```

which gives:

```text
z = [-1, +1, -1]
```

Now compare:

```text
QUBO energy  = x.T Q x
Ising energy = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + offset
```

If the conversion is correct, these energies match for every possible binary
assignment. This exhaustive check is practical for small examples and provides a
good test before implementing QAOA.
