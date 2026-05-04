# QAOA Hamiltonian

QAOA uses Hamiltonians to describe the optimization problem and the search
process. In this project, the first QAOA-facing step is to convert Ising
coefficients into Qiskit `SparsePauliOp` objects.

## Pauli Operators

Pauli operators are basic quantum operators that act on qubits. The operators
used here are:

- `Z`, which measures or phases a qubit according to its computational-basis
  state
- `X`, which flips a qubit between basis states
- `I`, the identity operator, which leaves a qubit unchanged

Multi-qubit operators are written as strings such as:

```text
ZII
IZZ
XII
```

Each character describes the operator applied to one qubit.

## Z Terms

The Ising linear coefficients `h_i` become single-qubit `Z_i` terms.

For example, if the first asset has coefficient `h_0`, the Hamiltonian receives:

```text
h_0 * ZII
```

These terms represent the individual contribution of each spin variable to the
cost energy.

## ZZ Couplings

The Ising pairwise coefficients `J_ij` become two-qubit `Z_i Z_j` terms.

For example, a coupling between the first and third assets becomes:

```text
J_02 * ZIZ
```

These couplings represent interactions between pairs of selected or unselected
assets. In the portfolio QUBO pipeline, they come from covariance and selection
penalty terms.

## Mixer Hamiltonian

The standard QAOA mixer is:

```text
sum_i X_i
```

For three qubits, this is:

```text
XII + IXI + IIX
```

The mixer lets the quantum state move between different candidate bitstrings.
Without a mixer, the algorithm would not explore the solution space.

## Why QAOA Alternates Cost and Mixer

QAOA alternates between two operations:

- the cost Hamiltonian, which phases states according to the objective value
- the mixer Hamiltonian, which moves amplitude between candidate solutions

This alternation is the core QAOA idea. The cost Hamiltonian encodes what makes a
portfolio good or bad, while the mixer helps search across possible portfolios.

This step only builds the Hamiltonian objects. Circuit construction and parameter
optimization come later.
