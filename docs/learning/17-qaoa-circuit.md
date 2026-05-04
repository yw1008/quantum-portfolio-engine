# QAOA Circuit

The QAOA circuit turns the Ising portfolio Hamiltonian into a sequence of quantum
gates. This step builds a circuit for fixed `gamma` and `beta` values. It does
not optimize those parameters yet.

## |+> Initialization

QAOA starts by placing every qubit into the `|+>` state.

Starting from `|0>`, a Hadamard gate creates:

```text
|+> = (|0> + |1>) / sqrt(2)
```

With one qubit per asset, this creates an equal superposition over all possible
portfolio bitstrings. The circuit begins with every candidate selection present
in the quantum state.

## Cost Layer

The cost layer encodes the Ising objective.

Linear `h_i Z_i` terms become `RZ` rotations. Pairwise `J_ij Z_i Z_j` couplings
become `RZZ` rotations, or an equivalent decomposition using controlled gates
and an `RZ` rotation.

The cost layer changes phases according to portfolio quality. Bitstrings with
different Ising energies receive different phase changes.

## Mixer Layer

The mixer layer uses `RX` rotations.

The standard mixer corresponds to:

```text
sum_i X_i
```

It moves amplitude between neighboring bitstrings by rotating each qubit around
the X axis. This helps the circuit explore the binary selection space.

## Gamma and Beta

`gamma` controls the strength of the cost layer. Larger values apply stronger
phase changes from the Ising objective.

`beta` controls the strength of the mixer layer. Larger values apply stronger X
rotations and move more amplitude between candidate bitstrings.

QAOA alternates these layers:

```text
cost(gamma) -> mixer(beta)
```

With more layers, the pattern repeats. Later parameter optimization will search
for `gamma` and `beta` values that make good portfolio bitstrings more likely.

## Measurement Bitstrings

At the end of the circuit, measuring the qubits produces a bitstring.

Each bit corresponds to one asset in ticker order:

```text
1 = selected
0 = not selected
```

For example, with tickers:

```text
SPY, QQQ, VIG
```

the bitstring `101` means select `SPY` and `VIG`, and do not select `QQQ`.

This step only builds the circuit. It does not run the circuit on real IBM
hardware and does not optimize QAOA parameters.
