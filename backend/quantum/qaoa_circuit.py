"""QAOA circuit builder for Ising portfolio Hamiltonians."""

from __future__ import annotations

from numbers import Real

import pandas as pd
from qiskit import QuantumCircuit


def build_qaoa_circuit(
    h: pd.Series,
    J: pd.DataFrame,
    gamma: float,
    beta: float,
    num_layers: int = 1,
    add_measurements: bool = True,
) -> QuantumCircuit:
    """Build a fixed-parameter QAOA circuit from Ising coefficients.

    Args:
        h: Linear Ising coefficients indexed by ticker.
        J: Pairwise Ising couplings indexed and columned by ticker.
        gamma: Cost-layer angle.
        beta: Mixer-layer angle.
        num_layers: Number of repeated QAOA layers.
        add_measurements: Whether to measure all qubits at the end.

    Returns:
        QuantumCircuit with one qubit per asset in ``h.index`` order.

    Raises:
        ValueError: If inputs are empty, labels do not match, angles are not
            numeric, or ``num_layers`` is not positive.
    """
    _validate_inputs(h, J, gamma, beta, num_layers)

    tickers = h.index
    circuit = QuantumCircuit(len(tickers), name="qaoa_portfolio")
    circuit.metadata = {"tickers": list(tickers)}

    for qubit in range(len(tickers)):
        circuit.h(qubit)

    aligned_J = J.loc[tickers, tickers]
    for _ in range(num_layers):
        _apply_cost_layer(circuit, h, aligned_J, float(gamma))
        _apply_mixer_layer(circuit, float(beta))

    if add_measurements:
        circuit.measure_all()

    return circuit


def _apply_cost_layer(
    circuit: QuantumCircuit,
    h: pd.Series,
    J: pd.DataFrame,
    gamma: float,
) -> None:
    """Apply one QAOA cost layer."""
    tickers = h.index

    for qubit, ticker in enumerate(tickers):
        coefficient = float(h.loc[ticker])
        if coefficient != 0.0:
            circuit.rz(2.0 * gamma * coefficient, qubit)

    for row_position, row_ticker in enumerate(tickers):
        for column_position, column_ticker in enumerate(tickers[row_position + 1 :]):
            actual_column_position = row_position + column_position + 1
            coefficient = float(J.loc[row_ticker, column_ticker])
            if coefficient != 0.0:
                _apply_rzz(
                    circuit,
                    2.0 * gamma * coefficient,
                    row_position,
                    actual_column_position,
                )


def _apply_mixer_layer(circuit: QuantumCircuit, beta: float) -> None:
    """Apply one standard X-mixer layer."""
    for qubit in range(circuit.num_qubits):
        circuit.rx(2.0 * beta, qubit)


def _apply_rzz(
    circuit: QuantumCircuit,
    angle: float,
    first_qubit: int,
    second_qubit: int,
) -> None:
    """Apply RZZ directly when available, otherwise decompose it."""
    if hasattr(circuit, "rzz"):
        circuit.rzz(angle, first_qubit, second_qubit)
        return

    circuit.cx(first_qubit, second_qubit)
    circuit.rz(angle, second_qubit)
    circuit.cx(first_qubit, second_qubit)


def _validate_inputs(
    h: pd.Series,
    J: pd.DataFrame,
    gamma: float,
    beta: float,
    num_layers: int,
) -> None:
    """Validate QAOA circuit inputs."""
    if h.empty:
        raise ValueError("h must not be empty.")
    if J.empty:
        raise ValueError("J must not be empty.")
    if len(J.index) != len(J.columns):
        raise ValueError("J must be square.")
    if list(J.index) != list(J.columns):
        raise ValueError("J must have matching row and column labels.")
    if set(h.index) != set(J.index):
        raise ValueError("Tickers in h and J must match.")
    if not isinstance(gamma, Real):
        raise ValueError("gamma must be numeric.")
    if not isinstance(beta, Real):
        raise ValueError("beta must be numeric.")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
