"""Hamiltonian builders for QAOA portfolio experiments."""

from __future__ import annotations

import pandas as pd
from qiskit.quantum_info import SparsePauliOp


def build_cost_hamiltonian(
    h: pd.Series,
    J: pd.DataFrame,
) -> SparsePauliOp:
    """Build a Qiskit cost Hamiltonian from Ising coefficients.

    Args:
        h: Linear Ising coefficients indexed by ticker.
        J: Pairwise Ising couplings indexed and columned by ticker.

    Returns:
        SparsePauliOp containing Z and ZZ terms in ticker order.

    Raises:
        ValueError: If inputs are empty, labels do not match, or ``J`` is not
            square.
    """
    _validate_ising_coefficients(h, J)

    tickers = h.index
    terms: list[tuple[str, float]] = []

    for position, ticker in enumerate(tickers):
        coefficient = float(h.loc[ticker])
        if coefficient != 0.0:
            terms.append((_single_pauli_label("Z", position, len(tickers)), coefficient))

    aligned_J = J.loc[tickers, tickers]
    for row_position, row_ticker in enumerate(tickers):
        for column_position, column_ticker in enumerate(tickers[row_position + 1 :]):
            actual_column_position = row_position + column_position + 1
            coefficient = float(aligned_J.loc[row_ticker, column_ticker])
            if coefficient != 0.0:
                terms.append(
                    (
                        _pair_pauli_label(
                            "Z",
                            row_position,
                            actual_column_position,
                            len(tickers),
                        ),
                        coefficient,
                    )
                )

    return _sparse_pauli_op_from_terms(terms, len(tickers))


def build_mixer_hamiltonian(num_qubits: int) -> SparsePauliOp:
    """Build the standard QAOA mixer Hamiltonian.

    Args:
        num_qubits: Number of qubits in the mixer.

    Returns:
        SparsePauliOp representing ``sum_i X_i``.

    Raises:
        ValueError: If ``num_qubits`` is not positive.
    """
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive.")

    terms = [
        (_single_pauli_label("X", position, num_qubits), 1.0)
        for position in range(num_qubits)
    ]
    return SparsePauliOp.from_list(terms)


def _validate_ising_coefficients(h: pd.Series, J: pd.DataFrame) -> None:
    """Validate Ising coefficient labels and shapes."""
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


def _single_pauli_label(operator: str, position: int, num_qubits: int) -> str:
    """Build a single-qubit Pauli label in ticker order."""
    label = ["I"] * num_qubits
    label[position] = operator

    return "".join(label)


def _pair_pauli_label(
    operator: str,
    first_position: int,
    second_position: int,
    num_qubits: int,
) -> str:
    """Build a two-qubit Pauli label in ticker order."""
    label = ["I"] * num_qubits
    label[first_position] = operator
    label[second_position] = operator

    return "".join(label)


def _sparse_pauli_op_from_terms(
    terms: list[tuple[str, float]],
    num_qubits: int,
) -> SparsePauliOp:
    """Create a SparsePauliOp, preserving a zero Hamiltonian when needed."""
    if not terms:
        return SparsePauliOp.from_list([("I" * num_qubits, 0.0)])

    return SparsePauliOp.from_list(terms)
