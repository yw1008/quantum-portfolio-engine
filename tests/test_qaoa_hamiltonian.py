from __future__ import annotations

import pandas as pd
import pytest
from qiskit.quantum_info import SparsePauliOp

from backend.quantum.qaoa_hamiltonian import (
    build_cost_hamiltonian,
    build_mixer_hamiltonian,
)


def test_build_cost_hamiltonian_maps_h_and_j_to_pauli_terms():
    h = pd.Series({"AAPL": 0.50, "MSFT": -0.25, "BND": 0.10}, name="h")
    J = pd.DataFrame(
        [
            [0.0, 0.20, -0.05],
            [0.20, 0.0, 0.30],
            [-0.05, 0.30, 0.0],
        ],
        index=h.index,
        columns=h.index,
    )

    hamiltonian = build_cost_hamiltonian(h, J)

    assert isinstance(hamiltonian, SparsePauliOp)
    assert _operator_terms(hamiltonian) == {
        "ZII": pytest.approx(0.50),
        "IZI": pytest.approx(-0.25),
        "IIZ": pytest.approx(0.10),
        "ZZI": pytest.approx(0.20),
        "ZIZ": pytest.approx(-0.05),
        "IZZ": pytest.approx(0.30),
    }


def test_build_cost_hamiltonian_preserves_h_ticker_order_when_j_is_reordered():
    h = pd.Series({"AAPL": 0.50, "MSFT": -0.25}, name="h")
    J = pd.DataFrame(
        [[0.0, 0.20], [0.20, 0.0]],
        index=["MSFT", "AAPL"],
        columns=["MSFT", "AAPL"],
    )

    hamiltonian = build_cost_hamiltonian(h, J)

    assert _operator_terms(hamiltonian) == {
        "ZI": pytest.approx(0.50),
        "IZ": pytest.approx(-0.25),
        "ZZ": pytest.approx(0.20),
    }


def test_build_cost_hamiltonian_returns_zero_operator_for_zero_coefficients():
    h = pd.Series({"AAPL": 0.0}, name="h")
    J = pd.DataFrame([[0.0]], index=["AAPL"], columns=["AAPL"])

    hamiltonian = build_cost_hamiltonian(h, J)

    assert _operator_terms(hamiltonian) == {"I": pytest.approx(0.0)}


def test_build_mixer_hamiltonian_returns_sum_of_x_terms():
    mixer = build_mixer_hamiltonian(3)

    assert isinstance(mixer, SparsePauliOp)
    assert _operator_terms(mixer) == {
        "XII": pytest.approx(1.0),
        "IXI": pytest.approx(1.0),
        "IIX": pytest.approx(1.0),
    }


def test_build_cost_hamiltonian_raises_value_error_for_empty_h():
    J = pd.DataFrame([[0.0]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="h must not be empty"):
        build_cost_hamiltonian(pd.Series(dtype=float), J)


def test_build_cost_hamiltonian_raises_value_error_for_empty_j():
    h = pd.Series({"AAPL": 0.50}, name="h")

    with pytest.raises(ValueError, match="J must not be empty"):
        build_cost_hamiltonian(h, pd.DataFrame())


def test_build_cost_hamiltonian_raises_value_error_for_non_square_j():
    h = pd.Series({"AAPL": 0.50}, name="h")
    J = pd.DataFrame([[0.0, 0.1]], index=["AAPL"], columns=["AAPL", "MSFT"])

    with pytest.raises(ValueError, match="J must be square"):
        build_cost_hamiltonian(h, J)


def test_build_cost_hamiltonian_raises_value_error_for_mismatched_j_labels():
    h = pd.Series({"AAPL": 0.50, "MSFT": -0.25}, name="h")
    J = pd.DataFrame(
        [[0.0, 0.1], [0.1, 0.0]],
        index=["AAPL", "MSFT"],
        columns=["MSFT", "AAPL"],
    )

    with pytest.raises(ValueError, match="J must have matching row and column labels"):
        build_cost_hamiltonian(h, J)


def test_build_cost_hamiltonian_raises_value_error_for_label_mismatch():
    h = pd.Series({"AAPL": 0.50}, name="h")
    J = pd.DataFrame([[0.0]], index=["MSFT"], columns=["MSFT"])

    with pytest.raises(ValueError, match="Tickers in h and J must match"):
        build_cost_hamiltonian(h, J)


def test_build_mixer_hamiltonian_raises_value_error_for_invalid_num_qubits():
    with pytest.raises(ValueError, match="num_qubits must be positive"):
        build_mixer_hamiltonian(0)


def _operator_terms(operator: SparsePauliOp) -> dict[str, complex]:
    """Return SparsePauliOp terms as a label-to-coefficient mapping."""
    return {label: coefficient for label, coefficient in operator.to_list()}
