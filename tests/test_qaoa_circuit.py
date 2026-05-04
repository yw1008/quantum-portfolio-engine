from __future__ import annotations

import pandas as pd
import pytest
from qiskit import QuantumCircuit

from backend.quantum.qaoa_circuit import build_qaoa_circuit


def test_build_qaoa_circuit_creates_expected_one_layer_circuit():
    h = pd.Series({"AAPL": 0.50, "MSFT": -0.25, "BND": 0.0}, name="h")
    J = pd.DataFrame(
        [
            [0.0, 0.20, -0.05],
            [0.20, 0.0, 0.0],
            [-0.05, 0.0, 0.0],
        ],
        index=h.index,
        columns=h.index,
    )

    circuit = build_qaoa_circuit(
        h,
        J,
        gamma=0.70,
        beta=0.30,
        add_measurements=False,
    )

    assert isinstance(circuit, QuantumCircuit)
    assert circuit.num_qubits == 3
    assert circuit.num_clbits == 0
    assert circuit.metadata == {"tickers": ["AAPL", "MSFT", "BND"]}
    assert circuit.count_ops() == {"h": 3, "rz": 2, "rzz": 2, "rx": 3}


def test_build_qaoa_circuit_uses_expected_gate_angles():
    h = pd.Series({"AAPL": 0.50, "MSFT": -0.25}, name="h")
    J = pd.DataFrame(
        [[0.0, 0.20], [0.20, 0.0]],
        index=h.index,
        columns=h.index,
    )

    circuit = build_qaoa_circuit(
        h,
        J,
        gamma=0.70,
        beta=0.30,
        add_measurements=False,
    )
    operations = list(circuit.data)

    assert operations[0].operation.name == "h"
    assert operations[1].operation.name == "h"
    assert operations[2].operation.name == "rz"
    assert float(operations[2].operation.params[0]) == pytest.approx(0.70)
    assert operations[3].operation.name == "rz"
    assert float(operations[3].operation.params[0]) == pytest.approx(-0.35)
    assert operations[4].operation.name == "rzz"
    assert float(operations[4].operation.params[0]) == pytest.approx(0.28)
    assert operations[5].operation.name == "rx"
    assert float(operations[5].operation.params[0]) == pytest.approx(0.60)
    assert operations[6].operation.name == "rx"
    assert float(operations[6].operation.params[0]) == pytest.approx(0.60)


def test_build_qaoa_circuit_adds_measurements_by_default():
    h = pd.Series({"AAPL": 0.50, "MSFT": -0.25}, name="h")
    J = pd.DataFrame(
        [[0.0, 0.20], [0.20, 0.0]],
        index=h.index,
        columns=h.index,
    )

    circuit = build_qaoa_circuit(h, J, gamma=0.70, beta=0.30)

    assert circuit.num_clbits == 2
    assert circuit.count_ops()["measure"] == 2


def test_build_qaoa_circuit_repeats_layers():
    h = pd.Series({"AAPL": 0.50, "MSFT": -0.25}, name="h")
    J = pd.DataFrame(
        [[0.0, 0.20], [0.20, 0.0]],
        index=h.index,
        columns=h.index,
    )

    circuit = build_qaoa_circuit(
        h,
        J,
        gamma=0.70,
        beta=0.30,
        num_layers=2,
        add_measurements=False,
    )

    assert circuit.count_ops() == {"h": 2, "rz": 4, "rzz": 2, "rx": 4}


def test_build_qaoa_circuit_aligns_j_to_h_order():
    h = pd.Series({"AAPL": 0.50, "MSFT": -0.25}, name="h")
    J = pd.DataFrame(
        [[0.0, 0.20], [0.20, 0.0]],
        index=["MSFT", "AAPL"],
        columns=["MSFT", "AAPL"],
    )

    circuit = build_qaoa_circuit(
        h,
        J,
        gamma=0.70,
        beta=0.30,
        add_measurements=False,
    )

    assert circuit.metadata == {"tickers": ["AAPL", "MSFT"]}
    rzz_operation = [item.operation for item in circuit.data if item.operation.name == "rzz"][0]
    assert float(rzz_operation.params[0]) == pytest.approx(0.28)


def test_build_qaoa_circuit_raises_value_error_for_empty_h():
    J = pd.DataFrame([[0.0]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="h must not be empty"):
        build_qaoa_circuit(pd.Series(dtype=float), J, gamma=0.1, beta=0.2)


def test_build_qaoa_circuit_raises_value_error_for_empty_j():
    h = pd.Series({"AAPL": 0.50}, name="h")

    with pytest.raises(ValueError, match="J must not be empty"):
        build_qaoa_circuit(h, pd.DataFrame(), gamma=0.1, beta=0.2)


def test_build_qaoa_circuit_raises_value_error_for_non_square_j():
    h = pd.Series({"AAPL": 0.50}, name="h")
    J = pd.DataFrame([[0.0, 0.1]], index=["AAPL"], columns=["AAPL", "MSFT"])

    with pytest.raises(ValueError, match="J must be square"):
        build_qaoa_circuit(h, J, gamma=0.1, beta=0.2)


def test_build_qaoa_circuit_raises_value_error_for_mismatched_j_labels():
    h = pd.Series({"AAPL": 0.50, "MSFT": -0.25}, name="h")
    J = pd.DataFrame(
        [[0.0, 0.1], [0.1, 0.0]],
        index=["AAPL", "MSFT"],
        columns=["MSFT", "AAPL"],
    )

    with pytest.raises(ValueError, match="J must have matching row and column labels"):
        build_qaoa_circuit(h, J, gamma=0.1, beta=0.2)


def test_build_qaoa_circuit_raises_value_error_for_label_mismatch():
    h = pd.Series({"AAPL": 0.50}, name="h")
    J = pd.DataFrame([[0.0]], index=["MSFT"], columns=["MSFT"])

    with pytest.raises(ValueError, match="Tickers in h and J must match"):
        build_qaoa_circuit(h, J, gamma=0.1, beta=0.2)


@pytest.mark.parametrize("gamma,beta,error", [("bad", 0.2, "gamma"), (0.1, "bad", "beta")])
def test_build_qaoa_circuit_raises_value_error_for_non_numeric_angles(
    gamma: object,
    beta: object,
    error: str,
):
    h = pd.Series({"AAPL": 0.50}, name="h")
    J = pd.DataFrame([[0.0]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match=f"{error} must be numeric"):
        build_qaoa_circuit(h, J, gamma=gamma, beta=beta)


def test_build_qaoa_circuit_raises_value_error_for_invalid_num_layers():
    h = pd.Series({"AAPL": 0.50}, name="h")
    J = pd.DataFrame([[0.0]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="num_layers must be positive"):
        build_qaoa_circuit(h, J, gamma=0.1, beta=0.2, num_layers=0)
