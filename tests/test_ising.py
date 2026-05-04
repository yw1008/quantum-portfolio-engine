from __future__ import annotations

from itertools import product

import pandas as pd
import pytest

from backend.quantum.bruteforce_solver import evaluate_qubo
from backend.quantum.ising import evaluate_ising, qubo_to_ising
from backend.quantum.qubo import build_portfolio_qubo


def test_qubo_to_ising_returns_labeled_coefficients():
    Q = pd.DataFrame(
        [[-1.0, 0.25], [0.25, -0.50]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    h, J, offset = qubo_to_ising(Q)

    expected_h = pd.Series({"AAPL": 0.375, "MSFT": 0.125}, name="h")
    expected_J = pd.DataFrame(
        [[0.0, 0.125], [0.125, 0.0]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    pd.testing.assert_series_equal(h, expected_h)
    pd.testing.assert_frame_equal(J, expected_J)
    assert offset == pytest.approx(-0.625)


def test_qubo_to_ising_matches_qubo_energy_for_all_bitstrings():
    Q = pd.DataFrame(
        [
            [-1.0, 0.25, 0.10],
            [0.25, -0.50, 0.20],
            [0.10, 0.20, -0.25],
        ],
        index=["AAPL", "MSFT", "BND"],
        columns=["AAPL", "MSFT", "BND"],
    )
    h, J, offset = qubo_to_ising(Q)

    for values in product([0, 1], repeat=len(Q.index)):
        selection = pd.Series(values, index=Q.index)
        spins = 1 - 2 * selection

        assert evaluate_ising(spins, h, J, offset) == pytest.approx(
            evaluate_qubo(selection, Q)
        )


def test_qubo_to_ising_handles_non_symmetric_qubo_by_pair_sum():
    Q = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    h, J, offset = qubo_to_ising(Q)

    assert h["AAPL"] == pytest.approx(-1.75)
    assert h["MSFT"] == pytest.approx(-3.25)
    assert J.loc["AAPL", "MSFT"] == pytest.approx(1.25)
    assert J.loc["MSFT", "AAPL"] == pytest.approx(1.25)
    assert offset == pytest.approx(3.75)


def test_qubo_to_ising_works_with_portfolio_qubo():
    expected_returns = pd.Series({"AAPL": 0.12, "MSFT": 0.08, "BND": 0.04})
    covariance_matrix = pd.DataFrame(
        [
            [0.040, 0.010, 0.002],
            [0.010, 0.030, 0.001],
            [0.002, 0.001, 0.010],
        ],
        index=expected_returns.index,
        columns=expected_returns.index,
    )
    Q = build_portfolio_qubo(
        expected_returns,
        covariance_matrix,
        num_assets_to_select=2,
        risk_aversion=2.0,
        penalty_strength=5.0,
    )
    h, J, offset = qubo_to_ising(Q)
    selection = pd.Series({"AAPL": 1, "MSFT": 1, "BND": 0})
    spins = 1 - 2 * selection

    assert evaluate_ising(spins, h, J, offset) == pytest.approx(
        evaluate_qubo(selection, Q)
    )


def test_evaluate_ising_aligns_spins_by_ticker_label():
    h = pd.Series({"AAPL": 0.50, "MSFT": -0.25}, name="h")
    J = pd.DataFrame(
        [[0.0, 0.10], [0.10, 0.0]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )
    spins = pd.Series({"MSFT": -1, "AAPL": 1})

    energy = evaluate_ising(spins, h, J, offset=0.20)

    assert energy == pytest.approx(0.50 + 0.25 - 0.10 + 0.20)


def test_qubo_to_ising_raises_value_error_for_empty_q():
    with pytest.raises(ValueError, match="Q must not be empty"):
        qubo_to_ising(pd.DataFrame())


def test_qubo_to_ising_raises_value_error_for_non_square_q():
    Q = pd.DataFrame([[1.0, 2.0]], index=["AAPL"], columns=["AAPL", "MSFT"])

    with pytest.raises(ValueError, match="Q must be square"):
        qubo_to_ising(Q)


def test_qubo_to_ising_raises_value_error_for_mismatched_q_labels():
    Q = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["AAPL", "MSFT"],
        columns=["MSFT", "AAPL"],
    )

    with pytest.raises(ValueError, match="Q must have matching row and column labels"):
        qubo_to_ising(Q)


def test_evaluate_ising_raises_value_error_for_invalid_spins():
    h = pd.Series({"AAPL": 0.50}, name="h")
    J = pd.DataFrame([[0.0]], index=["AAPL"], columns=["AAPL"])
    spins = pd.Series({"AAPL": 0})

    with pytest.raises(ValueError, match="spins values must be only -1 or \\+1"):
        evaluate_ising(spins, h, J, offset=0.0)


def test_evaluate_ising_raises_value_error_for_mismatched_spin_labels():
    h = pd.Series({"AAPL": 0.50}, name="h")
    J = pd.DataFrame([[0.0]], index=["AAPL"], columns=["AAPL"])
    spins = pd.Series({"MSFT": 1})

    with pytest.raises(ValueError, match="Tickers in spins and h must match"):
        evaluate_ising(spins, h, J, offset=0.0)


def test_evaluate_ising_raises_value_error_for_mismatched_j_labels():
    h = pd.Series({"AAPL": 0.50}, name="h")
    J = pd.DataFrame([[0.0]], index=["MSFT"], columns=["MSFT"])
    spins = pd.Series({"AAPL": 1})

    with pytest.raises(ValueError, match="Tickers in h and J must match"):
        evaluate_ising(spins, h, J, offset=0.0)
