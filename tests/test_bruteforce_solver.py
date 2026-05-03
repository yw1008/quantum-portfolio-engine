from __future__ import annotations

import pandas as pd
import pytest

from backend.quantum.bruteforce_solver import evaluate_qubo, solve_qubo_bruteforce
from backend.quantum.qubo import build_portfolio_qubo


def test_evaluate_qubo_computes_quadratic_objective():
    Q = pd.DataFrame(
        [[-1.0, 0.25], [0.25, -0.50]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )
    selection = pd.Series({"AAPL": 1, "MSFT": 1})

    objective = evaluate_qubo(selection, Q)

    assert objective == pytest.approx(-1.0 + 0.25 + 0.25 - 0.50)


def test_evaluate_qubo_aligns_selection_by_ticker_label():
    Q = pd.DataFrame(
        [[-1.0, 0.25], [0.25, -0.50]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )
    selection = pd.Series({"MSFT": 1, "AAPL": 0})

    objective = evaluate_qubo(selection, Q)

    assert objective == pytest.approx(-0.50)


def test_solve_qubo_bruteforce_returns_global_minimum():
    Q = pd.DataFrame(
        [[-1.0, 0.75], [0.75, -0.50]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    selection, objective = solve_qubo_bruteforce(Q)

    expected_selection = pd.Series({"AAPL": 1, "MSFT": 0})
    pd.testing.assert_series_equal(selection, expected_selection)
    assert objective == pytest.approx(-1.0)


def test_solve_qubo_bruteforce_works_with_portfolio_qubo():
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

    selection, objective = solve_qubo_bruteforce(Q)

    expected_selection = pd.Series({"AAPL": 1, "MSFT": 1, "BND": 0})
    pd.testing.assert_series_equal(selection, expected_selection)
    assert objective == pytest.approx(evaluate_qubo(expected_selection, Q))


def test_evaluate_qubo_raises_value_error_for_non_binary_selection():
    Q = pd.DataFrame([[-1.0]], index=["AAPL"], columns=["AAPL"])
    selection = pd.Series({"AAPL": 2})

    with pytest.raises(ValueError, match="selection values must be only 0 or 1"):
        evaluate_qubo(selection, Q)


def test_evaluate_qubo_raises_value_error_for_empty_selection():
    Q = pd.DataFrame([[-1.0]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="selection must not be empty"):
        evaluate_qubo(pd.Series(dtype=int), Q)


def test_evaluate_qubo_raises_value_error_for_mismatched_tickers():
    Q = pd.DataFrame([[-1.0]], index=["AAPL"], columns=["AAPL"])
    selection = pd.Series({"MSFT": 1})

    with pytest.raises(ValueError, match="Tickers in selection and Q must match"):
        evaluate_qubo(selection, Q)


def test_evaluate_qubo_raises_value_error_for_empty_q():
    selection = pd.Series({"AAPL": 1})

    with pytest.raises(ValueError, match="Q must not be empty"):
        evaluate_qubo(selection, pd.DataFrame())


def test_solve_qubo_bruteforce_raises_value_error_for_empty_q():
    with pytest.raises(ValueError, match="Q must not be empty"):
        solve_qubo_bruteforce(pd.DataFrame())


def test_solve_qubo_bruteforce_raises_value_error_for_non_square_q():
    Q = pd.DataFrame([[1.0, 2.0]], index=["AAPL"], columns=["AAPL", "MSFT"])

    with pytest.raises(ValueError, match="Q must be square"):
        solve_qubo_bruteforce(Q)


def test_solve_qubo_bruteforce_raises_value_error_for_mismatched_q_labels():
    Q = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=["AAPL", "MSFT"],
        columns=["MSFT", "AAPL"],
    )

    with pytest.raises(ValueError, match="Q must have matching row and column labels"):
        solve_qubo_bruteforce(Q)
