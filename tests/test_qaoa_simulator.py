from __future__ import annotations

import pandas as pd
import pytest

from backend.quantum.qaoa_simulator import bitstring_to_selection, run_qaoa_simulation


def test_run_qaoa_simulation_returns_sorted_measurement_dataframe():
    h = pd.Series({"AAPL": 0.50, "MSFT": -0.25}, name="h")
    J = pd.DataFrame(
        [[0.0, 0.20], [0.20, 0.0]],
        index=h.index,
        columns=h.index,
    )

    results = run_qaoa_simulation(
        h,
        J,
        gamma=0.70,
        beta=0.30,
        shots=128,
    )

    assert list(results.columns) == ["bitstring", "count", "probability"]
    assert results["count"].sum() == 128
    assert results["probability"].sum() == pytest.approx(1.0)
    assert results["count"].is_monotonic_decreasing
    assert set("".join(results["bitstring"])).issubset({"0", "1"})
    assert results["bitstring"].str.len().eq(2).all()


def test_run_qaoa_simulation_preserves_h_ticker_order_in_bitstrings():
    h = pd.Series({"AAPL": 0.0, "MSFT": 0.0, "BND": 0.0}, name="h")
    J = pd.DataFrame(0.0, index=h.index, columns=h.index)

    results = run_qaoa_simulation(
        h,
        J,
        gamma=0.0,
        beta=0.0,
        shots=64,
    )

    selection = bitstring_to_selection(results.loc[0, "bitstring"], list(h.index))

    assert list(selection.index) == ["AAPL", "MSFT", "BND"]
    assert set(selection).issubset({0, 1})


def test_bitstring_to_selection_converts_bits_to_ticker_indexed_series():
    selection = bitstring_to_selection("101", ["SPY", "QQQ", "TLT"])

    expected_selection = pd.Series(
        {"SPY": 1, "QQQ": 0, "TLT": 1},
        name="selection",
    )
    pd.testing.assert_series_equal(selection, expected_selection)


def test_run_qaoa_simulation_raises_value_error_for_invalid_shots():
    h = pd.Series({"AAPL": 0.50}, name="h")
    J = pd.DataFrame([[0.0]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="shots must be positive"):
        run_qaoa_simulation(h, J, gamma=0.1, beta=0.2, shots=0)


def test_run_qaoa_simulation_raises_value_error_for_invalid_num_layers():
    h = pd.Series({"AAPL": 0.50}, name="h")
    J = pd.DataFrame([[0.0]], index=["AAPL"], columns=["AAPL"])

    with pytest.raises(ValueError, match="num_layers must be positive"):
        run_qaoa_simulation(h, J, gamma=0.1, beta=0.2, num_layers=0)


def test_run_qaoa_simulation_raises_value_error_for_label_mismatch():
    h = pd.Series({"AAPL": 0.50}, name="h")
    J = pd.DataFrame([[0.0]], index=["MSFT"], columns=["MSFT"])

    with pytest.raises(ValueError, match="Tickers in h and J must match"):
        run_qaoa_simulation(h, J, gamma=0.1, beta=0.2)


def test_bitstring_to_selection_raises_value_error_for_empty_tickers():
    with pytest.raises(ValueError, match="tickers must not be empty"):
        bitstring_to_selection("", [])


def test_bitstring_to_selection_raises_value_error_for_length_mismatch():
    with pytest.raises(ValueError, match="bitstring length must match"):
        bitstring_to_selection("10", ["AAPL"])


def test_bitstring_to_selection_raises_value_error_for_invalid_characters():
    with pytest.raises(ValueError, match="bitstring must contain only 0 or 1"):
        bitstring_to_selection("1A0", ["AAPL", "MSFT", "BND"])
