"""Local QAOA simulation utilities."""

from __future__ import annotations

import pandas as pd
from qiskit_aer import AerSimulator

from backend.quantum.qaoa_circuit import build_qaoa_circuit


def run_qaoa_simulation(
    h: pd.Series,
    J: pd.DataFrame,
    gamma: float,
    beta: float,
    num_layers: int = 1,
    shots: int = 1024,
) -> pd.DataFrame:
    """Run a fixed-parameter QAOA circuit on a local Aer simulator.

    Args:
        h: Linear Ising coefficients indexed by ticker.
        J: Pairwise Ising couplings indexed and columned by ticker.
        gamma: Cost-layer angle.
        beta: Mixer-layer angle.
        num_layers: Number of repeated QAOA layers.
        shots: Number of simulated circuit measurements.

    Returns:
        DataFrame with ``bitstring``, ``count``, and ``probability`` columns,
        sorted by count descending. Bitstrings are returned in ticker order.

    Raises:
        ValueError: If ``shots`` or ``num_layers`` is not positive, or if Ising
            coefficient labels are invalid.
    """
    if shots <= 0:
        raise ValueError("shots must be positive.")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")

    circuit = build_qaoa_circuit(
        h=h,
        J=J,
        gamma=gamma,
        beta=beta,
        num_layers=num_layers,
        add_measurements=True,
    )

    simulator = AerSimulator(seed_simulator=42)
    result = simulator.run(circuit, shots=shots).result()
    counts = result.get_counts(circuit)

    rows = [
        {
            "bitstring": _to_ticker_order_bitstring(bitstring),
            "count": int(count),
            "probability": int(count) / shots,
        }
        for bitstring, count in counts.items()
    ]

    return (
        pd.DataFrame(rows)
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def bitstring_to_selection(
    bitstring: str,
    tickers: list[str],
) -> pd.Series:
    """Convert a measured bitstring into a binary asset selection.

    Args:
        bitstring: Binary measurement string in ticker order.
        tickers: Ticker labels in the same order as the bitstring.

    Returns:
        Binary selection Series indexed by ticker.

    Raises:
        ValueError: If tickers are empty, length does not match, or the bitstring
            contains characters other than 0 or 1.
    """
    if not tickers:
        raise ValueError("tickers must not be empty.")
    if len(bitstring) != len(tickers):
        raise ValueError("bitstring length must match number of tickers.")
    if any(character not in {"0", "1"} for character in bitstring):
        raise ValueError("bitstring must contain only 0 or 1.")

    return pd.Series(
        [int(character) for character in bitstring],
        index=tickers,
        name="selection",
    )


def _to_ticker_order_bitstring(raw_bitstring: str) -> str:
    """Convert Qiskit count keys to qubit/ticker order."""
    compact_bitstring = raw_bitstring.replace(" ", "")
    return compact_bitstring[::-1]
