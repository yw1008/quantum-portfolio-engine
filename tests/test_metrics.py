from __future__ import annotations

import pandas as pd
import pytest

from backend.analysis.metrics import (
    compute_portfolio_return,
    compute_portfolio_volatility,
    compute_sharpe_ratio,
)


def test_compute_portfolio_return_uses_weighted_expected_returns():
    weights = pd.Series({"AAPL": 0.60, "MSFT": 0.40})
    expected_returns = pd.Series({"AAPL": 0.10, "MSFT": 0.05})

    portfolio_return = compute_portfolio_return(weights, expected_returns)

    assert portfolio_return == pytest.approx(0.08)


def test_compute_portfolio_return_aligns_by_ticker_label():
    weights = pd.Series({"AAPL": 0.60, "MSFT": 0.40})
    expected_returns = pd.Series({"MSFT": 0.05, "AAPL": 0.10})

    portfolio_return = compute_portfolio_return(weights, expected_returns)

    assert portfolio_return == pytest.approx(0.08)


def test_compute_portfolio_volatility_uses_covariance_matrix():
    weights = pd.Series({"AAPL": 0.60, "MSFT": 0.40})
    covariance_matrix = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["AAPL", "MSFT"],
        columns=["AAPL", "MSFT"],
    )

    portfolio_volatility = compute_portfolio_volatility(weights, covariance_matrix)

    expected_variance = 0.60**2 * 0.04 + 2 * 0.60 * 0.40 * 0.01 + 0.40**2 * 0.09
    assert portfolio_volatility == pytest.approx(expected_variance**0.5)


def test_compute_portfolio_volatility_aligns_by_ticker_label():
    weights = pd.Series({"AAPL": 0.60, "MSFT": 0.40})
    covariance_matrix = pd.DataFrame(
        [[0.09, 0.01], [0.01, 0.04]],
        index=["MSFT", "AAPL"],
        columns=["MSFT", "AAPL"],
    )

    portfolio_volatility = compute_portfolio_volatility(weights, covariance_matrix)

    expected_variance = 0.60**2 * 0.04 + 2 * 0.60 * 0.40 * 0.01 + 0.40**2 * 0.09
    assert portfolio_volatility == pytest.approx(expected_variance**0.5)


def test_compute_sharpe_ratio_uses_excess_return():
    sharpe_ratio = compute_sharpe_ratio(
        portfolio_return=0.12,
        portfolio_volatility=0.20,
        risk_free_rate=0.02,
    )

    assert sharpe_ratio == pytest.approx(0.50)


def test_compute_portfolio_return_raises_value_error_for_mismatched_tickers():
    weights = pd.Series({"AAPL": 0.60, "MSFT": 0.40})
    expected_returns = pd.Series({"AAPL": 0.10, "GOOGL": 0.05})

    with pytest.raises(ValueError, match="Ticker labels must match"):
        compute_portfolio_return(weights, expected_returns)


def test_compute_portfolio_volatility_raises_value_error_for_mismatched_tickers():
    weights = pd.Series({"AAPL": 0.60, "MSFT": 0.40})
    covariance_matrix = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["AAPL", "GOOGL"],
        columns=["AAPL", "GOOGL"],
    )

    with pytest.raises(ValueError, match="Ticker labels must match"):
        compute_portfolio_volatility(weights, covariance_matrix)


def test_compute_portfolio_return_raises_value_error_for_empty_weights():
    expected_returns = pd.Series({"AAPL": 0.10})

    with pytest.raises(ValueError, match="weights must not be empty"):
        compute_portfolio_return(pd.Series(dtype=float), expected_returns)


def test_compute_portfolio_volatility_raises_value_error_for_empty_covariance():
    weights = pd.Series({"AAPL": 1.0})

    with pytest.raises(ValueError, match="covariance_matrix must not be empty"):
        compute_portfolio_volatility(weights, pd.DataFrame())


def test_compute_sharpe_ratio_raises_value_error_for_non_positive_volatility():
    with pytest.raises(ValueError, match="portfolio_volatility must be positive"):
        compute_sharpe_ratio(portfolio_return=0.10, portfolio_volatility=0.0)
