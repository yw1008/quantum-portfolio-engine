# Returns

Returns measure how asset prices change over time. Phase 1 includes utilities
for daily percentage returns and annualized mean returns.

## Current Implementation

- Module: `backend/data/returns.py`
- Function: `compute_daily_returns`
- Function: `compute_annualized_returns`

## Notes

Daily returns are computed from consecutive price observations. Annualized mean
returns are calculated by multiplying each asset's mean daily return by the
assumed number of trading days in a year.
