"""
Subsystem-return correlations and the IDM.

    python 4_correlation.py <csv> <csv> [<csv> ...]

Writes three files to data/correlations/:
    subsystem_returns_<tickers>.csv     the panel
    correlation_matrix_<tickers>.csv    the negatives floored at zero
    allocation_<tickers>.csv            weights + IDM, a record of the run

This script measures. It does not set anything. The live path and the backtest
both read the IDM from config.py, the same way they read the FDM and the
forecast scalars. To adopt a new value, edit config.py by hand. Nothing reads
the allocation file; it is kept as the record of what was measured and when.

Instruments are joined on every date any of them traded rather than only the dates all of them did, so one short history does not shorten the rest. Each correlation is then measured on its own pair's overlap.

The matrix covers every file given. The multiplier covers only the instruments that carry a weight, so a candidate can be measured against the book before it is admitted to it.
"""

import pandas as pd
import numpy as np
import os
import sys
from config import (IDM_CAP, WARMUP, BUSINESS_DAYS, CORRELATIONS_DIR,
                    WINDOW_START, WINDOW_END, IDM as IDM_IN_CONFIG)
from universe import WEIGHTS, ticker_from_path


def load_subsystem_returns(path):
    df = pd.read_csv(path, usecols=['date', 'subsystem_returns'])
    df = df.iloc[WARMUP:]
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')['subsystem_returns'].rename(ticker_from_path(path))


def build_panel(paths):
    """
    The panel of subsystem returns, restricted to the backtest window.

    The window matters here: the multiplier is an estimate from the data, so
    an out-of-sample run must not compute it from rows the run is pretending
    not to have seen.

    Gaps are left empty rather than dropped. A row where one instrument did
    not yet exist is still a real row for the others, and discarding it would
    throw away the history the longer instruments have.
    """
    series = [load_subsystem_returns(p) for p in paths]
    names = [s.name for s in series]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"same ticker given more than once: {sorted(duplicates)}")
    panel = pd.concat(series, axis=1, join='outer')

    if WINDOW_START is not None:
        panel = panel[panel.index >= pd.Timestamp(WINDOW_START)]
    if WINDOW_END is not None:
        panel = panel[panel.index < pd.Timestamp(WINDOW_END)]
    if panel.empty:
        raise ValueError("the configured window leaves no rows")
    return panel


def pair_rows(panel, a, b):
    """Rows on which both instruments have a return."""
    return int((panel[a].notna() & panel[b].notna()).sum())


def pairwise_correlations(panel, min_rows=BUSINESS_DAYS):
    """
    Each correlation measured on its own pair's overlap.

    A pair with less than a year of shared history is not estimated at all.
    The number would be noise, and it would go straight into the IDM, where
    nothing downstream could tell it from a measurement.
    """
    raw = panel.corr(min_periods=min_rows)
    thin = [f"{a}/{b} ({pair_rows(panel, a, b)} rows)"
            for i, a in enumerate(raw.columns) for b in raw.columns[i + 1:]
            if np.isnan(raw.loc[a, b])]
    if thin:
        raise ValueError(f"under {min_rows} rows of overlap: {', '.join(thin)}")
    return raw


def correlation_matrix(raw, floor_negatives=True):
    return raw.clip(lower=0) if floor_negatives else raw


def correlation_standard_error(r, n):
    """
    SE of a sample correlation, (1 - r^2) / sqrt(n-1).
    The familiar 1/sqrt(n) is only the r=0 case and overstates the uncertainty badly once r is large.
    """
    return (1 - r ** 2) / np.sqrt(n - 1)

def instrument_diversification_multiplier(H, weights, cap=IDM_CAP):
    """
    IDM = 1 / sqrt(W x H x W'), capped.
    H must already have negative correlations floored at zero.
    """
    missing = set(H.columns) - set(weights)
    if missing:
        raise KeyError(f"no weight supplied for: {sorted(missing)}")
    W = np.array([weights[c] for c in H.columns])
    if not np.isclose(W.sum(), 1.0):
        raise ValueError(f"weights sum to {W.sum():.4f}, expected 1.0")

    v = W @ H.values @ W.T
    return min(1 / np.sqrt(v), cap), v

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <csv1> <csv2> [<csv3> ...]")
        sys.exit(1)

    paths = sys.argv[1:]
    panel = build_panel(paths)
    n = len(panel)

    raw = pairwise_correlations(panel)
    H = correlation_matrix(raw)

    # the matrix covers every file given, so a candidate can be measured
    # against the book before it is given a weight. The multiplier is only
    # defined over what is actually traded.
    traded = [c for c in H.columns if c in WEIGHTS]
    extra = [c for c in H.columns if c not in WEIGHTS]
    idm, v = instrument_diversification_multiplier(
            H.loc[traded, traded], WEIGHTS)

    allocation = pd.DataFrame({
        'ticker': traded,
        'instrument_weight': [WEIGHTS[c] for c in traded],
        'idm': idm,
        })

    print(f"window: {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"rows: {n} ({n /BUSINESS_DAYS:.1f} years)\n")

    print("rows and annualised vol per subsystem:")
    for col in panel.columns:
        rows = int(panel[col].notna().sum())
        vol = panel[col].std() * np.sqrt(BUSINESS_DAYS)
        print(f"  {col:6s} {rows:6d} rows   {vol:.1%}   "
              f"from {panel[col].first_valid_index().date()}")
    print()

    print("correlation matrix (raw):")
    print(raw.round(3).to_string(), "\n")

    if not raw.equals(H):
        print("correlation matrix (negatives floored at zero):")
        print(H.round(3).to_string(), "\n")

    print("pairwise correlation and standard error:")
    cols = list(raw.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = raw.loc[a, b]
            shared = pair_rows(panel, a, b)
            se = correlation_standard_error(r, shared)
            print(f"  {a:6s} {b:6s}  r  {r:+.3f}  se  {se:.3f}  "
                  f"95% ci [{r - 1.96 * se:+.3f}, {r + 1.96 * se:+.3f}]  "
                  f"on {shared} shared rows")

    print("\ninstrument weights:")
    for c in traded:
        print(f" {c:6s} {WEIGHTS[c]:.0%}")
    if extra:
        print(f" not weighted: {', '.join(extra)}")

    print(f"W x H x W' = {v:.5f}")
    print(f"IDM        = {idm:.4f}" + ("  (capped)" if idm == IDM_CAP else ""))

    change = idm - IDM_IN_CONFIG
    print(f"config.py holds IDM = {IDM_IN_CONFIG:.4f}, "
          f"a difference of {change:+.4f}")
    if abs(change) > 1e-4:
        print("nothing uses the measured value. To adopt it, edit config.py.")

    CORRELATIONS_DIR.mkdir(parents=True, exist_ok=True)
    stem = '_'.join(sorted(panel.columns))
    traded_stem = '_'.join(sorted(traded))

    out = CORRELATIONS_DIR / f'subsystem_returns_{stem}.csv'
    panel.to_csv(out)
    print(f"\npanel written to {out}")

    corr_out = CORRELATIONS_DIR / f'correlation_matrix_{stem}.csv'
    H.to_csv(corr_out)
    print(f"correlation matrix written to {corr_out}")

    alloc_out = CORRELATIONS_DIR / f'allocation_{traded_stem}.csv'
    allocation.to_csv(alloc_out, index=False)
    print(f"allocation written to {alloc_out}")

if __name__ == '__main__':
    main()
