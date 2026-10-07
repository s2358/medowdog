"""
Subsystem-return correlations, instrument weights and the IDM.

    python 4_correlations.py <csv> <csv> [<csv> ...]

Writes three files to data/correlations/:
    subsystem_returns_<tickers>.csv     the aligned panel
    correlation_matrix_<tickers>.csv    the negatives floored at zero
    allocation_<tickers>.csv            weights + IDM, read by script 6

Everything is computed before anything is written, so a bad WEIGHTS dict fails before it can leave a half updated allocation file on disk.
"""

import pandas as pd
import numpy as np
import os
import sys
from config import (WEIGHTS, IDM_CAP, WARMUP, BUSINESS_DAYS, CORRELATIONS_DIR)

def ticker_from_path(path):
    """trim the filename to retain only the ticker."""
    return os.path.basename(path).split('_')[0].upper()

def load_subsystem_returns(path):
    df = pd.read_csv(path, usecols=['date', 'subsystem_returns'])
    df = df.iloc[WARMUP:]
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')['subsystem_returns'].rename(ticker_from_path(path))

def build_panel(paths):
    series = [load_subsystem_returns(p) for p in paths]
    names = [s.name for s in series]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"same ticker given more than once: {sorted(duplicates)}")
    return pd.concat(series, axis=1, join='inner').dropna()

def correlation_matrix(panel, floor_negatives=True):
    H = panel.corr()
    if floor_negatives:
        H = H.clip(lower=0)
    return H

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

    raw = panel.corr()
    H = correlation_matrix(panel)
    idm, v = instrument_diversification_multiplier(H, WEIGHTS)
    
    allocation = pd.DataFrame({
        'ticker': list(H.columns),
        'instrument_weight': [WEIGHTS[c] for c in H.columns],
        'idm': idm,
        })

    print(f"overlapping window: {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"rows: {n} ({n /BUSINESS_DAYS:.1f} years)\n")

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
            se = correlation_standard_error(r, n)
            print(f"  {a:6s} {b:6s}  r  {r:+.3f}  se  {se:.3f}  "
                  f"95% ci [{r - 1.96 * se:+.3f}, {r + 1.96 * se:+.3f}]")

    print("\nannualised vol of each subsystem over this window:")
    for col in panel.columns:
        print(f"  {col:6s} {panel[col].std() * np.sqrt(BUSINESS_DAYS):.1%}")
    
    print("\ninstrument weights:")
    for c in H.columns:
        print(f" {c:6s} {WEIGHTS[c]:.0%}")
    print(f"W x H x W' = {v:.5f}")
    print(f"IDM        = {idm:.4f}" + ("  (capped)" if idm == IDM_CAP else ""))

    CORRELATIONS_DIR.mkdir(parents=True, exist_ok=True)
    stem = '_'.join(sorted(panel.columns))

    out = CORRELATIONS_DIR / f'subsystem_returns_{stem}.csv'
    panel.to_csv(out)
    print(f"\npanel written to {out}")

    corr_out = CORRELATIONS_DIR / f'correlation_matrix_{stem}.csv'
    H.to_csv(corr_out)
    print(f"correlation matrix written to {corr_out}")
        
    alloc_out = CORRELATIONS_DIR / f'allocation_{stem}.csv'
    allocation.to_csv(alloc_out, index=False)
    print(f"allocation written to {alloc_out}")

if __name__ == '__main__':
    main()
