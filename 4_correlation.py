import pandas as pd
import numpy as np
import os
import sys


WEIGHTS = {'SPY': 0.40, 'GLDM': 0.30, 'SLV': 0.30}
IDM_CAP = 2.5
WARMUP = 256
OUT_DIR = 'data/correlations'

def ticker_from_path(path):
    """trim the filename to retain only the ticker"""
    return os.path.basename(path).split('_')[0].upper()

def load_subsystem_returns(path):
    df = pd.read_csv(path, usecols=['date', 'subsystem_returns'])
    df = df.iloc[WARMUP:]
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')['subsystem_returns'].rename(ticker_from_path(path))

def build_panel(paths):
    series = [load_subsystem_returns(p) for p in paths]
    return pd.concat(series, axis=1, join='inner').dropna()

def correlation_matrix(panel, floor_negatives=True):
    H = panel.corr()
    if floor_negatives:
        H = H.clip(lower=0)
    return H

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
        print("Usage: python 4_corrleations.py <csv1> <csv2> [<csv3> ...]")
        sys.exit(1)

    paths = sys.argv[1:]
    panel = build_panel(paths)
    raw = panel.corr()
    H = correlation_matrix(panel)
    idm, v = instrument_diversification_multiplier(H, WEIGHTS)
    allocation = pd.DataFrame({
        'ticker': list(H.columns),
        'instrument_weight': [WEIGHTS[c] for c in H.columns],
        'idm': idm,
        })

    print(f"overlapping window: {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"rows: {len(panel)} ({len(panel)/256:.1f} years)\n")

    print("correlation matrix (raw):")
    print(raw.round(3).to_string(), "\n")

    if not raw.equals(H):
        print("correlation matrix (negatives floored at zero):")
        print(H.round(3).to_string(), "\n")

    n = len(panel)
    print(f"standard error on each corrleation: ~{1/np.sqrt(n):.3f}\n")

    print("annulised vol of each subsystem over this window:")
    for col in panel.columns:
        print(f"  {col:6s} {panel[col].std()*16:.1%}")
    
    print("\ninstrument weights:")
    for c in H.columns:
        print(f" {c:6s} {WEIGHTS[c]:.0%}")
    print(f"W x H x W' = {v:.5f}")
    print(f"IDM        = {idm:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = '_'.join(sorted(panel.columns))

    out = os.path.join(OUT_DIR, f'subsystem_returns_{stem}.csv')
    panel.to_csv(out)
    print(f"\npanel written to {out}")

    corr_out = os.path.join(OUT_DIR, f'correlation_matrix_{stem}.csv')
    H.to_csv(corr_out)
    print(f"correlation matrix written to {corr_out}")
        
    alloc_out = os.path.join(OUT_DIR, f'allocation_{stem}.csv')
    allocation.to_csv(alloc_out, index=False)
    print(f"allocation written to {alloc_out}")

if __name__ == '__main__':
    main()
