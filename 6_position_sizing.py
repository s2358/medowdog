"""
Portfolio position sizing with compounding capital.

python 6_positional_sizing.py data/correlations/allocation_SPY_GLDM_SLV.csv data/SPY/*.csv data/GLDM/*.csv data/SLV/*.csv

Capital is not constant. Each day's position are sized against the equity produced by every prior day's P&L, which makes the whole calculation sequential: equity -> vol scalar -> position -> next day's P&L -> equity.

Costs are read from data/costs/ if a measurement exists and are deducted from equity as they are paid. The first run has no cost file and reports gross; once costs.py has run, re-run this and the compounding includes them.

The FX rate comes from the cache written by fx_rate.py
"""

import pandas as pd
import numpy as np
import os
import sys

import fx_rate
from config import (CAPITAL, VOL_TARGET, HOME_CURRENCY, BUSINESS_DAYS, WARMUP, INERTIA, PORTFOLIO_DIR, COSTS_DIR)

PER_TICKER_COLS = ('ivv', 'vol_scalar', 'subsystem_position', 'portfolio_position', 'rounded', 'held', 'trade', 'notional_home', 'pnl', 'cost')

def ticker_from_path(path):
    return os.path.basename(path).split('_')[0].upper()

def load_instrument(path):
    cols = ['date', 'close', 'combined_forecast', 'icv']
    df = pd.read_csv(path, usecols=cols).iloc[WARMUP:]
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')

def load_allocation(path):
    alloc = pd.read_csv(path).set_index('ticker')
    idm = alloc['idm'].iloc[0]
    if not np.allclose(alloc['idm'], idm):
        raise ValueError("idm differs between rows of the allocation file")
    return alloc['instrument_weight'].to_dict(), float(idm)

def load_costs():
    """Latest measured cost per share, per ticker. None if never measured."""
    files = sorted(COSTS_DIR.glob('costs_*.csv')) if COSTS_DIR.exists() else []
    if not files:
        return None, None
    latest = files[-1]
    series = pd.read_csv(latest).set_index('ticker')['cost_per_share_C']
    return series.to_dict(), latest.name

def run(frames, weights, idm, fx, capital0, vol_target, costs=None):
    """
    Walk the history day by day, compounding equity.
    
    Within day i: yesterday's position earns todays'
    """

def apply_inertia(target, buffer=INERTIA):
    """
    Walk the rounded target series, holding the current position until the gap exceeds 'buffer'. Returns    the position actually held.

    Sequential by nature - today's held position depends on yesterday's - so this can't be vectorised.
    """
    held = np.empty(len(target))
    current = 0.0
    for i, want in enumerate(target):
        if np.isnan(want):
            held[i] = current
            continue
        denom = max(abs(current), 1.0)
        if abs(want - current) / denom > buffer:
            current = want
        held[i] = current
    return pd.Series(held, index=target.index)

def build_positions(df, weight, idm, capital, vol_target, fx):
    """Add the position columns for one instrument."""
    daily_cash_vol_target = capital * vol_target / np.sqrt(BUSINESS_DAYS)
    
    df = df.copy()
    df['ivv'] = df['icv'] * fx
    df['vol_scalar'] = daily_cash_vol_target / df['ivv']
    df['subsystem_position'] = df['combined_forecast'] / 10 * df['vol_scalar']
    df['portfolio_position'] = df['subsystem_position'] * weight * idm
    df['rounded'] = df['portfolio_position'].round()
    df['held'] = apply_inertia(df['rounded'])
    df['trade'] = df['held'].diff().fillna(df['held'])
    df['notional_home'] = df['held'] * df['close'] * fx
    return df

def turnover_stats(df, years):
    """
    turnover = avg blocks traded per year / (2 x avg abs blocks held).
    The 2 is because a round trip is two trades.
    """
    blocks_traded = df['trade'].abs().sum()
    avg_held = df['held'].abs().mean()
    traded_per_year = blocks_traded / years
    turnover = traded_per_year / (2 * avg_held) if avg_held else np.nan

    trades = df.loc[df['trade'] != 0 , 'trade'].abs()
    return {
            'avg_abs_position': avg_held,
            'blocks_traded_per_year': traded_per_year,
            'turnover': turnover,
            'n_trades': len(trades),
            'trades_per_year': len(trades) / years,
            'typical_trade': trades.mean() if len(trades) else np.nan,
            'median_trade': trades.median() if len(trades) else np.nan,
            'pct_days_zero_position': (df['held'] == 0).mean(),
            }

def main():
    if len(sys.argv) < 3:
        print("Usage: python 6_position_sizing.py <allocation_csv> "
              "<instrument_csv> [<instrument_csv> ...]")
        sys.exit(1)

    fx = fx_rate.load()
    cached = fx_rate.describe()

    alloc_path, inst_paths = sys.argv[1], sys.argv[2:]
    weights, idm = load_allocation(alloc_path)

    frames = {ticker_from_path(p): load_instrument(p) for p in inst_paths}

    missing = set(frames) - set(weights)
    if missing:
        raise KeyError(f"no weight in allocation file for: {sorted(missing)}")

    # restrict every instrument to the common trading days
    common = None
    for df in frames.values():
        common = df.index if common is None else common.intersection(df.index)
    frames = {t: df.loc[common] for t, df in frames.items()}

    years = len(common) / BUSINESS_DAYS
    print(f"capital {CAPITAL:,.0f} {HOME_CURRENCY}   "
          f"vol target {VOL_TARGET:.0%}   IDM {idm:.4f}")
    print(f"fx {cached['pair']} {fx:.5f} (fetched {cached['fetched_at']})")
    print(f"window: {common[0].date()} -> {common[-1].date()}")
    print(f"rows: {len(common)} ({years:.1f} years)\n")

    built, stats = {}, {}
    for ticker, df in frames.items():
        built[ticker] = build_positions(df, weights[ticker], idm, CAPITAL, VOL_TARGET, fx)
        stats[ticker] = turnover_stats(built[ticker], years)

    summary = pd.DataFrame(stats).T
    summary.index.name = 'ticker'
    print(summary.round(3).to_string())

    print("\nTYPICAL_TRADE for costs.py:")
    print("  {" + ", ".join(
        f"'{t}': {max(1, int(round(s['typical_trade'])))}"
        for t, s in stats.items() if not np.isnan(s['typical_trade'])) + "}")

    # gross leverage over time, from positions actually held
    notional = pd.DataFrame({t: d['notional_home'] for t, d in built.items()})
    gross = notional.abs().sum(axis=1) / CAPITAL
    print(f"\ngross leverage   mean {gross.mean():.2f}x   "
          f"median {gross.median():.2f}x   max {gross.max():.2f}x   "
          f"days over 2x: {(gross > 2).mean():.1%}")

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    stem = '_'.join(sorted(built))
    summary.to_csv(PORTFOLIO_DIR / f'turnover_{stem}.csv')
    for ticker, df in built.items():
        df.to_csv(PORTFOLIO_DIR / f'positions_{ticker}.csv')
    print(f"\nwritten to PORTFLIO_DIR/")

if __name__ == '__main__':
    main()










