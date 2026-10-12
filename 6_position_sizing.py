"""
Portfolio position sizing with compounding capital.

    python 6_position_sizing.py data/correlations/allocation_GLDM_SLV_SPY.csv \
        data/SPY/*.csv data/GLDM/*.csv data/SLV/*.csv

Capital is not constant. Each day's positions are sized against the equity
produced by every prior day's P&L, which makes the whole calculation
sequential: equity -> vol scalar -> position -> next day's P&L -> equity.

Costs are read from data/costs/ if a measurement exists and are deducted from
equity as they are paid. The first run has no cost file and reports gross;
once the cost script has run, re-run this and the compounding includes them.

Dividends are read from data/dividends/ and credited to whatever position was
held into each ex-date, net of US withholding - negative when the position was
short. A ticker with no file there contributes nothing.

The FX rate comes from the cache written by fx_rate.py.
"""

import os
import sys

import numpy as np
import pandas as pd

import dividends
import fx_rate
from config import (CAPITAL, VOL_TARGET, HOME_CURRENCY, BUSINESS_DAYS,
                    WARMUP, INERTIA, PORTFOLIO_DIR, COSTS_DIR, US_WITHHOLDING)

PER_TICKER_COLS = ('ivv', 'vol_scalar', 'subsystem_position',
                   'portfolio_position', 'rounded', 'held', 'trade',
                   'notional_home', 'pnl', 'cost', 'dividend')


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

def load_income(tickers, index):
    """
    Net dividend per share on each row of the backtest, per ticker.

    Stored amounts are gross. Withholding comes off here: a Singapore account has no US treaty rate, so 30% of every distribution never arrives.

    An ex-date that is not itself a row - the index holds only the days every instrument traded - is carried to the next row rather than dropped. The entitlement is still the position held before the real ex-date, which is what the loop uses.
    """
    out, missing = {}, []
    for ticker in tickers:
        series = dividends.load(ticker)
        if series is None:
            missing.append(ticker)
        paid = np.zeros(len(index))
        if series is not None and not series.empty:
            net = series * (1 - US_WITHHOLDING)
            for position, amount in zip(index.searchsorted(net.index),
                                        net.values):
                if position < len(index):
                    paid[position] += amount
        out[ticker] = paid
    return out, missing


def round_half_away(x):
    """
    Round half away from zero. np.round uses banker's rounding, which sends
    0.5 to 0 and 1.5 to 2 - not what the framework means by rounding.
    """
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def run(frames, weights, idm, fx, capital0, vol_target, costs=None, income=None):
    """
    Walk the history day by day, compounding equity.

    Within day i: yesterday's position earns today's price change, equity
    updates, then today's forecast is sized against that new equity and the
    resulting trade is paid for.
    """
    tickers = sorted(frames)
    index = frames[tickers[0]].index
    n = len(index)

    close = {t: frames[t]['close'].values for t in tickers}
    fcast = {t: frames[t]['combined_forecast'].values for t in tickers}
    icv = {t: frames[t]['icv'].values for t in tickers}

    income = income or {t: np.zeros(n) for t in tickers}

    cols = {t: {k: np.full(n, np.nan) for k in PER_TICKER_COLS}
            for t in tickers}
    equity = np.full(n, np.nan)
    gross_pnl = np.zeros(n)
    day_costs = np.zeros(n)

    cap = capital0
    held_prev = {t: 0.0 for t in tickers}
    ruin_at = None

    for i in range(n):
        # 1. yesterday's position meets today's price, and is paid whatever went ex today
        total = 0.0
        for t in tickers:
            p = (held_prev[t] * (close[t][i] - close[t][i - 1]) * fx
                 if i > 0 else 0.0)
            d = held_prev[t] * income[t][i] * fx
            cols[t]['pnl'][i] = p
            cols[t]['dividend'][i] = d
            total += p + d
        gross_pnl[i] = total
        cap += total

        if cap <= 0:
            ruin_at = i
            break

        # 2. size today's positions against the equity we now hold
        daily_cash_vol_target = cap * vol_target / np.sqrt(BUSINESS_DAYS)
        spent = 0.0

        for t in tickers:
            ivv = icv[t][i] * fx
            vol_scalar = daily_cash_vol_target / ivv if ivv > 0 else np.nan
            subsystem = fcast[t][i] / 10 * vol_scalar
            target = subsystem * weights[t] * idm
            rounded = round_half_away(target) if np.isfinite(target) else np.nan

            current = held_prev[t]
            if np.isfinite(rounded):
                denom = max(abs(current), 1.0)
                new = rounded if abs(rounded - current) / denom > INERTIA \
                    else current
            else:
                new = current

            trade = new - current
            cost = abs(trade) * costs[t] * fx if costs and t in costs else 0.0

            c = cols[t]
            c['ivv'][i] = ivv
            c['vol_scalar'][i] = vol_scalar
            c['subsystem_position'][i] = subsystem
            c['portfolio_position'][i] = target
            c['rounded'][i] = rounded
            c['held'][i] = new
            c['trade'][i] = trade
            c['notional_home'][i] = new * close[t][i] * fx
            c['cost'][i] = cost

            spent += cost
            held_prev[t] = new

        # 3. pay for the trades
        day_costs[i] = spent
        cap -= spent
        equity[i] = cap

    # if the account was wiped out, keep only the days that happened
    end = ruin_at if ruin_at is not None else n
    index = index[:end]

    built = {}
    for t in tickers:
        df = pd.DataFrame({k: v[:end] for k, v in cols[t].items()}, index=index)
        df.insert(0, 'close', close[t][:end])
        df.insert(1, 'combined_forecast', fcast[t][:end])
        built[t] = df

    notional = pd.DataFrame(
        {t: built[t]['notional_home'] for t in tickers}).abs().sum(axis=1)

    paid = pd.DataFrame({t: built[t]['dividend'] for t in tickers}).sum(axis=1)


    portfolio = pd.DataFrame({
        'gross_pnl': gross_pnl[:end],
        'dividend': paid,
        'cost': day_costs[:end],
        'net_pnl': gross_pnl[:end] - day_costs[:end],
        'equity': equity[:end],
        'gross_notional': notional,
    }, index=index)
    portfolio['gross_leverage'] = (portfolio['gross_notional']
                                   / portfolio['equity'])

    return built, portfolio, ruin_at


def turnover_stats(df, years):
    """Carver: turnover = avg blocks traded per year / (2 x avg abs held)."""
    blocks_traded = df['trade'].abs().sum()
    avg_held = df['held'].abs().mean()
    traded_per_year = blocks_traded / years
    turnover = traded_per_year / (2 * avg_held) if avg_held else np.nan

    trades = df.loc[df['trade'] != 0, 'trade'].abs()
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
        print(f"Usage: python {os.path.basename(sys.argv[0])} <allocation_csv> "
              "<instrument_csv> [<instrument_csv> ...]")
        sys.exit(1)

    fx = fx_rate.load()
    cached = fx_rate.describe()
    costs, costs_file = load_costs()

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

    income, no_file = load_income(sorted(frames), common)
    counts = ', '.join(f"{t} {int((income[t] > 0).sum())}"
                       for t in sorted(income))
   
    print(f"capital {CAPITAL:,.0f} {HOME_CURRENCY} (compounding)   "
          f"vol target {VOL_TARGET:.0%}   IDM {idm:.4f}")
    print(f"fx {cached['pair']} {fx:.5f} (fetched {cached['fetched_at']})")
    print(f"costs: {costs_file if costs_file else 'not measured - gross only'}")
    print(f"dividends: {counts}"
          + (f"  (never fetched: {', '.join(no_file)})" if no_file else ''))
    print(f"window: {common[0].date()} -> {common[-1].date()}  "
          f"({len(common)} rows)\n")

    built, portfolio, ruin_at = run(
        frames, weights, idm, fx, CAPITAL, VOL_TARGET, costs, income)

    if ruin_at is not None:
        print(f"*** account wiped out on {common[ruin_at].date()} "
              f"(day {ruin_at} of {len(common)}) - truncated there ***\n")

    years = len(portfolio) / BUSINESS_DAYS

    stats = {t: turnover_stats(df, years) for t, df in built.items()}
    summary = pd.DataFrame(stats).T
    summary.index.name = 'ticker'
    print(summary.round(3).to_string())

    equity = portfolio['equity']
    lev = portfolio['gross_leverage']
    print(f"\nequity   start {CAPITAL:,.0f}   end {equity.iloc[-1]:,.0f}   "
          f"min {equity.min():,.0f}   max {equity.max():,.0f} {HOME_CURRENCY}")
    print(f"gross leverage   mean {lev.mean():.2f}x   "
          f"median {lev.median():.2f}x   max {lev.max():.2f}x   "
          f"days over 2x: {(lev > 2).mean():.1%}")
    if costs:
        print(f"costs paid   {portfolio['cost'].sum():,.0f} {HOME_CURRENCY} "
              f"({portfolio['cost'].sum()/years:,.0f}/yr)")
    paid = portfolio['dividend'].sum()
    if paid:
        print(f"dividends   {paid:,.0f} {HOME_CURRENCY}  "
              f"({paid/years:,.0f}/yr, net of {US_WITHHOLDING:.0%} "
              f"withholding)")
    
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    stem = '_'.join(sorted(built))
    summary.to_csv(PORTFOLIO_DIR / f'turnover_{stem}.csv')
    portfolio.to_csv(PORTFOLIO_DIR / 'portfolio.csv')
    for ticker, df in built.items():
        df.to_csv(PORTFOLIO_DIR / f'positions_{ticker}.csv')
    print(f"\nwritten to {PORTFOLIO_DIR}/")


if __name__ == '__main__':
    main()
