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

Margin financing is charged on the balance borrowed overnight - gross notional less equity, which assumes the account was converted to USD up front. The rate is the Fed funds effective rate from data/rates/ plus FINANCING_SPREAD, accrued actual/360 on calendar days, so a Friday position carries three days of interest into Monday.

Gross leverage is capped at MAX_LEVERAGE. The broker's limit applies to the whole portfolio, so it can only be checked once every instrument has been sized - a second pass scales every position by the same factor and re-rounds. Whole block means the result lands near the cap rather than exactly on it.

The loop runs in USD, because the account is funded by converting SGD up front. Each day's P&L, costs, financing and dividends are USD, and so is the equity the vol target is sized against. The home-currency curve is that equity at each day's rate, which exposes the whole balance to USDSGD, not just the day's flows - reported as its own fx_pnl line so the identity equity.diff() == net_pnl still holds exactly.
"""

import os
import sys

import numpy as np
import pandas as pd

import dividends
import fx_data
from config import (CAPITAL, VOL_TARGET, HOME_CURRENCY, BUSINESS_DAYS,
                    WARMUP, INERTIA, US_WITHHOLDING, FINANCING_SPREAD, MAX_LEVERAGE, 
                    WINDOW_START, WINDOW_END, BENCHMARK_FILE, PORTFOLIO_DIR, COSTS_DIR)

PER_TICKER_COLS = ('ivv', 'vol_scalar', 'subsystem_position',
                   'portfolio_position', 'rounded', 'held', 'trade',
                   'notional', 'pnl', 'cost', 'dividend')


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


def load_financing(index, path=BENCHMARK_FILE):
    """
    Annual financing rate on each row, as a decimal.

    FRED's DFF is in percent and covers every calendar day, but publishes a few days late, so the last rows forward-fill off the end of the file. IBKR Pro tier 1 charges this benchmark plus FINANCING_SPREAD.
    """
    raw = pd.read_csv(path, parse_dates=['observation_date'])
    rate = raw.set_index('observation_date')['DFF'] / 100
    aligned = rate.reindex(rate.index.union(index)).ffill().reindex(index)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index[0].date()
        raise ValueError(f"{path} has no rate on or before {missing}")
    return (aligned + FINANCING_SPREAD).values
    

def load_fx(index):
    """
    Home currency per unit of instrument currency on each row.

    The pair trades on days the US equity market is shut, so the series is aligned to the backtest index and carried forward.
    """
    series = fx_data.load_series()
    aligned = series.reindex(series.index.union(index)).ffill().reindex(index)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index[0].date()
        raise ValueError(f"no FX rate on or before {missing} - "
                         "run: python fx_data.py")
    return aligned.values


def round_half_away(x):
    """
    Round half away from zero. np.round uses banker's rounding, which sends
    0.5 to 0 and 1.5 to 2 - not what the framework means by rounding.
    """
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def run(frames, weights, idm, fx, capital0, vol_target, max_leverage, costs=None, income=None, rates=None):
    """
    Walk the history day by day, compounding equity.

    Within day i: yesterday's position earns today's price change, equity
    updates, then today's forecast is sized against that new equity and the
    resulting trade is paid for.
    """
    tickers = sorted(frames)
    index = frames[tickers[0]].index
    n = len(index)

    if len(fx) != n:
        raise ValueError(f"fx has {len(fx)} rows, the index has {n}")

    close = {t: frames[t]['close'].values for t in tickers}
    fcast = {t: frames[t]['combined_forecast'].values for t in tickers}
    icv = {t: frames[t]['icv'].values for t in tickers}

    income = income or {t: np.zeros(n) for t in tickers}

    rate = rates if rates is not None else np.zeros(n)

    # IBKR accrues actual/360 on the overnight balance every calendar day,
    # so a Friday row carries three days of interest into Monday.
    days = np.ones(n)
    if n > 1:
        days[1:] = np.diff(index.values).astype('timedelta64[D]').astype(float)

    cols = {t: {k: np.full(n, np.nan) for k in PER_TICKER_COLS}
            for t in tickers}
    equity = np.full(n, np.nan)
    gross_pnl = np.zeros(n)
    day_costs = np.zeros(n)
    day_financing = np.zeros(n)
    day_capped = np.zeros(n, dtype=bool)
    day_scale = np.ones(n)

    cap = capital0
    held_prev = {t: 0.0 for t in tickers}
    debt_prev = 0.0
    ruin_at = None

    for i in range(n):
        # 1. yesterday's position meets today's price, and is paid whatever went ex today
        total = 0.0
        for t in tickers:
            p = (held_prev[t] * (close[t][i] - close[t][i - 1])
                 if i > 0 else 0.0)
            d = held_prev[t] * income[t][i]
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
        want = {}

        for t in tickers:
            ivv = icv[t][i]
            vol_scalar = daily_cash_vol_target / ivv if ivv > 0 else np.nan
            subsystem = fcast[t][i] / 10 * vol_scalar
            target = subsystem * weights[t] * idm
            rounded = round_half_away(target) if np.isfinite(target) else np.nan

            current = held_prev[t]
            if np.isfinite(rounded):
                denom = max(abs(current), 1.0)
                new = rounded if abs(rounded - current) / denom > INERTIA else current
            else:
                new = current

            c = cols[t]
            c['ivv'][i] = ivv
            c['vol_scalar'][i] = vol_scalar
            c['subsystem_position'][i] = subsystem
            c['portfolio_position'][i] = target
            c['rounded'][i] = rounded
            want[t] = new

        # 2b. The broker will not carry more than max_leverage x equity overnight,
        # and the limit is on the portfolio, so it can only be applied now that 
        # every instrument has been sized.
        wanted = sum(abs(want[t] * close[t][i]) for t in tickers)
        scale = 1.0
        if wanted > max_leverage * cap:
            scale = max_leverage * cap / wanted
            for t in tickers:
                want[t] = round_half_away(want[t] * scale)

        day_capped[i] = scale < 1.0
        day_scale[i] = scale

        # 2c. Pay for the trades the capped positions imply, not the ones the 
        # forecast asked for.
        spent = 0.0
        for t in tickers:
            new = want[t]
            current = held_prev[t]
            trade = new - current
            cost = abs(trade) * costs[t] if costs and t in costs else 0.0

            c = cols[t]
            c['held'][i] = new
            c['trade'][i] = trade
            c['notional'][i] = new * close[t][i]
            c['cost'][i] = cost

            spent += cost
            held_prev[t] = new
      
        # 3. pay for the trades, and for the money borrowed overnight
        held_value = sum(abs(held_prev[t] * close[t][i]) for t in tickers)
        charge = debt_prev * rate[i] / 360 * days[i]

        day_costs[i] = spent
        day_financing[i] = charge
        cap -= spent + charge
        equity[i] = cap
        debt_prev = max(0.0, held_value - cap)

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
        {t: built[t]['notional'] for t in tickers}).abs().sum(axis=1)

    paid = pd.DataFrame({t: built[t]['dividend'] for t in tickers}).sum(axis=1)

    # the account is USD, so the home-currency curve revalues the whole
    # balance, not just the day's flows:
    #   equity_home[i] - equity_home[i-1]
    #       = net_pnl_usd[i] * conv[i] + equity_usd[i-1] * (conv[i] - conv[i-1])
    # the second term is fx_pnl, and the two together make the difference exact

    eq = equity[:end]
    conv = fx[:end]
    prev_eq = np.concatenate(([capital0], eq[:-1]))
    prev_conv = np.concatenate(([conv[0]], conv[:-1]))
    fx_pnl = prev_eq * (conv - prev_conv)   

    gross = gross_pnl[:end] * conv
    spent = day_costs[:end] * conv
    charged = day_financing[:end] * conv

    portfolio = pd.DataFrame({
        'gross_pnl': gross,
        'dividend': paid * conv,
        'cost': spent,
        'financing': charged,
        'capped': day_capped[:end],
        'cap_scale': day_scale[:end],
        'fx_pnl': fx_pnl,
        'net_pnl': gross - spent - charged + fx_pnl,
        'equity': eq * conv,
        'equity_usd': eq,
        'fx': conv,
        'gross_notional': notional * conv,
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

    spot = fx_data.load()
    cached = fx_data.describe()
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

    if WINDOW_START is not None:
        common = common[common >= pd.Timestamp(WINDOW_START)]
    if WINDOW_END is not None:
        common = common[common < pd.Timestamp(WINDOW_END)]
    if len(common) == 0:
        raise ValueError("the configured window leaves no rows")

    frames = {t: df.loc[common] for t, df in frames.items()}

    income, no_file = load_income(sorted(frames), common)
    rates = load_financing(common)
    
    fx = load_fx(common)
    capital0 = CAPITAL / fx[0]

    counts = ', '.join(f"{t} {int((income[t] > 0).sum())}"
                       for t in sorted(income))
   
    print(f"capital {CAPITAL:,.0f} {HOME_CURRENCY} = {capital0:,.0f} USD "
          f"at {fx[0]:.5f} (compounding)   "
          f"vol target {VOL_TARGET:.0%}   IDM {idm:.4f}")
    print(f"fx {cached['pair']} spot {spot:.5f} (fetched "
          f"{cached['fetched_at']})   history {fx[0]:.5f} -> {fx[-1]:.5f}")

    print(f"costs: {costs_file if costs_file else 'not measured - gross only'}")
    print(f"financing: benchmark + {FINANCING_SPREAD:.2%}, "
          f"{rates.min():.2%} to {rates.max():.2%} over the window")
    print(f"dividends: {counts}"
          + (f"  (never fetched: {', '.join(no_file)})" if no_file else ''))
    print(f"window: {common[0].date()} -> {common[-1].date()}  "
          f"({len(common)} rows)\n")

    built, portfolio, ruin_at = run(
        frames, weights, idm, fx, capital0, VOL_TARGET, MAX_LEVERAGE, costs, income, rates)

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
          f"cap {MAX_LEVERAGE:.0f}x binds on {portfolio['capped'].mean():.1%} "
          f"of days")
    if costs:
        print(f"costs paid   {portfolio['cost'].sum():,.0f} {HOME_CURRENCY} "
              f"({portfolio['cost'].sum()/years:,.0f}/yr)")
    fin = portfolio['financing'].sum()
    if fin:
        print(f"financing    {fin:,.0f} {HOME_CURRENCY}  "
              f"({fin/years:,.0f}/yr)")

    moved = portfolio['fx_pnl'].sum()
    print(f"fx           {moved:,.0f} {HOME_CURRENCY}  "
          f"({moved/years:,.0f}/yr on the whole balance)")

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
