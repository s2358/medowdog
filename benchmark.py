"""
Buy and hold benchmark on the same instruments.

    python benchmark.py <allocation_csv> <instrument_csv> [<instrument_csv> ...]

Fixed weights, no leverage, rebalanced back to those weights on the first trading day of each year. Whole shares only, so a little cash sits idle.

Capital, window, weights, costs, dividends and exchange rates are the ones the trading system uses, leaving the strategy as the only difference.

Writes benchmark.csv with the same columns as portfolio.csv:
    python 7_evaluate.py data/portfolio/benchmark.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import dividends
import fx_data
from config import (CAPITAL, HOME_CURRENCY, BUSINESS_DAYS, WARMUP, US_WITHHOLDING, PORTFOLIO_DIR, COSTS_DIR)


def ticker_from_path(path):
    return os.path.basename(path).split('_')[0].upper()


def load_prices(path):
    df = pd.read_csv(path, usecols=['date', 'close']).iloc[WARMUP:]
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')['close']


def load_weights(path):
    return pd.read_csv(path).set_index('ticker')['instrument_weight'].to_dict()


def load_costs():
    """Latest measured cost per share, per ticker. None if never measured."""
    files = sorted(COSTS_DIR.glob('costs_*.csv')) if COSTS_DIR.exists() else []
    if not files:
        return None, None
    latest = files[-1]
    series = pd.read_csv(latest).set_index('ticker')['cost_per_share_C']
    return series.to_dict(), latest.name


def load_income(tickers, index):
    """Net dividend per share on each row, withholding already deducted."""
    out = {}
    for ticker in tickers:
        series = dividends.load(ticker)
        paid = np.zeros(len(index))
        if series is not None and not series.empty:
            net = series * (1 - US_WITHHOLDING)
            for position, amount in zip(index.searchsorted(net.index),
                                        net.values):
                if position < len(index):
                    paid[position] += amount
        out[ticker] = paid
    return out


def load_fx(index):
    series = fx_data.load_series()
    aligned = series.reindex(series.index.union(index)).ffill().reindex(index)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index[0].date()
        raise ValueError(f"no FX rate on or before {missing} - "
                         "run: python fx_data.py")
    return aligned.values


def run(prices, weights, capital0, costs=None, income=None):
    """
    Walk the history day by day.

    Holdings are whole shares and cash is never negative, so equity is only ever
    the value of what is owned plus what was left over.
    """

    tickers = sorted(prices)
    index = prices[tickers[0]].index
    n = len(index)
    close = {t: prices[t].values for t in tickers}
    income = income or {t: np.zeros(n) for t in tickers}

    held = {t: 0.0 for t in tickers}
    cash = capital0

    equity = np.zeros(n)
    gross_pnl = np.zeros(n)
    day_income = np.zeros(n)
    day_costs = np.zeros(n)
    rebalanced = np.zeros(n, dtype=bool)

    for i in range(n):
        # 1. yesterday's holding meets today's price, and is paid what went ex
        change, paid = 0.0, 0.0
        for t in tickers:
            if i > 0:
                change += held[t] * (close[t][i] - close[t][i - 1])
            paid += held[t] * income[t][i]
        cash += paid
        day_income[i] = paid
        gross_pnl[i] = change + paid

        # 2. on the first row and at each turn of the year, buy back to weight
        if i == 0 or index[i].year != index[i -1].year:
            value = cash + sum(held[t] * close[t][i] for t in tickers)
            spend = 0.0
            for t in tickers:
                target = np.floor(value * weights[t] / close[t][i])
                trade = target - held[t]
                cash -= trade * close[t][i]
                if costs and t in costs:
                    spend += abs(trade) * costs[t]
                held[t] = target
            cash -= spend
            day_costs[i] = spend
            rebalanced[i] = True

        equity[i] = cash + sum(held[t] * close[t][i] for t in tickers)

    return index, equity, gross_pnl, day_income, day_costs, rebalanced


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {os.path.basename(sys.argv[0])} "
              "<allocation_csv> <instrument_csv> [<instrument_csv> ...]")
        sys.exit(1)

    weights = load_weights(sys.argv[1])
    prices = {ticker_from_path(p): load_prices(p) for p in sys.argv[2:]}

    missing = set(prices) - set(weights)
    if missing:
        raise KeyError(f"no weight in allocation file for: {sorted(missing)}")

    # restrict every instrument to the common trading days
    common = None
    for series in prices.values():
        common = (series.index if common is None
                  else common.intersection(series.index))
    prices = {t: s.loc[common] for t, s in prices.items()}

    costs, costs_file = load_costs()
    income = load_income(sorted(prices), common)
    fx = load_fx(common)
    capital0 = CAPITAL / fx[0]

    print(f"capital {CAPITAL:,.0f} {HOME_CURRENCY} = {capital0:,.0f} USD "
      f"at {fx[0]:.5f}")
    print("weights  " + "   ".join(f"{t} {weights[t]:.0%}"
                                   for t in sorted(prices)))
    print(f"costs: {costs_file if costs_file else 'not measured - gross only'}")
    print(f"window: {common[0].date()} -> {common[-1].date()}  "
          f"({len(common)} rows)\n")

    index, equity, gross_pnl, day_income, day_costs, rebalanced = run(
        prices, weights, capital0, costs, income)

    # the account is USD, so the home-currency curve revalues the whole
    # balance, not just the day's flows - the second term below is fx_pnl
    prev_eq = np.concatenate(([capital0], equity[:-1]))
    prev_conv = np.concatenate(([fx[0]], fx[:-1]))
    fx_pnl = prev_eq * (fx - prev_conv)

    gross = gross_pnl * fx
    spend = day_costs * fx

    portfolio = pd.DataFrame({
        'gross_pnl': gross,
        'dividend': day_income * fx,
        'cost': spend,
        'financing': np.zeros(len(index)),
        'fx_pnl': fx_pnl,
        'net_pnl': gross - spend + fx_pnl,
        'equity': equity * fx,
        'equity_usd': equity,
        'fx': fx,
        'rebalanced': rebalanced,
    }, index=index)
    portfolio.index.name = 'date'

    years = len(portfolio) / BUSINESS_DAYS
    final = portfolio['equity'].iloc[-1]
    print(f"equity   start {CAPITAL:,.0f}   end {final:,.0f}   "
          f"min {portfolio['equity'].min():,.0f}   "
          f"max {portfolio['equity'].max():,.0f} {HOME_CURRENCY}")
    print(f"cagr     {(final / CAPITAL) ** (1 / years) - 1:.2%}")
    print(f"rebalances   {int(rebalanced.sum())}")
    print(f"costs paid   {portfolio['cost'].sum():,.0f} {HOME_CURRENCY}")
    print(f"dividends    {portfolio['dividend'].sum():,.0f} {HOME_CURRENCY}")
    print(f"fx           {portfolio['fx_pnl'].sum():,.0f} {HOME_CURRENCY}")

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    out = PORTFOLIO_DIR / 'benchmark.csv'
    portfolio.to_csv(out)
    print(f"\nwritten to {out}")


if __name__ == '__main__':
    main()
