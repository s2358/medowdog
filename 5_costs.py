"""
Measure trading costs from live bid/ask spreads.

python 5_costs.py data/SPY/*.csv data/GLDM/*.csv data/SLV/*.csv

Must run during US market hours - the quote stream returns IBKR's -1 sentinel when the market is closed, which the pre-flight probe catches. 

Trade sizes come from the turnover file the position sizing script writes. The $1 commission minimum is divided by them, so they dominate the result: at 5 shares the minimum is $0.20/share, at 23 it is $0.04 
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

from ib_insync import IB, Stock

from config import (IB_HOST, IB_PORT, IB_MARKET_DATA_TYPE, WARMUP, BUSINESS_DAYS, COMMISSION_PER_SHARE, COMMISSION_MINIMUM, COMMISSION_MAX_PCT, SPEED_LIMIT_SR, COSTS_DIR, PORTFOLIO_DIR, WEIGHTS)

CLIENT_ID = 21

ANNUALISE = np.sqrt(BUSINESS_DAYS)

SAMPLES = 20
SAMPLE_INTERVAL = 15        # seconds
PROBE_TIMEOUT = 15          # seconds to wait for the first quote

SPINNER = '|/-\\'
ANIMATE = sys.stdout.isatty()   # no spinner when redirected to a log

def ticker_from_path(path):
    return os.path.basename(path).split('_')[0].upper()

def latest_icv(path):
    """ICV = ewma_vol x close: the volatility of one share, in USD."""
    df = pd.read_csv(path, usecols=['date', 'close', 'ewma_vol']).iloc[WARMUP:]
    row = df.iloc[-1]
    return row['ewma_vol'] * row['close'], row['date'], row['close']

def wait_for_quote(ib, ticker, timeout=PROBE_TIMEOUT):
    """Poll until a usable quote arrives, or the timeout expires."""
    waited = 0.0
    while waited < timeout:
        ib.sleep(0.5)
        waited += 0.5
        if ticker.bid and ticker.ask and ticker.bid > 0:
            return True
    return False

def sample_quotes(ib, symbol, samples=SAMPLES, interval=SAMPLE_INTERVAL):
    """Sample bid/ask repeatedly; return the list of spreads seen"""
    contract = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(contract)
    ticker = ib.reqMktData(contract, '', False, False)

    spreads = []
    tick = 0
    total_secs = samples * interval

    for i in range(samples):
        # animate during the wait instead of blocking entirely
        waited = 0.0
        while waited < interval:
            ib.sleep(0.25)
            waited += 0.25
            tick += 1
            if ANIMATE:
                elapsed = i * interval + waited
                remaining = total_secs - elapsed
                quote = (f"{ticker.bid:.2f}/{ticker.ask:.2f}"
                     if ticker.bid and ticker.ask else "  --/--  "
                     )
                print(f"\r {SPINNER[tick % 4]} {symbol:6s}  "
                  f"sample {i+1:2d}/{samples}  "
                  f"quote {quote}  "
                  f"captured {len(spreads):2d}  "
                  f"{remaining:3.0f}s left  ",
                  end='', flush=True
                    )
        bid, ask = ticker.bid, ticker.ask
        if bid and ask and bid > 0 and ask > bid:
            spreads.append(ask - bid)

    if ANIMATE:
        print("\r" + " " * 78 + "\r", end='')
    ib.cancelMktData(contract)
    return spreads

def commission_per_share(shares, price):
    total = max(COMMISSION_MINIMUM, shares * COMMISSION_PER_SHARE)
    total = min(total, COMMISSION_MAX_PCT * shares * price)
    return total / shares

def standardised_cost(half_spread, comm_per_share, icv):
    """ (2 x C / (annualised ICV), in annual Sharpe ratio units per round trip)"""
    C = half_spread + comm_per_share
    return 2 * C / (ANNUALISE * icv), C

def load_typical_trade():
    """
    Trade sizes measured by the position sizing script. The $1 commission minimum is divided by these, so they drive the whole result - reading them beats keeping a hand-copied dict in sync.
    """
    path = PORTFOLIO_DIR / f"turnover_{'_'.join(sorted(WEIGHTS))}.csv"
    if not path.exists():
        raise FileNotFoundError(
                f"{path} not found - run the position sizing script first")
    series = pd.read_csv(path).set_index('ticker')['typical_trade']
    return {t: max(1, int(round(v))) for t, v in series.items()}, path.name

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} " "<instrument_csv> [<instrument_csv> ...]")
        sys.exit(1)

    paths = sys.argv[1:]
    tickers = [ticker_from_path(p) for p in paths]

    typical_trade, trade_file = load_typical_trade()
    missing = [t for t in tickers if t not in typical_trade]
    if missing:
        raise KeyError(f"no typical_trade entry for: {missing}")

    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, readonly=True)
    ib.reqMarketDataType(IB_MARKET_DATA_TYPE)         # 1 = live, 3 = delayed (spreads will be stale)

    # pre-flight: if the market is closed, fail in seconds rather than 
    # grinding through samples x interval x tickers of dead air
    probe = Stock(tickers[0], 'SMART', 'USD')
    ib.qualifyContracts(probe)
    probe_ticker = ib.reqMktData(probe, '', False, False)
    live = wait_for_quote(ib, probe_ticker)
    bid, ask = probe_ticker.bid, probe_ticker.ask
    ib.cancelMktData(probe)

    if not live:
        print(f"no quotes after {PROBE_TIMEOUT}s (bid={bid}, ask={ask}) - market likely closed, aborting")
        ib.disconnect()
        sys.exit(1)

    print(f"{datetime.now().isoformat(timespec='seconds')} "
          f"market open, sampling {len(tickers)} instruments "
          f"({SAMPLES} X {SAMPLE_INTERVAL}s each)")

    print(f"trade sizes from {trade_file}: "
          + ", ".join(f"{t} {typical_trade[t]}" for t in tickers))

    rows = []
    try:
        for path, ticker in zip(paths, tickers):
            icv, asof, close = latest_icv(path)
            spreads = sample_quotes(ib, ticker)

            if not spreads:
                print(f"{ticker}: no valid quotes - market closed, or no data subscription")
                continue

            spread = float(np.median(spreads))
            half = spread / 2
            shares = typical_trade[ticker]
            comm = commission_per_share(shares, close)
            sc, C = standardised_cost(half, comm, icv)

            rows.append({
                'ticker': ticker,
                'sampled_at': datetime.now().isoformat(timespec='seconds'),
                'vol_asof': asof,
                'n_quotes': len(spreads),
                'spread_median': round(spread, 4),
                'spread_min': round(min(spreads), 4),
                'spread_max': round(max(spreads), 4),
                'half_spread': round(half, 5),
                'typical_shares': shares,
                'commission_per_share': round(comm, 5),
                'cost_per_share_C': round(C, 5),
                'icv': round(icv, 4),
                'annualised_icv': round(ANNUALISE * icv, 3),
                'standardised_cost': round(sc, 6),
                'speed_limit_turnover': round(SPEED_LIMIT_SR / sc, 1),
                'trade_sizes_from': trade_file,
            })
            print(f"{ticker:6s} spread {spread:.4f} C {C:.5f} "
                  f"std cost {sc:.5f} SR speed limit {SPEED_LIMIT_SR/sc:.0f}/yr")
    finally:
        ib.disconnect()

    if not rows:
        print("nothing collected")
        sys.exit(1)

    out = pd.DataFrame(rows)
    COSTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = COSTS_DIR / f'costs_{stamp}.csv'
    out.to_csv(out_path, index=False)
    print(f"\nwritten to {out_path}")

if __name__ == '__main__':
    main()



