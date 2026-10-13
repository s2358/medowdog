"""
Spot rate and daily history for the FX pair.

    python fx_data.py                   # refresh both
    from fx_data import load            # cached spot, to size today's position
    from fx_data import load_series     # daily history, to convert past P&L

Two different jobs. The spot cache is the rate a position is sized at right now. The series is the rate each historical day's P&L was earned at.

The cache exists so the backtest doesn't need IB Gateway running. ib_async is imported inside the fetch functions for the same reason - reading stored data must not require the broker library.
"""

import json
import datetime
import pandas as pd
import bars
from config import (FX_PAIR, FX_CACHE, FX_SERIES, FX_MAX_AGE_DAYS, DATA_DIR, IB_HOST, IB_PORT, IB_MARKET_DATA_TYPE)


CLIENT_ID = 22
QUOTE_TIMEOUT = 10

def load(path=FX_CACHE, max_age_days=FX_MAX_AGE_DAYS):
    """
    Return the cached rate (home currency per unit of instrument currency).
    Raises rather than silently returning a stale number - a wrong FX rate corrupts every position size     with no visible symptoms
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run: python fx_data.py")

    data = json.loads(path.read_text())
    fetched = datetime.datetime.fromisoformat(data['fetched_at'])
    age = (datetime.datetime.now() - fetched).days

    if age > max_age_days:
        raise ValueError(
                        f"{data['pair']} rate is {age} days old (fetched {data['fetched_at']}, "
                        f"max {max_age_days}) - run: python fx_data.py")

    return float(data['rate'])

def describe(path=FX_CACHE):
    """The cached entry in full, for printing provenance."""
    return json.loads(path.read_text())

def fetch(pair=FX_PAIR):
    """Connect to IB Gateway and return (mid, bid, ask)."""
    from ib_async import IB, Forex
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, readonly=True)
    try:
        ib.reqMarketDataType(IB_MARKET_DATA_TYPE)
        contract = Forex(pair)
        ib.qualifyContracts(contract)
        ticker = ib.reqMktData(contract, '', False, False)

        waited = 0.0
        while waited < QUOTE_TIMEOUT:
            ib.sleep(0.5)
            waited += 0.5
            if ticker.bid and ticker.ask and ticker.bid > 0:
                break
        bid, ask = ticker.bid, ticker.ask
        ib.cancelMktData(contract)
    finally:
        ib.disconnect()

    if not (bid and ask and bid > 0 and ask > bid):
        raise RuntimeError(
            f"no valid {pair} quote after {QUOTE_TIMEOUT}s (bid={bid}, ask={ask}) - "
            "FX spot needs its own IBKR data permission, separate from equities")
    
    return (bid + ask) / 2, bid, ask

def save(rate, bid, ask, pair=FX_PAIR, path=FX_CACHE):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'pair': pair,
        'rate': rate,
        'bid': bid,
        'ask': ask,
        'fetched_at': datetime.datetime.now().isoformat(timespec='seconds'),
        }, indent=2))


def load_series(path=FX_SERIES):
    """Daily rate of the pair, indexed by date."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run: python fx_data.py")
    df = pd.read_csv(path, parse_dates=['date'])
    return df.set_index('date')['rate']


def fetch_series(pair=FX_PAIR):
    """Daily bars for the pair, as (DataFrame, complete)."""
    from ib_async import IB, Forex

    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, readonly=True)
    try:
        contract = Forex(pair)
        ib.qualifyContracts(contract)
        # cash forex has no consolidated tape, so TRADES returns nothing.
        # IDEALPRO runs Sunday 17:00 to Friday 17:00 ET, and useRTH=True cuts
        # that down to a session the pair does not keep.
        frame, complete = bars.daily(ib, contract, 'MIDPOINT', use_rth=False, label=pair)
    finally:
        ib.disconnect()

    return frame[['date', 'close']].rename(columns={'close': 'rate'}), complete


def save_series(frame, path=FX_SERIES):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def refresh_spot():
    try:
        rate, bid, ask = fetch()
    except Exception as exc:
        if not FX_CACHE.exists():
            raise
        cached = describe()
        fetched = datetime.datetime.fromisoformat(cached['fetched_at'])
        age = (datetime.datetime.now() - fetched).days
        if age > FX_MAX_AGE_DAYS:
            raise
        print(f"fetch failed: {exc}")
        print(f"keeping cached {cached['pair']} {cached['rate']:.5f} from "
              f"{cached['fetched_at']} ({age}d old, max {FX_MAX_AGE_DAYS})")
        return

    save(rate, bid, ask)

    spread_bp = (ask - bid) / rate * 1e4
    print(f"{FX_PAIR}  bid {bid:.5f}  ask {ask:.5f}  mid {rate:.5f}  "
          f"spread {spread_bp:.1f} bp")
    print(f"written to {FX_CACHE}")


def refresh_series():
    try:
        frame, complete = fetch_series()
    except Exception as exc:
        if not FX_SERIES.exists():
            raise
        print(f"series fetch failed: {exc}")
        print(f"keeping {FX_SERIES}")
        return

    if not complete:
        if not FX_SERIES.exists():
            raise RuntimeError(
                    f"{FX_PAIR} download did not reach the start of the history "
                    f"and there is no stored series to keep")
        print(f"{FX_PAIR} NOT SAVED - the walk did not reach the start of "
              f"the history, keeping {FX_SERIES}")
        return

    save_series(frame)
    print(f"{FX_PAIR}  {len(frame):5d} daily bars {frame['date'].iloc[0]} -> "
          f"{frame['date'].iloc[-1]}  written to {FX_SERIES}")


def main():
    refresh_spot()
    refresh_series()



if __name__ == '__main__':
    main()
