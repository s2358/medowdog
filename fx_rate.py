"""
Fetch and cache the FX rate.

Run this to refresh: python fx_rate.py
Read it from a script: from fx_rate import load

The cache exists so that the backtest doesn't need IB gateway running. ib_async is imported inside fetch() for the same reason - reading the cached rate must not require the broker library to be installed.
"""

import json
import datetime
from config import (FX_PAIR, FX_CACHE, FX_MAX_AGE_DAYS, DATA_DIR, IB_HOST, IB_PORT, IB_MARKET_DATA_TYPE)


CLIENT_ID = 22
QUOTE_TIMEOUT = 10

def load(path=FX_CACHE, max_age_days=FX_MAX_AGE_DAYS):
    """
    Return the cached rate (home currency per unit of instrument currency).
    Raises rather than silently returning a stale number - a wrong FX rate corrupts every position size     with no visible symptoms
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run: python fx_rate.py")

    data = json.loads(path.read_text())
    fetched = datetime.datetime.fromisoformat(data['fetched_at'])
    age = (datetime.datetime.now() - fetched).days

    if age > max_age_days:
        raise ValueError(
                        f"{data['pair']} rate is {age} days old (fetched {data['fetched_at']}, "
                        f"max {max_age_days}) - run: python fx_rate.py")

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

def main():
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

if __name__ == '__main__':
    main()
