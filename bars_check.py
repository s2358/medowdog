"""
Which historical request shapes work for one instrument.

    python bars_check.py TLT
    python bars_check.py TLT NASDAQ

A sweep of durations, each a single request ending now, so a failure points
at the duration rather than at the paging walk in bars.py. IBKR enforces the
per-request limit per contract, not globally, so an instrument that refuses
'25 Y' can still answer a shorter window.

The second argument asks the primary exchange directly instead of routing
through SMART.

Read-only.
"""

import os
import sys
import time

from ib_async import IB, Stock

import universe
from config import IB_HOST, IB_PORT, IB_MARKET_DATA_TYPE

CLIENT_ID = 42
PAUSE = 2.0

DURATIONS = ('1 M', '1 Y', '5 Y', '10 Y', '15 Y', '20 Y', '25 Y')

# run once the duration limit is known, to confirm the other request shapes
# the system uses are also served - dividends.py needs ADJUSTED_LAST
SHAPES = (
    ('TRADES, regular hours', 'TRADES', True),
    ('TRADES, all hours', 'TRADES', False),
    ('ADJUSTED_LAST', 'ADJUSTED_LAST', True),
    ('MIDPOINT', 'MIDPOINT', True),
    ('BID_ASK', 'BID_ASK', True),
)


def try_one(ib, contract, label, what, rth, duration):
    try:
        bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting='1 day',
                whatToShow=what,
                useRTH=rth,
                formatDate=1)
    except Exception as exc:
        print(f"  {label:24s} raised {type(exc).__name__}: {exc}")
        return
    if not bars:
        print(f"  {label:24s} empty")
    else:
        print(f"  {label:24s} {len(bars):5d} bars  "
              f"{bars[0].date} -> {bars[-1].date}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} "
              "<TICKER> [EXCHANGE]")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    route = sys.argv[2].upper() if len(sys.argv) > 2 else None

    entry = universe.instrument(ticker)
    contract = Stock(ticker, route or entry['exchange'], entry['currency'])

    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    ib.reqMarketDataType(IB_MARKET_DATA_TYPE)
    try:
        details = ib.reqContractDetails(contract)
        print(f"\n{ticker} via {route or entry['exchange']}: "
              f"{len(details)} contract(s) resolve")
        for d in details:
            c = d.contract
            print(f"  conId {c.conId}  {c.primaryExchange or c.exchange}  "
                  f"{c.localSymbol}  {c.currency}")

        print("\nduration sweep, TRADES, regular hours:")
        for duration in DURATIONS:
            try_one(ib, contract, duration, 'TRADES', True, duration)
            time.sleep(PAUSE)

        print("\nrequest shapes at 1 M:")
        for label, what, rth in SHAPES:
            try_one(ib, contract, label, what, rth, '1 M')
            time.sleep(PAUSE)
    finally:
        ib.disconnect()
    print()


if __name__ == '__main__':
    main()
