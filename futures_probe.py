"""
What futures history IBKR will actually give us.

    python futures_probe.py MCL 10Y MJY
    python futures_probe.py CL 6J --max=6

Lists every contract IBKR knows for each root, expired ones included, then
asks each expired one for a year of daily bars ending on its last trading day
and reports what came back.

The question this answers is how far back per-expiry data survives. The micros
list nothing before late 2024, which is why the full-size contracts - CL for
WTI, 6J for yen - have to be probed as well: they are the same market in the
same units, differing only in how much one contract holds.

Read-only. Nothing is written and no order is placed.
"""

import os
import sys
import time
from datetime import datetime

from ib_async import IB, Future

import universe
from config import IB_HOST, IB_PORT, IB_MARKET_DATA_TYPE

CLIENT_ID = 41
PAUSE = 2.0     # seconds between historical requests, to stay inside pacing
DEFAULT_MAX = 16

# roots that are not in the universe, so the full-size contracts can be
# probed for the history the micros do not have
EXTRA = {
    'CL': {'exchange': 'NYMEX', 'currency': 'USD'},
    'JPY': {'exchange': 'CME', 'currency': 'USD'},
    'ZN': {'exchange': 'CBOT', 'currency': 'USD'},
    'GC': {'exchange': 'COMEX', 'currency': 'USD'},
}


def spec(ticker):
    """The instrument record, or a bare exchange for a root we do not trade."""
    try:
        return universe.instrument(ticker)
    except KeyError:
        if ticker not in EXTRA:
            raise
        return EXTRA[ticker]


def stub(ticker):
    """A root contract with no expiry, which lists every delivery month."""
    detail = spec(ticker)
    return Future(symbol=ticker,
                  exchange=detail['exchange'],
                  currency=detail['currency'],
                  includeExpired=True)


def listed(ib, ticker):
    """Every contract IBKR knows for this root, oldest expiry first."""
    details = ib.reqContractDetails(stub(ticker))
    rows = [(d.contract.lastTradeDateOrContractMonth,
             d.contract.localSymbol,
             d.contract.conId,
             d.contract.multiplier) for d in details]
    return sorted(rows)


def probe(ib, ticker, last_trade, con_id):
    """
    A year of daily bars ending on this contract's last trading day.

    The contract is addressed by conId so there is no ambiguity about which
    delivery month answered, and includeExpired is required or an expired
    contract is refused outright.
    """
    detail = spec(ticker)
    contract = Future(symbol=ticker,
                      exchange=detail['exchange'],
                      currency=detail['currency'],
                      lastTradeDateOrContractMonth=last_trade,
                      includeExpired=True)
    contract.conId = con_id
    bars = ib.reqHistoricalData(
            contract,
            endDateTime=f'{last_trade} 23:59:59',
            durationStr='1 Y',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1)
    if not bars:
        return None
    return bars[0].date, bars[-1].date, len(bars)


def report(ib, ticker, how_many):
    rows = listed(ib, ticker)
    if not rows:
        print(f"{ticker}: IBKR lists no contracts")
        return

    today = datetime.now().strftime('%Y%m%d')
    past = [r for r in rows if r[0] <= today]

    spans = {m for _, _, _, m in rows}
    print(f"\n{ticker}: {len(rows)} contracts listed, "
          f"{rows[0][0]} to {rows[-1][0]}, multiplier {sorted(spans)}")
    print(f"  {len(past)} already expired, "
          f"oldest {past[0][0] if past else '-'}")

    if not past:
        print("  nothing expired to probe")
        return

    # newest expired first, since the question is where the data runs out
    sample = past[::-1][:how_many]
    print(f"probing the newest {len(sample)} expired:\n")
    print(f"  {'expiry':10s} {'symbol':10s} {'bars':>5s}  first        last")

    for last_trade, symbol, con_id, _ in sample:
        got = probe(ib, ticker, last_trade, con_id)
        if got is None:
            print(f"  {last_trade:10s} {symbol:10s} {0:5d}  no data")
        else:
            first, last, n = got
            print(f"  {last_trade:10s} {symbol:10s} {n:5d}  {first}   {last}")
        time.sleep(PAUSE)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    how_many = DEFAULT_MAX
    for a in sys.argv[1:]:
        if a.startswith('--max'):
            how_many = int(a.split('=')[1]) if '=' in a else DEFAULT_MAX

    if not args:
        print(f"Usage: python {os.path.basename(sys.argv[0])} "
              "<ROOT> [<ROOT> ...] [--max=N]")
        sys.exit(1)

    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID)
    ib.reqMarketDataType(IB_MARKET_DATA_TYPE)
    try:
        for ticker in args:
            report(ib, ticker, how_many)
    finally:
        ib.disconnect()
    print()


if __name__ == '__main__':
    main()
