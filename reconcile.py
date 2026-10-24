"""
Check that the broker's positions are explained by the executions recorded.

    python reconcile.py

Each position is the signed sum of the fills recorded for that instrument: a
buy adds, a sell subtracts. If the broker agrees, every share held has a
recorded execution behind it and nothing has moved unobserved.

This is not a test against the target. An order can fill partly or not at
all, which leaves the position short of its target by design, and the next
order run closes the gap because it trades the difference between the target
and what is actually held. The distance to target is reported below as
information, not as a fault.

What the test does catch is a share that changed hands without an execution
to explain it: an order placed outside the system, a dividend taken in
shares, a corporate action. Any of those breaks the assumption that the
recorded history is complete, and every later trade inherits the error.

Exits non-zero on any mismatch.

The connection is read-only.
"""

import sys
from datetime import datetime

import pandas as pd

from config import (PORTFOLIO_DIR, FILLS_DIR, IB_HOST, IB_PORT)
from universe import WEIGHTS


CLIENT_ID = 31
SETTLE = 1.0        # seconds to let the position stream populate

FILLS_FILE = 'fills.csv'
TARGETS_FILE = 'live_targets.csv'


def recorded_positions():
    """
    Shares accounted for by recorded executions, by ticker, and how many
    executions were read.

    No file means nothing has been traded, in which case every position is
    expected to be flat.
    """
    path = FILLS_DIR / FILLS_FILE
    if not path.exists():
        return {}, 0

    frame = pd.read_csv(path)
    signed = frame['shares'].where(frame['side'].str.upper() == 'BUY',
                                   -frame['shares'])
    totals = signed.groupby(frame['ticker'].str.upper()).sum()
    return {t: int(n) for t, n in totals.items()}, len(frame)


def target_positions(tickers):
    """
    Shares the system intends to hold, by ticker, with the date they were
    computed. Reported but not tested.
    """
    path = PORTFOLIO_DIR / TARGETS_FILE
    if not path.exists():
        return {}, None
    frame = pd.read_csv(path, parse_dates=['date']).set_index('ticker')
    dates = {frame.loc[t, 'date'].date() for t in tickers if t in frame.index}
    return ({t: int(frame.loc[t, 'rounded'])
             for t in tickers if t in frame.index},
            dates.pop() if len(dates) == 1 else None)


def broker_positions(host=IB_HOST, port=IB_PORT):
    """
    Shares IBKR reports holding, by symbol, plus the accounts seen and any
    non-share positions found.

    Currency balances arrive here as positions too; they are listed rather
    than compared, because the system tracks shares only.
    """
    from ib_async import IB

    ib = IB()
    try:
        ib.connect(host, port, clientId=CLIENT_ID, readonly=True)
        accounts = set(ib.managedAccounts())
        ib.sleep(SETTLE)
        positions = ib.positions()
    finally:
        ib.disconnect()

    shares, other = {}, []
    for p in positions:
        accounts.add(p.account)
        if p.contract.secType != 'STK':
            other.append(f"{p.contract.localSymbol} "
                         f"({p.contract.secType}) {p.position:g}")
            continue
        symbol = p.contract.symbol.upper()
        shares[symbol] = shares.get(symbol, 0) + int(p.position)
    return shares, sorted(accounts), other


def main():
    tickers = sorted(WEIGHTS)
    recorded, executions = recorded_positions()
    broker, accounts, other = broker_positions()

    print(f"{datetime.now().isoformat(timespec='seconds')} "
          f"account {', '.join(accounts) if accounts else 'none reported'}")
    print(f"{executions} execution(s) recorded\n")

    mismatches = 0
    for symbol in sorted(set(tickers) | set(broker) | set(recorded)):
        explained = recorded.get(symbol, 0)
        have = broker.get(symbol, 0)
        known = '' if symbol in WEIGHTS else '   NOT IN THE SYSTEM'
        if explained == have and not known:
            print(f"{symbol:6s} fills {explained:+6d}  "
                  f"broker {have:+6d}   ok")
        else:
            mismatches += 1
            print(f"{symbol:6s} fills {explained:+6d}  broker {have:+6d}   "
                  f"MISMATCH {have - explained:+d}{known}")

    if other:
        print("\nnot shares, ignored: " + "; ".join(other))

    if len(accounts) > 1:
        print(f"\n{len(accounts)} accounts reported - the comparison pooled "
              f"them, which is only right if the system owns all of them")

    if mismatches:
        print(f"\n{mismatches} mismatch(es) - stopping")
        sys.exit(1)

    print(f"\nreconciled: every share held is explained by a recorded fill")

    target, asof = target_positions(tickers)
    if not target:
        return

    print(f"\ndistance to the targets of {asof}:")
    for ticker in tickers:
        have = broker.get(ticker, 0)
        want = target.get(ticker, 0)
        gap = want - have
        print(f"{ticker:6s} holding {have:+6d}  target {want:+6d}   "
              + ("flat" if gap == 0 else f"{gap:+d} to trade"))


if __name__ == '__main__':
    main()
