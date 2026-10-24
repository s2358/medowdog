"""Cancel working orders at the broker.

Reports every order the broker currently has working and, with --cancel,
sends a cancellation for each. Dry by default. An API order can only be
cancelled by the client that placed it, so this connects under the same
identifier as orders.py; --global bypasses ownership and cancels every
working order on the account. Run before orders.py, which derives trades
from positions and does not inspect working orders.

Usage:
    python cancel.py [--cancel] [--global]
"""

import sys

from ib_async import IB

from config import IB_HOST, IB_PORT

CLIENT_ID = 32          # must match orders.py - cancels are owned by client
WORKING = {'ApiPending', 'PendingSubmit', 'PreSubmitted', 'Submitted'}


def connect(send):
    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, readonly=not send)
    return ib


def working_orders(ib):
    """ Every working order on the account, not just this client's."""
    ib.reqAllOpenOrders()
    ib.sleep(1)
    return [trade for trade in ib.trades()
            if trade.orderStatus.status in WORKING]


def describe(trade):
    order = trade.order
    return (f"{order.orderId:>6}  {trade.contract.symbol:6s} "
            f"{order.action:4s} {int(order.totalQuantity):5d} "
            f"@ {order.lmtPrice:8.2f}  {trade.orderStatus.status}")


def main():
    send = '--cancel' in sys.argv
    everything = '--global' in sys.argv

    ib = connect(send)
    try:
        trades = working_orders(ib)

        if not trades:
            print("no working orders")
            return 0

        for trade in trades:
            print(describe(trade))

        if not send:
            print(f"\n{len(trades)} working - pass --cancel to cancel them")
            return 0

        if everything:
            ib.reqGlobalCancel()
        else:
            for trade in trades:
                ib.cancelOrder(trade.order)
        ib.sleep(2)

        remaining = working_orders(ib)
        print(f"\ncancelled {len(trades) - len(remaining)} of {len(trades)}")
        for trade in remaining:
            print(f"still working: {describe(trade)}")
        return 1 if remaining else 0
    finally:
        ib.disconnect()


if __name__ == '__main__':
    sys.exit(main())
