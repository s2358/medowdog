"""
Report the broker's account value.

    python account.py

A read-only look at what the account is worth. capital.py records the figure
the system sizes against and is what the pipeline runs; this prints the
broker's numbers in full, for checking that record by hand.

Net liquidation value is the figure to use: cash plus the market value of
everything held, in the account's base currency.

The connection is read-only.
"""

import sys
from datetime import datetime

from config import IB_HOST, IB_PORT

CLIENT_ID = 33

TAGS = ('NetLiquidation', 'TotalCashValue', 'GrossPositionValue', 'AvailableFunds', 'ExcessLiquidity', 'BuyingPower')

def summary(host=IB_HOST, port=IB_PORT):
    """Account numbers and the summary values the broker reports."""
    from ib_async import IB

    ib = IB()
    try:
        ib.connect(host, port, clientId=CLIENT_ID, readonly=True)
        accounts = sorted(ib.managedAccounts())
        values = ib.accountSummary()
    finally:
        ib.disconnect()
    return accounts, values


def main():
    accounts, values = summary()
    print(f"{datetime.now().isoformat(timespec='seconds')} "
          f"account {', '.join(accounts) if accounts else 'none reported'}")

    found = {v.tag: v for v in values if v.tag in TAGS}
    if not found:
        print("no account summary returned")
        sys.exit(1)

    for tag in TAGS:
        value = found.get(tag)
        if value is None:
            continue
        print(f"  {tag:20s} {float(value.value):14,.2f} {value.currency}")


if __name__ == '__main__':
    main()
