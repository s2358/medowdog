"""
Report the broker's account value, and check the financing model against it.

    python account.py

A read-only look at what the account is worth. capital.py records the figure
the system sizes against and is what the pipeline runs; this prints the
broker's numbers in full, for checking that record by hand.

Net liquidation value is the figure to use: cash plus the market value of
everything held, in the account's base currency.

It also prints what the backtest's financing model would charge on today's
book, next to what the broker has actually accrued. The model charges the
benchmark rate plus FINANCING_SPREAD, actual/360, on gross position value
less equity - which counts a short position as borrowed money. A short in
fact generates cash and carries a borrow fee instead, so the two numbers are
not expected to agree, and the gap is the thing worth knowing.

The connection is read-only.
"""

import sys
from datetime import datetime

import pandas as pd

from config import IB_HOST, IB_PORT, BENCHMARK_FILE, FINANCING_SPREAD

CLIENT_ID = 33

TAGS = ('NetLiquidation', 'TotalCashValue', 'GrossPositionValue',
        'AvailableFunds', 'ExcessLiquidity', 'BuyingPower',
        'AccruedCash', 'AccruedDividend',
        'FullInitMarginReq', 'FullMaintMarginReq')


def benchmark_rate(path=BENCHMARK_FILE):
    """The latest benchmark rate plus the broker's spread, as a decimal."""
    raw = pd.read_csv(path, parse_dates=['observation_date'])
    last = raw.dropna(subset=['DFF']).iloc[-1]
    return last['DFF'] / 100 + FINANCING_SPREAD, last['observation_date'].date()


def summary(host=IB_HOST, port=IB_PORT):
    """Account numbers, the summary values, and what is held."""
    from ib_async import IB

    ib = IB()
    try:
        ib.connect(host, port, clientId=CLIENT_ID, readonly=True)
        accounts = sorted(ib.managedAccounts())
        values = ib.accountSummary()
        holdings = ib.portfolio()
    finally:
        ib.disconnect()
    return accounts, values, holdings


def main():
    accounts, values, holdings = summary()
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

    if holdings:
        print("\npositions")
        longs = shorts = 0.0
        for item in sorted(holdings, key=lambda h: h.contract.symbol):
            market = float(item.marketValue)
            if market >= 0:
                longs += market
            else:
                shorts += -market
            print(f"  {item.contract.symbol:6s} "
                  f"{item.position:>10,.0f} @ {item.marketPrice:>10,.2f}   "
                  f"{market:>14,.2f} {item.contract.currency}   "
                  f"unrealised {item.unrealizedPNL:>12,.2f}")
        print(f"  {'long':6s} {longs:>40,.2f}")
        print(f"  {'short':6s} {shorts:>40,.2f}")
        print(f"  {'gross':6s} {longs + shorts:>40,.2f}")
        print(f"  {'net':6s} {longs - shorts:>40,.2f}")

    # what the backtest would charge on this book, against what the broker
    # has accrued. The model treats a short as debt; the broker does not.
    gross = found.get('GrossPositionValue')
    equity = found.get('NetLiquidation')
    accrued = found.get('AccruedCash')

    if gross is not None and equity is not None:
        rate, asof = benchmark_rate()
        debit = max(0.0, float(gross.value) - float(equity.value))
        daily = debit * rate / 360
        print(f"\nfinancing check   benchmark + {FINANCING_SPREAD:.2%} "
              f"= {rate:.2%}, as of {asof}")
        print(f"  modelled debit   {debit:>14,.2f} {gross.currency}   "
              f"(gross less equity)")
        print(f"  modelled charge  {daily:>14,.2f} {gross.currency}/day   "
              f"({daily * 30:,.0f}/30d, {daily * 365:,.0f}/yr)")
        if accrued is not None:
            print(f"  broker accrued   {float(accrued.value):>14,.2f} "
                  f"{accrued.currency}   (interest, month to date)")
        print("  a short is counted as debt here and is not one at the "
              "broker - the gap is the error")


if __name__ == '__main__':
    main()
