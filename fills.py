"""
Record what each order actually got.

    python fills.py

Every cost figure in the system is a model: the quoted spread is sampled and
half of it assumed paid, and the commission is taken from the published
schedule. This is the only measurement.

Shortfall is the fill price against the close the target was sized on, signed
so that buying above it is a cost. That close is the price the simulation
assumes the trade happened at, so the difference is what acting a day later
really costs. It is not the same quantity as the modelled half-spread: it
carries a day of price movement as well, which has no sign of its own and
averages out across many trades, leaving the cost behind.

The benchmark comes from the order file rather than the current targets. The
target file is rewritten every night, so a fill picked up a day later would
otherwise be measured against a close its order never saw.

The quotes recorded beside each fill are whatever the market data connection
returned. They are stored for the record and not used in any measurement,
because delayed quotes are stale by an unknown amount.

Run it once the orders have had time to fill. The order script watches only
briefly, so a fill arriving later is invisible to it. Executions carry the
broker's own identifier, so running this twice adds nothing.

Appends to data/fills/fills.csv.
"""

import pandas as pd

from config import ORDERS_DIR, FILLS_DIR, IB_HOST, IB_PORT

# client 0 is shown every execution on the account, including those a
# different session placed; a numbered client sees only its own

CLIENT_ID = 0
SETTLE = 2.0                # seconds to let executions and commission arrive

FILLS_FILE = 'fills.csv'


def latest_orders():
    """
    The most recent order file indexed by order id, with its name.

    A dry run writes no order ids, so a file without that column is treated
    as nothing to join against.
    """
    matches = sorted(ORDERS_DIR.glob('orders_*.csv'))
    if not matches:
        return None, None
    frame = pd.read_csv(matches[-1])
    if 'order_id' not in frame.columns:
        return None, matches[-1].name
    return frame.set_index('order_id'), matches[-1].name


def broker_fills(host=IB_HOST, port=IB_PORT):
    """Today's executions, with the commission charged on each."""
    from ib_async import IB

    ib = IB()
    try:
        ib.connect(host, port, clientId=CLIENT_ID, readonly=True)
        ib.reqExecutions()
        ib.sleep(SETTLE)
        fills = ib.fills()
    finally:
        ib.disconnect()
    return fills


def describe(fill, orders):
    """
    One row: what the broker did, and what it cost against the close its
    order was sized on.

    An execution with no matching order row is still recorded, unmeasured -
    it is a trade this system did not place, or one from an earlier order
    file, and either is worth knowing about.
    """
    execution = fill.execution
    report = fill.commissionReport
    side = 'BUY' if execution.side.upper().startswith('B') else 'SELL'
    price = float(execution.price)
    shares = int(execution.shares)

    row = {
        'exec_id': execution.execId,
        'order_id': execution.orderId,
        'filled_at': pd.Timestamp(execution.time).isoformat(),
        'ticker': fill.contract.symbol.upper(),
        'side': side,
        'shares': shares,
        'price': price,
        'commission': round(float(report.commission), 4) if report else None,
        'currency': report.currency if report else None,
        }

    if orders is None or execution.orderId not in orders.index:
        return row

    asked = orders.loc[execution.orderId]
    bid, ask = float(asked['bid']), float(asked['ask'])
    close = float(asked['target_close'])
    paid = (price - close) if side == 'BUY' else (close - price)

    row.update({
        'requested_shares': int(asked['shares']),
        'limit': float(asked['limit']),
        'quoted_bid': bid,
        'quoted_ask': ask,
        'modelled_half_spread': round((ask - bid) / 2, 4),
        'benchmark_close': close,
        'benchmark_date': asked['target_date'],
        'shortfall_per_share': round(paid, 4),
        'shortfall_total': round(paid * shares, 2),
        })
    return row


def append(rows):
    """
    Add the executions not already recorded, and return the new ones.

    The file is evidence, so nothing in it is ever rewritten.
    """
    FILLS_DIR.mkdir(parents=True, exist_ok=True)
    path = FILLS_DIR / FILLS_FILE

    if path.exists():
        kept = pd.read_csv(path)
        fresh = rows[~rows['exec_id'].isin(set(kept['exec_id']))]
        combined = pd.concat([kept, fresh], ignore_index=True)
    else:
        fresh, combined = rows, rows

    combined.to_csv(path, index=False)
    return fresh, combined, path


def main():
    orders, orders_name = latest_orders()
    if orders is None:
        print(f"{orders_name} holds no order ids - nothing to join against"
              if orders_name else
              f"no order file in {ORDERS_DIR} - nothing to join against")
    else:
        print(f"joining against {orders_name}")

    fills = broker_fills()
    if not fills:
        print("no executions reported today")
        return

    rows = pd.DataFrame([describe(f, orders) for f in fills])
    fresh, combined, path = append(rows)

    print(f"{len(fresh)} new execution(s), {len(combined)} recorded in total")
    if fresh.empty:
        return

    for _, row in fresh.iterrows():
        shortfall = row.get('shortfall_per_share')
        if pd.notna(shortfall):
            tail = (f"   {shortfall:+.4f}/share against the "
                    f"{row['benchmark_date']} close of "
                    f"{row['benchmark_close']:.2f}")
        else:
            tail = "   no order row, unmeasured"
        print(f"{row['ticker']:6s} {row['side']:4s} {int(row['shares']):5d} "
              f"@ {row['price']:8.2f}   commission {row['commission']}{tail}")

    if 'shortfall_per_share' in fresh.columns:
        measured = fresh.dropna(subset=['shortfall_per_share'])
    else:
        measured = fresh.iloc[0:0]

    if not measured.empty:
        traded = (measured['price'] * measured['shares']).sum()
        shortfall = measured['shortfall_total'].sum()
        commission = measured['commission'].sum()
        print(f"\ntraded {traded:,.0f}   "
              f"shortfall {shortfall:,.2f} ({shortfall / traded * 1e4:+.1f} bp)"
              f"   commission {commission:,.2f} "
              f"({commission / traded * 1e4:.1f} bp)")

    print(f"\nwritten to {path}")


if __name__ == '__main__':
    main()
