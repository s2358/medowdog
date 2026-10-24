"""
Place the orders that move the book to today's target positions.

    python orders.py              # dry run, sends nothing
    python orders.py --place      # transmits

Nothing reaches the broker unless --place is given.

Targets come from live_targets.csv, sized against the account value rather
than the equity the history ends on. The trade is that target minus what the
broker actually holds, so no record of yesterday's position is needed and an
unrecorded fill corrects itself on the next run.

A target within the inertia band of the position held is left alone, the band
being a fraction of the position so that small adjustments are not worth their
cost. Gross notional is then capped at MAX_LEVERAGE times capital, which is a
portfolio limit and can only be applied once every instrument has been sized.

Orders are limit orders priced through the far side of the spread by
LIMIT_BUFFER: a buy above the ask, a sell below the bid. The quotes read here
are delayed, so the touch may be some minutes old and the order would not be
marketable if the price has since moved away; crossing by a margin restores
that while still refusing a price far from the quote.

A trade too small to carry the per-order commission floor is dropped rather
than sent. The floor is charged per order, so the smaller the trade the larger
the cost per share; the smallest acceptable trade is the one whose commission
stays inside the ceiling the cost model uses.

A position in an instrument the system does not know about stops the run.

Writes data/orders/orders_<timestamp>.csv.
"""

import sys
from datetime import datetime

import pandas as pd

import capital
import sizing
import fx_data
import universe
from reconcile import broker_positions
from cancel import working_orders, describe as describe_order
from config import (PORTFOLIO_DIR, ORDERS_DIR, IB_HOST, IB_PORT,
                    IB_MARKET_DATA_TYPE, INERTIA, MAX_LEVERAGE, LIMIT_BUFFER)
from universe import WEIGHTS


CLIENT_ID = 32
QUOTE_TIMEOUT = 15      # seconds to wait for each quote
FILL_WAIT = 30          # seconds to watch for fills before reporting

TARGETS_FILE = 'live_targets.csv'


def limit_price(shares, bid, ask, ticker):
    """The far touch, crossed by the buffer, at a price the exchange takes."""
    raw = ask * (1 + LIMIT_BUFFER) if shares > 0 else bid * (1 - LIMIT_BUFFER)
    return universe.round_to_tick(raw, ticker)


def load_targets(tickers):
    """Today's target position and the close it was sized on, by ticker."""
    path = PORTFOLIO_DIR / TARGETS_FILE
    if not path.exists():
        raise FileNotFoundError(
                f"{path} not found - run the position sizing script first")
    frame = pd.read_csv(path, parse_dates=['date']).set_index('ticker')
    missing = set(tickers) - set(frame.index)
    if missing:
        raise KeyError(f"no target for: {sorted(missing)}")
    return {t: {'target': int(frame.loc[t, 'rounded']),
                'close': float(frame.loc[t, 'close']),
                'asof': frame.loc[t, 'date'].date()}
            for t in tickers}


def wanted_positions(targets, broker, capital_usd):
    """
    The position to hold in each instrument, and the scale the cap imposed.
    """
    want = {}
    for ticker, entry in targets.items():
        held = broker.get(ticker, 0)
        want[ticker] = sizing.apply_inertia(entry['target'], held, INERTIA)

    prices = {t: targets[t]['close'] for t in want}
    want, scale = sizing.cap_leverage(want, prices, capital_usd, MAX_LEVERAGE)
    return {t: int(n) for t, n in want.items()}, scale



def quote(ib, contract, timeout=QUOTE_TIMEOUT):
    """Current bid and ask, or None if no usable quote arrives."""
    ticker = ib.reqMktData(contract, '', False, False)
    waited = 0.0
    while waited < timeout:
        ib.sleep(0.5)
        waited += 0.5
        if ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > ticker.bid:
            bid, ask = float(ticker.bid), float(ticker.ask)
            ib.cancelMktData(contract)
            return bid, ask
    ib.cancelMktData(contract)
    return None


def build_and_send(targets, want, broker, place):
    """
    Price every trade against a quote and, if asked, send it.

    Returns the rows describing what was built, whether or not it was sent.
    """
    from ib_async import IB, LimitOrder

    ib = IB()
    rows = []
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, readonly=(not place))
        ib.reqMarketDataType(IB_MARKET_DATA_TYPE)

        working = working_orders(ib)
        if working:
            print(f"{len(working)} order(s) already working at the broker:")
            for trade in working:
                print(f"  {describe_order(trade)}")
            print("\nthe trade is the difference between the target and the "
                  "position held, and these are in neither - run cancel.py "
                  "first")
            sys.exit(1)

        trades = []
        for ticker in sorted(want):
            held = broker.get(ticker, 0)
            shares = want[ticker] - held
            if shares == 0:
                print(f"{ticker:6s} holding {held:+6d}, no trade")
                continue

            contract = universe.contract(ticker)
            ib.qualifyContracts(contract)
            prices = quote(ib, contract)
            if prices is None:
                print(f"{ticker:6s} no quote after {QUOTE_TIMEOUT}s - skipped")
                continue

            bid, ask = prices
            floor = universe.minimum_trade(ticker, ask)
            if abs(shares) < floor:
                print(f"{ticker:6s} {shares:+d} below the minimum of "
                      f"{floor} shares at {ask:.2f} - dropped")
                continue

            side = 'BUY' if shares > 0 else 'SELL'
            limit = limit_price(shares, bid, ask, ticker)
            order = LimitOrder(side, abs(shares), limit, tif='DAY')

            row = {
                'ticker': ticker,
                'side': side,
                'shares': abs(shares),
                'limit': limit,
                'bid': bid,
                'ask': ask,
                'buffer': LIMIT_BUFFER,
                'target': targets[ticker]['target'],
                'target_close': targets[ticker]['close'],
                'target_date': str(targets[ticker]['asof']),
                'wanted': want[ticker],
                'held_before': held,
                'minimum_shares': floor,
                'sent': place,
            }

            print(f"{ticker:6s} {side:4s} {abs(shares):5d} @ {limit:8.2f}   "
                  f"quote {bid:.2f}/{ask:.2f}   "
                  f"{held:+d} -> {want[ticker]:+d}   min {floor}")

            if place:
                trades.append((row, ib.placeOrder(contract, order)))
            else:
                rows.append(row)

        if place and trades:
            print(f"\nsent {len(trades)} order(s), watching {FILL_WAIT}s")
            ib.sleep(FILL_WAIT)
            for row, trade in trades:
                status = trade.orderStatus
                row.update({
                        'order_id': trade.order.orderId,
                        'status': status.status,
                        'filled': int(status.filled),
                        'remaining': int(status.remaining),
                        'avg_fill_price': round(status.avgFillPrice, 4)
                                        if status.filled else None,
                    })
                print(f"{row['ticker']:6s} {status.status:12s} "
                      f"filled {int(status.filled):5d} of {row['shares']:5d} "
                      f"at {status.avgFillPrice or 0:.4f}")
                rows.append(row)
    finally:
        ib.disconnect()

    return rows


def main():
    place = '--place' in sys.argv[1:]
    tickers = sorted(WEIGHTS)

    stored, age = capital.load()
    spot = fx_data.load()
    targets = load_targets(tickers)

    dates = {entry['asof'] for entry in targets.values()}
    if len(dates) > 1:
        print("the target file disagrees with itself on the date: "
              + ", ".join(f"{t} {targets[t]['asof']}" for t in tickers))
        sys.exit(1)

    broker, accounts, other = broker_positions()

    print(f"account {', '.join(accounts) if accounts else 'none reported'}")
    print(f"capital {stored['value']:,.0f} {stored['currency']}   "
          f"read {age.total_seconds()/3600:.1f}h ago")
    print(f"targets as of {dates.pop()}")

    extra = sorted(set(broker) - set(targets))
    for symbol in extra:
        print(f"{symbol:6s} held {broker[symbol]:6d}, not in the system")
    if other:
        print("not shares, ignored: " + "; ".join(other))
    if extra:
        print("\nthe account holds something the system does not know about "
              "- stopping")
        sys.exit(1)

    capital_usd = stored['value'] / spot
    want, scale = wanted_positions(targets, broker, capital_usd)
    if scale < 1.0:
        print(f"leverage cap binds - every position scaled by {scale:.3f}")

    print("\nDRY RUN - nothing will be sent\n" if not place
          else "\nPLACING ORDERS\n")

    rows = build_and_send(targets, want, broker, place)

    if not rows:
        print("\nnothing to do")
        return

    ORDERS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = ORDERS_DIR / f'orders_{stamp}.csv'
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nwritten to {out_path}")


if __name__ == '__main__':
    main()
