"""
The live equity curve, from recorded account readings.

    python equity.py

capital.py appends every reading of the account's value. This turns those
readings into one row per US trading session, in the columns portfolio.csv
uses, so the evaluation script reads it unchanged:

    python 7_evaluate.py data/portfolio/live.csv

A reading marks the session that closed most recently before it was taken.
One taken while the US market is open marks nothing - the positions are not
settled - and is dropped. Where a day has several readings the last one
stands.

gross_pnl is written equal to net_pnl and cost as zero. A net liquidation
figure already has commission inside it and nothing in it separates the two,
so the drag breakdown the evaluation prints is meaningless on this file. What
trading actually cost is in data/fills/fills.csv.

Writes data/portfolio/live.csv.
"""

import sys
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from capital import READINGS_FILE
from config import HOME_CURRENCY, PORTFOLIO_DIR

EXCHANGE_TZ = ZoneInfo('America/New_York')
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

LIVE_FILE = 'live.csv'


def session_date(read_at):
    """
    The session a reading marks, or None if the market was open at the time.

    Readings carry the local time of the machine that took them, so they are
    converted to exchange time before being placed. A reading after the close
    marks that day; one before the open marks the day before.
    """
    moment = datetime.fromisoformat(read_at).astimezone(EXCHANGE_TZ)

    if MARKET_OPEN <= moment.time() < MARKET_CLOSE:
        return None

    marked = moment.date() if moment.time() >= MARKET_CLOSE \
        else (moment - timedelta(days=1)).date()

    # a reading over the weekend marks the Friday
    while marked.weekday() > 4:
        marked -= timedelta(days=1)
    return marked


def load_readings(path=READINGS_FILE):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run capital.py first")
    frame = pd.read_csv(path)
    currencies = set(frame['currency'])
    if currencies != {HOME_CURRENCY}:
        raise ValueError(f"readings are in {sorted(currencies)}, "
                         f"the system reports in {HOME_CURRENCY}")
    return frame


def daily_curve(readings):
    """One row per session, carrying that session's closing account value."""
    readings = readings.sort_values('read_at')
    marked = [(session_date(row['read_at']), float(row['value']))
              for _, row in readings.iterrows()]
    kept = [(date, value) for date, value in marked if date is not None]
    if not kept:
        raise ValueError("every reading was taken while the market was open")

    frame = pd.DataFrame(kept, columns=['date', 'equity'])
    frame = frame.groupby('date', as_index=False).last()
    frame['date'] = pd.to_datetime(frame['date'])
    frame = frame.sort_values('date').set_index('date')

    frame['net_pnl'] = frame['equity'].diff().fillna(0.0).round(2)
    frame['gross_pnl'] = frame['net_pnl']
    frame['cost'] = 0.0
    return frame[['gross_pnl', 'cost', 'net_pnl', 'equity']]


def main():
    readings = load_readings()
    curve = daily_curve(readings)

    dropped = len(readings) - len(curve)
    start, end = curve['equity'].iloc[0], curve['equity'].iloc[-1]
    change = end - start

    print(f"{len(readings)} reading(s), {len(curve)} session(s)"
          + (f", {dropped} not used" if dropped else ""))
    print(f"{curve.index[0].date()} -> {curve.index[-1].date()}")
    print(f"equity {start:,.2f} -> {end:,.2f} {HOME_CURRENCY}   "
          f"{change:+,.2f} ({change / start:+.2%})")

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    out = PORTFOLIO_DIR / LIVE_FILE
    curve.to_csv(out)
    print(f"\nwritten to {out}")


if __name__ == '__main__':
    main()
