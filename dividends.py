"""
Dividend schedule for the instruments in WEIGHTS.

    python dividends.py
    python dividends.py SPY

Distributions are derived from the two series IBKR serves for the same
contract. TRADES is split-adjusted only; ADJUSTED_LAST is split- and
dividend-adjusted. Their ratio steps only on an ex-date:

    r_t = adjusted_t / trades_t
    r_{d-1} / r_d = 1 - D_d / P_{d-1}  ->  D_d = P_{d-1} * (1 - r_{d-1}/r_d)

reqFundamentalData is not used - IBKR's fundamentals are thin for ETFs, and it
would be a second source to reconcile against the prices the backtest uses.

This is the only script that touches ADJUSTED_LAST, and it keeps nothing but
the extracted dividends, so the price history stays unadjusted and a rerun
never restates it.

A short answer here is indistinguishable from a correct one, so nothing is written unless the bars reach back to the start of the backtest window - the latest price history start among the instruments in WEIGHTS. bars.py walks the history and reports whether it reached the start.

Writes data/dividends/<TICKER>.csv and _fetched.json.
"""

import json
import sys
from datetime import datetime, timedelta
import pandas as pd
import bars
from config import (WEIGHTS, DATA_DIR, DIVIDENDS_DIR, DIVIDEND_MAX_AGE_DAYS,
                    IB_HOST, IB_PORT)


# a ratio step below this is rounding noise in IBKR's adjusted prices, not a
# distribution. SPY pays around 30 bp a quarter, so the gap is wide.
MIN_YIELD = 0.0005

STAMPS = DIVIDENDS_DIR / '_fetched.json'


def path_for(ticker):
    return DIVIDENDS_DIR / f'{ticker}.csv'


def load(ticker):
    """
    Stored dividends per share, indexed by ex-date. None if never fetched.

    An empty series is a real answer - the instrument was checked and pays
    nothing - and is not the same as a missing file.
    """
    path = path_for(ticker)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=['ex_date'])
    return df.set_index('ex_date')['dps']


def stamps():
    if not STAMPS.exists():
        return {}
    return json.loads(STAMPS.read_text())


def describe(ticker):
    """Cache metadata for a ticker, or None if it was never fetched."""
    fetched = stamps().get(ticker)
    if fetched is None:
        return None
    series = load(ticker)
    age = datetime.now() - datetime.fromisoformat(fetched['fetched_at'])
    return {
        'ticker': ticker,
        'fetched_at': fetched['fetched_at'],
        'covers_from': fetched['covers_from'],
        'age_days': age.days,
        'stale': age > timedelta(days=DIVIDEND_MAX_AGE_DAYS),
        'n': 0 if series is None else len(series),
        'last_ex_date': None if series is None or series.empty
                        else series.index[-1].date(),
    }


def save(ticker, series, covers_from):
    DIVIDENDS_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({
        'ex_date': series.index.date,
        'dps': series.values,
        'currency': 'USD',
    })
    out.to_csv(path_for(ticker), index=False)

    marks = stamps()
    marks[ticker] = {
        'fetched_at': datetime.now().isoformat(timespec='seconds'),
        'covers_from': str(covers_from.date()),
    }
    STAMPS.write_text(json.dumps(marks, indent=2, sort_keys=True))


def price_start(ticker):
    """
    First date in the instrument's price CSV - the date the dividend record
    has to reach back to. None when there is no price file to check against.
    """
    folder = DATA_DIR / ticker
    matches = sorted(folder.glob('*.csv')) if folder.exists() else []
    if len(matches) != 1:
        return None
    dates = pd.read_csv(matches[0], usecols=['date'])['date']
    return pd.to_datetime(dates, utc=True).dt.tz_localize(None).min()


def implied_dividends(trades, adjusted, min_yield=MIN_YIELD):
    """
    Dividend per share on each ex-date, from the ratio of the two series.

    Splits move both series identically and cancel in the ratio, so only
    distributions survive.
    """
    pair = pd.DataFrame({'p': trades, 'a': adjusted}).dropna()
    if len(pair) < 2:
        return pd.Series(dtype=float, name='dps')

    ratio = pair['a'] / pair['p']
    prev_ratio = ratio.shift(1)
    prev_price = pair['p'].shift(1)

    dps = prev_price * (1 - prev_ratio / ratio)
    keep = (dps / prev_price) > min_yield
    out = dps[keep.fillna(False)]
    out.name = 'dps'
    out.index.name = 'ex_date'
    return out



def close_series(ib, contract, what):
    """Daily closes, date-indexed, as (Series, complete)."""
    frame, complete = bars.daily(ib, contract, what)
    out = frame[['date', 'close']].copy()
    out['date'] = pd.to_datetime(out['date'])
    return out.set_index('date')['close'], complete


def fetch(tickers, host=IB_HOST, port=IB_PORT):
    """
    Connect once and derive the schedule for every ticker.

    Each result carries where its bars start and whether both downloads ran to
    completion, so main() can refuse to store a truncated answer.
    """
    from ib_async import IB, Stock

    ib = IB()
    results = {}
    try:
        ib.connect(host, port, clientId=7, readonly=True)
        for ticker in tickers:
            contract = Stock(ticker, 'SMART', 'USD')
            ib.qualifyContracts(contract)

            trades, ok_t = close_series(ib, contract, 'TRADES')
            ib.sleep(bars.PAUSE)
            adjusted, ok_a = close_series(ib, contract, 'ADJUSTED_LAST')
            ib.sleep(bars.PAUSE)

            results[ticker] = {
                'dividends': implied_dividends(trades, adjusted),
                'covers_from': max(trades.index[0], adjusted.index[0]),
                'complete': ok_t and ok_a,
            }
    finally:
        ib.disconnect()
    return results


def verdict(result, needed):
    """Why this result may not be stored, or None if it may."""
    if needed is not None:
        if result['covers_from'] > needed:
            return (f"bars start {result['covers_from'].date()}, "
                    f"backtest window starts {needed.date()}")
        return None          # reaches the window the backtest uses
    if not result['complete']:
        return 'downloads were cut short and there is no price file to check'
    return None


def main():
    tickers = [t.upper() for t in sys.argv[1:]] or sorted(WEIGHTS)
    print(f"deriving dividends for {', '.join(tickers)} "
          f"({bars.CHUNK}, TRADES vs ADJUSTED_LAST)\n")

    starts = [price_start(t) for t in sorted(WEIGHTS)]
    starts = [d for d in starts if d is not None]
    needed = max(starts) if starts else None

    results = fetch(tickers)
    kept = 0

    for ticker in tickers:
        result = results[ticker]
        series = result['dividends']
        problem = verdict(result, needed)

        if problem:
            print(f"{ticker:6s} NOT SAVED - {problem}")
            continue

        save(ticker, series, result['covers_from'])
        kept += 1
        if series.empty:
            print(f"{ticker:6s} no distributions, "
                  f"checked from {result['covers_from'].date()}")
            continue
        recent = series[series.index > series.index[-1] - pd.Timedelta(days=365)]
        print(f"{ticker:6s} {len(series):3d} distributions   "
              f"{series.index[0].date()} -> {series.index[-1].date()}   "
              f"last {series.iloc[-1]:.4f}   "
              f"trailing 12m {recent.sum():.4f} USD/share")

    print(f"\n{kept} of {len(tickers)} written to {DIVIDENDS_DIR}/")
    if kept < len(tickers):
        sys.exit(1)


if __name__ == '__main__':
    main()
