"""
Measure trading costs from live bid/ask spreads.

    python 5_costs.py data/features/SPY.csv data/features/GLDM.csv data/features/SLV.csv

Must run during US market hours - the quote stream returns IBKR's -1 sentinel when the market is closed, which the pre-flight probe catches.

Trade sizes come from the turnover file the position sizing script writes. The
$1 commission minimum is divided by them, so they dominate the result: at 5
shares the minimum is $0.20 a share, at 200 it is $0.005.

An instrument that has never been traded has no turnover row, so its trade
size is estimated from the sizing formula instead. The estimate is marked in
the output and is replaced by the measurement as soon as the instrument
carries a weight.

Every run appends one row per instrument to costs/history.csv and compares
itself against the median of the recent runs there, and against the figure
universe.py is actually using. Anything that has moved by half is printed as a
spike. The history is the point of the daily schedule: one sample says what
tonight costs, a hundred say whether tonight is unusual.

Short sale availability is sampled alongside the spread. The borrow fee itself
is only published in the broker's Borrow Fee Details statement, and IBKR may
not serve it over the API at all, so it is recorded when it arrives and left
blank when it does not. Shortable shares always arrive, and a collapse there is
what precedes a borrow rate spike.

This script measures. It sets nothing. The figure it prints is written into
universe.py by hand, because a cost is a property of the instrument, not of
the night it was sampled.
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from ib_async import IB

from config import (IB_HOST, IB_PORT, IB_MARKET_DATA_TYPE, WARMUP, BUSINESS_DAYS, SPEED_LIMIT_SR, COSTS_DIR, PORTFOLIO_DIR, VOL_TARGET)
import capital
import fx_data
import sizing
import universe
from universe import WEIGHTS, ticker_from_path


CLIENT_ID = 21

ANNUALISE = np.sqrt(BUSINESS_DAYS)

SAMPLES = 20
SAMPLE_INTERVAL = 15        # seconds
PROBE_TIMEOUT = 15          # seconds to wait for the first quote

# 236 asks for short sale availability alongside the quote. The shortable
# share count always arrives; the fee rate may not, and is recorded only
# when it does.
GENERIC_TICKS = '236'

# a candidate has never been traded, so it has no measured turnover. Its
# average position is the full-strength position, since the mean absolute
# forecast is 10 by construction, and its average trade is close to the
# inertia band times that - measured at 0.118 to 0.153 on the traded three
# against a band of 0.10.
CANDIDATE_WEIGHT = 0.20
CANDIDATE_IDM = 2.0
TRADE_FRACTION = 0.13

# the running record, appended to on every run
HISTORY_FILE = 'history.csv'
HISTORY_LOOKBACK = 20       # runs the baseline median is taken over
HISTORY_MINIMUM = 3         # runs needed before a comparison means anything

SPIKE_UP = 1.5              # flag anything half again as expensive
SPIKE_DOWN = 0.5            # or half as available

# (column, label, which direction is the alarm)
SPIKE_METRICS = [
    ('spread_median', 'spread', 'high'),
    ('cost_per_share_C', 'cost/share', 'high'),
    ('standardised_cost', 'std cost', 'high'),
    ('fee_rate', 'borrow fee', 'high'),
    ('shortable_shares', 'shortable', 'low'),
]

HISTORY_COLS = ['sampled_at', 'ticker', 'spread_median', 'half_spread',
                'typical_shares', 'shares_estimated', 'commission_per_share',
                'cost_per_share_C', 'icv', 'standardised_cost',
                'fee_rate', 'shortable_shares', 'n_quotes']

SPINNER = '|/-\\'
ANIMATE = sys.stdout.isatty()   # no spinner when redirected to a log


def latest_icv(path):
    """ICV: the currency volatility of one unit, as script 3 computed it."""
    df = pd.read_csv(path, usecols=['date', 'close', 'icv']).iloc[WARMUP:]
    row = df.iloc[-1]
    return row['icv'], row['date'], row['close']


def wait_for_quote(ib, ticker, timeout=PROBE_TIMEOUT):
    """Poll until a usable quote arrives, or the timeout expires."""
    waited = 0.0
    while waited < timeout:
        ib.sleep(0.5)
        waited += 0.5
        if ticker.bid and ticker.ask and ticker.bid > 0:
            return True
    return False


def clean(value):
    """IBKR sends -1 for 'no data' and numpy sends nan. Both mean absent."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number < 0:
        return None
    return number


def borrow_fields(ticker):
    """
    Short sale availability, as far as the API gives it.

    shortableShares is served against generic tick 236 and is the number of
    shares the broker can currently lend. The fee rate is not part of the
    standard tick set, so it is read defensively: present on some builds and
    some instruments, absent on others, and never required.
    """
    shares = clean(getattr(ticker, 'shortableShares', None))
    rate = clean(getattr(ticker, 'feeRate', None))
    if rate is not None and rate > 1:
        rate = rate / 100        # some feeds quote percent, not a decimal
    return rate, shares


def sample_quotes(ib, symbol, samples=SAMPLES, interval=SAMPLE_INTERVAL):
    """
    Sample bid/ask repeatedly.

    Returns the spreads seen, plus the last short sale figures observed. The
    borrow numbers move far more slowly than the quote, so the last reading
    is as good as a median of them.
    """
    contract = universe.contract(symbol)
    ib.qualifyContracts(contract)
    ticker = ib.reqMktData(contract, GENERIC_TICKS, False, False)

    spreads = []
    fee_rate = None
    shortable = None
    tick = 0
    total_secs = samples * interval

    for i in range(samples):
        # animate during the wait instead of blocking entirely
        waited = 0.0
        while waited < interval:
            ib.sleep(0.25)
            waited += 0.25
            tick += 1
            if ANIMATE:
                elapsed = i * interval + waited
                remaining = total_secs - elapsed
                quote = (f"{ticker.bid:.2f}/{ticker.ask:.2f}"
                     if ticker.bid and ticker.ask else "  --/--  "
                     )
                print(f"\r {SPINNER[tick % 4]} {symbol:6s}  "
                  f"sample {i+1:2d}/{samples}  "
                  f"quote {quote}  "
                  f"captured {len(spreads):2d}  "
                  f"{remaining:3.0f}s left  ",
                  end='', flush=True
                    )
        bid, ask = ticker.bid, ticker.ask
        if bid and ask and bid > 0 and ask > bid:
            spreads.append(ask - bid)

        rate, shares = borrow_fields(ticker)
        if rate is not None:
            fee_rate = rate
        if shares is not None:
            shortable = shares

    if ANIMATE:
        print("\r" + " " * 78 + "\r", end='')
    ib.cancelMktData(contract)
    return spreads, fee_rate, shortable


def standardised_cost(half_spread, comm_per_share, icv):
    """ (2 x C / (annualised ICV), in annual Sharpe ratio units per round trip)"""
    C = half_spread + comm_per_share
    return 2 * C / (ANNUALISE * icv), C


def load_typical_trade():
    """
    Trade sizes measured by the position sizing script. The $1 commission minimum is divided by these, so they drive the whole result - reading them beats keeping a hand-copied dict in sync.
    """
    path = PORTFOLIO_DIR / f"turnover_{'_'.join(sorted(WEIGHTS))}.csv"
    if not path.exists():
        raise FileNotFoundError(
                f"{path} not found - run the position sizing script first")
    series = pd.read_csv(path).set_index('ticker')['typical_trade']
    return {t: max(1, int(round(v))) for t, v in series.items()}, path.name


def estimated_trade(icv, capital_usd, weight=CANDIDATE_WEIGHT,
                    idm=CANDIDATE_IDM, fraction=TRADE_FRACTION):
    """
    Trade size for an instrument that has never been traded.

    Replaced by the measured figure as soon as the instrument carries a
    weight and the position sizing script has walked a history with it in.
    """
    daily_target = sizing.daily_cash_vol_target(capital_usd, VOL_TARGET)
    full = sizing.volatility_scalar(daily_target, icv) * weight * idm
    return max(1, int(round(fraction * full)))


def load_history(path=None):
    """
    Every row this script has ever written, or an empty frame on the first
    run. Read before today's rows are appended, so the baseline never
    contains the sample it is judging.
    """
    path = path or COSTS_DIR / HISTORY_FILE
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def append_history(rows, path=None):
    """One row per instrument, appended. The file is never rewritten."""
    path = path or COSTS_DIR / HISTORY_FILE
    frame = pd.DataFrame(rows).reindex(columns=HISTORY_COLS)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode='a', header=not path.exists(), index=False)
    return path


def baseline(history, ticker, column, lookback=HISTORY_LOOKBACK):
    """
    The median of an instrument's recent readings, or None when there are too
    few to mean anything. A median rather than a mean, because one bad night
    should move the alarm threshold as little as possible.
    """
    if history.empty or column not in history.columns:
        return None
    rows = history[history['ticker'] == ticker]
    if len(rows) < HISTORY_MINIMUM:
        return None
    values = pd.to_numeric(rows[column], errors='coerce').dropna()
    values = values[values > 0].tail(lookback)
    if len(values) < HISTORY_MINIMUM:
        return None
    return float(values.median())


def human(value):
    """Share counts run to the millions and are unreadable in full."""
    if value is None:
        return '--'
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.0f}k"
    if abs(value) < 1:
        return f"{value:.4f}"
    return f"{value:,.0f}"


def spike_report(rows, history):
    """
    Compare tonight against the recent past, per instrument and per metric.

    A cost that has risen by half, or availability that has halved, is worth
    looking at. Everything else is printed as context so that a quiet night
    looks quiet rather than looking like nothing ran.
    """
    lines = []
    alarms = 0

    for row in rows:
        ticker = row['ticker']
        for column, label, direction in SPIKE_METRICS:
            now = clean(row.get(column))
            was = baseline(history, ticker, column)
            if now is None or was is None or was == 0:
                continue
            ratio = now / was
            hit = (ratio >= SPIKE_UP if direction == 'high'
                   else ratio <= SPIKE_DOWN)
            alarms += hit
            change = (ratio - 1) * 100
            lines.append(f"  {ticker:6s} {label:11s} "
                         f"{human(now):>10s} vs {human(was):>10s}  "
                         f"{change:+6.0f}%  {'SPIKE' if hit else ''}")

    if not lines:
        print(f"\nspike check   no baseline yet - needs {HISTORY_MINIMUM} "
              "runs per instrument")
        return 0

    print(f"\nspike check   against the median of the last "
          f"{HISTORY_LOOKBACK} runs")
    print('\n'.join(lines))
    return alarms


def drift_report(rows):
    """
    Tonight's cost against the number the backtest is actually using.

    The history check catches a sudden move. This catches a slow one: a cost
    that has crept away from universe.py over months without any single night
    looking unusual.
    """
    lines = []
    for row in rows:
        ticker = row['ticker']
        try:
            stored = universe.cost_per_share(ticker)
        except KeyError:
            lines.append(f"  {ticker:6s} {row['cost_per_share_C']:9.5f}  "
                         "vs       --   not set in universe.py")
            continue
        if not stored:
            continue
        change = (row['cost_per_share_C'] / stored - 1) * 100
        flag = '  update universe.py' if abs(change) >= 50 else ''
        lines.append(f"  {ticker:6s} {row['cost_per_share_C']:9.5f}  vs "
                     f"{stored:9.5f}  {change:+6.0f}%{flag}")

    if lines:
        print("\nagainst universe.py cost_per_share")
        print('\n'.join(lines))


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} " "<instrument_csv> [<instrument_csv> ...]")
        sys.exit(1)

    paths = sys.argv[1:]
    tickers = [ticker_from_path(p) for p in paths]

    typical_trade, trade_file = load_typical_trade()
    candidates = [t for t in tickers if t not in typical_trade]

    stored, _ = capital.load()
    capital_usd = stored['value'] / fx_data.load()

    history = load_history()

    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, readonly=True)
    ib.reqMarketDataType(IB_MARKET_DATA_TYPE)         # 1 = live, 3 = delayed (spreads will be stale)

    # pre-flight: if the market is closed, fail in seconds rather than
    # grinding through samples x interval x tickers of dead air
    probe = universe.contract(tickers[0])
    ib.qualifyContracts(probe)
    probe_ticker = ib.reqMktData(probe, GENERIC_TICKS, False, False)
    live = wait_for_quote(ib, probe_ticker)
    bid, ask = probe_ticker.bid, probe_ticker.ask
    ib.cancelMktData(probe)

    if not live:
        print(f"no quotes after {PROBE_TIMEOUT}s (bid={bid}, ask={ask}) - market likely closed, aborting")
        ib.disconnect()
        sys.exit(1)

    print(f"{datetime.now().isoformat(timespec='seconds')} "
          f"market open, sampling {len(tickers)} instruments "
          f"({SAMPLES} X {SAMPLE_INTERVAL}s each)")

    measured = [t for t in tickers if t in typical_trade]
    if measured:
        print(f"trade sizes from {trade_file}: "
              + ", ".join(f"{t} {typical_trade[t]}" for t in measured))
    if candidates:
        print(f"trade sizes estimated for {', '.join(candidates)} "
              f"at weight {CANDIDATE_WEIGHT:.0%}, IDM {CANDIDATE_IDM}, "
              f"capital {capital_usd:,.0f} USD")

    runs = 0 if history.empty else history['sampled_at'].nunique()
    print(f"history: {len(history)} rows over {runs} runs")

    rows = []
    failed = []
    try:
        for path, ticker in zip(paths, tickers):
            try:
                icv, asof, close = latest_icv(path)
                spreads, fee_rate, shortable = sample_quotes(ib, ticker)
            except ConnectionError as exc:
                # the socket is gone, so nothing after this can succeed
                print(f"{ticker:6s} connection lost: {exc}")
                failed.extend(tickers[tickers.index(ticker):])
                break
            except Exception as exc:
                print(f"{ticker:6s} FAILED: {exc}")
                failed.append(ticker)
                continue

            if not spreads:
                print(f"{ticker}: no valid quotes - market closed, or no data subscription")
                failed.append(ticker)
                continue

            spread = float(np.median(spreads))
            half = spread / 2 * universe.multiplier(ticker)

            if ticker in typical_trade:
                shares, estimated = typical_trade[ticker], False
            else:
                shares, estimated = estimated_trade(icv, capital_usd), True

            comm = universe.commission(ticker, shares, close) / shares
            sc, C = standardised_cost(half, comm, icv)

            rows.append({
                'ticker': ticker,
                'sampled_at': datetime.now().isoformat(timespec='seconds'),
                'vol_asof': asof,
                'n_quotes': len(spreads),
                'spread_median': round(spread, 4),
                'spread_min': round(min(spreads), 4),
                'spread_max': round(max(spreads), 4),
                'half_spread': round(half, 5),
                'typical_shares': shares,
                'shares_estimated': estimated,
                'commission_per_share': round(comm, 5),
                'cost_per_share_C': round(C, 5),
                'icv': round(icv, 4),
                'annualised_icv': round(ANNUALISE * icv, 3),
                'standardised_cost': round(sc, 6),
                'speed_limit_turnover': round(SPEED_LIMIT_SR / sc, 1),
                'fee_rate': round(fee_rate, 5) if fee_rate is not None else None,
                'shortable_shares': shortable,
                'trade_sizes_from': trade_file,
            })
            mark = ' (estimated size)' if estimated else ''
            borrow = (f" borrow {fee_rate:.2%}" if fee_rate is not None
                      else "")
            avail = (f" shortable {human(shortable)}"
                     if shortable is not None else "")
            print(f"{ticker:6s} spread {spread:.4f} C {C:.5f} "
                  f"std cost {sc:.5f} SR speed limit "
                  f"{SPEED_LIMIT_SR/sc:.0f}/yr{borrow}{avail}{mark}")
    finally:
        ib.disconnect()

    if not rows:
        print("nothing collected")
        sys.exit(1)

    if failed:
        print(f"\nnot measured: {', '.join(failed)}")

    alarms = spike_report(rows, history)
    drift_report(rows)

    if any(row.get('fee_rate') is None for row in rows):
        print("\nno borrow fee from the API for at least one instrument. "
              "The authoritative source is the broker's Borrow Fee Details "
              "statement; shortable shares above are the available proxy.")

    print("\nto adopt these, set cost_per_share in universe.py:")
    for row in rows:
        mark = "   (estimated trade size)" if row['shares_estimated'] else ""
        print(f"    {row['ticker']:6s} {row['cost_per_share_C']:.5f}{mark}")

    out = pd.DataFrame(rows)
    COSTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = COSTS_DIR / f'costs_{stamp}.csv'
    out.to_csv(out_path, index=False)

    history_path = append_history(rows)
    print(f"\nwritten to {out_path}")
    print(f"appended to {history_path}")

    if alarms:
        print(f"\n*** {alarms} spike{'s' if alarms > 1 else ''} above - "
              "check before trading ***")


if __name__ == '__main__':
    main()
