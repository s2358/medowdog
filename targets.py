"""
Today's target positions.

    python targets.py <allocation_csv> <features_csv> [<features_csv> ...]

The shares to hold in each instrument right now, sized against the account
value recorded by capital.py rather than any simulated equity.

Only the latest row of each features file is used, each instrument on its own
last trading date. Instruments do not share a calendar, so there is no date
all of them have; a file that lags the others by more than TARGET_STALE_DAYS
is a failed download rather than a holiday, and the run refuses it. No
history is walked, so none of what the backtest needs - measured costs,
dividends, financing rates - is read here, and a fault in any of them cannot
stop the targets being written.

Writes data/portfolio/live_targets.csv, which the order script reads.
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import capital
import fx_data
import sizing
from config import (VOL_TARGET, HOME_CURRENCY, WARMUP, PORTFOLIO_DIR, TARGET_STALE_DAYS, BENCHMARK_FILE)
from universe import ticker_from_path

TARGETS_FILE = 'live_targets.csv'
COLUMNS = ['date', 'close', 'combined_forecast', 'icv']
ARCHIVE_DIR = PORTFOLIO_DIR / 'archive'


def archive_previous(path):
    """
    Move the standing targets aside under the time they were written.

    The file is what the orders were derived from, so a run that replaces it
    must not be able to destroy the record of what the previous run decided.
    """
    if not path.exists():
        return None
    stamp = datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y%m%d_%H%M%S')
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    kept = ARCHIVE_DIR / f'{path.stem}_{stamp}{path.suffix}'
    path.rename(kept)
    return kept


def load_instrument(path):
    """The columns a position needs, indexed by date."""
    df = pd.read_csv(path, usecols=COLUMNS).iloc[WARMUP:]
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')


def load_allocation(path):
    """Instrument weights and the diversification multiplier."""
    alloc = pd.read_csv(path).set_index('ticker')
    idm = alloc['idm'].iloc[0]
    if not np.allclose(alloc['idm'], idm):
        raise ValueError("idm differs between rows of the allocation file")
    return alloc['instrument_weight'].to_dict(), float(idm)


def on_index(frame, index):
    """
    The frame on a shared index, carried forward within its own span.

    Before the instrument's first row it stays empty, which is what makes the
    position flat rather than wrong. After it, a day the instrument did not
    trade repeats the last close, so the P&L for that day is zero.
    """
    aligned = frame.reindex(index)
    live = index >= frame.index[0]
    aligned.loc[live] = aligned.loc[live].ffill()
    return aligned


def support_start(path=BENCHMARK_FILE):
    """
    The first date both support series cover.

    Every row needs an fx rate to convert its P&L and a benchmark rate to
    price its borrowing. The instruments' own price history reaches further
    back than either, so the support series set the start, not the prices.

    The benchmark borrows nothing and so needs only the fx series, but it
    uses this same date anyway: a comparison over a different window is not
    a comparison.
    """
    rates = pd.read_csv(path, parse_dates=['observation_date'])
    return max(fx_data.load_series().index[0],
               rates['observation_date'].iloc[0])



def latest_rows(frames, stale_days=TARGET_STALE_DAYS):
    """
    Each instrument's own last row, and the newest date among them.

    Instruments do not all trade on the same calendar, so there is no single
    date every one of them has. A file further behind than the others is not
    a holiday, though - it is a download that did not happen, and sizing from
    it would set today's position from a stale forecast.
    """
    newest = max(frame.index[-1] for frame in frames.values())
    rows, behind = {}, []
    for ticker, frame in frames.items():
        last = frame.index[-1]
        if (newest - last).days > stale_days:
            behind.append(f"{ticker} {last.date()}")
        rows[ticker] = frame.loc[last]
    if behind:
        raise ValueError(f"the newest data is {newest.date()}, but "
                         f"{', '.join(behind)} is behind it - re-run the "
                         "downloader")
    return rows, newest


def live_targets(frames, weights, idm, capital_usd, vol_target):
    """Today's positions, sized against the capital actually held."""
    rows, newest = latest_rows(frames)
    daily_target = sizing.daily_cash_vol_target(capital_usd, vol_target)

    built = []
    for ticker in sorted(frames):
        last = rows[ticker]
        icv = last['icv']
        forecast = last['combined_forecast']
        vol_scalar = sizing.volatility_scalar(daily_target, icv)
        target = sizing.target_position(forecast, icv, daily_target,
                                        weights[ticker], idm)
        built.append({
            'ticker': ticker,
            'date': last.name.date(),
            'close': last['close'],
            'combined_forecast': forecast,
            'icv': icv,
            'vol_scalar': vol_scalar,
            'subsystem_position': forecast / 10 * vol_scalar,
            'portfolio_position': target,
            'rounded': sizing.round_half_away(target),
        })
    return pd.DataFrame(built).set_index('ticker')


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {os.path.basename(sys.argv[0])} "
              "<allocation_csv> <features_csv> [<features_csv> ...]")
        sys.exit(1)

    spot = fx_data.load()
    stored, age = capital.load()
    if stored['currency'] != HOME_CURRENCY:
        raise ValueError(f"the account is in {stored['currency']}, "
                         f"the system reports in {HOME_CURRENCY}")
    account_value = stored['value']
    capital_usd = account_value / spot

    weights, idm = load_allocation(sys.argv[1])
    frames = {ticker_from_path(p): load_instrument(p) for p in sys.argv[2:]}

    missing = set(frames) - set(weights)
    if missing:
        raise KeyError(f"no weight in allocation file for: {sorted(missing)}")

    live = live_targets(frames, weights, idm, capital_usd, VOL_TARGET)
    dates = sorted(set(live['date']))
    asof = (f"{dates[-1]}" if len(dates) == 1
            else f"{dates[-1]}, oldest {dates[0]}")
    
    print(f"capital {account_value:,.0f} {HOME_CURRENCY} = "
          f"{capital_usd:,.0f} USD at {spot:.5f}   "
          f"read {age.total_seconds() / 3600:.1f}h ago   "
          f"vol target {VOL_TARGET:.0%}   IDM {idm:.4f}")
    print(f"targets as of {asof}\n")
    print(live[['close', 'combined_forecast', 'rounded']].round(2).to_string())

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    out = PORTFOLIO_DIR / TARGETS_FILE
    kept = archive_previous(out)
    live.to_csv(out)
    print(f"\nwritten to {out}"
          + (f"   (previous kept as {kept.name})" if kept else ""))


if __name__ == '__main__':
    main()
