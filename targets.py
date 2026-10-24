"""
Today's target positions.

    python targets.py <allocation_csv> <features_csv> [<features_csv> ...]

The shares to hold in each instrument right now, sized against the account
value recorded by capital.py rather than any simulated equity.

Only the latest row of each features file is used, taken on the last date
every instrument traded. No history is walked, so none of what the backtest
needs - measured costs, dividends, financing rates - is read here, and a
fault in any of them cannot stop the targets being written.

Writes data/portfolio/live_targets.csv, which the order script reads.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import capital
import fx_data
import sizing
from config import VOL_TARGET, HOME_CURRENCY, WARMUP, PORTFOLIO_DIR
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


def common_index(frames):
    """
    The dates every instrument traded.

    A position is a portfolio decision, so it can only be taken on a day the
    whole portfolio has a price for.
    """
    common = None
    for df in frames.values():
        common = df.index if common is None else common.intersection(df.index)
    if common is None or len(common) == 0:
        raise ValueError("the instruments share no trading days")
    return common


def live_targets(frames, weights, idm, capital_usd, vol_target):
    """Today's positions, sized against the capital actually held."""
    asof = common_index(frames)[-1]
    daily_target = sizing.daily_cash_vol_target(capital_usd, vol_target)

    rows = []
    for ticker in sorted(frames):
        last = frames[ticker].loc[asof]
        ivv = last['icv']
        forecast = last['combined_forecast']
        vol_scalar = sizing.volatility_scalar(daily_target, ivv)
        target = sizing.target_position(forecast, ivv, daily_target,
                                        weights[ticker], idm)
        rows.append({
            'ticker': ticker,
            'date': asof.date(),
            'close': last['close'],
            'combined_forecast': forecast,
            'icv': ivv,
            'vol_scalar': vol_scalar,
            'subsystem_position': forecast / 10 * vol_scalar,
            'portfolio_position': target,
            'rounded': sizing.round_half_away(target),
        })
    return pd.DataFrame(rows).set_index('ticker')


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

    print(f"capital {account_value:,.0f} {HOME_CURRENCY} = "
          f"{capital_usd:,.0f} USD at {spot:.5f}   "
          f"read {age.total_seconds() / 3600:.1f}h ago   "
          f"vol target {VOL_TARGET:.0%}   IDM {idm:.4f}")
    print(f"targets as of {live['date'].iloc[0]}\n")
    print(live[['close', 'combined_forecast', 'rounded']].round(2).to_string())

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    out = PORTFOLIO_DIR / TARGETS_FILE
    kept = archive_previous(out)
    live.to_csv(out)
    print(f"\nwritten to {out}"
          + (f"   (previous kept as {kept.name})" if kept else ""))


if __name__ == '__main__':
    main()
