"""
Fit the EWMAC forecast scalars from the price files.

    python fit_scalars.py <csv> [<csv> ...]

A forecast scalar is the constant that brings a rule's forecast to an average absolute value of 10. For each instrument, take the mean absolute vol-standardised forecast with no cap applied; average those means across instruments; divide the result into 10.

Reads the EWMAC_<rule>_vol_std columns written by script 3. Those columns do not depend on the scalars.

Nothing is written back to config.py.
"""

import os 
import sys
from datetime import datetime
import pandas as pd
from config import FORECAST_SCALARS, WARMUP, FITTED_DIR, WINDOW_START, WINDOW_END

def ticker_from_path(path):
    """Trim the filename to retain only the ticker."""
    return os.path.basename(path).split('_')[0].upper()


def in_window(frame):
    """Rows inside the configured window. Needs a 'date' column."""
    if WINDOW_START is not None:
        frame = frame[frame['date'] >= pd.Timestamp(WINDOW_START)]
    if WINDOW_END is not None:
        frame = frame[frame['date'] < pd.Timestamp(WINDOW_END)]
    return frame

def absolute_means(path):
    """Mean |vol-standardised forecast| per rule, with the row count."""
    columns = [f'EWMAC_{rule}_vol_std' for rule in FORECAST_SCALARS]
    frame = pd.read_csv(path, usecols=['date'] + columns,
                        parse_dates=['date']).iloc[WARMUP:].dropna()
    frame = in_window(frame)
    means = {rule: frame[f'EWMAC_{rule}_vol_std'].abs().mean()
             for rule in FORECAST_SCALARS}
    return pd.Series(means, name=ticker_from_path(path)), len(frame)

def fit(paths):
    """Per-instrument mean |forecast| and the pooled scalars, as a DataFrame"""
    rows, counts = [], {}
    for path in paths:
        means, n = absolute_means(path)
        if means.name in counts:
            raise ValueError(f"same ticker given more than once: {means.name}")
        rows.append(means)
        counts[means.name] = n

    table = pd.DataFrame(rows)
    table.index.name = 'instrument'

    # average the absolute values across instruments, then divide into 10,
    # in that order - averaging the per-instrument scalars instead would 
    # weight the instrument with the smallest forecasts most heavily
    fitted = 10.0 / table.mean()
    current = pd.Series(FORECAST_SCALARS)
    out = pd.DataFrame({
        'mean_abs_forecast': table.mean(),
        'fitted': fitted,
        'current': current,
        'ratio': fitted / current,
        'spread': table.max() / table.min(),
        })
    out.index.name = 'rule'
    return table, out, counts

def archive_previous(out_dir, pattern='forecast_scalars_*.csv'):
    old = sorted(out_dir.glob(pattern))
    if not old:
        return 0
    archive_dir = out_dir / 'archive'
    archive_dir.mkdir(exist_ok=True)
    for path in old:
        path.rename(archive_dir / path.name)
    return len(old)

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <csv> [<csv> ...]")
        sys.exit(1)

    table, out, counts = fit(sys.argv[1:])

    print(f"\nscalar each instrument would give on its own, "
          f"warmup of {WARMUP} rows dropped")
    print((10.0 / table).round(2).to_string())
    print("rows: " + "  ".join(f"{t} {n}" for t, n in counts.items()))

    print("\npooled")
    print(out.round(3).to_string())

    worst = out['spread'].max()
    print(f"\nwidest spread across instruments {worst:.2f}x")
    if worst > 1.5:
        print(" a properly vol-standardised rule gives a similar value on "
              "every instrument -\n a wide spread points at the volatility "
              "normalisation, not the scalar")

    FITTED_DIR.mkdir(parents=True, exist_ok=True)
    archived = archive_previous(FITTED_DIR)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = FITTED_DIR / f'forecast_scalars_{stamp}.csv'
    out.to_csv(path)
    note = f"  ({archived} archived)" if archived else ""
    print(f"\nwritten to {path}{note}")


if __name__ == '__main__':
    main()

