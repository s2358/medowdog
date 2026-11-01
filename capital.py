"""
Record the account value that position sizing uses.

    python capital.py

Net liquidation value is the figure used for capital: cash plus the market
value of everything held, in the account's base currency.

Two files. capital.json holds the latest reading, which is all position
sizing needs. readings.csv holds every reading ever taken and is never
rewritten - without it there is no record of what the account did, only of
what the simulation did.

The change since the last reading is printed.

Writes data/capital.json and appends to data/capital/readings.csv.
"""

import csv
import json
from datetime import datetime, timedelta

from account import summary
from config import DATA_DIR, CAPITAL_MAX_AGE_HOURS

CAPITAL_FILE = DATA_DIR / 'capital.json'
READINGS_FILE = DATA_DIR / 'capital' / 'readings.csv'
TAG = 'NetLiquidation'


def fetch():
    """Net liquidation value, its currency, and the account it belongs to."""
    accounts, values, _ = summary()
    found = next((v for v in values if v.tag == TAG), None)
    if found is None:
        raise RuntimeError(f"the broker reported no {TAG}")
    return {
        'account': ', '.join(accounts) if accounts else None,
        'value': float(found.value),
        'currency': found.currency,
        'fetched_at': datetime.now().isoformat(timespec='seconds'),
    }


def load(max_age_hours=CAPITAL_MAX_AGE_HOURS):
    """
    The stored account value and its age.

    Risk is scaled to the capital actually held, so a figure older than the
    limit is refused rather than used. Pass None to read it regardless.
    """
    if not CAPITAL_FILE.exists():
        raise FileNotFoundError(f"{CAPITAL_FILE} not found - run capital.py first")
    stored = json.loads(CAPITAL_FILE.read_text())
    age = datetime.now() - datetime.fromisoformat(stored['fetched_at'])
    if max_age_hours is not None and age > timedelta(hours=max_age_hours):
        raise ValueError(
                f"the account value was read "
                f"{age.total_seconds() / 3600:.1f}h ago, past the "
                f"{max_age_hours}h limit - run capital.py")
    return stored, age


def record(reading, path=READINGS_FILE):
    """
    Append one reading to the permanent record.

    Duplicates are kept: two runs an hour apart are two true observations of
    the account, and deciding which one counts belongs further downstream.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    if not fresh and path.stat().st_size:
        with path.open('rb') as existing:
            existing.seek(-1, 2)
            if existing.read(1) != b'\n':
                with path.open('a') as repair:
                    repair.write('\n')
    with path.open('a', newline='') as out:
        writer = csv.writer(out)
        if fresh:
            writer.writerow(['read_at', 'account', 'value', 'currency'])
        writer.writerow([reading['fetched_at'], reading['account'],
                         f"{reading['value']:.2f}", reading['currency']])
    return path


def main():
    previous = None
    if CAPITAL_FILE.exists():
        previous = json.loads(CAPITAL_FILE.read_text())

    current = fetch()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CAPITAL_FILE.write_text(json.dumps(current, indent=2))
    readings = record(current)

    print(f"{current['fetched_at']}  account {current['account']}")
    line = f"  capital {current['value']:,.2f} {current['currency']}"
    if previous and previous.get('currency') == current['currency']:
        change = current['value'] - previous['value']
        pct = change / previous['value'] * 100 if previous['value'] else 0.0
        line += (f"  {change:+,.2f} ({pct:+.2f}%) since "
                 f"{previous['fetched_at']}")
    print(line)
    print(f"written to {CAPITAL_FILE}, appended to {readings}")


if __name__ == '__main__':
    main()
