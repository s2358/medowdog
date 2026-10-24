"""
Record the account value that position sizing uses.

    python capital.py

Net liquidation value is the figure used for capital. cash plus the market value of everything held, in the account's base currency.

The change since the last reading is printed.

Write data/capital.json.
"""

import json
from datetime import datetime, timedelta
from account import summary
from config import DATA_DIR, CAPITAL_MAX_AGE_HOURS

CAPITAL_FILE = DATA_DIR / 'capital.json'
TAG = 'NetLiquidation'

def fetch():
    """Net liquidation value, its currency, and he account it belongs to."""
    accounts, values = summary()
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
    limit is refused rather than used. Pass None to read it regardless, which
    is what this script does when reporting the change since last time.
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


def main():
    previous = None
    if CAPITAL_FILE.exists():
        previous = json.loads(CAPITAL_FILE.read_text())

    current = fetch()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CAPITAL_FILE.write_text(json.dumps(current, indent=2))

    print(f"{current['fetched_at']}  account {current['account']}")
    line = f"  capital {current['value']:,.2f} {current['currency']}"
    if previous and previous.get('currency') == current['currency']:
        change = current['value'] - previous['value']
        pct = change / previous['value'] * 100 if previous['value'] else 0.0
        line += (f"  {change:+,.2f} ({pct:+.2f}%) since "
                 f"{previous['fetched_at']}")
    print(line)
    print(f"written to {CAPITAL_FILE}")



if __name__ == '__main__':
    main()
