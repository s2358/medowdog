"""
Daily price change and the volatility of one unit.

    python 2_volatility.py <price_csv> [<price_csv> ...]

Reads each downloaded price file and writes data/features/<TICKER>.csv with
the price columns plus 'change', 'simple_vol' and 'ewma_vol'. The downloaded
file is never modified, so a bad run is discarded by deleting the features
file rather than downloading the history again.

ewma_vol holds the currency move of one unit of the underlying rather than a
fraction, so position sizing can use it directly. The recursion is the
RiskMetrics one, sigma^2_t = lambda*sigma^2_{t-1} + (1 - lambda)*x^2_t, which
needs adjust=False. simple_vol is a rolling standard deviation in the same
units, kept only as a sanity check against it.
"""

import os
import sys

import numpy as np
import pandas as pd

import universe
from universe import ticker_from_path
from config import VOL_WINDOW, VOL_LAMBDA, BUSINESS_DAYS, FEATURES_DIR


def add_volatility(df, ticker, window=VOL_WINDOW, lam=VOL_LAMBDA):
    """
    Daily price change, and the volatility of one unit in currency.

    A share has a meaningful price level, so its volatility is estimated as a
    fraction and scaled by today's price: a 1% day carries the same risk
    whatever the price has done since. A back-adjusted futures series has no
    meaningful level - it can cross zero - so its volatility is estimated
    from price differences directly.
    """
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    df = df.sort_values('date').reset_index(drop=True)
    df['change'] = df['close'].diff()
    if universe.instrument(ticker)['type'] == 'STK':
        # a share's volatility is roughly proportional to its price, so it is
        # estimated as a percentage and scaled by today's close
        fraction = df['close'].pct_change()
        ewma_var = (fraction ** 2).ewm(alpha=1 - lam, adjust=False).mean()
        df['ewma_vol'] = np.sqrt(ewma_var) * df['close']
        df['simple_vol'] = fraction.rolling(window=window).std() * df['close']
    else:
        # a back-adjusted contract has no meaningful level - it is fictitious
        # and can cross zero - and its volatility does not scale with it, so
        # the price differences are used directly
        ewma_var = (df['change'] ** 2).ewm(alpha=1 - lam, adjust=False).mean()
        df['ewma_vol'] = np.sqrt(ewma_var)
        df['simple_vol'] = df['change'].rolling(window=window).std()

    return df


def process(path):
    ticker = ticker_from_path(path)
    df = add_volatility(pd.read_csv(path), ticker)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FEATURES_DIR / f'{ticker}.csv'
    df.to_csv(out, index=False)
    return df, out


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} "
              "<price_csv> [<price_csv> ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        ticker = ticker_from_path(path)
        df, out = process(path)
        last = df.iloc[-1]
        annual = last['ewma_vol'] * np.sqrt(BUSINESS_DAYS)
        print(f"{path}   ({len(df)} rows, to {last['date'].date()})")
        print(f"   simple vol ({VOL_WINDOW}d) {last['simple_vol']:.4f}   "
              f"ewma vol {last['ewma_vol']:.4f}/unit   "
              f"annualised {annual:.2f}/unit"
              + (f"  ({annual / last['close']:.1%})"
                 if universe.instrument(ticker)['type'] == 'STK' else ''))
        print(f"   written to {out}")

    span = 2 / (1 - VOL_LAMBDA) - 1
    print(f"\nlambda {VOL_LAMBDA} (span ~{span:.0f} days)")


if __name__ == '__main__':
    main()
