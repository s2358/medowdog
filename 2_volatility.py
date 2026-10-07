"""
Daily returns and EWMA volatility.

    python 2_volatility.py <csv> [<csv> ...]

Each file is rewritten in place with 'returns', 'simple_vol' and 'ewma_vol'.
Run after the downloader, before the EWMAC calculator script.

ewma_vol is the RiskMetrics recursion, sigma^2_t = lambda*sigma^2_{t-1} + (1 - lambda)*r^2_t, which needs adjust=False. simple_vol is a rolling standard deviation kept only as a sanity check against it.
"""

import os
import sys
import pandas as pd
import numpy as np
from config import VOL_WINDOW, VOL_LAMBDA, BUSINESS_DAYS

def add_volatility(df, window=VOL_WINDOW, lam=VOL_LAMBDA):
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
    df = df.sort_values('date').reset_index(drop=True)

    df['returns'] = df['close'].pct_change()
    df['simple_vol'] = df['returns'].rolling(window=window).std()

    ewma_var = (df['returns'] ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    df['ewma_vol'] = np.sqrt(ewma_var)
    return df

def process(path):
    df = add_volatility(pd.read_csv(path))
    df.to_csv(path, index=False)
    return df

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} "
              "<csv> [<csv> ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        df = process(path)
        last = df.iloc[-1]
        print(f"{path}   ({len(df)}  rows, to {last['date'].date()})")
        print(f"   simple vol  ({VOL_WINDOW}d) {last['simple_vol']:.4f}   "
              f"ewma vol {last['ewma_vol']:.4f}   "
              f"annualised {last['ewma_vol'] * np.sqrt(BUSINESS_DAYS):.1%}")
    span = 2 / (1 - VOL_LAMBDA) - 1
    print(f"\nlambda {VOL_LAMBDA} (span ~{span:.0f} days)")

if __name__ == '__main__':
    main()
