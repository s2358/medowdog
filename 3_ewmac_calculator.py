"""
EWMAC forecasts, combined forecast, subsystem returns and ICV.

    python 3_ewmac_calculator.py <features_csv> [<features_csv> ...]

Each features file is rewritten with the new columns added. Run after the
volatility script, which creates those files and supplies 'returns' and
'ewma_vol'.
"""

import pandas as pd
import numpy as np
import os
import sys
import universe
from universe import ticker_from_path
from config import (FORECAST_SCALARS, FORECAST_WEIGHTS, FDM, FORECAST_CAP, BUSINESS_DAYS, VOL_TARGET)

REPORT_COLS = ['date', 'close', 'combined_forecast', 'subsystem_returns', 'icv']


def calculate_ewma(prices, span):
    return pd.Series(prices).ewm(span=span, adjust=False).mean().values


def subsystem_returns(df, vol_target=VOL_TARGET, business_days=BUSINESS_DAYS):
    """
        Daily subsystem return as a fraction of capital.
        df must contain: 'returns', 'ewma_vol', 'combined_forecast'
        Forecast and vol are lagged one day - you trade on yesterday's signal.
    """
    return(
        df['combined_forecast'].shift(1) / 10
        * (vol_target / business_days ** 0.5)
        * (df['returns'] / df['ewma_vol'].shift(1))
    )


def calculate_forecasts(df, ticker):
    close_prices = df['close'].values

    # the currency value of a one-day move in one unit. Dividing each
    # crossover by it turns a price difference into a risk-adjusted number,
    # which is what makes a fast rule and a slow rule comparable, and it is
    # also what position sizing needs to know how risky a unit is.
    df['icv'] = df['ewma_vol'] * df['close'] * universe.multiplier(ticker)
    icv = df['icv'].values

    for name in FORECAST_SCALARS:
        fast, slow = (int(n) for n in name.split('_'))
        crossover = (calculate_ewma(close_prices, fast)
                     - calculate_ewma(close_prices, slow))
        df[f'EWMAC_{name}'] = crossover
        df[f'EWMAC_{name}_vol_std'] = crossover / icv
        df[f'EWMAC_{name}_scaled'] = (df[f'EWMAC_{name}_vol_std']
                                      * FORECAST_SCALARS[name]).clip(
                                              -FORECAST_CAP, FORECAST_CAP)

    raw = sum(FORECAST_WEIGHTS[name] * df[f'EWMAC_{name}_scaled']
              for name in FORECAST_WEIGHTS)
    df['combined_forecast'] = (raw * FDM).clip(-FORECAST_CAP, FORECAST_CAP)
    return df


def process(path):
    df = pd.read_csv(path)
    df = calculate_forecasts(df, ticker_from_path(path))
    df['subsystem_returns'] = subsystem_returns(df)
    df.to_csv(path, index=False)
    return df

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} " "<csv> [<csv> ...] ")
        sys.exit(1)
    
    missing = set(FORECAST_SCALARS) ^ set(FORECAST_WEIGHTS)
    if missing:
        raise KeyError(
                f"FORECAST_SCALARS and FORECAST_WEIGHTS disagree on: {sorted(missing)}")

    for path in sys.argv[1:]:
        df = process(path)
        print(f"\n{path}  ({len(df)} rows)")
        print(df[REPORT_COLS].tail().to_string(index=False))

    print(f"\nrules: {', '.join(FORECAST_SCALARS)} FDM {FDM}   "
          f"cap +/-{FORECAST_CAP}")


if __name__ == '__main__':
    main()
