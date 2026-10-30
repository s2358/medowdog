"""
EWMAC forecasts, combined forecast, subsystem returns and ICV.

    python 3_ewmac_calculator.py <features_csv> [<features_csv> ...]

Each features file is rewritten with the new columns added. Run after the
volatility script, which creates those files and supplies 'change' and
'ewma_vol'.
"""

import os
import sys

import numpy as np
import pandas as pd

import universe
from universe import ticker_from_path
from config import (FORECAST_SCALARS, FORECAST_WEIGHTS, FDM, FORECAST_CAP,
                    BUSINESS_DAYS, VOL_TARGET)

REPORT_COLS = ['date', 'close', 'combined_forecast', 'subsystem_returns', 'icv']


def calculate_ewma(prices, span):
    return pd.Series(prices).ewm(span=span, adjust=False).mean().values


def subsystem_returns(df, vol_target=VOL_TARGET, business_days=BUSINESS_DAYS):
    """
    Daily subsystem return as a fraction of capital.

    df must contain 'change', 'ewma_vol' and 'combined_forecast'. Both are in
    the same units, so the ratio is the move in units of its own volatility
    whether the instrument is a share or a contract.

    Forecast and volatility are lagged one day - you trade on yesterday's
    signal.
    """
    return (
        df['combined_forecast'].shift(1) / 10
        * (vol_target / business_days ** 0.5)
        * (df['change'] / df['ewma_vol'].shift(1))
    )


def calculate_forecasts(df, ticker):
    close_prices = df['close'].values

    # the currency value of a one-day move in one contract - what position
    # sizing needs to know how risky a contract is
    df['icv'] = df['ewma_vol'] * universe.multiplier(ticker)

    # the crossover is a price difference, so it is standardised by the
    # volatility of price differences, not by a currency amount. The two are
    # the same number only when one contract is one unit.
    vol = df['ewma_vol'].values

    for name in FORECAST_SCALARS:
        fast, slow = (int(n) for n in name.split('_'))
        crossover = (calculate_ewma(close_prices, fast)
                     - calculate_ewma(close_prices, slow))
        df[f'EWMAC_{name}'] = crossover
        df[f'EWMAC_{name}_vol_std'] = crossover / vol
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
        print(f"Usage: python {os.path.basename(sys.argv[0])} "
              "<features_csv> [<features_csv> ...]")
        sys.exit(1)

    missing = set(FORECAST_SCALARS) ^ set(FORECAST_WEIGHTS)
    if missing:
        raise KeyError("FORECAST_SCALARS and FORECAST_WEIGHTS disagree on: "
                       f"{sorted(missing)}")

    for path in sys.argv[1:]:
        df = process(path)
        print(f"\n{path}  ({len(df)} rows)")
        print(df[REPORT_COLS].tail().to_string(index=False))

    print(f"\nrules: {', '.join(FORECAST_SCALARS)} FDM {FDM}   "
          f"cap +/-{FORECAST_CAP}")


if __name__ == '__main__':
    main()
