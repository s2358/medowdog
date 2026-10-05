import pandas as pd
import numpy as np
import sys
from config import (FORECAST_SCALARS, FORECAST_WEIGHTS, FDM, FORECAST_CAP, BUSINESS_DAYS, load_settings)


def calculate_ewma(prices, span):
    return pd.Series(prices).ewm(span=span, adjust=False).mean().values


def subsystem_returns(df, vol_target=0.3, business_days=256):
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


def calculate_forecasts(df):
    close_prices = df['close'].values
    ewma_vol = df['ewma_vol'].values

    ewmac_16_64 = calculate_ewma(close_prices, 16) - calculate_ewma(close_prices, 64)
    ewmac_32_128 = calculate_ewma(close_prices, 32) - calculate_ewma(close_prices, 128)
    ewmac_64_256 = calculate_ewma(close_prices, 64) - calculate_ewma(close_prices, 256)
    vol_in_price_points = ewma_vol * close_prices

    df['EWMAC_16_64'] = ewmac_16_64
    df['EWMAC_32_128'] = ewmac_32_128
    df['EWMAC_64_256'] = ewmac_64_256
    df['EWMAC_16_64_vol_std'] = ewmac_16_64 / vol_in_price_points
    df['EWMAC_32_128_vol_std'] = ewmac_32_128 / vol_in_price_points
    df['EWMAC_64_256_vol_std'] = ewmac_64_256 / vol_in_price_points
    df['EWMAC_16_64_scaled'] = (df['EWMAC_16_64_vol_std'] * 3.75).clip(-20, 20)
    df['EWMAC_32_128_scaled'] = (df['EWMAC_32_128_vol_std'] * 2.65).clip(-20, 20)
    df['EWMAC_64_256_scaled'] = (df['EWMAC_64_256_vol_std'] * 1.87).clip(-20, 20)

    W = [0.42, 0.16, 0.42]
    FDM = 1.07
    df['combined_forecast'] = ((W[0]*df['EWMAC_16_64_scaled']
                                + W[1]*df['EWMAC_32_128_scaled']
                                + W[2]*df['EWMAC_64_256_scaled']) * FDM).clip(-20, 20)

    # instrument currency volatility of one block (one share)
    # = daily price vol x block value x 100, and block value = close/100
    df['icv'] = df['ewma_vol'] * df['close']
    return df


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file_path>")
        sys.exit(1)
    input_file = sys.argv[1]

    df = pd.read_csv(input_file)
    df = calculate_forecasts(df)
    df['subsystem_returns'] = subsystem_returns(df)
    df.to_csv(input_file, index=False)

    print("EWMAC calculation with vol stdization, scaling, combined forecast and subsystem returns complete")
    print(df[['date', 'close', 'EWMAC_16_64', 'EWMAC_32_128', 'EWMAC_64_256',
              'EWMAC_16_64_vol_std', 'EWMAC_32_128_vol_std', 'EWMAC_64_256_vol_std',
              'EWMAC_16_64_scaled', 'EWMAC_32_128_scaled', 'EWMAC_64_256_scaled',
              'combined_forecast', 'subsystem_returns', 'icv']].tail())


if __name__ == '__main__':
    main()
