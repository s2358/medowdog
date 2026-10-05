import pandas as pd
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python script.py <csv_path>")
    sys.exit(1)

csv_path = sys.argv[1] 

df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
df = df.sort_values('date').reset_index(drop=True)
df['returns'] = df['close'].pct_change()

window = 30

df['simple_vol'] = df['returns'].rolling(window=window).std()
lambda_param = 0.94
df['squared_returns'] = df['returns'] ** 2
ewma = df['squared_returns'].ewm(alpha=1-lambda_param).mean()

df['ewma_vol'] = np.sqrt(ewma)
df.to_csv(csv_path, index=False)

print("Last 20 days:")
print(df[['date', 'close', 'returns', 'simple_vol', 'ewma_vol']].tail(20).to_string())
print(f"\nLatest simple vol (30-day): {df['simple_vol'].iloc[-1]:.4f}")
print(f"Latest EWMA vol: {df['ewma_vol'].iloc[-1]:.4f}")
