"""
Project configuration: values only, no behaviour.
Anything that reads a file or talks to a broker lives in its own module.
"""

from pathlib import Path

# --- paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
FEATURES_DIR = DATA_DIR / 'features'
CORRELATIONS_DIR = DATA_DIR / 'correlations'
PORTFOLIO_DIR = DATA_DIR / 'portfolio'
COSTS_DIR = DATA_DIR / 'costs'
FX_CACHE = DATA_DIR / 'fx_rate.json'
DIVIDENDS_DIR = DATA_DIR / 'dividends'
FITTED_DIR = DATA_DIR / 'fitted'
ORDERS_DIR = DATA_DIR / 'orders'
FILLS_DIR = DATA_DIR / 'fills'

# --- account ---
VOL_TARGET = 0.20
HOME_CURRENCY = 'SGD'

# --- capital ---
# the backtest's starting capital. Fixed, not the live account value, so the
# same prices and the same code always give the same result. Rounding to
# whole shares and the commission floor are not linear in size, so a moving
# start moves the CAGR, the drawdown and the Sharpe.
BACKTEST_CAPITAL = 200_000
CAPITAL_MAX_AGE_HOURS = 12   # the account value must be re-read at least this often

TARGET_STALE_DAYS = 3    # how far behind the newest an instrument's data may be

# --- fx ---
FX_PAIR = 'USDSGD'      # home currency per unit of instrument currency
FX_SERIES = DATA_DIR / 'fx' / f'{FX_PAIR}.csv'
FX_MAX_AGE_DAYS = 7

# --- rates ---
RATES_DIR = DATA_DIR / 'rates'
BENCHMARK_FILE = RATES_DIR / 'usd_benchmark.csv'

# paid above the benchmark on money the broker has lent. Published tier 1
# (0 to 100,000 USD), not measured: the account has never held a debit
# balance long enough to solve the rate from its own statements.
FINANCING_SPREAD = 0.015

# given up below the benchmark on a cash credit. Measured from the 23 October
# 2026 daily statement: (150,891.51 - 10,000) x 3.38% / 360 = 13.23, which is
# the accrual to the cent. 3.38% is the published benchmark less 0.5%.
CREDIT_SPREAD = 0.005

# USD. A balance below this earns nothing. From the same calculation: without
# the 10,000 deduction the implied rate is 3.157%, which matches no published
# figure.
CASH_THRESHOLD = 10_000

# annual fee on the market value of a short stock position, charged whatever
# the cash balance. Measured over 23 to 29 October 2026: 2.59 USD on 253,440
# USD-days. The quoted rate moved between 0.25% and 0.43% day by day.
BORROW_FEE = 0.0037

# --- leverage ---
# Reg T sets 25% overnight maintenance, so 4x is where IBKR liquidates rather than margin calls
MAX_LEVERAGE = 4.0

# --- dividends ---
DIVIDEND_MAX_AGE_DAYS = 90

# --- interactive brokers ---
IB_HOST = '127.0.0.1'
IB_PORT = 4002          # 4001 gateway live, 4002 gateway paper
IB_MARKET_DATA_TYPE = 3 # 1 live, 3 delayed
LIMIT_BUFFER = 0.003    # cross the quoted touch - quotes are delayed

# --- framework parameters ---
VOL_WINDOW = 30     # rolling window for the sanity-check vol
VOL_LAMBDA = 0.9459   # EWMA decay; span = 2/(1-lambda) - 1 ~36 days
BUSINESS_DAYS = 256
WARMUP = 256            # slowest EWMA span

FORECAST_CAP = 20
FORECAST_SCALARS = {'16_64': 3.75, '32_128': 2.65, '64_256': 1.87}
FORECAST_WEIGHTS = {'16_64': 0.42, '32_128': 0.16, '64_256': 0.42}
FDM = 1.099              # forecast diversification multiplier
IDM = 1.4239             # instrument diversification multiplier, measured by 4_correlation.py

IDM_CAP = 2.5           # instrument diversification multiplier ceiling
FDM_CAP = 2.5           # forecast diversification multiplier ceiling
INERTIA = 0.10         # don't trade if the gap is under 10%
SPEED_LIMIT_SR = 0.13   # max annual cost in Sharpe ratio units

COST_MULTIPLIER = 1.0       # scale measured costs, for sensitivity runs

# --- tax? ---
US_WITHHOLDING = 0.30

# --- backtest window (None means all available rows) ---
# the portfolio is tested with every constituent present. GLDM is the
# newest: its file starts 2018-06-26 and its first usable row, after the
# 256-row warmup, is 2019-07-03. Revisit this date whenever the universe
# changes.
WINDOW_START = '2019-07-03'
WINDOW_END = None
