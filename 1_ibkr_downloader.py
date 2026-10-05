import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from ib_async import IB, Stock, util

# Longest duration IB Gateway has accepted in testing (34 Y was rejected)
CHUNK = "25 Y"

parser = argparse.ArgumentParser()
parser.add_argument("ticker")
args = parser.parse_args()

ib = IB()
ib.connect("127.0.0.1", 4001, clientId=1, readonly=True)

contract = Stock(args.ticker, "SMART", "USD")
ib.qualifyContracts(contract)
assert contract.conId, f"Could not resolve contract for {args.ticker}"

# Earliest available bar
head = ib.reqHeadTimeStamp(contract, whatToShow="TRADES", useRTH=True, formatDate=2)

all_bars = []
end = ""
while True:
    bars = ib.reqHistoricalData(
        contract,
        endDateTime=end,
        durationStr=CHUNK,
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    if not bars:
        break
    all_bars = bars + all_bars
    first = bars[0].date
    if first <= head.date():
        break
    # Next request ends at the start of the earliest day received
    end = datetime(first.year, first.month, first.day, tzinfo=timezone.utc)

ib.disconnect()

if not all_bars:
    sys.exit(f"No data returned for {args.ticker}")

df = util.df(all_bars).drop_duplicates(subset="date").sort_values("date")

out_dir = Path(__file__).parent / "data" / args.ticker
out_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = out_dir / f"{args.ticker}_{timestamp}.csv"
df.to_csv(out_path, index=False)

print(f"{len(df)} bars from {df['date'].iloc[0]} to {df['date'].iloc[-1]} saved to {out_path}")
