"""
Download daily bars from IB Gateway.
    python 1_ibkr_downloader.py SPY GLDM SLV

Each run writes data/<TICKER>/<TICKER>_<timestamp>.csv and moves any previous download into data/<TICKER>/archive/. The live folder therefore always holds exactly one file per ticker, so data/<TICKER>/*.csv is unambiguous, while every earlier download is kept.
"""

import argparse
import sys
from datetime import datetime, timezone
from ib_async import IB, Stock, util
from config import IB_HOST, IB_PORT, DATA_DIR

CLIENT_ID = 1
CHUNK = "25 Y"        # Longest duration IB Gateway has accepted in testing (34 Y was rejected)

def download(ib, ticker):
    """Page backwards through history until the earliest available bar."""
    contract = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(contract)
    if not contract.conId:
        raise ValueError(f"could not resolve contract for {ticker}")

    head = ib.reqHeadTimeStamp(contract, whatToShow="TRADES", useRTH=True, formatDate=2)

    all_bars = []
    end = ""
    previous_first = None

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
        if previous_first is not None and first >= previous_first:
            # not moving backwards - stop rather than loop forever
            print(f"  {ticker}: pagination stalled at {first}, stopping")
            break
        previous_first = first

        # next request ends at the start of the earliest day received
        end = datetime(first.year, first.month, first.day, tzinfo=timezone.utc)

    return all_bars

def archive_previous(out_dir, ticker):
    """Move earlier downloads aside so the live folder holds only the newest."""
    old = sorted(out_dir.glob(f'{ticker}_*.csv'))
    if not old:
        return 0

    archive_dir = out_dir / 'archive'
    archive_dir.mkdir(exist_ok=True)
    for path in old:
        path.rename(archive_dir / path.name)
    return len(old)

def save(ticker, bars):
    df = util.df(bars).drop_duplicates(subset="date").sort_values("date")

    out_dir = DATA_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    moved = archive_previous(out_dir, ticker)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ticker}_{stamp}.csv"
    df.to_csv(out_path, index=False)

    note = f"   ({moved} archived)" if moved else ""

    print(f"{ticker:6s} {len(df):5d} bars  {df['date'].iloc[0]} -> "
          f"{df['date'].iloc[-1]}  {out_path.name}{note}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+")
    args = parser.parse_args()

    ib = IB()
    ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, readonly=True)

    failed = []
    try:
        for ticker in args.tickers:
            ticker = ticker.upper()
            try:
                bars = download(ib, ticker)
            except Exception as exc:
                print(f"{ticker:6s} FAILED: {exc}")
                failed.append(ticker)
                continue

            if not bars:
                print(f"{ticker:6s} no data returned")
                failed.append(ticker)
                continue

            save(ticker, bars)
    finally:
        ib.disconnect()

    if failed:
        sys.exit(f"\nfailed: {', '.join(failed)}")


if __name__ == '__main__':
    main()



