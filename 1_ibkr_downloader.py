"""
Download daily bars from IB Gateway.
    python 1_ibkr_downloader.py SPY GLDM SLV

Each run writes data/<TICKER>/<TICKER>_<timestamp>.csv and moves any previous download into data/<TICKER>/archive/. The live folder therefore always holds exactly one file per ticker, so data/<TICKER>/*.csv is unambiguous, while every earlier download is kept.

A download that does not reach the start of the available history is not saved, so the previous file stays live.
"""

import argparse
import sys
from datetime import datetime, time
from ib_async import IB
import bars
import universe
from config import IB_HOST, IB_PORT, DATA_DIR

CLIENT_ID = 1
SESSION_CLOSE = {'STK': time(16, 0), 'FUT': time(17, 0)}

def download(ib, ticker):
    """
    Daily bars for one ticker, as (DataFrame, complete). 
    
    A future trades nearly around the clock, so its daily bar is taken from the whole session rather than the hours an equity exchange keeps.
    """
    contract = universe.contract(ticker)
    ib.qualifyContracts(contract)
    if not contract.conId:
        raise ValueError(f"could not resolve contract for {ticker}")

    use_rth = universe.instrument(ticker)['type'] == 'STK'
    return bars.daily(ib, contract, "TRADES", use_rth=use_rth, label=ticker)


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

def save(ticker, df):
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
                frame, complete = download(ib, ticker)
            except Exception as exc:
                print(f"{ticker:6s} FAILED: {exc}")
                failed.append(ticker)
                continue

            if not complete:
                print(f"{ticker:6s} NOT SAVED - bars start "
                      f"{frame['date'].iloc[0]} and the walk did not reach "
                      f"the start of the history")
                failed.append(ticker)
                continue

            closes_at = SESSION_CLOSE[universe.instrument(ticker)['type']]
            frame, dropped = bars.drop_unsettled(frame, closes_at)
            if dropped is not None:
                print(f"{ticker:6s} dropped the unsettled {dropped} bar")

            save(ticker, frame)
    finally:
        ib.disconnect()

    if failed:
        sys.exit(f"\nfailed: {', '.join(failed)}")


if __name__ == '__main__':
    main()



