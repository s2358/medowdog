from ib_async import IB, Contract, util
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

Path("data").mkdir(parents=True, exist_ok=True)

TICKERS = {
    "IAU": 474829650,
    "EEM": 6604766,
    "TLT": 15547841,
    "DBC": 319355208
}

ib = IB()
ib.connect("127.0.0.1", 4001, clientId=7, readonly=True)

for symbol, conid in TICKERS.items():
    contract = Contract(secType="STK", conId=conid, exchange="ARCA" if symbol == "TLT" else "SMART", currency="USD")
    ib.qualifyContracts(contract)
    assert contract.symbol == symbol, f"conid {conid} resolved to {contract.symbol}, expected {symbol}"
    bars_t = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="25 Y",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
        timeout=120,
    )
    print(f"{symbol} TRADES: {len(bars_t)} bars, {bars_t[0].date} to {bars_t[-1].date}")
    bars_a = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="25 Y",
        barSizeSetting="1 day",
        whatToShow="ADJUSTED_LAST",
        useRTH=True,
        formatDate=1,
        timeout=120,
    )

    df_t = util.df(bars_t)[["date", "open", "high", "low", "close", "volume"]]
    df_a = util.df(bars_a)[["date", "close"]].rename(columns={"close": "adj_close"})
    assert set(df_t.date) == set(df_a.date), f"{symbol}: TRADES and ADJ date sets differ"
    print(f"{symbol} ADJ: {len(bars_a)} bars, {bars_a[0].date} to {bars_a[-1].date}")
    df = df_t.merge(df_a, on="date").sort_values("date")

    today_us = datetime.now(ZoneInfo("America/New_York")).date()
    df = df[df.date < today_us]

    out = Path("data") / f"{symbol}.csv"
    df.to_csv(out, index=False)
    print(f"{symbol}: wrote {len(df)} rows to {out}")

    ib.sleep(2)




