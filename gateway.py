"""
Is IB Gateway up and able to serve data?

    python gateway.py

Three checks, because they fail differently. The port answering says some
process is listening. The API handshake completing says that process is a
gateway willing to talk. A short historical request says the data farm behind
it is alive.

The third check matters most. A gateway whose historical farm has dropped
answers the port and completes the handshake, then times out on every bar
request. The downloader then grinds through four chunks, each tried twice at
a 120-second timeout, for every instrument. The scheduled run would hang for
most of an hour before failing.

Prints what it found and exits 0 when the gateway is usable, 1 when it is
not, so a scheduled job can stop cleanly instead of failing halfway through.
"""

import socket
import sys
from datetime import datetime

from ib_async import IB, Stock

from config import IB_HOST, IB_PORT

CLIENT_ID = 99      # kept clear of orders.py (32) and the probe (41)
TIMEOUT = 4

# the farm probe. One month of daily bars on the most liquid instrument
# there is, so a failure means the farm and not the contract.
PROBE_SYMBOL = 'SPY'
PROBE_DURATION = '1 M'
PROBE_TIMEOUT = 20


def port_open(host, port, timeout=TIMEOUT):
    """True if something is listening, without speaking the API."""
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


def farm_alive(ib, symbol=PROBE_SYMBOL):
    """
    One short historical request. Returns the bar count, or None.

    None covers every way this fails - a timeout, an error from the farm, an
    empty answer - because the caller does the same thing in each case.
    """
    contract = Stock(symbol, 'SMART', 'USD')
    try:
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=PROBE_DURATION,
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1,
                timeout=PROBE_TIMEOUT)
    except Exception as exc:
        print(f"  historical request raised {type(exc).__name__}: {exc}")
        return None
    return len(bars) if bars else None


def main():
    print(f"{datetime.now().isoformat(timespec='seconds')}  "
          f"{IB_HOST}:{IB_PORT}")

    if not port_open(IB_HOST, IB_PORT):
        print("  nothing listening - the gateway is not running")
        sys.exit(1)
    print("  port open")

    ib = IB()
    try:
        ib.connect(IB_HOST, IB_PORT, clientId=CLIENT_ID, timeout=TIMEOUT)
    except Exception as exc:
        print(f"  port open but the api refused the handshake: {exc}")
        sys.exit(1)

    try:
        accounts = ib.managedAccounts()
        print(f"  connected, server version {ib.client.serverVersion()}")
        print(f"  accounts   {', '.join(accounts) if accounts else '(none)'}")
        print(f"  broker time {ib.reqCurrentTime()}")

        bars = farm_alive(ib)
        if bars is None:
            print(f"  historical farm served nothing for {PROBE_SYMBOL} - "
                  "the gateway is up but cannot supply data")
            sys.exit(1)
        print(f"  historical farm ok, {bars} bars of {PROBE_SYMBOL}")
    finally:
        ib.disconnect()

    sys.exit(0)


if __name__ == '__main__':
    main()
