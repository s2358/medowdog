"""
Is IB Gateway up?

    python gateway.py

Two checks, because they fail differently. The port answering says some
process is listening; the API handshake completing says that process is a
gateway willing to talk. A gateway in the middle of its daily restart holds
the port open while refusing the handshake, so the first check alone would
report it as healthy.

Prints what it found and exits 0 when the gateway is usable, 1 when it is
not, so a scheduled job can stop cleanly instead of failing halfway through.
"""

import socket
import sys
from datetime import datetime

from ib_async import IB

from config import IB_HOST, IB_PORT

CLIENT_ID = 99      # kept clear of orders.py (32) and the probe (41)
TIMEOUT = 4


def port_open(host, port, timeout=TIMEOUT):
    """True if something is listening, without speaking the API."""
    try:
        with socket.create_connection((host, port), timeout):
            return True
    except OSError:
        return False


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
    finally:
        ib.disconnect()

    sys.exit(0)


if __name__ == '__main__':
    main()
