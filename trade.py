"""
Run the trading sequence.

    python trade.py              # dry run, sends nothing
    python trade.py --place      # transmits

Steps in order: capital, reconcile, orders, fills. Everything each step prints
goes to the terminal and to logs/trade_<timestamp>.log, and the sequence stops
at the first failure.

The order is the point. Capital leads so the staleness limit on the account
value can never be the thing that stops a trade. Reconcile then refuses to
continue unless every share held is explained by a recorded execution, so
orders are never sized against a book whose history has a hole in it. Fills
runs last and records what the orders got.

Only the order step is told about --place. The others never write anything to
the broker.

An order still working when fills runs is not yet in the record. Run fills.py
again later to pick it up.
"""

import sys
from datetime import datetime

from run import emit, run_step, RULE, LOG_DIR


def steps(place):
    return [
            ('capital',   ['capital.py']),
            ('reconcile', ['reconcile.py']),
            ('orders',    ['orders.py'] + (['--place'] if place else [])),
            ('fills',     ['fills.py']),
            ]


def main():
    place = '--place' in sys.argv[1:]
    sequence = steps(place)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = LOG_DIR / f'trade_{stamp}.log'

    results = []
    failed = None

    with open(log_path, 'w') as log:
        emit(f"trade run {datetime.now().isoformat(timespec='seconds')}", log)
        emit(f"python    {sys.executable}", log)
        emit(f"mode      {'PLACING ORDERS' if place else 'DRY RUN'}", log)
        emit(f"log       {log_path}", log)

        for label, argv in sequence:
            code, seconds = run_step(label, argv, log)
            results.append((label, 'ok' if code == 0 else f'exit {code}',
                            seconds))
            if code != 0:
                failed = label
                break

        emit(f"\n{RULE}", log)
        emit("summary", log)
        emit(RULE, log)

        done = {r[0] for r in results}
        for label, status, seconds in results:
            emit(f"  {label:18s}  {status:10s}  {seconds:6.1f}s", log)
        for label, _ in sequence:
            if label not in done:
                emit(f"  {label:18s}  {'skipped':10s}", log)

        total = sum(r[2] for r in results)
        if failed:
            emit(f"\nFAILED at '{failed}' after {total:.1f}s", log)
        else:
            emit(f"\nall steps ok in {total:.1f}s", log)
        emit(f"log written to {log_path}", log)

    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
