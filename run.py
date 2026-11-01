"""
Run the daily pipeline.

    python run.py               # the trading inputs
    python run.py --research    # and the correlations, backtest and evaluation

The daily run builds today's targets. Acting on them is trade.py.

The research steps are separate on purpose. The correlations and the
diversification multiplier are estimates, not observations. If they run every
night, every position moves for reasons that have nothing to do with the
forecast. The backtest reads nothing that the live path uses, and its starting
capital is today's account value, so a daily run rescales nineteen years of
history every morning.

The cost script is not here either. It needs US market hours and runs from its
own cron entry.

Each step prints to the terminal and to logs/run_<timestamp>.log. The run stops
at the first failure, so a broken step never leaves a later one working from
stale inputs.
"""

import subprocess
import sys
import time
from datetime import datetime

from config import PROJECT_ROOT, DATA_DIR, FEATURES_DIR
from universe import WEIGHTS

TICKERS = sorted(WEIGHTS)
LOG_DIR = PROJECT_ROOT / 'logs'
RULE = '=' * 72

def emit(text, log):
    """Write a line to the terminal and the log file."""
    print(text, flush=True)
    log.write(text + '\n')
    log.flush()

def one_csv(ticker):
    """
    The single live CSV for a ticker.

    The downloader archives previous downloads, so exactly one file should match. Two means an archive step was skipped, and the scripts that take sys.argv[1:] would then process the stale file too.
    """
    folder = DATA_DIR / ticker
    matches = sorted(folder.glob('*.csv'))
    if len(matches) != 1:
        names = [m.name for m in matches] or ['(none)']
        raise FileNotFoundError(
                f"{folder} holds {len(matches)} csv files, expected exactly 1: "
                f"{', '.join(names)}"
                )
    return str(matches[0])

def instrument_csvs():
    return [one_csv(t) for t in TICKERS]


def features_csvs():
    paths = [FEATURES_DIR / f'{t}.csv' for t in TICKERS]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
                f"not written by the volatility step: {', '.join(missing)}")
    return [str(p) for p in paths]


# what today's position depends on. Each changes daily, and the target
# cannot be computed without it.
DAILY = [
        ('fx data',        lambda: ['fx_data.py']),
        ('download',       lambda: ['1_ibkr_downloader.py'] + TICKERS),
        ('volatility',     lambda: ['2_volatility.py'] + instrument_csvs()),
        ('ewmac',          lambda: ['3_ewmac_calculator.py'] + features_csvs()),
        ('dividends',      lambda: ['dividends.py']),
        ('capital',        lambda: ['capital.py']),
        ('equity',         lambda: ['equity.py']),
        ('targets',        lambda: ['targets.py'] + features_csvs()),
        ]

# research, run deliberately rather than nightly
RESEARCH = [
        ('correlation',    lambda: ['4_correlation.py'] + features_csvs()),
        ('position sizing',lambda: ['6_position_sizing.py'] + features_csvs()),
        ('evaluate',       lambda: ['7_evaluate.py']),
        ]


def run_step(label, argv, log):
    """Run one step, streaming its output. Returns (returncode, seconds)."""
    emit(f"\n{RULE}", log)
    emit(f"{label}", log)
    emit(f"  {' '.join(argv)}", log)
    emit(RULE, log)

    start = time.time()
    proc = subprocess.Popen(
            [sys.executable] + argv,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            )
    for line in proc.stdout:
        emit(line.rstrip('\n'), log)
    proc.wait()
    return proc.returncode, time.time() - start

def main():
    research = '--research' in sys.argv[1:]
    sequence = DAILY + RESEARCH if research else DAILY

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = LOG_DIR / f'run_{stamp}.log'

    results = []
    failed = None

    with open(log_path, 'w') as log:
        emit(f"pipeline run {datetime.now().isoformat(timespec='seconds')}", log)
        emit(f"python    {sys.executable}", log)
        emit(f"mode      {'DAILY + RESEARCH' if research else 'DAILY'}", log)
        emit(f"tickers   {', '.join(TICKERS)}", log)
        emit(f"log       {log_path}", log)

        for label, build in sequence:
            try:
                argv = build()
            except Exception as exc:
                emit(f"\n{RULE}", log)
                emit(f"{label}", log)
                emit(RULE, log)
                emit(f"could not build arguments: {exc}", log)
                results.append((label, 'SETUP FAIL', 0.0))
                failed = label
                break

            code, seconds = run_step(label, argv, log)
            results.append((label, 'ok' if code == 0 else f'exit {code}', seconds))

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
