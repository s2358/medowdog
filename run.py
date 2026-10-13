"""
Run the daily pipeline.

    python run.py

Steps in order: fx rate, download prices, volatility, ewmac, correlation, dividends, position sizing, evaluate. The cost script is deliberately not here - it needs US market hours and runs from its own cron entry.

Everything each step prints goes to the terminal and to logs/run_<timestamp>.log. The run stops at the first failure, so a broken step never leaves a later one working from stale inputs.
"""

import subprocess
import sys
import time
from datetime import datetime

from config import PROJECT_ROOT, DATA_DIR, CORRELATIONS_DIR, WEIGHTS

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

def allocation_csv():
    path = CORRELATIONS_DIR / f"allocation_{'_'.join(TICKERS)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not written by the correlation step")
    return str(path)

# each entry builds its argv lazily - the downloader creates the files the 
# later steps glob for, so the paths cannot be resolved up front
STEPS = [
        ('fx data',        lambda: ['fx_data.py']),
        ('download',       lambda: ['1_ibkr_downloader.py'] + TICKERS),
        ('volatility',     lambda: ['2_volatility.py'] + instrument_csvs()),
        ('ewmac',          lambda: ['3_ewmac_calculator.py'] + instrument_csvs()),
        ('correlation',    lambda: ['4_correlation.py'] + instrument_csvs()),
        ('dividends',      lambda: ['dividends.py']),         
        ('position sizing',lambda: ['6_position_sizing.py', allocation_csv()]
                                    + instrument_csvs()),
        ('evaluate',        lambda: ['7_evaluate.py']),
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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = LOG_DIR / f'run_{stamp}.log'

    results = []
    failed = None

    with open(log_path, 'w') as log:
        emit(f"pipeline run {datetime.now().isoformat(timespec='seconds')}", log)
        emit(f"python    {sys.executable}", log)
        emit(f"tickers   {', '.join(TICKERS)}", log)
        emit(f"log       {log_path}", log)

        for label, build in STEPS:
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
        for label, _ in STEPS:
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
