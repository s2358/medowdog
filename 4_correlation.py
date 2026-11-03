"""
Subsystem-return correlations and the IDM.

    python 4_correlation.py <csv> <csv> [<csv> ...]
    python 4_correlation.py <csv> ... --weights SPY=20,GLDM=10,SLV=10
    python 4_correlation.py <csv> ... --weights equal

Writes three files to data/correlations/:
    subsystem_returns_<tickers>.csv     the panel
    correlation_matrix_<tickers>.csv    the negatives floored at zero
    allocation_<tickers>.csv            weights + IDM, a record of the run

This script measures. It does not set anything. The live path and the backtest
both read the IDM from config.py, the same way they read the FDM and the
forecast scalars. To adopt a new value, edit config.py by hand. Nothing reads
the allocation file; it is kept as the record of what was measured and when.

Instruments are joined on every date any of them traded rather than only the dates all of them did, so one short history does not shorten the rest. Each correlation is then measured on its own pair's overlap.

The matrix covers every file given. The multiplier covers only the instruments that carry a weight, so a candidate can be measured against the book before it is admitted to it.

--weights answers the other half of that question. A candidate has no weight in
universe.py until it is adopted, so the multiplier cannot show what adopting it
would do. Given a weighting on the command line, the run uses that instead and
writes its record under allocation_whatif_, leaving universe.py untouched.
Numbers may be percentages or fractions and are normalised to sum to one,
because that is the only form the multiplier is defined for.
"""

import pandas as pd
import numpy as np
import os
import sys
from config import (IDM_CAP, WARMUP, BUSINESS_DAYS, CORRELATIONS_DIR,
                    WINDOW_START, WINDOW_END, IDM as IDM_IN_CONFIG)
from universe import WEIGHTS, ticker_from_path


def parse_args(argv):
    """Paths, and an optional weighting to test instead of the universe's."""
    paths, spec = [], None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--weights':
            i += 1
            if i >= len(argv):
                raise ValueError("--weights needs a value")
            spec = argv[i]
        elif arg.startswith('--weights='):
            spec = arg.split('=', 1)[1]
        else:
            paths.append(arg)
        i += 1
    return paths, spec


def parse_weights(spec, tickers):
    """
    TICKER=NUMBER pairs, or the word 'equal'.

    Whatever is given is normalised to sum to one. Carver: the weights sum to
    one over what you hold, and where they do not, you divide until they do.
    So 20,10,10,20,20,20 and 0.2,0.1,0.1,0.2,0.2,0.2 mean the same thing.

    A weight of zero is allowed and removes the instrument from the
    multiplier while leaving it in the matrix, which is how a candidate is
    looked at without being counted.
    """
    if spec.strip().lower() == 'equal':
        share = 1.0 / len(tickers)
        return {t: share for t in tickers}

    given = {}
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(f"expected TICKER=NUMBER, got {part!r}")
        name, number = part.split('=', 1)
        given[name.strip().upper()] = float(number)

    unknown = set(given) - set(tickers)
    if unknown:
        raise ValueError(f"not among the files given: {sorted(unknown)}")
    missing = set(tickers) - set(given)
    if missing:
        raise ValueError(f"no weight given for: {sorted(missing)}")
    if any(w < 0 for w in given.values()):
        raise ValueError("a negative instrument weight has no meaning here")

    total = sum(given.values())
    if total <= 0:
        raise ValueError("the weights sum to zero")
    return {t: given[t] / total for t in tickers}


def load_subsystem_returns(path):
    df = pd.read_csv(path, usecols=['date', 'subsystem_returns'])
    df = df.iloc[WARMUP:]
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')['subsystem_returns'].rename(ticker_from_path(path))


def build_panel(paths):
    """
    The panel of subsystem returns, restricted to the backtest window.

    The window matters here: the multiplier is an estimate from the data, so
    an out-of-sample run must not compute it from rows the run is pretending
    not to have seen.

    Gaps are left empty rather than dropped. A row where one instrument did
    not yet exist is still a real row for the others, and discarding it would
    throw away the history the longer instruments have.
    """
    series = [load_subsystem_returns(p) for p in paths]
    names = [s.name for s in series]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"same ticker given more than once: {sorted(duplicates)}")
    panel = pd.concat(series, axis=1, join='outer')

    if WINDOW_START is not None:
        panel = panel[panel.index >= pd.Timestamp(WINDOW_START)]
    if WINDOW_END is not None:
        panel = panel[panel.index < pd.Timestamp(WINDOW_END)]
    if panel.empty:
        raise ValueError("the configured window leaves no rows")
    return panel


def pair_rows(panel, a, b):
    """Rows on which both instruments have a return."""
    return int((panel[a].notna() & panel[b].notna()).sum())


def pairwise_correlations(panel, min_rows=BUSINESS_DAYS):
    """
    Each correlation measured on its own pair's overlap.

    A pair with less than a year of shared history is not estimated at all.
    The number would be noise, and it would go straight into the IDM, where
    nothing downstream could tell it from a measurement.
    """
    raw = panel.corr(min_periods=min_rows)
    thin = [f"{a}/{b} ({pair_rows(panel, a, b)} rows)"
            for i, a in enumerate(raw.columns) for b in raw.columns[i + 1:]
            if np.isnan(raw.loc[a, b])]
    if thin:
        raise ValueError(f"under {min_rows} rows of overlap: {', '.join(thin)}")
    return raw


def correlation_matrix(raw, floor_negatives=True):
    return raw.clip(lower=0) if floor_negatives else raw


def correlation_standard_error(r, n):
    """
    SE of a sample correlation, (1 - r^2) / sqrt(n-1).
    The familiar 1/sqrt(n) is only the r=0 case and overstates the uncertainty badly once r is large.
    """
    return (1 - r ** 2) / np.sqrt(n - 1)

def instrument_diversification_multiplier(H, weights, cap=IDM_CAP):
    """
    IDM = 1 / sqrt(W x H x W'), capped.
    H must already have negative correlations floored at zero.
    """
    missing = set(H.columns) - set(weights)
    if missing:
        raise KeyError(f"no weight supplied for: {sorted(missing)}")
    W = np.array([weights[c] for c in H.columns])
    if not np.isclose(W.sum(), 1.0):
        raise ValueError(f"weights sum to {W.sum():.4f}, expected 1.0")

    v = W @ H.values @ W.T
    return min(1 / np.sqrt(v), cap), v

def main():
    try:
        paths, spec = parse_args(sys.argv[1:])
    except ValueError as exc:
        print(exc)
        sys.exit(1)

    if len(paths) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} "
              "<csv1> <csv2> [<csv3> ...] "
              "[--weights TICKER=N,TICKER=N,... | --weights equal]")
        sys.exit(1)

    panel = build_panel(paths)
    n = len(panel)

    raw = pairwise_correlations(panel)
    H = correlation_matrix(raw)

    # the matrix covers every file given, so a candidate can be measured
    # against the book before it is given a weight. The multiplier is only
    # defined over what is actually traded - or, with --weights, over what a
    # proposed book would hold.
    what_if = spec is not None
    if what_if:
        try:
            weights = parse_weights(spec, list(H.columns))
        except ValueError as exc:
            print(f"--weights: {exc}")
            sys.exit(1)
        traded = [c for c in H.columns if weights[c] > 0]
        extra = [c for c in H.columns if weights[c] == 0]
    else:
        weights = WEIGHTS
        traded = [c for c in H.columns if c in WEIGHTS]
        extra = [c for c in H.columns if c not in WEIGHTS]

    if not traded:
        print("no instrument carries a weight - nothing to measure")
        sys.exit(1)

    idm, v = instrument_diversification_multiplier(
            H.loc[traded, traded], weights)

    allocation = pd.DataFrame({
        'ticker': traded,
        'instrument_weight': [weights[c] for c in traded],
        'idm': idm,
        })

    print(f"window: {panel.index[0].date()} -> {panel.index[-1].date()}")
    print(f"rows: {n} ({n /BUSINESS_DAYS:.1f} years)\n")

    print("rows and annualised vol per subsystem:")
    for col in panel.columns:
        rows = int(panel[col].notna().sum())
        vol = panel[col].std() * np.sqrt(BUSINESS_DAYS)
        print(f"  {col:6s} {rows:6d} rows   {vol:.1%}   "
              f"from {panel[col].first_valid_index().date()}")
    print()

    print("correlation matrix (raw):")
    print(raw.round(3).to_string(), "\n")

    if not raw.equals(H):
        print("correlation matrix (negatives floored at zero):")
        print(H.round(3).to_string(), "\n")

    print("pairwise correlation and standard error:")
    cols = list(raw.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = raw.loc[a, b]
            shared = pair_rows(panel, a, b)
            se = correlation_standard_error(r, shared)
            print(f"  {a:6s} {b:6s}  r  {r:+.3f}  se  {se:.3f}  "
                  f"95% ci [{r - 1.96 * se:+.3f}, {r + 1.96 * se:+.3f}]  "
                  f"on {shared} shared rows")

    source = "proposed on the command line" if what_if else "from universe.py"
    print(f"\ninstrument weights ({source}):")
    for c in traded:
        print(f" {c:6s} {weights[c]:.1%}")
    if extra:
        label = "at zero weight" if what_if else "not weighted"
        print(f" {label}: {', '.join(extra)}")

    print(f"W x H x W' = {v:.5f}")
    print(f"IDM        = {idm:.4f}" + ("  (capped)" if idm == IDM_CAP else ""))

    change = idm - IDM_IN_CONFIG
    print(f"config.py holds IDM = {IDM_IN_CONFIG:.4f}, "
          f"a difference of {change:+.4f}")

    if what_if:
        print("this is a what-if. universe.py and config.py are untouched, "
              "and nothing has been adopted.")
        if abs(change) > 1e-4:
            print(f"adopting this book would scale every position by "
                  f"{idm / IDM_IN_CONFIG:.2f}x.")
    elif abs(change) > 1e-4:
        print("nothing uses the measured value. To adopt it, edit config.py.")

    CORRELATIONS_DIR.mkdir(parents=True, exist_ok=True)
    stem = '_'.join(sorted(panel.columns))
    traded_stem = '_'.join(sorted(traded))

    out = CORRELATIONS_DIR / f'subsystem_returns_{stem}.csv'
    panel.to_csv(out)
    print(f"\npanel written to {out}")

    corr_out = CORRELATIONS_DIR / f'correlation_matrix_{stem}.csv'
    H.to_csv(corr_out)
    print(f"correlation matrix written to {corr_out}")

    prefix = 'allocation_whatif' if what_if else 'allocation'
    alloc_out = CORRELATIONS_DIR / f'{prefix}_{traded_stem}.csv'
    allocation.to_csv(alloc_out, index=False)
    print(f"allocation written to {alloc_out}")

if __name__ == '__main__':
    main()
