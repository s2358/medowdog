"""
Fit the EWMAC forecast weights.
    
    python fit_weights.py <csv> [<csv> ...]

Weights come from a handcrafting table: pre-computed allocations for small groups of assets whose correlation pattern matches one of a handful of cases. It is indexed by the three correlations in the order AB, AC, BC and returns the weights for A, B, C.

This measures the correlations between the three rules' forecasts, one instrument at a time, averages those matrices across instruments, rounds each correlation to the nearest value the table is defined at, and looks the weights up. Rounding sends any negative correlation to zero, which is also the floor the method requires.

Every triple of rounded correlations matches either a listed row or a reordering of one, so the lookup always resolves.

Reads the EWMAC_<rule>_scaled columns written by script 3, which are the values the weighted sum actually operates on. Nothing is written back to the config.
"""

import os 
import sys
from datetime import datetime
from itertools import permutations

import numpy as np
import pandas as pd
from config import (FORECAST_SCALARS, FORECAST_WEIGHTS, FDM, FDM_CAP, WARMUP, FITTED_DIR)
from fit_scalars import ticker_from_path, archive_previous, in_window

# the correlation values the handcrafting table is defined at
GRID = (0.0, 0.5, 0.9)

# correlations (AB, AC, BC) -> weights for A, B, C
HANDCRAFTED = {
        (0.0, 0.5, 0.0): (0.30, 0.40, 0.30),
        (0.0, 0.9, 0.0): (0.27, 0.46, 0.27),
        (0.5, 0.0, 0.5): (0.37, 0.26, 0.37),
        (0.0, 0.5, 0.9): (0.45, 0.45, 0.10),
        (0.9, 0.0, 0.9): (0.39, 0.22, 0.39),
        (0.5, 0.9, 0.5): (0.29, 0.42, 0.29),
        (0.9, 0.5, 0.9): (0.42, 0.16, 0.42),
        }

def rule_names():
    """The rules fastest first, so the printed order is the span order."""
    return sorted(FORECAST_SCALARS, key=lambda name: int(name.split('_')[1]))


def snap(value):
    """The nearest correlation the table is defined at."""
    return min(GRID, key=lambda point: abs(value - point))


def margin(value):
    """How far a correlation sits from the nearest rounding boundary."""
    edges = [(GRID[i] + GRID[i+1]) / 2 for i in range(len(GRID) - 1)]
    return min(abs(value - edge) for edge in edges)


def correlations(path, names):
    """Correlation matrix of one instrument's scaled forecasts."""
    columns = [f'EWMAC_{rule}_scaled' for rule in names]
    frame = pd.read_csv(path, usecols=['date'] + columns, 
                        parse_dates=['date']).iloc[WARMUP:].dropna()
    frame = in_window(frame)
    matrix = frame[columns].corr()      # usecols does not preserve the order asked for
    matrix.index = names
    matrix.columns = names
    return matrix, len(frame)


def average_correlations(paths):
    """Each instrument's matrix, and the elementwise average of them."""
    names = rule_names()
    matrices, counts = {}, {}
    for path in paths:
        ticker = ticker_from_path(path)
        if ticker in counts:
            raise ValueError(f"same ticker given more than once: {ticker}")
        matrices[ticker], counts[ticker] = correlations(path, names)
    mean = sum(matrices.values()) / len(matrices)
    return names, matrices, mean, counts


def handcraft(matrix, names):
    """
    Weights per rule, the rounded correlations used, and the order they were read in. The table
    lists one ordering of each pattern, so the values are tried in every order until a listed triple is found.
    """
    if len(names) != 3:
        raise ValueError(f"the table covers groups of three, got {len(names)}")

    a, b, c = names
    measured = (snap(matrix.loc[a, b]), snap(matrix.loc[a, c]),
                snap(matrix.loc[b,c]))
    if measured[0] == measured[1] == measured[2]:
        # identical correlations: nothing distinguishes the rules
        equal = 1.0 / len(names)
        return {name: equal for name in names}, measured, tuple(names)

    for order in permutations(names):
        x, y, z = order
        key = (snap(matrix.loc[x, y]), snap(matrix.loc[x, z]),
               snap(matrix.loc[y, z]))
        if key in HANDCRAFTED:
            return dict(zip(order, HANDCRAFTED[key])), key, order

    raise LookupError(f"no handcrafting row for correlations {measured}")


def diversification_multiplier(matrix, weights, cap=FDM_CAP):
    """
    1 / sqrt(W x H x W'), capped.

    Negative correlations are floored at zero first. Two rules moving in opposite directions would 
    otherwise inflate the multiplier on the least stable relationship in the data.
    """

    missing = set(matrix.columns) - set(weights)
    if missing:
        raise KeyError(f"no weight supplied for: {sorted(missing)}")
    W = np.array([weights[name] for name in matrix.columns])
    if not np.isclose(W.sum(), 1.0):
        raise ValueError(f"weights sum to {W.sum():.4f}, not 1")
    H = matrix.clip(lower = 0.0).values
    return min(float(1.0 / np.sqrt(W @ H @ W)), cap)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <csv> [<csv> ...]")
        sys.exit(1)

    names, matrices, mean, counts = average_correlations(sys.argv[1:])
    weights, key, order = handcraft(mean, names)
    fdm = diversification_multiplier(mean, weights)

    for ticker, matrix in matrices.items():
        print(f"\n{ticker}  ({counts[ticker]} rows)")
        print(matrix.round(3).to_string())

    print("\naveraged across instruments")
    print(mean.round(3).to_string())

    print("\nrounded")
    pairs = ((order[0], order[1]), (order[0], order[2]), (order[1], order[2]))
    tight = False
    for (left, right), rounded in zip(pairs, key):
        measured = mean.loc[left, right]
        close = margin(measured) < 0.05
        tight = tight or close
        flag = '  <- close to the boundary' if close else ''
        print(f"  {left:7s} {right:7s} {measured:6.3f} -> {rounded:.1f}{flag}")
    if tight:
        print(" a correlation near a boundary can switch the row, and with it"
              "\n the weights - worth knowing before adopting the result")

    table = pd.DataFrame({
        'fitted': pd.Series({**weights, 'fdm': fdm}),
        'current': pd.Series({**FORECAST_WEIGHTS, 'fdm': FDM}),
        }).loc[names + ['fdm']]
    table.index.name = 'parameter'
    print("\nweights and multiplier")
    print(table.round(3).to_string())

    FITTED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    archive_previous(FITTED_DIR, 'forecast_weights_*.csv')
    weights_path = FITTED_DIR / f'forecast_weights_{stamp}.csv'
    table.to_csv(weights_path)

    archive_previous(FITTED_DIR, 'forecast_correlations_*.csv')
    correlations_path = FITTED_DIR / f'forecast_correlations_{stamp}.csv'
    mean.to_csv(correlations_path)

    print(f"\nwritten to {weights_path}\n            {correlations_path}")


if __name__ == '__main__':
    main()

