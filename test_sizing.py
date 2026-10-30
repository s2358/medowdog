"""
Tests for the position formula.

    python test_sizing.py

Known inputs and hand-computed outputs, run at a multiplier of 1 and again at
1000. Every instrument traded today has a multiplier of 1, which makes a
missing multiplication and a correct one indistinguishable in any output the
system produces, so these cases are the only place the difference shows.

Prints one line per case and exits non-zero if any fails.
"""

import sys
import importlib
import pandas as pd
import numpy as np
import sizing
import universe
from config import FORECAST_SCALARS
ewmac = importlib.import_module('3_ewmac_calculator')

CAPITAL = 150_000.0
VOL_TARGET = 0.20

# a share at 5.00 of daily move, and a contract whose price moves 0.05 a day
# with 1000 units behind it - deliberately the same currency risk per unit of
# position after the multiplier, so the two columns can be compared directly
SHARE_VOL, SHARE_MULT = 5.0, 1
FUT_VOL, FUT_MULT = 0.05, 1000

WEIGHT, IDM = 0.4, 1.4

results = []


def same(got, want, tol=1e-9):
    """Equal within tolerance, with nan equal to nan."""
    got, want = float(got), float(want)
    if np.isnan(want):
        return np.isnan(got)
    return abs(got - want) <= tol


def check(name, got, want):
    ok = same(got, want)
    results.append(ok)
    mark = 'ok  ' if ok else 'FAIL'
    print(f"  {mark}  {name:<46}  got {got!r:>22}  want {want!r}")


def check_dict(name, got, want):
    ok = (set(got) == set(want)
          and all(same(got[k], want[k]) for k in want))
    results.append(ok)
    mark = 'ok  ' if ok else 'FAIL'
    print(f"  {mark}  {name:<46}  got {got}  want {want}")


def test_daily_cash_vol_target():
    print("\ndaily cash vol target")
    # 150,000 x 0.20 / sqrt(256) = 30,000 / 16
    check('capital x target / sqrt(256)',
          sizing.daily_cash_vol_target(CAPITAL, VOL_TARGET), 1875.0)
    check('zero capital', sizing.daily_cash_vol_target(0.0, VOL_TARGET), 0.0)


def test_volatility_scalar():
    print("\nvolatility scalar")
    # icv is currency per unit of position, so the scalar is a position count
    check('share, icv 5.00',
          sizing.volatility_scalar(1875.0, SHARE_VOL * SHARE_MULT), 375.0)
    check('contract, icv 50.00',
          sizing.volatility_scalar(1875.0, FUT_VOL * FUT_MULT), 37.5)
    check('icv zero is not sizeable',
          sizing.volatility_scalar(1875.0, 0.0), np.nan)
    check('icv negative is not sizeable',
          sizing.volatility_scalar(1875.0, -5.0), np.nan)
    check('icv missing is not sizeable',
          sizing.volatility_scalar(1875.0, np.nan), np.nan)


def test_target_position():
    print("\ntarget position")
    target = 1875.0
    # 375 x 0.4 x 1.4, at a full-strength forecast
    check('share, forecast 10',
          sizing.target_position(10.0, SHARE_VOL * SHARE_MULT, target,
                                 WEIGHT, IDM), 210.0)
    # 37.5 x 0.4 x 1.4 - the multiplier divides the position, because the
    # same currency risk is reached with fewer contracts
    check('contract, forecast 10',
          sizing.target_position(10.0, FUT_VOL * FUT_MULT, target,
                                 WEIGHT, IDM), 21.0)
    check('contract, forecast 6',
          sizing.target_position(6.0, FUT_VOL * FUT_MULT, target,
                                 WEIGHT, IDM), 12.6)
    check('contract, forecast -10 is the mirror',
          sizing.target_position(-10.0, FUT_VOL * FUT_MULT, target,
                                 WEIGHT, IDM), -21.0)
    check('forecast 0 holds nothing',
          sizing.target_position(0.0, FUT_VOL * FUT_MULT, target,
                                 WEIGHT, IDM), 0.0)
    check('unsizeable icv gives no position',
          sizing.target_position(10.0, 0.0, target, WEIGHT, IDM), np.nan)


def test_risk_is_multiplier_free():
    """
    The test that catches a multiplier applied twice or not at all.

    Position x icv is the currency risk the position carries. It must come to
    the same number whatever the contract size, because the contract size is
    not a risk preference - it is a unit of account. A multiplier counted
    twice makes this 1000x small, one missing makes it 1000x large, and
    nothing else in the system reports it.
    """
    print("\ncurrency risk per position, which the multiplier must not change")
    target = 1875.0
    want = 10.0 / 10 * WEIGHT * IDM * target        # 1050.0

    for label, vol, mult in (('share   ', SHARE_VOL, SHARE_MULT),
                             ('contract', FUT_VOL, FUT_MULT)):
        icv = vol * mult
        position = sizing.target_position(10.0, icv, target, WEIGHT, IDM)
        check(f'{label} position x icv', position * icv, want)


def test_round_half_away():
    print("\nrounding")
    # np.round would give 0, 2, 0, -2 on these
    check('0.5 up', sizing.round_half_away(0.5), 1.0)
    check('1.5 up', sizing.round_half_away(1.5), 2.0)
    check('-0.5 away from zero', sizing.round_half_away(-0.5), -1.0)
    check('-2.5 away from zero', sizing.round_half_away(-2.5), -3.0)
    check('0.49 down', sizing.round_half_away(0.49), 0.0)
    check('a whole contract is unchanged', sizing.round_half_away(21.0), 21.0)


def test_apply_inertia():
    print("\ninertia")
    band = 0.10
    check('inside the band, keep what is held',
          sizing.apply_inertia(105.0, 100.0, band), 100.0)
    check('on the band, keep what is held',
          sizing.apply_inertia(110.0, 100.0, band), 100.0)
    check('outside the band, take the target',
          sizing.apply_inertia(120.0, 100.0, band), 120.0)
    check('short side is symmetric',
          sizing.apply_inertia(-120.0, -100.0, band), -120.0)
    check('from flat, one contract is a full move',
          sizing.apply_inertia(1.0, 0.0, band), 1.0)
    check('a missing target holds the position',
          sizing.apply_inertia(np.nan, 100.0, band), 100.0)
    check('crossing zero is outside any band',
          sizing.apply_inertia(-5.0, 5.0, band), -5.0)


def test_cap_leverage():
    print("\nleverage cap")
    # prices here are per unit of position and already carry the multiplier,
    # which is what script 6 passes in
    prices = {'A': 1000.0, 'B': 500.0}

    # 10 x 1000 + 20 x 500 = 20,000 gross against a 10,000 limit
    capped, scale = sizing.cap_leverage({'A': 10.0, 'B': 20.0}, prices,
                                        5_000.0, 2.0)
    check('scale factor', scale, 0.5)
    check_dict('scaled pro rata and re-rounded', capped,
               {'A': 5.0, 'B': 10.0})

    under, scale = sizing.cap_leverage({'A': 10.0, 'B': 20.0}, prices,
                                       20_000.0, 2.0)
    check('under the cap, no scaling', scale, 1.0)
    check_dict('under the cap, positions untouched', under,
               {'A': 10.0, 'B': 20.0})

    # exactly on the limit is not over it
    on, scale = sizing.cap_leverage({'A': 10.0, 'B': 20.0}, prices,
                                    10_000.0, 2.0)
    check('on the cap, no scaling', scale, 1.0)

    shorts, scale = sizing.cap_leverage({'A': -10.0, 'B': 20.0}, prices,
                                        5_000.0, 2.0)
    check('shorts count toward gross', scale, 0.5)
    check_dict('shorts scale toward zero', shorts,
               {'A': -5.0, 'B': 10.0})

    flat, scale = sizing.cap_leverage({'A': 0.0, 'B': 0.0}, prices,
                                      5_000.0, 2.0)
    check('a flat book is never capped', scale, 1.0)


def test_forecast_is_multiplier_free():
    """
    The crossover is a price difference, so standardising it must not depend
    on how many units a contract holds.

    Two instruments are given the identical price series and the identical
    price volatility, and differ only in their multiplier - 1 for SPY, 1000
    for 10Y. The standardised crossover must come out the same for both, and
    only ICV may differ, by exactly the ratio of the multipliers.
    """
    print("\nforecast standardisation, which the multiplier must not change")

    n = 400
    close = 100.0 + np.cumsum(np.sin(np.arange(n) / 17.0)) + np.arange(n) * 0.02
    frame = pd.DataFrame({'close': close, 'ewma_vol': np.full(n, 1.0)})

    share = ewmac.calculate_forecasts(frame.copy(), 'SPY')
    contract = ewmac.calculate_forecasts(frame.copy(), '10Y')

    ratio = universe.multiplier('10Y') / universe.multiplier('SPY')
    check('icv scales with the multiplier',
          contract['icv'].iloc[-1] / share['icv'].iloc[-1], ratio)

    for name in sorted(FORECAST_SCALARS):
        column = f'EWMAC_{name}_vol_std'
        worst = (contract[column] - share[column]).abs().max()
        check(f'{column} identical across multipliers', worst, 0.0)

    worst = (contract['combined_forecast']
             - share['combined_forecast']).abs().max()
    check('combined forecast identical across multipliers', worst, 0.0)



def main():
    for test in (test_daily_cash_vol_target,
                 test_volatility_scalar,
                 test_target_position,
                 test_risk_is_multiplier_free,
                 test_forecast_is_multiplier_free,
                 test_round_half_away,
                 test_apply_inertia,
                 test_cap_leverage):
        test()

    failed = results.count(False)
    print(f"\n{len(results) - failed} passed, {failed} failed "
          f"of {len(results)}")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
