"""
The position formula.

Every position in the system - each day of the backtest, today's live target,
and the order script's view of what should be held - comes from the same four
steps:

    daily cash vol target   capital x vol target / sqrt(business days)
    volatility scalar       that, divided by the currency volatility of one
                            share: the shares a full-strength forecast holds
    subsystem position      scaled by the forecast, which is 10 at full
                            strength
    portfolio position      scaled by the instrument's weight and the
                            diversification multiplier

Two adjustments follow and are not part of the formula, because each depends
on something outside the instrument: inertia compares the result to what is
already held, and the leverage cap compares the whole portfolio to capital.

Written once here so the backtest and the live run cannot drift apart.
"""

import numpy as np

from config import BUSINESS_DAYS


def round_half_away(x):
    """
    Round half away from zero. np.round uses banker's rounding, which sends
    0.5 to 0 and 1.5 to 2 - not what is meant by rounding a position.

    Takes a single number or an array.
    """
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def daily_cash_vol_target(capital, vol_target, business_days=BUSINESS_DAYS):
    """The currency amount one day's returns are meant to move the account."""
    return capital * vol_target / np.sqrt(business_days)


def volatility_scalar(daily_target, icv):
    """
    Shares a full-strength forecast would hold.

    icv is the currency volatility of one share. Zero or missing means the
    instrument cannot be sized today.
    """
    return daily_target / icv if icv > 0 else np.nan


def target_position(forecast, icv, daily_target, weight, idm):
    """
    Shares to hold, before rounding, inertia and the leverage cap.

    The forecast is 10 at full strength, so dividing by 10 gives the fraction
    of the full-strength position to take.
    """
    return forecast / 10 * volatility_scalar(daily_target, icv) * weight * idm


def apply_inertia(target, held, band):
    """
    The target, or the position already held when the two are close enough.

    The band is a fraction of the position held, so the same proportional
    move costs the same whatever the size. A flat position has no fraction to
    take, so one share stands in as the denominator.
    """
    if not np.isfinite(target):
        return held
    denom = max(abs(held), 1.0)
    return target if abs(target - held) / denom > band else held


def cap_leverage(positions, prices, capital, max_leverage):
    """
    Every position scaled down together when gross notional is over the cap,
    and the factor applied.

    The limit is on the portfolio, so it can only be applied once every
    instrument has been sized. One factor for all of them keeps the relative
    sizes the forecast asked for.
    """
    gross = sum(abs(positions[t] * prices[t]) for t in positions)
    if gross <= max_leverage * capital:
        return dict(positions), 1.0
    scale = max_leverage * capital / gross
    return ({t: round_half_away(n * scale) for t, n in positions.items()},
            scale)
