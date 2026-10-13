"""
Paged daily-bar downloads from IB Gateway.

One request will not reach back far enough, so the window is walked backwards until it meets the contract's head timestamp, stops moving, or uses up MAX_CHUNKS requests.

An empty response is ambiguous - the end of the history, a timeout, or a pacing violation - so it is
requested once more before being accepted. `complete` is then decided from how far back the bars reach, not from which condition stopped the walk.

Callers build their own contracts.
"""

from datetime import datetime, timezone
import pandas as pd

# longest duration IB Gateway has accepted in testing
CHUNK = '25 Y'

# loop guard, not a span limit - 4 x 25 Y exceeds any available history
MAX_CHUNKS = 4

# IBKR allows 60 historical requests per 10 minutes
PAUSE = 1.0
RETRY_PAUSE = 10.0

# the head timestamp and the first bar actually served can differ by a few sessions. This separates that gap from a gateway truncating by years, so the exact value does not matter.
HEAD_SLACK_DAYS = 30


def _as_date(value):
    """A bar date or head timestamp as a plain date, whatever it arrives as."""
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert('UTC').tz_localize(None)
    return stamp.date()


def head_date(ib, contract, what, use_rth=True, label=None):
    """Earliest date IBKR has for this contract, or None if it will not say."""
    name = label or getattr(contract, 'symbol', '?')
    try:
        stamp = ib.reqHeadTimeStamp(contract, whatToShow=what, useRTH=use_rth, formatDate=2)

    except Exception as exc:
        print(f"  {name} {what}: no head timestamp ({exc})")
        return None
    if stamp is None:
        print(f"  {name} {what}: no head timestamp")
        return None
    return _as_date(stamp)


def _request(ib, contract, what, use_rth, chunk, end, timeout):
    return ib.reqHistoricalData(
            contract,
            endDateTime=end,
            durationStr=chunk,
            barSizeSetting='1 day',
            whatToShow=what,
            useRTH=use_rth,
            formatDate=1,
            timeout=timeout,
            )


def daily(ib, contract, what, use_rth=True, chunk=CHUNK, max_chunks=MAX_CHUNKS, pause=PAUSE, timeout=120, label=None):
    """
    Daily bars, oldest first, as (DataFrame, complete).
    'complete' says whether the frame reaches the start of the available history: its oldest bar is within HEAD_SLACK_DAYS of the head timestamp, or there is no head timestamp and the window stopped moving. Anything else leaves the frame possibly truncated.

    Bar dates are left exactly as ib_async returns them, so a caller writing a CSV keeps the format it already has.
    """
    from ib_async import util
    
    name = label or getattr(contract, 'symbol', '?')
    head = head_date(ib, contract, what, use_rth, name)

    frames, end, previous_first, stalled = [], '', None, False

    for _ in range(max_chunks):
        bars = _request(ib, contract, what, use_rth, chunk, end, timeout)
        if not bars:
            # empty is ambiguous. Pause and ask once more before believing it.
            ib.sleep(RETRY_PAUSE)
            bars = _request(ib, contract, what, use_rth, chunk, end, timeout)
        if not bars:
            print(f"  {name} {what}: empty twice ending {end or 'now'}, "
                  f"stopping")
            break

        frames.append(util.df(bars))
        first = _as_date(bars[0].date)

        if head is not None and first <= head:
            break       # reached the contract's first bar
        if previous_first is not None and first >= previous_first:
            print(f"  {name} {what}: pagination stalled at {first}, stopping")
            stalled = True
            break
        previous_first = first

        # next request ends at the start of the earliest day received
        end = datetime(first.year, first.month, first.day, tzinfo=timezone.utc)
        ib.sleep(pause)
    else:
        print(f"  {name} {what}: {max_chunks} chunks of {chunk} used up "
              f"without reaching the start")

    if not frames:
        raise RuntimeError(f"no {what} bars returned for {name}")

    out = (pd.concat(frames, ignore_index=True)
           .drop_duplicates(subset='date')
           .sort_values('date')
           .reset_index(drop=True))

    start = _as_date(out['date'].iloc[0])
    reached_head = head is not None and (start - head).days <= HEAD_SLACK_DAYS
    complete = reached_head or (stalled and head is None)
    return out, complete


