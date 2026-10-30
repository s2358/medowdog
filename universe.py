"""
The traded universe: one record per instrument.

Everything that differs between instruments lives here - what the broker
calls it, what one unit is worth, how small a price move can be, and what a
trade costs. Everything in config.py applies to the system as a whole.

An instrument with zero weight is described but not traded: it stays listed
so its contract and multiplier are ready before it is brought in.

Weights are handcrafted and must sum to 1 across the traded instruments.
"""

import math
from pathlib import Path


STOCK_COMMISSION = {'per_share': 0.005, 'minimum': 1.00, 'max_pct': 0.01}

# per-contract fees are commission plus exchange and regulatory charges, and
# differ by contract - check each against a real fill before trusting it.
# Nothing carries this today; it is kept as the record of what a micro
# contract costs, from when futures were costed.
MICRO_COMMISSION = {'per_contract': 0.62}

INSTRUMENTS = {
    'SPY': {
        'weight': 0.40,
        'type': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'multiplier': 1,
        'tick': 0.01,
        'commission': STOCK_COMMISSION,
    },
    'GLDM': {
        'weight': 0.30,
        'type': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'multiplier': 1,
        'tick': 0.01,
        'commission': STOCK_COMMISSION,
    },
    'SLV': {
        'weight': 0.30,
        'type': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'multiplier': 1,
        'tick': 0.01,
        'commission': STOCK_COMMISSION,
    },

    # candidates, described but not traded. Each covers a sector the book has
    # no exposure to: rates, currency, energy. TLT was the obvious rates
    # choice and is not here - IBKR serves it only from 2016-02-03, when its
    # listing moved to NASDAQ, where EDV goes back to 2007.
    'EDV': {
        'weight': 0.0,
        'type': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'multiplier': 1,
        'tick': 0.01,
        'commission': STOCK_COMMISSION,
    },
    'FXY': {
        'weight': 0.0,
        'type': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'multiplier': 1,
        'tick': 0.01,
        'commission': STOCK_COMMISSION,
    },
    'USO': {
        'weight': 0.0,
        'type': 'STK',
        'exchange': 'SMART',
        'currency': 'USD',
        'multiplier': 1,
        'tick': 0.01,
        'commission': STOCK_COMMISSION,
    },
}

REQUIRED = ('weight', 'type', 'exchange', 'currency', 'multiplier', 'tick',
            'commission')
TYPES = ('STK', 'FUT')


def instrument(ticker):
    """The record for one instrument."""
    try:
        return INSTRUMENTS[ticker]
    except KeyError:
        raise KeyError(f"{ticker} is not in the universe") from None


def tickers():
    """The instruments actually traded, in a stable order."""
    return sorted(t for t, e in INSTRUMENTS.items() if e['weight'] > 0)


def weights():
    """Instrument weight by ticker, for the traded instruments."""
    return {t: INSTRUMENTS[t]['weight'] for t in tickers()}


def multiplier(ticker):
    """Units of the underlying in one contract. One, for a share."""
    return instrument(ticker)['multiplier']


def contract(ticker, expiry=None):
    """
    The ib_async contract for an instrument.

    A future with no expiry becomes the continuous contract, which is what a
    price history is built from. Sending an order needs a real expiry.

    ib_async is imported here rather than at the top, so reading the universe
    does not require the broker library.
    """
    from ib_async import Stock, Future, ContFuture

    entry = instrument(ticker)
    if entry['type'] == 'STK':
        return Stock(ticker, entry['exchange'], entry['currency'])
    if expiry is None:
        return ContFuture(ticker, exchange=entry['exchange'],
                          currency=entry['currency'])
    return Future(ticker, lastTradeDateOrContractMonth=expiry,
                  exchange=entry['exchange'], currency=entry['currency'])


def commission(ticker, quantity, price):
    """
    What the broker charges for a trade of this size.

    A per-contract fee is flat. A per-share rate carries a floor charged per
    order and a ceiling set as a fraction of the trade's value.
    """
    model = instrument(ticker)['commission']
    quantity = abs(quantity)
    if 'per_contract' in model:
        return quantity * model['per_contract']
    charged = max(model['minimum'], quantity * model['per_share'])
    return min(charged, model['max_pct'] * quantity * price)


def minimum_trade(ticker, price):
    """
    Smallest trade whose commission stays inside the ceiling.

    A flat per-contract fee costs the same however few are traded, so there
    is nothing to spread and one contract is always acceptable. A floor
    charged per order does not spread until the trade is large enough.
    """
    model = instrument(ticker)['commission']
    if 'per_contract' in model:
        return 1
    return max(1, math.ceil(model['minimum'] / (model['max_pct'] * price)))


def round_to_tick(price, ticker):
    """The nearest price the exchange will accept."""
    tick = instrument(ticker)['tick']
    return round(round(price / tick) * tick, 10)


def validate():
    """Refuse a universe the rest of the system cannot act on."""
    for ticker, entry in INSTRUMENTS.items():
        missing = [k for k in REQUIRED if k not in entry]
        if missing:
            raise ValueError(f"{ticker} is missing: {', '.join(missing)}")
        if entry['type'] not in TYPES:
            raise ValueError(f"{ticker} has an unknown type: {entry['type']}")
        if entry['multiplier'] <= 0 or entry['tick'] <= 0:
            raise ValueError(f"{ticker} has a non-positive multiplier or tick")

    traded = tickers()
    if not traded:
        raise ValueError("no instrument carries a weight")
    total = sum(weights().values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"weights sum to {total:.6f}, expected 1.0")

def ticker_from_path(path):
    """
    The ticker a file belongs to.

    Handles both the downloader's SPY_<timestamp>.csv and the features file's
    plain SPY.csv.
    """
    return Path(path).stem.split('_')[0].upper()


validate()

WEIGHTS = weights()
