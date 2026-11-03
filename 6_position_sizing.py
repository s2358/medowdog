"""
Portfolio position sizing with compounding capital.

    python 6_position_sizing.py data/features/SPY.csv data/features/GLDM.csv \
        data/features/SLV.csv

Capital is fixed at BACKTEST_CAPITAL. This is research, not trading, so the result must not move because the live account moved. Nothing here reads the account, the allocation file, or anything else the live path writes.

Capital is not constant inside the run. Each day's positions are sized against the equity produced by every prior day's P&L, which makes the whole calculation sequential: equity -> vol scalar -> position -> next day's P&L -> equity.

Costs come from universe.py, one figure per instrument: half the quoted spread plus commission, measured by 5_costs.py and written in by hand. COST_MULTIPLIER scales them all for a sensitivity run.

Dividends are read from data/dividends/ and credited to whatever position was
held into each ex-date, net of US withholding - negative when the position was
short. A ticker with no file there contributes nothing.

The loop carries a cash balance, not only an equity figure. Buying a share spends cash, shorting one raises it, and a price change moves equity without touching cash at all. Equity is then cash plus the market value of what is held. The two numbers are needed separately because the broker charges on the cash balance, not on the account value.

Three charges come out of that balance every calendar day, accrued actual/360, so a Friday position carries three days into Monday:

  interest  the benchmark plus FINANCING_SPREAD on everything the broker has
            lent - the short positions it holds shares against, plus any cash
            debit. Measured over 23 to 26 October 2026, this base reproduces
            the broker's own accrual to within four percent. The cash balance
            alone does not: it implies rates above eight percent.
  credit    the benchmark less CREDIT_SPREAD on a cash balance above
            CASH_THRESHOLD. Paid, not charged.
  borrow    BORROW_FEE on the market value of the short positions, whatever
            the cash balance.

A futures position is excluded from all three: it borrows nothing, because the cost of carry is already in the contract's price, and its profit settles into cash every day rather than sitting on the balance sheet.

Gross leverage is capped at MAX_LEVERAGE. The broker's limit applies to the whole portfolio, so it can only be checked once every instrument has been sized - a second pass scales every position by the same factor and re-rounds. Whole block means the result lands near the cap rather than exactly on it.

The loop runs in USD, because the account is funded by converting SGD up front. Each day's P&L, costs, financing and dividends are USD, and so is the equity the vol target is sized against. The home-currency curve is that equity at each day's rate, which exposes the whole balance to USDSGD, not just the day's flows - reported as its own fx_pnl line so the identity equity.diff() == net_pnl still holds exactly.

Writes positions_<TICKER>.csv and portfolio.csv for the history, and turnover
statistics per instrument. Today's targets are targets.py, which does not
depend on any of this.
"""

import os
import sys

import numpy as np
import pandas as pd

import sizing
import dividends
import fx_data
from targets import load_instrument, on_index, support_start
import universe
from universe import ticker_from_path, WEIGHTS
from config import (VOL_TARGET, HOME_CURRENCY, BUSINESS_DAYS, BACKTEST_CAPITAL,
                    INERTIA, US_WITHHOLDING, FINANCING_SPREAD, CREDIT_SPREAD,
                    CASH_THRESHOLD, BORROW_FEE, MAX_LEVERAGE,
                    WINDOW_START, WINDOW_END, BENCHMARK_FILE, PORTFOLIO_DIR,
                    COST_MULTIPLIER, IDM)


PER_TICKER_COLS = ('icv', 'vol_scalar', 'subsystem_position',
                   'portfolio_position', 'rounded', 'held', 'trade',
                   'notional', 'pnl', 'cost', 'dividend')


def load_income(tickers, index):
    """
    Net dividend per share on each row of the backtest, per ticker.

    Stored amounts are gross. Withholding comes off here: a Singapore account has no US treaty rate, so 30% of every distribution never arrives.

    An ex-date that is not itself a row - the index holds every day any instrument traded - is carried to the next row rather than dropped. The entitlement is still the position held before the real ex-date, which is what the loop uses.
    """
    out, missing = {}, []
    for ticker in tickers:
        series = dividends.load(ticker)
        if series is None:
            missing.append(ticker)
        paid = np.zeros(len(index))
        if series is not None and not series.empty:
            net = series * (1 - US_WITHHOLDING)
            for position, amount in zip(index.searchsorted(net.index),
                                        net.values):
                if position < len(index):
                    paid[position] += amount
        out[ticker] = paid
    return out, missing


def load_benchmark(index, path=BENCHMARK_FILE):
    """
    The bare benchmark rate on each row, as a decimal.

    FRED's DFF is in percent and covers every calendar day, but publishes a few days late, so the last rows forward-fill off the end of the file. The spreads are applied in the loop, because the debit side and the credit side move in opposite directions from this number.
    """
    raw = pd.read_csv(path, parse_dates=['observation_date'])
    rate = raw.set_index('observation_date')['DFF'] / 100
    aligned = rate.reindex(rate.index.union(index)).ffill().reindex(index)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index[0].date()
        raise ValueError(f"{path} has no rate on or before {missing}")
    return aligned.values


def load_fx(index):
    """
    Home currency per unit of instrument currency on each row.

    The pair trades on days the US equity market is shut, so the series is aligned to the backtest index and carried forward.
    """
    series = fx_data.load_series()
    aligned = series.reindex(series.index.union(index)).ffill().reindex(index)
    if aligned.isna().any():
        missing = aligned[aligned.isna()].index[0].date()
        raise ValueError(f"no FX rate on or before {missing} - "
                         "run: python fx_data.py")
    return aligned.values


def run(frames, weights, idm, fx, capital0, vol_target, max_leverage, costs=None, income=None, rates=None):
    """
    Walk the history day by day, compounding equity.

    Within day i: yesterday's position earns today's price change, equity
    updates, then today's forecast is sized against that new equity and the
    resulting trade is paid for.
    """
    tickers = sorted(frames)
    index = frames[tickers[0]].index
    n = len(index)

    if len(fx) != n:
        raise ValueError(f"fx has {len(fx)} rows, the index has {n}")

    close = {t: frames[t]['close'].values for t in tickers}
    fcast = {t: frames[t]['combined_forecast'].values for t in tickers}
    icv = {t: frames[t]['icv'].values for t in tickers}

    # what one unit is worth, which is the price for a share and the price
    # times the contract size for a future
    value = {t: close[t] * universe.multiplier(t) for t in tickers}

    # a future borrows nothing - the cost of carry is already in its price -
    # and it holds no balance-sheet value, so only cash instruments sit in
    # equity and only they are financed
    financed = {t: universe.instrument(t)['type'] == 'STK' for t in tickers}

    income = income or {t: np.zeros(n) for t in tickers}

    rate = rates if rates is not None else np.zeros(n)

    # IBKR accrues actual/360 on the overnight balance every calendar day,
    # so a Friday row carries three days of interest into Monday.
    days = np.ones(n)
    if n > 1:
        days[1:] = np.diff(index.values).astype('timedelta64[D]').astype(float)

    cols = {t: {k: np.full(n, np.nan) for k in PER_TICKER_COLS}
            for t in tickers}
    equity = np.full(n, np.nan)
    cash_at = np.full(n, np.nan)
    short_at = np.zeros(n)
    gross_pnl = np.zeros(n)
    day_costs = np.zeros(n)
    day_interest = np.zeros(n)
    day_credit = np.zeros(n)
    day_borrow = np.zeros(n)
    day_capped = np.zeros(n, dtype=bool)
    day_scale = np.ones(n)

    cash = capital0
    held_prev = {t: 0.0 for t in tickers}

    # what was on the books overnight, carried so that day i's accrual is
    # charged on the balance that was actually held through the gap
    borrowed_prev = 0.0
    lent_prev = 0.0
    short_prev = 0.0
    ruin_at = None

    def book_value(i):
        """Market value of everything that sits on the balance sheet."""
        return sum(held_prev[t] * value[t][i]
                   for t in tickers
                   if financed[t] and held_prev[t] != 0)

    for i in range(n):
        # 1. yesterday's position meets today's price, and is paid whatever
        # went ex today. A dividend is cash. A price change is not, unless
        # the instrument is a future, which settles its profit every day.
        total = 0.0
        for t in tickers:
            p = (held_prev[t] * (value[t][i] - value[t][i - 1])
                 if i > 0 and held_prev[t] != 0 else 0.0)
            d = held_prev[t] * income[t][i]
            cols[t]['pnl'][i] = p
            cols[t]['dividend'][i] = d
            total += p + d
            cash += d
            if not financed[t]:
                cash += p
        gross_pnl[i] = total

        # 2. the money borrowed overnight is paid for at today's rate
        debit_rate = rate[i] + FINANCING_SPREAD
        credit_rate = max(0.0, rate[i] - CREDIT_SPREAD)
        accrual = days[i] / 360

        charged = borrowed_prev * debit_rate * accrual
        earned = lent_prev * credit_rate * accrual
        fee = short_prev * BORROW_FEE * accrual

        day_interest[i] = charged
        day_credit[i] = earned
        day_borrow[i] = fee
        cash += earned - charged - fee

        cap = cash + book_value(i)

        if cap <= 0:
            ruin_at = i
            break

        # 3. size today's positions against the equity we now hold
        daily_target = sizing.daily_cash_vol_target(cap, vol_target)
        want = {}

        for t in tickers:
            icv_i = icv[t][i]
            vol_scalar = sizing.volatility_scalar(daily_target, icv_i)
            subsystem = fcast[t][i] / 10 * vol_scalar
            target = sizing.target_position(fcast[t][i], icv_i, daily_target,
                                            weights[t], idm)
            rounded = (sizing.round_half_away(target)
                       if np.isfinite(target) else np.nan)

            c = cols[t]
            c['icv'][i] = icv_i
            c['vol_scalar'][i] = vol_scalar
            c['subsystem_position'][i] = subsystem
            c['portfolio_position'][i] = target
            c['rounded'][i] = rounded
            want[t] = sizing.apply_inertia(rounded, held_prev[t], INERTIA)

        # 3b. the cap is on the portfolio, so it can only be applied once
        # every instrument has been sized. An instrument with no price yet is
        # not in the book and cannot contribute to gross notional.
        priced = [t for t in tickers if np.isfinite(value[t][i])]
        capped, scale = sizing.cap_leverage(
                {t: want[t] for t in priced},
                {t: value[t][i] for t in priced}, cap, max_leverage)
        want = {**want, **capped}
        day_capped[i] = scale < 1.0
        day_scale[i] = scale

        # 3c. Pay for the trades the capped positions imply, not the ones the
        # forecast asked for. Buying spends cash and shorting raises it.
        spent = 0.0
        for t in tickers:
            new = want[t]
            current = held_prev[t]
            trade = new - current
            cost = abs(trade) * costs[t] if costs and t in costs else 0.0

            c = cols[t]
            c['held'][i] = new
            c['trade'][i] = trade
            c['notional'][i] = new * value[t][i] if new != 0 else 0.0
            c['cost'][i] = cost

            if financed[t] and trade != 0:
                cash -= trade * value[t][i]
            cash -= cost
            spent += cost
            held_prev[t] = new

        day_costs[i] = spent

        # 4. close the day. The broker lends two things: the shares behind
        # every short, and whatever cash was spent beyond the balance. It
        # charges on the sum of the two, and pays on a credit balance above
        # the threshold.
        short_value = sum(-held_prev[t] * value[t][i]
                          for t in tickers
                          if financed[t] and held_prev[t] < 0)

        borrowed_prev = short_value + max(0.0, -cash)
        lent_prev = max(0.0, cash - CASH_THRESHOLD)
        short_prev = short_value

        cash_at[i] = cash
        short_at[i] = short_value
        equity[i] = cash + book_value(i)

    # if the account was wiped out, keep only the days that happened
    end = ruin_at if ruin_at is not None else n
    index = index[:end]

    built = {}
    for t in tickers:
        df = pd.DataFrame({k: v[:end] for k, v in cols[t].items()}, index=index)
        df.insert(0, 'close', close[t][:end])
        df.insert(1, 'combined_forecast', fcast[t][:end])
        built[t] = df

    notional = pd.DataFrame(
        {t: built[t]['notional'] for t in tickers}).abs().sum(axis=1)

    paid = pd.DataFrame({t: built[t]['dividend'] for t in tickers}).sum(axis=1)

    # the account is USD, so the home-currency curve revalues the whole
    # balance, not just the day's flows:
    #   equity_home[i] - equity_home[i-1]
    #       = net_pnl_usd[i] * conv[i] + equity_usd[i-1] * (conv[i] - conv[i-1])
    # the second term is fx_pnl, and the two together make the difference exact

    eq = equity[:end]
    conv = fx[:end]
    prev_eq = np.concatenate(([capital0], eq[:-1]))
    prev_conv = np.concatenate(([conv[0]], conv[:-1]))
    fx_pnl = prev_eq * (conv - prev_conv)

    gross = gross_pnl[:end] * conv
    spent = day_costs[:end] * conv
    interest = day_interest[:end] * conv
    credit = day_credit[:end] * conv
    borrow = day_borrow[:end] * conv
    financing = interest + borrow - credit

    portfolio = pd.DataFrame({
        'gross_pnl': gross,
        'dividend': paid * conv,
        'cost': spent,
        'interest': interest,
        'credit': credit,
        'borrow': borrow,
        'financing': financing,
        'capped': day_capped[:end],
        'cap_scale': day_scale[:end],
        'fx_pnl': fx_pnl,
        'net_pnl': gross - spent - financing + fx_pnl,
        'equity': eq * conv,
        'equity_usd': eq,
        'cash_usd': cash_at[:end],
        'short_notional': short_at[:end] * conv,
        'fx': conv,
        'gross_notional': notional * conv,
    }, index=index)
    portfolio['gross_leverage'] = (portfolio['gross_notional']
                                   / portfolio['equity'])

    return built, portfolio, ruin_at


def turnover_stats(df, years):
    """
    Per-instrument turnover and what holding it actually cost.

    Carver: turnover = avg blocks traded per year / (2 x avg abs held).

    The cost and dividend columns are in USD, as the loop runs. Dividend is
    signed by the position: a short pays the distribution out, and pays it in
    full, where a long receives it net of 30% withholding. That asymmetry is
    invisible in the speed-limit test, which looks only at the spread and the
    commission, so an instrument with a yield can be cheap to trade and
    expensive to hold.
    """
    blocks_traded = df['trade'].abs().sum()
    avg_held = df['held'].abs().mean()
    traded_per_year = blocks_traded / years
    turnover = traded_per_year / (2 * avg_held) if avg_held else np.nan

    trades = df.loc[df['trade'] != 0, 'trade'].abs()
    notional = df['notional'].abs()
    avg_notional = notional[notional > 0].mean()
    dividend = df['dividend'].sum()

    return {
        'avg_abs_position': avg_held,
        'avg_notional_usd': avg_notional,
        'pct_days_short': float((df['held'] < 0).mean()),
        'blocks_traded_per_year': traded_per_year,
        'turnover': turnover,
        'n_trades': len(trades),
        'trades_per_year': len(trades) / years,
        'typical_trade': trades.mean() if len(trades) else np.nan,
        'median_trade': trades.median() if len(trades) else np.nan,
        'pct_days_zero_position': (df['held'] == 0).mean(),
        'gross_pnl_per_year': df['pnl'].sum() / years,
        'cost_per_year': df['cost'].sum() / years,
        'dividend_per_year': dividend / years,
        'dividend_pct_notional': (dividend / years / avg_notional
                                  if avg_notional else np.nan),
    }


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} "
              "<instrument_csv> [<instrument_csv> ...]")
        sys.exit(1)

    account_value = BACKTEST_CAPITAL
    cached = fx_data.describe()

    inst_paths = sys.argv[1:]
    frames = {ticker_from_path(p): load_instrument(p) for p in inst_paths}

    missing = set(frames) - set(WEIGHTS)
    if missing:
        raise KeyError(f"no weight in the universe for: {sorted(missing)}")

    costs = {t: universe.cost_per_share(t) * COST_MULTIPLIER
             for t in sorted(frames)}

    # every trading day any instrument had, so a short history truncates
    # nothing - an instrument contributes from the day its own data begins
    dates = None
    for df in frames.values():
        dates = df.index if dates is None else dates.union(df.index)

    if WINDOW_START is not None:
        dates = dates[dates >= pd.Timestamp(WINDOW_START)]
    if WINDOW_END is not None:
        dates = dates[dates < pd.Timestamp(WINDOW_END)]

    start = support_start()
    if len(dates) and dates[0] < start:
        dropped = int((dates < start).sum())
        print(f"trimmed {dropped} rows before {start.date()}, where the fx "
              "and benchmark rate series begin")
        dates = dates[dates >= start]

    if len(dates) == 0:
        raise ValueError("the configured window leaves no rows")

    frames = {t: on_index(df, dates) for t, df in frames.items()}

    income, no_file = load_income(sorted(frames), dates)
    rates = load_benchmark(dates)

    fx = load_fx(dates)
    capital0 = account_value / fx[0]

    counts = ', '.join(f"{t} {int((income[t] > 0).sum())}"
                       for t in sorted(income))

    print(f"capital {account_value:,.0f} {HOME_CURRENCY} = {capital0:,.0f} USD "
          f"at {fx[0]:.5f} (fixed, compounding)   "
          f"vol target {VOL_TARGET:.0%}   IDM {IDM:.4f}")

    print(f"fx {cached['pair']} history {fx[0]:.5f} -> {fx[-1]:.5f}")

    multiple = f" x{COST_MULTIPLIER:g}" if COST_MULTIPLIER != 1.0 else ""
    print("costs per share: "
          + ", ".join(f"{t} {c:.5f}" for t, c in sorted(costs.items()))
          + multiple)
    print(f"benchmark {rates.min():.2%} to {rates.max():.2%} over the window")
    print(f"financing: debit benchmark + {FINANCING_SPREAD:.2%} on shorts plus "
          f"any cash debit, credit benchmark - {CREDIT_SPREAD:.2%} above "
          f"{CASH_THRESHOLD:,.0f} USD, borrow {BORROW_FEE:.2%} on short value")
    print(f"dividends: {counts}"
          + (f"  (never fetched: {', '.join(no_file)})" if no_file else ''))
    print(f"window: {dates[0].date()} -> {dates[-1].date()}  "
          f"({len(dates)} rows)\n")

    built, portfolio, ruin_at = run(
        frames, WEIGHTS, IDM, fx, capital0, VOL_TARGET, MAX_LEVERAGE, costs, income, rates)

    if ruin_at is not None:
        print(f"*** account wiped out on {dates[ruin_at].date()} "
              f"(day {ruin_at} of {len(dates)}) - truncated there ***\n")

    years = len(portfolio) / BUSINESS_DAYS

    stats = {t: turnover_stats(df, years) for t, df in built.items()}
    summary = pd.DataFrame(stats).T
    summary.index.name = 'ticker'
    print(summary.round(3).to_string())

    equity = portfolio['equity']
    lev = portfolio['gross_leverage']
    print(f"\nequity   start {account_value:,.0f}   end {equity.iloc[-1]:,.0f}   "
          f"min {equity.min():,.0f}   max {equity.max():,.0f} {HOME_CURRENCY}")
    print(f"gross leverage   mean {lev.mean():.2f}x   "
          f"median {lev.median():.2f}x   max {lev.max():.2f}x   "
          f"cap {MAX_LEVERAGE:.0f}x binds on {portfolio['capped'].mean():.1%} "
          f"of days")
    if costs:
        print(f"costs paid   {portfolio['cost'].sum():,.0f} {HOME_CURRENCY} "
              f"({portfolio['cost'].sum()/years:,.0f}/yr)")

    interest = portfolio['interest'].sum()
    credit = portfolio['credit'].sum()
    borrow = portfolio['borrow'].sum()
    fin = portfolio['financing'].sum()
    print(f"interest     {interest:,.0f} {HOME_CURRENCY}  "
          f"({interest/years:,.0f}/yr)")
    print(f"credit       {credit:,.0f} {HOME_CURRENCY}  "
          f"({credit/years:,.0f}/yr, received)")
    print(f"borrow       {borrow:,.0f} {HOME_CURRENCY}  "
          f"({borrow/years:,.0f}/yr)")
    print(f"financing    {fin:,.0f} {HOME_CURRENCY}  "
          f"({fin/years:,.0f}/yr, net)")

    days_short = (portfolio['short_notional'] > 0).mean()
    days_debit = (portfolio['cash_usd'] < 0).mean()
    print(f"             short on {days_short:.0%} of days, "
          f"cash debit on {days_debit:.0%}")

    moved = portfolio['fx_pnl'].sum()
    print(f"fx           {moved:,.0f} {HOME_CURRENCY}  "
          f"({moved/years:,.0f}/yr on the whole balance)")

    paid = portfolio['dividend'].sum()
    if paid:
        print(f"dividends   {paid:,.0f} {HOME_CURRENCY}  "
              f"({paid/years:,.0f}/yr, net of {US_WITHHOLDING:.0%} "
              f"withholding)")

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    stem = '_'.join(sorted(built))
    summary.to_csv(PORTFOLIO_DIR / f'turnover_{stem}.csv')
    portfolio.to_csv(PORTFOLIO_DIR / 'portfolio.csv')
    for ticker, df in built.items():
        df.to_csv(PORTFOLIO_DIR / f'positions_{ticker}.csv')
    print(f"\nwritten to {PORTFOLIO_DIR}/")


if __name__ == '__main__':
    main()
