"""
Live trading dashboard.

    python dashboard.py

Everything the live account has done since the first order, on one page. It
reads only files the daily run has already written, so it needs no broker
connection and can be rebuilt at any hour.

    data/portfolio/live.csv          the equity curve, from equity.py
    data/portfolio/live_targets.csv  what the system wants to hold today
    data/fills/fills.csv             every execution
    data/portfolio/portfolio.csv     the backtest, for the comparison

The book is the signed sum of the fills rather than a reading from the
broker. That is the same quantity reconcile.py checks the broker against, so
if the two ever disagree it is reconcile.py that says so, and this page shows
what the system believes it holds.

Three questions, in the order they matter at this stage:

  Is the book where the system wants it?   A gap is a missed fill, a stale
  features file, or a target that moved since the last trade.

  Did a trade cost what the backtest assumes?   Every result in the backtest
  rests on cost_per_share, and these are the first real executions to test it.

  Is the live curve tracking the model?   Rounding, fills and timing make the
  two diverge. The size of the divergence matters more than the return when
  there are only a few weeks of it.

Writes data/portfolio/dashboard.html.
"""

import importlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from capital import READINGS_FILE
from config import (FILLS_DIR, HOME_CURRENCY, IDM, INERTIA, PORTFOLIO_DIR,
                    VOL_TARGET)
import universe
from universe import WEIGHTS

# the evaluation report's stylesheet and tile helper, so the two pages look
# like one system rather than two
report = importlib.import_module('7_evaluate')

LIVE_FILE = PORTFOLIO_DIR / 'live.csv'
TARGETS_FILE = PORTFOLIO_DIR / 'live_targets.csv'
MODEL_FILE = PORTFOLIO_DIR / 'portfolio.csv'
FILLS_FILE = FILLS_DIR / 'fills.csv'
OUT_FILE = PORTFOLIO_DIR / 'dashboard.html'

CHART_W = 880
PAD_L, PAD_R, PAD_T, PAD_B = 56, 14, 14, 30

SIGN = {'BUY': 1, 'SELL': -1}


# ------------------------------------------------------------------ loading

def load_live(path=LIVE_FILE):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run equity.py first")
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = 'date'
    return frame.sort_index()


def load_targets(path=TARGETS_FILE):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run targets.py first")
    return pd.read_csv(path).set_index('ticker')


def load_fills(path=FILLS_FILE):
    """
    Every execution, oldest first.

    Nothing is dropped. A fill that failed to record its slippage still
    counts as a trade and still paid commission, so the columns are coerced
    rather than filtered.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - nothing has traded yet")
    frame = pd.read_csv(path)
    frame['filled_at'] = pd.to_datetime(frame['filled_at'], utc=True)
    for column in ('shares', 'price', 'commission', 'slippage_per_share',
                   'slippage_total', 'modelled_half_spread',
                   'shortfall_per_share', 'shortfall_total'):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
    return frame.sort_values('filled_at')


def load_model(path=MODEL_FILE):
    """The backtest curve, or None. The page works without it."""
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = 'date'
    return frame.sort_index()


def load_readings(path=READINGS_FILE):
    if not path.exists():
        return None
    return pd.read_csv(path)


# ------------------------------------------------------------------ the book

def held_from_fills(fills):
    """Signed shares per ticker. A sell is negative."""
    signed = fills['shares'] * fills['side'].map(SIGN).fillna(0)
    return signed.groupby(fills['ticker']).sum().to_dict()


def book_rows(targets, held, idm=IDM, inertia=INERTIA):
    """
    Target against holding, per instrument.

    The gap is expressed against the average position - the size the
    instrument would carry at a forecast of 10 - because a ten-share gap
    means something different on a 400-share position than on a 40-share one.
    That is the same denominator the inertia rule uses to decide whether to
    trade, so a gap above the band is a gap the system would act on.
    """
    rows = []
    for ticker in sorted(set(targets.index) | set(held)):
        target = (float(targets.loc[ticker, 'rounded'])
                  if ticker in targets.index else 0.0)
        have = float(held.get(ticker, 0.0))
        gap = target - have

        weight = WEIGHTS.get(ticker, 0.0)
        scalar = (float(targets.loc[ticker, 'vol_scalar'])
                  if ticker in targets.index else np.nan)
        average = abs(scalar * weight * idm) if np.isfinite(scalar) else np.nan
        share = gap / average if average else np.nan

        rows.append({
            'ticker': ticker,
            'target': target,
            'held': have,
            'gap': gap,
            'average_position': average,
            'gap_fraction': share,
            'would_trade': bool(np.isfinite(share) and abs(share) >= inertia),
            'close': (float(targets.loc[ticker, 'close'])
                      if ticker in targets.index else np.nan),
            'forecast': (float(targets.loc[ticker, 'combined_forecast'])
                         if ticker in targets.index else np.nan),
        })
    return rows


# ------------------------------------------------------------------ the cost

def cost_rows(fills):
    """
    What trading has cost, per instrument, measured against the price the
    target was computed from.

    Implementation shortfall is the honest measure. The decision was taken on
    a session close and the fill happened later at whatever the market then
    was, so the difference carries both the delay and the execution - the
    whole cost of turning a target into a position. It is signed: a trade
    that filled better than its decision price shows as a gain.

    Slippage against the quote is not used. The paper account has no live
    data subscription, so that quote is fifteen minutes old and the figure
    measures market drift rather than execution.

    Everything is in basis points of the value traded. A cent a share means
    something different on a 760-dollar instrument than on a 45-dollar one,
    and only basis points can be added together.
    """
    rows = []
    for ticker, group in fills.groupby('ticker'):
        shares = float(group['shares'].abs().sum())
        traded = float((group['shares'].abs() * group['price']).sum())
        commission = float(group['commission'].fillna(0).sum())

        # only the fills that recorded a decision price can be measured
        per_share = group.get('shortfall_per_share')
        covered = (group[per_share.notna()] if per_share is not None
                   else group.iloc[0:0])
        covered_shares = float(covered['shares'].abs().sum())
        covered_value = float((covered['shares'].abs()
                               * covered['price']).sum())
        shortfall = float(covered['shortfall_total'].fillna(0).sum())
        covered_commission = float(covered['commission'].fillna(0).sum())

        cost_bp = ((covered_commission + shortfall) / covered_value * 1e4
                   if covered_value else np.nan)

        try:
            modelled = universe.cost_per_share(ticker)
        except KeyError:
            modelled = np.nan
        average_price = traded / shares if shares else np.nan
        modelled_bp = (modelled / average_price * 1e4
                       if np.isfinite(modelled) and average_price else np.nan)

        rows.append({
            'ticker': ticker,
            'fills': len(group),
            'shares': shares,
            'traded': traded,
            'commission': commission,
            'commission_bp': commission / traded * 1e4 if traded else np.nan,
            'covered_shares': covered_shares,
            'covered_value': covered_value,
            'shortfall': shortfall,
            'cost_bp': cost_bp,
            'modelled_bp': modelled_bp,
            'ratio': (cost_bp / modelled_bp
                      if np.isfinite(cost_bp) and modelled_bp else np.nan),
        })
    return rows

# -------------------------------------------------------------- the tracking

def tracking(live, model):
    """
    The live curve and the model's, over the days they share, rebased so the
    first shared day is 100.

    They cannot match. The model holds whatever its own history led it to
    hold, the account holds what it was able to fill, and neither rounds the
    same way. The point is the size of the difference, not its sign.
    """
    if model is None:
        return None
    shared = live.index.intersection(model.index)
    if len(shared) < 2:
        return None

    a = live.loc[shared, 'equity'].astype(float)
    b = model.loc[shared, 'equity'].astype(float)
    return pd.DataFrame({
        'live': a / a.iloc[0] * 100,
        'model': b / b.iloc[0] * 100,
    }, index=shared)


# ----------------------------------------------------------------- the chart

def x_positions(n):
    span = CHART_W - PAD_L - PAD_R
    if n < 2:
        return [PAD_L + span / 2]
    return [PAD_L + i * span / (n - 1) for i in range(n)]


def path_from(xs, ys):
    return 'M' + ' L'.join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def date_labels(index, limit=10):
    """Every point when there are few, otherwise an even sample."""
    n = len(index)
    step = 1 if n <= limit else int(np.ceil(n / limit))
    return [(i, index[i].strftime('%d %b')) for i in range(0, n, step)]


def curve_svg(frame, height=240):
    """
    Two rebased curves on a linear scale.

    Linear rather than log: a few weeks of a curve that starts at 100 has no
    decades in it, and a log axis would draw one gridline.
    """
    xs = x_positions(len(frame))
    values = frame.values.astype(float)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    pad = max((hi - lo) * 0.12, 0.4)
    top, bottom = hi + pad, lo - pad
    span_px = height - PAD_T - PAD_B

    def y_of(v):
        return PAD_T + (1 - (v - bottom) / (top - bottom)) * span_px

    parts = []
    for tick in np.linspace(bottom, top, 5):
        y = y_of(tick)
        parts.append(f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" '
                     f'x2="{CHART_W - PAD_R}" y2="{y:.1f}"/>')
        parts.append(f'<text class="ytick" x="{PAD_L - 8}" '
                     f'y="{y + 3.5:.1f}">{tick:.1f}</text>')

    base = y_of(100.0)
    parts.append(f'<line class="start" x1="{PAD_L}" y1="{base:.1f}" '
                 f'x2="{CHART_W - PAD_R}" y2="{base:.1f}"/>')

    for i, label in date_labels(frame.index):
        parts.append(f'<text class="xtick" x="{xs[i]:.1f}" '
                     f'y="{height - PAD_B + 16}">{label}</text>')

    if 'model' in frame.columns:
        ys = [y_of(v) for v in frame['model'].values]
        parts.append(f'<path class="modelline" d="{path_from(xs, ys)}"/>')
    ys = [y_of(v) for v in frame['live'].values]
    parts.append(f'<path class="line" d="{path_from(xs, ys)}"/>')
    for x, y in zip(xs, ys):
        parts.append(f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="3"/>')

    return (f'<svg class="chart" viewBox="0 0 {CHART_W} {height}" '
            f'preserveAspectRatio="none" role="img">{"".join(parts)}</svg>')


# ------------------------------------------------------------------ the page

EXTRA_STYLE = """
.grid2 { font-size: 13px; }
.grid2 th { color: var(--muted); font-weight: 500; font-size: 11px;
  text-transform: uppercase; letter-spacing: .04em; padding: 4px 8px;
  text-align: right; }
.grid2 th.l, .grid2 td.l { text-align: left; }
.grid2 td { text-align: right; padding: 6px 8px; background: var(--surface);
  border-radius: 4px; }
.grid2 tr.sum td, .grid2 tr.sum th { font-weight: 600;
  border-top: 1px solid var(--border); }
.flag { color: var(--neg); font-weight: 600; }
.ok { color: var(--muted); }
.modelline { fill: none; stroke: var(--axis); stroke-width: 2;
  stroke-dasharray: 5 4; stroke-linejoin: round;
  vector-effect: non-scaling-stroke; }
.dot { fill: var(--surface); stroke: var(--line); stroke-width: 2; }
.key { display: flex; gap: 18px; color: var(--muted); font-size: 11.5px;
  margin: 8px 0 0; }
.key i { display: inline-block; width: 18px; height: 0;
  border-top: 2px solid var(--line); vertical-align: middle;
  margin-right: 6px; }
.key i.m { border-top: 2px dashed var(--axis); }
.warn { color: var(--ink2); font-size: 12.5px; background: var(--surface);
  border: 1px solid var(--border); border-left: 3px solid var(--neg);
  border-radius: 0 8px 8px 0; padding: 9px 12px; margin: 10px 0 0; }
"""


def num(value, places=0, dash='--'):
    if value is None or not np.isfinite(value):
        return dash
    return f"{value:,.{places}f}"


def book_table_html(rows):
    head = ('<tr><th class="l">instrument</th><th>forecast</th><th>target</th>'
            '<th>held</th><th>gap</th><th>average</th><th>gap / average</th>'
            '<th class="l">status</th></tr>')
    body = []
    for row in rows:
        flag = ('<span class="flag">trade</span>' if row['would_trade']
                else '<span class="ok">inside band</span>')
        body.append(
            f'<tr><td class="l">{row["ticker"]}</td>'
            f'<td>{num(row["forecast"], 1)}</td>'
            f'<td>{num(row["target"])}</td>'
            f'<td>{num(row["held"])}</td>'
            f'<td>{num(row["gap"])}</td>'
            f'<td>{num(row["average_position"])}</td>'
            f'<td>{"--" if not np.isfinite(row["gap_fraction"]) else f"{row['gap_fraction']:+.1%}"}</td>'
            f'<td class="l">{flag}</td></tr>')
    return (f'<div class="scroll"><table class="grid2"><thead>{head}</thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')



def cost_table_html(rows):
    head = ('<tr><th class="l">instrument</th><th>fills</th><th>shares</th>'
            '<th>measured on</th><th>commission bp</th><th>shortfall</th>'
            '<th>cost bp</th><th>modelled bp</th><th>ratio</th></tr>')
    body = []
    for row in rows:
        ratio, cost = row['ratio'], row['cost_bp']
        cls = ' class="flag"' if np.isfinite(ratio) and ratio >= 2 else ''
        covered = (f"{row['covered_shares']:,.0f}"
                   if row['covered_shares'] else '--')
        body.append(
            f'<tr><td class="l">{row["ticker"]}</td>'
            f'<td>{row["fills"]}</td>'
            f'<td>{num(row["shares"])}</td>'
            f'<td>{covered}</td>'
            f'<td>{num(row["commission_bp"], 1)}</td>'
            f'<td>{"--" if not row["covered_shares"] else f"{row['shortfall']:+,.2f}"}</td>'
            f'<td>{"--" if not np.isfinite(cost) else f"{cost:+,.1f}"}</td>'
            f'<td>{num(row["modelled_bp"], 1)}</td>'
            f'<td{cls}>{"--" if not np.isfinite(ratio) else f"{ratio:+.1f}x"}</td>'
            f'</tr>')

    traded = sum(r['traded'] for r in rows)
    covered_value = sum(r['covered_value'] for r in rows)
    commission = sum(r['commission'] for r in rows)
    shortfall = sum(r['shortfall'] for r in rows)
    covered_commission = sum(
        r['commission'] * (r['covered_value'] / r['traded'])
        if r['traded'] else 0.0 for r in rows)
    total_bp = ((covered_commission + shortfall) / covered_value * 1e4
                if covered_value else np.nan)

    body.append(
        f'<tr class="sum"><th class="l">total</th>'
        f'<td>{sum(r["fills"] for r in rows)}</td>'
        f'<td>{num(sum(r["shares"] for r in rows))}</td>'
        f'<td>{num(sum(r["covered_shares"] for r in rows))}</td>'
        f'<td>{num(commission / traded * 1e4 if traded else np.nan, 1)}</td>'
        f'<td>{shortfall:+,.2f}</td>'
        f'<td>{"--" if not np.isfinite(total_bp) else f"{total_bp:+,.1f}"}</td>'
        f'<td></td><td></td></tr>')
    return (f'<div class="scroll"><table class="grid2"><thead>{head}</thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def fills_table_html(fills, limit=25):
    recent = fills.tail(limit).iloc[::-1]
    head = ('<tr><th class="l">filled</th><th class="l">instrument</th>'
            '<th class="l">side</th><th>shares</th><th>price</th>'
            '<th>limit</th><th>commission</th><th>slippage /share</th></tr>')
    body = []
    for _, row in recent.iterrows():
        body.append(
            f'<tr><td class="l">'
            f'{row["filled_at"].strftime("%Y-%m-%d %H:%M")}</td>'
            f'<td class="l">{row["ticker"]}</td>'
            f'<td class="l">{row["side"]}</td>'
            f'<td>{num(row["shares"])}</td>'
            f'<td>{num(row["price"], 2)}</td>'
            f'<td>{num(row.get("limit", np.nan), 2)}</td>'
            f'<td>{num(row["commission"], 2)}</td>'
            f'<td>{num(row.get("slippage_per_share", np.nan), 3)}</td></tr>')
    return (f'<div class="scroll"><table class="grid2"><thead>{head}</thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def write_dashboard(path, live, book, costs, track, fills, readings):
    equity = live['equity'].astype(float)
    start = float(equity.iloc[0] - live['net_pnl'].iloc[0])
    change = float(equity.iloc[-1]) - start

    gaps = [abs(r['gap_fraction']) for r in book
            if np.isfinite(r['gap_fraction'])]
    worst_gap = max(gaps) if gaps else np.nan

    if track is not None:
        spread = float(track['live'].iloc[-1] - track['model'].iloc[-1])
        track_note = f"{spread:+.2f} points against the model"
    else:
        spread = np.nan
        track_note = 'no overlap with the backtest'

    commission = sum(r['commission'] for r in costs)
    shortfall = sum(r['shortfall'] for r in costs)

    tiles = ''.join([
        report.tile('sessions', f"{len(live)}",
                    f"since {equity.index[0].date()}"),
        report.tile('equity', f"{equity.iloc[-1]:,.0f}",
                    f"from {start:,.0f} {HOME_CURRENCY}"),
        report.tile('result', f"{change:+,.0f}",
                    f"{change / start:+.2%}"),
        report.tile('fills', f"{len(fills)}",
                    f"{commission:,.2f} commission, {shortfall:+,.0f} shortfall"),
        report.tile('worst gap',
                    '--' if not np.isfinite(worst_gap) else f"{worst_gap:.0%}",
                    f"band is {INERTIA:.0%} of average position"),
        report.tile('tracking',
                    '--' if not np.isfinite(spread) else f"{spread:+.2f}",
                    track_note),
    ])

    chart = (f'<div class="panel">{curve_svg(track)}</div>'
             '<div class="key"><span><i></i>live</span>'
             '<span><i class="m"></i>model</span></div>'
             if track is not None else
             '<p class="cap">Not enough shared days with the backtest yet.</p>')

    reading_count = 0 if readings is None else len(readings)

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Medowdog live</title>
<style>{report.STYLE}{EXTRA_STYLE}</style></head>
<body><div class="wrap">
<h1>Medowdog live</h1>
<p class="sub">{equity.index[0].date()} to {equity.index[-1].date()} &middot;
{len(live)} sessions &middot; {reading_count} account readings &middot;
vol target {VOL_TARGET:.0%} &middot; IDM {IDM:.4f} &middot;
built {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="tiles">{tiles}</div>

<h2>Book against target</h2>
<p class="cap">Holdings are the signed sum of recorded fills. The gap is
measured against the average position, which is what the inertia rule
uses.</p>
{book_table_html(book)}

<h2>Cost, realised against modelled</h2>
<p class="cap">Implementation shortfall against the close the target was
computed from, plus commission, in basis points of the value traded.
Negative is a gain. "Measured on" is the number of shares whose fill recorded
a decision price; the rest cannot be measured.</p>
{cost_table_html(costs)}
<p class="warn">Shortfall carries the delay as well as the execution. The
SPY sell filled three sessions after its decision close, so most of that
number is the market moving, not the broker. It is still the right thing to
measure, because the delay is a real cost of trading by hand - it just
happened to pay this time.</p>

<h2>Equity, live against the model</h2>
<p class="cap">Both rebased to 100 on the first shared day. They will
diverge; the size of the gap is the tracking error.</p>
{chart}

<h2>Recent fills</h2>
{fills_table_html(fills)}

<footer>built from {LIVE_FILE}, {TARGETS_FILE}, {FILLS_FILE}</footer>
</div>
</body></html>
"""
    path.write_text(html, encoding='utf-8')


def main():
    try:
        live = load_live()
        targets = load_targets()
        fills = load_fills()
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)

    model = load_model()
    readings = load_readings()

    held = held_from_fills(fills)
    book = book_rows(targets, held)
    costs = cost_rows(fills)
    track = tracking(live, model)

    equity = live['equity'].astype(float)
    start = float(equity.iloc[0] - live['net_pnl'].iloc[0])
    change = float(equity.iloc[-1]) - start

    print(f"{len(live)} session(s)  {equity.index[0].date()} -> "
          f"{equity.index[-1].date()}")
    print(f"equity {start:,.2f} -> {equity.iloc[-1]:,.2f} {HOME_CURRENCY}   "
          f"{change:+,.2f} ({change / start:+.2%})")

    print("\nbook against target:")
    for row in book:
        mark = '  TRADE' if row['would_trade'] else ''
        share = ('--' if not np.isfinite(row['gap_fraction'])
                 else f"{row['gap_fraction']:+.1%}")
        print(f"  {row['ticker']:6s} target {row['target']:>8,.0f}   "
              f"held {row['held']:>8,.0f}   gap {row['gap']:>7,.0f}   "
              f"{share:>8s}{mark}")

    print("\ncost, realised against modelled (bp of value traded):")
    for row in costs:
        if not row['covered_shares']:
            print(f"  {row['ticker']:6s} no fill recorded a decision price "
                  f"({row['fills']} fill(s), {row['shares']:,.0f} shares)")
            continue
        ratio = ('--' if not np.isfinite(row['ratio'])
                 else f"{row['ratio']:+.1f}x")
        print(f"  {row['ticker']:6s} {row['cost_bp']:+8.1f} bp   vs "
              f"{row['modelled_bp']:7.1f} bp   {ratio:>7s}   "
              f"shortfall {row['shortfall']:+,.2f}, commission "
              f"{row['commission_bp']:.1f} bp   on "
              f"{row['covered_shares']:,.0f} of {row['shares']:,.0f} shares")

    if track is not None:
        spread = float(track['live'].iloc[-1] - track['model'].iloc[-1])
        print(f"\ntracking: live {track['live'].iloc[-1]:.2f} against model "
              f"{track['model'].iloc[-1]:.2f}, both from 100 on "
              f"{track.index[0].date()}  ({spread:+.2f})")
    else:
        print("\ntracking: no overlap with the backtest curve")

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    write_dashboard(OUT_FILE, live, book, costs, track, fills, readings)
    print(f"\nwritten to {OUT_FILE}")


if __name__ == '__main__':
    main()
