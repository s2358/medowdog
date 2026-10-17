"""
Evaluation statistics for the equity curve.

    python 7_evaluate.py
    python 7_evaluate.py path/to/portfolio.csv # incase the equity curve is not the current one

Daily returns are net_pnl_t / equity_{t-1}. Sharpe is on raw returns, not excess. Skew carries SE = sqrt(6/n).

The drag figures split sharpe_gross - sharpe_net into trading costs,
financing and currency, applied in that order.

Writes stats, monthly_returns, drawdown and report files to data/portfolio/,
named after the source file and the time of the run.
"""

import json
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from config import (CAPITAL, VOL_TARGET, HOME_CURRENCY, BUSINESS_DAYS, PORTFOLIO_DIR)

ANNUALISE = np.sqrt(BUSINESS_DAYS)
REQUIRED = ('gross_pnl', 'cost', 'net_pnl', 'equity')

def load_portfolio(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = 'date'
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df.sort_index()

def returns_on_capital(pnl, equity, capital0):
    """
    Daily return on the equity the position was sized against.

    equity holds end-of-day values, so yesterday's close is today's base.
    The first row has no predecessor and uses the starting capital.
    """
    base = equity.shift(1)
    base.iloc[0] = capital0
    return pnl / base

def period_returns(equity, capital0, freq):
    """
    Returns over calendar periods, compounded from the equity curve.

    to_period is used rather than resample so the frequency aliases stay valid across pandas versions
    """
    closes = equity.groupby(equity.index.to_period(freq)).last()
    prev = closes.shift(1)
    prev.iloc[0] = capital0
    return closes / prev - 1

def sharpe(returns):
    sd = returns.std()
    if not np.isfinite(sd) or sd == 0:
        return np.nan
    return returns.mean() / sd * ANNUALISE

def skew_with_se(returns, label):
    """
    pandas .skew() is the bias-corrected sample skew; the SE is the asymptotic one under normality, which is the usual reference point.
    """
    n = int(returns.count())
    return {
            'horizon': label,
            'n': n,
            'skew': returns.skew() if n > 2 else np.nan,
            'se': np.sqrt(6 / n) if n > 2 else np.nan,
            }

def drawdown_curve(equity, capital0):
    """Fraction below the running peak, with the start included as a peak."""
    peak = equity.cummax().clip(lower=capital0)
    return equity / peak - 1

def underwater_runs(dd):
    """Length in rows of each consecutive stretch below the prior peak"""
    runs, current = [], 0
    for below in (dd < 0).values:
        if below:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs

def drawdown_stats(equity, capital0):
    dd = drawdown_curve(equity, capital0)
    trough = dd.idxmin()

    # the peak this drawdown fell from, and the first day back above it
    before = equity.loc[:trough]
    start = before.idxmax() if before.max() > capital0 else equity.index[0]
    after = dd.loc[trough:]
    back = after[after >= 0]

    runs = underwater_runs(dd)
    return {
            'max_drawdown': dd.min(),
            'trough_on': trough,
            'fell_from': start,
            'recovered_on': back.index[0] if len(back) else None,
            'pct_days_underwater': float((dd < 0).mean()),
            'longest_drawdown_days': max(runs) if runs else 0,
            'curve': dd,
            }

def monthly_grid(monthly):
    """Monthly returns as a year x month table, in percent."""
    frame = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'ret': monthly.values * 100,
        })
    grid = frame.pivot(index='year', columns='month', values='ret')
    grid.columns = [pd.Timestamp(2000, m, 1).strftime('%b')
                    for m in grid.columns]
    grid['year'] = [((1 + monthly[monthly.index.year == y]).prod() - 1) * 100
                    for y in grid.index]
    return grid


def archive_previous(out_dir, pattern='stats_*.csv'):
    """Move earlier files matching the pattern into archive/"""
    old = sorted(out_dir.glob(pattern))
    if not old:
        return 0

    archive_dir = out_dir / 'archive'
    archive_dir.mkdir(exist_ok=True)
    for path in old:
        path.rename(archive_dir / path.name)
    return len(old)


# --------------------------------------------------------------- html report
#
# Charts are inline SVG built here, so the report needs no plotting library
# and no network. Geometry is shared between the two charts so their x axes
# line up: same width, same padding, same point positions.

CHART_W = 880
PAD_L, PAD_R, PAD_T, PAD_B = 62, 14, 12, 24


def x_positions(n):
    span = CHART_W - PAD_L - PAD_R
    if n < 2:
        return [PAD_L + span / 2]
    return [PAD_L + i * span / (n - 1) for i in range(n)]


def year_ticks(index):
    """(position, year) for the first row of each calendar year."""
    out, seen = [], set()
    for i, ts in enumerate(index):
        if ts.year not in seen:
            seen.add(ts.year)
            out.append((i, ts.year))
    return out


def log_ticks(lo, hi):
    """Decade ticks covering [lo, hi]. Half steps keep a short range labelled."""
    ticks = []
    mult = 10 ** int(np.floor(np.log10(lo)))
    while mult <= hi:
        for m in (1, 1.5, 2, 3, 5, 7):
            v = m * mult
            if lo <= v <= hi:
                ticks.append(v)
        mult *= 10
    return ticks or [lo, hi]


def pct_ticks(low):
    """Ticks from 0 down past low, in 10 or 20 point steps."""
    step = 0.1 if low > -0.35 else 0.2
    ticks, v = [], 0.0
    while v > low - step:
        ticks.append(round(v, 4))
        v -= step
    return ticks


def path_from(xs, ys):
    return 'M' + ' L'.join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def axis_svg(xs, index, ticks, label_of, y_of, height):
    """Horizontal gridlines with labels, plus year markers along the bottom."""
    parts = []
    for t in ticks:
        y = y_of(t)
        parts.append(f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" '
                     f'x2="{CHART_W - PAD_R}" y2="{y:.1f}"/>')
        parts.append(f'<text class="ytick" x="{PAD_L - 8}" y="{y + 3.5:.1f}">'
                     f'{label_of(t)}</text>')
    for i, year in year_ticks(index):
        if i == 0:
            continue
        parts.append(f'<line class="yearline" x1="{xs[i]:.1f}" y1="{PAD_T}" '
                     f'x2="{xs[i]:.1f}" y2="{height - PAD_B}"/>')
        parts.append(f'<text class="xtick" x="{xs[i]:.1f}" '
                     f'y="{height - PAD_B + 15}">{year}</text>')
    return ''.join(parts)


def equity_svg(equity, height=270):
    xs = x_positions(len(equity))
    lo, hi = float(equity.min()), float(equity.max())
    top, bottom = hi * 1.08, lo * 0.92
    span_px = height - PAD_T - PAD_B

    def y_of(v):
        frac = (np.log(v) - np.log(bottom)) / (np.log(top) - np.log(bottom))
        return PAD_T + (1 - frac) * span_px

    ys = [y_of(v) for v in equity.values]
    base = y_of(CAPITAL)
    body = [
        axis_svg(xs, equity.index, log_ticks(bottom, top),
                 lambda v: (f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"),
                 y_of, height),
        f'<line class="start" x1="{PAD_L}" y1="{base:.1f}" '
        f'x2="{CHART_W - PAD_R}" y2="{base:.1f}"/>',
        f'<path class="line" d="{path_from(xs, ys)}"/>',
    ]
    return svg_frame('eq', height, ''.join(body)), xs, ys


def drawdown_svg(dd, height=200):
    xs = x_positions(len(dd))
    low = float(dd.min())
    span_px = height - PAD_T - PAD_B
    floor = low * 1.06

    def y_of(v):
        return PAD_T + (v / floor) * span_px if floor else PAD_T

    ys = [y_of(v) for v in dd.values]
    area = (path_from(xs, ys)
            + f" L{xs[-1]:.1f},{y_of(0):.1f} L{xs[0]:.1f},{y_of(0):.1f} Z")
    body = [
        axis_svg(xs, dd.index, pct_ticks(low), lambda v: f"{v:.0%}", y_of,
                 height),
        f'<path class="area" d="{area}"/>',
        f'<path class="negline" d="{path_from(xs, ys)}"/>',
    ]
    return svg_frame('dd', height, ''.join(body)), xs, ys


def svg_frame(ident, height, body):
    return (f'<svg id="svg-{ident}" class="chart" viewBox="0 0 {CHART_W} '
            f'{height}" preserveAspectRatio="none" role="img">'
            f'{body}'
            f'<line class="cross" id="cross-{ident}" y1="{PAD_T}" '
            f'y2="{height - PAD_B}" x1="0" x2="0" style="display:none"/>'
            f'<circle class="dot" id="dot-{ident}" r="4" '
            f'style="display:none"/></svg>')


def grid_table_html(grid):
    """The monthly table, cells shaded by sign and magnitude."""
    body = grid.drop(columns='year')
    scale = float(np.nanmax(np.abs(body.values))) or 1.0

    head = ''.join(f'<th>{c}</th>' for c in body.columns)
    rows = []
    for year, row in grid.iterrows():
        cells = []
        for col in body.columns:
            v = row[col]
            if pd.isna(v):
                cells.append('<td class="empty"></td>')
                continue
            alpha = min(abs(v) / scale, 1.0) * 0.5
            sign = 'pos' if v >= 0 else 'neg'
            cells.append(f'<td class="cell {sign}" style="--a:{alpha:.3f}">'
                         f'{v:.1f}</td>')
        total = row['year']
        cells.append(f'<td class="total">{total:+.1f}</td>')
        rows.append(f'<tr><th class="yr">{year}</th>{"".join(cells)}</tr>')

    return (f'<div class="scroll"><table class="months">'
            f'<thead><tr><th></th>{head}'
            f'<th class="total">year</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def stats_table_html(stats):
    rows = ''.join(
        f'<tr><th>{k}</th><td>{v}</td></tr>' for k, v in stats)
    return f'<table class="stats"><tbody>{rows}</tbody></table>'


def tile(label, value, note=''):
    sub = f'<span class="note">{note}</span>' if note else ''
    return (f'<div class="tile"><span class="label">{label}</span>'
            f'<span class="value">{value}</span>{sub}</div>')


STYLE = """
:root {
  color-scheme: light;
  --plane: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --line: #2a78d6; --neg: #e34948;
  --pos-pole: #2a78d6; --neg-pole: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --plane: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --line: #3987e5; --neg: #e66767;
    --pos-pole: #3987e5; --neg-pole: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --plane: #0d0d0d; --surface: #1a1a19;
  --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --line: #3987e5; --neg: #e66767;
  --pos-pole: #3987e5; --neg-pole: #e66767;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--plane); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 14px; line-height: 1.5;
}
.wrap { max-width: 960px; margin: 0 auto; padding: 36px 20px 72px; }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 2px; }
h2 { font-size: 14px; font-weight: 600; margin: 32px 0 2px; }
.sub, .cap { color: var(--ink2); font-size: 13px; margin: 0; }
.cap { margin: 0 0 10px; }
.tiles {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px; margin: 20px 0 8px;
}
.tile {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 14px; display: flex;
  flex-direction: column; gap: 2px;
}
.tile .label { color: var(--muted); font-size: 11px; letter-spacing: .04em;
  text-transform: uppercase; }
.tile .value { font-size: 21px; font-weight: 600; }
.tile .note { color: var(--ink2); font-size: 11.5px; }
.panel {
  position: relative; background: var(--surface);
  border: 1px solid var(--border); border-radius: 10px; padding: 8px 6px 2px;
}
.chart { width: 100%; height: auto; display: block; touch-action: none; }
.grid { stroke: var(--grid); stroke-width: 1; }
.yearline { stroke: var(--grid); stroke-width: 1; stroke-dasharray: 2 3; }
.start { stroke: var(--axis); stroke-width: 1; stroke-dasharray: 4 3; }
.line { fill: none; stroke: var(--line); stroke-width: 2;
  stroke-linejoin: round; vector-effect: non-scaling-stroke; }
.negline { fill: none; stroke: var(--neg); stroke-width: 2;
  stroke-linejoin: round; vector-effect: non-scaling-stroke; }
.area { fill: var(--neg); fill-opacity: 0.16; stroke: none; }
.cross { stroke: var(--axis); stroke-width: 1; vector-effect: non-scaling-stroke; }
.dot { fill: var(--surface); stroke: var(--line); stroke-width: 2; }
#dot-dd { stroke: var(--neg); }
text { font-family: system-ui, -apple-system, sans-serif;
  font-variant-numeric: tabular-nums; }
.ytick { fill: var(--muted); font-size: 11px; text-anchor: end; }
.xtick { fill: var(--muted); font-size: 11px; text-anchor: middle; }
.tip {
  position: absolute; pointer-events: none; display: none;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 7px; padding: 5px 9px; font-size: 12px;
  font-variant-numeric: tabular-nums; white-space: nowrap;
  box-shadow: 0 2px 10px rgba(0,0,0,0.12); color: var(--ink);
}
.tip b { font-weight: 600; }
table { border-collapse: separate; border-spacing: 2px; width: 100%;
  font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
.months { font-size: 12.5px; min-width: 620px; }
.months th { color: var(--muted); font-weight: 500; font-size: 11px;
  text-transform: uppercase; letter-spacing: .04em; padding: 2px; }
.months th.yr, .months th.total { color: var(--ink2); }
.months td { text-align: right; padding: 5px 7px; border-radius: 4px;
  color: var(--ink); }
.months td.cell.pos {
  background: color-mix(in srgb, var(--pos-pole) calc(var(--a) * 100%),
                        var(--surface));
}
.months td.cell.neg {
  background: color-mix(in srgb, var(--neg-pole) calc(var(--a) * 100%),
                        var(--surface));
}
.months td.empty { background: var(--surface); }
.months td.total { font-weight: 600; background: var(--surface);
  border: 1px solid var(--border); }
.months th.yr { text-align: left; }
.legend { display: flex; align-items: center; gap: 8px; margin: 8px 0 0;
  color: var(--muted); font-size: 11.5px; }
.ramp { height: 8px; width: 160px; border-radius: 99px;
  background: linear-gradient(90deg,
    color-mix(in srgb, var(--neg-pole) 50%, var(--surface)),
    var(--surface),
    color-mix(in srgb, var(--pos-pole) 50%, var(--surface)));
  border: 1px solid var(--border); }
.stats { font-size: 13px; }
.stats th { text-align: left; font-weight: 400; color: var(--ink2);
  padding: 4px 8px; background: var(--surface); border-radius: 4px; }
.stats td { text-align: right; padding: 4px 10px; background: var(--surface);
  border-radius: 4px; font-weight: 500; width: 40%; }
footer { color: var(--muted); font-size: 12px; margin-top: 28px;
  border-top: 1px solid var(--border); padding-top: 12px; }
"""

SCRIPT = """
const P = __POINTS__;
function hook(id, fmt) {
  const svg = document.getElementById('svg-' + id);
  if (!svg) return;
  const panel = svg.parentElement;
  const tip = panel.querySelector('.tip');
  const cross = document.getElementById('cross-' + id);
  const dot = document.getElementById('dot-' + id);
  const n = P.x.length;

  function hide() {
    cross.style.display = dot.style.display = tip.style.display = 'none';
  }
  function move(ev) {
    const box = svg.getBoundingClientRect();
    const vx = (ev.clientX - box.left) / box.width * __W__;
    const span = __W__ - __PL__ - __PR__;
    let i = Math.round((vx - __PL__) / span * (n - 1));
    i = Math.max(0, Math.min(n - 1, i));
    const x = P.x[i], y = P[id][i];
    cross.setAttribute('x1', x); cross.setAttribute('x2', x);
    dot.setAttribute('cx', x); dot.setAttribute('cy', y);
    cross.style.display = dot.style.display = '';
    tip.innerHTML = '<b>' + fmt(P[id + 'v'][i]) + '</b> &middot; ' + P.d[i];
    tip.style.display = 'block';
    const px = x / __W__ * box.width;
    tip.style.left = Math.min(Math.max(px - tip.offsetWidth / 2, 4),
                              box.width - tip.offsetWidth - 4) + 'px';
    tip.style.top = '6px';
  }
  svg.addEventListener('mousemove', move);
  svg.addEventListener('mouseleave', hide);
  svg.addEventListener('touchmove', function (e) {
    move(e.touches[0]); e.preventDefault();
  });
}
hook('eq', v => v.toLocaleString(undefined, {maximumFractionDigits: 0}));
hook('dd', v => (v * 100).toFixed(1) + '%');
"""


def write_report(path, df, net, monthly, grid, dd, stats, src):
    equity = df['equity']
    eq_svg, xs, eq_ys = equity_svg(equity)
    dd_svg, _, dd_ys = drawdown_svg(dd['curve'])

    points = {
        'd': [d.strftime('%Y-%m-%d') for d in equity.index],
        'x': [round(x, 1) for x in xs],
        'eq': [round(y, 1) for y in eq_ys],
        'eqv': [round(float(v), 2) for v in equity.values],
        'dd': [round(y, 1) for y in dd_ys],
        'ddv': [round(float(v), 5) for v in dd['curve'].values],
    }
    script = (SCRIPT.replace('__POINTS__', json.dumps(points))
              .replace('__W__', str(CHART_W))
              .replace('__PL__', str(PAD_L))
              .replace('__PR__', str(PAD_R)))

    total_drag = (stats['cost_drag_sr'] + stats['financing_drag_sr']
                  + stats['fx_drag_sr'])

    tiles = ''.join([
        tile('terminal equity', f"{equity.iloc[-1]:,.0f}",
             f"from {CAPITAL:,.0f} {HOME_CURRENCY}"),
        tile('cagr', f"{stats['cagr']:.1%}",
             f"{stats['annual_vol']:.1%} realised vol"),
        tile('sharpe (net)', f"{stats['sharpe_net']:.2f}",
             f"{stats['sharpe_gross']:.2f} gross"),
        tile('max drawdown', f"{stats['max_drawdown']:.1%}",
             f"{stats['longest_drawdown_days']:.0f} days underwater"),
        tile('drag', f"{total_drag:.2f}",
             f"SR/yr &middot; {stats['cost_drag_sr']:.2f} cost (limit 0.13), "
             f"{stats['financing_drag_sr']:.2f} financing, "
             f"{stats['fx_drag_sr']:.2f} fx"),
            ])

    table = stats_table_html([
        ('window', f"{equity.index[0].date()} to {equity.index[-1].date()}"),
        ('rows', f"{stats['rows']:,.0f}"),
        ('years', f"{stats['years']:.2f}"),
        ('equity low / high',
         f"{stats['equity_min']:,.0f} / {stats['equity_max']:,.0f}"),
        ('mean annual return (arithmetic)',
         f"{stats['mean_annual_return']:.2%}"),
        ('annual vol (target '
         f"{VOL_TARGET:.0%})", f"{stats['annual_vol']:.2%}"),
        ('costs paid',
         f"{stats['costs_paid']:,.0f} ({stats['costs_per_year']:,.0f}/yr)"),
        ('financing paid', 
         f"{stats['financing_paid']:,.0f}  "
         f"({stats['financing_per_year']:,.0f}/yr)"),
        ('fx on the balance',
         f"{stats['fx_pnl']:,.0f} ({stats['fx_per_year']:,.0f}/yr)"),
        ('dividends received', 
         f"{stats['dividends']:,.0f} ({stats['dividends_per_year']:,.0f}/yr)"),
        ('days underwater', f"{stats['pct_days_underwater']:.1%}"),
        ('positive days', f"{stats['pct_positive_days']:.1%}"),
        ('worst day / month',
         f"{stats['worst_day']:.2%} / {stats['worst_month']:.2%}"),
        ('skew daily',
         f"{stats['skew_daily']:+.2f} &plusmn; {stats['skew_daily_se']:.2f}"),
        ('skew weekly',
         f"{stats['skew_weekly']:+.2f} &plusmn; {stats['skew_weekly_se']:.2f}"),
        ('skew monthly',
         f"{stats['skew_monthly']:+.2f} &plusmn; "
         f"{stats['skew_monthly_se']:.2f}"),
    ])

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Medowdog evaluation</title>
<style>{STYLE}</style></head>
<body><div class="wrap">
<h1>Medowdog evaluation</h1>
<p class="sub">{equity.index[0].date()} to {equity.index[-1].date()} &middot;
{stats['rows']:,.0f} rows &middot; capital {CAPITAL:,.0f} {HOME_CURRENCY},
compounding &middot; vol target {VOL_TARGET:.0%}</p>

<div class="tiles">{tiles}</div>

<h2>Equity</h2>
<p class="cap">Log scale. Dashed line is starting capital.</p>
<div class="panel">{eq_svg}<div class="tip"></div></div>

<h2>Drawdown</h2>
<p class="cap">Fraction below the running high-water mark, starting capital
included as a peak.</p>
<div class="panel">{dd_svg}<div class="tip"></div></div>

<h2>Monthly returns</h2>
<p class="cap">Percent. Shading is magnitude; the year column compounds the
months.</p>
{grid_table_html(grid)}
<div class="legend"><span>loss</span><span class="ramp"></span>
<span>gain</span></div>

<h2>Statistics</h2>
{table}

<footer>{src}</footer>
</div>
<script>{script}</script>
</body></html>
"""
    path.write_text(html, encoding='utf-8')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else PORTFOLIO_DIR / 'portfolio.csv'
    df = load_portfolio(path)

    equity = df['equity']
    years = len(df) / BUSINESS_DAYS

    net = returns_on_capital(df['net_pnl'], equity, CAPITAL)
    gross = returns_on_capital(df['gross_pnl'], equity, CAPITAL)

    sr_net = sharpe(net)
    sr_gross = sharpe(gross)
    cagr = (equity.iloc[-1] / CAPITAL) ** (1 / years) - 1

    dd = drawdown_stats(equity, CAPITAL)

    weekly = period_returns(equity, CAPITAL, 'W')
    monthly = period_returns(equity, CAPITAL, 'M')
    skews = [skew_with_se(net, 'daily'),
            skew_with_se(weekly, 'weekly'),
            skew_with_se(monthly, 'monthly')]

    costs_paid = df['cost'].sum()
    gross_profit = df['gross_pnl'].sum()
    income = df['dividend'].sum() if 'dividend' in df.columns else 0.0
    financing_col = df['financing'] if 'financing' in df.columns else pd.Series(0.0, index=df.index)
    fx_col = (df['fx_pnl'] if 'fx_pnl' in df.columns else pd.Series(0.0, index=df.index))
    capped = (df['capped'].astype(bool) if 'capped' in df.columns
              else pd.Series(False, index=df.index))
    financing = financing_col.sum()
    moved = fx_col.sum()

    # gross -> after trading costs -> after financing -> after fx, so each drag is the 
    # Sharpe given up at that step and the three add to sharpe_gross - sharpe_net
    after_costs = returns_on_capital(df['gross_pnl'] - df['cost'], equity, CAPITAL)
    after_financing = returns_on_capital(df['gross_pnl'] - df['cost'] - financing_col, equity, CAPITAL)
    
    sr_after_costs = sharpe(after_costs)
    sr_after_financing = sharpe(after_financing)
    cost_drag = sr_gross - sr_after_costs
    financing_drag = sr_after_costs - sr_after_financing
    fx_drag = sr_after_financing - sr_net

    print(f"{path}")
    print(f"window   {equity.index[0].date()} -> {equity.index[-1].date()}  "
          f"({len(df)} rows, {years:.2f} years)\n")

    print("return")
    print(f"  equity          {CAPITAL:,.0f} -> {equity.iloc[-1]:,.0f} "
          f"{HOME_CURRENCY}")
    print(f"  cagr            {cagr:>8.2%}")
    print(f"  mean annual     {net.mean() * BUSINESS_DAYS:>8.2%}   "
          f"(arithmetic, not compounded)")
    print(f"  annual vol      {net.std() * ANNUALISE:>8.2%}   "
          f"(target {VOL_TARGET:.0%})")

    print("\nrisk-adjusted")
    print(f"  sharpe net      {sr_net:>8.2f}")
    print(f"  sharpe gross    {sr_gross:>8.2f}")
    print(f"  cost drag       {cost_drag:>8.2f} SR/yr   "
          f"(speed limit 0.13)")
    print(f"  financing drag  {financing_drag:>8.2f} SR/yr")
    print(f"  fx drag         {fx_drag:>8.2f} SR/yr")
    share = (f", {costs_paid / gross_profit:.1%} of gross profit"
             if gross_profit > 0 else '')
    print(f"  costs paid      {costs_paid:>8,.0f} {HOME_CURRENCY}   "
          f"({costs_paid / years:,.0f}/yr{share})")

    if financing:
        print(f"  financing       {financing:>8,.0f} {HOME_CURRENCY}   "
              f"({financing / years:,.0f}/yr)")

    if moved:
        print(f"  fx              {moved:>8,.0f} {HOME_CURRENCY}  "
              f"({moved / years:,.0f}/yr on the whole balance)") 

    if income:
        part = (f", {income / gross_profit:.1%} of gross profit"
                if gross_profit > 0 else '')
        print(f"  dividends       {income:>8,.0f} {HOME_CURRENCY}   "
              f"({income / years:,.0f}/yr{part})")

    print("\ndrawdown")
    print(f"  max             {dd['max_drawdown']:>8.2%}   "
          f"peak {dd['fell_from'].date()} -> trough "
          f"{dd['trough_on'].date()}")
    recovered = dd['recovered_on']
    back = recovered.date().isoformat() if recovered is not None else 'not yet'
    print(f"  recovered on    {back:>10}   "
          f"longest spell {dd['longest_drawdown_days']} days")
    print(f"  days underwater {dd['pct_days_underwater']:>8.1%}")

    print("\nskew")
    for s in skews:
        verdict = ('' if not np.isfinite(s['skew'])
                   else '  (within 1 SE of zero)'
                   if abs(s['skew']) < s['se'] else '')
        print(f"  {s['horizon']:8s} n={s['n']:>5d}   "
              f"{s['skew']:>6.2f} +/- {s['se']:.2f}{verdict}")

    print("\ntails")
    print(f"  worst day       {net.min():>8.2%}   "
          f"best day {net.max():.2%}")
    print(f"  worst month     {monthly.min():>8.2%}   "
          f"best month {monthly.max():.2%}")
    print(f"  positive days   {(net > 0).mean():>8.1%}")

    grid = monthly_grid(monthly)
    print("\nmonthly returns (%)")
    print(grid.round(1).to_string(na_rep=''))

    stats = {
        'rows': len(df),
        'years': years,
        'equity_start': CAPITAL,
        'equity_end': equity.iloc[-1],
        'equity_min': equity.min(),
        'equity_max': equity.max(),
        'cagr': cagr,
        'mean_annual_return': net.mean() * BUSINESS_DAYS,
        'annual_vol': net.std() * ANNUALISE,
        'sharpe_net': sr_net,
        'sharpe_gross': sr_gross,
        'cost_drag_sr': cost_drag,
        'financing_drag_sr': financing_drag,
        'financing_paid': financing,
        'financing_per_year': financing / years,
        'fx_drag_sr': fx_drag,
        'fx_pnl': moved,
        'fx_per_year': moved / years,
        'capped_days': int(capped.sum()),
        'capped_fraction': float(capped.mean()),
        'costs_paid': costs_paid,
        'costs_per_year': costs_paid / years,
        'dividends': income,
        'dividends_per_year': income / years,
        'max_drawdown': dd['max_drawdown'],
        'longest_drawdown_days': dd['longest_drawdown_days'],
        'pct_days_underwater': dd['pct_days_underwater'],
        'worst_day': net.min(),
        'worst_month': monthly.min(),
        'pct_positive_days': float((net > 0).mean()),
    }
    for s in skews:
        stats[f"skew_{s['horizon']}"] = s['skew']
        stats[f"skew_{s['horizon']}_se"] = s['se']

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(path).stem
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # one set of outputs per source file, so the benchmark and the system
    # never overwrite each other
    archived = sum(archive_previous(PORTFOLIO_DIR, f'{name}_{stem}_*.{ext}')
                   for name, ext in (('stats', 'csv'),
                                     ('monthly_returns', 'csv'),
                                     ('drawdown', 'csv'),
                                     ('report', 'html')))

    stats_path = PORTFOLIO_DIR / f'stats_{stem}_{stamp}.csv'
    pd.Series(stats, name='value').rename_axis('statistic').to_csv(stats_path)
    grid.to_csv(PORTFOLIO_DIR / f'monthly_returns_{stem}_{stamp}.csv')
    dd['curve'].rename('drawdown').to_csv(
        PORTFOLIO_DIR / f'drawdown_{stem}_{stamp}.csv')
    report_path = PORTFOLIO_DIR / f'report_{stem}_{stamp}.html'
    write_report(report_path, df, net, monthly, grid, dd, stats,
                 f"generated from {path}")

    note = f"  ({archived} archived)" if archived else ""
    print(f"\nwritten to {PORTFOLIO_DIR}/ ({stats_path.name}, "
          f"{report_path.name} and 2 more){note}")


if __name__ == '__main__':
    main()
