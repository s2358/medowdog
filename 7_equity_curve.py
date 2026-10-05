"""
Portfolio equity curve and performance statistics.

    python equity_curve.py data/portfolio/positions_*.csv

P&L on day t comes from the position held at close of day t-1, so the forecast that set that position had no knowledge of day t's price.

Capital is constant across the whole window (positions were sized against a fixed CAPITAL), 
"""
