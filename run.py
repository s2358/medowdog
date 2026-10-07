"""
Run the daily pipeline.

    python run.py

Steps in order: fx rate, download prices, volatility, ewmac, correlation, position sizing. The cost 
script is deliberately not here - it needs US market hours and runs from its own cron entry.


"""
