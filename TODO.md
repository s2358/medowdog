# Medowdog — TODO

## Current approach (revised 2026-08-28)

Side project scope: download daily bars for all four tickers to CSV, then
analyse. The SQLite plan below is shelved, not deleted — revisit if the CSV
workflow outgrows itself.

Working script (in progress, built line by line):
- [x] Connect to Gateway (4001, readonly), qualify all 4 contracts by conId,
      assert symbol matches.
- [x] TRADES request, 25 Y / 1 day, one request per ticker. Verified counts:
      IAU 5428 (2005-01-28), EEM 5881 (2003-04-11), TLT 5783 (2002-08-19,
      via ARCA), DBC 5172 (2006-02-03).
- [ ] ADJUSTED_LAST request per ticker — also settles the entitlement question.
- [ ] Merge on date -> date, open, high, low, close, volume, adj_close;
      check the two series have identical date sets; write <symbol>.csv.
- [ ] Analyse: XNYS calendar diff (locates TLT's missing days), structural
      checks, TRADES vs adj_close ratio to see dividend adjustment factors.

Empirical notes that override earlier assumptions:
- 25 Y single request works but can stall Gateway: use timeout=120 on every
  reqHistoricalData and ib.sleep(2) between tickers (ib.sleep, never
  time.sleep — time.sleep blocks the event loop).
- DBC's real start is 2006-02-03 (fund inception). The 2006-08-10 start seen
  earlier was an MCP-connector artifact. Common window for all four:
  **2006-02-03** onward.
- TLT: SMART returns only 2016-02-03 onward (2657 bars); ARCA returns
  2002-08-19 onward (5783 bars). Always pass exchange="ARCA" for TLT.

## Open questions

- [ ] Does ADJUSTED_LAST work under the current entitlement (US Real-Time
      Non Consolidated Streaming Quotes, no consolidated tape)? Being tested
      by the next script step.
- [ ] Locate TLT's missing days. ~24y ≈ 6,050 trading days vs 5,783 returned,
      ~265 missing. Known Feb–Sep 2003 gap ≈ 165; find the rest via calendar
      diff during analysis.
- [ ] Dividend sourcing for EEM/TLT/DBC — only needed if we later move off
      storing adj_close directly (e.g. back to the append-only DB design).
      Candidate: derive dividends from the TRADES/ADJUSTED_LAST ratio.
- [ ] NOT FINALISED: append-only storage rule and whether an adjusted close
      is stored or computed at read time. Moot while on CSV (full rewrite
      each run); decide if/when returning to the DB design.

## Shelved: SQLite design (Module 0)

Decisions made but not currently in use:
- schema.sql — five tables (instrument, price, price_flag, corporate_action,
  metadata), STRICT, CHECK constraints, dates as ISO TEXT,
  PRAGMA foreign_keys=ON per connection.
- price_flag as separate table with observed_value; flag, never repair/drop.
- Structural checks over statistical; |return| > 25% as the one loose
  statistical check. Calendar gaps reported, not stored.
- db committed to git, superseded copies to archive/ (gitignored),
  named by snapshot date.

## Dropped

- ~~Add EEM, TLT, DBC (was under "Later" in the SQLite plan).~~ Superseded
  2026-08-28: the CSV script covers all four tickers from the start.

## Later

- [ ] IDM correlation window decision given ragged start dates
      (2002/2003/2005/2006 starts vs common window 2006-02-03).
