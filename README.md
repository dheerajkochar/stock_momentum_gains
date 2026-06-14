# Price Shock Anatomy — Nifty 200 Momentum Study

> A systematic analysis of 3,195 single-day 7%+ events in Indian large-cap 
> equities, resulting in a rules-based overnight trading signal with 77% win 
> rate across 7 years.

## What This Is

A complete quantitative research project studying what happens after a Nifty 200 
stock gains 7%+ in a single day. Built entirely on public OHLCV data — no 
earnings feeds, no news APIs, no Bloomberg terminal required.

## Key Findings

| Finding | Result |
|---|---|
| Unfiltered base rate (7–10% events) | 22.1% next-day win rate |
| Conventional research misses | Day+1 intraday high averages +3.69% vs +0.39% close |
| CLV > 0.942 alone | Lifts target hit rate to 72.3% |
| Final 3-signal strategy | 77% win rate, +24.9% annual return |
| Years positive (Exit 1 or 2) | 8 out of 8 |


## The Signal
Entry Conditions (ALL required):

Event day return  : 7% – 10%
CLV               : > 0.942  [(Close-Low)/(High-Low)]
ATR Ratio         : < 2.0    [DayRange / ATR_20]

Market Filter : Skip if Nifty 50 fell > 3% on event day

Position      : ₹50,000 per trade, max 3/day, ranked by CLV

Exit          : Sell at open if gap ≥ +3%

Limit at +3% if gap < +3%

Stop at open if gap ≤ -2%



## Data & Stack

- **Universe** : Nifty 200 (191 active tickers)
- **Period**   : March 2018 – March 2025
- **Source**   : Yahoo Finance via `yfinance`
- **Stack**    : Python, pandas, numpy, scikit-learn
- **Files**    : `event_days_typed.csv`, `event_days_intraday.csv`, 
                 `master_open/high/low/close.csv`


## Limitations

- 213 total trades — robust overall, thin in individual years
- Gap risk: stop orders execute at open price, not at -2% level
- No earnings filter — some events are post-earnings gaps
- Survivorship bias: 2025 Nifty 200 constituent list used throughout

## Disclaimer

Research only. Not financial advice. Past performance does not 
guarantee future results.
