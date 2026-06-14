import pandas as pd
import numpy as np

d1    = pd.read_csv("data/phase5_day1_intraday.csv", parse_dates=["date"])
micro = pd.read_csv("data/event_days_microstructure.csv", parse_dates=["date"])

# Drop the fake target_hit/stop_hit from micro (close-based, wrong)
micro = micro.drop(columns=["target_hit","stop_hit"], errors="ignore")

# Also fix duplicate atr_ratio columns
micro = micro.drop(columns=["atr_ratio_x","atr_ratio_y"], errors="ignore")

# Merge with real intraday d1 data
df = micro.merge(
    d1[["date","ticker","open_ret","high_ret",
        "low_ret","close_ret","target_hit","stop_hit","outcome"]],
    on=["date","ticker"], how="left"
)

# Market direction
nifty = pd.read_csv("data/NIFTY50.csv", index_col=0, dtype=str)
nifty.columns = [str(c).split()[0] for c in nifty.columns]
mask = nifty.index.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
nifty = nifty[mask]
nifty.index = pd.to_datetime(nifty.index)
nifty = nifty.apply(pd.to_numeric, errors="coerce")
nifty_ret = nifty["Close"].pct_change()
df = df.merge(nifty_ret.rename("market_ret"),
              left_on="date", right_index=True, how="left")
df["market_up"] = (df["market_ret"] > 0).astype(int)

# Win = target hit before stop
df["win"] = df["outcome"].str.contains("target", na=False).astype(int)

clv_p90 = df["clv"].quantile(0.90)

print(f"CLV p90: {clv_p90:.3f}")
print(f"\nBaseline (all {len(df)} events):")
print(f"  Target hit (High>=+3%): {df.target_hit.mean()*100:.1f}%")
print(f"  Open avg:               {df.open_ret.mean():.2f}%")
print(f"  Win rate:               {df.win.mean()*100:.1f}%")

print(f"\n{'Signal':<50} {'N':>5} {'Tgt%':>7} {'Open%':>8} {'Win%':>7}")
print("-"*75)

combos = [
    ("CLV > p90",
     df.clv > clv_p90),
    ("CLV > p90 + atr<2",
     (df.clv > clv_p90) & (df.atr_ratio_event < 2.0)),
    ("CLV > p90 + atr<1.5",
     (df.clv > clv_p90) & (df.atr_ratio_event < 1.5)),
    ("CLV > p90 + atr<2 + 7-10%",
     (df.clv > clv_p90) & (df.atr_ratio_event < 2.0) &
     (df.bucket == "7-10%")),
    ("CLV > p90 + atr<1.5 + 7-10%",
     (df.clv > clv_p90) & (df.atr_ratio_event < 1.5) &
     (df.bucket == "7-10%")),
    ("CLV > p90 + atr<2 + vol<5x",
     (df.clv > clv_p90) & (df.atr_ratio_event < 2.0) &
     (df.vol_ratio_event < 5)),
    ("CLV > p90 + atr<1.5 + vol<5x + 7-10%",
     (df.clv > clv_p90) & (df.atr_ratio_event < 1.5) &
     (df.vol_ratio_event < 5) & (df.bucket == "7-10%")),
    ("CLV > p90 + atr<2 + market NOT up",
     (df.clv > clv_p90) & (df.atr_ratio_event < 2.0) &
     (df.market_up == 0)),
    ("CLV > p90 + atr<2 + upper_wick<0.05",
     (df.clv > clv_p90) & (df.atr_ratio_event < 2.0) &
     (df.upper_wick_ratio < 0.05)),
    ("day_ret>p90 + CLV>p90",
     (df.day_return > df.day_return.quantile(0.90)) &
     (df.clv > clv_p90)),
    ("day_ret>p90 + CLV>p90 + atr<2",
     (df.day_return > df.day_return.quantile(0.90)) &
     (df.clv > clv_p90) & (df.atr_ratio_event < 2.0)),
]

for label, mask in combos:
    sub = df[mask].dropna(subset=["target_hit","open_ret","win"])
    if len(sub) < 20: continue
    tgt   = sub.target_hit.mean()*100
    open_r = sub.open_ret.mean()
    win   = sub.win.mean()*100
    lift  = tgt - df.target_hit.mean()*100
    marker = " ◄◄" if lift > 10 else (" ◄" if lift > 5 else "")
    print(f"{label:<50} {len(sub):>5} "
          f"{tgt:>7.1f}% {open_r:>8.2f}% {win:>7.1f}%{marker}")

# ── Best signal deep dive ──
best_mask = (
    (df.clv > clv_p90) &
    (df.atr_ratio_event < 1.5) &
    (df.bucket == "7-10%")
)
best = df[best_mask].dropna(subset=["target_hit","open_ret","win"])

print(f"\n{'='*60}")
print(f"BEST SIGNAL: CLV>p90 + atr<1.5 + 7-10%  n={len(best)}")
print(f"{'='*60}")
print(f"  Target hit (intraday High>=+3%): {best.target_hit.mean()*100:.1f}%")
print(f"  Open avg return:                 {best.open_ret.mean():.2f}%")
print(f"  Win rate (target before stop):   {best.win.mean()*100:.1f}%")
print(f"  High avg return:                 {best.high_ret.mean():.2f}%")
print(f"  Close avg return:                {best.close_ret.mean():.2f}%")

print(f"\n  Gap breakdown:")
print(f"  Opened >=+3% (sell at open):   "
      f"{(best.open_ret>=3).mean()*100:.1f}%  "
      f"avg={best[best.open_ret>=3].open_ret.mean():.2f}%")
print(f"  Opened 0-3% (watch intraday):  "
      f"{((best.open_ret>=0)&(best.open_ret<3)).mean()*100:.1f}%")
print(f"  Opened negative (stop loss):   "
      f"{(best.open_ret<0).mean()*100:.1f}%  "
      f"avg={best[best.open_ret<0].open_ret.mean():.2f}%")

print(f"\n  Outcome breakdown:")
print(best.outcome.value_counts().to_string())

print(f"\n  Year by year:")
print(f"  {'Year':<6} {'N':>5} {'Tgt%':>7} "
      f"{'Open%':>8} {'Win%':>7}")
for yr in sorted(best.date.dt.year.unique()):
    sub = best[best.date.dt.year==yr]
    print(f"  {yr:<6} {len(sub):>5} "
          f"{sub.target_hit.mean()*100:>7.1f}% "
          f"{sub.open_ret.mean():>8.2f}% "
          f"{sub.win.mean()*100:>7.1f}%")