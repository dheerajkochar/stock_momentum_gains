"""
PHASE 4B — BULLISH SIGNAL WIN RATE TEST
Test: if all bullish conditions met, what is actual win rate and return?

Bullish conditions from logistic regression:
  1. price_vs_ma20 < 0     (not overextended above MA)
  2. vol_event < 5x        (no climax volume)
  3. vol_5d_ratio > 1.0    (volume building before event)
  4. d52w_low > 30         (far above 52w low = has base)
  5. pre_60d > 0           (longer trend positive)
  6. atr_ratio < 1.2       (volatility not expanded)
"""

import pandas as pd
import numpy as np

DATA_DIR = "data"
df = pd.read_csv(f"{DATA_DIR}/event_days_typed.csv", parse_dates=["date"])
print(f"Total events: {len(df)}\n")

# ─────────────────────────────────────────────
# DEFINE BULLISH CONDITIONS
# ─────────────────────────────────────────────

conditions = {
    "price_vs_ma20 < 0":    df["price_vs_ma20"] < 0,
    "vol_event < 5x":       df["vol_event"] < 3,
    "vol_5d_ratio > 1.0":   df["vol_5d_ratio"] > 1.0,
    "d52w_low > 30%":       df["d52w_low"] > 30,
    "pre_60d > 0%":         df["pre_60d"] > 0,

}

# ─────────────────────────────────────────────
# SINGLE CONDITION IMPACT
# ─────────────────────────────────────────────

print("=" * 65)
print("  SINGLE CONDITION IMPACT (fwd_5d >= +3%)")
print("=" * 65)
print(f"\n  {'Condition':<28} {'N':>5}  "
      f"{'Win%':>6} {'Loss%':>6}  "
      f"{'Avg 5d':>8} {'Avg 1d':>8}")
print("  " + "-"*60)

# Baseline
base_win  = (df.fwd_5d >= 5).mean()*100
base_loss = (df.fwd_5d <= -1).mean()*100
base_avg5 = df.fwd_5d.mean()
base_avg1 = df.fwd_1d.mean()
print(f"  {'BASELINE (all)':<28} {len(df):>5}  "
      f"{base_win:>6.1f} {base_loss:>6.1f}  "
      f"{base_avg5:>8.2f}% {base_avg1:>8.2f}%")
print()

for label, mask in conditions.items():
    sub = df[mask]
    if len(sub) < 20: continue
    win  = (sub.fwd_5d >= 3).mean()*100
    loss = (sub.fwd_5d <= -1).mean()*100
    avg5 = sub.fwd_5d.mean()
    avg1 = sub.fwd_1d.mean()
    delta = avg5 - base_avg5
    marker = " ◄" if abs(delta) > 0.3 else ""
    print(f"  {label:<28} {len(sub):>5}  "
          f"{win:>6.1f} {loss:>6.1f}  "
          f"{avg5:>8.2f}% {avg1:>8.2f}%  "
          f"[Δ{delta:+.2f}%]{marker}")

# ─────────────────────────────────────────────
# PROGRESSIVE SIGNAL STACKING
# How does win rate change as we add conditions?
# ─────────────────────────────────────────────

print(f"\n{'='*65}")
print("  PROGRESSIVE SIGNAL STACKING")
print(f"{'='*65}")
print(f"\n  {'Conditions Met':<35} {'N':>5}  "
      f"{'Win%':>6} {'Loss%':>6}  "
      f"{'Avg 5d':>8} {'Avg 1d':>8}")
print("  " + "-"*65)

cond_list = list(conditions.items())
mask_cumulative = pd.Series([True]*len(df))

for i in range(len(cond_list)):
    label, mask = cond_list[i]
    mask_cumulative = mask_cumulative & mask
    sub = df[mask_cumulative]
    if len(sub) < 10: continue
    win  = (sub.fwd_5d >= 3).mean()*100
    loss = (sub.fwd_5d <= -1).mean()*100
    avg5 = sub.fwd_5d.mean()
    avg1 = sub.fwd_1d.mean()
    conds_str = f"{i+1} cond: ...+{label.split()[0]}"
    print(f"  {conds_str:<35} {len(sub):>5}  "
          f"{win:>6.1f} {loss:>6.1f}  "
          f"{avg5:>8.2f}% {avg1:>8.2f}%")

# ─────────────────────────────────────────────
# ALL CONDITIONS MET — DEEP ANALYSIS
# ─────────────────────────────────────────────

all_bull = pd.Series([True]*len(df))
for _, mask in conditions.items():
    all_bull = all_bull & mask

bull_df = df[all_bull]

print(f"\n{'='*65}")
print(f"  ALL BULLISH CONDITIONS MET: n={len(bull_df)}")
print(f"{'='*65}")

print(f"\n── Return Distribution ─────────────────────")
print(f"  fwd_1d: mean={bull_df.fwd_1d.mean():.2f}%  "
      f"median={bull_df.fwd_1d.median():.2f}%  "
      f"std={bull_df.fwd_1d.std():.2f}%")
print(f"  fwd_3d: mean={bull_df.fwd_3d.mean():.2f}%  "
      f"median={bull_df.fwd_3d.median():.2f}%  "
      f"std={bull_df.fwd_3d.std():.2f}%")
print(f"  fwd_5d: mean={bull_df.fwd_5d.mean():.2f}%  "
      f"median={bull_df.fwd_5d.median():.2f}%  "
      f"std={bull_df.fwd_5d.std():.2f}%")

print(f"\n── Win/Loss Rates ──────────────────────────")
for horizon, col in [("Day+1","fwd_1d"),
                     ("Day+3","fwd_3d"),
                     ("Day+5","fwd_5d")]:
    win  = (bull_df[col] >= 3).mean()*100
    loss = (bull_df[col] <= -1).mean()*100
    flat = 100 - win - loss
    avg  = bull_df[col].mean()
    print(f"  {horizon}: win={win:.1f}%  "
          f"loss={loss:.1f}%  flat={flat:.1f}%  "
          f"avg={avg:.2f}%")

print(f"\n── vs Baseline ─────────────────────────────")
print(f"  {'':25} {'Signal':>8} {'Baseline':>9} {'Edge':>7}")
print(f"  {'─'*50}")
for horizon, col in [("Day+1 win%","fwd_1d"),
                     ("Day+3 win%","fwd_3d"),
                     ("Day+5 win%","fwd_5d")]:
    sig  = (bull_df[col] >= 3).mean()*100
    base = (df[col] >= 3).mean()*100
    edge = sig - base
    print(f"  {horizon:<25} {sig:>8.1f} {base:>9.1f} "
          f"{edge:>+7.1f}pp")

for horizon, col in [("Day+5 avg return","fwd_5d"),
                     ("Day+3 avg return","fwd_3d"),
                     ("Day+1 avg return","fwd_1d")]:
    sig  = bull_df[col].mean()
    base = df[col].mean()
    edge = sig - base
    print(f"  {horizon:<25} {sig:>8.2f}% {base:>8.2f}% "
          f"{edge:>+7.2f}%")

print(f"\n── By Bucket ───────────────────────────────")
for bucket in ["7-10%","10-15%","15%+"]:
    sub = bull_df[bull_df.bucket==bucket]
    if len(sub) < 5: continue
    win  = (sub.fwd_5d >= 3).mean()*100
    loss = (sub.fwd_5d <= -1).mean()*100
    avg5 = sub.fwd_5d.mean()
    print(f"  {bucket}: n={len(sub):3d} | "
          f"win={win:.1f}%  loss={loss:.1f}%  "
          f"avg_5d={avg5:.2f}%")

print(f"\n── By Event Type ───────────────────────────")
for etype in ["A_ColdBreakout","B_MomentumRun",
              "C_Shock","D_DeadCat","E_Mixed"]:
    sub = bull_df[bull_df.event_type==etype]
    if len(sub) < 5: continue
    win  = (sub.fwd_5d >= 3).mean()*100
    loss = (sub.fwd_5d <= -1).mean()*100
    avg5 = sub.fwd_5d.mean()
    print(f"  {etype:<20}: n={len(sub):3d} | "
          f"win={win:.1f}%  loss={loss:.1f}%  "
          f"avg_5d={avg5:.2f}%")

# ─────────────────────────────────────────────
# THRESHOLD SENSITIVITY
# What if we tighten conditions?
# ─────────────────────────────────────────────

print(f"\n{'='*65}")
print("  THRESHOLD SENSITIVITY")
print(f"{'='*65}")

tests = [
    ("Base signal (6 conds)",
     (df.price_vs_ma20 < 0) &
     (df.vol_event < 5) &
     (df.vol_5d_ratio > 1.0) &
     (df.d52w_low > 30) &
     (df.pre_60d > 0) &
     (df.atr_ratio < 1.2)),

    ("Tighter: price_vs_ma20 < -2%",
     (df.price_vs_ma20 < -2) &
     (df.vol_event < 5) &
     (df.vol_5d_ratio > 1.0) &
     (df.d52w_low > 30) &
     (df.pre_60d > 0) &
     (df.atr_ratio < 1.2)),

    ("Tighter: vol_event < 3x",
     (df.price_vs_ma20 < 0) &
     (df.vol_event < 3) &
     (df.vol_5d_ratio > 1.0) &
     (df.d52w_low > 30) &
     (df.pre_60d > 0) &
     (df.atr_ratio < 1.2)),

    ("Tighter: vol_5d_ratio > 1.2",
     (df.price_vs_ma20 < 0) &
     (df.vol_event < 5) &
     (df.vol_5d_ratio > 1.2) &
     (df.d52w_low > 30) &
     (df.pre_60d > 0) &
     (df.atr_ratio < 1.2)),

    ("Strictest: all tightened",
     (df.price_vs_ma20 < -2) &
     (df.vol_event < 3) &
     (df.vol_5d_ratio > 1.2) &
     (df.d52w_low > 40) &
     (df.pre_60d > 5) &
     (df.atr_ratio < 1.1)),

    ("Add: 7-10% bucket only",
     (df.price_vs_ma20 < 0) &
     (df.vol_event < 5) &
     (df.vol_5d_ratio > 1.0) &
     (df.d52w_low > 30) &
     (df.pre_60d > 0) &
     (df.atr_ratio < 1.2) &
     (df.bucket == "7-10%")),

    ("Add: ColdBreakout type only",
     (df.price_vs_ma20 < 0) &
     (df.vol_event < 5) &
     (df.vol_5d_ratio > 1.0) &
     (df.d52w_low > 30) &
     (df.pre_60d > 0) &
     (df.atr_ratio < 1.2) &
     (df.event_type == "A_ColdBreakout")),
]

print(f"\n  {'Filter':<35} {'N':>5}  "
      f"{'Win%':>6} {'Loss%':>6}  "
      f"{'Avg 5d':>8} {'Edge':>7}")
print("  " + "-"*70)

base_avg = df.fwd_5d.mean()
for label, mask in tests:
    sub = df[mask]
    if len(sub) < 10: continue
    win  = (sub.fwd_5d >= 3).mean()*100
    loss = (sub.fwd_5d <= -1).mean()*100
    avg5 = sub.fwd_5d.mean()
    edge = avg5 - base_avg
    print(f"  {label:<35} {len(sub):>5}  "
          f"{win:>6.1f} {loss:>6.1f}  "
          f"{avg5:>8.2f}% {edge:>+7.2f}%")

print("\n✅ Done")