"""
PARAMETER VALIDATION — Final 8 parameters
1. Check score_imom < p10 trap (is it reversal or real?)
2. Validate each parameter independently
3. Test all 8 stacked together
4. Find the minimum viable combination
"""

import pandas as pd
import numpy as np

DATA_DIR = "data"

s1 = pd.read_csv(f"event_days_typed.csv",    parse_dates=["date"])
s2 = pd.read_csv(f"event_days_intraday.csv", parse_dates=["date"])


df = pd.merge(s1, s2[["date","ticker","clv","imom","gap",
                        "atr_ratio","d52w","pre_rng",
                        "score_clv","score_imom","score_gap",
                        "score_atr","score_d52w","score_prng",
                        "intraday_score","structure"]],
              on=["date","ticker"], how="left", suffixes=("","_s2"))

WIN    = 3.0
LOSS   = -1.0
BASE_WIN  = (df.fwd_1d >= WIN).mean() * 100
BASE_LOSS = (df.fwd_1d <= LOSS).mean() * 100
BASE_AVG  = df.fwd_1d.mean()

def show(label, sub, base=df):
    n    = len(sub)
    if n < 10: return
    win  = (sub.fwd_1d >= WIN).mean()  * 100
    loss = (sub.fwd_1d <= LOSS).mean() * 100
    avg1 = sub.fwd_1d.mean()
    avg5 = sub.fwd_5d.mean()
    lift = win - BASE_WIN
    print(f"  {label:<42} n={n:>4}  win={win:>5.1f}%  "
          f"loss={loss:>5.1f}%  avg1d={avg1:>+5.2f}%  lift={lift:>+5.1f}pp")

# ═══════════════════════════════════════════════════════════════
# 1. SCORE_IMOM TRAP CHECK
# ═══════════════════════════════════════════════════════════════
print("=" * 72)
print("  SCORE_IMOM TRAP CHECK")
print("=" * 72)

p10_imom = df.score_imom.quantile(0.10)
print(f"\n  score_imom p10 = {p10_imom}")
print(f"  What do stocks with score_imom < {p10_imom} look like?\n")

trap = df[df.score_imom < p10_imom].copy()

# Check pre_60d distribution — are these beaten-down stocks?
print(f"  pre_60d median  : trap={trap.pre_60d.median():.1f}%  "
      f"all={df.pre_60d.median():.1f}%")
print(f"  pre_20d median  : trap={trap.pre_20d.median():.1f}%  "
      f"all={df.pre_20d.median():.1f}%")
print(f"  d52w_low median : trap={trap.d52w_low.median():.1f}  "
      f"all={df.d52w_low.median():.1f}")
print(f"  clv median      : trap={trap.clv.median():.3f}  "
      f"all={df.clv.median():.3f}")
print(f"  event_type dist :")
print(trap.event_type.value_counts().to_string())

print(f"\n  Win rate: trap={( trap.fwd_1d >= WIN).mean()*100:.1f}%  "
      f"non-trap={(df[df.score_imom >= p10_imom].fwd_1d >= WIN).mean()*100:.1f}%")

# Split trap into momentum vs reversal
trap_mom = trap[trap.pre_60d > 0]
trap_rev = trap[trap.pre_60d < 0]
print(f"\n  Within score_imom < p10:")
show("  pre_60d > 0  (momentum stocks)",  trap_mom)
show("  pre_60d < 0  (reversal/beaten down)", trap_rev)

print(f"\n  Verdict: ", end="")
if trap_rev.fwd_1d.mean() > trap_mom.fwd_1d.mean() + 0.5:
    print("TRAP CONFIRMED — win rate driven by beaten-down reversals")
    print("  → Do NOT use score_imom < p10 as a momentum signal")
    print("  → Could use it as a separate reversal strategy")
else:
    print("SIGNAL VALID — works across both momentum and reversal")
    print("  → Safe to include in your parameter set")

# ═══════════════════════════════════════════════════════════════
# 2. VALIDATE EACH OF THE 8 PARAMETERS INDEPENDENTLY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print("  INDIVIDUAL PARAMETER VALIDATION")
print(f"  Baseline: win={BASE_WIN:.1f}%  loss={BASE_LOSS:.1f}%  avg1d={BASE_AVG:.2f}%")
print(f"{'=' * 72}\n")

# Thresholds from threshold scan
params_8 = {
    "score_imom < p10":      df.score_imom  < df.score_imom.quantile(0.10),
    "clv > p90":             df.clv         > df.clv.quantile(0.90),
    "d52w_high > p85":       df.d52w_high   > df.d52w_high.quantile(0.85),
    "pre_range_5d > p90":    df.pre_range_5d > df.pre_range_5d.quantile(0.90),
    "vol_event < p10":       df.vol_event   < df.vol_event.quantile(0.10),
    "pre_5d > p90":          df.pre_5d      > df.pre_5d.quantile(0.90),
    "vol_5d_ratio > p90":    df.vol_5d_ratio > df.vol_5d_ratio.quantile(0.90),
    "day_return > p90":      df.day_return  > df.day_return.quantile(0.90),
}

# Also add the EXCLUDING score_imom version
params_7 = {k: v for k, v in params_8.items() if "score_imom" not in k}

for label, mask in params_8.items():
    show(label, df[mask])

# ═══════════════════════════════════════════════════════════════
# 3. STACKING — progressive combination
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print("  PROGRESSIVE STACKING — 8 parameters (sorted by lift)")
print(f"{'=' * 72}\n")

# Sort by single-param lift to stack in best order
param_lifts = []
for label, mask in params_8.items():
    sub = df[mask]
    if len(sub) < 10: continue
    win = (sub.fwd_1d >= WIN).mean() * 100
    param_lifts.append((win - BASE_WIN, label, mask))
param_lifts.sort(reverse=True)

cumulative = pd.Series([True] * len(df))
for lift, label, mask in param_lifts:
    cumulative = cumulative & mask
    show(f"+{label}", df[cumulative])

# ═══════════════════════════════════════════════════════════════
# 4. SAME STACKING WITHOUT SCORE_IMOM
# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 72}")
print("  PROGRESSIVE STACKING — 7 parameters (score_imom excluded)")
print(f"{'=' * 72}\n")

param_lifts_7 = [(l, lb, m) for l, lb, m in param_lifts if "score_imom" not in lb]
cumulative7 = pd.Series([True] * len(df))
for lift, label, mask in param_lifts_7:
    cumulative7 = cumulative7 & mask
    show(f"+{label}", df[cumulative7])

# ═══════════════════════════════════════════════════════════════
# 5. MINIMUM VIABLE COMBINATIONS (3–4 params)
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# 5. MINIMUM VIABLE COMBINATIONS (3–6 params)
# ═══════════════════════════════════════════════════════════════
from itertools import combinations

items = list(params_7.items())   # use 7-param set (no score_imom)

for n_params, min_n, top_k in [(3, 30, 15), (4, 10, 10), (5, 10, 10), (6, 10, 10)]:
    print(f"\n{'=' * 72}")
    print(f"  BEST {n_params}-PARAMETER COMBINATIONS (top {top_k} by win rate, n >= {min_n})")
    print(f"{'=' * 72}\n")
    col_w = 55 + (n_params - 3) * 15          # widen label column as combos get longer
    print(f"  {'Combination':<{col_w}} {'N':>4}  {'Win%':>6} {'Lift':>7}")
    print("  " + "-" * (col_w + 22))

    results = []
    for combo in combinations(items, n_params):
        mask = combo[0][1].copy()
        for _, m in combo[1:]:
            mask = mask & m
        sub = df[mask]
        if len(sub) < min_n:
            continue
        win  = (sub.fwd_1d >= WIN).mean() * 100
        loss = (sub.fwd_1d <= LOSS).mean() * 100
        lift = win - BASE_WIN
        label = "  +  ".join(l for l, _ in combo)
        results.append((win, lift, len(sub), label))

    results.sort(reverse=True)
    if results:
        for win, lift, n, label in results[:top_k]:
            print(f"  {label:<{col_w}} {n:>4}  {win:>6.1f}% {lift:>+6.1f}pp")
    else:
        print(f"  No combinations found with n >= {min_n}")

    # Keep combos_4 for the year-by-year section below
    if n_params == 4:
        combos_4 = results
# ═══════════════════════════════════════════════════════════════
# 6. YEAR-BY-YEAR for best combination
# ═══════════════════════════════════════════════════════════════
if combos_4:
    print(f"\n{'=' * 72}")
    print("  YEAR-BY-YEAR — best 4-param combo")
    print(f"{'=' * 72}")
    best_labels = combos_4[0][3]
    print(f"  {best_labels}\n")

    # Rebuild mask for best combo
    best_names = [l.strip() for l in best_labels.split("  +  ")]
    best_mask = pd.Series([True] * len(df))
    for name in best_names:
        best_mask = best_mask & params_7[name]

    best_df = df[best_mask].copy()
    best_df["year"] = best_df.date.dt.year

    print(f"  {'Year':<6} {'N':>4}  {'Win%':>6} {'Avg1d':>7} {'Avg5d':>7}")
    print("  " + "-"*36)
    for yr, g in best_df.groupby("year"):
        win = (g.fwd_1d >= WIN).mean() * 100
        a1  = g.fwd_1d.mean()
        a5  = g.fwd_5d.mean()
        print(f"  {yr:<6} {len(g):>4}  {win:>6.1f}% {a1:>+6.2f}% {a5:>+6.2f}%")

print("\n✅ Done")