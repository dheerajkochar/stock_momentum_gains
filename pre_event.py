"""
PHASE 3D — PRE-EVENT METRICS
Project: Post-Surge Momentum Analysis (Nifty 200, 2018–2025)
--------------------------------------------------------------
For each 10%+ event computes pre-event state across 4 dimensions:

  DIMENSION 1 — Price Momentum
    pre_1d, pre_5d, pre_20d, pre_60d

  DIMENSION 2 — Volatility Regime
    atr_5, atr_20, atr_ratio (compression/expansion)
    pre_range_5d (avg daily range % last 5 days)

  DIMENSION 3 — Volume Behaviour
    vol_event       (event day volume ratio vs 20d avg)
    vol_5d_ratio    (avg vol last 5d vs 20d avg)
    vol_trend       (volume building up or flat before event)
    vol_surprise    (event day vol vs last 5d avg)

  DIMENSION 4 — Technical Structure
    d52w_high       (distance from 52w high)
    d52w_low        (distance from 52w low)
    consolidation   (days stock stayed within ±3% before event)
    price_vs_ma20   (above/below 20d moving average)
    price_vs_ma50   (above/below 50d moving average)
    ma20_slope      (direction of 20d MA)

Output: data/event_days_pre.csv
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"

print("=" * 55)
print("  PHASE 3D — PRE-EVENT METRICS")
print("=" * 55)

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────

events = pd.read_csv(f"{DATA_DIR}/event_days_with_returns.csv",
                     parse_dates=["date"])
events = events.sort_values("date").reset_index(drop=True)
print(f"\n✅ Loaded {len(events)} events")

def load_matrix(name):
    path = f"{DATA_DIR}/{name}.csv"
    df   = pd.read_csv(path, index_col=0, dtype=str)
    df.columns = [str(c).split()[0] for c in df.columns]
    mask = df.index.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
    df   = df[mask]
    df.index = pd.to_datetime(df.index)
    df   = df.apply(pd.to_numeric, errors="coerce")
    return df.sort_index()

print("📂 Loading matrices...")
closes  = load_matrix("master_close")
highs   = load_matrix("master_high")
lows    = load_matrix("master_low")
volumes = load_matrix("master_volume")
print(f"  ✅ {closes.shape[1]} tickers × {closes.shape[0]} days")

# ─────────────────────────────────────────────
# COMPUTE PRE-EVENT METRICS
# ─────────────────────────────────────────────

print("\n⚙️  Computing pre-event metrics...")

records = []

for idx, row in events.iterrows():
    ticker = row["ticker"]
    date   = pd.Timestamp(row["date"])
    rec    = {}

    if ticker not in closes.columns:
        records.append(rec)
        continue

    # All history before event date
    c_all = closes[ticker].dropna()
    h_all = highs[ticker].dropna()  if ticker in highs.columns  else pd.Series(dtype=float)
    l_all = lows[ticker].dropna()   if ticker in lows.columns   else pd.Series(dtype=float)
    v_all = volumes[ticker].dropna() if ticker in volumes.columns else pd.Series(dtype=float)

    past_c = c_all[c_all.index < date]
    past_h = h_all[h_all.index < date]
    past_l = l_all[l_all.index < date]
    past_v = v_all[v_all.index < date]

    if len(past_c) < 60:
        records.append(rec)
        continue

    # ── DIMENSION 1: Price Momentum ──────────

    def price_ret(n):
        if len(past_c) >= n:
            return round((past_c.iloc[-1]/past_c.iloc[-n] - 1)*100, 2)
        return np.nan

    rec["pre_1d"]  = price_ret(2)   # yesterday vs day before
    rec["pre_5d"]  = price_ret(5)
    rec["pre_20d"] = price_ret(20)
    rec["pre_60d"] = price_ret(60)

    # ── DIMENSION 2: Volatility Regime ───────

    def atr_n(n):
        """Average True Range over last n days."""
        if len(past_c) < n+1 or len(past_h) < n or len(past_l) < n:
            return np.nan
        h = past_h.iloc[-n:].values.astype(float)
        l = past_l.iloc[-n:].values.astype(float)
        c = past_c.iloc[-n:].values.astype(float)
        cp= past_c.iloc[-n-1:-1].values.astype(float)
        tr = np.maximum(h-l, np.maximum(np.abs(h-cp), np.abs(l-cp)))
        return float(np.mean(tr))

    atr5  = atr_n(5)
    atr20 = atr_n(20)
    rec["atr_5"]   = round(atr5, 3)  if not np.isnan(atr5)  else np.nan
    rec["atr_20"]  = round(atr20, 3) if not np.isnan(atr20) else np.nan
    rec["atr_ratio"] = round(atr5/atr20, 3) \
        if (not np.isnan(atr5) and not np.isnan(atr20) and atr20 > 0) else np.nan

    # Average daily range % over last 5 days
    if len(past_h) >= 5 and len(past_l) >= 5:
        h5 = past_h.iloc[-5:].values.astype(float)
        l5 = past_l.iloc[-5:].values.astype(float)
        c5 = past_c.iloc[-5:].values.astype(float)
        rec["pre_range_5d"] = round(float(np.mean((h5-l5)/c5)*100), 2)
    else:
        rec["pre_range_5d"] = np.nan

    # ── DIMENSION 3: Volume Behaviour ────────

    if len(past_v) >= 20:
        avg_vol_20 = float(past_v.iloc[-20:].mean())
        avg_vol_5  = float(past_v.iloc[-5:].mean())

        # Event day volume ratio
        try:
            event_vol = float(volumes.loc[date, ticker])
            rec["vol_event"] = round(event_vol/avg_vol_20, 2) \
                if avg_vol_20 > 0 else np.nan
        except Exception:
            rec["vol_event"] = np.nan

        # Pre-event 5d avg vs 20d avg
        rec["vol_5d_ratio"] = round(avg_vol_5/avg_vol_20, 2) \
            if avg_vol_20 > 0 else np.nan

        # Volume trend: slope of volume over last 10 days
        if len(past_v) >= 10:
            v10 = past_v.iloc[-10:].values.astype(float)
            x   = np.arange(len(v10))
            slope = np.polyfit(x, v10, 1)[0]
            rec["vol_trend"] = round(slope/avg_vol_20*100, 3) \
                if avg_vol_20 > 0 else np.nan
        else:
            rec["vol_trend"] = np.nan

        # Vol surprise: event day vs pre-5d avg
        try:
            event_vol = float(volumes.loc[date, ticker])
            rec["vol_surprise"] = round(event_vol/avg_vol_5, 2) \
                if avg_vol_5 > 0 else np.nan
        except Exception:
            rec["vol_surprise"] = np.nan
    else:
        rec["vol_event"] = rec["vol_5d_ratio"] = np.nan
        rec["vol_trend"] = rec["vol_surprise"] = np.nan

    # ── DIMENSION 4: Technical Structure ─────

    # 52-week high/low distance
    if len(past_h) >= 20:
        high_252 = float(past_h.iloc[-252:].max()) if len(past_h)>=252 \
                   else float(past_h.max())
        low_252  = float(past_l.iloc[-252:].min()) if len(past_l)>=252 \
                   else float(past_l.min())
        curr     = float(past_c.iloc[-1])
        rec["d52w_high"] = round((high_252 - curr)/high_252*100, 2) \
            if high_252 > 0 else np.nan
        rec["d52w_low"]  = round((curr - low_252)/low_252*100, 2)  \
            if low_252  > 0 else np.nan
    else:
        rec["d52w_high"] = rec["d52w_low"] = np.nan

    # Consolidation: days in ±3% range before event
    if len(past_c) >= 5:
        base  = float(past_c.iloc[-1])
        count = 0
        for p in reversed(past_c.iloc[:-1].values):
            if abs(p/base - 1) <= 0.03:
                count += 1
            else:
                break
        rec["consolidation_days"] = count
    else:
        rec["consolidation_days"] = np.nan

    # Price vs MA20 and MA50
    if len(past_c) >= 50:
        ma20 = float(past_c.iloc[-20:].mean())
        ma50 = float(past_c.iloc[-50:].mean())
        curr = float(past_c.iloc[-1])
        rec["price_vs_ma20"] = round((curr/ma20 - 1)*100, 2)
        rec["price_vs_ma50"] = round((curr/ma50 - 1)*100, 2)

        # MA20 slope (% change over last 5 days of MA)
        ma20_series = past_c.rolling(20).mean().dropna()
        if len(ma20_series) >= 5:
            ma_slope = (ma20_series.iloc[-1] - ma20_series.iloc[-5]) \
                       / ma20_series.iloc[-5] * 100
            rec["ma20_slope"] = round(float(ma_slope), 3)
        else:
            rec["ma20_slope"] = np.nan
    else:
        rec["price_vs_ma20"] = rec["price_vs_ma50"] = rec["ma20_slope"] = np.nan

    records.append(rec)

# ─────────────────────────────────────────────
# MERGE & SAVE
# ─────────────────────────────────────────────

metrics_df = pd.DataFrame(records)
df = pd.concat([events.reset_index(drop=True),
                metrics_df.reset_index(drop=True)], axis=1)

df.to_csv(f"{DATA_DIR}/event_days_pre.csv", index=False)
print(f"  ✅ Saved: data/event_days_pre.csv  shape={df.shape}")

# ─────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────

pre_cols = [
    "pre_1d","pre_5d","pre_20d","pre_60d",
    "atr_ratio","pre_range_5d",
    "vol_event","vol_5d_ratio","vol_trend","vol_surprise",
    "d52w_high","d52w_low","consolidation_days",
    "price_vs_ma20","price_vs_ma50","ma20_slope"
]

print("\n" + "="*60)
print("  PRE-EVENT METRICS — SUMMARY STATISTICS")
print("="*60)
print(f"\n  {'Metric':<22} {'Mean':>8} {'Median':>8} "
      f"{'Std':>8} {'NaN%':>6}")
print("  " + "-"*56)
for col in pre_cols:
    if col not in df.columns:
        continue
    s    = df[col].dropna()
    nanp = df[col].isna().mean()*100
    print(f"  {col:<22} {s.mean():>8.2f} {s.median():>8.2f} "
          f"{s.std():>8.2f} {nanp:>5.1f}%")

# ── Bucket-level averages ─────────────────────
print("\n" + "="*60)
print("  PRE-EVENT PROFILES BY RETURN BUCKET")
print("="*60)

key_cols = ["pre_5d","pre_20d","pre_60d","atr_ratio",
            "vol_event","vol_5d_ratio","d52w_high",
            "consolidation_days","price_vs_ma20"]

for bucket in ["7-10%","10-15%","15%+"]:
    sub = df[df.bucket==bucket]
    print(f"\n  [{bucket}]  n={len(sub)}")
    for col in key_cols:
        if col not in sub.columns: continue
        s = sub[col].dropna()
        print(f"    {col:<22}: mean={s.mean():>7.2f}  "
              f"median={s.median():>7.2f}")

# ── Continuation vs Reversal pre-conditions ──
print("\n" + "="*60)
print("  PRE-CONDITIONS: CONTINUATION vs REVERSAL (Day+1)")
print("="*60)

cont = df[df.label_1d=="continuation"]
rev  = df[df.label_1d=="reversal"]

print(f"\n  {'Metric':<22} {'Cont Mean':>10} {'Rev Mean':>10} {'Diff':>8}")
print("  " + "-"*52)
for col in key_cols:
    if col not in df.columns: continue
    cm = cont[col].mean()
    rm = rev[col].mean()
    diff = cm - rm
    marker = " ◄" if abs(diff) > 1.0 else ""
    print(f"  {col:<22} {cm:>10.2f} {rm:>10.2f} {diff:>8.2f}{marker}")

print("\n  NEXT: Phase 3E — Event Typing & Clustering")