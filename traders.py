import pandas as pd
import numpy as np

DATA_DIR = "data"

def load_matrix(name):
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv", index_col=0, dtype=str)
    df.columns = [str(c).split()[0] for c in df.columns]
    mask = df.index.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
    df = df[mask]
    df.index = pd.to_datetime(df.index)
    return df.apply(pd.to_numeric, errors="coerce").sort_index()

closes = load_matrix("master_close")
events = pd.read_csv(f"{DATA_DIR}/event_days_typed.csv", parse_dates=["date"])

# ── Compute trader returns ────────────────────
# Entry at Day+1 close (you buy after seeing Day+1 continuation)
# Target: Day+3 and Day+5 returns from that entry

trader_ret_3d = []
trader_ret_5d = []
entry_price   = []

for _, row in events.iterrows():
    ticker = row["ticker"]
    date   = pd.Timestamp(row["date"])

    if ticker not in closes.columns:
        trader_ret_3d.append(np.nan)
        trader_ret_5d.append(np.nan)
        entry_price.append(np.nan)
        continue

    future = closes[ticker].dropna()
    future = future[future.index > date]

    def get_price(n):
        return float(future.iloc[n-1]) if len(future) >= n else np.nan

    p1 = get_price(1)  # Day+1 close = entry
    p3 = get_price(3)  # Day+3 close
    p5 = get_price(5)  # Day+5 close

    r3 = round((p3/p1 - 1)*100, 2) if not any(np.isnan([p1,p3])) else np.nan
    r5 = round((p5/p1 - 1)*100, 2) if not any(np.isnan([p1,p5])) else np.nan

    trader_ret_3d.append(r3)
    trader_ret_5d.append(r5)
    entry_price.append(p1)

events["entry_price"]    = entry_price
events["trader_ret_3d"]  = trader_ret_3d   # Day+3 return from Day+1 entry
events["trader_ret_5d"]  = trader_ret_5d   # Day+5 return from Day+1 entry

# Label based on trader returns
events["trader_label_3d"] = events["trader_ret_3d"].apply(
    lambda x: "profit" if x > 1 else ("loss" if x < -1 else "flat")
    if not pd.isna(x) else np.nan)

events["trader_label_5d"] = events["trader_ret_5d"].apply(
    lambda x: "profit" if x > 1 else ("loss" if x < -1 else "flat")
    if not pd.isna(x) else np.nan)

events.to_csv(f"{DATA_DIR}/event_days_typed.csv", index=False)
print("✅ Trader returns added")

# ── But this is only valid when Day+1 was continuation ──
# Because you only buy if Day+1 confirmed the move

d1_cont = events[events.label_1d == "continuation"]
print(f"\nEvents where Day+1 continued: {len(d1_cont)}")

print("\n── If you buy at Day+1 close (after seeing continuation) ──")
print(f"  {'Bucket':<10} {'N':>5}  "
      f"{'D3 Profit%':>11} {'D3 Loss%':>9}  "
      f"{'D5 Profit%':>11} {'D5 Loss%':>9}  "
      f"{'Avg D5 Ret':>10}")
print("  " + "-"*65)

for bucket in ["7-10%","10-15%","15%+","ALL"]:
    if bucket == "ALL":
        sub = d1_cont
    else:
        sub = d1_cont[d1_cont.bucket==bucket]
    if len(sub) < 10: continue

    p3=(sub.trader_label_3d=="profit").mean()*100
    l3=(sub.trader_label_3d=="loss").mean()*100
    p5=(sub.trader_label_5d=="profit").mean()*100
    l5=(sub.trader_label_5d=="loss").mean()*100
    avg5=sub.trader_ret_5d.mean()

    print(f"  {bucket:<10} {len(sub):>5}  "
          f"{p3:>11.1f} {l3:>9.1f}  "
          f"{p5:>11.1f} {l5:>9.1f}  "
          f"{avg5:>10.2f}%")

print("\n── By Event Type (Day+1 entry, Day+5 target) ──")
for etype in ["A_ColdBreakout","B_MomentumRun","C_Shock","D_DeadCat","E_Mixed"]:
    sub = d1_cont[d1_cont.event_type==etype]
    if len(sub) < 10: continue
    p5=(sub.trader_label_5d=="profit").mean()*100
    l5=(sub.trader_label_5d=="loss").mean()*100
    avg5=sub.trader_ret_5d.mean()
    print(f"  {etype:<20} n={len(sub):4d} | "
          f"profit={p5:.1f}% loss={l5:.1f}% | "
          f"avg_ret={avg5:.2f}%")