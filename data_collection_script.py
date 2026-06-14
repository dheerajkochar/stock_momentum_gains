"""
PHASE 1 — DATA COLLECTION SCRIPT
Project: Post-Surge Momentum Analysis (Nifty 100, 2018–2025)
--------------------------------------------------------------
Run this script in your LOCAL environment.
Requirements: pip install yfinance pandas numpy tqdm requests
"""

import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
START_DATE = "2018-03-01"
END_DATE   = "2025-03-31"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — NIFTY 100 TICKERS
# Point-in-time constituents are hard to get freely.
# This is the current Nifty 100 list (as of early 2025).
# NOTE: For survivorship-bias-free analysis, manually cross-check
# historical additions/removals from:
# https://www.niftyindices.com/IndexConstituents/ind_nifty100list.csv
# ─────────────────────────────────────────────

# ── Nifty 100 (large cap) ─────────────────────
NIFTY100_TICKERS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "HCLTECH",
    "SUNPHARMA", "WIPRO", "ULTRACEMCO", "NESTLEIND", "TITAN",
    "POWERGRID", "NTPC", "TECHM", "BAJFINANCE", "BAJAJFINSV",
    "ONGC", "COALINDIA", "INDUSINDBK", "JSWSTEEL", "TATASTEEL",
    "HINDALCO", "GRASIM", "ADANIPORTS", "ADANIENT", "ADANIGREEN",
    "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP", "EICHERMOT",
    "BRITANNIA", "DABUR", "MARICO", "GODREJCP", "PIDILITIND",
    "BERGEPAINT", "HAVELLS", "VOLTAS", "WHIRLPOOL", "UBL",
    "TATACONSUM", "TATACOMM", "TATAPOWER", "TATAMOTORS", "M&M",
    "HEROMOTOCO", "BAJAJ-AUTO", "BOSCHLTD", "MOTHERSON", "ESCORTS",
    "AMBUJACEM", "SHREECEM", "ACC", "DALMIACEM", "RAMCOCEM",
    "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "PHOENIXLTD",
    "HDFCLIFE", "SBILIFE", "ICICIPRULI", "MAXHEALTH", "LICI",
    "HDFCAMC", "MUTHOOTFIN", "CHOLAFIN", "PIRAMALENT", "LICHSGFIN",
    "ZOMATO", "NYKAA", "PAYTM", "POLICYBZR", "DELHIVERY",
    "IRCTC", "IRFC", "HAL", "BEL", "BHEL",
    "SIEMENS", "ABB", "CUMMINSIND", "THERMAX", "HONAUT",
    "OFSS", "MPHASIS", "LTIM", "PERSISTENT", "COFORGE",
    "PAGEIND", "ABFRL", "RAYMOND", "TRENT", "VEDL",
]

# ── Nifty 101–200 (additional large/mid cap) ──
NIFTY_NEXT100_TICKERS = [
    # Financials
    "FEDERALBNK", "IDFCFIRSTB", "RBLBANK", "PNB", "BANKBARODA",
    "CANBK", "UNIONBANK", "INDIANB", "KARURVYSYA", "DCBBANK",
    "SUNDARMFIN", "M&MFIN", "MANAPPURAM", "IIFL", "BAJAJHLDNG",
    # IT / Tech
    "KPITTECH", "TATAELXSI", "HEXAWARE", "NIITTECH", "INOXWIND",
    "TANLA", "ROUTE", "ZENSAR", "RATEGAIN", "NEWGEN",
    # Pharma / Healthcare
    "AUROPHARMA", "TORNTPHARM", "ALKEM", "IPCALAB", "GLENMARK",
    "NATCOPHARM", "AJANTPHARM", "LALPATHLAB", "METROPOLIS", "THYROCARE",
    # Auto & Auto Ancillary
    "ASHOKLEY", "TVSMOTOR", "BAJAJCON", "EXIDEIND", "AMARAJABAT",
    "BALKRISIND", "APOLLOTYRE", "CEATLTD", "SUPRAJIT", "SUNDRMFAST",
    # Industrials / Capital Goods
    "AIAENG", "GRINDWELL", "ELGIEQUIP", "KAJARIACER", "ORIENTELEC",
    "IFBIND", "GMMPFAUDLR", "PNCINFRA", "KEC", "KALPATPOWR",
    # Metals & Mining
    "NMDC", "NATIONALUM", "HINDCOPPER", "WELCORP", "RATNAMANI",
    "JINDALSAW", "MSTCLTD", "SAILESH", "MOIL", "GMRINFRA",
    # Consumer / FMCG
    "COLPAL", "EMAMILTD", "JYOTHYLAB", "BAJAJCON", "VBL",
    "RADICO", "MCDOWELL-N", "GODFRYPHLP", "VST", "PATANJALI",
    # Real Estate / Infra
    "BRIGADE", "SOBHA", "MAHLIFE", "SUNTECK", "KOLTEPATIL",
    # Energy
    "IGL", "MGL", "GUJGASLTD", "PETRONET", "GSPL",
    "TORNTPOWER", "CESC", "JPPOWER", "RPOWER", "NHPC",
    # Cement
    "HEIDELBERG", "BIRLACORPN", "JKCEMENT", "JKLAKSHMI", "NCLIND",
    # Logistics / Aviation
    "BLUEDART", "GATI", "TCI", "INTERGLOBE", "SPICEJET",
    # Misc
    "NAVINFLUOR", "DEEPAKNTR", "AAVAS", "HOMEFIRST", "APTUS",
]

# Combined Nifty 200
NIFTY200_TICKERS = NIFTY100_TICKERS + NIFTY_NEXT100_TICKERS

# Append .NS for NSE
TICKERS_NS = [t + ".NS" for t in NIFTY200_TICKERS]


# ─────────────────────────────────────────────
# STEP 2 — DOWNLOAD STOCK PRICE DATA
# ─────────────────────────────────────────────

def download_stock_data(tickers, start, end, output_dir):
    """Download daily OHLCV for all tickers, save per stock."""
    failed = []
    print(f"\n📥 Downloading {len(tickers)} stocks ({start} to {end})...\n")

    for ticker in tqdm(tickers):
        save_path = os.path.join(output_dir, f"{ticker}.csv")
        if os.path.exists(save_path):
            continue  # skip if already downloaded

        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            if df.empty:
                failed.append(ticker)
                continue

            df.index = pd.to_datetime(df.index)
            df.to_csv(save_path)
            time.sleep(0.3)  # be polite to API

        except Exception as e:
            failed.append(ticker)

    print(f"\n✅ Done. Failed tickers ({len(failed)}): {failed}")
    return failed


# ─────────────────────────────────────────────
# STEP 3 — DOWNLOAD INDIA VIX & NIFTY 50
# ─────────────────────────────────────────────

def download_indices(start, end, output_dir):
    indices = {
        "INDIA_VIX": "^INDIAVIX",
        "NIFTY50":   "^NSEI",
    }
    print("\n📥 Downloading indices...")
    for name, ticker in indices.items():
        save_path = os.path.join(output_dir, f"{name}.csv")
        try:
            df = yf.download(ticker, start=start, end=end,
                             progress=False, auto_adjust=True)
            df.to_csv(save_path)
            print(f"  ✅ {name} saved ({len(df)} rows)")
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")


# ─────────────────────────────────────────────
# STEP 4 — BUILD MASTER PRICE MATRIX
# ─────────────────────────────────────────────

def read_yf_csv(path):
    """
    Robustly read a yfinance-saved CSV.
    Strategy: read raw as strings, keep only YYYY-MM-DD indexed rows.
    Handles all yfinance versions regardless of header format.
    """
    raw = pd.read_csv(path, index_col=0, dtype=str, header=0)
    # Flatten multi-level column names (e.g. "Close  RELIANCE.NS" -> "Close")
    raw.columns = [str(c).split()[0] for c in raw.columns]
    # Keep only rows where index matches YYYY-MM-DD
    mask = raw.index.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
    raw = raw[mask]
    raw.index = pd.to_datetime(raw.index, errors="coerce")
    raw = raw[raw.index.notna()]
    raw = raw.apply(pd.to_numeric, errors="coerce")
    return raw


def build_master_close(tickers, output_dir):
    """Combine all close prices into one wide DataFrame."""
    print("\n🔧 Building master close price matrix...")
    frames = {}
    for ticker in tickers:
        path = os.path.join(output_dir, f"{ticker}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = read_yf_csv(path)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # yfinance may name it 'Close' or 'Adj Close'
            col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
            if col:
                frames[ticker] = df[col]
        except Exception as e:
            print(f"  ⚠ Skipped {ticker}: {e}")
            continue

    master = pd.DataFrame(frames)
    master.index = pd.to_datetime(master.index)
    master = master.sort_index()
    master.to_csv(os.path.join(output_dir, "master_close.csv"))
    print(f"  ✅ Master close matrix: {master.shape}")
    return master


def build_master_volume(tickers, output_dir):
    """Combine all volume data into one wide DataFrame."""
    print("\n🔧 Building master volume matrix...")
    frames = {}
    for ticker in tickers:
        path = os.path.join(output_dir, f"{ticker}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = read_yf_csv(path)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Volume" in df.columns:
                frames[ticker] = df["Volume"]
        except Exception:
            continue

    master = pd.DataFrame(frames)
    master.index = pd.to_datetime(master.index)
    master = master.sort_index()
    master.to_csv(os.path.join(output_dir, "master_volume.csv"))
    print(f"  ✅ Master volume matrix: {master.shape}")
    return master


# ─────────────────────────────────────────────
# STEP 4B — BUILD MASTER OHLC (needed for Phase 3B)
# ─────────────────────────────────────────────

def build_master_ohlc(tickers, output_dir):
    """Save Open, High, Low per ticker as separate master matrices."""
    print("\n🔧 Building master OHLC matrices...")
    opens, highs, lows = {}, {}, {}

    for ticker in tickers:
        path = os.path.join(output_dir, f"{ticker}.csv")
        if not os.path.exists(path):
            continue
        try:
            df = read_yf_csv(path)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Open"  in df.columns: opens[ticker]  = df["Open"]
            if "High"  in df.columns: highs[ticker]  = df["High"]
            if "Low"   in df.columns: lows[ticker]   = df["Low"]
        except Exception:
            continue

    for name, frames in [("master_open", opens),
                          ("master_high", highs),
                          ("master_low",  lows)]:
        m = pd.DataFrame(frames)
        m.index = pd.to_datetime(m.index)
        m = m.sort_index()
        m.to_csv(os.path.join(output_dir, f"{name}.csv"))
        print(f"  ✅ {name}: {m.shape}")


# ─────────────────────────────────────────────
# STEP 5 — IDENTIFY EVENT DAYS (7–10% and 10%+)
# ─────────────────────────────────────────────

def find_event_days(master_close, master_volume,
                    threshold=0.07):
    """
    For each day, find ALL stocks with return >= 7%.
    Tag each into three tiers:
      - "7-10%"   : moderate excitation
      - "10-15%"  : strong excitation
      - "15%+"    : extreme excitation (near upper circuit)
    Also compute volume ratio vs 20-day average.
    Caps returns at 50% to remove split/bonus artifacts.
    """
    print("\n🔍 Finding event days (all stocks >= 7% per day)...")

    # Daily returns — cap at 50% to remove adjustment artifacts
    returns = master_close.pct_change().clip(upper=0.50, lower=-0.50)

    # 20-day average volume
    avg_vol = master_volume.rolling(20).mean()
    vol_ratio = master_volume / avg_vol

    events = []

    for date in returns.index:
        day_returns = returns.loc[date].dropna()
        if day_returns.empty:
            continue

        # ALL stocks hitting >= 7% that day
        hits = day_returns[day_returns >= threshold]
        if hits.empty:
            continue

        for ticker, ret in hits.items():
            # Three-tier bucketing
            if ret >= 0.15:
                bucket = "15%+"
            elif ret >= 0.10:
                bucket = "10-15%"
            else:
                bucket = "7-10%"

            # Volume ratio
            try:
                vr = vol_ratio.loc[date, ticker]
                vr = round(float(vr), 2) if not np.isnan(vr) else np.nan
            except Exception:
                vr = np.nan

            events.append({
                "date":         date,
                "ticker":       ticker,
                "day_return":   round(ret * 100, 2),
                "bucket":       bucket,
                "volume_ratio": vr,
            })

    events_df = pd.DataFrame(events)
    events_df.to_csv(os.path.join("data", "event_days.csv"), index=False)
    print(f"  ✅ {len(events_df)} event observations found")
    print(f"     7-10%:  {(events_df.bucket=='7-10%').sum()}")
    print(f"     10-15%: {(events_df.bucket=='10-15%').sum()}")
    print(f"     15%+:   {(events_df.bucket=='15%+').sum()}")
    return events_df


# ─────────────────────────────────────────────
# STEP 6 — COMPUTE FORWARD RETURNS (+1, +3, +5)
# ─────────────────────────────────────────────

def compute_forward_returns(events_df, master_close):
    """Add +1, +3, +5 day forward returns to event days."""
    print("\n📈 Computing forward returns...")

    fwd1, fwd3, fwd5 = [], [], []

    for _, row in events_df.iterrows():
        ticker = row["ticker"]
        date   = pd.Timestamp(row["date"])

        if ticker not in master_close.columns:
            fwd1.append(np.nan); fwd3.append(np.nan); fwd5.append(np.nan)
            continue

        prices = master_close[ticker].dropna()
        future = prices[prices.index > date]

        def get_fwd(n):
            if len(future) >= n:
                return round((future.iloc[n-1] / prices[date] - 1) * 100, 2)
            return np.nan

        fwd1.append(get_fwd(1))
        fwd3.append(get_fwd(3))
        fwd5.append(get_fwd(5))

    events_df["fwd_1d"] = fwd1
    events_df["fwd_3d"] = fwd3
    events_df["fwd_5d"] = fwd5

    # Label: continuation / reversal / flat (threshold: ±1%)
    for col, label_col in [("fwd_1d", "label_1d"),
                            ("fwd_3d", "label_3d"),
                            ("fwd_5d", "label_5d")]:
        events_df[label_col] = events_df[col].apply(
            lambda x: "continuation" if x > 3
            else ("reversal" if x < -2 else "flat")
            if not np.isnan(x) else np.nan
        )

    events_df.to_csv("data/event_days_with_returns.csv", index=False)
    print("  ✅ Forward returns computed and saved")
    return events_df





# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  PHASE 1 — DATA COLLECTION")
    print("  Nifty 100 Momentum Study (2018–2025)")
    print("=" * 55)

    # 1. Download stocks
    failed = download_stock_data(TICKERS_NS, START_DATE, END_DATE, OUTPUT_DIR)

    # 2. Download indices
    download_indices(START_DATE, END_DATE, OUTPUT_DIR)

    # 3. Build master matrices
    master_close  = build_master_close(TICKERS_NS, OUTPUT_DIR)
    master_volume = build_master_volume(TICKERS_NS, OUTPUT_DIR)
    build_master_ohlc(TICKERS_NS, OUTPUT_DIR)

    # 4. Find event days
    events_df = find_event_days(master_close, master_volume)

    # 5. Forward returns
    events_df = compute_forward_returns(events_df, master_close)

    # ── Summary ──────────────────────────────
    print("\n" + "=" * 55)
    print("  PHASE 1 COMPLETE — FILES SAVED IN /data/")
    print("=" * 55)
    print("""
  data/
  ├── <TICKER>.NS.csv          ← individual stock OHLCV
  ├── INDIA_VIX.csv            ← VIX daily
  ├── NIFTY50.csv              ← Nifty 50 daily
  ├── master_close.csv         ← all closes in one matrix
  ├── master_volume.csv        ← all volumes in one matrix
  ├── event_days.csv           ← raw event days
  └── event_days_with_returns.csv  ← with +1/+3/+5 returns

  NEXT: Run Phase 2 — add catalyst tags, sector tags,
        VIX regime, and Nifty market direction columns.
    """)

    # Quick peek at results
    if not events_df.empty:
        print("\n── Sample Event Days ──────────────────────────")
        print(events_df[["date","ticker","day_return","bucket",
                          "fwd_1d","fwd_3d","fwd_5d","label_1d"]].head(10).to_string(index=False))

        print("\n── Continuation Rates by Tier ──────────────────")
        for bucket in ["7-10%", "10-15%", "15%+"]:
            sub = events_df[events_df.bucket == bucket]
            if sub.empty:
                continue
            print(f"\n  [{bucket}]  n={len(sub)}")
            for col, label in [("label_1d","Day+1"),("label_3d","Day+3"),("label_5d","Day+5")]:
                rate = (sub[col] == "continuation").mean() * 100
                rev  = (sub[col] == "reversal").mean() * 100
                flat = (sub[col] == "flat").mean() * 100
                print(f"    {label}: {rate:.1f}% cont | {rev:.1f}% rev | {flat:.1f}% flat")