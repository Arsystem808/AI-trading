import argparse
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

from core.polygon_client import PolygonClient


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build technical features with proper NaN handling."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df.get("volume", pd.Series(index=df.index)).astype(float)

    # Technical indicators
    atr14 = (high - low).rolling(14, min_periods=14).mean()
    vol = close.pct_change().rolling(30, min_periods=30).std() * np.sqrt(252)

    def _slope(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr), dtype=float)
        coef = np.polyfit(x, arr.astype(float), 1)[0]
        return float(coef)

    slope = close.rolling(20, min_periods=20).apply(
        lambda x: _slope(np.asarray(x)), raw=False
    )

    # Additional momentum indicators
    returns_1d = close.pct_change(1)
    returns_5d = close.pct_change(5)
    
    # RSI (14-period)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.clip(lower=1e-9)
    rsi = 100 - (100 / (1 + rs))

    # Volume features (if available)
    vol_ma_20 = volume.rolling(20, min_periods=20).mean() if not volume.isna().all() else pd.Series(0, index=df.index)

    feats = pd.DataFrame(
        {
            "atr14": atr14,
            "vol": vol,
            "slope": slope / close.clip(lower=1e-9),
            "returns_1d": returns_1d,
            "returns_5d": returns_5d,
            "rsi": rsi,
            "vol_ma_20": vol_ma_20,
        },
        index=df.index,
    )
    return feats


def build_targets(df: pd.DataFrame, horizon_days: int = 5) -> pd.Series:
    """Build forward-looking targets WITHOUT filling unknown future values."""
    close = df["close"].astype(float)
    fwd = close.shift(-horizon_days)
    # Return NaN for unknown future - do NOT fill with 0
    y = (fwd - close) > 0.0
    return y.astype(float)  # Keeps NaN preserved


def validate_features(feats: pd.DataFrame, ticker: str) -> bool:
    """Validate feature quality and detect anomalies."""
    if feats.isnull().all().any():
        print(f"⚠️  Warning [{ticker}]: Feature column entirely null")
        return False
    
    # Check for extreme outliers (beyond 99th percentile * 10)
    for col in ["atr14", "vol"]:
        if col in feats.columns:
            q99 = feats[col].quantile(0.99)
            if q99 > 0 and (feats[col] > q99 * 10).any():
                outlier_count = (feats[col] > q99 * 10).sum()
                print(f"⚠️  Warning [{ticker}]: {outlier_count} extreme outliers in {col}")
    
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily ETL for M7 features/targets")
    ap.add_argument("--tickers", type=str, default="AAPL,MSFT,TSLA")
    ap.add_argument("--days", type=int, default=240)
    ap.add_argument("--out", type=str, default="data/latest.parquet")
    ap.add_argument("--horizon", type=int, default=5, help="Prediction horizon in days")
    args = ap.parse_args()

    tickers: List[str] = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    cli = PolygonClient()
    rows: List[pd.DataFrame] = []

    print(f"🔄 Starting ETL for {len(tickers)} tickers...")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for t in tickers:
        print(f"\n📊 Processing {t}...")
        df = cli.daily_ohlc(t, days=args.days)
        
        if not isinstance(df, pd.DataFrame) or len(df) < 60:
            print(f"❌ Skipped {t}: Insufficient data ({len(df) if isinstance(df, pd.DataFrame) else 0} rows)")
            continue

        feats = build_features(df)
        y = build_targets(df, horizon_days=args.horizon)

        # Validate before combining
        if not validate_features(feats, t):
            continue

        dat = feats.copy()
        dat["ticker"] = t
        dat["y"] = y
        
        # Drop rows with ANY NaN (features or target)
        initial_len = len(dat)
        dat = dat.dropna()
        dropped = initial_len - len(dat)
        
        if len(dat) < 30:
            print(f"❌ Skipped {t}: Too few valid rows after cleaning ({len(dat)} rows)")
            continue
        
        print(f"✅ {t}: {len(dat)} valid rows (dropped {dropped} incomplete rows)")
        rows.append(dat)

    if not rows:
        raise SystemExit("❌ No data rows produced. Check ticker symbols and data availability.")

    # Combine and add metadata
    out = pd.concat(rows, ignore_index=False).reset_index().rename(columns={"index": "date"})
    
    # Add train/test split based on time
    train_cutoff = out["date"].max() - timedelta(days=60)
    out["split"] = out["date"].apply(lambda d: "test" if d > train_cutoff else "train")
    
    # Add ETL timestamp for versioning
    out["etl_timestamp"] = datetime.now().isoformat()
    
    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = args.out.replace(".parquet", f"_{timestamp}.parquet")
    
    out.to_parquet(args.out, index=False)
    out.to_parquet(versioned_path, index=False)
    
    # Summary statistics
    print(f"\n{'='*50}")
    print(f"✅ ETL Complete!")
    print(f"{'='*50}")
    print(f"📦 Total rows: {len(out)}")
    print(f"🎯 Target distribution: {out['y'].value_counts().to_dict()}")
    print(f"📊 Tickers: {out['ticker'].nunique()} ({', '.join(sorted(out['ticker'].unique()))})")
    print(f"📅 Date range: {out['date'].min()} to {out['date'].max()}")
    print(f"🔀 Train/Test split: {(out['split']=='train').sum()} train, {(out['split']=='test').sum()} test")
    print(f"💾 Saved to: {args.out}")
    print(f"💾 Version backup: {versioned_path}")


if __name__ == "__main__":
    main()
