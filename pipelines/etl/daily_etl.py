import argparse
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

from core.polygon_client import PolygonClient


def ensure_datetime_column(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """
    Ensure date column is proper datetime type with validation.
    
    Args:
        df: DataFrame with date column
        col: Name of the date column
        
    Returns:
        DataFrame with validated datetime column
        
    Raises:
        AssertionError: If conversion to datetime fails
    """
    df[col] = pd.to_datetime(df[col])
    assert df[col].dtype == 'datetime64[ns]', f"{col} column must be datetime64[ns], got {df[col].dtype}"
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build technical features with proper NaN handling."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df.get("volume", pd.Series(index=df.index)).astype(float)

    # Technical indicators
    atr = (high - low).rolling(14).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    rsi = 100 - (100 / (1 + close.diff().clip(lower=0).rolling(14).mean() / 
                         (-close.diff().clip(upper=0)).rolling(14).mean()))
    
    # Momentum features
    momentum_5 = close.pct_change(5)
    momentum_10 = close.pct_change(10)
    
    # Volatility
    volatility = close.pct_change().rolling(20).std()
    
    # Volume features
    volume_ma = volume.rolling(20).mean()
    volume_ratio = volume / volume_ma.replace(0, np.nan)

    features = pd.DataFrame({
        "atr": atr,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "rsi": rsi,
        "momentum_5": momentum_5,
        "momentum_10": momentum_10,
        "volatility": volatility,
        "volume_ratio": volume_ratio,
        "close": close,
        "high": high,
        "low": low,
        "volume": volume
    }, index=df.index)
    
    return features.fillna(method='ffill').fillna(0)


def main():
    parser = argparse.ArgumentParser(description="Daily ETL for market data")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL", "TSLA"],
                       help="List of tickers to process")
    parser.add_argument("--days", type=int, default=252,
                       help="Number of days of historical data")
    parser.add_argument("--out", type=str, default="data/market_data.parquet",
                       help="Output parquet file path")
    args = parser.parse_args()

    print(f"🔄 Starting ETL for {len(args.tickers)} tickers...")
    print(f"📊 Fetching {args.days} days of data")
    
    client = PolygonClient()
    rows = []
    
    for ticker in args.tickers:
        try:
            print(f"  Fetching {ticker}...", end=" ")
            df = client.daily_ohlc(ticker, days=args.days)
            
            if df is None or df.empty:
                print("❌ No data")
                continue
                
            # Build features
            features = build_features(df)
            features["ticker"] = ticker
            rows.append(features)
            print(f"✅ {len(features)} rows")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue
    
    if not rows:
        print("⚠️ No data collected, exiting")
        return
    
    # Combine all data
    out = pd.concat(rows, ignore_index=False).reset_index().rename(columns={"index": "date"})
    
    # CRITICAL FIX: Ensure date is datetime type before arithmetic operations
    out = ensure_datetime_column(out, "date")
    
    # Train/test split based on time
    train_cutoff = out["date"].max() - timedelta(days=60)
    out["split"] = out["date"].apply(lambda x: "train" if x < train_cutoff else "test")
    
    # Save to parquet
    out.to_parquet(args.out, index=False, engine="pyarrow", compression="snappy")
    
    # Create versioned backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_path = args.out.replace(".parquet", f"_{timestamp}.parquet")
    out.to_parquet(versioned_path, index=False, engine="pyarrow", compression="snappy")
    
    # Summary
    print("\n" + "="*60)
    print(f"✅ ETL Complete!")
    print(f"📈 Total rows: {len(out)}")
    print(f"📅 Date range: {out['date'].min()} → {out['date'].max()}")
    print(f"🔀 Train/Test split: {(out['split']=='train').sum()} train, {(out['split']=='test').sum()} test")
    print(f"💾 Saved to: {args.out}")
    print(f"💾 Version backup: {versioned_path}")
    print("="*60)


def test_date_arithmetic():
    """
    Smoke test to verify date column handling.
    Run with: python -c "from pipelines.etl.daily_etl import test_date_arithmetic; test_date_arithmetic()"
    """
    print("🧪 Running date arithmetic smoke test...")
    
    # Create sample data
    sample = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=100),
        "close": np.random.randn(100).cumsum() + 100,
        "high": np.random.randn(100).cumsum() + 102,
        "low": np.random.randn(100).cumsum() + 98,
        "volume": np.random.randint(1000000, 10000000, 100)
    })
    
    # Simulate the reset_index operation that caused the bug
    sample = sample.reset_index(drop=True).reset_index().rename(columns={"index": "date"})
    
    # Apply fix
    sample = ensure_datetime_column(sample, "date")
    
    # Test arithmetic operation
    cutoff = sample["date"].max() - timedelta(days=60)
    assert isinstance(cutoff, pd.Timestamp), "Cutoff must be Timestamp"
    
    sample["split"] = sample["date"].apply(lambda x: "train" if x < cutoff else "test")
    
    train_count = (sample["split"] == "train").sum()
    test_count = (sample["split"] == "test").sum()
    
    print(f"✅ Test passed: {train_count} train, {test_count} test rows")
    assert train_count > 0 and test_count > 0, "Both splits must have data"
    print("✅ All smoke tests passed!")


if __name__ == "__main__":
    main()
