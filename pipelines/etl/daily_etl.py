import argparse
from datetime import datetime, timedelta
<<<<<<< HEAD
=======
from typing import List
>>>>>>> origin/main

import numpy as np
import pandas as pd

from core.polygon_client import PolygonClient


# Ежедневный ETL: формирует фичи и цели для M7 по тикерам
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    atr14 = (high - low).rolling(14, min_periods=1).mean()
    vol = close.pct_change().rolling(30, min_periods=1).std() * np.sqrt(252)
<<<<<<< HEAD
    slope = close.rolling(20).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=False
    )
    feats = pd.DataFrame(
        {"atr14": atr14, "vol": vol, "slope": slope / close.clip(lower=1e-9)},
=======

    def _slope(arr: np.ndarray) -> float:
        x = np.arange(len(arr), dtype=float)
        coef = np.polyfit(x, arr.astype(float), 1)[0]
        return float(coef)

    slope = close.rolling(20).apply(lambda x: _slope(np.asarray(x)), raw=False)

    feats = pd.DataFrame(
        {
            "atr14": atr14,
            "vol": vol,
            "slope": slope / close.clip(lower=1e-9),
        },
>>>>>>> origin/main
        index=df.index,
    )
    return feats


def build_targets(df: pd.DataFrame, horizon_days: int = 5) -> pd.Series:
    close = df["close"].astype(float)
    fwd = close.shift(-horizon_days)
    y = (fwd - close).fillna(0.0) > 0.0
    return y.astype(int)


<<<<<<< HEAD
def main():
    ap = argparse.ArgumentParser()
=======
def main() -> None:
    ap = argparse.ArgumentParser(description="Daily ETL for M7 features/targets")
>>>>>>> origin/main
    ap.add_argument("--tickers", type=str, default="AAPL,MSFT,TSLA")
    ap.add_argument("--days", type=int, default=240)
    ap.add_argument("--out", type=str, default="data/latest.parquet")
    args = ap.parse_args()

    tickers: List[str] = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    cli = PolygonClient()
    rows: List[pd.DataFrame] = []

    for t in tickers:
        df = cli.daily_ohlc(t, days=args.days)
        if not isinstance(df, pd.DataFrame) or len(df) < 60:
            continue
<<<<<<< HEAD
=======

>>>>>>> origin/main
        feats = build_features(df)
        y = build_targets(df)

        dat = feats.copy()
        dat["ticker"] = t
        dat["y"] = y
<<<<<<< HEAD
        dat = dat.dropna().tail(args.days - 10)
        rows.append(dat)
=======
        dat = dat.dropna().tail(max(1, args.days - 10))
        if len(dat):
            rows.append(dat)

>>>>>>> origin/main
    if not rows:
        raise SystemExit("No data rows")

    out = pd.concat(rows).reset_index().rename(columns={"index": "date"})
    out.to_parquet(args.out, index=False)


if __name__ == "__main__":
    main()
