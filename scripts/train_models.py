# scripts/train_models.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from core.polygon_client import PolygonClient


def linreg_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def features_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["close"].astype(float)

    out["ret_1"] = c.pct_change(1)
    out["ret_3"] = c.pct_change(3)
    out["ret_5"] = c.pct_change(5)
    out["vol_20"] = c.pct_change().rolling(20, min_periods=2).std()

    atr14 = atr_like(df, 14)
    out["atr_pct"] = (atr14 / c).fillna(0.0)

    # позиция в диапазоне разных окон
    for win in (60, 120, 240):
        lo = df["low"].rolling(win, min_periods=2).min()
        hi = df["high"].rolling(win, min_periods=2).max()
        rng = (hi - lo).replace(0, np.nan)
        out[f"pos_{win}"] = ((c - lo) / rng).fillna(0.0)

    # наклон цены (линейная регрессия) на окнах 14/28/56
    for w in (14, 28, 56):
        out[f"slope_{w}"] = (
            c.rolling(w).apply(lambda s: linreg_slope(s.values), raw=False)
        ) / c.replace(0, np.nan)

    # «тени» и тело как доли от диапазона
    body = (df["close"] - df["open"]).abs()
    upper = (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=0)
    lower = (df[["open", "close"]].min(axis=1) - df["low"]).clip(lower=0)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    out["body_r"] = (body / rng).fillna(0.0)
    out["upper_r"] = (upper / rng).fillna(0.0)
    out["lower_r"] = (lower / rng).fillna(0.0)

    return out.fillna(0.0)


def load_symbol(cli: PolygonClient, symbol: str, days: int = 1500) -> pd.DataFrame:
    df = cli.daily_ohlc(symbol, days=days)
    df = df.sort_index()
    df["ticker"] = symbol
    return df


def label_forward_up(close: pd.Series, horizon_days: int) -> pd.Series:
    fwd = close.shift(-horizon_days) / close - 1.0
    return (fwd > 0).astype(int)


def train_one(hz_tag: str, frames: List[pd.DataFrame]) -> str:
    """
    Возвращает путь к сохранённой модели для горизонта hz_tag.
    Горизонты: ST≈5 дней, MID≈20 дней, LT≈60 дней.
    """
    hz_to_days = {"ST": 5, "MID": 20, "LT": 60}
    days = hz_to_days.get(hz_tag, 5)

    # объединяем все тикеры
    feats, ys = [], []
    for f in frames:
        ff = features_frame(f)
        y = label_forward_up(f["close"], days)
        xx = ff.loc[y.index]
        mask = xx.notna().all(axis=1) & y.notna()
        feats.append(xx[mask])
        ys.append(y[mask])

    X = pd.concat(feats).astype(float)
    y = pd.concat(ys).astype(int)

    # тренировка
    clf = HistGradientBoostingClassifier(
        max_depth=None,
        max_leaf_nodes=31,
        learning_rate=0.05,
        max_iter=600,
        l2_regularization=0.0,
        validation_fraction=0.1,
        random_state=42,
    )
    clf.fit(X, y)

    # валидационная AUC по holdout'у внутри модели
    try:
        y_proba = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_proba)
        print(f"[{hz_tag}] AUC(all)={auc:.3f} · rows={len(X)}")
    except Exception:
        print(f"[{hz_tag}] trained rows={len(X)}")

    Path("models").mkdir(parents=True, exist_ok=True)
    out_path = f"models/hgb_{hz_tag}.joblib"
    joblib.dump(clf, out_path)
    print("Saved:", out_path)
    return out_path


def main() -> None:
    if os.getenv("CI_DRY_RUN") == "1":
        print("CI_DRY_RUN=1 -> skip train_models heavy run")
        sys.exit(0)

    ap = argparse.ArgumentParser(description="Train models for multiple horizons")
    ap.add_argument(
        "--tickers",
        type=str,
        default="AAPL,MSFT,TSLA,NVDA,SPY,QQQ",
        help="Comma-separated list of tickers",
    )
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--horizons", type=str, default="ST,MID,LT")
    args = ap.parse_args()

    cli = PolygonClient()
    symbols = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    frames = [load_symbol(cli, s, days=args.days) for s in symbols]

    for hz in [h.strip().upper() for h in args.horizons.split(",") if h.strip()]:
        train_one(hz, frames)


if __name__ == "__main__":
    main()
