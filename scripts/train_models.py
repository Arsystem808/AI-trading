# scripts/train_models.py
import os, sys, math, joblib, argparse
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from core.polygon_client import PolygonClient

def linreg_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2: return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0: return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)

def atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def features_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    out["ret_1"]  = c.pct_change(1)
    out["ret_3"]  = c.pct_change(3)
    out["ret_5"]  = c.pct_change(5)
    out["vol_20"] = c.pct_change().rolling(20).std()

    atr14 = atr_like(df, 14)
    out["atr_pct"] = (atr14 / c).fillna(0.0)

    # позиция в диапазоне разных окон
    for win in (60, 120, 240):
        lo = df["low"].rolling(win, min_periods=2).min()
        hi = df["high"].rolling(win, min_periods=2).max()
        rng = (hi - lo).replace(0, np.nan)
        out[f"pos_{win}"] = (c - lo) / rng

    # наклон цены (линейная регрессия) на окнах 14/28/56
    for w in (14, 28, 56):
        out[f"slope_{w}"] = (
            c.rolling(w).apply(lambda s: linreg_slope(s.values), raw=False)
        ) / c

    # «тени» и тело как доли от диапазона
    body  = (df["close"] - df["open"]).abs()
    upper = (df["high"] - df[["open","close"]].max(axis=1)).clip(lower=0)
    lower = (df[["open","close"]].min(axis=1) - df["low"]).clip(lower=0)
    rng   = (df["high"] - df["low"]).replace(0, np.nan)
    out["body_r"]  = (body  / rng).fillna(0.0)
    out["upper_r"] = (upper / rng).fillna(0.0)
    out["lower_r"] = (lower / rng).fillna(0.0)

    return out.fillna(0.0)

def load_symbol(cli, symbol: str, days: int = 1500) -> pd.DataFrame:
    df = cli.daily_ohlc(symbol, days=days)
    df = df.sort_index()
    df["ticker"] = symbol
    return df

def label_forward_up(close: pd.Series, horizon_days: int) -> pd.Series:
    fwd = close.shift(-horizon_days) / close - 1.0
    return (fwd > 0).astype(int)

def train_one(hz_tag: str, frames: list[pd.DataFrame]) -> str:
    """Возвращает путь к сохранённой модели."""
    # горизонты (примерно): ST~5дн, MID~20дн, LT~60дн
    H = {"ST": 5, "MID": 20, "LT": 60}[hz_tag]

    feats_all, y_all = [], []
    for df in frames:
        X = features_frame(df)
        y = label_forward_up(df["close"], H)
        # убираем хвост без метки
        ok = y.notna()
        feats_all.append(X[ok])
        y_all.append(y[ok].astype(int))

    X = pd.concat(feats_all, axis=0)
    y = pd.concat(y_all, axis=0)

    if len(X) < 500:
        raise RuntimeError(f"Мало данных для {hz_tag}: {len(X)}")

    clf = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.08, max_iter=400, l2_regularization=0.0
    )
    clf.fit(X, y)
    p = clf.predict_proba(X)[:,1]
    auc = roc_auc_score(y, p)

    models_dir = Path(os.getenv("ARXORA_MODEL_DIR","models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"arxora_lgbm_{hz_tag}.joblib"
    joblib.dump({"model": clf, "features": list(X.columns), "auc": float(auc)}, path)
    print(f"[{hz_tag}] saved {path} • AUC={auc:.3f} • n={len(X)}")
    return str(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=str,
        default="SPY,QQQ,AAPL,MSFT,TSLA,X:BTCUSD,X:ETHUSD")
    parser.add_argument("--days", type=int, default=1500)
    args = parser.parse_args()

    key = os.getenv("POLYGON_API_KEY")
    if not key:
        print("ERROR: POLYGON_API_KEY не задан", file=sys.stderr)
        sys.exit(1)

    cli = PolygonClient()
    frames = []
    for t in [s.strip() for s in args.tickers.split(",") if s.strip()]:
        try:
            frames.append(load_symbol(cli, t, days=args.days))
            print("loaded:", t)
        except Exception as e:
            print("skip", t, "→", e)

    if not frames:
        print("Нет данных для обучения", file=sys.stderr)
        sys.exit(1)

    for hz in ("ST","MID","LT"):
        try:
            train_one(hz, frames)
        except Exception as e:
            print(f"train {hz} failed:", e)

if __name__ == "__main__":
    main()
