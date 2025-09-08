# core/train_multi.py
import os
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta

# клиент
from core.polygon_client import PolygonClient

# ========= helpers =========
def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def _linreg_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    beta = ((x - xm) * (y - ym)).sum() / denom
    return float(beta)

def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index.copy())
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2.0]
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + ha["ha_close"].iloc[i-1]) / 2.0)
    ha["ha_open"] = pd.Series(ha_open, index=df.index)
    return ha

def _macd_hist(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

# ========= features/labels =========
def _build_features(df: pd.DataFrame, hz: str) -> pd.DataFrame:
    """
    Возвращает DataFrame с фичами и целевой переменной y.
    y = 1, если будущая доходность за N дней > 0
    """
    close = df["close"].astype(float)
    ret1  = close.pct_change(1)
    ret5  = close.pct_change(5)
    ret10 = close.pct_change(10)

    atr14 = _atr_like(df, 14)
    atr28 = _atr_like(df, 28)
    atr_rel = (atr14 / close).clip(0, 1)

    vol_ratio = (atr14 / atr28.replace(0, np.nan)).clip(0, 5)

    slope14 = pd.Series(
        [np.nan]* (len(close)-14) + [
            _linreg_slope(close.iloc[i-14+1:i+1].values) for i in range(14-1, len(close))
        ],
        index=close.index
    )
    slope_norm = (slope14 / close).clip(-1, 1)

    ha = _heikin_ashi(df)
    ha_diff = ha["ha_close"].diff()
    ha_up = (ha_diff > 0).astype(int).rolling(5, min_periods=1).sum()
    ha_dn = (ha_diff < 0).astype(int).rolling(5, min_periods=1).sum()

    _, _, hist = _macd_hist(close)
    macd_hist = hist
    macd_pos = (macd_hist > 0).astype(int).rolling(5, min_periods=1).sum()
    macd_neg = (macd_hist < 0).astype(int).rolling(5, min_periods=1).sum()

    X = pd.DataFrame({
        "ret1": ret1,
        "ret5": ret5,
        "ret10": ret10,
        "atr_rel": atr_rel,
        "vol_ratio": vol_ratio,
        "slope_norm": slope_norm,
        "ha_up5": ha_up,
        "ha_dn5": ha_dn,
        "macd_pos5": macd_pos,
        "macd_neg5": macd_neg,
    }, index=df.index).replace([np.inf, -np.inf], np.nan)

    # horizon → шаг целевой переменной
    fut_map = {"ST": 5, "MID": 20, "LT": 40}
    fut = fut_map.get(hz, 5)
    future_ret = close.shift(-fut) / close - 1.0
    y = (future_ret > 0).astype(int)

    out = X.copy()
    out["y"] = y
    return out

def _fetch_days_by_years(years: int) -> int:
    return int(365.25 * years) + 5

# ========= training =========
def train_multi(hz: str, tickers: list[str], years: int, out_path: str | None):
    cli = PolygonClient()

    frames = []
    days = _fetch_days_by_years(years)

    for t in tickers:
        print(f"[{t}] fetching {days} days...")
        df = cli.daily_ohlc(t, days=days).dropna()
        if df is None or len(df) < 200:
            print(f"  -> skip: not enough data for {t}")
            continue

        feats = _build_features(df, hz).dropna()
        if feats.empty:
            print(f"  -> skip: empty features for {t}")
            continue

        feats["ticker"] = t
        frames.append(feats)

    if not frames:
        raise ValueError("Нет данных для обучения (все тикеры отфильтрованы).")

    data = pd.concat(frames).sort_index()
    # добавим простую кодировку тикера (cat)
    ticker_map = {t: i for i, t in enumerate(sorted(data["ticker"].unique()))}
    data["ticker_id"] = data["ticker"].map(ticker_map).astype(int)
    X_all = data.drop(columns=["y", "ticker"])
    y_all = data["y"].astype(int)

    # temporal split: последние 20% на валидацию
    n = len(X_all)
    cut = int(n * 0.8)
    X_tr, X_val = X_all.iloc[:cut], X_all.iloc[cut:]
    y_tr, y_val = y_all.iloc[:cut], y_all.iloc[cut:]

    # модель: lightgbm если есть, иначе — RandomForest
    try:
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary",
            class_weight="balanced",
            random_state=42,
        )
    except Exception:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    print("Training...")
    clf.fit(X_tr, y_tr)

    # метрики
    try:
        from sklearn.metrics import roc_auc_score, classification_report
        p_val = getattr(clf, "predict_proba")(X_val)[:, 1]
        auc = roc_auc_score(y_val, p_val)
        print(f"VAL AUC: {auc:.3f}")
        print(classification_report(y_val, (p_val >= 0.5).astype(int)))
    except Exception as e:
        print(f"(metrics skipped: {e})")

    # ===== сохраняем =====
    model_pack = {
        "model": clf,
        "features": list(X_all.columns),
        "hz": hz,
        "tickers": tickers,
        "meta": {"tickers": tickers},
        "notes": "multi-ticker model; label = future N-day return > 0",
    }

    if out_path is None:
        out_path = os.path.join("models", f"arxora_lgbm_{hz}.joblib")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model_pack, out_path)
    print(f"Saved model to: {out_path}")

# ========= CLI =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", required=True, choices=["ST", "MID", "LT"], help="horizon")
    ap.add_argument("--tickers", required=True, help="comma-separated tickers")
    ap.add_argument("--years", type=int, default=6, help="years of history")
    ap.add_argument("--out", default=None, help="optional output path")
    args = ap.parse_args()

    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    train_multi(args.hz, tickers, args.years, args.out)

if __name__ == "__main__":
    main()
