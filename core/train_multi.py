# core/train_multi.py
import os
import argparse
import math
from typing import List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.utils import shuffle

from core.polygon_client import PolygonClient

# -------------------- базовые утилиты --------------------
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = (df["high"] - df["low"]).abs()
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def _linreg_slope_norm(x: pd.Series, win: int, price: pd.Series) -> pd.Series:
    # Наклон лин. регрессии на окне / текущую цену
    idx = np.arange(win, dtype=float)
    def _slope(a: np.ndarray) -> float:
        y = a
        xm, ym = idx.mean(), y.mean()
        denom = ((idx - xm) ** 2).sum()
        if denom == 0:
            return 0.0
        beta = ((idx - xm) * (y - ym)).sum() / denom
        return float(beta)
    raw = x.rolling(win).apply(_slope, raw=True)
    return raw / price.clip(lower=1e-9)

def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2.0]
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + ha["ha_close"].iloc[i-1]) / 2.0)
    ha["ha_open"] = pd.Series(ha_open, index=df.index)
    return ha

def _series_run_lengths(vals: np.ndarray, positive: bool=True) -> np.ndarray:
    # На каждом баре — длина текущей серии >0 (или <0)
    out = np.zeros_like(vals, dtype=int)
    run = 0
    for i in range(len(vals)):
        v = vals[i]
        ok = (v > 0) if positive else (v < 0)
        if ok:
            run += 1
        elif v == 0:
            # нулевой столбик — не сбиваем серию
            pass
        else:
            run = 0
        out[i] = run
    return out

def _macd_hist(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist

def _pos_in_range(close: pd.Series, win: int) -> pd.Series:
    lo = close.rolling(win).min()
    hi = close.rolling(win).max()
    w = (hi - lo).replace(0, np.nan)
    return (close - lo) / w

# -------------------- triple-barrier разметка --------------------
def _label_triple_barrier(df: pd.DataFrame,
                          horizon: int,
                          tp_mult: float = 1.0,
                          sl_mult: float = 0.8) -> pd.Series:
    """
    Для каждого бара t: смотрим максимум/минимум в окне (t+1 ... t+horizon),
    проверяем, кто встретился раньше — +TP или -SL (онлайн-приближение).
    """
    close = df["close"].values
    atr14 = df["atr14"].values
    n = len(df)
    y = np.full(n, np.nan)
    for i in range(n - horizon):
        entry = close[i]
        tp = entry + tp_mult * atr14[i]
        sl = entry - sl_mult * atr14[i]
        future_max = close[i+1:i+horizon+1].max()
        future_min = close[i+1:i+horizon+1].min()
        # Простейшая «первый достигнутый» эвристика:
        hit_tp = future_max >= tp
        hit_sl = future_min <= sl
        if hit_tp and not hit_sl:
            y[i] = 1.0
        elif hit_sl and not hit_tp:
            y[i] = 0.0
        elif hit_tp and hit_sl:
            # если оба в окне – решаем по ближней дистанции
            dist_tp = (np.argmax(close[i+1:i+horizon+1] >= tp) + 1
                       if hit_tp else math.inf)
            dist_sl = (np.argmax(close[i+1:i+horizon+1] <= sl) + 1
                       if hit_sl else math.inf)
            y[i] = 1.0 if dist_tp < dist_sl else 0.0
        else:
            # нейтрально — оставляем NaN (потом выкинем)
            pass
    return pd.Series(y, index=df.index)

# -------------------- датасет по одному тикеру --------------------
def _make_dataset_for_ticker(cli: PolygonClient,
                             ticker: str,
                             years: int,
                             look: int,
                             trend: int,
                             horizon_days: int) -> pd.DataFrame:
    days = int(years * 365)
    df = cli.daily_ohlc(ticker, days=days).copy()
    if df is None or len(df) < (max(look, trend) + horizon_days + 10):
        return pd.DataFrame()

    df["atr14"] = _atr(df, 14)
    df["atr28"] = _atr(df, 28)
    df["atr_d_over_price"] = df["atr14"] / df["close"].clip(lower=1e-9)
    df["vol_ratio"] = df["atr14"] / df["atr28"].replace(0, np.nan)

    df["pos"] = _pos_in_range(df["close"], look)
    df["slope_norm"] = _linreg_slope_norm(df["close"], trend, df["close"])

    # Heikin-Ashi runs
    ha = _heikin_ashi(df)
    ha_diff = ha["ha_close"].diff().fillna(0.0).values
    df["ha_up_run"]   = _series_run_lengths(ha_diff, positive=True)
    df["ha_down_run"] = _series_run_lengths(ha_diff, positive=False)

    # MACD runs
    _, _, hist = _macd_hist(df["close"])
    mh = hist.fillna(0.0).values
    df["macd_pos_run"] = _series_run_lengths(mh, positive=True)
    df["macd_neg_run"] = _series_run_lengths(mh, positive=False)

    # Разметка
    df["y"] = _label_triple_barrier(df, horizon_days)

    # Чистим
    feat_cols = [
        "pos", "slope_norm",
        "atr_d_over_price", "vol_ratio",
        "ha_up_run", "ha_down_run",
        "macd_pos_run", "macd_neg_run",
    ]
    out = df[feat_cols + ["y"]].dropna().copy()
    out["ticker"] = ticker
    return out

# -------------------- маппинги горизонта --------------------
HZ_MAP = {
    "ST":  dict(look=60,  trend=14, horizon_days=5),
    "MID": dict(look=120, trend=28, horizon_days=20),
    "LT":  dict(look=240, trend=56, horizon_days=60),
}

def _norm_hz(hz: str) -> str:
    hz = (hz or "").upper()
    if hz not in HZ_MAP:
        raise ValueError("hz must be one of ST|MID|LT")
    return hz

# -------------------- основной train() --------------------
def train(hz: str,
          tickers: List[str],
          years: int,
          out_path: str,
          group: Optional[str] = None):
    hz = _norm_hz(hz)
    cfg = HZ_MAP[hz]
    cli = PolygonClient()

    frames = []
    for t in tickers:
        print(f"[{hz}] building dataset for {t} …")
        df_t = _make_dataset_for_ticker(cli, t, years, cfg["look"], cfg["trend"], cfg["horizon_days"])
        if len(df_t):
            frames.append(df_t)
        else:
            print(f"  skip {t}: not enough data")

    if not frames:
        raise RuntimeError("dataset is empty — check tickers/years")

    data = pd.concat(frames).reset_index(drop=True)
    data = shuffle(data, random_state=42)

    feat_cols = [
        "pos", "slope_norm",
        "atr_d_over_price", "vol_ratio",
        "ha_up_run", "ha_down_run",
        "macd_pos_run", "macd_neg_run",
    ]
    X = data[feat_cols].values
    y = data["y"].values.astype(int)

    # Разделение по времени (упрощённо: последние 15% как валидация)
    split = int(len(data) * 0.85)
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    clf = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=6,
        max_iter=600,
        l2_regularization=0.0,
        random_state=42,
        class_weight={0:1.0, 1:1.0},
    )
    clf.fit(X_train, y_train)

    # Оценка
    proba = clf.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, proba)
    pred = (proba >= 0.5).astype(int)
    print(f"AUC: {auc:.3f}")
    print(classification_report(y_val, pred, digits=3))

    # Сохраняем артефакт
    artifact = {
        "model": clf,
        "features": feat_cols,
        "hz": hz,
        "group": group,
        "tickers_used": tickers,
        "years": years,
        "horizon_days": cfg["horizon_days"],
        "target": "triple_barrier_tp1_vs_sl",
        "version": 1,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(artifact, out_path)
    print(f"Saved model to: {out_path}")

# -------------------- CLI --------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train multi-ticker model for ST/MID/LT")
    p.add_argument("--hz", required=True, choices=["ST","MID","LT"])
    p.add_argument("--tickers", required=True,
                   help="Comma-separated tickers, e.g. QQQ,SPY,AAPL,ETHUSD,BTCUSD,NVDA")
    p.add_argument("--years", type=int, default=8)
    p.add_argument("--out", required=False, default=None,
                   help="Output path; default: models/arxora_lgbm_<HZ>.joblib or with group")
    p.add_argument("--group", required=False, default=None,
                   help="Optional asset class label: CRYPTO|ETF|EQUITY")
    args = p.parse_args()

    hz = args.hz
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    group = (args.group or "").upper() or None

    # имя файла по умолчанию
    if args.out:
        out_path = args.out
    else:
        suffix = f"-{group}" if group else ""
        out_path = os.path.join("models", f"arxora_lgbm_{hz}{suffix}.joblib")

    train(hz, tickers, args.years, out_path, group)
