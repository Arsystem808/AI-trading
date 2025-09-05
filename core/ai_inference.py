# core/ai_inference.py
import os
from typing import Tuple, List
import numpy as np
import pandas as pd

# ML
try:
    from lightgbm import LGBMClassifier
    _USE_LGBM = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier as LGBMClassifier  # fallback
    _USE_LGBM = False

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import joblib

# data
from .polygon_client import PolygonClient


# ---------- small helpers ----------
def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def _feat_eng(df: pd.DataFrame) -> pd.DataFrame:
    """Быстрые табличные фичи по дневным OHLC."""
    out = pd.DataFrame(index=df.index)
    c = df["close"]

    # доходности
    out["ret1"]  = c.pct_change(1)
    out["ret2"]  = c.pct_change(2)
    out["ret5"]  = c.pct_change(5)
    out["ret10"] = c.pct_change(10)

    # скользящие статистики
    out["ma5"]   = c.rolling(5).mean()   / c - 1.0
    out["ma10"]  = c.rolling(10).mean()  / c - 1.0
    out["ma20"]  = c.rolling(20).mean()  / c - 1.0
    out["std5"]  = c.rolling(5).std()
    out["std10"] = c.rolling(10).std()

    # волатильность
    out["atr14"] = _atr_like(df, 14)
    out["rng"]   = (df["high"] - df["low"]).rolling(5).mean()

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def _make_labels(close: pd.Series) -> pd.Series:
    """Бинарная цель: 1 если завтра close выше сегодняшнего."""
    y = (close.shift(-1) > close).astype(int)
    return y.loc[y.index.isin(close.index)]


def _fetch_and_build(cli: PolygonClient, ticker: str, months: int) -> pd.DataFrame:
    # ~21 торговых дня в месяц + запас
    days = int(months * 22 + 40)
    df = cli.daily_ohlc(ticker, days=days)
    if df is None or df.empty:
        raise ValueError(f"Нет данных для {ticker}")
    feats = _feat_eng(df)
    y = _make_labels(df["close"]).reindex(feats.index)
    feats["y"] = y
    feats["ticker_id"] = hash(ticker) % 10_000_000  # простая идентификация тикера
    feats = feats.dropna()
    return feats


def _train_quick(symbols: List[str], months: int, model_dir: str, tag: str) -> Tuple[str, float, tuple, float]:
    """
    Общий тренер: сохраняет модель и возвращает:
      (path, auc, X.shape, pos_share)
    """
    os.makedirs(model_dir, exist_ok=True)
    cli = PolygonClient()

    # сбор датасета
    frames = []
    for s in symbols:
        s = s.strip()
        if not s:
            continue
        frames.append(_fetch_and_build(cli, s, months))
    if not frames:
        raise ValueError("Пустой список тикеров.")

    data = pd.concat(frames).sort_index()
    y = data["y"].astype(int).values
    X = data.drop(columns=["y"]).values

    # train/valid split по времени (простой hold-out)
    # используем индексы, чтобы не тасовать
    idx = np.arange(len(X))
    cut = int(len(idx) * 0.8)
    X_train, X_valid = X[:cut], X[cut:]
    y_train, y_valid = y[:cut], y[cut:]

    # модель
    if _USE_LGBM:
        model = LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
    else:
        # fallback — помедленнее, но без внешних зависимостей
        model = LGBMClassifier(random_state=42)

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_valid)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_valid)
    try:
        auc = float(roc_auc_score(y_valid, proba))
    except Exception:
        auc = float("nan")

    pos_share = float(y.mean())
    shape = X.shape

    # имя файла — без двоеточий и слешей
    clean_names = "_".join([s.replace(":", "").replace("/", "").upper() for s in symbols[:4]])
    if len(symbols) > 4:
        clean_names += f"+{len(symbols)-4}"
    fname = f"arxora_lgbm_{tag}_{clean_names}.joblib"
    fpath = os.path.join(model_dir, fname)
    joblib.dump({"model": model, "feature_names": list(data.drop(columns=['y']).columns)}, fpath)

    return fpath, auc, shape, pos_share


# ---------- публичные функции, которые ждёт app.py ----------
def train_quick_st(tickers_csv: str, months: int, model_dir: str) -> Tuple[str, float, tuple, float]:
    symbols = [t.strip() for t in tickers_csv.split(",") if t.strip()]
    return _train_quick(symbols, months, model_dir, tag="ST")


def train_quick_mid(tickers_csv: str, months: int, model_dir: str) -> Tuple[str, float, tuple, float]:
    symbols = [t.strip() for t in tickers_csv.split(",") if t.strip()]
    return _train_quick(symbols, months, model_dir, tag="MID")


def train_quick_lt(tickers_csv: str, months: int, model_dir: str) -> Tuple[str, float, tuple, float]:
    symbols = [t.strip() for t in tickers_csv.split(",") if t.strip()]
    # для LT возьмём больше истории автоматически
    months = max(months, 24)
    return _train_quick(symbols, months, model_dir, tag="LT")

