# core/ai_inference.py
# Универсальный тренер/инференс для ST/MID/LT.
# - train_quick_model(hz_tag, tickers, months)  -> обучает и сохраняет модели в models/
# - load_model(hz_tag, ticker)                  -> подхватывает персональную или общую модель
# - predict_long_proba(hz_tag, ticker, df)      -> даёт P(long) по последней свече
# Сохранённый файл: joblib-словарь {"model": <sklearn/lightgbm>, "hz": "...", "features": [...]}

from __future__ import annotations

import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

from core.polygon_client import PolygonClient

# -------------------- Константы/путь --------------------
MODEL_DIR = os.getenv("ARXORA_MODEL_DIR", "models")
DEFAULT_FEATS: List[str] = [
    "ret1", "ret3", "ret5", "vol10", "atr14", "rng", "slope10", "slope20"
]

os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------- Утилиты --------------------
def _safe_ticker_name(t: str) -> str:
    t = (t or "").upper().strip()
    t = t.replace("X:", "").replace("C:", "").replace("O:", "")
    t = t.replace("USDT", "USD").replace("USDC", "USD")
    return re.sub(r"[^A-Z0-9]", "", t)


def model_path(hz_tag: str, ticker: Optional[str] = None) -> str:
    """
    Если указан один тикер при обучении — сохраняем персональную модель
    models/arxora_lgbm_<HZ>_<TICKER>.joblib
    Иначе — общую:
    models/arxora_lgbm_<HZ>.joblib
    """
    if ticker:
        return os.path.join(MODEL_DIR, f"arxora_lgbm_{hz_tag}_{_safe_ticker_name(ticker)}.joblib")
    return os.path.join(MODEL_DIR, f"arxora_lgbm_{hz_tag}.joblib")


def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def _lin_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def _build_Xy(df: pd.DataFrame, hz_tag: str) -> tuple[pd.DataFrame, pd.Series]:
    """Фичи + бэктаргет для обучения."""
    df = df.copy()
    df["ret1"]  = df["close"].pct_change(1)
    df["ret3"]  = df["close"].pct_change(3)
    df["ret5"]  = df["close"].pct_change(5)
    df["vol10"] = df["close"].pct_change().rolling(10).std()
    df["atr14"] = _atr_like(df, 14) / df["close"]
    df["rng"]   = (df["high"] - df["low"]) / df["close"]
    df["slope10"] = df["close"].rolling(10).apply(_lin_slope, raw=True)
    df["slope20"] = df["close"].rolling(20).apply(_lin_slope, raw=True)

    fwd = 5 if hz_tag == "ST" else (20 if hz_tag == "MID" else 60)
    df["fwd"] = df["close"].shift(-fwd) / df["close"] - 1.0
    df["y"]   = (df["fwd"] > 0).astype(int)

    X = df[DEFAULT_FEATS].dropna()
    y = df.loc[X.index, "y"]
    return X, y


def _featurize_last(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Фичи на последнюю доступную дату для инференса."""
    df = df.copy()
    df["ret1"]  = df["close"].pct_change(1)
    df["ret3"]  = df["close"].pct_change(3)
    df["ret5"]  = df["close"].pct_change(5)
    df["vol10"] = df["close"].pct_change().rolling(10).std()
    df["atr14"] = _atr_like(df, 14) / df["close"]
    df["rng"]   = (df["high"] - df["low"]) / df["close"]
    df["slope10"] = df["close"].rolling(10).apply(_lin_slope, raw=True)
    df["slope20"] = df["close"].rolling(20).apply(_lin_slope, raw=True)

    X = df[DEFAULT_FEATS].dropna()
    if len(X) == 0:
        return None
    return X.tail(1)


def _align_features(x_row: pd.DataFrame, feat_names: List[str]) -> pd.DataFrame:
    """
    Приводим порядок и набор колонок к тем, на которых обучалась модель:
    недостающие фичи -> 0.0; лишние — игнорируем.
    """
    aligned = pd.DataFrame(index=x_row.index)
    for f in feat_names:
        aligned[f] = x_row[f] if f in x_row.columns else 0.0
    return aligned


# -------------------- Обучение --------------------
def train_quick_model(hz_tag: str, tickers: list[str], months: int = 24) -> list[Dict[str, Any]]:
    """
    Обучает LGBM-классификатор(ы) для указанного горизонта.
    - hz_tag: "ST" | "MID" | "LT"
    - tickers: список тикеров Polygon (можно с X:/C:)
    - months: сколько месяцев истории грузить
    Возвращает список результатов [{'ticker','path','auc','shape','pos_rate'}]
    """
    cli = PolygonClient()
    results: list[Dict[str, Any]] = []

    for tk in (t.strip() for t in tickers):
        if not tk:
            continue

        days = max(200, int(months * 30))
        df = cli.daily_ohlc(tk, days=days)
        if df is None or len(df) < 80:
            continue

        X, y = _build_Xy(df, hz_tag)
        if len(X) < 100:
            continue

        model = LGBMClassifier(
            n_estimators=400, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42
        )
        model.fit(X, y)

        try:
            proba = model.predict_proba(X)[:, 1]
            auc = float(roc_auc_score(y, proba))
        except Exception:
            auc = float("nan")

        # Персональная, если ровно один тикер; иначе — общая
        out = model_path(hz_tag, ticker=tk if len(tickers) == 1 else None)
        joblib.dump(
            {"model": model, "hz": hz_tag, "features": list(X.columns)},
            out,
            compress=3
        )

        results.append({
            "ticker": tk,
            "path": out,
            "auc": auc,
            "shape": (int(X.shape[0]), int(X.shape[1])),
            "pos_rate": float(y.mean())
        })

    return results


# -------------------- Загрузка и инференс --------------------
def load_model(hz_tag: str, ticker: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Сначала пытаемся персональную (…_<TICKER>.joblib),
    потом общую (…_<HZ>.joblib). Возвращаем унифицированный словарь.
    """
    paths = []
    if ticker:
        paths.append(model_path(hz_tag, ticker=ticker))
    paths.append(model_path(hz_tag, ticker=None))

    for p in paths:
        if os.path.exists(p):
            obj = joblib.load(p)
            # Поддержка как "чистых" моделей, так и dict-контейнеров
            if isinstance(obj, dict) and "model" in obj:
                feats = obj.get("features", DEFAULT_FEATS)
                return {"model": obj["model"], "features": feats, "path": p, "hz": hz_tag}
            else:
                # Оборачиваем в контейнер на лету
                try:
                    feats = getattr(obj, "feature_name_", DEFAULT_FEATS)
                except Exception:
                    feats = DEFAULT_FEATS
                return {"model": obj, "features": list(feats), "path": p, "hz": hz_tag}
    return None


def predict_long_proba(hz_tag: str, ticker: str, df: pd.DataFrame) -> Optional[float]:
    """
    P(long) по последней свече. Возвращает None, если модели/фичей нет.
    """
    pack = load_model(hz_tag, ticker=ticker)
    if pack is None:
        pack = load_model(hz_tag, ticker=None)  # запасной вариант (общая модель)
    if pack is None:
        return None

    x = _featurize_last(df)
    if x is None:
        return None

    x = _align_features(x, pack["features"])
    try:
        proba = float(pack["model"].predict_proba(x)[:, 1][0])
        return proba
    except Exception:
        return None


# Удобный хелпер: решение по порогам из env
def decide_side_from_proba(p_long: float) -> str:
    """
    Возвращает 'BUY' | 'SHORT' | 'HOLD' исходя из порогов.
    ARXORA_AI_TH_LONG  (default 0.60)
    ARXORA_AI_TH_SHORT (default 0.40)
    """
    th_long  = float(os.getenv("ARXORA_AI_TH_LONG", 0.60))
    th_short = float(os.getenv("ARXORA_AI_TH_SHORT", 0.40))

    if p_long is None:
        return "HOLD"
    if p_long >= th_long:
        return "BUY"
    if (1.0 - p_long) >= (1.0 - th_short):  # симметрично в сторону SHORT
        return "SHORT"
    return "HOLD"
