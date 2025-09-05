# core/ai_inference.py
# -*- coding: utf-8 -*-
"""
Загрузка ML-моделей Arxora и инференс.

Поддержка:
- Файлы моделей: models/arxora_lgbm_{ST|MID|LT}.joblib
  или тикер-специфично: models/arxora_lgbm_{ST|MID|LT}_{TICKER}.joblib
- Формат joblib: либо сам estimator, либо {"model": estimator, "meta": {...}}
- Пороги решений читаются из переменных окружения/секретов:
    ARXORA_AI_TH_LONG  (по умолчанию 0.55)
    ARXORA_AI_TH_SHORT (по умолчанию 0.45)
- Совместимость со sklearn/LightGBM/XGBoost

API:
    load_model(horizon: str, ticker: str | None = None) -> (model, meta)
    predict_proba(model, X_df) -> float   # вероятность класса 1
    decide_action(proba: float,
                  th_long: float = TH_LONG,
                  th_short: float = TH_SHORT) -> str  # "BUY"/"SHORT"/"WAIT"
    is_ai_available(horizon: str, ticker: str | None = None) -> bool
    model_path(horizon: str, ticker: str | None = None) -> str
"""

from __future__ import annotations

import os
import re
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd


# ---- конфигурация из окружения/секретов -------------------------------------

MODEL_DIR = os.getenv("ARXORA_MODEL_DIR", "models").strip()
TH_LONG = float(os.getenv("ARXORA_AI_TH_LONG", "0.55"))
TH_SHORT = float(os.getenv("ARXORA_AI_TH_SHORT", "0.45"))


# ---- утилиты -----------------------------------------------------------------

def _hz_tag(text: str) -> str:
    """Нормализует название горизонта в ST/MID/LT."""
    t = (text or "").upper()
    if "КРАТКО" in t or "ST" in t:
        return "ST"
    if "СРЕДНЕ" in t or "MID" in t:
        return "MID"
    return "LT"


def _clean_symbol(sym: str) -> str:
    """X:BTCUSD -> BTCUSD; AAPL -> AAPL (оставляем только A-Z0-9)."""
    return re.sub(r"[^A-Z0-9]+", "", (sym or "").upper())


def model_path(horizon: str, ticker: str | None = None) -> str:
    """
    Возвращает путь к файлу модели. Приоритет:
    1) модель по тикеру (если файл существует)
    2) общая модель по горизонту
    """
    hz = _hz_tag(horizon)
    if ticker:
        t = _clean_symbol(ticker)
        p_ticker = os.path.join(MODEL_DIR, f"arxora_lgbm_{hz}_{t}.joblib")
        if os.path.exists(p_ticker):
            return p_ticker
    return os.path.join(MODEL_DIR, f"arxora_lgbm_{hz}.joblib")


def is_ai_available(horizon: str, ticker: str | None = None) -> bool:
    return os.path.exists(model_path(horizon, ticker))


def _align_features(X: pd.DataFrame, meta: Dict[str, Any] | None) -> pd.DataFrame:
    """
    Если в meta есть список признаков, выровнять порядок/пропуски.
    Лишние столбцы отбрасываются, недостающие — добавляются нулями.
    """
    if not isinstance(meta, dict):
        return X
    feats = meta.get("features")
    if not feats:
        return X
    X = X.copy()
    for f in feats:
        if f not in X.columns:
            X[f] = 0.0
    X = X[feats]
    return X


# ---- загрузка модели ---------------------------------------------------------

def load_model(horizon: str, ticker: str | None = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Загружает модель и метаданные.
    Поддерживает файлы-словари {"model": ..., "meta": {...}}.
    """
    path = model_path(horizon, ticker)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Модель не найдена: {path}")

    obj = joblib.load(path)
    # Если файл — словарь
    if isinstance(obj, dict):
        model = obj.get("model", obj)
        meta = obj.get("meta", {})
        return model, meta
    # Иначе это сам estimator
    return obj, {}


# ---- инференс ----------------------------------------------------------------

def _predict_proba_any(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Унифицированное получение вероятностей:
    - sklearn/LightGBM/XGBoost (классификатор с predict_proba)
    - Booster/модели без predict_proba -> fallback на predict/decision_function
    Возвращает массив shape (n_samples,) с P(y=1).
    """
    # 1) стандартный путь
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        p = np.asarray(p)
        if p.ndim == 1:
            return p  # уже (n,)
        # берём столбец класса 1
        return p[:, 1]

    # 2) некоторые бустинги (raw predict)
    try:
        p = model.predict(X)
        p = np.asarray(p)
        # если это уже вероятности (0..1)
        if p.ndim == 1:
            # Иногда модели выдают логиты — попробуем привести к (0..1)
            if p.min() < 0 or p.max() > 1:
                p = 1 / (1 + np.exp(-p))
            return p
        # двумерный выход — берём столбец 1
        return p[:, 1]
    except Exception:
        pass

    # 3) decision_function -> логистическая сигмоида
    if hasattr(model, "decision_function"):
        s = np.asarray(model.decision_function(X))
        return 1 / (1 + np.exp(-s))

    raise RuntimeError("Не удалось получить вероятности от модели.")


def predict_proba(model: Any, X_df: pd.DataFrame, meta: Dict[str, Any] | None = None) -> float:
    """
    Возвращает вероятность P(y=1) для последней строки X_df.
    Выполняет выравнивание признаков при наличии списка в meta["features"].
    """
    if not isinstance(X_df, pd.DataFrame):
        X_df = pd.DataFrame(X_df)
    X = _align_features(X_df, meta)
    proba = _predict_proba_any(model, X)
    return float(np.asarray(proba).reshape(-1)[-1])


def decide_action(proba: float,
                  th_long: float = TH_LONG,
                  th_short: float = TH_SHORT) -> str:
    """
    Простое правило:
        proba >= th_long  -> "BUY"
        proba <= th_short -> "SHORT"
        иначе              "WAIT"
    """
    if proba >= th_long:
        return "BUY"
    if proba <= th_short:
        return "SHORT"
    return "WAIT"
