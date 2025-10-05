# core/ai_inference.py
# ---------------------------------------------------------------------
# Универсальная подгрузка ML-моделей для Arxora + псевдо-оценка.
# Поддерживает имена файлов:
#   models/arxora_lgbm_{HZ}-{TICKER}.joblib
#   models/arxora_lgbm_{HZ}_{TICKER}.joblib
#   models/arxora_lgbm_{HZ}.joblib
#   models/ai_{st,mid,lt}.pkl
# где HZ ∈ {"ST","MID","LT"}, TICKER — верхним регистром (QQQ, ETHUSD, ...).
# Возвращает словарь: {"p_long": float in [0,1], "model_path": str|None, "meta": dict}
# Если модель не найдена и ARXORA_AI_PSEUDO=0 — вернёт None (стратегия тогда
# работает только по правилам). Если ARXORA_AI_PSEUDO=1 — вернёт псевдо-оценку.
# ---------------------------------------------------------------------

from __future__ import annotations

import math
import os
import pickle
import re
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import joblib  # предпочтительно

    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False


# ----------- ENV / глобальные настройки -----------
MODEL_DIR = os.getenv("ARXORA_MODEL_DIR", "models").strip() or "models"
PSEUDO_ON = str(os.getenv("ARXORA_AI_PSEUDO", "0")).strip() not in (
    "0",
    "false",
    "False",
    "",
)
# Пороги в самой стратегии; тут только возвращаем вероятность p_long.

# Кеш загрузок, чтобы не тянуть модель на каждый тик
_MODEL_CACHE: Dict[str, Any] = {}


# ----------- утилиты -----------
def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _tanh(x: float) -> float:
    # плавное ограничение для псевдо-оценки
    return math.tanh(x)


def _sanitize_ticker(t: Optional[str]) -> str:
    if not t:
        return ""
    t = t.upper().strip()
    # Уберём префикс "X:" для крипты, если есть (пример "X:BTCUSD")
    if t.startswith("X:"):
        t = t[2:]
    # Разрешим только буквы/цифры
    t = re.sub(r"[^A-Z0-9]", "", t)
    return t


def _possible_paths(hz: str, ticker: Optional[str]) -> Tuple[str, ...]:
    hz = hz.upper()
    t = _sanitize_ticker(ticker)
    cand = []
    if t:
        cand.append(os.path.join(MODEL_DIR, f"arxora_lgbm_{hz}-{t}.joblib"))
        cand.append(os.path.join(MODEL_DIR, f"arxora_lgbm_{hz}_{t}.joblib"))
    cand.append(os.path.join(MODEL_DIR, f"arxora_lgbm_{hz}.joblib"))
    cand.append(os.path.join(MODEL_DIR, f"ai_{hz.lower()}.pkl"))
    return tuple(cand)


def _load_model(path: str) -> Any:
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".joblib") and _HAVE_JOBLIB:
            model = joblib.load(path)
        else:
            with open(path, "rb") as f:
                model = pickle.load(f)
        _MODEL_CACHE[path] = model
        return model
    except Exception:
        return None


def _infer_proba_for_long(model: Any, Xrow: np.ndarray) -> Optional[float]:
    """
    Пытаемся получить p(long) из модели.
    Xrow: shape (1, n_features)
    """
    # sklearn-like API
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xrow)
            # classes_ может быть [0,1] или ['short','long'] и т.п.
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if "long" in classes:
                    j = classes.index("long")
                elif 1 in classes:
                    j = classes.index(1)
                else:
                    # fallback: берём вторую колонку
                    j = 1 if proba.shape[1] > 1 else 0
            else:
                j = 1 if proba.shape[1] > 1 else 0
            return float(proba[0, j])

        # decision_function -> сигмоид
        if hasattr(model, "decision_function"):
            z = float(model.decision_function(Xrow).reshape(-1)[0])
            return _clip01(1.0 / (1.0 + math.exp(-z)))

        # predict -> хард-класс, сделаем мягким 0.75/0.25
        if hasattr(model, "predict"):
            y = model.predict(Xrow)
            y = y[0] if hasattr(y, "__len__") else y
            if isinstance(y, str):
                return 0.75 if y.lower() == "long" else 0.25
            return 0.75 if int(y) == 1 else 0.25
    except Exception:
        pass
    return None


def _features_to_row(feats: Dict[str, Any], model: Any) -> np.ndarray:
    """
    Подгоняем порядок фич под модель.
    - Если у модели есть feature_names_in_, используем их.
    - Иначе берём алфавитный порядок ключей feats.
    Отсутствующие фичи заполняем 0.0, bool -> 0/1.
    """
    # 1) Попробуем достать признаки у модели/последнего шага пайплайна
    names = None
    m = model
    try:
        # sklearn pipeline: берём последний шаг
        if hasattr(m, "steps") and len(m.steps) > 0:
            m = m.steps[-1][1]
    except Exception:
        pass

    if hasattr(m, "feature_names_in_"):
        names = list(m.feature_names_in_)

    row = []
    if names:
        for k in names:
            v = feats.get(k, 0.0)
            if isinstance(v, bool):
                v = 1.0 if v else 0.0
            row.append(float(v))
    else:
        # стабильный порядок: отсортированные ключи
        for k in sorted(feats.keys()):
            v = feats[k]
            if isinstance(v, bool):
                v = 1.0 if v else 0.0
            row.append(float(v))

    X = np.array(row, dtype=float).reshape(1, -1)
    return X


# ----------- псевдо-оценка (эвристика) -----------
def _pseudo_score(feats: Dict[str, Any], hz: str) -> float:
    """
    Мягкая «интуиция» по твоей логике:
      - pos (позиция в диапазоне) — у дна +, у верха -;
      - slope_norm — положит. наклон +;
      - vol_ratio — растущая волатильность слегка +;
      - серии HA/MACD: long зеленая серия у «крыши» — минус к long;
                       длинная красная у «дна» — плюс к long;
      - дополнительные флаги: band/long_upper/long_lower, если есть.
    """
    hz = hz.upper()
    pos = float(feats.get("pos", 0.5))
    slope_norm = float(feats.get("slope_norm", 0.0))
    vol_ratio = float(feats.get("vol_ratio", 1.0))

    ha_up = float(feats.get("ha_up_run", 0.0))
    ha_dn = float(feats.get("ha_down_run", 0.0))
    macd_up = float(feats.get("macd_pos_run", 0.0))
    macd_dn = float(feats.get("macd_neg_run", 0.0))

    band = float(feats.get("band", 0.0))  # -3..+3, если есть
    long_upper = bool(feats.get("long_upper", False))
    long_lower = bool(feats.get("long_lower", False))

    # База
    p = 0.5

    # Наклон (плавно) — важен
    p += 0.25 * _tanh(120.0 * slope_norm)

    # Позиция в диапазоне — не гонимся за вершинами
    if pos >= 0.80:
        p -= 0.12
    elif pos <= 0.20:
        p += 0.12

    # Волатильность
    if vol_ratio > 1.05:
        p += 0.03
    elif vol_ratio < 0.95:
        p -= 0.03

    # Серии HA/MACD — пороги по горизонту
    thr_ha = {"ST": 4, "MID": 5, "LT": 6}.get(hz, 5)
    thr_mc = {"ST": 4, "MID": 6, "LT": 8}.get(hz, 6)

    # Длинная зелёная серия у верха — штраф лонгу
    if ha_up >= thr_ha or macd_up >= thr_mc:
        if pos >= 0.65 or band >= +1:
            p -= 0.10

    # Длинная красная серия у низа — бонус лонгу
    if ha_dn >= thr_ha or macd_dn >= thr_mc:
        if pos <= 0.35 or band <= -1:
            p += 0.10

    # Флаги теней — косвенно
    if long_upper and pos >= 0.60:
        p -= 0.03
    if long_lower and pos <= 0.40:
        p += 0.03

    # Лёгкая нормализация и клип
    return _clip01(p)


# ----------- публичная функция -----------
def score_signal(
    feats: Dict[str, Any], hz: str, ticker: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    feats: dict с признаками (любые из:
        pos, slope_norm, atr_d_over_price, vol_ratio,
        ha_up_run, ha_down_run, macd_pos_run, macd_neg_run,
        band, long_upper, long_lower, ...
    )
    hz: "ST" | "MID" | "LT"
    ticker: тикер (желательно).
    """
    hz = (hz or "").upper()
    paths = _possible_paths(hz, ticker)

    model = None
    used_path = None
    for p in paths:
        m = _load_model(p)
        if m is not None:
            model = m
            used_path = p
            break

    if model is not None:
        try:
            Xrow = _features_to_row(feats, model)
            p_long = _infer_proba_for_long(model, Xrow)
            if p_long is None:
                # если модель не умеет proba — fallback к псевдо
                if PSEUDO_ON:
                    p_long = _pseudo_score(feats, hz)
                    return {
                        "p_long": float(p_long),
                        "model_path": "pseudo_fallback",
                        "meta": {"reason": "no_proba"},
                    }
                return None
            # лёгкое ограничение, чтобы не было 0/1
            p_long = float(max(0.01, min(0.99, p_long)))
            return {"p_long": p_long, "model_path": used_path, "meta": {}}
        except Exception:
            # повреждённая модель/неожиданные фичи
            if PSEUDO_ON:
                p_long = _pseudo_score(feats, hz)
                return {
                    "p_long": float(p_long),
                    "model_path": "pseudo_fallback",
                    "meta": {"error": "model_inference_failed"},
                }
            return None

    # Модель не найдена
    if PSEUDO_ON:
        p_long = _pseudo_score(feats, hz)
        return {
            "p_long": float(p_long),
            "model_path": "pseudo",
            "meta": {"missing_model": True},
        }

    return None
