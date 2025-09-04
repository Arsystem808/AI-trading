# core/ai_inference.py
import os
import math
import joblib
import numpy as np

# Ленивая загрузка моделей по горизонту
_MODELS = {}  # {"ST": clf, "MID": clf, "LT": clf}

def _model_path(hz: str) -> str:
    # Можно переопределить через ENV, иначе берём из ./models/
    base = os.getenv("ARXORA_MODEL_DIR", "models")
    return os.path.join(base, f"arxora_lgbm_{hz}.joblib")

def _get_model(hz: str):
    hz = hz.upper()
    if hz in _MODELS:
        return _MODELS[hz]
    path = _model_path(hz)
    try:
        clf = joblib.load(path)
        _MODELS[hz] = clf
        return clf
    except Exception:
        _MODELS[hz] = None
        return None

def score_signal(feat: dict, hz: str):
    """
    feat — словарь признаков. Возвращает:
      {"p_long": float, "p_short": float} или None (если модели нет).
    """
    clf = _get_model(hz)
    if clf is None:
        return None

    # Минимальный вектор признаков (синхронен со стратегией)
    x = np.array([[
        float(feat.get("pos", 0.5)),
        float(feat.get("slope_norm", 0.0)),
        float(feat.get("atr_d_over_price", 0.0)),
        float(feat.get("vol_ratio", 1.0)),
        float(feat.get("streak", 0.0)),
        float(feat.get("band", 0)),
        1.0 if feat.get("long_upper") else 0.0,
        1.0 if feat.get("long_lower") else 0.0,
    ]], dtype=float)

    # Ожидается бинарная классификация: proba[:,1] — вероятность long
    try:
        proba = clf.predict_proba(x)[0, 1]
    except Exception:
        # На случай регрессии или модели без proba — используем predict и сигмоиду
        pred = float(np.ravel(clf.predict(x))[0])
        proba = 1.0 / (1.0 + math.exp(-pred))

    p_long = float(proba)
    p_short = float(1.0 - p_long)
    return {"p_long": p_long, "p_short": p_short}
