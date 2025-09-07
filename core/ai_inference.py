# core/ai_inference.py
import os, math
import numpy as np

try:
    import joblib  # pip install joblib scikit-learn
except Exception:
    joblib = None

_MODEL_CACHE = {}
_FEATURE_ORDER = [
    "pos", "slope_norm", "atr_d_over_price", "vol_ratio", "band",
    "ha_up_run", "ha_down_run", "macd_pos_run", "macd_neg_run",
    "long_upper", "long_lower"
]

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _vectorize(feats: dict) -> np.ndarray:
    vec = []
    for k in _FEATURE_ORDER:
        v = feats.get(k, 0.0)
        if isinstance(v, bool):
            v = 1.0 if v else 0.0
        try:
            vec.append(float(v))
        except Exception:
            vec.append(0.0)
    return np.array(vec, dtype=float).reshape(1, -1)

def _load_model(hz: str):
    if hz in _MODEL_CACHE:
        return _MODEL_CACHE[hz]
    model_dir = os.getenv("ARXORA_MODEL_DIR", "models")
    fname = {"ST": "ai_st.pkl", "MID": "ai_mid.pkl", "LT": "ai_lt.pkl"}.get(hz, "ai_mid.pkl")
    path = os.path.join(model_dir, fname)
    mdl = None
    if joblib and os.path.exists(path):
        try:
            mdl = joblib.load(path)
        except Exception:
            mdl = None
    _MODEL_CACHE[hz] = mdl
    return mdl

def score_signal(feats: dict, hz: str = "MID"):
    """
    Возвращает словарь {"p_long": 0..1, "src": "model|pseudo"} или None, если
    модели нет и псевдорежим выключен.
    feats — тот же набор фич, который формирует strategy: pos, slope_norm,
    atr_d_over_price, vol_ratio, band, ha_up_run, ha_down_run, macd_pos_run,
    macd_neg_run, long_upper, long_lower (любые могут отсутствовать — будут 0).
    """
    # 1) Псевдо-режим (быстрое эвристическое P(long))
    if int(os.getenv("ARXORA_AI_PSEUDO", "0")) == 1:
        x = feats
        # Логика: ап-сигналы повышают p_long, перекупленность у "крыши" — понижает.
        band = float(x.get("band", 0.0))
        w = (
            0.9 * float(x.get("slope_norm", 0.0)) +
            0.3 * (float(x.get("pos", 0.5)) - 0.5) +
            0.3 * (float(x.get("vol_ratio", 1.0)) - 1.0) +
            0.2 * (float(x.get("ha_up_run", 0.0)) - float(x.get("ha_down_run", 0.0))) * 0.10 +
            0.2 * (float(x.get("macd_pos_run", 0.0)) - float(x.get("macd_neg_run", 0.0))) * 0.07 -
            0.25 * max(0.0, band - 1.5)  # штраф за верхние зоны
        )
        p_long = _sigmoid(6.0 * w)
        return {"p_long": float(max(0.0, min(1.0, p_long))), "src": "pseudo"}

    # 2) Модель из файлов
    mdl = _load_model(hz)
    if mdl is None:
        return None

    x = _vectorize(feats)
    try:
        if hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(x)[0]
            if hasattr(mdl, "classes_") and 1 in list(mdl.classes_):
                idx = list(mdl.classes_).index(1)
            else:
                idx = 1 if len(proba) > 1 else 0
            p_long = float(proba[idx])
        elif hasattr(mdl, "decision_function"):
            score = float(mdl.decision_function(x))
            p_long = float(max(0.0, min(1.0, _sigmoid(score))))
        else:
            # регрессор → преобразуем в [0,1]
            score = float(mdl.predict(x))
            p_long = float(max(0.0, min(1.0, _sigmoid(score))))
        return {"p_long": p_long, "src": "model"}
    except Exception:
        return None
