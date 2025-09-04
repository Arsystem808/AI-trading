# core/ai_inference.py
import os, math
import joblib
import numpy as np

_MODELS = {}  # кэш {"ST": clf, ...}

def _model_path(hz: str) -> str:
    base = os.getenv("ARXORA_MODEL_DIR", "models")
    return os.path.join(base, f"arxora_lgbm_{hz.upper()}.joblib")

def _get_model(hz: str):
    hz = hz.upper()
    if hz in _MODELS:
        return _MODELS[hz]
    path = _model_path(hz)
    try:
        clf = joblib.load(path)
        _MODELS[hz] = clf
    except Exception:
        _MODELS[hz] = None
    return _MODELS[hz]

def _pseudo_prob_long(feat: dict) -> float:
    """
    Псевдо-ML на основе твоих признаков: даёт вероятности, если модели нет.
    Отключается через ARXORA_AI_PSEUDO=0.
    """
    if os.getenv("ARXORA_AI_PSEUDO", "1") not in ("1", "true", "True"):
        return None

    pos = float(feat.get("pos", 0.5))                 # 0..1
    slope = float(feat.get("slope_norm", 0.0)) * 100  # наклон, усилим масштаб
    atrp = float(feat.get("atr_d_over_price", 0.0)) * 100
    volr = float(feat.get("vol_ratio", 1.0)) - 1.0    # центрируем на 0
    streak = float(feat.get("streak", 0.0))
    band = float(feat.get("band", 0.0))
    lu = 1.0 if feat.get("long_upper") else 0.0
    ll = 1.0 if feat.get("long_lower") else 0.0

    # простая логистика: тренд+, низ диапазона+, верх диапазона-, рост волы-, длинная серия+,-
    z  =  3.00 * slope
    z += -0.80 * ((pos - 0.5) * 2.0)      # предпочитаем нижнюю/среднюю часть диапазона
    z += -0.50 * atrp
    z += -0.40 * (volr * 3.0)
    z += -0.15 * streak                   # после длинной серии вверх — аккуратнее
    z += -0.25 * band                     # ближе к R-зонам — консервативнее
    z += -0.60 * lu + 0.60 * ll           # «тень сверху» минус, «тень снизу» плюс
    z +=  0.20                            # небольшой сдвиг в сторону long

    p = 1.0 / (1.0 + math.exp(-z))
    return float(max(0.0, min(1.0, p)))

def score_signal(feat: dict, hz: str):
    """
    Возвращает {"p_long": float, "p_short": float}:
      – сначала пробуем реальную модель;
      – если её нет — используем псевдо-ML (если не отключён).
    """
    clf = _get_model(hz)
    if clf is not None:
        x = np.array([[
            float(feat.get("pos", 0.5)),
            float(feat.get("slope_norm", 0.0)),
            float(feat.get("atr_d_over_price", 0.0)),
            float(feat.get("vol_ratio", 1.0)),
            float(feat.get("streak", 0.0)),
            float(feat.get("band", 0.0)),
            1.0 if feat.get("long_upper") else 0.0,
            1.0 if feat.get("long_lower") else 0.0,
        ]], dtype=float)
        try:
            p_long = float(clf.predict_proba(x)[0, 1])
        except Exception:
            pred = float(np.ravel(clf.predict(x))[0])
            p_long = 1.0 / (1.0 + math.exp(-pred))
        return {"p_long": p_long, "p_short": 1.0 - p_long}

    # псевдо-ML
    p_long = _pseudo_prob_long(feat)
    if p_long is None:
        return None
    return {"p_long": p_long, "p_short": 1.0 - p_long}
