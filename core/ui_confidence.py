# core/ui_confidence.py
# Адаптер для UI: считает разбор уверенности из "базовых правил" и "ИИ", с безопасными fallback'ами.

from __future__ import annotations
from typing import Any, Dict, List
import math

def _clamp01(v: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except Exception:
        return 0.0

def _sigmoid(z: float) -> float:
    try:
        import numpy as np
        return 1.0 / (1.0 + np.exp(-float(z)))
    except Exception:
        return 0.5

def compute_breakdown_from_overall(rules_pct: float, overall_pct: float) -> Dict[str, float]:
    # Консистентная дельта ИИ = overall - rules, ограниченная 0..100
    raw_delta = float(overall_pct) - float(rules_pct)
    ai_delta = max(-float(rules_pct), min(100.0 - float(rules_pct), raw_delta))
    return {"rules_pct": float(rules_pct), "ai_override_delta_pct": float(ai_delta)}

def ui_get_confidence_breakdown_via_model(ticker: str) -> Dict[str, Any]:
    # Пытаемся оценить уверенность ИИ по модели (proba → margin → регрессия), иначе нейтрально.
    rules_pct = 44.0
    ai_delta = 0.0
    shap_top: List[Dict[str, Any]] = []
    try:
        from core.rules import rules_confidence_for_ticker
        rules_pct = float(rules_confidence_for_ticker(ticker))
    except Exception:
        rules_pct = 44.0

    try:
        import numpy as np
        from core.model_loader import load_model_for
        from core.ai_inference import build_features_for_ticker

        model = load_model_for(ticker)
        X = build_features_for_ticker(ticker)

        prob1 = None
        point = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                prob1 = float(np.ravel(proba[:, 1])[0])
            except Exception:
                prob1 = None
        if prob1 is None and hasattr(model, "predict"):
            try:
                point = float(np.ravel(model.predict(X))[0])
            except Exception:
                point = None

        if prob1 is not None:
            ai_conf = _clamp01(prob1) * 100.0
        else:
            margin = None
            if hasattr(model, "decision_function"):
                try:
                    margin = float(np.ravel(model.decision_function(X))[0])
                except Exception:
                    margin = None
            if margin is not None and math.isfinite(margin):
                ai_conf = _clamp01(2.0 * _sigmoid(abs(margin)) - 1.0) * 100.0
            elif point is not None and math.isfinite(point):
                ai_conf = _clamp01(np.tanh(abs(point))) * 100.0
            else:
                ai_conf = 50.0

        ai_delta = float(ai_conf - 50.0)

        # SHAP (опционально)
        try:
            import shap
            explainer = shap.Explainer(model)
            sv = explainer(X)
            vals = sv.values[0]
            names = getattr(sv, "feature_names", [f"f{i}" for i in range(len(vals))])
            order = np.argsort(-np.abs(vals))[:8]
            shap_top = [{"feature": names[i], "shap": float(vals[i])} for i in order]
        except Exception:
            shap_top = []
    except Exception:
        ai_delta = 0.0
        shap_top = []

    signal = "BUY" if ai_delta >= 0 else "SELL"
    overall = float(max(0.0, min(100.0, rules_pct + ai_delta)))
    return {
        "signal": signal,
        "overall_confidence_pct": overall,
        "breakdown": {"rules_pct": float(rules_pct), "ai_override_delta_pct": float(ai_delta)},
        "shap_top": shap_top,
    }
