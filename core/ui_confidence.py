# core/ui_confidence.py
# Адаптер для Streamlit: вычисляет breakdown из правил и ИИ с безопасными fallback'ами.

from __future__ import annotations

from typing import Any, Dict, List
import math

def _clamp01(v: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except Exception:
        return 0.0

def _sigmoid(z: float) -> float:
    # Нужен для преобразования margin -> уверенность
    try:
        import numpy as np  # локальный импорт на случай отсутствия numpy в ранней инициализации
        return 1.0 / (1.0 + np.exp(-float(z)))
    except Exception:
        # нейтральная уверенность, если что-то пошло не так
        return 0.5

def ui_get_confidence_breakdown(ticker: str) -> Dict[str, Any]:
    """
    Возвращает словарь для UI:
    {
      "signal": "BUY/SELL",
      "overall_confidence_pct": float,
      "breakdown": {"rules_pct": float, "ai_override_delta_pct": float},
      "shap_top": [{"feature": str, "shap": float}, ...]
    }
    """
    # 1) Базовая уверенность по правилам
    try:
        from core.rules import rules_confidence_for_ticker
        rules_pct = float(rules_confidence_for_ticker(ticker))
    except Exception:
        rules_pct = 44.0  # безопасный дефолт, чтобы UI не пустовал

    # 2) ИИ‑часть: пытаемся получить proba → margin → регрессию → нейтрально
    ai_delta = 0.0
    shap_top: List[Dict[str, Any]] = []

    try:
        import numpy as np
        from core.model_loader import load_model_for
        from core.ai_inference import build_features_for_ticker

        model = load_model_for(ticker)
        X = build_features_for_ticker(ticker)  # ожидается (1, n_features) или совместимая форма

        prob1 = None
        point = None

        # Классификация
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                prob1 = float(np.ravel(proba[:, 1])[0])
            except Exception:
                prob1 = None

        # Регрессия / общий predict
        if prob1 is None and hasattr(model, "predict"):
            try:
                point = float(np.ravel(model.predict(X))[0])
            except Exception:
                point = None

        # Перевод уверенности ИИ
        if prob1 is not None:
            ai_conf = _clamp01(prob1) * 100.0
        else:
            # Попытка через margin (decision_function)
            margin = None
            if hasattr(model, "decision_function"):
                try:
                    margin = float(np.ravel(model.decision_function(X))[0])
                except Exception:
                    margin = None

            if margin is not None and math.isfinite(margin):
                ai_conf = _clamp01(2.0 * _sigmoid(abs(margin)) - 1.0) * 100.0
            elif point is not None and math.isfinite(point):
                # Регрессионный сигнал -> уверенность через насыщаемую функцию
                ai_conf = _clamp01(np.tanh(abs(point))) * 100.0
            else:
                ai_conf = 50.0  # нейтрально

        ai_delta = float(ai_conf - 50.0)

        # SHAP (опционально, без падения UI)
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
        # Если модель/фичи недоступны — не ломаем UI
        ai_delta = 0.0
        shap_top = []

    # Итоговый сигнал: положительная дельта -> BUY, отрицательная -> SELL
    signal = "BUY" if ai_delta >= 0 else "SELL"
    overall = float(max(0.0, min(100.0, rules_pct + ai_delta)))

    return {
        "signal": signal,
        "overall_confidence_pct": overall,
        "breakdown": {
            "rules_pct": float(rules_pct),
            "ai_override_delta_pct": float(ai_delta),
        },
        "shap_top": shap_top,
    }

if __name__ == "__main__":
    # Ручной тест: python -m core.ui_confidence SPY
    import sys, json
    t = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    print(json.dumps(ui_get_confidence_breakdown(t), ensure_ascii=False))
