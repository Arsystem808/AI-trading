import json
import sys

import numpy as np

from core.confidence import intervals_from_quantiles, shap_breakdown
from core.model_loader import load_model_for
from core.rules import rule_score


def clamp01(x):
    return max(0.0, min(1.0, x))


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/signal_with_confidence.py <TICKER>")
        sys.exit(2)

    ticker = sys.argv[1]
    model = load_model_for(ticker)

    # Синтетический батч признаков по числу фич
    n = getattr(model, "n_features_in_", None)
    x = np.zeros((1, n)) if n else None

    is_clf = hasattr(model, "predict_proba")
    point = None
    prob1 = None

    if is_clf and x is not None:
        try:
            proba = model.predict_proba(x)
            if proba is not None and len(proba.shape) == 2 and proba.shape[1] >= 2:
                prob1 = float(proba[0, -1])
        except Exception:
            prob1 = None
    else:
        if x is not None:
            try:
                point = float(np.ravel(model.predict(x))[0])
            except Exception:
                point = None

    # Сигнал
    if prob1 is not None:
        signal = "BUY" if prob1 >= 0.5 else "SELL"
    else:
        signal = "BUY" if (point is not None and point >= 0) else "SELL"

    # Интервалы для регрессии (если доступны квантильные модели)
    family_guess = "global"
    intervals = intervals_from_quantiles(ticker, family_guess, x)

    # Перевод уверенности ИИ
    if prob1 is not None:
        ai_conf = clamp01(prob1) * 100.0
    else:
        if intervals and point is not None:
            width = abs(intervals["width"])
            ai_conf = clamp01(1.0 - (width / (abs(point) + 1e-6))) * 100.0
        else:
            ai_conf = 50.0  # нейтрально при отсутствии интервалов

    # Базовые правила и разложение
    rule_pct = clamp01(rule_score(ticker)) * 100.0
    ai_delta = ai_conf - 50.0  # ИИ корректирует базовую уверенность
    overall = max(0.0, min(100.0, rule_pct + ai_delta))

    # SHAP топ факторов
    shap_top = shap_breakdown(model, x)

    out = {
        "ticker": ticker,
        "signal": signal,
        "overall_confidence_pct": round(overall, 1),
        "breakdown": {
            "rules_pct": round(rule_pct, 1),
            "ai_override_delta_pct": round(ai_delta, 1),
        },
        "point": point,
        "interval": intervals,
        "shap_top": shap_top,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
