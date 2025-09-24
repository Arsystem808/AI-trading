from pathlib import Path
import os, joblib, numpy as np

try:
    import shap
except Exception:
    shap = None

MODELS = (Path(__file__).resolve().parents[1] / "models").resolve()

def load_optional(path: Path):
    return joblib.load(path) if os.path.exists(path) else None

def intervals_from_quantiles(ticker: str, family: str, x: np.ndarray):
    # Ожидаются модели с objective='quantile', alpha=0.1 и 0.9
    p10 = load_optional(MODELS / f"{family}_{ticker}_p10.joblib")
    p90 = load_optional(MODELS / f"{family}_{ticker}_p90.joblib")
    if p10 is None or p90 is None or x is None:
        return None
    lo = float(np.ravel(p10.predict(x))[0])
    hi = float(np.ravel(p90.predict(x))[0])
    return {"low": lo, "high": hi, "width": hi - lo}

def shap_breakdown(model, x: np.ndarray, feature_names=None, top_k=8):
    if shap is None or x is None:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(x)
        vals = sv if isinstance(sv, np.ndarray) else sv[0]
        vals = vals[0]
        order = np.argsort(np.abs(vals))[::-1][:top_k]
        feats = feature_names if feature_names is not None else [f"f{i}" for i in range(len(vals))]
        return [{"feature": feats[i], "shap": float(vals[i])} for i in order]
    except Exception:
        return None
# --- UI adapter API ---
def ui_get_confidence_breakdown(ticker: str) -> dict:
    """
    Минимальный контракт для Streamlit:
    {
      "signal": "BUY/SELL",
      "overall_confidence_pct": float,
      "breakdown": {"rules_pct": float, "ai_override_delta_pct": float},
      "shap_top": [{"feature": str, "shap": float}, ...]
    }
    """
    # 1) Базовая уверенность правил (должна существовать в вашем модуле правил)
    try:
        from core.rules import rules_confidence_for_ticker
        rules_pct = float(rules_confidence_for_ticker(ticker))
    except Exception:
        rules_pct = 44.0  # безопасный дефолт, чтобы UI не пустовал

    # 2) Уверенность ИИ (попытка классификатор → margin → регрессия → дефолт)
    ai_delta = 0.0
    shap_top = []
    try:
        import numpy as np
        from core.model_loader import load_model_for  # ожидается в проекте
        from core.ai_inference import build_features_for_ticker  # ожидается в проекте

        model = load_model_for(ticker)
        X = build_features_for_ticker(ticker)  # shape: (1, n_features)

        prob1 = None
        point = None

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            prob1 = float(np.ravel(proba[:, 1])[0])
        elif hasattr(model, "predict"):
            point = float(np.ravel(model.predict(X))[0])

        def clamp01(v: float) -> float:
            return max(0.0, min(1.0, float(v)))

        def _sigmoid(z: float) -> float:
            try:
                return 1.0 / (1.0 + np.exp(-z))
            except Exception:
                return 0.5

        if prob1 is not None:
            ai_conf = clamp01(prob1) * 100.0
        else:
            margin = None
            if hasattr(model, "decision_function"):
                try:
                    margin = float(np.ravel(model.decision_function(X))[0])
                except Exception:
                    margin = None
            if margin is not None:
                ai_conf = clamp01(2.0 * _sigmoid(abs(margin)) - 1.0) * 100.0
            elif point is not None:
                ai_conf = clamp01(np.tanh(abs(point))) * 100.0
            else:
                ai_conf = 50.0

        # Дельта ИИ относительно нейтрали 50%
        ai_delta = float(ai_conf - 50.0)

        # SHAP (опционально, не роняем UI)
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

    # Итоговый сигнал (пример: положительная дельта → BUY)
    signal = "BUY" if ai_delta >= 0 else "SELL"

    overall = float(max(0.0, min(100.0, rules_pct + ai_delta)))
    return {
        "signal": signal,
        "overall_confidence_pct": overall,
        "breakdown": {"rules_pct": float(rules_pct), "ai_override_delta_pct": float(ai_delta)},
        "shap_top": shap_top,
    }

if __name__ == "__main__":
    import sys, json
    t = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    print(json.dumps(ui_get_confidence_breakdown(t), ensure_ascii=False))
