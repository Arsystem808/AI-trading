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
