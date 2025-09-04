# core/ai_inference.py
import os, joblib, numpy as np, pandas as pd
from pathlib import Path

def _safe_sigmoid(x):  # для уверенности из логитов, если надо
    return 1.0 / (1.0 + np.exp(-x))

def _ensure_feats(dfX: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    for f in feats:
        if f not in dfX.columns:
            dfX[f] = 0.0
    return dfX[feats].astype(float)

def load_model(hz_tag: str):
    d = os.getenv("ARXORA_MODEL_DIR", "models")
    p = Path(d) / f"arxora_lgbm_{hz_tag}.joblib"
    if not p.exists():
        return None
    obj = joblib.load(p)
    return obj  # {"model":..., "features":[...], "auc":...}

def predict_up_proba(model_obj, dfX_row: pd.DataFrame) -> float:
    feats = model_obj["features"]
    X = _ensure_feats(dfX_row.copy(), feats)
    proba = float(model_obj["model"].predict_proba(X)[:,1][0])
    return max(0.0, min(1.0, proba))
