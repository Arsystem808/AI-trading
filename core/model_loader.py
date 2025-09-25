# 1) core/model_loader.py
cat > core/model_loader.py <<'PY'
from pathlib import Path
import joblib

MODELS = Path("models")

def load_model_for(ticker: str):
    t = ticker.upper()
    candidates = [
        MODELS / f"arxora_m7pro_{t}.joblib",
        MODELS / f"global_{t}.joblib",
        MODELS / f"alphapulse_{t}.joblib",
        MODELS / f"octopus_{t}.joblib",
        MODELS / "m7_model.pkl",
    ]
    for p in candidates:
        if p.exists():
            return joblib.load(p)
    return None
PY
