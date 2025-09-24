from pathlib import Path
import os, joblib

# Абсолютный путь до папки models от корня репозитория
MODELS = (Path(__file__).resolve().parents[1] / "models").resolve()

def load_model_for(ticker: str):
    # Порядок fallback: тикерные -> универсальная global_SPY.joblib
    candidates = [
        MODELS / f"alphapulse_{ticker}.joblib",
        MODELS / f"octopus_{ticker}.joblib",
        MODELS / f"global_{ticker}.joblib",
        MODELS / "global_SPY.joblib",
    ]
    for p in candidates:
        if os.path.exists(p):
            return joblib.load(p)
    raise FileNotFoundError(f"No model found for {ticker}: " + ", ".join(str(c) for c in candidates))
