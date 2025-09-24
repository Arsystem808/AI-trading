from pathlib import Path
import joblib

# Абсолютный путь до папки models от корня репозитория
MODELS = (Path(__file__).resolve().parents[1] / "models").resolve()

def load_model_for(ticker: str):
    t = (ticker or "").strip().upper()

    candidates = [
        # Агент‑специфичные файлы (включая M7pro)
        MODELS / f"m7pro_{t}.joblib",
        MODELS / f"m7pro_global.joblib",
        MODELS / "m7pro.joblib",

        MODELS / f"alphapulse_{t}.joblib",
        MODELS / f"octopus_{t}.joblib",
        MODELS / f"global_{t}.joblib",

        # Универсальные глобальные фоллбеки
        MODELS / "global_SPY.joblib",
    ]

    for p in candidates:
        if p.exists():                      # Path.exists — нативно и надёжно
            return joblib.load(p)           # joblib.load принимает Path
    # ВАЖНО: безопасный возврат, без исключения — стратегия уйдёт в fallback
    return None
