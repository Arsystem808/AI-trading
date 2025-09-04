# core/ai_inference.py
import os
from typing import Any, Dict, Tuple
import numpy as np

try:
    import joblib  # type: ignore
except Exception as e:
    raise RuntimeError("Не установлен joblib. Добавьте 'joblib' в requirements.txt") from e

MODEL_FILENAMES = {
    "ST": "arxora_lgbm_ST.joblib",
    "MID": "arxora_lgbm_MID.joblib",
    "LT": "arxora_lgbm_LT.joblib",
}

def _models_dir() -> str:
    return os.getenv("ARXORA_MODEL_DIR", "models")

def load_model_for_horizon(hz: str, models_dir: str | None = None) -> tuple[Any, Dict[str, Any], str]:
    """
    Возвращает (model, meta, path).
    Если в .joblib лежит dict, извлекаем поле 'model', остальное отдаём как meta.
    """
    models_dir = models_dir or _models_dir()
    fname = MODEL_FILENAMES.get(hz, MODEL_FILENAMES["ST"])
    path = os.path.join(models_dir, fname)
    if not os.path.exists(path):
        return None, {}, path

    obj = joblib.load(path)

    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        meta = {k: v for k, v in obj.items() if k != "model"}
    else:
        model, meta = obj, {}

    return model, meta, path

def apply_model(model: Any, meta: Dict[str, Any], X) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Делает предсказание вероятности класса 1.
    - Если есть meta['features'] — берём только эти признаки.
    - Если есть meta['scaler'] — применяем его.
    - Если у модели нет predict_proba, используем сигмоиду поверх predict.
    """
    if model is None:
        raise ValueError("Модель не загружена")

    feats = meta.get("features")
    if feats is not None:
        X = X[feats].copy()

    scaler = meta.get("scaler")
    if scaler is not None:
        X[feats] = scaler.transform(X[feats])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # иногда возвращается (n_samples,) — приведём к (n,2) или (n,)
        if isinstance(proba, (list, tuple)):
            proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] > 1:
            proba = proba[:, 1]
        else:
            proba = np.asarray(proba).reshape(-1)
    else:
        # регрессор/бинарный предикт без proba
        yhat = model.predict(X)
        yhat = np.asarray(yhat).reshape(-1)
        # мягкая нормализация в 0..1
        s = yhat.std() + 1e-9
        proba = 1.0 / (1.0 + np.exp(-(yhat - yhat.mean()) / s))

    return proba, meta
