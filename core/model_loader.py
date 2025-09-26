# core/model_loader.py
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Optional, Dict, Union

try:
    import joblib  # внешняя библиотека joblib, не sklearn.externals
except Exception:
    joblib = None  # мягкий импорт


# Директории для артефактов
MODELS_DIR = Path("models")
CONFIG_DIR = Path("configs")


# Нормализация текста для устранения «invalid character» в JSON
_FANCY = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "’": "'", "‘": "'",
    "—": "-", "–": "-",
    "，": ",", "：": ":", "；": ";"
}

def _normalize_text(s: str) -> str:
    # Заменяем типографские символы
    for k, v in _FANCY.items():
        s = s.replace(k, v)
    # Превращаем ': 'value'' -> ': "value"' (частый почти‑JSON)
    s = re.sub(r'(:\s*)\'([^\'\\n]*)\'', r'\1"\2"', s)
    # Удаляем запятую перед закрывающей скобкой/фигурной
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    return s


def _load_json_file(p: Path) -> Any:
    raw = p.read_text(encoding="utf-8", errors="replace")
    norm = _normalize_text(raw)
    try:
        return json.loads(norm)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {p}: {e}")


def _extract_feature_names_if_any(model: Any) -> Optional[list]:
    """
    Пытаемся получить список имён признаков у LightGBM моделей.
    Для sklearn-обёрток: .booster_.feature_name()
    Для Booster: .feature_name()
    Возвращаем None, если не удалось.
    """
    try:
        # sklearn API обёртка
        if hasattr(model, "booster_") and model.booster_ is not None:
            booster = model.booster_
            if hasattr(booster, "feature_name"):
                names = booster.feature_name()
                return list(names) if names is not None else None
        # raw Booster
        if hasattr(model, "feature_name"):
            names = model.feature_name()
            return list(names) if names is not None else None
        # некоторые версии экспонируют feature_name_
        if hasattr(model, "feature_name_"):
            names = getattr(model, "feature_name_")
            return list(names) if names is not None else None
    except Exception:
        return None
    return None


def _try_joblib(p: Path) -> Any:
    if not joblib:
        raise RuntimeError("joblib is not available to load model files")
    return joblib.load(p)


def load_model_for(ticker_or_name: str) -> Optional[Union[Any, Dict[str, Any]]]:
    """
    Универсальный загрузчик:
    - Пытается найти веса по приоритету для тикера/агента (m7pro/global/alphapulse/octopus).
    - Поддерживает .joblib/.pkl (через joblib) и .json конфиги.
    - Если модель — LightGBM (или содержит booster), возвращает dict {"model": obj, "feature_names": [...]}
      чтобы инференс мог выровнять DataFrame под train и избежать LGBM Fatal по фичам.
    - Иначе возвращает сам объект модели.
    """
    t = (ticker_or_name or "").upper()

    candidates = [
        MODELS_DIR / f"arxora_m7pro_{t}.joblib",
        MODELS_DIR / f"m7pro_{t}.joblib",
        MODELS_DIR / f"global_{t}.joblib",
        MODELS_DIR / f"alphapulse_{t}.joblib",
        MODELS_DIR / f"octopus_{t}.joblib",
        MODELS_DIR / "m7_model.pkl",  # общий fallback
        CONFIG_DIR / f"m7pro_{t}.json",
        CONFIG_DIR / f"octopus_{t}.json",
        CONFIG_DIR / "calibration.json",
    ]

    for p in candidates:
        if not p.exists():
            continue

        if p.suffix in (".joblib", ".pkl"):
            try:
                obj = _try_joblib(p)
            except Exception as e:
                raise RuntimeError(f"Joblib load failed for {p}: {e}")

            # Пытаемся вытащить feature names для LightGBM
            feats = _extract_feature_names_if_any(obj)
            if feats:
                return {"model": obj, "feature_names": feats}
            return obj

        if p.suffix == ".json":
            return _load_json_file(p)

    return None

