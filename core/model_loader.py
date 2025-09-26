# core/model_loader.py
from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Optional

try:
    import joblib  # sklearn persistence
except Exception:  # мягкий импорт
    joblib = None


# Директории с артефактами моделей
MODELS_DIR = Path("models")
CONFIG_DIR = Path("configs")


# Нормализация текста для устранения “invalid character”
# - заменяем фигурные/типографские кавычки на обычные
# - приводим JSON с одиночными кавычками к валидному формату (двойные)
_FANCY = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "’": "'", "‘": "'",
    "—": "-", "–": "-",
    "，": ",", "：": ":", "；": ";"
}


def _normalize_text(s: str) -> str:
    for k, v in _FANCY.items():
        s = s.replace(k, v)
    # Превращаем ': 'value'' в ': "value"' для часто встречающегося "почти-JSON"
    s = re.sub(r'(:\s*)\'([^\'\\n]*)\'', r'\1"\2"', s)
    # Не позволяйте висячим запятым ломать JSON в конце объектов/массивов
    s = re.sub(r',(\s*[}\]])', r'\1', s)
    return s


def _load_json_file(p: Path) -> Any:
    raw = p.read_text(encoding="utf-8", errors="replace")
    norm = _normalize_text(raw)
    try:
        return json.loads(norm)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {p}: {e}")  # JSON требует двойные кавычки и корректный синтаксис


def _try_joblib(p: Path) -> Any:
    if not joblib:
        return None
    return joblib.load(p)


def load_model_for(ticker_or_name: str) -> Optional[Any]:
    """
    Универсальный загрузчик:
    - Пытается найти веса по приоритету для тикера/агента (m7pro/global/alphapulse/octopus)
    - Поддерживает .joblib/.pkl (через joblib) и .json конфиги для метапараметров
    - Возвращает объект модели/конфиг или None, если ничего не найдено
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
                return _try_joblib(p)
            except Exception as e:
                raise RuntimeError(f"Joblib load failed for {p}: {e}")
        if p.suffix == ".json":
            return _load_json_file(p)

    return None
