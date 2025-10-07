from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import joblib  # внешняя библиотека joblib, не sklearn.externals
except Exception:
    joblib = None  # мягкий импорт

from core.utils_naming import sanitize_symbol  # единая санитизация имён

# Базовые директории артефактов
MODELS_DIR = Path("models")
CONFIG_DIR = Path("configs")

# Нормализация типографики/почти‑JSON (для конфигов)
_FANCY = {
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',
    "’": "'",
    "‘": "'",
    "—": "-",
    "–": "-",
    "，": ",",
    "：": ":",  # в JSON допустимо; для имён файлов используем sanitize_symbol
    "；": ";",
}

def _normalize_text(s: str) -> str:
    # Заменяем типографские символы
    for k, v in _FANCY.items():
        s = s.replace(k, v)
    # Превращаем ': 'value'' -> ': "value"' (частый почти‑JSON)
    s = re.sub(r"(:\s*)\'([^\'\\n]*)\'", r'\1"\2"', s)
    # Удаляем запятую перед закрывающей скобкой/фигурной
    s = re.sub(r",(\s*[}\]])", r"\1", s)
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

def normalize_symbol(sym: str) -> str:
    """
    Приводим тикер к верхнему регистру и добавляем префикс класса активов
    для совместимости с Polygon (crypto/forex/indices).
    Примеры:
      BTCUSD -> X:BTCUSD
      EURUSD -> C:EURUSD (если используется ваша логика для FX; при желании оставьте только crypto)
    """
    u = (sym or "").upper().strip()
    # Crypto/FX/Indices префиксы Polygon
    if u.endswith("USD") and not (u.startswith("X:") or u.startswith("C:") or u.startswith("I:")):
        # По умолчанию считаем крипто для *USD* тикеров без префикса
        return f"X:{u}"
    return u

def _candidate_names(base: str) -> list[str]:
    """
    Формирует шаблоны имён для разных агентов по одному тикеру.
    """
    return [
        f"arxora_m7pro_{base}.joblib",
        f"global_{base}.joblib",
        f"alphapulse_{base}.joblib",
        f"octopus_{base}.joblib",
    ]

def _dedup(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _build_candidates(ticker_or_name: str) -> list[Path]:
    """
    Приоритетный список путей поиска:
    1) Санитизированное нормализованное имя (совместимо с upload‑artifact)
    2) Санитизированное исходное имя (обратная совместимость)
    3) Общие/fallback‑модели и JSON‑конфиги
    """
    raw = (ticker_or_name or "").upper().strip()
    norm = normalize_symbol(raw)
    safe_norm = sanitize_symbol(norm)
    safe_raw = sanitize_symbol(raw)

    names = _candidate_names(safe_norm)
    if safe_raw != safe_norm:
        names += _candidate_names(safe_raw)

    # Модели
    model_paths = [MODELS_DIR / n for n in names]
    model_paths.append(MODELS_DIR / "m7_model.pkl")  # общий fallback

    # Конфиги
    cfgs = [
        CONFIG_DIR / f"m7pro_{safe_norm}.json",
        CONFIG_DIR / f"octopus_{safe_norm}.json",
        CONFIG_DIR / "calibration.json",
    ]
    if safe_raw != safe_norm:
        cfgs.insert(0, CONFIG_DIR / f"m7pro_{safe_raw}.json")
        cfgs.insert(1, CONFIG_DIR / f"octopus_{safe_raw}.json")

    return _dedup(model_paths + cfgs)

def load_model_for(ticker_or_name: str) -> Optional[Union[Any, Dict[str, Any]]]:
    """
    Универсальный загрузчик:
    - Ищет веса по приоритету для тикера/агента (arxora_m7pro/global/alphapulse/octopus),
      сначала по нормализованному и санитизированному имени, потом по исходному.
    - Поддерживает .joblib/.pkl (через joblib) и .json конфиги.
    - Если найден конфиг JSON с указанием model_artifact, загружает модель по этому пути
      и возвращает {"model": obj, "metadata": {...}}, где metadata включает feature_cols/dates при наличии.
    - Если модель — LightGBM (или содержит booster), добавляет feature_names для выравнивания инференса.
    """
    for p in _build_candidates(ticker_or_name):
        if not p.exists():
            continue

        if p.suffix == ".json":
            cfg = _load_json_file(p)
            if isinstance(cfg, dict) and "model_artifact" in cfg:
                art = Path(cfg["model_artifact"])
                if not art.is_absolute():
                    art = Path(".") / art
                if art.exists():
                    obj = _try_joblib(art)
                    feats = _extract_feature_names_if_any(obj)
                    metadata: Dict[str, Any] = dict(cfg)
                    if feats:
                        metadata.setdefault("feature_names", feats)
                    return {"model": obj, "metadata": metadata}
            # если нет пути к артефакту — возвращаем сам конфиг (обратная совместимость)
            return cfg

        if p.suffix in (".joblib", ".pkl"):
            obj = _try_joblib(p)
            feats = _extract_feature_names_if_any(obj)
            if feats:
                return {"model": obj, "feature_names": feats}
            return obj

    return None
