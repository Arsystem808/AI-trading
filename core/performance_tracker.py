# core/performance_tracker.py
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd  # основано на рекомендациях по безопасным операциям с DataFrame [web:3191]

# Директории вывода
PERF_DIR = "performance_data"   # помесячные CSV с дневной доходностью по агенту/тикеру [web:3191]
METRICS_DIR = "metrics"         # сводный CSV со событийными метриками сигналов [web:3191]
METRICS_CSV = Path(METRICS_DIR) / "agent_performance.csv"  # путь для единого CSV [web:3191]

os.makedirs(PERF_DIR, exist_ok=True)   # гарантируем существование каталогов перед записью [web:3191]
os.makedirs(METRICS_DIR, exist_ok=True)  # аналогично для метрик [web:3191]


# -------------------- безопасное добавление строки (без FutureWarning) --------------------
def _append_row(df: Optional[pd.DataFrame], row: Dict[str, Any]) -> pd.DataFrame:
    """
    Безопасно добавляет одну строку в DataFrame, избегая FutureWarning от pd.concat c пустыми/all‑NA столбцами,
    используя прямую вставку через df.loc[len(df)] после выравнивания схемы. [web:3191]
    """
    # Если df отсутствует или пуст — создаём с нужными колонками сразу из строки [web:3191]
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame([row])  # одноразовое создание без concat предупреждений [web:3191]

    # Удаляем полностью пустые столбцы (чтобы не тянуть all-NA типы) [web:3191]
    df = df.dropna(axis=1, how="all")

    # Обеспечиваем наличие всех ключей строки в df; отсутствующие столбцы добавляем с NA [web:3191]
    for k in row.keys():
        if k not in df.columns:
            df[k] = pd.NA  # выравниваем схему перед вставкой [web:3191]

    # Для отсутствующих в row столбцов добавляем значения по умолчанию (NA) [web:3191]
    for k in df.columns:
        if k not in row:
            row[k] = pd.NA  # заполняем пропуски для согласования типов [web:3191]

    # Вставляем строку без concat, что исключает FutureWarning [web:3191]
    df.loc[len(df)] = row
    return df  # возвращаем обновлённый фрейм [web:3191]


# -------------------- дневная доходность (совместимость со старым интерфейсом) --------------------
def _log_daily_return(
    agent_label: str,
    ticker: str,
    date: Union[str, datetime, pd.Timestamp],
    daily_return: float,
) -> bool:
    """
    Запись/обновление дневной доходности агента по тикеру в CSV performance_data/performance_{agent}_{TICKER}.csv. [web:3191]
    """
    filename = Path(PERF_DIR) / f"performance_{agent_label.lower()}_{ticker.upper()}.csv"  # формируем путь [web:3191]

    df: Optional[pd.DataFrame] = None  # инициализация буфера [web:3191]
    if filename.exists():
        try:
            df = pd.read_csv(filename, parse_dates=["date"])  # читаем существующие данные [web:3191]
        except Exception:
            df = None  # при ошибке читаем заново как пустой [web:3191]

    date_norm = pd.to_datetime(date).normalize()  # нормализуем дату до полуночи [web:3191]
    row = {"date": date_norm, "daily_return": float(daily_return)}  # строка для вставки [web:3191]
    df = _append_row(df, row)  # безопасно добавляем строку [web:3191]

    # нормализация: сортировка и уникализация по дате [web:3191]
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df.to_csv(filename, index=False)  # перезаписываем CSV атомарно [web:3191]
    return True  # успешная запись [web:3191]


def get_agent_performance(agent_label: str, ticker: str) -> Optional[pd.DataFrame]:
    """
    История доходности за последние 90 дней с накопленной доходностью, либо None если данных нет. [web:3191]
    """
    filename = Path(PERF_DIR) / f"performance_{agent_label.lower()}_{ticker.upper()}.csv"  # путь к CSV [web:3191]
    if not filename.exists():
        return None  # нет данных — возвращаем None [web:3191]
    try:
        df = pd.read_csv(filename, parse_dates=["date"])  # читаем CSV с датами [web:3191]
    except Exception:
        return None  # при ошибке чтения — None [web:3191]

    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=90)  # окно 90 дней [web:3191]
    df = df[df["date"] >= cutoff].copy()  # фильтрация по дате [web:3191]
    if df.empty:
        return None  # нет свежих записей — None [web:3191]
    df = df.sort_values("date")  # сортировка по дате [web:3191]
    df["cumulative_return"] = (1.0 + df["daily_return"]).cumprod() - 1.0  # накопленная доходность [web:3191]
    return df  # возвращаем подготовленный фрейм [web:3191]


# -------------------- событийные метрики сигналов (интерфейс strategy.py) --------------------
def _log_event_metrics(
    *,
    agent: Optional[str],
    ticker: str,
    horizon: str,
    action: str,
    confidence: float,
    levels: Dict[str, float],
    probs: Dict[str, float],
    meta: Optional[Dict[str, Any]] = None,
    ts: Optional[str] = None,
    out_path: Optional[Union[str, Path]] = None,
    df_cache: Optional[pd.DataFrame] = None,
    **kwargs,
) -> bool:
    """
    Логирование метрик сигнала в единый CSV (metrics/agent_performance.csv), совместимо с вызовами из core/strategy.py. [web:3191]
    """
    meta = dict(meta or {})  # копируем метаданные [web:3191]
    if agent and "agent" not in meta:
        meta["agent"] = agent  # пробрасываем агент в meta при необходимости [web:3191]

    row = {
        "ts": ts,
        "date": (
            pd.to_datetime(ts).date().isoformat()
            if ts
            else pd.Timestamp.utcnow().date().isoformat()
        ),
        "ticker": ticker,
        "horizon": horizon,
        "agent": meta.get("agent", agent),
        "action": str(action),
        "confidence": float(confidence) if confidence is not None else None,
        "entry": float(levels.get("entry", 0.0)) if isinstance(levels, dict) else None,
        "sl": float(levels.get("sl", 0.0)) if isinstance(levels, dict) else None,
        "tp1": float(levels.get("tp1", 0.0)) if isinstance(levels, dict) else None,
        "tp2": float(levels.get("tp2", 0.0)) if isinstance(levels, dict) else None,
        "tp3": float(levels.get("tp3", 0.0)) if isinstance(levels, dict) else None,
        "p_tp1": float(probs.get("tp1", 0.0)) if isinstance(probs, dict) else None,
        "p_tp2": float(probs.get("tp2", 0.0)) if isinstance(probs, dict) else None,
        "p_tp3": float(probs.get("tp3", 0.0)) if isinstance(probs, dict) else None,
        "meta_json": json.dumps(meta, ensure_ascii=False),
    }  # формируем единую строку метрик [web:3191]

    path = Path(out_path) if out_path else METRICS_CSV  # вычисляем путь вывода [web:3191]
    path.parent.mkdir(parents=True, exist_ok=True)  # убеждаемся, что директория существует [web:3191]

    df = df_cache if isinstance(df_cache, pd.DataFrame) else None  # используем кэш если передан [web:3191]
    if df is None and path.exists():
        try:
            df = pd.read_csv(path)  # читаем существующий CSV [web:3191]
        except Exception:
            df = None  # при ошибке начнём с пустого [web:3191]

    df = _append_row(df, row)  # безопасная вставка без concat предупреждений [web:3191]
    df.to_csv(path, index=False)  # перезапись файла целиком (атомарнее, чем append) [web:3191]
    return True  # успешное логирование [web:3191]


# -------------------- единая точка входа (обратная совместимость) --------------------
def log_agent_performance(*args, **kwargs) -> bool:
    """
    Универсальный логгер с авто‑диспетчеризацией:
    1) Старый формат (дневная доходность): log_agent_performance(agent_label, ticker, date, daily_return) → CSV в performance_data. [web:3191]
    2) Новый формат (событийные метрики): log_agent_performance(agent=..., ticker=..., horizon=..., action=..., confidence=..., levels=..., probs=..., meta=None, ts=None, out_path=None) → metrics/agent_performance.csv. [web:3191]
    """
    # Детект формата 2 (новые события) по ключевым аргументам [web:3191]
    wants_event = any(k in kwargs for k in ("agent", "action", "levels", "probs", "horizon", "confidence"))
    if wants_event:
        return _log_event_metrics(**kwargs)  # маршрутизация в событийный логгер [web:3191]

    # Иначе ожидаем старый позиционный формат (agent_label, ticker, date, daily_return) [web:3191]
    if len(args) == 4 and not kwargs:
        agent_label, ticker, date, daily_return = args  # распаковка позиционных аргументов [web:3191]
        return _log_daily_return(agent_label, ticker, date, daily_return)  # запись дневной доходности [web:3191]

    # Также поддержим старый формат через именованные параметры [web:3191]
    if all(k in kwargs for k in ("agent_label", "ticker", "date", "daily_return")):
        return _log_daily_return(kwargs["agent_label"], kwargs["ticker"], kwargs["date"], kwargs["daily_return"])  # вызов старого формата [web:3191]

    # Если сигнатура не распознана — ничего не делаем, чтобы не падать [web:3191]
    return False  # исправлено: без лишней точки, чтобы не ловить E999 [web:3191]
