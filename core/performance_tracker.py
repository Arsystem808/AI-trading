# core/performance_tracker.py
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

# Директории вывода
PERF_DIR = "performance_data"  # помесячные CSV с дневной доходностью по агенту/тикеру
METRICS_DIR = "metrics"  # сводный CSV со событийными метриками сигналов
METRICS_CSV = Path(METRICS_DIR) / "agent_performance.csv"

os.makedirs(PERF_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# -------------------- безопасное добавление строки (без FutureWarning) --------------------
def _append_row(df: Optional[pd.DataFrame], row: Dict[str, Any]) -> pd.DataFrame:
    """
    Безопасно добавляет одну строку в DataFrame, избегая FutureWarning при concat с пустыми/all‑NA столбцами. [web:667]
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        df = pd.DataFrame(columns=list(row.keys()))
    else:
        # убрать полностью пустые столбцы и обеспечить наличие всех ключей строки
        df = df.dropna(axis=1, how="all")
        for k in row.keys():
            if k not in df.columns:
                df[k] = pd.NA
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


# -------------------- дневная доходность (совместимость со старым интерфейсом) --------------------
def _log_daily_return(
    agent_label: str, ticker: str, date: Union[str, datetime, pd.Timestamp], daily_return: float
) -> bool:
    """
    Запись/обновление дневной доходности агента по тикеру в CSV performance_data/performance_{agent}_{TICKER}.csv. [attached_file:614]
    """
    filename = Path(PERF_DIR) / f"performance_{agent_label.lower()}_{ticker.upper()}.csv"
    df = None
    if filename.exists():
        try:
            df = pd.read_csv(filename, parse_dates=["date"])
        except Exception:
            df = None

    date_norm = pd.to_datetime(date).normalize()
    row = {"date": date_norm, "daily_return": float(daily_return)}
    df = _append_row(df, row)

    # нормализация: сортировка и уникализация по дате
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    df.to_csv(filename, index=False)
    return True


def get_agent_performance(agent_label: str, ticker: str) -> Optional[pd.DataFrame]:
    """
    История доходности за последние 90 дней с накопленной доходностью, либо None если данных нет. [attached_file:614]
    """
    filename = Path(PERF_DIR) / f"performance_{agent_label.lower()}_{ticker.upper()}.csv"
    if not filename.exists():
        return None
    try:
        df = pd.read_csv(filename, parse_dates=["date"])
    except Exception:
        return None

    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=90)
    df = df[df["date"] >= cutoff].copy()
    if df.empty:
        return None
    df = df.sort_values("date")
    df["cumulative_return"] = (1.0 + df["daily_return"]).cumprod() - 1.0
    return df


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
    Логирование метрик сигнала в единый CSV (metrics/agent_performance.csv), совместимо с вызовами из core/strategy.py. [attached_file:614]
    """
    meta = dict(meta or {})
    if agent and "agent" not in meta:
        meta["agent"] = agent

    row = {
        "ts": ts,
        "date": pd.to_datetime(ts).date().isoformat() if ts else pd.Timestamp.utcnow().date().isoformat(),
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
    }

    path = Path(out_path) if out_path else METRICS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)

    df = df_cache if isinstance(df_cache, pd.DataFrame) else None
    if df is None and path.exists():
        try:
            df = pd.read_csv(path)
        except Exception:
            df = None

    df = _append_row(df, row)
    df.to_csv(path, index=False)
    return True


# -------------------- единая точка входа (обратная совместимость) --------------------
def log_agent_performance(*args, **kwargs) -> bool:
    """
    Универсальный логгер с авто‑диспетчеризацией:
    1) Старый формат (дневная доходность): log_agent_performance(agent_label, ticker, date, daily_return) → CSV в performance_data. [attached_file:614]
    2) Новый формат (событийные метрики): log_agent_performance(agent=..., ticker=..., horizon=..., action=..., confidence=..., levels=..., probs=..., meta=None, ts=None, out_path=None) → metrics/agent_performance.csv. [attached_file:614]
    """
    # Детект формата 2 (новые события) по ключевым аргументам
    wants_event = any(k in kwargs for k in ("agent", "action", "levels", "probs", "horizon", "confidence"))
    if wants_event:
        return _log_event_metrics(**kwargs)

    # Иначе ожидаем старый позиционный формат (agent_label, ticker, date, daily_return)
    if len(args) == 4 and not kwargs:
        agent_label, ticker, date, daily_return = args
        return _log_daily_return(agent_label, ticker, date, daily_return)

    # Также поддержим старый формат через именованные параметры
    if all(k in kwargs for k in ("agent_label", "ticker", "date", "daily_return")):
        return _log_daily_return(kwargs["agent_label"], kwargs["ticker"], kwargs["date"], kwargs["daily_return"])

    # Если сигнатура не распознана — ничего не делаем, чтобы не падать
    return False
