# core/backtest_filters.py
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Tuple

__all__ = [
    "dedupe_within_horizon",
    "dedupe_across_horizons",
]


# ---------- утилиты ----------
def _risk(sig: Dict[str, Any]) -> float:
    lv = sig["levels"]
    return abs(float(lv["entry"]) - float(lv["sl"])) or 1e-9


def _rrs(sig: Dict[str, Any]) -> Tuple[float, float, float]:
    lv = sig["levels"]
    r = _risk(sig)
    rr1 = abs(float(lv["tp1"]) - float(lv["entry"])) / r
    rr2 = abs(float(lv["tp2"]) - float(lv["entry"])) / r
    rr3 = abs(float(lv["tp3"]) - float(lv["entry"])) / r
    return rr1, rr2, rr3


def _probs(sig: Dict[str, Any]) -> Tuple[float, float, float]:
    pr = sig.get("probs", {})
    return float(pr.get("tp1", 0)), float(pr.get("tp2", 0)), float(pr.get("tp3", 0))


def _expected_R(sig: Dict[str, Any]) -> float:
    rr1, rr2, rr3 = _rrs(sig)
    p1, p2, p3 = _probs(sig)
    ev_pos = (p1 * rr1 + p2 * rr2 + p3 * rr3) / 3.0
    ev_neg = (1.0 - p1) * 1.0  # штраф за недостижение TP1
    return ev_pos - ev_neg


def _score(sig: Dict[str, Any]) -> Tuple[float, float, float]:
    conf = float(sig["recommendation"].get("confidence", 0.0))
    evR = _expected_R(sig)
    rr1, _, _ = _rrs(sig)
    return (conf, evR, rr1)  # приоритет: conf → EV → RR1


def _ts(sig: Dict[str, Any]) -> dt.datetime:
    t = sig["ts"]
    return (
        t if not isinstance(t, str) else dt.datetime.fromisoformat(t.replace("Z", ""))
    )


def _same_side(a, b) -> bool:
    return a["recommendation"]["action"] == b["recommendation"]["action"]


def _entry_close(a, b, tol_mult=0.5, tol_mode="risk") -> bool:
    """считать дублями, если entry близки:
    tol_mode='risk' → |Δentry| <= tol_mult * min(risk_a, risk_b)
    tol_mode='atr'  → если есть features['atr_d'], использовать её
    """
    ea, eb = float(a["levels"]["entry"]), float(b["levels"]["entry"])
    if tol_mode == "atr":
        fa = a.get("features", {}) or {}
        fb = b.get("features", {}) or {}
        atr_a = float(fa.get("atr_d", 0) or 0)
        atr_b = float(fb.get("atr_d", 0) or 0)
        base = max(1e-9, min(atr_a, atr_b))
        return abs(ea - eb) <= tol_mult * base
    # по умолчанию — risk
    base = max(1e-9, min(_risk(a), _risk(b)))
    return abs(ea - eb) <= tol_mult * base


# ---------- 1) анти-дубли внутри одного горизонта ----------
def dedupe_within_horizon(
    signals: List[Dict[str, Any]],
    horizon_key: str,  # напр.: "Среднесрок (1–4 недели)"
    by_day: bool = True,  # кластеризовать по дню
    tol_mult: float = 0.5,  # жёсткость близости entry
    tol_mode: str = "risk",  # 'risk' | 'atr'
    cooldown_days: int | None = None,  # запретить новую идею по тикеру N дней
) -> List[Dict[str, Any]]:
    # Только нужный горизонт
    sigs = [s for s in signals if s["horizon"] == horizon_key]
    # Сортировка по времени, затем по убыванию score
    sigs.sort(key=lambda s: (_ts(s),) + tuple(-x for x in _score(s)))

    accepted: List[Dict[str, Any]] = []
    last_taken_at: dict[str, dt.datetime] = {}

    # Группировка: тикер × (дата если by_day, иначе весь поток)
    from collections import defaultdict

    buckets = defaultdict(list)
    for s in sigs:
        key = (s["ticker"], _ts(s).date()) if by_day else (s["ticker"], None)
        buckets[key].append(s)

    for (ticker, _), items in sorted(buckets.items(), key=lambda kv: kv[0][0]):
        # внутри дня/группы: убрать почти-дубли (одна сторона + близкий entry)
        uniq: List[Dict[str, Any]] = []
        for s in items:
            dup_idx = -1
            for i, u in enumerate(uniq):
                if _same_side(s, u) and _entry_close(s, u, tol_mult, tol_mode):
                    dup_idx = i
                    break
            if dup_idx >= 0:
                if _score(s) > _score(uniq[dup_idx]):  # оставляем лучший
                    uniq[dup_idx] = s
            else:
                uniq.append(s)

        # применить cooldown (по тикеру) внутри горизонта
        for s in sorted(uniq, key=_ts):
            if cooldown_days is not None and ticker in last_taken_at:
                if (_ts(s).date() - last_taken_at[ticker].date()).days < cooldown_days:
                    continue
            accepted.append(s)
            last_taken_at[ticker] = _ts(s)

    return accepted


# ---------- 2) (опционально) анти-дубли между горизонтами ----------
def dedupe_across_horizons(
    signals: List[Dict[str, Any]],
    cooldown_days: int = 3,
    conflict_policy: str = "pick_best",  # или "skip_conflict"
) -> List[Dict[str, Any]]:
    """
    В один день по тикеру оставляем 1 идею (лучший score).
    """
    # сортировка по времени, затем по убыванию score
    sigs = sorted(signals, key=lambda s: (_ts(s),) + tuple(-x for x in _score(s)))

    from collections import defaultdict

    by_day = defaultdict(list)
    for s in sigs:
        by_day[(s["ticker"], _ts(s).date())].append(s)

    accepted: List[Dict[str, Any]] = []
    last_taken_at: dict[str, dt.datetime] = {}

    for (ticker, day), items in sorted(by_day.items(), key=lambda kv: kv[0][1]):
        dirs = {i["recommendation"]["action"] for i in items}
        if "BUY" in dirs and "SHORT" in dirs and conflict_policy == "skip_conflict":
            continue
        best = max(items, key=_score)
        # cooldown
        last_ts = last_taken_at.get(ticker)
        if last_ts and (day - last_ts.date()).days < cooldown_days:
            continue
        accepted.append(best)
        last_taken_at[ticker] = _ts(best)

    return accepted
