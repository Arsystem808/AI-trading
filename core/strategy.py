# core/strategy.py
import hashlib
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.polygon_client import PolygonClient


# ----------------- вспомогательные функции (в UI не раскрываем) -----------------

def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def _linreg_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    beta = ((x - xm) * (y - ym)).sum() / denom
    return float(beta)


def _streak(closes: pd.Series) -> int:
    """кол-во подряд растущих (положит.) или падающих (отрицат.) свечей"""
    s = 0
    for i in range(len(closes) - 1, 0, -1):
        d = closes.iloc[i] - closes.iloc[i - 1]
        if d > 0:
            if s < 0:
                break
            s += 1
        elif d < 0:
            if s > 0:
                break
            s -= 1
        else:
            break
    return s


def _pick(seed: int, *variants: str) -> str:
    rng = random.Random(seed)
    return rng.choice(list(variants))


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _horizon_cfg(text: str):
    # окна под разные горизонты (в UI не показываем)
    if "Кратко" in text:
        return dict(look=60, trend=14, atr=14)
    if "Средне" in text:
        return dict(look=120, trend=28, atr=14)
    return dict(look=240, trend=56, atr=14)


# ----------------- основная логика -----------------

def analyze_asset(ticker: str, horizon: str):
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)

    # Данные
    days = max(90, cfg["look"] * 2)
    df = cli.daily_ohlc(ticker, days=days)
    price = cli.last_trade_price(ticker)

    tail = df.tail(cfg["look"])
    low = float(tail["low"].min())
    high = float(tail["high"].max())
    width = max(1e-9, high - low)
    pos = (price - low) / width  # 0..1 положение внутри коридора

    # Тренд/волатильность (внутренние метрики)
    closes = df["close"]
    slope = _linreg_slope(closes.tail(cfg["trend"]).values)  # наклон
    slope_norm = slope / max(1e-9, price)  # нормировка
    atr_s = float(_atr_like(df, n=cfg["atr"]).iloc[-1])
    atr_l = float(_atr_like(df, n=cfg["atr"] * 2).iloc[-1])
    vol_ratio = atr_s / max(1e-9, atr_l)  # >1 — волатильность расширяется
    streak = _streak(closes)

    # Сценарии: продолжение тренда, разворот от края, пробой из коридора
    near_top = pos >= 0.7
    near_bottom = pos <= 0.3
    mid_zone = 0.45 < pos < 0.55
    strong_up = slope_norm > 0.001 and vol_ratio > 1.05
    strong_down = slope_norm < -0.001 and vol_ratio > 1.05
    breakout_up = near_top and strong_up and (price >= high * 0.995)
    breakout_dn = near_bottom and strong_down and (price <= low * 1.005)

    # Решение
    if breakout_up:
        action = "BUY"
        scenario = "breakout_up"
    elif breakout_dn:
        action = "SHORT"
        scenario = "breakout_dn"
    elif near_top and not strong_up:
        action = "SHORT"
        scenario = "fade_top"
    elif near_bottom and not strong_down:
        action = "BUY"
        scenario = "bounce_bottom"
    elif mid_zone:
        action = "WAIT"
        scenario = "mid_range"
    else:
        # сдвинутые зоны — идём по тренду
        action = "BUY" if slope_norm >= 0 else "SHORT"
        scenario = "trend_follow"

    # Уверенность (0.55–0.90) из нескольких факторов
    edge = abs(pos - 0.5) * 2  # ближе к краю — выше
    trn = _clip01(abs(slope_norm) * 1800)  # масштабир. в удобный интервал
    vol = _clip01((vol_ratio - 0.9) / 0.6)
    base = 0.48 + 0.25 * edge + 0.17 * trn + 0.10 * vol
    if scenario in ("mid_range",):
        base -= 0.08
    conf = float(max(0.55, min(0.90, base)))

    # Размер шага (для уровней) — от текущей волатильности
    step = max(1e-6, atr_s)

    # Уровни зависят от сценария
    if action == "BUY":
        if scenario == "breakout_up":
            entry = price + step * 0.10
            note = "Сильный подъём у верхней границы; работать по импульсу после выхода и короткой паузы."
        elif scenario == "trend_follow":
            entry = price - step * 0.15
            note = "Преимущество у роста; берём откат внутри восходящего движения."
        else:  # bounce_bottom
            entry = price - step * 0.20
            note = "Цена ближе к зоне покупателя; ждём откат и реакцию снизу."
        sl = price - step * 1.0
        tp1 = price + step * 0.8
        tp2 = price + step * 1.6
        tp3 = price + step * 2.4
        alt = "Если уйдёт ниже зоны покупателя — пропустить вход; ждать возврата и подтверждения сверху."
    elif action == "SHORT":
        if scenario == "breakout_dn":
            entry = price - step * 0.10
            note = "Импульс вниз у нижней границы; работаем по движению после выхода и паузы."
        elif scenario == "trend_follow":
            entry = price + step * 0.15
            note = "Преимущество у снижения; берём откат в нисходящем движении."
        else:  # fade_top
            entry = price + step * 0.20
            note = "Цена ближе к зоне продавца; смотрим на слабость у верхней границы."
        sl = price + step * 1.0
        tp1 = price - step * 0.8
        tp2 = price - step * 1.6
        tp3 = price - step * 2.4
        alt = "Если пробьёт верх и удержится — не гнаться; ждать возврата и признаки слабости у максимумов."
    else:
        # WAIT — даём план на пробой (но действие остаётся WAIT)
        entry = price
        sl = price - step * 0.9
        tp1 = price + step * 0.7
        tp2 = price + step * 1.4
        tp3 = price + step * 2.1
        note = "Середина коридора — явного перевеса нет; ждём реакции рядом с ценой."
        alt = "При уверенном пробое диапазона работаем по направлению после ретеста и подтверждения."

    # Вероятности целей — завязаны на уверенности (детерминированные)
    p1 = 0.60 + 0.25 * (conf - 0.55) / 0.35
    p2 = 0.45 + 0.20 * (conf - 0.55) / 0.35
    p3 = 0.28 + 0.12 * (conf - 0.55) / 0.35

    # Небольшая вариативность формулировок: детерминированный seed
    seed = int(hashlib.sha1(f"{ticker}{df.index[-1].date()}".encode()).hexdigest(), 16) % (2**32)
    if "границ" in note:  # подменяем синонимы без рандомной «болтанки» сигнала
        note = note.replace(
            "границы",
            _pick(seed, "границы", "края", "верхней зоны", "верхнего диапазона"),
        ).replace(
            "зоне покупателя",
            _pick(seed, "зоне покупателя", "нижней зоне", "области спроса"),
        )

    return {
        "last_price": float(price),
        "recommendation": {"action": action, "confidence": float(round(conf, 4))},
        "levels": {
            "entry": float(entry),
            "sl": float(sl),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "tp3": float(tp3),
        },
        "probs": {"tp1": float(_clip01(p1)), "tp2": float(_clip01(p2)), "tp3": float(_clip01(p3))},
        "note": note,
        "alt": alt,
    }
