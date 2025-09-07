# core/strategy.py
from typing import List, Tuple, Dict, Optional
import hashlib
import random
import numpy as np
import pandas as pd
from core.polygon_client import PolygonClient

# ---------- helpers (в UI не раскрываем) ----------
def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def _weekly_atr(df: pd.DataFrame, n_weeks: int = 8) -> float:
    w = df.resample("W-FRI").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(w) < 2:
        return float((df["high"] - df["low"]).tail(14).mean())
    hl = w["high"] - w["low"]
    hc = (w["high"] - w["close"].shift(1)).abs()
    lc = (w["low"] - w["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return float(tr.rolling(n_weeks, min_periods=1).mean().iloc[-1])

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

def _wick_profile(row: pd.Series) -> Tuple[float, float, float]:
    o, c, h, l = row["open"], row["close"], row["high"], row["low"]
    body = abs(c - o)
    upper = max(0.0, h - max(o, c))
    lower = max(0.0, min(o, c) - l)
    return body, upper, lower

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _horizon_cfg(text: str) -> Dict[str, object]:
    # Недельные пивоты для всех горизонтов; weekly ATR — для средне/долгосрока
    if "Кратко" in text:
        return dict(look=60, trend=14, atr=14, pivot_period="W-FRI", use_weekly_atr=False)
    if "Средне" in text:
        return dict(look=120, trend=28, atr=14, pivot_period="W-FRI", use_weekly_atr=True)
    return dict(look=240, trend=56, atr=14, pivot_period="W-FRI", use_weekly_atr=True)

# ---------- Heikin Ashi ----------
def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy();  c = df["close"].to_numpy()
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_open)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low  = np.minimum.reduce([l, ha_open, ha_close])
    return pd.DataFrame(
        {"ha_open": ha_open, "ha_high": ha_high, "ha_low": ha_low, "ha_close": ha_close},
        index=df.index
    )

# ---------- Fibonacci pivots (внутренне) ----------
def _last_period_hlc(df: pd.DataFrame, rule: str) -> Optional[Tuple[float, float, float]]:
    g = df.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(g) < 2:
        return None
    row = g.iloc[-2]  # последняя ЗАКРЫТАЯ неделя/месяц
    return float(row["high"]), float(row["low"]), float(row["close"])

def _fib_pivots(H: float, L: float, C: float) -> Dict[str, float]:
    P = (H + L + C) / 3.0
    d = H - L
    R1 = P + 0.382 * d; R2 = P + 0.618 * d; R3 = P + 1.000 * d
    S1 = P - 0.382 * d; S2 = P - 0.618 * d; S3 = P - 1.000 * d
    return {"P": P, "R1": R1, "R2": R2, "R3": R3, "S1": S1, "S2": S2, "S3": S3}

def _pivot_ladder(piv: Dict[str, float]) -> List[float]:
    keys = ["S3", "S2", "S1", "P", "R1", "R2", "R3"]
    return [float(piv[k]) for k in keys if k in piv and piv[k] is not None]

def _three_targets_from_pivots(entry: float, direction: str, piv: Dict[str, float], step: float) -> Tuple[float, float, float]:
    """
    BUY — цели строго выше entry; SHORT — ниже.
    Если подходящих пивотов < 3, докладываем ATR-ступени.
    """
    ladder = sorted(set(_pivot_ladder(piv)))
    eps = 0.10 * step
    if direction == "BUY":
        ups = [x for x in ladder if x > entry + eps]
        while len(ups) < 3:
            k = len(ups) + 1
            ups.append(entry + (0.7 + 0.7 * (k - 1)) * step)
        return ups[0], ups[1], ups[2]
    else:
        dns = [x for x in ladder if x < entry - eps]
        dns = list(sorted(dns, reverse=True))
        while len(dns) < 3:
            k = len(dns) + 1
            dns.append(entry - (0.7 + 0.7 * (k - 1)) * step)
        return dns[0], dns[1], dns[2]

def _classify_band(price: float, piv: Dict[str, float], buf: float) -> int:
    """
    Полоса пивотов:
    -3: <S2, -2: [S2,S1), -1: [S1,P), 0: [P,R1), +1: [R1,R2), +2: [R2,R3), +3: >=R3
    """
    P, R1 = piv["P"], piv["R1"]
    R2, R3, S1, S2 = piv.get("R2"), piv.get("R3"), piv["S1"], piv.get("S2")
    neg_inf, pos_inf = -1e18, 1e18
    levels = [
        S2 if S2 is not None else neg_inf, S1, P, R1,
        R2 if R2 is not None else pos_inf, R3 if R3 is not None else pos_inf
    ]
    if price < levels[0] - buf: return -3
    if price < levels[1] - buf: return -2
    if price < levels[2] - buf: return -1
    if price < levels[3] - buf: return 0
    if R2 is None or price < levels[4] - buf: return +1
    if price < levels[5] - buf: return +2
    return +3

# ---------- основная логика ----------
def analyze_asset(ticker: str, horizon: str) -> Dict[str, object]:
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)

    # Данные
    days = max(90, cfg["look"] * 2)
    df = cli.daily_ohlc(ticker, days=days)
    price = cli.last_trade_price(ticker)

    # Heikin Ashi (D1 и W1)
    ha = _heikin_ashi(df)
    ha_w = ha.resample("W-FRI").agg({
        "ha_open": "first", "ha_high": "max", "ha_low": "min", "ha_close": "last"
    }).dropna()

    # Позиция в своём диапазоне
    tail = df.tail(cfg["look"])
    rng_low = float(tail["low"].min())
    rng_high = float(tail["high"].max())
    rng_w = max(1e-9, rng_high - rng_low)
    pos = (price - rng_low) / rng_w  # 0..1

    closes = df["close"]
    slope = _linreg_slope(closes.tail(cfg["trend"]).values)
    slope_norm = slope / max(1e-9, price)

    atr_d = float(_atr_like(df, n=cfg["atr"]).iloc[-1])
    atr_w = _weekly_atr(df) if cfg.get("use_weekly_atr") else atr_d
    vol_ratio = atr_d / max(1e-9, float(_atr_like(df, n=cfg["atr"] * 2).iloc[-1]))
    streak = _streak(closes)

    last_row = df.iloc[-1]
    body, upper, lower = _wick_profile(last_row)
    long_upper = upper > body * 1.3 and upper > lower * 1.1
    long_lower = lower > body * 1.3 and lower > upper * 1.1

    # D1/W1 Heikin Ashi признаки
    ha_last = ha.iloc[-1]
    ha_body_d = abs(float(ha_last["ha_close"] - ha_last["ha_open"]))
    ha_upper_d = max(0.0, float(ha_last["ha_high"] - max(ha_last["ha_open"], ha_last["ha_close"])))
    ha_lower_d = max(0.0, float(min(ha_last["ha_open"], ha_last["ha_close"]) - ha_last["ha_low"]))
    ha_long_upper_d = ha_upper_d > ha_body_d
    ha_long_lower_d = ha_lower_d > ha_body_d

    if len(ha_w) >= 1:
        ha_w_last = ha_w.iloc[-1]
        ha_w_body = abs(float(ha_w_last["ha_close"] - ha_w_last["ha_open"]))
        ha_w_upper = max(0.0, float(ha_w_last["ha_high"] - max(ha_w_last["ha_open"], ha_w_last["ha_close"])))
        ha_w_bear_reject = (ha_w_last["ha_close"] < ha_w_last["ha_open"]) and (ha_w_upper > ha_w_body)
    else:
        ha_w_bear_reject = False

    # Пивоты предыдущей недели (или fallback)
    hlc = _last_period_hlc(df, cfg["pivot_period"])
    if hlc:
        H, L, C = hlc
    else:
        H = float(df["high"].tail(60).max())
        L = float(df["low"].tail(60).min())
        C = float(df["close"].iloc[-1])
    piv = _fib_pivots(H, L, C)
    P, R1, R2 = piv["P"], piv["R1"], piv.get("R2")

    # Буфер «вблизи уровня» — от weekly ATR (для средне/долгосрока шире)
    buf = 0.25 * atr_w
    band = _classify_band(price, piv, buf)  # -3..+3

    # ---- Сценарии (жёсткий short-bias у верхней кромки) ----
    last_o, last_c = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
    last_h = float(df["high"].iloc[-1])
    upper_wick_d = max(0.0, last_h - max(last_o, last_c))
    body_d = abs(last_c - last_o)

    # МЕДВЕЖИЙ ОТКАЗ: обычные свечи ИЛИ HA-сигналы
    bearish_reject = ((last_c < last_o) and (upper_wick_d > body_d)) or ha_w_bear_reject or ha_long_upper_d
    very_high_pos = pos >= 0.80  # верхние 20% диапазона

    if very_high_pos:
        # приоритет: SHORT; BUY только при явном пробое верхней зоны
        if (R2 is not None) and (price > R2 + 0.6 * buf) and (slope_norm > 0):
            action, scenario = "BUY", "breakout_up"
        else:
            action, scenario = "SHORT", "fade_top"
    else:
        if band >= +2:
            action, scenario = ("BUY", "breakout_up") if ((R2 is not None) and (price > R2 + 0.6 * buf) and (slope_norm > 0)) else ("SHORT", "fade_top")
        elif band == +1:
            action, scenario = ("WAIT", "upper_wait") if (slope_norm > 0.0015 and not bearish_reject and not long_upper) else ("SHORT", "fade_top")
        elif band == 0:
            action, scenario = ("BUY", "trend_follow") if slope_norm >= 0 else ("WAIT", "mid_range")
        elif band == -1:
            action, scenario = ("BUY", "revert_from_bottom") if (streak <= -3 or long_lower or ha_long_lower_d) else ("BUY", "trend_follow")
        else:
            action, scenario = ("BUY", "revert_from_bottom") if band <= -2 else ("WAIT", "upper_wait")

    # Guard: на средне/долгосроке в верхних 20% диапазона BUY запрещён (кроме пробоя)
    if cfg.get("use_weekly_atr") and pos >= 0.8 and action == "BUY" and scenario != "breakout_up":
        action, scenario = "WAIT", "upper_wait"

    # Уверенность (корректируем от HA)
    base = 0.55 + 0.12 * _clip01(abs(slope_norm) * 1800) + 0.08 * _clip01((vol_ratio - 0.9) / 0.6)
    if action == "WAIT":
        base -= 0.07
    if band >= +1 and action == "BUY":
        base -= 0.10
    if band <= -1 and action == "BUY":
        base += 0.05
    if action == "SHORT" and (ha_w_bear_reject or ha_long_upper_d):
        base += 0.03
    if action == "BUY" and (ha_long_lower_d):
        base += 0.02
    conf = float(max(0.55, min(0.92, base)))

    # Шаги уровней
    step_d, step_w = atr_d, atr_w

    # Уровни (для WAIT UI их не показывает)
    if action == "BUY":
        if scenario == "breakout_up":
            base_ref = R2 if R2 is not None else R1
            entry = max(price, base_ref + 0.10 * step_w)
            sl    = base_ref - 1.00 * step_w
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
        elif price < P:
            entry = max(price, piv["S1"] + 0.15 * step_w)
            sl    = piv["S1"] - 0.60 * step_w
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
        else:
            entry = max(price, P + 0.10 * step_w)
            sl    = P - 0.60 * step_w
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
        alt = "Если продавят ниже и не вернут — не лезем; ждём возврата сверху и подтверждения."
    elif action == "SHORT":
        # fade сверху — якорим к верхней зоне
        if price >= R1:
            entry = min(price, R1 - 0.15 * step_w)
            sl    = R1 + 0.60 * step_w
        else:
            entry = price + 0.15 * step_d
            sl    = price + 1.00 * step_d
        tp1, tp2, tp3 = _three_targets_from_pivots(entry, "SHORT", piv, step_w)
        alt = "Если протолкнут выше и удержат — без погони; ждём возврата и признаков слабости сверху."
    else:  # WAIT
        entry = price
        sl    = price - 0.90 * step_d
        tp1, tp2, tp3 = entry + 0.7 * step_d, entry + 1.4 * step_d, entry + 2.1 * step_d
        alt = "Пока не вижу входа; работаю на пробое после ретеста или от отката к опоре."

    # Вероятности целей (монотонно убывают)
    p1 = _clip01(0.58 + 0.27 * (conf - 0.55) / 0.35)
    p2 = _clip01(0.44 + 0.21 * (conf - 0.55) / 0.35)
    p3 = _clip01(0.28 + 0.13 * (conf - 0.55) / 0.35)

    # «Живой» комментарий без терминов уровней
    seed = int(hashlib.sha1(f"{ticker}{df.index[-1].date()}".encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    if action == "BUY":
        lead = rng.choice([
            "Вход беру от зоны спроса. Не гонюсь за ценой.",
            "Работаю по тренду после возврата."
        ])
    elif action == "SHORT":
        lead = rng.choice([
            "Под верхней кромкой действую от отказа.",
            "Продаю слабость после выноса наверх."
        ])
    else:
        lead = rng.choice([
            "Пока без позиции, жду понятный сигнал.",
            "План появится после пробоя/ретеста или отката."
        ])
    note_html = f"<div style='margin-top:10px; opacity:0.95;'>{lead}</div>"

    return {
        "last_price": float(price),
        "recommendation": {"action": action, "confidence": float(round(conf, 4))},
        "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
        "probs": {"tp1": float(p1), "tp2": float(p2), "tp3": float(p3)},
        "context": [],  # UI не выводит чипы, оставляем пустым
        "note_html": note_html,
        "alt": alt,
        # dev: пивоты для внутреннего просмотра
        "pivots": {k: (float(piv[k]) if piv.get(k) is not None else None) for k in ["S3", "S2", "S1", "P", "R1", "R2", "R3"]},
    }
