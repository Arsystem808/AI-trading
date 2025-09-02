import hashlib
import random
import numpy as np
import pandas as pd
from core.polygon_client import PolygonClient

# --- вспомогательные (в UI не раскрываем) ---

def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def _linreg_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2: return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0: return 0.0
    beta = ((x - xm) * (y - ym)).sum() / denom
    return float(beta)

def _streak(closes: pd.Series) -> int:
    s = 0
    for i in range(len(closes) - 1, 0, -1):
        d = closes.iloc[i] - closes.iloc[i - 1]
        if d > 0:
            if s < 0: break
            s += 1
        elif d < 0:
            if s > 0: break
            s -= 1
        else:
            break
    return s

def _wick_profile(row):
    o, c, h, l = row["open"], row["close"], row["high"], row["low"]
    body = abs(c - o)
    upper = max(0.0, h - max(o, c))
    lower = max(0.0, min(o, c) - l)
    return body, upper, lower

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _horizon_cfg(text: str):
    if "Кратко" in text: return dict(look=60, trend=14, atr=14)
    if "Средне" in text: return dict(look=120, trend=28, atr=14)
    return dict(look=240, trend=56, atr=14)

def _choose(seed: int, variants):
    rng = random.Random(seed)
    return rng.choice(list(variants))

# --- основная логика ---

def analyze_asset(ticker: str, horizon: str):
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)

    days = max(90, cfg["look"] * 2)
    df = cli.daily_ohlc(ticker, days=days)
    price = cli.last_trade_price(ticker)

    tail = df.tail(cfg["look"])
    low = float(tail["low"].min()); high = float(tail["high"].max())
    width = max(1e-9, high - low)
    pos = (price - low) / width  # 0..1

    closes = df["close"]
    slope = _linreg_slope(closes.tail(cfg["trend"]).values)
    slope_norm = slope / max(1e-9, price)
    atr_s = float(_atr_like(df, n=cfg["atr"]).iloc[-1])
    atr_l = float(_atr_like(df, n=cfg["atr"] * 2).iloc[-1])
    vol_ratio = atr_s / max(1e-9, atr_l)
    streak = _streak(closes)

    last = df.iloc[-1]
    body, upper, lower = _wick_profile(last)
    long_upper = upper > body * 1.3 and upper > lower * 1.1
    long_lower = lower > body * 1.3 and lower > upper * 1.1

    prev_close = float(df["close"].iloc[-2])
    today_open = float(df["open"].iloc[-1])
    gap = today_open - prev_close
    gap_rel = abs(gap) / max(1e-9, prev_close)

    near_top = pos >= 0.7
    near_bottom = pos <= 0.3
    mid_zone = 0.45 < pos < 0.55
    strong_up = slope_norm > 0.001 and vol_ratio > 1.05
    strong_down = slope_norm < -0.001 and vol_ratio > 1.05
    breakout_up = near_top and strong_up and (price >= high * 0.995)
    breakout_dn = near_bottom and strong_down and (price <= low * 1.005)

    if breakout_up:
        action, scenario = "BUY", "breakout_up"
    elif breakout_dn:
        action, scenario = "SHORT", "breakout_dn"
    elif near_top and not strong_up:
        action, scenario = "SHORT", "fade_top"
    elif near_bottom and not strong_down:
        action, scenario = "BUY", "bounce_bottom"
    elif mid_zone:
        action, scenario = "WAIT", "mid_range"
    else:
        action, scenario = ("BUY" if slope_norm >= 0 else "SHORT"), "trend_follow"

    edge = abs(pos - 0.5) * 2
    trn = _clip01(abs(slope_norm) * 1800)
    vol = _clip01((vol_ratio - 0.9) / 0.6)
    base = 0.48 + 0.25 * edge + 0.17 * trn + 0.10 * vol
    if scenario == "mid_range": base -= 0.08
    conf = float(max(0.55, min(0.90, base)))

    step = max(1e-6, atr_s)

    if action == "BUY":
        entry = price + step * 0.10 if scenario == "breakout_up" else (price - step * 0.15 if scenario == "trend_follow" else price - step * 0.20)
        sl = price - step * 1.0
        tp1, tp2, tp3 = price + step * 0.8, price + step * 1.6, price + step * 2.4
        alt = "Если уйдёт ниже зоны покупателя — пропустить вход; ждать возврата и подтверждения сверху."
    elif action == "SHORT":
        entry = price - step * 0.10 if scenario == "breakout_dn" else (price + step * 0.15 if scenario == "trend_follow" else price + step * 0.20)
        sl = price + step * 1.0
        tp1, tp2, tp3 = price - step * 0.8, price - step * 1.6, price - step * 2.4
        alt = "Если пробьёт верх и удержится — не гнаться; ждать возврата и признаки слабости у максимумов."
    else:
        entry = price
        sl = price - step * 0.9
        tp1, tp2, tp3 = price + step * 0.7, price + step * 1.4, price + step * 2.1
        alt = "При уверенном пробое диапазона работаем по направлению после ретеста и подтверждения."

    p1 = _clip01(0.58 + 0.27 * (conf - 0.55) / 0.35)
    p2 = _clip01(0.44 + 0.21 * (conf - 0.55) / 0.35)
    p3 = _clip01(0.28 + 0.13 * (conf - 0.55) / 0.35)

    # Чипсы-контекст
    chips = []
    if near_top: chips.append("возле зоны продавца")
    if near_bottom: chips.append("возле зоны покупателя")
    if gap_rel > 0.012: chips.append("гэп " + ("вверх" if gap > 0 else "вниз"))
    if long_upper: chips.append("длинные тени сверху")
    if long_lower: chips.append("длинные тени снизу")
    if streak >= 3: chips.append(f"{streak} зелёных подряд")
    if streak <= -3: chips.append(f"{abs(streak)} красных подряд")
    if vol_ratio > 1.05: chips.append("волатильность растёт")
    if vol_ratio < 0.95: chips.append("волатильность сжимается")

    # «Живые» фразы
    seed = int(hashlib.sha1(f"{ticker}{df.index[-1].date()}".encode()).hexdigest(), 16) % (2**32)
    if action == "BUY":
        lead = _choose(seed, [
            "Снизу чувствуется спрос — ловим откат и реакцию.",
            "Покупатели рядом, цена подпирается снизу.",
            "Низ рядом, покупатель активен — вход на возврате."
        ])
    elif action == "SHORT":
        lead = _choose(seed, [
            "Сверху тяжело — ищем слабость у кромки.",
            "Цена под продавцом — сверху давят.",
            "Верх близко — работаем от отказа."
        ])
    else:
        lead = _choose(seed, [
            "Середина диапазона. Преимущества нет.",
            "Рынок посередине — спешить некуда.",
            "Баланс. Смотрим, куда толкнут."
        ])

    add = []
    if long_upper:
        add.append(_choose(seed+5, [
            "Длинные хвосты сверху — продавец отвечает.",
            "Сверху были выносы — импульс выдыхается.",
            "Под верхом оставили шипы — осторожно с погоней."
        ]))
    if long_lower:
        add.append(_choose(seed+6, [
            "Снизу хвосты — защитили.",
            "Прокалывали вниз и откупили — живой спрос.",
            "Дно прокалывали, закрыли выше — бычий намёк."
        ]))
    if gap_rel > 0.012:
        add.append(_choose(seed+7 if gap>0 else seed+8, [
            "Гэп держат — важно не сдуться.",
            "Разрыв по цене — следим за удержанием.",
            "После разрыва возвраты могут быть резкими."
        ]))

    if "breakout" in scenario:
        add.append("Импульс сильный, работаем по движению — после короткой паузы.")
    elif scenario == "trend_follow":
        add.append("Идём за текущим ходом — вход на откате, не догоняя.")
    elif scenario in ("fade_top", "bounce_bottom"):
        add.append("Играем от края диапазона — ждём реакцию и только потом входим.")
    elif scenario == "mid_range":
        add.append("Здесь у самой цены ловить нечего — ждём шага рынка.")

    note_text = " ".join([lead] + add[:2])  # не перегружаем
    note_html = f"<div style='margin-top:10px; opacity:0.95;'>{note_text}</div>"

    return {
        "last_price": float(price),
        "recommendation": {"action": action, "confidence": float(round(conf, 4))},
        "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
        "probs": {"tp1": float(p1), "tp2": float(p2), "tp3": float(p3)},
        "context": chips,
        "note_html": note_html,
        "alt": alt,
    }
