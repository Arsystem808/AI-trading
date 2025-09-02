import hashlib
import random
import numpy as np
import pandas as pd
from core.polygon_client import PolygonClient

# ---------- helpers (не раскрываются в UI) ----------

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
    for i in range(len(closes)-1, 0, -1):
        d = closes.iloc[i] - closes.iloc[i-1]
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
    body  = abs(c - o)
    upper = max(0.0, h - max(o, c))
    lower = max(0.0, min(o, c) - l)
    return body, upper, lower

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _horizon_cfg(text: str):
    # period: какую "завершенную" свечу брать для пивотов
    if "Кратко" in text:  return dict(look=60, trend=14, atr=14, pivot_period="W-FRI")
    if "Средне" in text:  return dict(look=120, trend=28, atr=14, pivot_period="W-FRI")
    return dict(look=240, trend=56, atr=14, pivot_period="M")

# ---------- Fibonacci pivots (скрытые) ----------

def _last_period_hlc(df: pd.DataFrame, rule: str):
    # Берем предыдущий завершенный период: high/low/close
    g = df.resample(rule).agg({"high":"max","low":"min","close":"last"}).dropna()
    if len(g) < 2:
        return None
    row = g.iloc[-2]  # последняя ЗАКРЫТАЯ неделя/месяц
    return float(row["high"]), float(row["low"]), float(row["close"])

def _fib_pivots(H: float, L: float, C: float):
    P = (H + L + C) / 3.0
    d = H - L
    R1 = P + 0.382 * d; R2 = P + 0.618 * d; R3 = P + 1.000 * d
    S1 = P - 0.382 * d; S2 = P - 0.618 * d; S3 = P - 1.000 * d
    return {"P":P,"R1":R1,"R2":R2,"R3":R3,"S1":S1,"S2":S2,"S3":S3}

# ---------- основная логика ----------

def analyze_asset(ticker: str, horizon: str):
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)

    # Данные
    days = max(90, cfg["look"]*2)
    df = cli.daily_ohlc(ticker, days=days)
    price = cli.last_trade_price(ticker)

    tail = df.tail(cfg["look"])
    low  = float(tail["low"].min())
    high = float(tail["high"].max())
    width = max(1e-9, high - low)
    pos   = (price - low) / width  # положение в коридоре 0..1

    closes = df["close"]
    slope  = _linreg_slope(closes.tail(cfg["trend"]).values)
    slope_norm = slope / max(1e-9, price)
    atr_s  = float(_atr_like(df, n=cfg["atr"]).iloc[-1])
    atr_l  = float(_atr_like(df, n=cfg["atr"]*2).iloc[-1])
    vol_ratio = atr_s / max(1e-9, atr_l)
    streak    = _streak(closes)

    last = df.iloc[-1]
    body, upper, lower = _wick_profile(last)
    long_upper = upper > body*1.3 and upper > lower*1.1
    long_lower = lower > body*1.3 and lower > upper*1.1

    # Пивоты предыдущего периода
    piv = None
    hlc = _last_period_hlc(df, cfg["pivot_period"])
    if hlc:
        H, L, C = hlc
        piv = _fib_pivots(H, L, C)

    # --- сценарные зоны (по пивотам) ---
    # ничего не называем в UI, но внутри используем точные уровни
    near_mid    = False
    near_upper  = False
    near_lower  = False
    upper_ext   = False
    lower_ext   = False

    if piv:
        P,R1,R2,R3,S1,S2,S3 = piv["P"],piv["R1"],piv["R2"],piv["R3"],piv["S1"],piv["S2"],piv["S3"]
        buf = 0.25 * atr_s  # "вблизи уровня" — четверть дневной волатильности

        near_mid   = abs(price - P)  <= buf
        near_upper = abs(price - R1) <= buf
        near_lower = abs(price - S1) <= buf
        upper_ext  = (price >= R1 + buf)  # выше R1 заметно
        lower_ext  = (price <= S1 - buf)  # ниже S1 заметно
    else:
        P = (high+low)/2  # fallback

    # --- принятие решения ---
    # 1) Пробой из зон (сильный импульс)
    breakout_up = piv and (upper_ext and slope_norm > 0 and vol_ratio > 1.02)
    breakout_dn = piv and (lower_ext and slope_norm < 0 and vol_ratio > 1.02)

    # 2) Разворот от верхней/нижней кромки после "перегрева" (серия свечей, тени)
    overheat_up   = streak >= 3 or long_upper
    overheat_down = streak <= -3 or long_lower

    if breakout_up:
        action, scenario = "BUY", "breakout_up"
    elif breakout_dn:
        action, scenario = "SHORT", "breakout_dn"
    elif near_upper and overheat_up:
        # пример из твоего ядра: длительный подъём + подход к "верхней зоне" → коррекция к области между верхом и серединой
        action, scenario = "SHORT", "revert_from_top"
    elif near_lower and overheat_down:
        action, scenario = "BUY", "revert_from_bottom"
    elif near_mid:
        action, scenario = "WAIT", "mid_range"
    else:
        # Внутри диапазона идём по наклону
        action, scenario = ("BUY" if slope_norm >= 0 else "SHORT"), "trend_follow"

    # Уверенность (0.55–0.90)
    edge = abs(pos-0.5)*2
    trn  = _clip01(abs(slope_norm)*1800)
    vol  = _clip01((vol_ratio-0.9)/0.6)
    base = 0.50 + 0.22*edge + 0.16*trn + 0.10*vol
    if scenario == "mid_range": base -= 0.08
    conf = float(max(0.55, min(0.90, base)))

    # Шаг уровней от текущей волатильности
    step = max(1e-6, atr_s)

    # --- уровни по пивот-логике ---
    # Стоп всегда за ближайшим "краем" + запас от волатильности.
    if action == "SHORT":
        if piv:
            # short от верхней зоны → цели: 1) середина между верхом и серединой, 2) середина (P), 3) нижняя зона
            upper_ref = R1 if scenario == "revert_from_top" else max(R1, price) if price > R1 else R1
            mid_ref   = P
            lower_ref = S1
            entry = min(price, upper_ref - 0.15*step)  # не гонимся; берем после микро-отката
            sl    = upper_ref + 0.60*step
            tp1   = (upper_ref + mid_ref) / 2.0
            tp2   = mid_ref
            tp3   = (mid_ref + lower_ref) / 2.0
        else:
            entry = price + 0.15*step; sl = price + 1.0*step
            tp1, tp2, tp3 = price - 0.8*step, price - 1.6*step, price - 2.4*step
        alt = "Если протолкнёт выше и удержит — без погони; ждём возврата и признаков слабости выше."
    elif action == "BUY":
        if piv:
            lower_ref = S1
            mid_ref   = P
            upper_ref = R1
            if scenario == "breakout_up":
                base_ref = R1
                entry = price + 0.10*step
                sl    = base_ref - 1.00*step
                tp1   = R2 if "R2" in piv else price + 0.8*step
                tp2   = (R2 + (piv.get("R3", R2) ))/2.0 if "R2" in piv else price + 1.6*step
                tp3   = piv.get("R3", tp2 + 0.8*step)
            else:
                # от нижней зоны/по тренду снизу → цели: 1) середина между низом и серединой, 2) середина, 3) верхняя зона
                entry = max(price, lower_ref + 0.15*step)
                sl    = lower_ref - 0.60*step
                tp1   = (lower_ref + mid_ref) / 2.0
                tp2   = mid_ref
                tp3   = (mid_ref + upper_ref) / 2.0
        else:
            entry = price - 0.15*step; sl = price - 1.0*step
            tp1, tp2, tp3 = price + 0.8*step, price + 1.6*step, price + 2.4*step
        alt = "Если продавят ниже и не вернут — не лезем; ждём возврата сверху и подтверждения."
    else:  # WAIT
        entry = price
        sl    = price - 0.9*step
        tp1, tp2, tp3 = price + 0.7*step, price + 1.4*step, price + 2.1*step
        alt = "На уверенном выходе из коридора — входим по направлению после возврата и подтверждения."

    # Вероятности целей ~ от уверенности
    p1 = _clip01(0.58 + 0.27*(conf-0.55)/0.35)
    p2 = _clip01(0.44 + 0.21*(conf-0.55)/0.35)
    p3 = _clip01(0.28 + 0.13*(conf-0.55)/0.35)

    # -------- «живой» текст и контекст-бейджи --------
    chips = []
    if near_upper: chips.append("верхняя кромка")
    if near_lower: chips.append("нижняя кромка")
    if near_mid:   chips.append("середина диапазона")
    if vol_ratio > 1.05: chips.append("волатильность растёт")
    if vol_ratio < 0.95: chips.append("волатильность сжимается")
    if streak >= 3: chips.append(f"{streak} зелёных подряд")
    if streak <= -3: chips.append(f"{abs(streak)} красных подряд")
    if long_upper: chips.append("тени сверху")
    if long_lower: chips.append("тени снизу")

    seed = int(hashlib.sha1(f"{ticker}{df.index[-1].date()}".encode()).hexdigest(), 16) % (2**32)
    rng  = random.Random(seed)

    if action == "BUY":
        lead = rng.choice([
            "Снизу чувствуется спрос — беру откат и реакцию.",
            "Покупатели рядом, цена подпирается — работаем по ходу.",
            "Низ близко, спрос живой — входим после возврата."
        ])
    elif action == "SHORT":
        lead = rng.choice([
            "Сверху тяжело — ищем слабость у кромки.",
            "Под потолком есть продавец — работаю от отказа.",
            "Верх близко, импульс выдыхается — готовлю шорт."
        ])
    else:
        lead = rng.choice([
            "Середина. Преимущества нет — без входа.",
            "Баланс сил, жду шага цены.",
            "Внутри коридора — спешить некуда."
        ])

    add = []
    if scenario == "breakout_up":
        add.append("Импульс вверх, работаем по движению — после короткой паузы.")
    if scenario == "breakout_dn":
        add.append("Импульс вниз, не ловлю нож — беру после отката.")
    if scenario in ("revert_from_top","revert_from_bottom"):
        add.append("Игра от края: сначала реакция, потом вход.")
    if scenario == "trend_follow":
        add.append("Идём за текущим ходом — вход не догоняя, а на откате.")

    note_html = f"<div style='margin-top:10px; opacity:0.95;'>{' '.join([lead]+add[:2])}</div>"

    return {
        "last_price": float(price),
        "recommendation": {"action": action, "confidence": float(round(conf,4))},
        "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
        "probs": {"tp1": float(p1), "tp2": float(p2), "tp3": float(p3)},
        "context": chips,
        "note_html": note_html,
        "alt": alt,
    }
