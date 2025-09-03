# core/strategy.py
import hashlib
import random
import numpy as np
import pandas as pd
from core.polygon_client import PolygonClient

# -------- helpers --------

def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def _weekly_atr(df: pd.DataFrame, n_weeks: int = 8) -> float:
    w = df.resample("W-FRI").agg({"high":"max","low":"min","close":"last"}).dropna()
    if len(w) < 2:
        return float((df["high"] - df["low"]).tail(14).mean())
    hl = w["high"] - w["low"]
    hc = (w["high"] - w["close"].shift(1)).abs()
    lc = (w["low"]  - w["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return float(tr.rolling(n_weeks, min_periods=1).mean().iloc[-1])

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
    body  = abs(c - o)
    upper = max(0.0, h - max(o, c))
    lower = max(0.0, min(o, c) - l)
    return body, upper, lower

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _horizon_cfg(text: str):
    # недельные пивоты для всех горизонтов, weekly ATR — для средне/долгосрока
    if "Кратко" in text:  return dict(look=60,  trend=14, atr=14, pivot_period="W-FRI", use_weekly_atr=False)
    if "Средне" in text:  return dict(look=120, trend=28, atr=14, pivot_period="W-FRI", use_weekly_atr=True)
    return dict(look=240, trend=56, atr=14, pivot_period="W-FRI", use_weekly_atr=True)

# -------- Fibonacci pivots (внутренне) --------

def _last_period_hlc(df: pd.DataFrame, rule: str):
    g = df.resample(rule).agg({"high":"max","low":"min","close":"last"}).dropna()
    if len(g) < 2: return None
    row = g.iloc[-2]
    return float(row["high"]), float(row["low"]), float(row["close"])

def _fib_pivots(H: float, L: float, C: float):
    P = (H + L + C) / 3.0
    d = H - L
    R1 = P + 0.382 * d; R2 = P + 0.618 * d; R3 = P + 1.000 * d
    S1 = P - 0.382 * d; S2 = P - 0.618 * d; S3 = P - 1.000 * d
    return {"P":P,"R1":R1,"R2":R2,"R3":R3,"S1":S1,"S2":S2,"S3":S3}

def _pivot_ladder(piv: dict) -> list[float]:
    keys = ["S3","S2","S1","P","R1","R2","R3"]
    return [float(piv[k]) for k in keys if k in piv and piv[k] is not None]

def _three_targets_from_pivots(entry: float, direction: str, piv: dict, step: float) -> tuple[float,float,float]:
    # цели всегда по сторону входа; если мало уровней — добавляем ATR-ступени
    ladder = sorted(set(_pivot_ladder(piv)))
    eps = 0.10 * step
    if direction == "BUY":
        ups = [x for x in ladder if x > entry + eps]
        while len(ups) < 3:
            k = len(ups) + 1
            ups.append(entry + (0.7 + 0.7*(k-1)) * step)
        return ups[0], ups[1], ups[2]
    else:
        dns = [x for x in ladder if x < entry - eps]
        dns = list(sorted(dns, reverse=True))
        while len(dns) < 3:
            k = len(dns) + 1
            dns.append(entry - (0.7 + 0.7*(k-1)) * step)
        return dns[0], dns[1], dns[2]

# -------- основная логика --------

def analyze_asset(ticker: str, horizon: str):
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)

    # данные
    days = max(90, cfg["look"]*2)
    df = cli.daily_ohlc(ticker, days=days)
    price = cli.last_trade_price(ticker)

    # позиция в своём диапазоне (для «не гнаться»)
    tail = df.tail(cfg["look"])
    rng_low  = float(tail["low"].min())
    rng_high = float(tail["high"].max())
    rng_w    = max(1e-9, rng_high - rng_low)
    pos = (price - rng_low) / rng_w  # 0..1

    closes = df["close"]
    slope  = _linreg_slope(closes.tail(cfg["trend"]).values)
    slope_norm = slope / max(1e-9, price)

    atr_d  = float(_atr_like(df, n=cfg["atr"]).iloc[-1])
    atr_w  = _weekly_atr(df) if cfg.get("use_weekly_atr") else atr_d
    vol_ratio = atr_d / max(1e-9, float(_atr_like(df, n=cfg["atr"]*2).iloc[-1]))
    streak    = _streak(closes)
    body, upper, lower = _wick_profile(df.iloc[-1])
    long_upper = upper > body*1.3 and upper > lower*1.1
    long_lower = lower > body*1.3 and lower > upper*1.1

    # пивоты предыдущей недели
    hlc = _last_period_hlc(df, cfg["pivot_period"])
    if hlc:
        H, L, C = hlc
    else:
        H, L, C = float(df["high"].tail(60).max()), float(df["low"].tail(60).min()), float(df["close"].iloc[-1])
    piv = _fib_pivots(H, L, C)

    # буфер «вблизи уровня»
    buf = 0.25 * atr_w
    P, R1, R2, R3, S1, S2, S3 = piv["P"], piv["R1"], piv.get("R2"), piv.get("R3"), piv["S1"], piv.get("S2"), piv.get("S3")

    # классификация полос (не выводим названия наружу)
    def band_of(price_):
        # -3: <S2, -2: [S2,S1), -1: [S1,P), 0: [P,R1), +1: [R1,R2), +2: [R2,R3), +3: >=R3
        neg_inf, pos_inf = -1e18, 1e18
        levels = [S2 if S2 is not None else neg_inf, S1, P, R1, R2 if R2 is not None else pos_inf, R3 if R3 is not None else pos_inf]
        if price_ < levels[0]-buf: return -3
        if price_ < levels[1]-buf: return -2
        if price_ < levels[2]-buf: return -1
        if price_ < levels[3]-buf: return 0
        if R2 is None or price_ < levels[4]-buf: return +1
        if price_ < levels[5]-buf: return +2
        return +3

    band = band_of(price)

    # сценарии (строже у верха) + «не гнаться» на средне/долгосроке
    overheat_up   = streak >= 3 or long_upper
    overheat_down = streak <= -3 or long_lower

    if band >= +2:
        if slope_norm > 0 and R2 is not None and price > (R2 + 0.6*buf):
            action, scenario = "BUY", "breakout_up"
        else:
            action, scenario = ("SHORT", "fade_top") if overheat_up else ("WAIT", "upper_wait")
    elif band == +1:
        action, scenario = ("BUY", "trend_follow") if (slope_norm > 0.001 and not overheat_up and pos < 0.8) else (("SHORT","fade_top") if overheat_up else ("WAIT","upper_wait"))
    elif band == 0:
        action, scenario = ("BUY", "trend_follow") if slope_norm >= 0 else ("WAIT", "mid_range")
    elif band == -1:
        action, scenario = "BUY", "revert_from_bottom" if overheat_down else ("BUY","trend_follow")[0]
    else:
        action, scenario = ("BUY","revert_from_bottom") if band <= -2 else ("WAIT","upper_wait")

    # guard: на средне/долгосроке в верхних 20% диапазона BUY запрещён, если нет пробоя
    if cfg.get("use_weekly_atr") and pos >= 0.8 and action == "BUY" and scenario != "breakout_up":
        action, scenario = "WAIT", "upper_wait"

    # уверенность
    base = 0.55 + 0.12*_clip01(abs(slope_norm)*1800) + 0.08*_clip01((vol_ratio-0.9)/0.6)
    if action == "WAIT": base -= 0.07
    if band >= +1 and action == "BUY": base -= 0.10
    if band <= -1 and action == "BUY": base += 0.05
    conf = float(max(0.55, min(0.90, base)))

    # уровни
    step_d, step_w = atr_d, atr_w
    if action == "BUY":
        if scenario == "breakout_up":
            base_ref = R2 if R2 is not None else R1
            entry = max(price, base_ref + 0.10*step_w)
            sl    = base_ref - 1.00*step_w
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
        elif band <= 0:
            if price < P:
                entry = max(price, S1 + 0.15*step_w); sl = S1 - 0.60*step_w
            else:
                entry = max(price, P  + 0.10*step_w); sl = P  - 0.60*step_w
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
        else:
            entry = max(price, R1 + 0.10*step_w); sl = R1 - 0.60*step_w
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
        alt = "Если продавят ниже и не вернут — не лезем; ждём возврата сверху и подтверждения."
    elif action == "SHORT":
        if band >= +1:
            entry = min(price, R1 - 0.15*step_w); sl = R1 + 0.60*step_w
        else:
            entry = price + 0.15*step_d;         sl = price + 1.00*step_d
        tp1, tp2, tp3 = _three_targets_from_pivots(entry, "SHORT", piv, step_w)
        alt = "Если протолкнут выше и удержат — без погони; ждём возврата и признаков слабости выше."
    else:  # WAIT
        entry = price; sl = price - 0.90*step_d
        tp1, tp2, tp3 = entry + 0.7*step_d, entry + 1.4*step_d, entry + 2.1*step_d
        alt = "Под верхом — не догоняю; работаю на пробое после ретеста или на откате к опоре."

    # вероятности целей
    p1 = _clip01(0.58 + 0.27*(conf-0.55)/0.35)
    p2 = _clip01(0.44 + 0.21*(conf-0.55)/0.35)
    p3 = _clip01(0.28 + 0.13*(conf-0.55)/0.35)

    # контекст-чипсы (без названий индикаторов)
    chips = []
    if band >= +2: chips.append("верхний диапазон")
    elif band == +1: chips.append("под верхним краем")
    elif band == 0: chips.append("средняя зона")
    elif band == -1: chips.append("нижняя половина")
    elif band <= -2: chips.append("ниже опоры")
    if vol_ratio > 1.05: chips.append("волатильность растёт")
    if vol_ratio < 0.95: chips.append("волатильность сжимается")
    if streak >= 3: chips.append(f"{streak} зелёных подряд")
    if streak <= -3: chips.append(f"{abs(streak)} красных подряд")
    if long_upper: chips.append("тени сверху")
    if long_lower: chips.append("тени снизу")

    # живой текст — без терминов S1/P/R2
    seed = int(hashlib.sha1(f"{ticker}{df.index[-1].date()}".encode()).hexdigest(), 16) % (2**32)
    rng  = random.Random(seed)

    if action == "BUY":
        lead = rng.choice([
            "Снизу чувствуется спрос — беру откат и реакцию.",
            "Поддержка рядом, цена подпирается — работаю по ходу.",
            "Опора близко, спрос живой — вхожу на возврате."
        ]) if band <= 0 else rng.choice([
            "Поддержка держит — вход после возврата, не догоняю.",
            "Отталкиваемся от опоры — беру после отката."
        ])
    elif action == "SHORT":
        lead = rng.choice([
            "Сверху тяжело — ищу слабость у кромки.",
            "Под потолком продавец — работаю от отказа.",
            "Импульс выдыхается у верха — готовлю шорт."
        ])
    else:
        lead = rng.choice([
            "Под кромкой. Преимущества нет — без входа.",
            "Баланс силы у верха — даю цене определиться.",
            "Здесь не гонюсь — жду откат к опоре/центру."
        ])

    adds = []
    if band >= +2 and action != "BUY": adds.append("Погоня сверху — не мой вход. Жду откат.")
    if action == "BUY" and band <= 0:  adds.append("Вход только после возврата/реакции, без рывка за ценой.")
    if streak >= 4:                     adds.append("Серия длинная — риски отката повыше.")

    note_html = f"<div style='margin-top:10px; opacity:0.95;'>{' '.join([lead]+adds[:2])}</div>"

    return {
        "last_price": float(price),
        "recommendation": {"action": action, "confidence": float(round(conf,4))},
        "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
        "probs": {"tp1": float(p1), "tp2": float(p2), "tp3": float(p3)},
        "context": chips,       # теперь без "между S1 и P"
        "note_html": note_html, # текст без упоминаний индикаторов
        "alt": alt,
    }
