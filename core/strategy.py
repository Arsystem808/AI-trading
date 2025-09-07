# core/strategy.py
import os
import hashlib
import random
import numpy as np
import pandas as pd
from core.polygon_client import PolygonClient

# ---------- helpers ----------
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

def _hz_tag(text: str) -> str:
    if "Кратко" in text:  return "ST"
    if "Средне" in text:  return "MID"
    return "LT"

def _horizon_cfg(text: str):
    # Только weekly для ST и monthly для MID/LT
    hz = _hz_tag(text)
    if hz == "ST":
        return dict(look=60, trend=14, atr=14, pivot_period="W-FRI", use_weekly_atr=False)
    else:  # MID и LT
        return dict(look=240 if hz == "LT" else 120,
                    trend=56 if hz == "LT" else 28,
                    atr=14,
                    pivot_period="MS",       # месяц
                    use_weekly_atr=True)

# ---------- Fibonacci-pivots ----------
def _last_period_hlc(df: pd.DataFrame, rule: str):
    g = df.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(g) < 2:
        return None
    row = g.iloc[-2]
    return float(row["high"]), float(row["low"]), float(row["close"])

def _fib_pivots(H: float, L: float, C: float):
    P = (H + L + C) / 3.0
    d = H - L
    R1 = P + 0.382 * d; R2 = P + 0.618 * d; R3 = P + 1.000 * d
    S1 = P - 0.382 * d; S2 = P - 0.618 * d; S3 = P - 1.000 * d
    return {"P": P, "R1": R1, "R2": R2, "R3": R3, "S1": S1, "S2": S2, "S3": S3}

def _pivot_ladder(piv: dict) -> list[float]:
    keys = ["S3", "S2", "S1", "P", "R1", "R2", "R3"]
    return [float(piv[k]) for k in keys if k in piv and piv[k] is not None]

def _three_targets_from_pivots(entry: float, direction: str, piv: dict, step: float):
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

def _classify_band(price: float, piv: dict, buf: float) -> int:
    # -3: <S2, -2: [S2,S1), -1: [S1,P), 0: [P,R1), +1: [R1,R2), +2: [R2,R3), +3: >=R3
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

# ---------- пост-фильтр целей ----------
def _apply_tp_floors(entry: float, sl: float, tp1: float, tp2: float, tp3: float,
                     action: str, hz_tag: str, price: float, atr_val: float):
    if action not in ("BUY", "SHORT"):
        return tp1, tp2, tp3

    risk = abs(entry - sl)
    if risk <= 1e-9:
        return tp1, tp2, tp3

    side = 1 if action == "BUY" else -1

    # Усилили LT
    min_rr   = {"ST": 0.80, "MID": 1.00, "LT": 1.40}
    min_pct  = {"ST": 0.006, "MID": 0.012, "LT": 0.020}
    atr_mult = {"ST": 0.50, "MID": 0.80, "LT": 1.40}

    floor1 = max(min_rr[hz_tag] * risk, min_pct[hz_tag] * price, atr_mult[hz_tag] * atr_val)

    if abs(tp1 - entry) < floor1:
        tp1 = entry + side * floor1

    floor2 = max(1.7 * floor1, (min_rr[hz_tag] * 1.9) * risk)
    if abs(tp2 - entry) < floor2:
        tp2 = entry + side * floor2

    min_gap3 = max(0.9 * floor1, 0.7 * risk)
    if abs(tp3 - tp2) < min_gap3:
        tp3 = tp2 + side * min_gap3

    return tp1, tp2, tp3

def _order_targets(entry: float, tp1: float, tp2: float, tp3: float, action: str, eps: float = 1e-6):
    side = 1 if action == "BUY" else -1
    arr = sorted([float(tp1), float(tp2), float(tp3)], key=lambda x: side * (x - entry))
    d0 = side * (arr[0] - entry)
    d1 = side * (arr[1] - entry)
    d2 = side * (arr[2] - entry)
    if d1 - d0 < eps:
        arr[1] = entry + side * max(d0 + max(eps, 0.1 * abs(d0)), d1 + eps)
    if side * (arr[2] - entry) - side * (arr[1] - entry) < eps:
        d1 = side * (arr[1] - entry)
        arr[2] = entry + side * max(d1 + max(eps, 0.1 * abs(d1)), d2 + eps)
    return arr[0], arr[1], arr[2]

# ---------- основная логика ----------
def analyze_asset(ticker: str, horizon: str):
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)
    hz = _hz_tag(horizon)

    # данные
    days = max(180, cfg["look"] * 2) if hz != "ST" else max(90, cfg["look"] * 2)
    df = cli.daily_ohlc(ticker, days=days)
    price = cli.last_trade_price(ticker)

    # позиция в своём диапазоне
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
    vol_ratio = atr_d / max(1e-9, float(_atr_like(df, n=cfg["atr"] * 2).iloc[-1]))
    streak    = _streak(closes)

    last_row = df.iloc[-1]
    body, upper, lower = _wick_profile(last_row)
    long_upper = upper > body * 1.3 and upper > lower * 1.1
    long_lower = lower > body * 1.3 and lower > upper * 1.1

    # --- pivots: ST weekly, MID/LT monthly (предыдущий период) ---
    hlc = _last_period_hlc(df, cfg["pivot_period"])
    if not hlc:
        # запасной вариант: широкий дневной диапазон
        look_fallback = 90 if hz == "ST" else 180
        hlc = (
            float(df["high"].tail(look_fallback).max()),
            float(df["low"].tail(look_fallback).min()),
            float(df["close"].iloc[-1]),
        )
    H, L, C = hlc
    piv = _fib_pivots(H, L, C)
    P, R1, R2 = piv["P"], piv["R1"], piv.get("R2")

    # классификация зоны
    buf  = 0.25 * atr_w
    band = _classify_band(price, piv, buf)  # -3..+3

    # ---- сценарии с тренд-фильтром ----
    very_high_pos = pos >= 0.80
    strong_up = slope_norm > (0.0012 if hz == "ST" else 0.0009)  # LT/MID мягче
    strong_down = slope_norm < -(0.0012 if hz == "ST" else 0.0009)

    if very_high_pos:
        # у крыши — SHORT только при слабости тренда, иначе WAIT/BUY-breakout
        if strong_up and (R2 is None or price <= R2 + 0.6 * buf):
            action, scenario = "WAIT", "upper_wait"
        elif (R2 is not None) and (price > R2 + 0.6 * buf) and strong_up:
            action, scenario = "BUY", "breakout_up"
        else:
            action, scenario = "SHORT", "fade_top"
    else:
        if band >= +2:
            if strong_up and hz != "ST":
                action, scenario = "WAIT", "upper_wait"
            else:
                action, scenario = ("BUY", "breakout_up") if ((R2 is not None) and (price > R2 + 0.6 * buf) and (slope_norm > 0)) else ("SHORT", "fade_top")
        elif band == +1:
            action, scenario = ("WAIT", "upper_wait") if (slope_norm > 0 and not long_upper) else ("SHORT", "fade_top")
        elif band == 0:
            action, scenario = ("BUY", "trend_follow") if slope_norm >= 0 else ("WAIT", "mid_range")
        elif band == -1:
            action, scenario = ("BUY", "revert_from_bottom") if (streak <= -3 or long_lower or strong_up) else ("BUY", "trend_follow")
        else:
            action, scenario = ("BUY", "revert_from_bottom") if band <= -2 else ("WAIT", "upper_wait")

    # уверенность
    base = 0.55 + 0.12 * _clip01(abs(slope_norm) * 1800) + 0.08 * _clip01((vol_ratio - 0.9) / 0.6)
    if action == "WAIT": base -= 0.07
    if band >= +1 and action == "BUY": base -= 0.10
    if band <= -1 and action == "BUY": base += 0.05
    # В длинном тренде не раздуваем уверенность в контртренд SHORT на MID/LT
    if action == "SHORT" and hz != "ST" and strong_up:
        base = min(base, 0.62)
    conf = float(max(0.55, min(0.90, base)))

    # ======== AI override (если есть модель) ========
    try:
        from core.ai_inference import score_signal
    except Exception:
        score_signal = None

    if score_signal is not None:
        feats = dict(
            pos=pos,
            slope_norm=slope_norm,
            atr_d_over_price=(atr_d / max(1e-9, price)),
            vol_ratio=vol_ratio,
            streak=float(streak),
            band=float(band),
            long_upper=bool(long_upper),
            long_lower=bool(long_lower),
        )
        out_ai = score_signal(feats, hz=hz)
        if out_ai is not None:
            p_long = float(out_ai.get("p_long", 0.5))
            th_long = float(os.getenv("ARXORA_AI_TH_LONG", "0.55"))
            th_short = float(os.getenv("ARXORA_AI_TH_SHORT", "0.45"))
            if p_long >= th_long:
                ai_action = "BUY"
                ai_conf = 0.55 + 0.35 * (p_long - th_long) / max(1e-9, 1.0 - th_long)
            elif p_long <= th_short:
                ai_action = "SHORT"
                ai_conf = 0.55 + 0.35 * ((th_short - p_long) / max(1e-9, th_short))
            else:
                ai_action = "WAIT"
                ai_conf = 0.55 - 0.07
            # в контртренд на MID/LT — прижимаем уверенность
            if ai_action == "SHORT" and hz != "ST" and strong_up:
                ai_conf = min(ai_conf, 0.62)
            action = ai_action
            conf = float(max(0.55, min(0.90, ai_conf)))
    # ======== конец AI override ========

    # уровни
    step_d, step_w = atr_d, atr_w
    if action == "BUY":
        if (R2 is not None) and (price > R2 + 0.6 * step_w) and strong_up:
            base_ref = R2
            entry = max(price, base_ref + 0.10 * step_w)
            sl    = base_ref - 1.00 * step_w
        elif price < P:
            entry = max(price, piv["S1"] + 0.20 * step_w); sl = piv["S1"] - 0.70 * step_w
        else:
            entry = max(price, P + 0.15 * step_w); sl = P - 0.70 * step_w
        tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
        alt = "Если продавят ниже и не вернут — не лезем; ждём возврата сверху и подтверждения."
    elif action == "SHORT":
        # у верха контртренд в долгосрок — осторожнее
        if price >= R1:
            entry = min(price, R1 - 0.20 * step_w)
            sl    = R1 + 0.70 * step_w
        else:
            entry = price + 0.20 * step_d
            sl    = price + 1.10 * step_d
        tp1, tp2, tp3 = _three_targets_from_pivots(entry, "SHORT", piv, step_w)
        # при сильном ап-тренде не «зарываем» TP3 слишком глубоко
        if hz != "ST" and strong_up:
            tp3 = max(tp2 - 0.9 * step_w, entry - 2.2 * step_w)
        alt = "Если протолкнут выше и удержат — без погони; ждём возврата и признаков слабости сверху."
    else:  # WAIT
        entry = price
        sl    = price - 1.10 * step_d
        tp1, tp2, tp3 = entry + 0.8 * step_d, entry + 1.6 * step_d, entry + 2.4 * step_d
        alt = "Под верхом — не догоняю; работаю на пробое после ретеста или на откате к опоре."

    # минимальные дистанции и порядок
    atr_for_floor = atr_w if hz != "ST" else atr_d
    tp1, tp2, tp3 = _apply_tp_floors(entry, sl, tp1, tp2, tp3, action, hz, price, atr_for_floor)
    tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)

    # вероятности
    p1 = _clip01(0.58 + 0.27 * (conf - 0.55) / 0.35)
    p2 = _clip01(0.44 + 0.21 * (conf - 0.55) / 0.35)
    p3 = _clip01(0.28 + 0.13 * (conf - 0.55) / 0.35)

    # контекст-чипсы
    chips = []
    if pos >= 0.8: chips.append("верхний диапазон")
    elif pos >= 0.6: chips.append("под верхним краем")
    elif pos >= 0.4: chips.append("средняя зона")
    elif pos >= 0.2: chips.append("нижняя половина")
    else:            chips.append("ниже опоры")
    if vol_ratio > 1.05: chips.append("волатильность растёт")
    if vol_ratio < 0.95: chips.append("волатильность сжимается")
    if streak >= 3: chips.append(f"{streak} зелёных подряд")
    if streak <= -3: chips.append(f"{abs(streak)} красных подряд")
    if long_upper: chips.append("тени сверху")
    if long_lower: chips.append("тени снизу")

    # «живой» текст
    seed = int(hashlib.sha1(f"{ticker}{df.index[-1].date()}".encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    if action == "BUY":
        lead = rng.choice([
            "Поддержка рядом — работаю по ходу, только после отката.",
            "Опора держит — вход не догоняя, на возврате.",
            "Спрос живой — беру реакцию от опоры.",
        ]) if pos <= 0.6 else rng.choice([
            "Поддержка снизу — вход аккуратно, без погони.",
            "Отталкиваемся от опоры — беру после паузы.",
        ])
    elif action == "SHORT":
        lead = rng.choice([
            "Сверху тяжело — ищу слабость у кромки.",
            "Под потолком продавец — работаю от отказа.",
            "Импульс выдыхается у верха — готовлю шорт.",
        ])
    else:
        lead = rng.choice([
            "Без входа: жду пробой с ретестом или откат к опоре/центру.",
            "Под кромкой — даю цене определиться.",
            "Здесь не гонюсь — план от отката.",
        ])
    adds = []
    if pos >= 0.8 and action != "BUY": adds.append("Погоня сверху — не мой вход. Жду откат.")
    if action == "BUY" and pos <= 0.4:  adds.append("Вход только после реакции, без рывка.")
    if streak >= 4:                      adds.append("Серия длинная — риски отката выше.")
    note_html = f"<div style='margin-top:10px; opacity:0.95;'>{' '.join([lead] + adds[:2])}</div>"

    return {
        "last_price": float(price),
        "recommendation": {"action": action, "confidence": float(round(conf, 4))},
        "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
        "probs": {"tp1": float(p1), "tp2": float(p2), "tp3": float(p3)},
        "context": chips,
        "note_html": note_html,
        "alt": alt,
    }
