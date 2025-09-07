# core/strategy.py
import os
import hashlib
import random
import numpy as np
import pandas as pd
from core.polygon_client import PolygonClient

# ---------- math helpers ----------
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

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

def _monthly_atr(df: pd.DataFrame, n_months: int = 6) -> float:
    m = df.resample("M").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(m) < 2:
        return float((df["high"] - df["low"]).tail(30).mean())
    hl = m["high"] - m["low"]
    hc = (m["high"] - m["close"].shift(1)).abs()
    lc = (m["low"] - m["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return float(tr.rolling(n_months, min_periods=1).mean().iloc[-1])

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

# ---------- horizon & pivots ----------
def _horizon_cfg(text: str):
    # ST: weekly pivots; MID/LT: monthly pivots
    if "Кратко" in text:
        return dict(look=60,  trend=14, atr=14, pivot_period="W-FRI", use_weekly_atr=True,  use_monthly_atr=False)
    if "Средне" in text:
        return dict(look=120, trend=28, atr=14, pivot_period="M",     use_weekly_atr=False, use_monthly_atr=True)
    return dict(look=240, trend=56, atr=14, pivot_period="M",         use_weekly_atr=False, use_monthly_atr=True)

def _hz_tag(text: str) -> str:
    if "Кратко" in text:  return "ST"
    if "Средне" in text:  return "MID"
    return "LT"

def _last_period_hlc(df: pd.DataFrame, rule: str):
    g = df.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(g) < 2: return None
    row = g.iloc[-2]  # строго завершённый период
    return float(row["high"]), float(row["low"]), float(row["close"])

def _fib_pivots(H: float, L: float, C: float):
    P = (H + L + C) / 3.0
    d = H - L
    R1 = P + 0.382 * d; R2 = P + 0.618 * d; R3 = P + 1.000 * d
    S1 = P - 0.382 * d; S2 = P - 0.618 * d; S3 = P - 1.000 * d
    return {"P": P, "R1": R1, "R2": R2, "R3": R3, "S1": S1, "S2": S2, "S3": S3, "_range": d}

def _pivot_ladder(piv: dict) -> list[float]:
    keys = ["S3", "S2", "S1", "P", "R1", "R2", "R3"]
    return [float(piv[k]) for k in keys if piv.get(k) is not None]

# (1) К какому «поясу» пивотов относится цена (-3..+3)
def _classify_band(price: float, piv: dict, buf: float) -> int:
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

# (2) Три цели от лестницы пивотов с fallback на ATR-шаг
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

# ---------- HA & MACD (лёгкие признаки) ----------
def _ha_streak(df: pd.DataFrame) -> int:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    ha_close = (o + h + l + c) / 4.0
    ha_open = ha_close.copy()
    ha_open.iloc[0] = o.iloc[0]
    for i in range(1, len(df)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2.0
    green = (ha_close > ha_open).astype(int)
    red   = (ha_close < ha_open).astype(int)
    s = 0
    for i in range(len(df)-1, -1, -1):
        if green.iloc[i] == 1:
            if s < 0: break
            s += 1
        elif red.iloc[i] == 1:
            if s > 0: break
            s -= 1
        else:
            break
    return int(s)

def _macd_hist_streak(closes: pd.Series) -> int:
    macd = _ema(closes, 12) - _ema(closes, 26)
    sig  = _ema(macd, 9)
    hist = macd - sig
    s = 0
    for i in range(len(hist)-1, 0, -1):
        if hist.iloc[i] > 0:
            if s < 0: break
            s += 1
        elif hist.iloc[i] < 0:
            if s > 0: break
            s -= 1
        else:
            break
    return int(s)

# ---------- пост-фильтр целей: минимальные дистанции ----------
def _apply_tp_floors(entry: float, sl: float, tp1: float, tp2: float, tp3: float,
                     action: str, hz_tag: str, price: float, atr_val: float):
    if action not in ("BUY", "SHORT"):
        return tp1, tp2, tp3
    risk = abs(entry - sl)
    if risk <= 1e-9:
        return tp1, tp2, tp3
    side = 1 if action == "BUY" else -1

    # Чуть выше пороги для LT
    min_rr   = {"ST": 0.80, "MID": 1.00, "LT": 1.30}
    min_pct  = {"ST": 0.006, "MID": 0.012, "LT": 0.025}
    atr_mult = {"ST": 0.50, "MID": 0.90, "LT": 1.40}

    floor1 = max(min_rr[hz_tag] * risk, min_pct[hz_tag] * price, atr_mult[hz_tag] * atr_val)
    if abs(tp1 - entry) < floor1:
        tp1 = entry + side * floor1

    floor2 = max(1.6 * floor1, (min_rr[hz_tag] * 1.8) * risk)
    if abs(tp2 - entry) < floor2:
        tp2 = entry + side * floor2

    min_gap3 = max(0.8 * floor1, 0.6 * risk)
    if abs(tp3 - tp2) < min_gap3:
        tp3 = tp2 + side * min_gap3

    return tp1, tp2, tp3

# ---------- «рельсы» по тренду и RR-потолки ----------
def _trend_score(pos: float, slope_norm: float, band: int, streak_p: int,
                 ha_s: int, macd_s: int, hz: str) -> float:
    s = 0.0
    s += 1.2 if slope_norm > 0 else -1.2
    s += 0.6 if pos >= 0.6 else (-0.6 if pos <= 0.4 else 0.0)
    s += 0.5 if band >= +1 else (-0.5 if band <= -1 else 0.0)
    if streak_p >= 3: s += 0.3
    if streak_p <= -3: s -= 0.3
    ha_thr = {"ST":4,"MID":5,"LT":6}[hz]
    mac_thr= {"ST":4,"MID":6,"LT":8}[hz]
    if ha_s >= ha_thr: s += 0.5
    if ha_s <= -ha_thr: s -= 0.5
    if macd_s >= mac_thr: s += 0.5
    if macd_s <= -mac_thr: s -= 0.5
    return s

def _cap_tp_tails(entry: float, sl: float, tp1: float, tp2: float, tp3: float,
                  action: str, hz: str, price: float, piv: dict,
                  trend_score: float, atr_ref: float):
    risk = abs(sl - entry) if abs(sl - entry) > 1e-9 else max(1.0, 0.3*atr_ref)
    rr_cap_tp2 = {"ST": 1.6, "MID": 1.9, "LT": 2.1}[hz]
    rr_cap_tp3 = {"ST": 2.0, "MID": 2.3, "LT": 2.5}[hz]

    if action == "SHORT" and trend_score >= 1.0:  # сильный ап-тренд
        rr_cap_tp2 -= 0.2; rr_cap_tp3 -= 0.5
        s1 = piv.get("S1", piv["P"])
        floor3 = max(piv["P"] - 0.4*atr_ref, (piv["P"] + s1) / 2.0)
        tp3 = max(tp3, floor3)
        tp2 = max(tp2, tp1 - 0.6*(tp1 - floor3))
    if action == "BUY" and trend_score <= -1.0:  # сильный даун-тренд
        rr_cap_tp2 -= 0.2; rr_cap_tp3 -= 0.5
        r1 = piv.get("R1", piv["P"])
        ceil3 = min(piv["P"] + 0.4*atr_ref, (piv["P"] + r1) / 2.0)
        tp3 = min(tp3, ceil3)
        tp2 = min(tp2, tp1 + 0.6*(ceil3 - tp1))

    if action == "BUY":
        tp2 = min(tp2, entry + rr_cap_tp2 * risk)
        tp3 = min(tp3, entry + rr_cap_tp3 * risk)
    elif action == "SHORT":
        tp2 = max(tp2, entry - rr_cap_tp2 * risk)
        tp3 = max(tp3, entry - rr_cap_tp3 * risk)
    return tp1, tp2, tp3

# ---------- упорядочивание целей ----------
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
    days = max(90, cfg["look"] * 2)
    df = cli.daily_ohlc(ticker, days=days)      # DatetimeIndex обязателен
    price = cli.last_trade_price(ticker)

    # позиция в диапазоне
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
    atr_m  = _monthly_atr(df) if cfg.get("use_monthly_atr") else atr_d
    atr_ref = atr_w if hz == "ST" else atr_m

    vol_ratio = atr_d / max(1e-9, float(_atr_like(df, n=cfg["atr"] * 2).iloc[-1]))
    streak_p  = _streak(closes)

    last_row = df.iloc[-1]
    body, upper, lower = _wick_profile(last_row)
    long_upper = upper > body * 1.3 and upper > lower * 1.1
    long_lower = lower > body * 1.3 and lower > upper * 1.1

    # HA & MACD признаки
    ha_s   = _ha_streak(df.tail(120))
    mac_s  = _macd_hist_streak(closes)

    # пивоты (последний завершённый период)
    hlc = _last_period_hlc(df, cfg["pivot_period"])
    if not hlc:
        hlc = (
            float(df["high"].tail(60).max()),
            float(df["low"].tail(60).min()),
            float(df["close"].iloc[-1]),
        )
    H, L, C = hlc
    piv = _fib_pivots(H, L, C)
    P, R1, R2 = piv["P"], piv["R1"], piv.get("R2")

    buf  = 0.25 * atr_ref
    band = _classify_band(price, piv, buf)  # -3..+3

    # ---- сценарии ----
    last_o, last_c = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
    last_h = float(df["high"].iloc[-1])
    upper_wick_d = max(0.0, last_h - max(last_o, last_c))
    body_d = abs(last_c - last_o)
    bearish_reject = (last_c < last_o) and (upper_wick_d > body_d)
    very_high_pos = pos >= 0.80

    if very_high_pos:
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
            action, scenario = ("BUY", "revert_from_bottom") if (streak_p <= -3 or long_lower) else ("BUY", "trend_follow")
        else:
            action, scenario = ("BUY", "revert_from_bottom") if band <= -2 else ("WAIT", "upper_wait")

    # overheat/oversold фильтр по HA+MACD
    ha_thr = {"ST":4,"MID":5,"LT":6}[hz]
    mac_thr= {"ST":4,"MID":6,"LT":8}[hz]
    overheat = (band >= +2) and (ha_s >= ha_thr) and (mac_s >= mac_thr)
    oversold = (band <= -2) and (ha_s <= -ha_thr) and (mac_s <= -mac_thr)
    if action == "BUY" and overheat:
        action, scenario = "WAIT", "overheat_top"
    if action == "SHORT" and oversold:
        action, scenario = "WAIT", "exhaust_bottom"

    # уверенность
    base = 0.55 + 0.12 * _clip01(abs(slope_norm) * 1800) + 0.08 * _clip01((vol_ratio - 0.9) / 0.6)
    if action == "WAIT": base -= 0.07
    if band >= +1 and action == "BUY": base -= 0.10
    if band <= -1 and action == "BUY": base += 0.05
    conf = float(max(0.55, min(0.90, base)))

    # ======== AI override (если есть модель) ========
    try:
        from core.ai_inference import score_signal
    except Exception:
        score_signal = None
    if score_signal is not None:
        feats = dict(
            pos=pos, slope_norm=slope_norm,
            atr_over_price=(atr_ref / max(1e-9, price)),
            vol_ratio=vol_ratio, streak=float(streak_p), band=float(band),
            ha_streak=int(ha_s), macd_streak=int(mac_s),
            long_upper=bool(long_upper), long_lower=bool(long_lower),
        )
        hz_tag = _hz_tag(horizon)
        out_ai = score_signal(feats, hz=hz_tag)
        if out_ai is not None:
            p_long = float(out_ai.get("p_long", 0.5))
            th_long = float(os.getenv("ARXORA_AI_TH_LONG", "0.55"))
            th_short = float(os.getenv("ARXORA_AI_TH_SHORT", "0.45"))
            if p_long >= th_long:
                action = "BUY";  conf = max(conf, 0.55 + 0.35 * (p_long - th_long) / max(1e-9, 1.0 - th_long))
            elif p_long <= th_short:
                action = "SHORT"; conf = max(conf, 0.55 + 0.35 * ((th_short - p_long) / max(1e-9, th_short)))
            else:
                action = "WAIT";  conf = max(0.55, conf - 0.07)
    # ======== конец override ========

    # уровни
    step_ref = atr_ref
    if action == "BUY":
        if scenario == "breakout_up":
            base_ref = R2 if R2 is not None else R1
            entry = max(price, base_ref + 0.10 * step_ref)
            sl    = base_ref - 1.00 * step_ref
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_ref)
        elif price < P:
            entry = max(price, piv["S1"] + 0.15 * step_ref); sl = piv["S1"] - 0.60 * step_ref
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_ref)
        else:
            entry = max(price, P + 0.10 * step_ref); sl = P - 0.60 * step_ref
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_ref)
        alt = "Если продавят ниже и не вернут — не лезем; ждём возврата сверху и подтверждения."
    elif action == "SHORT":
        if price >= R1:
            entry = min(price, R1 - 0.15 * step_ref)
            sl    = R1 + 0.60 * step_ref
        else:
            entry = price + 0.15 * step_ref
            sl    = price + 1.00 * step_ref
        tp1, tp2, tp3 = _three_targets_from_pivots(entry, "SHORT", piv, step_ref)
        alt = "Если протолкнут выше и удержат — без погони; ждём возврата и признаков слабости сверху."
    else:  # WAIT
        entry = price
        sl    = price - 0.90 * step_ref
        tp1, tp2, tp3 = entry + 0.7 * step_ref, entry + 1.4 * step_ref, entry + 2.1 * step_ref
        alt = "Под верхом — не догоняю; работаю на пробое после ретеста или на откате к опоре."

    # фиксы целей
    tp1, tp2, tp3 = _apply_tp_floors(entry, sl, tp1, tp2, tp3, action, hz, price, atr_ref)
    tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)
    t_score = _trend_score(pos, slope_norm, band, streak_p, ha_s, mac_s, hz)
    tp1, tp2, tp3 = _cap_tp_tails(entry, sl, tp1, tp2, tp3, action, hz, price, piv, t_score, atr_ref)
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
    if streak_p >= 3: chips.append(f"{streak_p} зелёных подряд")
    if streak_p <= -3: chips.append(f"{abs(streak_p)} красных подряд")

    # живой текст
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
    note_html = f"<div style='margin-top:10px;opacity:.95;'>{lead}</div>"

    return {
        "last_price": float(price),
        "recommendation": {"action": action, "confidence": float(round(conf, 4))},
        "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
        "probs": {"tp1": float(p1), "tp2": float(p2), "tp3": float(p3)},
        "context": chips,
        "note_html": note_html,
        "alt": alt,
    }
