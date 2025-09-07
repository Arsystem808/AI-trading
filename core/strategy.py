# core/strategy.py
import os
import hashlib
import random
import numpy as np
import pandas as pd
from core.polygon_client import PolygonClient

# ========================== helpers ==========================

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
    lc = (w["low"] - w["close"].shift(1)).abs()
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
    body = abs(c - o)
    upper = max(0.0, h - max(o, c))
    lower = max(0.0, min(o, c) - l)
    return body, upper, lower

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

# ========================== horizons ==========================

def _horizon_cfg(text: str):
    # Уровни сделок: ST=weekly, MID=monthly, LT=yearly
    # Интуиция (HA/MACD) — всегда weekly.
    if "Кратко" in text:
        return dict(look=120, trend=14, atr=14, pivot_rule="W-FRI", ha_macd_rule="W-FRI", use_weekly_atr=False)
    if "Средне" in text:
        return dict(look=240, trend=28, atr=14, pivot_rule="M",     ha_macd_rule="W-FRI", use_weekly_atr=True)
    return dict(look=540, trend=56, atr=14, pivot_rule="A-DEC",     ha_macd_rule="W-FRI", use_weekly_atr=True)

def _hz_tag(text: str) -> str:
    if "Кратко" in text:  return "ST"
    if "Средне" in text:  return "MID"
    return "LT"

# ========================== pivots ==========================

def _last_period_hlc(df: pd.DataFrame, rule: str):
    g = df.resample(rule).agg({"high":"max","low":"min","close":"last"}).dropna()
    if len(g) < 2: return None
    row = g.iloc[-2]
    return float(row["high"]), float(row["low"]), float(row["close"])

def _fib_pivots(H: float, L: float, C: float):
    P = (H + L + C) / 3.0
    d = max(1e-12, H - L)
    R1 = P + 0.382*d; R2 = P + 0.618*d; R3 = P + 1.000*d
    S1 = P - 0.382*d; S2 = P - 0.618*d; S3 = P - 1.000*d
    return {"P":P,"R1":R1,"R2":R2,"R3":R3,"S1":S1,"S2":S2,"S3":S3}

def _pivot_ladder(piv: dict) -> list:
    keys = ["S3","S2","S1","P","R1","R2","R3"]
    return [float(piv[k]) for k in keys if piv.get(k) is not None]

def _classify_band(price: float, piv: dict, buf: float) -> int:
    # -3 <S2, -2 [S2,S1), -1 [S1,P), 0 [P,R1), +1 [R1,R2), +2 [R2,R3), +3 >=R3
    P, R1 = piv["P"], piv["R1"]
    R2, R3, S1, S2 = piv.get("R2"), piv.get("R3"), piv["S1"], piv.get("S2")
    neg_inf, pos_inf = -1e18, 1e18
    levels = [S2 if S2 is not None else neg_inf, S1, P, R1,
              R2 if R2 is not None else pos_inf, R3 if R3 is not None else pos_inf]
    if price < levels[0]-buf: return -3
    if price < levels[1]-buf: return -2
    if price < levels[2]-buf: return -1
    if price < levels[3]-buf: return 0
    if R2 is None or price < levels[4]-buf: return +1
    if price < levels[5]-buf: return +2
    return +3

# ========================== HA & MACD (weekly) ==========================

def _heikin_ashi(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df_ohlc.index, columns=["open","high","low","close"], dtype=float)
    ha["close"] = (df_ohlc["open"]+df_ohlc["high"]+df_ohlc["low"]+df_ohlc["close"]) / 4.0
    ha["open"] = np.nan
    for i in range(len(df_ohlc)):
        if i == 0: ha.iloc[i,0] = float(df_ohlc["open"].iloc[0])
        else:      ha.iloc[i,0] = (ha["open"].iloc[i-1] + ha["close"].iloc[i-1]) / 2.0
    ha["high"] = np.vstack([df_ohlc["high"].values, ha["open"].values, ha["close"].values]).max(axis=0)
    ha["low"]  = np.vstack([df_ohlc["low"].values,  ha["open"].values, ha["close"].values]).min(axis=0)
    return ha

def _ha_green_streak(df_w: pd.DataFrame) -> int:
    ha = _heikin_ashi(df_w)
    up = ha["close"] > ha["open"]
    s = 0
    for i in range(len(up)-1, -1, -1):
        if up.iloc[i]:
            if s < 0: break
            s += 1
        else:
            if s > 0: break
            s -= 1
    return s

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _macd_hist_weekly(df_w: pd.DataFrame) -> pd.Series:
    close = df_w["close"]
    macd_line = _ema(close,12) - _ema(close,26)
    signal = _ema(macd_line,9)
    return macd_line - signal

def _macd_pos_streak(df_w: pd.DataFrame) -> int:
    hist = _macd_hist_weekly(df_w)
    s = 0
    for i in range(len(hist)-1, -1, -1):
        v = hist.iloc[i]
        if v > 0:
            if s < 0: break
            s += 1
        elif v < 0:
            if s > 0: break
            s -= 1
        else:
            break
    return s

def _weekly_trend_score(df_w: pd.DataFrame) -> float:
    slope = _linreg_slope(df_w["close"].tail(16).values) / max(1e-9, float(df_w["close"].iloc[-1]))
    hist = _macd_hist_weekly(df_w).iloc[-1]
    hist_norm = float(hist) / (abs(float(df_w["close"].pct_change().tail(26).std())) + 1e-9)
    x = 0.65*np.tanh(180*slope) + 0.35*np.tanh(hist_norm)
    return float(max(-1.0, min(1.0, x)))

# ========================== targets helpers ==========================

def _three_targets_from_pivots(entry: float, direction: str, piv: dict, step: float):
    ladder = sorted(set(_pivot_ladder(piv)))
    eps = 0.10 * step
    if direction == "BUY":
        ups = [x for x in ladder if x > entry + eps]
        while len(ups) < 3:
            k = len(ups)+1
            ups.append(entry + (0.7 + 0.7*(k-1))*step)
        return ups[0], ups[1], ups[2]
    else:
        dns = [x for x in ladder if x < entry - eps]
        dns = list(sorted(dns, reverse=True))
        while len(dns) < 3:
            k = len(dns)+1
            dns.append(entry - (0.7 + 0.7*(k-1))*step)
        return dns[0], dns[1], dns[2]

def _order_targets(entry: float, tp1: float, tp2: float, tp3: float, action: str, eps: float=1e-6):
    side = 1 if action=="BUY" else -1
    arr = sorted([float(tp1),float(tp2),float(tp3)], key=lambda x: side*(x-entry))
    d0 = side*(arr[0]-entry); d1 = side*(arr[1]-entry); d2 = side*(arr[2]-entry)
    if d1 - d0 < eps:
        arr[1] = entry + side*max(d0 + max(eps,0.1*abs(d0)), d1+eps)
    if side*(arr[2]-entry) - side*(arr[1]-entry) < eps:
        d1 = side*(arr[1]-entry)
        arr[2] = entry + side*max(d1 + max(eps,0.1*abs(d1)), d2+eps)
    return arr[0], arr[1], arr[2]

def _apply_tp_floors(entry: float, sl: float, tp1: float, tp2: float, tp3: float,
                     action: str, hz_tag: str, price: float, atr_val: float):
    if action not in ("BUY","SHORT"): return tp1,tp2,tp3
    risk = abs(entry - sl)
    if risk <= 1e-9: return tp1,tp2,tp3
    side = 1 if action=="BUY" else -1
    min_rr   = {"ST":0.80,"MID":1.00,"LT":1.25}
    min_pct  = {"ST":0.006,"MID":0.012,"LT":0.020}
    atr_mult = {"ST":0.50,"MID":0.80,"LT":1.20}
    floor1 = max(min_rr[hz_tag]*risk, min_pct[hz_tag]*price, atr_mult[hz_tag]*atr_val)
    if abs(tp1-entry) < floor1: tp1 = entry + side*floor1
    floor2 = max(1.6*floor1, (min_rr[hz_tag]*1.8)*risk)
    if abs(tp2-entry) < floor2: tp2 = entry + side*floor2
    min_gap3 = max(0.8*floor1, 0.6*risk)
    if abs(tp3-tp2) < min_gap3: tp3 = tp2 + side*min_gap3
    return tp1,tp2,tp3

# >>>> ЖЁСТКИЕ интуитивные пакеты целей у weekly-крыши/дна
def _pack_short_at_top_weekly(piv_w: dict, price: float, atr_w: float):
    """TP для шорта у крыши в бычьем тренде: TP1 между R2 и R1, TP2≈R1, TP3 между P и R1."""
    R1, R2, P = piv_w["R1"], piv_w.get("R2", None), piv_w["P"]
    if R2 is None: R2 = R1 + 0.8*atr_w  # если R2 нет, синтетика
    tp1 = max(min(price - 0.5*atr_w, R2), R1 - 0.3*atr_w)
    tp2 = R1 - 0.05*atr_w
    tp3 = max(P + 0.25*atr_w, (R1 + P)/2.0)
    return float(tp1), float(tp2), float(tp3)

def _pack_long_at_bottom_weekly(piv_w: dict, price: float, atr_w: float):
    """TP для лонга у дна в медвежьем тренде: TP1 между S2 и S1, TP2≈S1, TP3 между P и S1."""
    S1, S2, P = piv_w["S1"], piv_w.get("S2", None), piv_w["P"]
    if S2 is None: S2 = S1 - 0.8*atr_w
    tp1 = min(max(price + 0.5*atr_w, S2), S1 + 0.3*atr_w)
    tp2 = S1 + 0.05*atr_w
    tp3 = min(P - 0.25*atr_w, (S1 + P)/2.0)
    return float(tp1), float(tp2), float(tp3)

# ========================== main ==========================

def analyze_asset(ticker: str, horizon: str):
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)
    hz = _hz_tag(horizon)

    days = max(90, cfg["look"]*2)
    df = cli.daily_ohlc(ticker, days=days)
    price = cli.last_trade_price(ticker)

    df_w = df.resample(cfg["ha_macd_rule"]).agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()

    tail = df.tail(cfg["look"])
    rng_low  = float(tail["low"].min())
    rng_high = float(tail["high"].max())
    rng_w    = max(1e-9, rng_high - rng_low)
    pos = (price - rng_low) / rng_w

    closes = df["close"]
    slope  = _linreg_slope(closes.tail(cfg["trend"]).values)
    slope_norm = slope / max(1e-9, price)

    atr_d  = float(_atr_like(df, n=cfg["atr"]).iloc[-1])
    atr_w  = _weekly_atr(df) if cfg.get("use_weekly_atr") else atr_d
    vol_ratio = atr_d / max(1e-9, float(_atr_like(df, n=cfg["atr"]*2).iloc[-1]))
    streak_px = _streak(closes)

    last_row = df.iloc[-1]
    body, upper, lower = _wick_profile(last_row)
    long_upper = upper > body*1.3 and upper > lower*1.1
    long_lower = lower > body*1.3 and lower > upper*1.1

    # Пивоты для уровней сделки
    hlc = _last_period_hlc(df, cfg["pivot_rule"])
    if not hlc:
        hlc = (float(df["high"].tail(60).max()), float(df["low"].tail(60).min()), float(df["close"].iloc[-1]))
    piv = _fib_pivots(*hlc)

    # Weekly-пивоты (интуиция/кэппинг)
    hlc_w = _last_period_hlc(df, "W-FRI")
    piv_w = _fib_pivots(*hlc_w) if hlc_w else piv

    buf_hz = 0.25 * (atr_w if hz!="ST" else atr_d)
    buf_w  = 0.25 * atr_w

    band_hz = _classify_band(price, piv,   buf_hz)
    band_w  = _classify_band(price, piv_w, buf_w)

    ha_streak   = _ha_green_streak(df_w)
    macd_streak = _macd_pos_streak(df_w)
    trend_score = _weekly_trend_score(df_w)

    long_green = ha_streak >= {"ST":4,"MID":5,"LT":6}[hz] and macd_streak >= {"ST":4,"MID":6,"LT":8}[hz]
    long_red   = ha_streak <= -{"ST":4,"MID":5,"LT":6}[hz] and macd_streak <= -{"ST":4,"MID":6,"LT":8}[hz]

    # ----- базовый сценарий
    last_o, last_c = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
    last_h = float(df["high"].iloc[-1])
    upper_wick_d = max(0.0, last_h - max(last_o, last_c))
    body_d = abs(last_c - last_o)
    bearish_reject = (last_c < last_o) and (upper_wick_d > body_d)
    very_high_pos = pos >= 0.80

    if very_high_pos:
        if (piv.get("R2") is not None) and (price > piv["R2"] + 0.6*buf_hz) and (slope_norm > 0):
            action, scenario = "BUY", "breakout_up"
        else:
            action, scenario = "SHORT", "fade_top"
    else:
        if band_hz >= +2:
            action, scenario = ("BUY","breakout_up") if ((piv.get("R2") is not None) and (price > piv["R2"] + 0.6*buf_hz) and (slope_norm > 0)) else ("SHORT","fade_top")
        elif band_hz == +1:
            action, scenario = ("WAIT","upper_wait") if (slope_norm > 0.0015 and not bearish_reject and not long_upper) else ("SHORT","fade_top")
        elif band_hz == 0:
            action, scenario = ("BUY","trend_follow") if slope_norm >= 0 else ("WAIT","mid_range")
        elif band_hz == -1:
            action, scenario = ("BUY","revert_from_bottom") if (streak_px <= -3 or long_lower) else ("BUY","trend_follow")
        else:
            action, scenario = ("BUY","revert_from_bottom") if band_hz <= -2 else ("WAIT","upper_wait")

    # ----- weekly-интуиция у крыши/дна
    if long_green and band_w >= +1:
        if (piv_w.get("R2") is not None) and (price <= piv_w["R2"] + 0.6*buf_w):
            action, scenario = "SHORT", "fade_top_confirmed"
        else:
            action, scenario = "WAIT", "overheat_top"
    if long_red and band_w <= -1:
        if (piv_w.get("S2") is not None) and (price >= piv_w["S2"] - 0.6*buf_w):
            action, scenario = "BUY", "rebound_confirmed"
        else:
            action, scenario = "WAIT", "washout_bottom"

    # уверенность
    base = 0.55 + 0.12*_clip01(abs(slope_norm)*1800) + 0.08*_clip01((vol_ratio-0.9)/0.6)
    if action=="WAIT": base -= 0.07
    if band_hz >= +1 and action=="BUY": base -= 0.10
    if band_hz <= -1 and action=="BUY": base += 0.05
    base += 0.05*trend_score
    if (long_green and action=="SHORT") or (long_red and action=="BUY"): base -= 0.04
    conf = float(max(0.55, min(0.90, base)))

    # ----- уровни
    step_d, step_w = atr_d, atr_w
    P, R1 = piv["P"], piv["R1"]

    if action == "BUY":
        if (piv.get("R2") is not None) and (price > piv["R2"] + 0.6*buf_hz):
            base_ref = piv["R2"]; entry = max(price, base_ref + 0.10*step_w); sl = base_ref - 1.00*step_w
        elif price < P:
            entry = max(price, piv["S1"] + 0.15*step_w); sl = piv["S1"] - 0.60*step_w
        else:
            entry = max(price, P + 0.10*step_w);        sl = P - 0.60*step_w
        tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)

    elif action == "SHORT":
        # >>>> confirm-вход у крыши: не опускаем entry ниже текущей
        if price >= R1:
            entry = price
            sl    = R1 + 0.60*step_w
        else:
            entry = price + 0.15*step_d
            sl    = price + 1.00*step_d
        tp1, tp2, tp3 = _three_targets_from_pivots(entry, "SHORT", piv, step_w)
    else:
        entry = price
        sl    = price - 0.90*step_d
        tp1, tp2, tp3 = entry + 0.7*step_d, entry + 1.4*step_d, entry + 2.1*step_d

    # >>>> если шорт у weekly-крыши в бычьем тренде — принудительно ужимаем цели
    if action=="SHORT" and trend_score > 0.45 and band_w >= +1:
        t1,t2,t3 = _pack_short_at_top_weekly(piv_w, price, atr_w)
        tp1,tp2,tp3 = t1,t2,t3

    # >>>> если лонг у weekly-дна в медвежьем — зеркальная логика
    if action=="BUY" and trend_score < -0.45 and band_w <= -1:
        t1,t2,t3 = _pack_long_at_bottom_weekly(piv_w, price, atr_w)
        tp1,tp2,tp3 = t1,t2,t3

    # фиксы
    atr_for_floor = atr_w if hz!="ST" else atr_d
    tp1,tp2,tp3 = _apply_tp_floors(entry, sl, tp1, tp2, tp3, action, hz, price, atr_for_floor)
    tp1,tp2,tp3 = _order_targets(entry, tp1, tp2, tp3, action)

    # вероятности
    p1 = _clip01(0.58 + 0.27*(conf-0.55)/0.35)
    p2 = _clip01(0.44 + 0.21*(conf-0.55)/0.35)
    p3 = _clip01(0.28 + 0.13*(conf-0.55)/0.35)

    # чипсы
    chips = []
    if pos >= 0.8: chips.append("верхний диапазон")
    elif pos >= 0.6: chips.append("под верхним краем")
    elif pos >= 0.4: chips.append("средняя зона")
    elif pos >= 0.2: chips.append("нижняя половина")
    else:            chips.append("ниже опоры")
    if vol_ratio > 1.05: chips.append("волатильность растёт")
    if vol_ratio < 0.95: chips.append("волатильность сжимается")
    if streak_px >= 3: chips.append(f"{streak_px} зелёных подряд")
    if streak_px <= -3: chips.append(f"{abs(streak_px)} красных подряд")
    if _ha_green_streak(df_w) >= 5: chips.append(f"HA {ha_streak} зелёных")
    if _ha_green_streak(df_w) <= -5: chips.append(f"HA {abs(ha_streak)} красных")
    if macd_streak >= 6: chips.append("MACD-hist зелёная серия")
    if macd_streak <= -6: chips.append("MACD-hist красная серия")

    # текст
    seed = int(hashlib.sha1(f"{ticker}{df.index[-1].date()}".encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    if action == "BUY":
        lead = rng.choice(["Опора рядом — беру после отката, без погони.",
                           "Покупатель держит зону — работаю по ходу.",
                           "Спрос живой — ищу реакцию от поддержки."])
        alt = "Если продавят ниже и не вернут — не лезем; ждём возврата и подтверждения."
    elif action == "SHORT":
        lead = rng.choice(["Сверху кромка — играю от отказа.",
                           "У крыши тяжело — готовлю аккуратный шорт.",
                           "Импульс выдыхается — работаю от сопротивления."])
        alt = "Если протолкнут выше и удержат — без погони; ждём возврата и признаков слабости сверху."
    else:
        lead = rng.choice(["Здесь без входа: жду пробой с ретестом.",
                           "Под кромкой — даю цене определиться.",
                           "План от отката к опоре/центру."])
        alt = "Под верхом — не догоняю; работаю на пробое после ретеста или на откате к опоре."

    adds = []
    if (band_w >= +1) and (action != "BUY"):  adds.append("Weekly: под крышей — без погони.")
    if (band_w <= -1) and (action != "SHORT"): adds.append("Weekly: у дна — не шорчу ямы.")
    note_html = f"<div style='margin-top:10px;opacity:.95'>{' '.join([lead]+adds[:2])}</div>"

    return {
        "last_price": float(price),
        "recommendation": {"action": action, "confidence": float(round(conf,4))},
        "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
        "probs": {"tp1": float(p1), "tp2": float(p2), "tp3": float(p3)},
        "context": chips,
        "note_html": note_html,
        "alt": alt,
    }
