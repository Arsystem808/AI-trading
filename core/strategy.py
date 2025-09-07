# core/strategy.py
import os
import hashlib
import random
import numpy as np
import pandas as pd
from core.polygon_client import PolygonClient

# ========================== helpers: math / series ==========================

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


# ========================== horizons / config ==========================

def _horizon_cfg(text: str):
    """
    ST: пивоты по прошлой неделе, HA/MACD считаем на неделях
    MID: пивоты по прошлому месяцу, HA/MACD на неделях
    LT: пивоты по прошлому году, HA/MACD на неделях
    """
    if "Кратко" in text:
        return dict(
            look=120, trend=14, atr=14,
            pivot_rule="W-FRI",      # last completed week
            ha_macd_rule="W-FRI",    # weekly HA/MACD (визуально как на твоих графиках)
            use_weekly_atr=False
        )
    if "Средне" in text:
        return dict(
            look=240, trend=28, atr=14,
            pivot_rule="M",          # last completed month
            ha_macd_rule="W-FRI",
            use_weekly_atr=True
        )
    # LT
    return dict(
        look=540, trend=56, atr=14,
        pivot_rule="A-DEC",          # last completed calendar year
        ha_macd_rule="W-FRI",
        use_weekly_atr=True
    )

def _hz_tag(text: str) -> str:
    if "Кратко" in text:  return "ST"
    if "Средне" in text:  return "MID"
    return "LT"


# ========================== pivots (Fibonacci) ==========================

def _last_period_hlc(df: pd.DataFrame, rule: str):
    g = df.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(g) < 2:
        return None
    row = g.iloc[-2]  # последняя ЗАВЕРШЁННАЯ периодика
    return float(row["high"]), float(row["low"]), float(row["close"])

def _fib_pivots(H: float, L: float, C: float):
    P = (H + L + C) / 3.0
    d = max(1e-12, H - L)
    R1 = P + 0.382 * d; R2 = P + 0.618 * d; R3 = P + 1.000 * d
    S1 = P - 0.382 * d; S2 = P - 0.618 * d; S3 = P - 1.000 * d
    return {"P": P, "R1": R1, "R2": R2, "R3": R3, "S1": S1, "S2": S2, "S3": S3}

def _pivot_ladder(piv: dict) -> list:
    keys = ["S3", "S2", "S1", "P", "R1", "R2", "R3"]
    return [float(piv[k]) for k in keys if piv.get(k) is not None]

def _classify_band(price: float, piv: dict, buf: float) -> int:
    # -3: <S2, -2: [S2,S1), -1: [S1,P), 0: [P,R1), +1: [R1,R2), +2: [R2,R3), +3: >=R3
    P, R1 = piv["P"], piv["R1"]
    R2, R3, S1, S2 = piv.get("R2"), piv.get("R3"), piv["S1"], piv.get("S2")
    neg_inf, pos_inf = -1e18, 1e18
    levels = [S2 if S2 is not None else neg_inf, S1, P, R1,
              R2 if R2 is not None else pos_inf, R3 if R3 is not None else pos_inf]
    if price < levels[0] - buf: return -3
    if price < levels[1] - buf: return -2
    if price < levels[2] - buf: return -1
    if price < levels[3] - buf: return 0
    if R2 is None or price < levels[4] - buf: return +1
    if price < levels[5] - buf: return +2
    return +3


# ========================== Heikin-Ashi & MACD (weekly) ==========================

def _heikin_ashi(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    """Ожидает ohlc с равномерным шагом (здесь weekly)."""
    ha = pd.DataFrame(index=df_ohlc.index, columns=["open","high","low","close"], dtype=float)
    ha["close"] = (df_ohlc["open"] + df_ohlc["high"] + df_ohlc["low"] + df_ohlc["close"]) / 4.0
    ha["open"]  = np.nan
    for i in range(len(df_ohlc)):
        if i == 0:
            ha.iloc[i, ha.columns.get_loc("open")] = float(df_ohlc["open"].iloc[0])
        else:
            ha.iloc[i, ha.columns.get_loc("open")] = (ha["open"].iloc[i-1] + ha["close"].iloc[i-1]) / 2.0
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
    macd_line = _ema(close, 12) - _ema(close, 26)
    signal = _ema(macd_line, 9)
    hist = macd_line - signal
    return hist

def _macd_pos_streak(df_w: pd.DataFrame) -> int:
    hist = _macd_hist_weekly(df_w)
    s = 0
    for i in range(len(hist)-1, -1, -1):
        if hist.iloc[i] > 0:
            if s < 0: break
            s += 1
        elif hist.iloc[i] < 0:
            if s > 0: break
            s -= 1
        else:
            break
    return s

def _weekly_trend_score(df_w: pd.DataFrame) -> float:
    """-1..+1. Смешиваем наклон и MACD-гистограмму."""
    slope = _linreg_slope(df_w["close"].tail(16).values) / max(1e-9, float(df_w["close"].iloc[-1]))
    hist = _macd_hist_weekly(df_w).iloc[-1]
    hist_norm = float(hist) / (abs(float(df_w["close"].pct_change().tail(26).std())) + 1e-9)
    x = 0.65 * np.tanh(180*slope) + 0.35 * np.tanh(hist_norm)
    return float(max(-1.0, min(1.0, x)))


# ========================== targets helpers ==========================

def _three_targets_from_pivots(entry: float, direction: str, piv: dict, step: float):
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

def _order_targets(entry: float, tp1: float, tp2: float, tp3: float, action: str, eps: float = 1e-6):
    side = 1 if action == "BUY" else -1
    arr = sorted([float(tp1), float(tp2), float(tp3)], key=lambda x: side * (x - entry))
    d0 = side * (arr[0] - entry)
    d1 = side * (arr[1] - entry)
    d2 = side * (arr[2] - entry)
    if d1 - d0 < eps:
        arr[1] = entry + side * max(d0 + max(eps, 0.1*abs(d0)), d1 + eps)
    if side * (arr[2] - entry) - side * (arr[1] - entry) < eps:
        d1 = side * (arr[1] - entry)
        arr[2] = entry + side * max(d1 + max(eps, 0.1*abs(d1)), d2 + eps)
    return arr[0], arr[1], arr[2]

def _apply_tp_floors(entry: float, sl: float, tp1: float, tp2: float, tp3: float,
                     action: str, hz_tag: str, price: float, atr_val: float):
    if action not in ("BUY", "SHORT"):
        return tp1, tp2, tp3
    risk = abs(entry - sl)
    if risk <= 1e-9:
        return tp1, tp2, tp3
    side = 1 if action == "BUY" else -1
    min_rr   = {"ST": 0.80, "MID": 1.00, "LT": 1.25}
    min_pct  = {"ST": 0.006, "MID": 0.012, "LT": 0.020}
    atr_mult = {"ST": 0.50, "MID": 0.80, "LT": 1.20}
    floor1 = max(min_rr[hz_tag]*risk, min_pct[hz_tag]*price, atr_mult[hz_tag]*atr_val)
    if abs(tp1 - entry) < floor1:
        tp1 = entry + side * floor1
    floor2 = max(1.6*floor1, (min_rr[hz_tag]*1.8)*risk)
    if abs(tp2 - entry) < floor2:
        tp2 = entry + side * floor2
    min_gap3 = max(0.8*floor1, 0.6*risk)
    if abs(tp3 - tp2) < min_gap3:
        tp3 = tp2 + side * min_gap3
    return tp1, tp2, tp3

def _cap_targets_by_trend(tp1, tp2, tp3, *, action: str, piv: dict, step: float,
                          trend_score: float, band: int):
    """
    «Интуитивная защита» от нереалистичных целей против сильного тренда.
    - Сильный ап-тренд (score>0.6): для SHORT не опускаем TP3 ниже P, TP2 ~ R1, TP1 между R2 и R1.
    - Сильный даун-тренд (score<-0.6): для BUY не задираем TP3 выше P, TP2 ~ S1, TP1 между S1 и P.
    """
    P, R1, R2, S1, S2 = piv["P"], piv["R1"], piv.get("R2"), piv["S1"], piv.get("S2")

    if action == "SHORT" and trend_score > 0.6 and band >= +1:
        # потолок сверху, бычья нога — цели ближе
        # TP1: в коридоре [R1-δ, R2] (если доступно)
        if R2 is not None:
            tp1 = max(min(tp1, R2), R1 - 0.3*step)
        else:
            tp1 = max(tp1, R1 - 0.3*step)
        # TP2: удерживаем вокруг R1
        tp2 = max(min(tp2, R1 - 0.1*step), R1 - 0.6*step)
        # TP3: не ниже P
        tp3 = max(tp3, P + 0.2*step)

    if action == "BUY" and trend_score < -0.6 and band <= -1:
        # дно снизу, медвежья нога — цели ближе
        # TP1: в коридоре [S1, P+δ]
        tp1 = min(max(tp1, S1 + 0.3*step), P + 0.3*step)
        # TP2: вокруг S1
        tp2 = min(max(tp2, S1 + 0.1*step), S1 + 0.8*step)
        # TP3: не выше P
        tp3 = min(tp3, P - 0.2*step)

    return tp1, tp2, tp3


# ========================== main ==========================

def analyze_asset(ticker: str, horizon: str):
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)
    hz = _hz_tag(horizon)

    # ---- данные (дневные) ----
    days = max(90, cfg["look"] * 2)
    df = cli.daily_ohlc(ticker, days=days)  # DatetimeIndex
    price = cli.last_trade_price(ticker)

    # недельная агрегация (для HA/MACD)
    df_w = df.resample(cfg["ha_macd_rule"]).agg(
        {"open":"first", "high":"max", "low":"min", "close":"last"}
    ).dropna()

    # позиция в своём диапазоне (по дневным)
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
    streak_px = _streak(closes)

    # свечной профиль последнего дня
    last_row = df.iloc[-1]
    body, upper, lower = _wick_profile(last_row)
    long_upper = upper > body * 1.3 and upper > lower * 1.1
    long_lower = lower > body * 1.3 and lower > upper * 1.1

    # пивоты по выбранному горизонту (последний завершённый период)
    hlc = _last_period_hlc(df, cfg["pivot_rule"])
    if not hlc:
        hlc = (float(df["high"].tail(60).max()),
               float(df["low"].tail(60).min()),
               float(df["close"].iloc[-1]))
    H, L, C = hlc
    piv = _fib_pivots(H, L, C)
    P, R1, R2 = piv["P"], piv["R1"], piv.get("R2")

    buf  = 0.25 * atr_w
    band = _classify_band(price, piv, buf)  # -3..+3

    # недельные индикаторы (интуиция)
    ha_streak   = _ha_green_streak(df_w)   # >0 зелёная серия, <0 красная
    macd_streak = _macd_pos_streak(df_w)   # >0 положительная серия, <0 отрицательная
    trend_score = _weekly_trend_score(df_w)  # -1..+1

    long_green = ha_streak >= { "ST":4, "MID":5, "LT":6 }[hz] and macd_streak >= { "ST":4, "MID":6, "LT":8 }[hz]
    long_red   = ha_streak <= -{ "ST":4, "MID":5, "LT":6 }[hz] and macd_streak <= -{ "ST":4, "MID":6, "LT":8 }[hz]

    # ---- сценарии (человекоподобно) ----
    last_o, last_c = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
    last_h = float(df["high"].iloc[-1])
    upper_wick_d = max(0.0, last_h - max(last_o, last_c))
    body_d = abs(last_c - last_o)
    bearish_reject = (last_c < last_o) and (upper_wick_d > body_d)
    very_high_pos = pos >= 0.80

    # Базовое решение
    if very_high_pos:
        if (R2 is not None) and (price > R2 + 0.6*buf) and (slope_norm > 0):
            action, scenario = "BUY", "breakout_up"
        else:
            action, scenario = "SHORT", "fade_top"
    else:
        if band >= +2:
            action, scenario = ("BUY", "breakout_up") if ((R2 is not None) and (price > R2 + 0.6*buf) and (slope_norm > 0)) else ("SHORT", "fade_top")
        elif band == +1:
            action, scenario = ("WAIT", "upper_wait") if (slope_norm > 0.0015 and not bearish_reject and not long_upper) else ("SHORT", "fade_top")
        elif band == 0:
            action, scenario = ("BUY", "trend_follow") if slope_norm >= 0 else ("WAIT", "mid_range")
        elif band == -1:
            action, scenario = ("BUY", "revert_from_bottom") if (streak_px <= -3 or long_lower) else ("BUY", "trend_follow")
        else:
            action, scenario = ("BUY", "revert_from_bottom") if band <= -2 else ("WAIT", "upper_wait")

    # Интуитивные поправки (HA+MACD)
    if long_green and band >= +1 and trend_score > 0.4:
        # у крыши на сильной зелёной серии — не покупаем вершину
        if action == "BUY":
            action, scenario = "WAIT", "overheat_top"
        elif action == "SHORT":
            scenario = "fade_top_confirmed"

    if long_red and band <= -1 and trend_score < -0.4:
        # у дна на длинной красной — не шортим дно
        if action == "SHORT":
            action, scenario = "WAIT", "washout_bottom"
        elif action == "BUY":
            scenario = "rebound_confirmed"

    # уверенность
    base = 0.55 + 0.12 * _clip01(abs(slope_norm) * 1800) + 0.08 * _clip01((vol_ratio - 0.9) / 0.6)
    if action == "WAIT": base -= 0.07
    if band >= +1 and action == "BUY": base -= 0.10
    if band <= -1 and action == "BUY": base += 0.05
    # усиливаем за счёт тренда/серий
    base += 0.05 * trend_score
    if (long_green and action=="SHORT") or (long_red and action=="BUY"):
        base -= 0.04  # конфликт с интуицией — уверенность ниже
    conf = float(max(0.55, min(0.90, base)))

    # ======== AI override (если загружена модель) ========
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
            streak=float(streak_px),
            band=float(band),
            long_upper=bool(long_upper),
            long_lower=bool(long_lower),
            ha_streak=float(ha_streak),
            macd_streak=float(macd_streak),
            trend_score=float(trend_score),
        )
        hz_tag = _hz_tag(horizon)
        out_ai = score_signal(feats, hz=hz_tag)
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
            action = ai_action
            conf = float(max(0.55, min(0.90, ai_conf)))
    # ======== конец override ========

    # уровни (для WAIT UI всё равно покажет планово)
    step_d, step_w = atr_d, atr_w
    if action == "BUY":
        if (R2 is not None) and scenario == "breakout_up":
            base_ref = R2
            entry = max(price, base_ref + 0.10*step_w)
            sl    = base_ref - 1.00*step_w
        elif price < P:
            entry = max(price, piv["S1"] + 0.15*step_w); sl = piv["S1"] - 0.60*step_w
        else:
            entry = max(price, P + 0.10*step_w);        sl = P - 0.60*step_w
        tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
        alt = "Если продавят ниже и не вернут — не лезем; ждём возврата сверху и подтверждения."
    elif action == "SHORT":
        if price >= R1:
            entry = min(price, R1 - 0.15*step_w)
            sl    = R1 + 0.60*step_w
        else:
            entry = price + 0.15*step_d
            sl    = price + 1.00*step_d
        tp1, tp2, tp3 = _three_targets_from_pivots(entry, "SHORT", piv, step_w)
        alt = "Если протолкнут выше и удержат — без погони; ждём возврата и признаков слабости сверху."
    else:  # WAIT — дадим ориентиры для наблюдения
        entry = price
        sl    = price - 0.90*step_d
        tp1, tp2, tp3 = entry + 0.7*step_d, entry + 1.4*step_d, entry + 2.1*step_d
        alt = "Под верхом — не догоняю; работаю на пробое после ретеста или на откате к опоре."

    # ФИКСЫ: минимальные дистанции и порядок
    atr_for_floor = atr_w if hz != "ST" else atr_d
    tp1, tp2, tp3 = _apply_tp_floors(entry, sl, tp1, tp2, tp3, action, hz, price, atr_for_floor)
    # тренд-защита от слишком дальних TP против тренда
    tp1, tp2, tp3 = _cap_targets_by_trend(tp1, tp2, tp3, action=action, piv=piv,
                                          step=step_w, trend_score=trend_score, band=band)
    tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)

    # вероятности достижения целей
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
    if streak_px >= 3: chips.append(f"{streak_px} зелёных подряд")
    if streak_px <= -3: chips.append(f"{abs(streak_px)} красных подряд")
    if long_upper: chips.append("тени сверху")
    if long_lower: chips.append("тени снизу")
    if ha_streak >= 5: chips.append(f"HA {ha_streak} зелёных")
    if ha_streak <= -5: chips.append(f"HA {abs(ha_streak)} красных")
    if macd_streak >= 6: chips.append("MACD-hist зелёная серия")
    if macd_streak <= -6: chips.append("MACD-hist красная серия")

    # «живой» текст
    seed = int(hashlib.sha1(f"{ticker}{df.index[-1].date()}".encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    if action == "BUY":
        lead = rng.choice([
            "Опора рядом — беру после отката, без погони.",
            "Покупатель держит зону — работаю по ходу.",
            "Спрос живой — ищу реакцию от поддержки.",
        ])
    elif action == "SHORT":
        lead = rng.choice([
            "Сверху кромка — играю от отказа.",
            "У крыши тяжело — готовлю аккуратный шорт.",
            "Импульс выдыхается — работаю от сопротивления.",
        ])
    else:
        lead = rng.choice([
            "Здесь без входа: жду пробой с ретестом.",
            "Под кромкой — даю цене определиться.",
            "План от отката к опоре/центру.",
        ])

    adds = []
    if (band >= +1) and (action != "BUY"): adds.append("Под крышей — без погони.")
    if (band <= -1) and (action != "SHORT"): adds.append("У дна — не шорчу ямы.")
    if long_green and action == "SHORT": adds.append("Длинная зелёная серия — цели ближе.")
    if long_red   and action == "BUY":   adds.append("Длинная красная — беру осторожно.")

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
