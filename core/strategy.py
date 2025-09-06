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

def _wick_profile(row: pd.Series):
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

def _pivot_rule(text: str) -> str:
    hz = _hz_tag(text)
    if hz == "ST":  return "W-FRI"  # прошлая неделя
    if hz == "MID": return "M"      # прошлый месяц
    return "Y"                      # прошлый год

# ---------- pivots ----------
def _last_period_hlc(df: pd.DataFrame, rule: str):
    g = df.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(g) < 2:
        return None
    row = g.iloc[-2]  # последняя ЗАВЕРШЁННАЯ
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

# ---------- Heikin Ashi / MACD / RSI ----------
def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = ha_close.copy()
    if len(df) > 0:
        ha_open.iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2.0
    ha["ha_open"] = ha_open
    ha["ha_close"] = ha_close
    return ha

def _macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_sig
    return hist

def _signed_streak_last(series: pd.Series) -> int:
    """Стрик по знаку на конце ряда (положит. → >0, отрицат. → <0)."""
    s = 0
    for i in range(len(series)-1, 0, -1):
        d = series.iloc[i]
        if d > 0:
            if s < 0: break
            s += 1
        elif d < 0:
            if s > 0: break
            s -= 1
        else:
            break
    return s

def _hist_fade(series: pd.Series, k: int = 3) -> bool:
    """Замедление: по модулю столбики 3 бара подряд уменьшаются."""
    if len(series) < k + 1:
        return False
    a = series.tail(k+1).values
    # сравниваем |h[i]| с |h[i-1]|, последние k шагов убывают
    dec = True
    for i in range(1, len(a)):
        if abs(a[i]) >= abs(a[i-1]) - 1e-12:
            dec = False
            break
    return dec

def _rsi_wilder(close: pd.Series, n: int = 14) -> float:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

# ---------- пост-фильтр целей ----------
def _apply_tp_floors(entry: float, sl: float, tp1: float, tp2: float, tp3: float,
                     action: str, hz_tag: str, price: float, atr_val: float):
    if action not in ("BUY", "SHORT"):
        return tp1, tp2, tp3
    risk = abs(entry - sl)
    if risk <= 1e-9:
        return tp1, tp2, tp3
    side = 1 if action == "BUY" else -1

    min_rr   = {"ST": 0.80, "MID": 1.00, "LT": 1.20}
    min_pct  = {"ST": 0.006, "MID": 0.012, "LT": 0.020}
    atr_mult = {"ST": 0.50, "MID": 0.80, "LT": 1.20}

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
    hz = _hz_tag(horizon)

    # данные
    look = {"ST": 120, "MID": 240, "LT": 480}[hz]
    trend = {"ST": 14, "MID": 28, "LT": 56}[hz]
    atr_n = 14
    days = max(90, look * 2)

    df = cli.daily_ohlc(ticker, days=days)  # index должен быть DatetimeIndex
    price = cli.last_trade_price(ticker)
    if df is None or len(df) < 30:
        raise RuntimeError("Недостаточно данных для анализа")

    # позиция в своём диапазоне
    tail = df.tail(look)
    rng_low  = float(tail["low"].min())
    rng_high = float(tail["high"].max())
    rng_w    = max(1e-9, rng_high - rng_low)
    pos = (price - rng_low) / rng_w  # 0..1

    closes = df["close"]
    slope  = _linreg_slope(closes.tail(trend).values)
    slope_norm = slope / max(1e-9, price)

    atr_d  = float(_atr_like(df, n=atr_n).iloc[-1])
    atr_w  = _weekly_atr(df) if hz != "ST" else atr_d
    vol_ratio = atr_d / max(1e-9, float(_atr_like(df, n=atr_n * 2).iloc[-1]))
    streak_px = _streak(closes)

    last_row = df.iloc[-1]
    body, upper, lower = _wick_profile(last_row)
    long_upper = upper > body * 1.3 and upper > lower * 1.1
    long_lower = lower > body * 1.3 and lower > upper * 1.1

    # Heikin Ashi / MACD / RSI
    ha = _heikin_ashi(df)
    ha_streak = _streak(ha["ha_close"])  # >0 зелёная серия, <0 красная
    hist = _macd_hist(closes)
    macd_streak = _signed_streak_last(hist)
    macd_fade = _hist_fade(hist, k=3)
    rsi_last = _rsi_wilder(closes, n=14)

    # Пороговые серии по горизонту
    HA_SER = {"ST": 4, "MID": 5, "LT": 6}[hz]
    MACD_SER = {"ST": 4, "MID": 6, "LT": 8}[hz]

    # пивоты (W/M/Y по горизонту)
    rule = _pivot_rule(horizon)
    hlc = _last_period_hlc(df, rule)
    if not hlc:
        # fallback — текущий диапазон
        hlc = (float(df["high"].tail(60).max()),
               float(df["low"].tail(60).min()),
               float(df["close"].iloc[-1]))
    H, L, C = hlc
    piv = _fib_pivots(H, L, C)
    P, R1, R2, R3 = piv["P"], piv["R1"], piv.get("R2"), piv.get("R3")
    S1, S2, S3 = piv["S1"], piv.get("S2"), piv.get("S3")

    # зона относительно пивотов
    step_w = atr_w
    buf  = 0.25 * step_w
    band = _classify_band(price, piv, buf)  # -3..+3

    # Композитные сигналы (перегрев/перепроданность)
    overheat_top = (band >= +2) and (ha_streak >= HA_SER) and (macd_streak >= MACD_SER)
    oversold_bot = (band <= -2) and (ha_streak <= -HA_SER) and (macd_streak <= -MACD_SER)

    # ---- первичное решение ----
    if overheat_top:
        # у крыши: базово WAIT; при fade/слабой динамике — SHORT
        if macd_fade or long_upper or slope_norm <= 0:
            action, scenario = "SHORT", "overheat_top"
        else:
            action, scenario = "WAIT", "overheat_wait"
    elif oversold_bot:
        if macd_fade or long_lower or slope_norm >= 0:
            action, scenario = "BUY", "oversold_bottom"
        else:
            action, scenario = "WAIT", "oversold_wait"
    else:
        # прежняя логика
        very_high_pos = pos >= 0.80
        last_o, last_c = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
        last_h = float(df["high"].iloc[-1])
        upper_wick_d = max(0.0, last_h - max(last_o, last_c))
        body_d = abs(last_c - last_o)
        bearish_reject = (last_c < last_o) and (upper_wick_d > body_d)

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
                action, scenario = ("BUY", "revert_from_bottom") if (streak_px <= -3 or long_lower) else ("BUY", "trend_follow")
            else:
                action, scenario = ("BUY", "revert_from_bottom") if band <= -2 else ("WAIT", "upper_wait")

    # уверенность (базовая от правил + бонусы за композитные сигналы)
    base = 0.55 + 0.12 * _clip01(abs(slope_norm) * 1800) + 0.08 * _clip01((vol_ratio - 0.9) / 0.6)
    if action == "WAIT": base -= 0.07
    if band >= +1 and action == "BUY": base -= 0.10
    if band <= -1 and action == "BUY": base += 0.05
    if overheat_top or oversold_bot:
        base += 0.05
        if macd_fade: base += 0.03
    conf = float(max(0.55, min(0.90, base)))

    # ======== AI override (если есть модель — подменяет action/conf) ========
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
            rsi=rsi_last,
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

            action = ai_action
            conf = float(max(0.55, min(0.90, ai_conf)))
    # ======== конец AI override ========

    # уровни
    step_d, step_w = atr_d, atr_w
    if action == "BUY":
        if scenario in ("breakout_up",):
            base_ref = R2 if (R2 is not None) else R1
            entry = max(price, base_ref + 0.10 * step_w)
            sl    = base_ref - 1.00 * step_w
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
            alt = "Если продавят ниже и не вернут — не лезем; ждём возврата сверху и подтверждения."
        elif scenario in ("oversold_bottom",):
            # у S3→S2→P
            near = S2 if (S2 is not None) else S1
            entry = max(price, (near or P) + 0.05 * step_w)
            sl    = (S3 if S3 is not None else (near or P)) - 0.80 * step_w
            tp1   = (S2 if S2 is not None else P) + 0.10 * step_w
            tp2   = P + 0.50 * step_w
            tp3   = (R1 if R1 is not None else (P + 1.0 * step_w))
            alt   = "Если импульс вниз продолжится без реакции — WAIT; возвращаюсь после стабилизации."
        else:
            if price < P:
                entry = max(price, S1 + 0.15 * step_w); sl = S1 - 0.60 * step_w
            else:
                entry = max(price, P + 0.10 * step_w);  sl = P - 0.60 * step_w
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
            alt = "Если продавят ниже и не вернут — не лезем; ждём возврата сверху и подтверждения."
    elif action == "SHORT":
        if scenario in ("overheat_top",):
            # если зашли к R3 — цели R2→P, иначе от R2 — (P↔S1)→S1/S2
            base_ref = None
            if R3 is not None and price >= R3 - 0.2 * step_w:
                base_ref = R3
                entry = min(price, base_ref - 0.05 * step_w)
                sl    = base_ref + 0.80 * step_w
                tp1   = R2 if R2 is not None else R1
                tp2   = P
                tp3   = (S1 if S1 is not None else P - 0.8 * step_w)
            else:
                base_ref = R2 if R2 is not None else R1
                entry = min(price, base_ref - 0.10 * step_w)
                sl    = base_ref + 0.60 * step_w
                mid   = (P + S1)/2.0 if (S1 is not None) else (P - 0.5 * step_w)
                tp1   = mid
                tp2   = S1 if S1 is not None else (P - 1.0 * step_w)
                tp3   = S2 if S2 is not None else (tp2 - 0.8 * step_w)
            alt = "Если протолкнут выше и удержат — без погони; жду возврата и признаков слабости сверху."
        else:
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
        alt = "Под верхом — не догоняю; работаю на пробое после ретеста или на откате к опоре."

    # фиксы: минимальные дистанции и порядок
    atr_for_floor = atr_w if hz != "ST" else atr_d
    tp1, tp2, tp3 = _apply_tp_floors(entry, sl, tp1, tp2, tp3, action, hz, price, atr_for_floor)
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
    if ha_streak >= HA_SER: chips.append("HA длинная зелёная серия")
    if ha_streak <= -HA_SER: chips.append("HA длинная красная серия")
    if macd_streak >= MACD_SER: chips.append("MACD+ серия")
    if macd_streak <= -MACD_SER: chips.append("MACD− серия")
    if macd_fade: chips.append("MACD усталость")
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
    if abs(macd_streak) >= MACD_SER and macd_fade: adds.append("Гистограмма выдыхается — не жадничаю с целью.")
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
