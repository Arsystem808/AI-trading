# core/strategy.py
import os
import hashlib
import random
import numpy as np
import pandas as pd
from core.polygon_client import PolygonClient

# ---------- базовые helpers ----------
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

# ---------- HA / MACD / RSI ----------
def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index)
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = np.zeros(len(df))
    if len(df) > 0:
        ha_open[0] = (float(df["open"].iloc[0]) + float(df["close"].iloc[0])) / 2.0
        for i in range(1, len(df)):
            ha_open[i] = (ha_open[i-1] + float(ha["ha_close"].iloc[i-1])) / 2.0
    ha["ha_open"] = ha_open
    ha["ha_high"] = pd.concat([ha["ha_open"], ha["ha_close"], df["high"]], axis=1).max(axis=1)
    ha["ha_low"]  = pd.concat([ha["ha_open"], ha["ha_close"], df["low"]],  axis=1).min(axis=1)
    return ha

def _ha_color_series_len(ha: pd.DataFrame) -> int:
    if len(ha) < 2: return 0
    color = np.where(ha["ha_close"] >= ha["ha_open"], 1, -1)
    sgn = int(color[-1])
    cnt = 0
    for v in color[::-1]:
        if v != sgn: break
        cnt += 1
    return cnt if sgn > 0 else -cnt

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd = _ema(close, fast) - _ema(close, slow)
    sig  = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def _streak_sign(s: pd.Series) -> int:
    s = s.dropna()
    if len(s) == 0: return 0
    last = np.sign(float(s.iloc[-1]))
    if last == 0: return 0
    cnt = 0
    for v in s.values[::-1]:
        sv = np.sign(v)
        if sv != last or sv == 0: break
        cnt += 1
    return cnt if last > 0 else -cnt

def _macd_slowdown(hist: pd.Series, look: int = 4) -> tuple[bool, bool]:
    h = hist.dropna().tail(look)
    if len(h) < look: return False, False
    dec_pos = bool((h > 0).all() and all(h.iloc[i] <= h.iloc[i-1] for i in range(1, len(h))))
    dec_neg = bool((h < 0).all() and all(abs(h.iloc[i]) <= abs(h.iloc[i-1]) for i in range(1, len(h))))
    return dec_pos, dec_neg

def _rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up  = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

# ---------- горизонты ----------
def _horizon_cfg(text: str):
    if "Кратко" in text:
        return dict(
            look=60, trend=14, atr=14,
            pivot_period="W-FRI", use_weekly_atr=False,
            tol_pct=0.008, ha_long=4, macd_long=4
        )
    if "Средне" in text:
        return dict(
            look=120, trend=28, atr=14,
            pivot_period="M", use_weekly_atr=True,
            tol_pct=0.010, ha_long=5, macd_long=6
        )
    return dict(
        look=240, trend=56, atr=14,
        pivot_period="A-DEC", use_weekly_atr=True,
        tol_pct=0.012, ha_long=6, macd_long=8
    )

def _hz_tag(text: str) -> str:
    if "Кратко" in text:  return "ST"
    if "Средне" in text:  return "MID"
    return "LT"

# ---------- pivot helpers ----------
def _last_period_hlc(df: pd.DataFrame, rule: str):
    g = df.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(g) < 2: return None
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
    P, R1 = piv["P"], piv["R1"]
    R2, R3, S1, S2 = piv.get("R2"), piv.get("R3"), piv["S1"], piv.get("S2")
    neg_inf, pos_inf = -1e18, 1e18
    levels = [S2 if S2 is not None else neg_inf, S1, P, R1, R2 if R2 is not None else pos_inf, R3 if R3 is not None else pos_inf]
    if price < levels[0] - buf: return -3
    if price < levels[1] - buf: return -2
    if price < levels[2] - buf: return -1
    if price < levels[3] - buf: return 0
    if R2 is None or price < levels[4] - buf: return +1
    if price < levels[5] - buf: return +2
    return +3

# ---------- пост-фильтры TP ----------
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
    d0 = side * (arr[0] - entry); d1 = side * (arr[1] - entry); d2 = side * (arr[2] - entry)
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
    df = cli.daily_ohlc(ticker, days=days)
    price = cli.last_trade_price(ticker)

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
    vol_ratio = atr_d / max(1e-9, float(_atr_like(df, n=cfg["atr"] * 2).iloc[-1]))
    streak    = _streak(closes)

    last_row = df.iloc[-1]
    body, upper, lower = _wick_profile(last_row)
    long_upper = upper > body * 1.3 and upper > lower * 1.1
    long_lower = lower > body * 1.3 and lower > upper * 1.1

    # --- Heikin Ashi / MACD / RSI ---
    ha = _heikin_ashi(df)
    ha_series = _ha_color_series_len(ha)
    macd_line, macd_sig, macd_hist = _macd(closes)
    macd_streak = _streak_sign(macd_hist)
    macd_dec_pos, macd_dec_neg = _macd_slowdown(macd_hist, look=4)
    rsi = _rsi_wilder(closes, 14); rsi_last = float(rsi.iloc[-1])

    # --- пивоты по горизонту ---
    hlc = _last_period_hlc(df, cfg["pivot_period"])
    if not hlc:
        hlc = (float(df["high"].tail(60).max()), float(df["low"].tail(60).min()), float(df["close"].iloc[-1]))
    H, L, C = hlc
    piv = _fib_pivots(H, L, C)
    P, R1, R2, R3, S1, S2, S3 = piv["P"], piv["R1"], piv["R2"], piv["R3"], piv["S1"], piv["S2"], piv["S3"]

    tol = max(cfg["tol_pct"] * price, 0.20 * atr_w)
    near_R1 = abs(price - R1) <= tol
    near_R2 = abs(price - R2) <= tol
    near_R3 = abs(price - R3) <= tol
    near_S1 = abs(price - S1) <= tol
    near_S2 = abs(price - S2) <= tol
    near_S3 = abs(price - S3) <= tol

    buf  = 0.25 * atr_w
    band = _classify_band(price, piv, buf)  # -3..+3

    # --- Weekly подтверждение ---
    w = df.resample("W-FRI").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    if len(w) >= 6:
        w_ha = _heikin_ashi(w)
        w_ha_series = _ha_color_series_len(w_ha)
        w_macd_hist = _macd(w["close"])[2]
        w_macd_streak = _streak_sign(w_macd_hist)
    else:
        w_ha_series = 0; w_macd_streak = 0

    # ---- базовая rule-логика (как было) ----
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
            action, scenario = ("BUY", "revert_from_bottom") if (_streak(closes) <= -3 or long_lower) else ("BUY", "trend_follow")
        else:
            action, scenario = ("BUY", "revert_from_bottom") if band <= -2 else ("WAIT", "upper_wait")

    # ---- интуитивные паттерны на пивотах (добавлено) ----
    # Overheat у крыши R2/R3
    overheat_top = (
        ((price >= R2 - tol) or near_R3)
        and (ha_series >= cfg["ha_long"])
        and ((macd_streak >= cfg["macd_long"]) or macd_dec_pos)
    )
    # Oversold у дна S2/S3
    oversold_bottom = (
        ((price <= S2 + tol) or near_S3)
        and (ha_series <= -cfg["ha_long"])
        and ((macd_streak <= -cfg["macd_long"]) or macd_dec_neg)
    )
    # Растяжение у R1 / S1
    r1_stretch = (
        near_R1
        and (ha_series >= cfg["ha_long"])
        and ((macd_streak >= cfg["macd_long"]) or macd_dec_pos or w_macd_streak >= 2)
    )
    s1_stretch = (
        near_S1
        and (ha_series <= -cfg["ha_long"])
        and ((macd_streak <= -cfg["macd_long"]) or macd_dec_neg or w_macd_streak <= -2)
    )

    # Приоритет: край диапазона важнее R1/S1
    if overheat_top:
        action   = "SHORT" if (macd_dec_pos or long_upper or rsi_last >= 63) else "WAIT"
        scenario = "overheat_top"
    elif oversold_bottom:
        action   = "BUY" if (macd_dec_neg or long_lower or rsi_last <= 37) else "WAIT"
        scenario = "oversold_bottom"
    elif r1_stretch:
        action   = "SHORT" if (macd_dec_pos or long_upper) else "WAIT"
        scenario = "fade_r1"
    elif s1_stretch:
        action   = "BUY" if (macd_dec_neg or long_lower) else "WAIT"
        scenario = "fade_s1"

    # уверенность
    base = 0.55 + 0.12 * _clip01(abs(slope_norm) * 1800) + 0.08 * _clip01((vol_ratio - 0.9) / 0.6)
    if action == "WAIT": base -= 0.07
    if band >= +1 and action == "BUY": base -= 0.10
    if band <= -1 and action == "BUY": base += 0.05
    # усиление за счёт паттернов и Weekly
    if overheat_top or oversold_bottom: base += 0.10
    if r1_stretch or s1_stretch: base += 0.05
    if abs(w_ha_series) >= 2 or abs(w_macd_streak) >= 2: base += 0.03
    conf = float(max(0.55, min(0.90, base)))

    # ======== AI override (если есть ML-модель) ========
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
    # ======== конец AI override ========

    # уровни
    step_d, step_w = atr_d, atr_w
    if action == "BUY":
        if scenario == "overheat_top":
            # Обычно WAIT, но если BUY — защитный режим
            entry = max(price, P + 0.10 * step_w); sl = P - 0.80 * step_w
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
            alt = "Рынок перегрет у крыши — не догоняю; работаю от возврата к опоре."
        elif scenario == "oversold_bottom":
            base_ref = S2 if near_S3 else S1
            entry = max(price, base_ref + 0.10 * step_w)
            sl    = base_ref - 0.80 * step_w
            tp1, tp2, tp3 = S2, P, R1
            alt = "Если продавливают ниже и не возвращают — без входа до признаков разворота."
        elif scenario == "fade_s1":
            mid = (S1 + P) / 2.0
            entry = max(price, S1 + 0.05 * step_w)
            sl    = S1 - 0.60 * step_w
            tp1, tp2, tp3 = mid, P, R1
            alt = "Если слабость — жду ретест S1/P и реакцию."
        elif price < P:
            entry = max(price, piv["S1"] + 0.15 * step_w); sl = piv["S1"] - 0.60 * step_w
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
            alt = "Вход от опоры — после реакции."
        else:
            entry = max(price, P + 0.10 * step_w); sl = P - 0.60 * step_w
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
            alt = "По тренду — не догоняю, беру после паузы."
    elif action == "SHORT":
        if scenario == "overheat_top":
            if near_R3:
                entry = min(price, R3 - 0.05 * step_w)
                sl    = R3 + 0.60 * step_w
                tp1, tp2, tp3 = R2, P, S1
            else:
                entry = min(price, R2 - 0.05 * step_w)
                sl    = R2 + 0.60 * step_w
                mid   = (P + S1) / 2.0
                tp1, tp2, tp3 = mid, S1, S2
            alt = "Если удержат выше — не гонюсь; жду отказа и возврата под уровень."
        elif scenario == "fade_r1":
            entry = min(price, R1 - 0.05 * step_w)
            sl    = R1 + 0.60 * step_w
            mid   = (P + R1) / 2.0
            tp1, tp2, tp3 = mid, P, (S1 if S1 is not None else P - 0.8 * step_w)
            alt = "Если выше R1 закрепятся — без погони, жду слабость сверху."
        else:
            if price >= R1:
                entry = min(price, R1 - 0.15 * step_w)
                sl    = R1 + 0.60 * step_w
            else:
                entry = price + 0.15 * step_d
                sl    = price + 1.00 * step_d
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "SHORT", piv, step_w)
            alt = "Работа от кромки — без погони, только по отказу."
    else:  # WAIT
        entry = price
        sl    = price - 0.90 * step_d
        tp1, tp2, tp3 = entry + 0.7 * step_d, entry + 1.4 * step_d, entry + 2.1 * step_d
        alt = "Жду пробой/ретест или откат к опоре."

    # фиксы и вероятности
    atr_for_floor = atr_w if hz != "ST" else atr_d
    tp1, tp2, tp3 = _apply_tp_floors(entry, sl, tp1, tp2, tp3, action, hz, price, atr_for_floor)
    tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)

    p1 = _clip01(0.58 + 0.27 * (conf - 0.55) / 0.35)
    p2 = _clip01(0.44 + 0.21 * (conf - 0.55) / 0.35)
    p3 = _clip01(0.28 + 0.13 * (conf - 0.55) / 0.35)

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
    if overheat_top: chips.append("overheat")
    if oversold_bottom: chips.append("oversold")
    if r1_stretch: chips.append("R1 stretch")
    if s1_stretch: chips.append("S1 stretch")
    if abs(w_ha_series) >= 2 or abs(w_macd_streak) >= 2: chips.append("weekly confirm")

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
