# core/strategy.py
import os
import hashlib
import random
import numpy as np
import pandas as pd
from core.polygon_client import PolygonClient

# ----------------- utils -----------------
def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

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

# ----------------- ATR -----------------
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

# ----------------- HA & MACD -----------------
def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index.copy())
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2.0]
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + ha["ha_close"].iloc[i-1]) / 2.0)
    ha["ha_open"] = pd.Series(ha_open, index=df.index)
    return ha

def _streak_by_sign(series: pd.Series, positive: bool=True) -> int:
    sgn = 1 if positive else -1
    run = 0
    vals = series.values
    for i in range(len(vals)-1, -1, -1):
        v = vals[i]
        if (v > 0 and sgn == 1) or (v < 0 and sgn == -1):
            run += 1
        elif v == 0:
            continue
        else:
            break
    return run

def _macd_hist(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

# ----------------- horizons & pivots -----------------
def _horizon_cfg(text: str):
    # ST — weekly pivots; MID/LT — monthly pivots (как и просил)
    if "Кратко" in text:
        return dict(look=60, trend=14, atr=14, pivot_rule="W-FRI", use_weekly_atr=False, hz="ST")
    if "Средне" in text:
        return dict(look=120, trend=28, atr=14, pivot_rule="M", use_weekly_atr=True, hz="MID")
    return dict(look=240, trend=56, atr=14, pivot_rule="M", use_weekly_atr=True, hz="LT")

def _last_period_hlc(df: pd.DataFrame, rule: str):
    g = df.resample(rule).agg({"high":"max","low":"min","close":"last"}).dropna()
    if len(g) < 2:
        return None
    row = g.iloc[-2]  # завершённый период
    return float(row["high"]), float(row["low"]), float(row["close"])

def _fib_pivots(H: float, L: float, C: float):
    P = (H + L + C) / 3.0
    d = H - L
    R1 = P + 0.382 * d; R2 = P + 0.618 * d; R3 = P + 1.000 * d
    S1 = P - 0.382 * d; S2 = P - 0.618 * d; S3 = P - 1.000 * d
    return {"P":P,"R1":R1,"R2":R2,"R3":R3,"S1":S1,"S2":S2,"S3":S3}

def _classify_band(price: float, piv: dict, buf: float) -> int:
    # -3: <S2, -2:[S2,S1), -1:[S1,P), 0:[P,R1), +1:[R1,R2), +2:[R2,R3), +3:>=R3
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

# ----------------- guards & clamps -----------------
def _apply_tp_floors(entry: float, sl: float, tp1: float, tp2: float, tp3: float,
                     action: str, hz: str, price: float, atr_val: float):
    """Минимальная дистанция до TP и монотонность целей."""
    if action not in ("BUY", "SHORT"):
        return tp1, tp2, tp3
    risk = abs(entry - sl)
    if risk <= 1e-9:
        return tp1, tp2, tp3
    side = 1 if action == "BUY" else -1
    min_rr   = {"ST":0.80,"MID":1.00,"LT":1.25}
    min_pct  = {"ST":0.006,"MID":0.012,"LT":0.020}
    atr_mult = {"ST":0.50,"MID":0.80,"LT":1.20}
    floor1 = max(min_rr[hz]*risk, min_pct[hz]*price, atr_mult[hz]*atr_val)
    if abs(tp1 - entry) < floor1:
        tp1 = entry + side * floor1
    floor2 = max(1.6*floor1, (min_rr[hz]*1.8)*risk)
    if abs(tp2 - entry) < floor2:
        tp2 = entry + side * floor2
    min_gap3 = max(0.8*floor1, 0.6*risk)
    if abs(tp3 - tp2) < min_gap3:
        tp3 = tp2 + side * min_gap3
    return tp1, tp2, tp3

def _clamp_tp_span(entry: float, tps: tuple, action: str, hz: str, price: float, atr_w: float):
    """Ограничивает дальность TP от entry (фикс «улетающих» TP3 на LT/крипте)."""
    max_pct = {"ST":0.06, "MID":0.10, "LT":0.16}[hz]   # не дальше X% от entry
    max_atr = {"ST":2.5,  "MID":3.2,  "LT":3.8}[hz]    # и не дальше N×ATR (weekly для MID/LT)
    lim = min(max_pct*price, max_atr*atr_w)
    side = 1 if action == "BUY" else -1
    out = []
    for tp in tps:
        dist = side * (tp - entry)
        if dist > lim:
            tp = entry + side * lim
        out.append(tp)
    return tuple(out)

def _cap_sl(entry: float, sl: float, action: str, hz: str, price: float, atr_w: float):
    """Ограничивает максимально допустимый риск."""
    max_pct = {"ST":0.015,"MID":0.030,"LT":0.050}[hz]
    max_atr = {"ST":1.2, "MID":1.6, "LT":2.2}[hz]
    lim = min(max_pct*price, max_atr*atr_w)
    if action == "BUY":
        sl = max(entry - lim, sl)
    elif action == "SHORT":
        sl = min(entry + lim, sl)
    return sl

# ----------------- pivot-aware targets -----------------
def _between(a: float, b: float) -> float:
    return (a + b) / 2.0

def _targets_pivot_aware(entry: float, action: str, piv: dict, band: int,
                         hz: str, atr_w: float, atr_d: float, price: float):
    """
    Цели «с пониманием лестницы»:
    - SHORT в верхних зонах: TP1 ~ R1, TP2 ~ P, TP3 ~ между P и S1 (не ниже S1).
    - LONG у дна:            TP1 ~ S1, TP2 ~ P, TP3 ~ между P и R1 (не выше R1).
    В середине — мягче, по ATR.
    """
    P, R1, R2, S1, S2 = piv["P"], piv["R1"], piv.get("R2"), piv["S1"], piv.get("S2")
    step = atr_w if hz != "ST" else atr_d

    if action == "SHORT":
        if band >= +2 and R2 is not None:
            tp1 = max(_between(R2, R1), entry - 0.9*step)
            tp2 = max(R1,            entry - 1.6*step)
            tp3 = max(_between(P, R1), entry - 2.2*step)
        elif band == +1:
            tp1 = max(R1,            entry - 0.9*step)
            tp2 = max(P,             entry - 1.6*step)
            tp3 = max(_between(P, S1), entry - 2.2*step)
        elif band == 0:
            tp1 = entry - 0.9*step; tp2 = entry - 1.6*step; tp3 = entry - 2.2*step
        else:  # ниже середины — не агрессивничаем
            tp1 = entry - 0.8*step; tp2 = entry - 1.3*step; tp3 = entry - 1.8*step
    else:  # BUY
        if band <= -2 and S2 is not None:
            tp1 = min(_between(S2, S1), entry + 0.9*step)
            tp2 = min(S1,             entry + 1.6*step)
            tp3 = min(_between(P, S1), entry + 2.2*step)
        elif band == -1:
            tp1 = min(S1,             entry + 0.9*step)
            tp2 = min(P,              entry + 1.6*step)
            tp3 = min(_between(P, R1), entry + 2.2*step)
        elif band == 0:
            tp1 = entry + 0.9*step; tp2 = entry + 1.6*step; tp3 = entry + 2.2*step
        else:  # выше середины — аккуратно
            tp1 = entry + 0.8*step; tp2 = entry + 1.3*step; tp3 = entry + 1.8*step

    return tp1, tp2, tp3

# ----------------- main -----------------
def analyze_asset(ticker: str, horizon: str):
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)
    hz = cfg["hz"]

    days = max(90, cfg["look"] * 2)
    df = cli.daily_ohlc(ticker, days=days)  # DatetimeIndex обязателен
    price = cli.last_trade_price(ticker)

    closes = df["close"]
    tail = df.tail(cfg["look"])
    rng_low, rng_high = float(tail["low"].min()), float(tail["high"].max())
    rng_w = max(1e-9, rng_high - rng_low)
    pos = (price - rng_low) / rng_w

    slope  = _linreg_slope(closes.tail(cfg["trend"]).values)
    slope_norm = slope / max(1e-9, price)

    atr_d = float(_atr_like(df, n=cfg["atr"]).iloc[-1])
    atr_w = _weekly_atr(df) if cfg.get("use_weekly_atr") else atr_d
    vol_ratio = atr_d / max(1e-9, float(_atr_like(df, n=cfg["atr"]*2).iloc[-1]))

    # HA & MACD streaks (перегрев)
    ha = _heikin_ashi(df)
    ha_diff = ha["ha_close"].diff()
    ha_up_run   = _streak_by_sign(ha_diff, positive=True)
    ha_down_run = _streak_by_sign(ha_diff, positive=False)
    _, _, hist = _macd_hist(closes)
    macd_pos_run = _streak_by_sign(hist, positive=True)
    macd_neg_run = _streak_by_sign(hist, positive=False)

    # pivots
    hlc = _last_period_hlc(df, cfg["pivot_rule"])
    if not hlc:
        hlc = (float(df["high"].tail(60).max()),
               float(df["low"].tail(60).min()),
               float(df["close"].iloc[-1]))
    H, L, C = hlc
    piv = _fib_pivots(H, L, C)
    P, R1, R2, S1, S2 = piv["P"], piv["R1"], piv.get("R2"), piv["S1"], piv.get("S2")

    # буфер зон
    tol_k = {"ST":0.18, "MID":0.22, "LT":0.28}[hz]
    buf = tol_k * (atr_w if hz != "ST" else atr_d)

    def _near_from_below(level: float) -> bool:
        return (level is not None) and (0 <= level - price <= buf)

    def _near_from_above(level: float) -> bool:
        return (level is not None) and (0 <= price - level <= buf)

    # фильтры перегрева у порогов
    thr_ha   = {"ST":4,"MID":5,"LT":6}[hz]
    thr_macd = {"ST":4,"MID":6,"LT":8}[hz]
    long_up   = (ha_up_run >= thr_ha)  or (macd_pos_run >= thr_macd)
    long_down = (ha_down_run >= thr_ha) or (macd_neg_run >= thr_macd)

    # базовый сценарий
    band = _classify_band(price, piv, buf)
    very_high_pos = pos >= 0.80

    if long_up and (_near_from_below(R1) or _near_from_below(R2) or _near_from_below(S1)):
        action, scenario = "WAIT", "stall_after_long_up_at_pivot"
    elif long_down and (_near_from_above(R1) or _near_from_above(S1) or _near_from_above(S2)):
        action, scenario = "WAIT", "stall_after_long_down_at_pivot"
    else:
        if very_high_pos:
            if (R2 is not None) and (price > R2 + 0.6*buf) and (slope_norm > 0):
                action, scenario = "BUY", "breakout_up"
            else:
                action, scenario = "SHORT", "fade_top"
        else:
            if band >= +2:
                action, scenario = ("BUY","breakout_up") if ((R2 is not None) and (price > R2 + 0.6*buf) and (slope_norm > 0)) else ("SHORT","fade_top")
            elif band == +1:
                action, scenario = "WAIT","upper_wait"
            elif band == 0:
                action, scenario = ("BUY","trend_follow") if slope_norm >= 0 else ("WAIT","mid_range")
            elif band == -1:
                action, scenario = "BUY","revert_from_bottom"
            else:
                action, scenario = ("BUY","revert_from_bottom") if band <= -2 else ("WAIT","upper_wait")

    # уверенность
    base = 0.55 + 0.12*_clip01(abs(slope_norm)*1800) + 0.08*_clip01((vol_ratio-0.9)/0.6)
    if action == "WAIT": base -= 0.07
    conf = float(max(0.55, min(0.90, base)))

    # ===== AI override (опционально) =====
    try:
        from core.ai_inference import score_signal
    except Exception:
        score_signal = None
    if score_signal is not None:
        feats = dict(
            pos=pos, slope_norm=slope_norm,
            atr_d_over_price=(atr_d/max(1e-9, price)),
            vol_ratio=vol_ratio,
            ha_up_run=float(ha_up_run), ha_down_run=float(ha_down_run),
            macd_pos_run=float(macd_pos_run), macd_neg_run=float(macd_neg_run),
            band=float(band)
        )
        out_ai = score_signal(feats, hz=hz)
        if out_ai is not None:
            p_long = float(out_ai.get("p_long", 0.5))
            th_long  = float(os.getenv("ARXORA_AI_TH_LONG",  "0.55"))
            th_short = float(os.getenv("ARXORA_AI_TH_SHORT", "0.45"))
            if p_long >= th_long:
                action = "BUY"
                conf = float(max(0.55, min(0.90, 0.55 + 0.35*(p_long - th_long) / max(1e-9, 1.0 - th_long))))
            elif p_long <= th_short:
                action = "SHORT"
                conf = float(max(0.55, min(0.90, 0.55 + 0.35*((th_short - p_long) / max(1e-9, th_short)))))
            else:
                action = "WAIT"
                conf = float(max(0.48, min(0.83, conf - 0.05)))
    # ===== end override =====

    # ------- уровни (entry якорим к текущей цене для UI) -------
    entry_display = price  # показываем текущую цену как точку входа
    step_d, step_w = atr_d, atr_w

    if action == "BUY":
        # идеальный триггер (для логики), но в UI показываем текущую
        ideal_sl = (P - 0.60*step_w) if price >= P else (S1 - 0.60*step_w)
        sl = ideal_sl
        tp1, tp2, tp3 = _targets_pivot_aware(entry_display, "BUY", piv, band, hz, atr_w, atr_d, price)
        alt = "Если продавят ниже и не вернут — не лезем; ждём возврата и подтверждения сверху."
    elif action == "SHORT":
        ideal_sl = (R1 + 0.60*step_w) if price >= R1 else (price + 1.00*step_d)
        sl = ideal_sl
        tp1, tp2, tp3 = _targets_pivot_aware(entry_display, "SHORT", piv, band, hz, atr_w, atr_d, price)
        alt = "Если протолкнут выше и удержат — без погони; ждём возврата и слабости сверху."
    else:  # WAIT
        sl = price - 0.90*step_d
        tp1, tp2, tp3 = entry_display + 0.7*step_d, entry_display + 1.4*step_d, entry_display + 2.1*step_d
        alt = "Под кромкой — не догоняю; план на пробой/ретест или откат к опоре."

    # ограничения риска/дальности
    sl  = _cap_sl(entry_display, sl, action, hz, price, atr_w)
    tp1, tp2, tp3 = _apply_tp_floors(entry_display, sl, tp1, tp2, tp3, action, hz, price, atr_w if hz!="ST" else atr_d)
    tp1, tp2, tp3 = _clamp_tp_span(entry_display, (tp1, tp2, tp3), action, hz, price, atr_w)
    tp1, tp2, tp3 = _order_targets(entry_display, tp1, tp2, tp3, action)

    # вероятности
    p1 = _clip01(0.58 + 0.27*(conf-0.55)/0.35)
    p2 = _clip01(0.44 + 0.21*(conf-0.55)/0.35)
    p3 = _clip01(0.28 + 0.13*(conf-0.55)/0.35)

    # контекст
    chips = []
    if vol_ratio > 1.05: chips.append("волатильность растёт")
    if vol_ratio < 0.95: chips.append("волатильность сжимается")
    if long_up   and (_near_from_below(R1) or _near_from_below(R2) or _near_from_below(S1)): chips.append("длинная зелёная серия у порога")
    if long_down and (_near_from_above(R1) or _near_from_above(S1) or _near_from_above(S2)): chips.append("длинная красная серия у порога")

    # «живой» текст
    seed = int(hashlib.sha1(f"{ticker}{df.index[-1].date()}".encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    if action == "WAIT":
        lead = rng.choice(["Под кромкой — жду пробой/ретест.","Импульс длинный — не гонюсь.","Даём цене определиться."])
    elif action == "BUY":
        lead = rng.choice(["Опора близко — беру по ходу после паузы.","Спрос живой — вход от поддержки.","Восстановление держится — беру аккуратно."])
    else:
        lead = rng.choice(["Слабость у кромки — работаю от отказа.","Под потолком тяжело — шорт со стопом.","Импульс выдыхается — фиксирую вниз."])
    note_html = f"<div style='margin-top:10px; opacity:0.95;'>{lead}</div>"

    return {
        "last_price": float(price),
        "recommendation": {"action": action, "confidence": float(round(conf,4))},
        "levels": {"entry": float(entry_display), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
        "probs": {"tp1": float(p1), "tp2": float(p2), "tp3": float(p3)},
        "context": chips,
        "note_html": note_html,
        "alt": alt,
    }
