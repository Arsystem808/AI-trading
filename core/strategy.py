# core/strategy.py
import os
import hashlib
import random
import numpy as np
import pandas as pd
from core.polygon_client import PolygonClient


# =========================
# БАЗОВЫЕ ХЕЛПЕРЫ
# =========================
def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


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


def _streak_simple(closes: pd.Series) -> int:
    """Последовательность обычных свечей (рост/падение)."""
    s = 0
    for i in range(len(closes) - 1, 0, -1):
        d = closes.iloc[i] - closes.iloc[i - 1]
        if d > 0:
            if s < 0:
                break
            s += 1
        elif d < 0:
            if s > 0:
                break
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


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


# =========================
# HEIKIN ASHI / MACD / RSI
# =========================
def _heikin_ashi(ohlc: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=ohlc.index, dtype=float)
    ha["ha_close"] = (ohlc["open"] + ohlc["high"] + ohlc["low"] + ohlc["close"]) / 4.0
    ha_open = []
    prev_ho = float(ohlc["open"].iloc[0])
    prev_hc = float(ohlc["close"].iloc[0])
    for i in range(len(ohlc)):
        if i == 0:
            cur = (prev_ho + prev_hc) / 2.0
        else:
            prev_ho = ha_open[-1]
            prev_hc = ha["ha_close"].iloc[i - 1]
            cur = (prev_ho + prev_hc) / 2.0
        ha_open.append(cur)
    ha["ha_open"] = pd.Series(ha_open, index=ohlc.index)
    ha["ha_high"] = pd.concat([ohlc["high"], ha["ha_open"], ha["ha_close"]], axis=1).max(axis=1)
    ha["ha_low"]  = pd.concat([ohlc["low"],  ha["ha_open"], ha["ha_close"]], axis=1).min(axis=1)
    return ha


def _ha_streak(ha: pd.DataFrame) -> int:
    sign = np.sign(ha["ha_close"] - ha["ha_open"])
    s = 0
    for i in range(len(sign) - 1, -1, -1):
        if sign.iloc[i] > 0:
            if s < 0: break
            s += 1
        elif sign.iloc[i] < 0:
            if s > 0: break
            s -= 1
        else:
            break
    return int(s)


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=1).mean()


def _macd_hist(close: pd.Series, fast=12, slow=26, signal=9) -> pd.Series:
    macd = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd, signal)
    return macd - sig


def _macd_streak(hist: pd.Series) -> int:
    s = 0
    for i in range(len(hist) - 1, -1, -1):
        h = hist.iloc[i]
        if h > 0:
            if s < 0: break
            s += 1
        elif h < 0:
            if s > 0: break
            s -= 1
        else:
            break
    return int(s)


def _macd_decelerating(hist: pd.Series, look: int = 4) -> bool:
    if len(hist) < look + 1:
        return False
    tail = hist.tail(look)
    if (tail > 0).all():
        return bool((tail.diff().dropna() < 0).all())
    if (tail < 0).all():
        return bool((tail.diff().dropna() > 0).all())
    return False


def _rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _rsi_percentile_flag(rsi: pd.Series, upper_q=0.8, lower_q=0.2, window=200):
    if len(rsi) < 30:
        return False, False
    tail = rsi.tail(min(window, len(rsi)))
    cur = float(tail.iloc[-1])
    lo = float(tail.quantile(lower_q))
    hi = float(tail.quantile(upper_q))
    return (cur >= hi), (cur <= lo)  # (overbought, oversold)


# =========================
# ПИВОТЫ (Fibonacci)
# =========================
def _horizon_cfg(text: str):
    if "Кратко" in text:
        return dict(look=60,  trend=14, atr=14, pivot_period="W-FRI", use_weekly_atr=False)
    if "Средне" in text:
        return dict(look=120, trend=28, atr=14, pivot_period="M",     use_weekly_atr=True)
    return dict(look=240, trend=56, atr=14, pivot_period="M",     use_weekly_atr=True)


def _hz_tag(text: str) -> str:
    if "Кратко" in text:  return "ST"
    if "Средне" in text:  return "MID"
    return "LT"


def _last_period_hlc(df: pd.DataFrame, rule: str):
    g = _resample_ohlc(df, rule)[["high", "low", "close"]]
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


# =========================
# ФИКСЫ ДЛЯ ЦЕЛЕЙ
# =========================
def _apply_tp_floors(entry, sl, tp1, tp2, tp3, action, hz_tag, price, atr_val):
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
    if abs(tp1 - entry) < floor1:  tp1 = entry + side * floor1
    floor2 = max(1.6 * floor1, (min_rr[hz_tag] * 1.8) * risk)
    if abs(tp2 - entry) < floor2:  tp2 = entry + side * floor2
    min_gap3 = max(0.8 * floor1, 0.6 * risk)
    if abs(tp3 - tp2) < min_gap3:  tp3 = tp2 + side * min_gap3
    return tp1, tp2, tp3


def _order_targets(entry, tp1, tp2, tp3, action, eps: float = 1e-6):
    side = 1 if action == "BUY" else -1
    arr = sorted([float(tp1), float(tp2), float(tp3)], key=lambda x: side * (x - entry))
    d0 = side * (arr[0] - entry); d1 = side * (arr[1] - entry); d2 = side * (arr[2] - entry)
    if d1 - d0 < eps: arr[1] = entry + side * max(d0 + max(eps, 0.1 * abs(d0)), d1 + eps)
    if side * (arr[2] - entry) - side * (arr[1] - entry) < eps:
        d1 = side * (arr[1] - entry)
        arr[2] = entry + side * max(d1 + max(eps, 0.1 * abs(d1)), d2 + eps)
    return arr[0], arr[1], arr[2]


# === НОВОЕ: режимный каппинг целей (чтобы не ставить чрезмерно дальние контртрендовые цели)
def _regime_cap_targets(action: str, hz: str, tp1: float, tp2: float, tp3: float,
                        piv: dict, strong_bull: bool, strong_bear: bool):
    P, R1, R2, R3, S1, S2, S3 = piv["P"], piv["R1"], piv["R2"], piv["R3"], piv["S1"], piv["S2"], piv["S3"]
    if action == "SHORT" and strong_bull:
        # Контртрендовый шорт в бычьем режиме: тянем цели ближе — не глубже P/S1
        tp1 = max(tp1, (P + R1) / 2.0)             # первая фиксация ближе к центру
        tp2 = max(tp2, P)                           # не глубже центра диапазона
        tp3 = max(tp3, P if hz == "ST" else S1)     # ST: максимум до P, MID/LT: до S1
    elif action == "BUY" and strong_bear:
        # Контртрендовый лонг в медвежьем режиме: не выше P/R1
        tp1 = min(tp1, (P + S1) / 2.0)
        tp2 = min(tp2, P)
        tp3 = min(tp3, P if hz == "ST" else R1)
    return tp1, tp2, tp3


# =========================
# ОСНОВНОЙ АНАЛИЗ
# =========================
def analyze_asset(ticker: str, horizon: str):
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)
    hz = _hz_tag(horizon)

    # данные (дневные)
    days = max(360, cfg["look"] * 4)
    df = cli.daily_ohlc(ticker, days=days)
    price = cli.last_trade_price(ticker)

    # производные ТФ
    df_w = _resample_ohlc(df, "W-FRI")
    df_m = _resample_ohlc(df, "M")
    base_df = df if hz == "ST" else df_w
    senior_df = df_w if hz == "ST" else df_m

    # позиция/вола/тренд
    tail = df.tail(cfg["look"])
    rng_low, rng_high = float(tail["low"].min()), float(tail["high"].max())
    rng_w = max(1e-9, rng_high - rng_low)
    pos = (price - rng_low) / rng_w

    closes = df["close"]
    slope = _linreg_slope(closes.tail(cfg["trend"]).values)
    slope_norm = slope / max(1e-9, price)
    atr_d = float(_atr_like(df, n=cfg["atr"]).iloc[-1])
    atr_2d = float(_atr_like(df, n=cfg["atr"] * 2).iloc[-1])
    vol_ratio = atr_d / max(1e-9, atr_2d)
    simple_streak = _streak_simple(closes)

    last_row = df.iloc[-1]
    body, upper, lower = _wick_profile(last_row)
    long_upper = upper > body * 1.3 and upper > lower * 1.1
    long_lower = lower > body * 1.3 and lower > upper * 1.1

    atr_w = float(_atr_like(df_w, 8).iloc[-1]) if len(df_w) > 8 else atr_d
    step_d, step_w = atr_d, atr_w

    # Пивоты (последняя завершённая неделя/месяц)
    hlc = _last_period_hlc(df, cfg["pivot_period"])
    if not hlc:
        hlc = (float(df["high"].tail(60).max()), float(df["low"].tail(60).min()), float(df["close"].iloc[-1]))
    H, L, C = hlc
    piv = _fib_pivots(H, L, C)
    P, R1, R2, R3, S1, S2, S3 = piv["P"], piv["R1"], piv["R2"], piv["R3"], piv["S1"], piv["S2"], piv["S3"]

    buf = 0.25 * atr_w
    band = _classify_band(price, piv, buf)

    # Индикаторы base/senior
    ha_base = _heikin_ashi(base_df); ha_stk = _ha_streak(ha_base)
    macd_base = _macd_hist(base_df["close"]); macd_stk = _macd_streak(macd_base)
    macd_tired = _macd_decelerating(macd_base, look=4)
    rsi_base = _rsi_wilder(base_df["close"])
    rsi_overbought, rsi_oversold = _rsi_percentile_flag(rsi_base)

    ha_sen = _heikin_ashi(senior_df)
    ha_sen_color = np.sign(ha_sen["ha_close"].iloc[-1] - ha_sen["ha_open"].iloc[-1])
    macd_sen = _macd_hist(senior_df["close"]); macd_sen_sign = np.sign(float(macd_sen.iloc[-1]))

    ha_thr   = {"ST": 4, "MID": 5, "LT": 6}[hz]
    macd_thr = {"ST": 4, "MID": 6, "LT": 8}[hz]

    tol_pct = {"ST": 0.008, "MID": 0.010, "LT": 0.012}[hz]
    tol_abs = tol_pct * price
    near_R3 = abs(price - R3) <= (tol_abs + 0.6 * buf)
    near_R2 = abs(price - R2) <= (tol_abs + 0.6 * buf)
    near_S2 = abs(price - S2) <= (tol_abs + 0.6 * buf)
    near_S3 = abs(price - S3) <= (tol_abs + 0.6 * buf)

    overheat_top = (
        (near_R2 or near_R3)
        and ((ha_stk >= ha_thr) or (abs(macd_stk) >= macd_thr) or macd_tired)
        and (rsi_overbought or long_upper or band >= +2)
    )
    exhaustion_bottom = (
        (near_S2 or near_S3)
        and ((-ha_stk >= ha_thr) or (abs(macd_stk) >= macd_thr) or macd_tired)
        and (rsi_oversold or long_lower or band <= -2)
    )

    # Режим рынка (для каппинга)
    strong_bull = (ha_sen_color > 0) and (macd_sen_sign >= 0) and (slope_norm > 0) and (price >= P)
    strong_bear = (ha_sen_color < 0) and (macd_sen_sign <= 0) and (slope_norm < 0) and (price <= P)

    # Выбор сценария
    very_high_pos = pos >= 0.80
    last_o, last_c = float(df["open"].iloc[-1]), float(df["close"].iloc[-1])
    last_h = float(df["high"].iloc[-1])
    upper_wick_d = max(0.0, last_h - max(last_o, last_c))
    body_d = abs(last_c - last_o)
    bearish_reject = (last_c < last_o) and (upper_wick_d > body_d)

    action, scenario = None, "base"
    if overheat_top:
        action, scenario = "SHORT", "overheat_top"
    elif exhaustion_bottom:
        action, scenario = "BUY", "exhaustion_bottom"
    else:
        if very_high_pos:
            if price > R2 + 0.6 * buf and slope_norm > 0:
                action, scenario = "BUY", "breakout_up"
            else:
                action, scenario = "SHORT", "fade_top"
        else:
            if band >= +2:
                action, scenario = (("BUY", "breakout_up") if (price > R2 + 0.6 * buf and slope_norm > 0)
                                    else ("SHORT", "fade_top"))
            elif band == +1:
                action, scenario = (("WAIT", "upper_wait") if (slope_norm > 0.0015 and not bearish_reject and not long_upper)
                                    else ("SHORT", "fade_top"))
            elif band == 0:
                action, scenario = ("BUY", "trend_follow") if slope_norm >= 0 else ("WAIT", "mid_range")
            elif band == -1:
                action, scenario = ("BUY", "revert_from_bottom") if (simple_streak <= -3 or long_lower) else ("BUY", "trend_follow")
            else:
                action, scenario = ("BUY", "revert_from_bottom") if band <= -2 else ("WAIT", "upper_wait")

    # Уверенность + подтверждение старшего ТФ
    base_conf = 0.55 + 0.12 * _clip01(abs(slope_norm) * 1800) + 0.08 * _clip01((vol_ratio - 0.9) / 0.6)
    if action == "WAIT": base_conf -= 0.07
    if band >= +1 and action == "BUY": base_conf -= 0.10
    if band <= -1 and action == "BUY": base_conf += 0.05
    if action == "BUY":
        if (ha_sen_color > 0) and (macd_sen_sign >= 0): base_conf += 0.05
        if rsi_oversold: base_conf += 0.02
        if overheat_top: base_conf -= 0.12
    elif action == "SHORT":
        if (ha_sen_color < 0) and (macd_sen_sign <= 0): base_conf += 0.05
        if rsi_overbought: base_conf += 0.02
        if exhaustion_bottom: base_conf -= 0.12
    conf = float(max(0.55, min(0.90, base_conf)))

    # ===== AI override (опционально)
    try:
        from core.ai_inference import score_signal
    except Exception:
        score_signal = None
    if score_signal is not None:
        feats = dict(
            pos=pos, slope_norm=slope_norm, atr_d_over_price=(atr_d / max(1e-9, price)),
            vol_ratio=vol_ratio, streak=float(simple_streak), band=float(band),
            long_upper=bool(long_upper), long_lower=bool(long_lower),
            ha_streak=float(ha_stk), macd_streak=float(macd_stk),
            macd_tired=bool(macd_tired), rsi_overbought=bool(rsi_overbought),
            rsi_oversold=bool(rsi_oversold),
        )
        hz_tag = _hz_tag(horizon)
        out_ai = score_signal(feats, hz=hz_tag)
        if out_ai is not None:
            p_long = float(out_ai.get("p_long", 0.5))
            th_long  = float(os.getenv("ARXORA_AI_TH_LONG",  "0.55"))
            th_short = float(os.getenv("ARXORA_AI_TH_SHORT", "0.45"))
            if p_long >= th_long:
                action_ai, conf_ai = "BUY",   0.55 + 0.35 * (p_long - th_long)  / max(1e-9, 1.0 - th_long)
            elif p_long <= th_short:
                action_ai, conf_ai = "SHORT", 0.55 + 0.35 * ((th_short - p_long) / max(1e-9, th_short))
            else:
                action_ai, conf_ai = "WAIT",  0.55 - 0.07
            action, conf = action_ai, float(max(0.55, min(0.90, conf_ai)))
    # ===== конец AI override

    # УРОВНИ
    if action == "BUY":
        if scenario == "exhaustion_bottom":
            if near_S3:
                entry = max(price, S3 + 0.15 * step_w); sl = S3 - 0.80 * step_w
                tp1, tp2, tp3 = S2, P, R1
            elif near_S2:
                entry = max(price, S2 + 0.15 * step_w); sl = S2 - 0.80 * step_w
                tp1, tp2, tp3 = P, R1, R2
            else:
                entry = max(price, P + 0.10 * step_w); sl = P - 0.60 * step_w
                tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
            alt = "Если продавят ниже и не вернут — не лезем; ждём возврата сверху и подтверждения."
        elif scenario == "breakout_up":
            base_ref = R2
            entry = max(price, base_ref + 0.10 * step_w); sl = base_ref - 1.00 * step_w
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
            alt = "Если возврат под уровень и удержание — без погони; ждём ретест."
        else:
            if price < P:
                entry = max(price, S1 + 0.15 * step_w); sl = S1 - 0.60 * step_w
            else:
                entry = max(price, P + 0.10 * step_w); sl = P - 0.60 * step_w
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
            alt = "Поддержка держит — работаем по реакции; без погони."
    elif action == "SHORT":
        if scenario == "overheat_top":
            if near_R3:
                entry = min(price, R3 - 0.15 * step_w); sl = R3 + 0.80 * step_w
                tp1, tp2, tp3 = R2, P, S1
            elif near_R2:
                entry = min(price, R2 - 0.15 * step_w); sl = R2 + 0.80 * step_w
                mid_PS1 = (P + S1) / 2.0
                tp1, tp2, tp3 = mid_PS1, S1, S2
            else:
                entry = price + 0.15 * step_d; sl = price + 1.00 * step_d
                tp1, tp2, tp3 = _three_targets_from_pivots(entry, "SHORT", piv, step_w)
            alt = "Если протолкнут выше и удержат — без погони; ждём возврата и признаков слабости сверху."
        else:
            if price >= R1:
                entry = min(price, R1 - 0.15 * step_w); sl = R1 + 0.60 * step_w
            else:
                entry = price + 0.15 * step_d; sl = price + 1.00 * step_d
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "SHORT", piv, step_w)
            alt = "Слабость у кромки — работаем аккуратно; без охоты за движением."
    else:  # WAIT
        entry = price; sl = price - 0.90 * step_d
        tp1, tp2, tp3 = entry + 0.7 * step_d, entry + 1.4 * step_d, entry + 2.1 * step_d
        alt = "Под верхом — не догоняю; план на пробой с ретестом или откат к опоре."

    # Мин. дистанции -> Режимный каппинг -> Порядок
    atr_for_floor = step_w if hz != "ST" else step_d
    tp1, tp2, tp3 = _apply_tp_floors(entry, sl, tp1, tp2, tp3, action, hz, price, atr_for_floor)
    tp1, tp2, tp3 = _regime_cap_targets(action, hz, tp1, tp2, tp3, piv, strong_bull, strong_bear)
    tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)

    # Вероятности
    p1 = _clip01(0.58 + 0.27 * (conf - 0.55) / 0.35)
    p2 = _clip01(0.44 + 0.21 * (conf - 0.55) / 0.35)
    p3 = _clip01(0.28 + 0.13 * (conf - 0.55) / 0.35)

    # Контекст
    chips = []
    if pos >= 0.8: chips.append("верхний диапазон")
    elif pos >= 0.6: chips.append("под верхним краем")
    elif pos >= 0.4: chips.append("средняя зона")
    elif pos >= 0.2: chips.append("нижняя половина")
    else: chips.append("ниже опоры")
    if vol_ratio > 1.05: chips.append("волатильность растёт")
    if vol_ratio < 0.95: chips.append("волатильность сжимается")
    if simple_streak >= 3: chips.append(f"{simple_streak} зелёных подряд")
    if simple_streak <= -3: chips.append(f"{abs(simple_streak)} красных подряд")
    if ha_stk >= ha_thr: chips.append(f"HA-серия {ha_stk}")
    if ha_stk <= -ha_thr: chips.append(f"HA-серия {abs(ha_stk)} красных")
    if macd_tired: chips.append("MACD: усталость")

    # Текст
    seed = int(hashlib.sha1(f"{ticker}{df.index[-1].date()}".encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    if action == "BUY":
        lead = rng.choice([
            "Опора держит — беру реакцию без погони.",
            "Спрос живой — вход после паузы.",
            "Поддержка рядом — работаю по тренду.",
        ])
        if exhaustion_bottom: lead += " Перегиб вниз иссяк — играю от восстановления."
        if rsi_oversold: lead += " RSI внизу поддерживает идею."
        if strong_bear: lead += " Старший ТФ медвежий — цели укорочены к P/R1."
    elif action == "SHORT":
        lead = rng.choice([
            "Под потолком продавец — работаю от отказа.",
            "Сверху тяжело — готовлю шорт.",
            "Импульс выдыхается у кромки — шорт по слабости.",
        ])
        if overheat_top: lead += " Перегрев у крыши — беру слабость."
        if rsi_overbought: lead += " RSI сверху — риск разворота повышен."
        if strong_bull: lead += " Старший ТФ бычий — цели укорочены к P/S1."
    else:
        lead = rng.choice([
            "Даю цене определиться: пробой/ретест или откат к опоре.",
            "Без входа — план от отката.",
            "Под кромкой — без погони.",
        ])

    note_html = f"<div style='margin-top:10px; opacity:0.95;'>{lead}</div>"

    return {
        "last_price": float(price),
        "recommendation": {"action": action, "confidence": float(round(conf, 4))},
        "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
        "probs": {"tp1": float(p1), "tp2": float(p2), "tp3": float(p3)},
        "context": chips,
        "note_html": note_html,
        "alt": alt,
    }
