# core/strategy.py
import hashlib
import random
import numpy as np
import pandas as pd
from core.polygon_client import PolygonClient

# ---------- helpers (в UI не раскрываем) ----------

def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def _weekly_atr(df: pd.DataFrame, n_weeks: int = 8) -> float:
    """ATR по недельным свечам (закрытие пятницы)."""
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
    """
    Для сопоставления с твоим недельным графиком:
    - краткосрок: опираемся на дневной ATR; пивоты — недельные
    - среднесрок/долгосрок: используем weekly ATR (шире стоп/буфер)
    """
    if "Кратко" in text:  return dict(look=60,  trend=14, atr=14, pivot_period="W-FRI", use_weekly_atr=False)
    if "Средне" in text:  return dict(look=120, trend=28, atr=14, pivot_period="W-FRI", use_weekly_atr=True)
    return dict(look=240, trend=56, atr=14, pivot_period="W-FRI", use_weekly_atr=True)

# ---------- Fibonacci pivots (скрытые) ----------

def _last_period_hlc(df: pd.DataFrame, rule: str):
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

def _pivot_ladder(piv: dict) -> list[float]:
    keys = ["S3","S2","S1","P","R1","R2","R3"]
    return [float(piv[k]) for k in keys if k in piv and piv[k] is not None]

def _three_targets_from_pivots(entry: float, direction: str, piv: dict, step: float) -> tuple[float,float,float]:
    """
    Гарантируем монотонность: для BUY — цели строго выше entry, для SHORT — ниже.
    Если подходящих пивотов меньше трёх, докладываем ATR-ступени.
    """
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
    pos   = (price - low) / width  # 0..1

    closes = df["close"]
    slope  = _linreg_slope(closes.tail(cfg["trend"]).values)
    slope_norm = slope / max(1e-9, price)
    atr_s  = float(_atr_like(df, n=cfg["atr"]).iloc[-1])
    atr_w  = _weekly_atr(df) if cfg.get("use_weekly_atr") else atr_s
    vol_ratio = atr_s / max(1e-9, float(_atr_like(df, n=cfg["atr"]*2).iloc[-1]))
    streak    = _streak(closes)
    last = df.iloc[-1]
    body, upper, lower = _wick_profile(last)
    long_upper = upper > body*1.3 and upper > lower*1.1
    long_lower = lower > body*1.3 and lower > upper*1.1

    # Пивоты предыдущей недели
    piv = None
    hlc = _last_period_hlc(df, cfg["pivot_period"])
    if hlc:
        H, L, C = hlc
        piv = _fib_pivots(H, L, C)
        P   = piv["P"]; R1 = piv["R1"]; S1 = piv["S1"]; R2 = piv.get("R2")
    else:
        P, R1, S1, R2 = (high+low)/2, high, low, None

    # Буфер «вблизи уровня» — от weekly ATR для сред/долгосрока
    buf = 0.25 * atr_w

    # Зоны / условия
    near_top    = pos >= 0.7
    near_bottom = pos <= 0.3
    mid_zone    = 0.45 < pos < 0.55
    strong_up   = slope_norm > 0.001 and vol_ratio > 1.05
    strong_down = slope_norm < -0.001 and vol_ratio > 1.05
    breakout_up = near_top and strong_up and (price >= high * 0.995)
    breakout_dn = near_bottom and strong_down and (price <= low * 1.005)

    near_R2 = bool(R2) and abs(price - R2) <= buf
    touch_upper_band = price >= R1 + buf

    overheat_up   = streak >= 3 or long_upper
    overheat_down = streak <= -3 or long_lower

    # Сценарии (недельная логика у верха)
    if breakout_up:
        action, scenario = "BUY", "breakout_up"
    elif breakout_dn:
        action, scenario = "SHORT", "breakout_dn"
    elif near_R2 or touch_upper_band:
        action, scenario = ("SHORT", "fade_top") if overheat_up else ("WAIT", "upper_wait")
    elif near_bottom and overheat_down:
        action, scenario = "BUY", "revert_from_bottom"
    elif mid_zone:
        action, scenario = "WAIT", "mid_range"
    else:
        action, scenario = ("BUY" if slope_norm >= 0 else "SHORT"), "trend_follow"

    # Уверенность
    edge = abs(pos-0.5)*2
    trn  = _clip01(abs(slope_norm)*1800)
    vol  = _clip01((vol_ratio-0.9)/0.6)
    base = 0.50 + 0.22*edge + 0.16*trn + 0.10*vol
    if scenario == "mid_range": base -= 0.08
    conf = float(max(0.55, min(0.90, base)))

    # Шаг уровней
    step = atr_s
    step_w = atr_w  # для недельных якорей

    # Уровни (якорим к пивотам + weekly ATR на долгосроке)
    if action == "BUY":
        if scenario == "breakout_up":
            entry = price + 0.10*step_w
            sl    = (R1 - 1.00*step_w) if piv else (price - 1.00*step_w)
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w) if piv else (entry + 0.8*step_w, entry + 1.6*step_w, entry + 2.4*step_w)
        else:
            # вход после отката к R1/P (не гонимся под R2)
            if piv:
                if price < P:
                    entry = max(price, S1 + 0.15*step_w)
                    sl    = S1 - 0.60*step_w
                else:
                    entry = max(price, P + 0.10*step_w)
                    sl    = P - 0.60*step_w
                tp1, tp2, tp3 = _three_targets_from_pivots(entry, "BUY", piv, step_w)
            else:
                entry = price - 0.15*step; sl = price - 1.00*step
                tp1, tp2, tp3 = entry + 0.8*step, entry + 1.6*step, entry + 2.4*step
        alt = "Если продавят ниже и не вернут — не лезем; ждём возврата сверху и подтверждения."

    elif action == "SHORT":
        if scenario == "breakout_dn":
            entry = price - 0.10*step_w
            sl    = (S1 + 1.00*step_w) if piv else (price + 1.00*step_w)
            tp1, tp2, tp3 = _three_targets_from_pivots(entry, "SHORT", piv, step_w) if piv else (entry - 0.8*step_w, entry - 1.6*step_w, entry - 2.4*step_w)
        else:
            if piv:
                entry = min(price, R1 - 0.15*step_w)
                sl    = R1 + 0.60*step_w
                tp1, tp2, tp3 = _three_targets_from_pivots(entry, "SHORT", piv, step_w)
            else:
                entry = price + 0.15*step; sl = price + 1.00*step
                tp1, tp2, tp3 = entry - 0.8*step, entry - 1.6*step, entry - 2.4*step
        alt = "Если протолкнут выше и удержат — без погони; ждём возврата и признаков слабости выше."

    else:  # WAIT
        entry = price
        sl    = price - 0.90*step
        tp1, tp2, tp3 = entry + 0.7*step, entry + 1.4*step, entry + 2.1*step
        alt = "Под самой кромкой — не гонюсь; работаю на пробое после ретеста или на откате к R1/P."

    # Вероятности целей
    p1 = _clip01(0.58 + 0.27*(conf-0.55)/0.35)
    p2 = _clip01(0.44 + 0.21*(conf-0.55)/0.35)
    p3 = _clip01(0.28 + 0.13*(conf-0.55)/0.35)

    # Контекст-бейджи
    chips = []
    if near_R2: chips.append("под верхней кромкой (R2)")
    if touch_upper_band and not near_R2: chips.append("верхний диапазон")
    if near_bottom: chips.append("нижняя кромка")
    if mid_zone: chips.append("середина диапазона")
    if vol_ratio > 1.05: chips.append("волатильность растёт")
    if vol_ratio < 0.95: chips.append("волатильность сжимается")
    if streak >= 3: chips.append(f"{streak} зелёных подряд")
    if streak <= -3: chips.append(f"{abs(streak)} красных подряд")
    if long_upper: chips.append("тени сверху")
    if long_lower: chips.append("тени снизу")

    # «живой» текст
    seed = int(hashlib.sha1(f"{ticker}{df.index[-1].date()}".encode()).hexdigest(), 16) % (2**32)
    rng  = random.Random(seed)
    if action == "BUY":
        lead = rng.choice([
            "Снизу чувствуется спрос — беру откат и реакцию.",
            "Покупатели рядом, цена подпирается — работаю по ходу.",
            "Низ близко, спрос живой — входим после возврата."
        ])
    elif action == "SHORT":
        lead = rng.choice([
            "Сверху тяжело — ищу слабость у кромки.",
            "Под потолком продавец — работаю от отказа.",
            "Верх близко, импульс выдыхается — готовлю шорт."
        ])
    else:
        lead = rng.choice([
            "Под самой кромкой. Преимущества нет — без входа.",
            "Баланс силы у верха — даю цене определиться.",
            "Здесь не догоняю — жду отката к R1/P."
        ])

    add = []
    if scenario == "breakout_up": add.append("Импульс вверх — работаю по движению после короткой паузы и ретеста.")
    if scenario == "breakout_dn": add.append("Импульс вниз — не ловлю нож, беру после отката.")
    if scenario == "fade_top":    add.append("Игра от края: сперва реакция, потом вход.")
    if scenario == "trend_follow": add.append("Идём за текущим ходом — вход на откате, не догоняя.")

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
