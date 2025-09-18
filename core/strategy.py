# core/strategy.py

import os
import hashlib
import random
import numpy as np
import pandas as pd
from datetime import datetime, time
import pytz
from core.polygon_client import PolygonClient

# –––––––––– small utils ––––––––––

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
return float(((x - xm) * (y - ym)).sum() / denom)

# –––––––––– ATR ––––––––––

def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
hl = df[“high”] - df[“low”]
hc = (df[“high”] - df[“close”].shift(1)).abs()
lc = (df[“low”] - df[“close”].shift(1)).abs()
tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
return tr.rolling(n, min_periods=1).mean()

def _weekly_atr(df: pd.DataFrame, n_weeks: int = 8) -> float:
w = df.resample(“W-FRI”).agg({“high”:“max”,“low”:“min”,“close”:“last”}).dropna()
if len(w) < 2:
return float((df[“high”] - df[“low”]).tail(14).mean())
hl = w[“high”] - w[“low”]
hc = (w[“high”] - w[“close”].shift(1)).abs()
lc = (w[“low”] - w[“close”].shift(1)).abs()
tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
return float(tr.rolling(n_weeks, min_periods=1).mean().iloc[-1])

# –––––––––– Heikin Ashi & MACD ––––––––––

def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
ha = pd.DataFrame(index=df.index.copy())
ha[“ha_close”] = (df[“open”] + df[“high”] + df[“low”] + df[“close”]) / 4.0
ha_open = [(df[“open”].iloc[0] + df[“close”].iloc[0]) / 2.0]
for i in range(1, len(df)):
ha_open.append((ha_open[-1] + ha[“ha_close”].iloc[i-1]) / 2.0)
ha[“ha_open”] = pd.Series(ha_open, index=df.index)
return ha

def _streak_by_sign(series: pd.Series, positive: bool = True) -> int:
want_pos = 1 if positive else -1
run = 0
vals = series.values
for i in range(len(vals) - 1, -1, -1):
v = vals[i]
if (v > 0 and want_pos == 1) or (v < 0 and want_pos == -1):
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

# –––––––––– horizons & pivots ––––––––––

def _horizon_cfg(text: str):
# ST — weekly; MID/LT — monthly
if “Кратко” in text:
return dict(look=60, trend=14, atr=14, pivot_rule=“W-FRI”, use_weekly_atr=False, hz=“ST”)
if “Средне” in text:
return dict(look=120, trend=28, atr=14, pivot_rule=“M”, use_weekly_atr=True, hz=“MID”)
return dict(look=240, trend=56, atr=14, pivot_rule=“M”, use_weekly_atr=True, hz=“LT”)

def _last_period_hlc(df: pd.DataFrame, rule: str):
g = df.resample(rule).agg({“high”:“max”,“low”:“min”,“close”:“last”}).dropna()
if len(g) < 2:
return None
row = g.iloc[-2]  # последняя завершённая
return float(row[“high”]), float(row[“low”]), float(row[“close”])

def _fib_pivots(H: float, L: float, C: float):
P = (H + L + C) / 3.0
d = H - L
R1 = P + 0.382 * d; R2 = P + 0.618 * d; R3 = P + 1.000 * d
S1 = P - 0.382 * d; S2 = P - 0.618 * d; S3 = P - 1.000 * d
return {“P”:P,“R1”:R1,“R2”:R2,“R3”:R3,“S1”:S1,“S2”:S2,“S3”:S3}

def _classify_band(price: float, piv: dict, buf: float) -> int:
# -3: <S2, -2:[S2,S1), -1:[S1,P), 0:[P,R1), +1:[R1,R2), +2:[R2,R3), +3:>=R3
P, R1 = piv[“P”], piv[“R1”]
R2, R3, S1, S2 = piv.get(“R2”), piv.get(“R3”), piv[“S1”], piv.get(“S2”)
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

# –––––––––– wick profile (для AI/псевдо) ––––––––––

def _wick_profile(row: pd.Series):
o, c, h, l = float(row[“open”]), float(row[“close”]), float(row[“high”]), float(row[“low”])
body = max(1e-9, abs(c - o))
up_wick = max(0.0, h - max(o, c))
dn_wick = max(0.0, min(o, c) - l)
return body, up_wick, dn_wick

# –––––––––– Order kind (STOP/LIMIT/NOW) ––––––––––

def _entry_kind(action: str, entry: float, price: float, step_d: float) -> str:
tol = max(0.0015 * max(1.0, price), 0.15 * step_d)  # копеечный допуск (без нулевой цены)
if action == “BUY”:
if entry > price + tol:  return “buy-stop”
if entry < price - tol:  return “buy-limit”
return “buy-now”
if action == “SHORT”:
if entry < price - tol:  return “sell-stop”
if entry > price + tol:  return “sell-limit”
return “sell-now”
return “wait”

# –––––––––– TP/SL guards ––––––––––

def _apply_tp_floors(entry: float, sl: float, tp1: float, tp2: float, tp3: float,
action: str, hz_tag: str, price: float, atr_val: float):
“”“Минимальные разумные дистанции до целей — чтобы TP не были слишком близко.”””
if action not in (“BUY”, “SHORT”):
return tp1, tp2, tp3
risk = abs(entry - sl)
if risk <= 1e-9:
return tp1, tp2, tp3
side = 1 if action == “BUY” else -1
min_rr   = {“ST”:0.80,“MID”:1.00,“LT”:1.20}
min_pct  = {“ST”:0.006,“MID”:0.012,“LT”:0.018}
atr_mult = {“ST”:0.50,“MID”:0.80,“LT”:1.20}
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

def _order_targets(entry: float, tp1: float, tp2: float, tp3: float, action: str, eps: float = 1e-6):
“”“Гарантирует порядок целей: TP1 ближе, TP3 дальше (в сторону сделки).”””
side = 1 if action == “BUY” else -1
arr = sorted([float(tp1), float(tp2), float(tp3)], key=lambda x: side*(x-entry))
d0 = side * (arr[0] - entry)
d1 = side * (arr[1] - entry)
d2 = side * (arr[2] - entry)
if d1 - d0 < eps:
arr[1] = entry + side * max(d0 + max(eps, 0.1*abs(d0)), d1 + eps)
if side*(arr[2]-entry) - side*(arr[1]-entry) < eps:
d1 = side * (arr[1] - entry)
arr[2] = entry + side * max(d1 + max(eps, 0.1*abs(d1)), d2 + eps)
return arr[0], arr[1], arr[2]

def _clamp_tp_by_trend(action: str, hz: str,
tp1: float, tp2: float, tp3: float,
piv: dict, step_w: float,
slope_norm: float, macd_pos_run: int, macd_neg_run: int):
“”“Интуитивные «предохранители» целей при мощном тренде (чтобы не ставить TP против паровоза слишком далеко).”””
thr_macd = {“ST”:3, “MID”:5, “LT”:6}[hz]
bullish = (slope_norm > 0.0006) and (macd_pos_run >= thr_macd)
bearish = (slope_norm < -0.0006) and (macd_neg_run >= thr_macd)

```
P, R1, S1 = piv["P"], piv["R1"], piv["S1"]

if action == "SHORT" and bullish:
    limit = max(R1 - 1.2*step_w, (P + R1)/2.0)
    tp1 = max(tp1, limit - 0.2*step_w)
    tp2 = max(tp2, limit)
    tp3 = max(tp3, limit + 0.4*step_w)
if action == "BUY" and bearish:
    limit = min(S1 + 1.2*step_w, (P + S1)/2.0)
    tp1 = min(tp1, limit + 0.2*step_w)
    tp2 = min(tp2, limit)
    tp3 = min(tp3, limit - 0.4*step_w)
return tp1, tp2, tp3
```

def _sanity_levels(action: str, entry: float, sl: float,
tp1: float, tp2: float, tp3: float,
price: float, step_d: float, step_w: float, hz: str):
“””
Жёсткая проверка сторон/зазоров:
- SL обязательно по «неправильную» сторону
- TP обязательно по «правильную» сторону
- минимальные зазоры для TP1/TP2/TP3
“””
side = 1 if action == “BUY” else -1
# минимальные зазоры (зависит от горизонта)
min_tp_gap = {“ST”:0.40, “MID”:0.70, “LT”:1.10}[hz] * step_w
min_tp_pct = {“ST”:0.004,“MID”:0.009,“LT”:0.015}[hz] * price
floor_gap = max(min_tp_gap, min_tp_pct, 0.35*abs(entry - sl) if sl != entry else 0.0)

```
# SL sanity
if action == "BUY" and sl >= entry - 0.25*step_d:
    sl = entry - max(0.60*step_w, 0.90*step_d)
if action == "SHORT" and sl <= entry + 0.25*step_d:
    sl = entry + max(0.60*step_w, 0.90*step_d)

# TP sanity: строго в сторону сделки + минимальная дистанция
def _push_tp(tp, rank):
    need = floor_gap * (1.0 if rank == 1 else (1.6 if rank == 2 else 2.2))
    want = entry + side * need
    # если TP по неправильную сторону — перекидываем
    if side * (tp - entry) <= 0:
        return want
    # если слишком близко — отодвигаем
    if abs(tp - entry) < need:
        return want
    return tp

tp1 = _push_tp(tp1, 1)
tp2 = _push_tp(tp2, 2)
tp3 = _push_tp(tp3, 3)

# упорядочим точно
tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)
return sl, tp1, tp2, tp3
```

# –––––––––– main engine ––––––––––

def analyze_asset(ticker: str, horizon: str):
cli = PolygonClient()
cfg = _horizon_cfg(horizon)
hz = cfg[“hz”]

```
# данные
days = max(90, cfg["look"] * 2)
df = cli.daily_ohlc(ticker, days=days)  # DatetimeIndex обязателен
price = cli.last_trade_price(ticker)

closes = df["close"]
tail = df.tail(cfg["look"])
rng_low, rng_high = float(tail["low"].min()), float(tail["high"].max())
rng_w = max(1e-9, rng_high - rng_low)
pos = (price - rng_low) / rng_w

slope = _linreg_slope(closes.tail(cfg["trend"]).values)
slope_norm = slope / max(1e-9, price)

atr_d = float(_atr_like(df, n=cfg["atr"]).iloc[-1])
atr_w = _weekly_atr(df) if cfg.get("use_weekly_atr") else atr_d
vol_ratio = atr_d / max(1e-9, float(_atr_like(df, n=cfg["atr"] * 2).iloc[-1]))

# HA & MACD
ha = _heikin_ashi(df)
ha_diff = ha["ha_close"].diff()
ha_up_run = _streak_by_sign(ha_diff, True)
ha_down_run = _streak_by_sign(ha_diff, False)
macd, sig, hist = _macd_hist(closes)
macd_pos_run = _streak_by_sign(hist, True)
macd_neg_run = _streak_by_sign(hist, False)

# текущая свеча: тени (для AI/псевдо)
last_row = df.iloc[-1]
body, up_wick, dn_wick = _wick_profile(last_row)
long_upper = (up_wick > body * 1.3) and (up_wick > dn_wick * 1.1)
long_lower = (dn_wick > body * 1.3) and (dn_wick > up_wick * 1.1)

# pivots (ST weekly, MID/LT monthly) — last completed period
hlc = _last_period_hlc(df, cfg["pivot_rule"])
if not hlc:
    hlc = (float(df["high"].tail(60).max()),
           float(df["low"].tail(60).min()),
           float(df["close"].iloc[-1]))
H, L, C = hlc
piv = _fib_pivots(H, L, C)
P, R1, R2, S1, S2 = piv["P"], piv["R1"], piv.get("R2"), piv["S1"], piv.get("S2")

# буфер около уровней
tol_k = {"ST":0.18, "MID":0.22, "LT":0.28}[hz]
buf = tol_k * (atr_w if hz != "ST" else atr_d)

def _near_from_below(level: float) -> bool:
    return (level is not None) and (0 <= level - price <= buf)

def _near_from_above(level: float) -> bool:
    return (level is not None) and (0 <= price - level <= buf)

# guard: длинные серии HA/MACD у кромок -> WAIT
thr_ha   = {"ST":4, "MID":5, "LT":6}[hz]
thr_macd = {"ST":4, "MID":6, "LT":8}[hz]
long_up   = (ha_up_run >= thr_ha)  or (macd_pos_run >= thr_macd)
long_down = (ha_down_run >= thr_ha) or (macd_neg_run >= thr_macd)

if long_up and (_near_from_below(S1) or _near_from_below(R1) or _near_from_below(R2)):
    action, scenario = "WAIT", "stall_after_long_up_at_pivot"
elif long_down and (_near_from_above(R1) or _near_from_above(S1) or _near_from_above(S2)):
    action, scenario = "WAIT", "stall_after_long_down_at_pivot"
else:
    band = _classify_band(price, piv, buf)
    very_high_pos = pos >= 0.80
    if very_high_pos:
        if (R2 is not None) and (price > R2 + 0.6*buf) and (slope_norm > 0):
            action, scenario = "BUY", "breakout_up"
        else:
            action, scenario = "SHORT", "fade_top"
    else:
        if band >= +2:
            action, scenario = ("BUY","breakout_up") if ((R2 is not None) and (price > R2 + 0.6*buf) and (slope_norm > 0)) else ("SHORT","fade_top")
        elif band == +1:
            action, scenario = "WAIT", "upper_wait"
        elif band == 0:
            action, scenario = ("BUY","trend_follow") if slope_norm >= 0 else ("WAIT","mid_range")
        elif band == -1:
            action, scenario = "BUY","revert_from_bottom"
        else:
            action, scenario = ("BUY","revert_from_bottom") if band <= -2 else ("WAIT","upper_wait")

# базовая уверенность
base = 0.55 + 0.12*_clip01(abs(slope_norm)*1800) + 0.08*_clip01((vol_ratio-0.9)/0.6)
if action == "WAIT":
    base -= 0.07
conf = float(max(0.55, min(0.90, base)))

# ===== optional AI override =====
try:
    from core.ai_inference import score_signal
except Exception:
    score_signal = None

if score_signal is not None:
    feats = dict(
        pos=pos, slope_norm=slope_norm,
        atr_d_over_price=(atr_d / max(1e-9, price)),
        vol_ratio=vol_ratio,
        ha_up_run=float(ha_up_run), ha_down_run=float(ha_down_run),
        macd_pos_run=float(macd_pos_run), macd_neg_run=float(macd_neg_run),
        band=float(_classify_band(price, piv, buf)),
        long_upper=bool(long_upper), long_lower=bool(long_lower),
    )
    out_ai = score_signal(feats, hz=hz, ticker=ticker)
    if out_ai is not None:
        p_long  = float(out_ai.get("p_long", 0.5))
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

# уровни (черновик)
step_d, step_w = atr_d, atr_w
if action == "BUY":
    if price < P:
        entry = max(price, S1 + 0.15*step_w); sl = S1 - 0.60*step_w
    else:
        entry = max(price, P + 0.10*step_w);  sl = P - 0.60*step_w
    tp1 = entry + 0.9*step_w; tp2 = entry + 1.6*step_w; tp3 = entry + 2.3*step_w
    alt = "Если продавят ниже и не вернут — не заходим; ждём возврата и подтверждения сверху."
elif action == "SHORT":
    if price >= R1:
        entry = min(price, R1 - 0.15*step_w); sl = R1 + 0.60*step_w
    else:
        entry = price + 0.10*step_d;         sl = price + 1.00*step_d
    tp1 = entry - 0.9*step_w; tp2 = entry - 1.6*step_w; tp3 = entry - 2.3*step_w
    alt = "Если протолкнут выше и удержат — без погони; ждём возврата и слабости сверху."
else:  # WAIT
    entry, sl = price, price - 0.90*step_d
    tp1, tp2, tp3 = entry + 0.7*step_d, entry + 1.4*step_d, entry + 2.1*step_d
    alt = "Ниже уровня — не пытаюсь догонять; стратегия — ждать пробоя, ретеста или отката к поддержке."

# трендовые «предохранители» целей (интуитивные)
tp1, tp2, tp3 = _clamp_tp_by_trend(action, hz, tp1, tp2, tp3, piv, step_w, slope_norm, macd_pos_run, macd_neg_run)

# TP floors + порядок
atr_for_floor = atr_w if hz != "ST" else atr_d
tp1, tp2, tp3 = _apply_tp_floors(entry, sl, tp1, tp2, tp3, action, hz, price, atr_for_floor)
tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)

# финальная sanity-проверка сторон/зазоров
sl, tp1, tp2, tp3 = _sanity_levels(action, entry, sl, tp1, tp2, tp3, price, step_d, step_w, hz)

# тип входа (STOP/LIMIT/NOW) — для UI
entry_kind = _entry_kind(action, entry, price, step_d)
entry_label = {
    "buy-stop":"Buy STOP", "buy-limit":"Buy LIMIT", "buy-now":"Buy NOW",
    "sell-stop":"Sell STOP","sell-limit":"Sell LIMIT","sell-now":"Sell NOW"
}.get(entry_kind, "")

# контекст-чипсы
chips = []
if vol_ratio > 1.05: chips.append("волатильность растёт")
if vol_ratio < 0.95: chips.append("волатильность сжимается")
if (ha_up_run >= thr_ha):   chips.append(f"HA зелёных: {ha_up_run}")
if (ha_down_run >= thr_ha): chips.append(f"HA красных: {ha_down_run}")

# вероятности
p1 = _clip01(0.58 + 0.27*(conf - 0.55)/0.35)
p2 = _clip01(0.44 + 0.21*(conf - 0.55)/0.35)
p3 = _clip01(0.28 + 0.13*(conf - 0.55)/0.35)

# «живой» текст
seed = int(hashlib.sha1(f"{ticker}{df.index[-1].date()}".encode()).hexdigest(), 16) % (2**32)
rng = random.Random(seed)
if action == "WAIT":
    lead = rng.choice(["Под кромкой — жду пробой/ретест.", "Импульс длинный — не гонюсь.", "Даём цене определиться."])
elif action == "BUY":
    lead = rng.choice(["Опора близко — беру по ходу после паузы.", "Спрос живой — вход от поддержки.", "Восстановление держится — беру аккуратно."])
else:
    lead = rng.choice(["Слабость у кромки — работаю от отказа.", "Под потолком тяжело — шорт со стопом.", "Импульс выдыхается — фиксирую вниз."])
note_html = f"<div style='margin-top:10px; opacity:0.95;'>{lead}</div>"

return {
    "last_price": float(price),
    "recommendation": {"action": action, "confidence": float(round(conf, 4))},
    "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
    "probs": {"tp1": float(p1), "tp2": float(p2), "tp3": float(p3)},
    "context": chips,
    "note_html": note_html,
    "alt": alt,
    "entry_kind": entry_kind,
    "entry_label": entry_label,
}
```

# –––––––––– Universal Strategy (400+ trades analysis) ––––––––––

class TimeWeightedStrategy:
“””
Гибкая стратегия с весами по времени
Не жесткие рамки, а плавные переходы
“””

```
def __init__(self):
    # Временные зоны с весами (от 0.0 до 1.0)
    self.time_weights = {
        # ВХОДНЫЕ ОКНА
        'entry_prime': {     # Основное время входа
            'start': time(9, 45),   # 9:45 AM EST/EDT
            'peak': time(10, 15),   # 10:15 AM (пик)
            'end': time(11, 45),    # 11:45 AM
            'weight_multiplier': 1.0
        },
        'entry_lunch': {     # Дообеденные входы
            'start': time(12, 0),   # 12:00 PM EST/EDT
            'peak': time(12, 30),   # 12:30 PM
            'end': time(13, 30),    # 1:30 PM
            'weight_multiplier': 0.6
        },
        'entry_scalp':
```
