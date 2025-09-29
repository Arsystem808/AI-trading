# core/strategy.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Callable
import logging

# -------------------- инфраструктура --------------------
try:
    from core.polygon_client import PolygonClient
except Exception:
    class PolygonClient:
        def daily_ohlc(self, ticker, days=120):
            raise RuntimeError("PolygonClient unavailable")
        def last_trade_price(self, ticker):
            raise RuntimeError("PolygonClient unavailable")

try:
    from core.performance_tracker import log_agent_performance, get_agent_performance
except Exception:
    def log_agent_performance(*args, **kwargs): pass
    def get_agent_performance(*args, **kwargs): return None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- small utils --------------------
def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _linreg_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2: return 0.0
    x = np.arange(n, dtype=float); xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0: return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)

# -------------------- ATR --------------------
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

# -------------------- Heikin Ashi & MACD --------------------
def _heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha = pd.DataFrame(index=df.index.copy())
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2.0]
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + ha["ha_close"].iloc[i-1]) / 2.0)
    ha["ha_open"] = pd.Series(ha_open, index=df.index)
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

# -------------------- horizons (для W7) --------------------
def _horizon_cfg(text: str):
    if "Кратко" in text:  return dict(look=60, trend=14, atr=14, pivot_rule="W-FRI", use_weekly_atr=False, hz="ST")
    if "Средне" in text:  return dict(look=120, trend=28, atr=14, pivot_rule="M",     use_weekly_atr=True,  hz="MID")
    return dict(look=240, trend=56, atr=14, pivot_rule="M", use_weekly_atr=True, hz="LT")

def _last_period_hlc(df: pd.DataFrame, rule: str):
    g = df.resample(rule).agg({"high":"max","low":"min","close":"last"}).dropna()
    if len(g) < 2: return None
    row = g.iloc[-2]
    return float(row["high"]), float(row["low"]), float(row["close"])

def _fib_pivots(H: float, L: float, C: float):
    P = (H + L + C) / 3.0; d = H - L
    return {"P":P, "R1":P + 0.382*d, "R2":P + 0.618*d, "R3":P + 1.000*d, "S1":P - 0.382*d, "S2":P - 0.618*d, "S3":P - 1.000*d}

def _classify_band(price: float, piv: dict, buf: float) -> int:
    P, R1 = piv["P"], piv["R1"]; R2, R3, S1, S2 = piv.get("R2"), piv.get("R3"), piv["S1"], piv.get("S2")
    neg_inf, pos_inf = -1e18, 1e18
    levels = [S2 if S2 is not None else neg_inf, S1, P, R1, R2 if R2 is not None else pos_inf, R3 if R3 is not None else pos_inf]
    if price < levels[0] - buf: return -3
    if price < levels[1] - buf: return -2
    if price < levels[2] - buf: return -1
    if price < levels[3] - buf: return 0
    if R2 is None or price < levels[4] - buf: return +1
    if price < levels[5] - buf: return +2
    return +3

# -------------------- wick profile --------------------
def _wick_profile(row: pd.Series):
    o, c, h, l = float(row["open"]), float(row["close"]), float(row["high"]), float(row["low"])
    body = max(1e-9, abs(c - o)); up_wick = max(0.0, h - max(o, c)); dn_wick = max(0.0, min(o, c) - l)
    return body, up_wick, dn_wick

# -------------------- Order kind --------------------
def _entry_kind(action: str, entry: float, price: float, step_d: float) -> str:
    tol = max(0.0015 * max(1.0, price), 0.15 * step_d)
    if action == "BUY":
        if entry > price + tol:  return "buy-stop"
        if entry < price - tol:  return "buy-limit"
        return "buy-now"
    if action == "SHORT":
        if entry < price - tol:  return "sell-stop"
        if entry > price + tol:  return "sell-limit"
        return "sell-now"
    return "wait"

# -------------------- TP/SL helpers --------------------
def _apply_tp_floors(entry: float, sl: float, tp1: float, tp2: float, tp3: float,
                     action: str, hz_tag: str, price: float, atr_val: float):
    if action not in ("BUY", "SHORT"): return tp1, tp2, tp3
    risk = abs(entry - sl)
    if risk <= 1e-9: return tp1, tp2, tp3
    side = 1 if action == "BUY" else -1
    min_rr  = {"ST": 0.80, "MID": 1.00, "LT": 1.20}
    min_pct = {"ST": 0.006, "MID": 0.012, "LT": 0.018}
    atr_mult = {"ST": 0.50, "MID": 0.80, "LT": 1.20}
    floor1 = max(min_rr[hz_tag]*risk, min_pct[hz_tag]*price, atr_mult[hz_tag]*atr_val)
    if abs(tp1 - entry) < floor1: tp1 = entry + side*floor1
    floor2 = max(1.6*floor1, min_rr[hz_tag]*1.8*risk)
    if abs(tp2 - entry) < floor2: tp2 = entry + side*floor2
    min_gap3 = max(0.8*floor1, 0.6*risk)
    if abs(tp3 - tp2) < min_gap3: tp3 = tp2 + side*min_gap3
    return tp1, tp2, tp3

def _order_targets(entry: float, tp1: float, tp2: float, tp3: float, action: str, eps: float = 1e-6):
    side = 1 if action == "BUY" else -1
    arr = sorted([float(tp1), float(tp2), float(tp3)], key=lambda x: side * (x - entry))
    d0 = side*(arr[0]-entry); d1 = side*(arr[1]-entry); d2 = side*(arr[2]-entry)
    if d1 - d0 < eps: arr[1] = entry + side * max(d0 + max(eps, 0.1*abs(d0)), d1 + eps)
    if side*(arr[2]-entry) - side*(arr[1]-entry) < eps:
        d1 = side*(arr[1]-entry); arr[2] = entry + side * max(d1 + max(eps, 0.1*abs(d1)), d2 + eps)
    return arr[0], arr[1], arr[2]

def _clamp_tp_by_trend(action: str, hz: str,
                       tp1: float, tp2: float, tp3: float,
                       piv: dict, step_w: float,
                       slope_norm: float, macd_pos_run: int, macd_neg_run: int):
    thr_macd = {"ST": 3, "MID": 5, "LT": 6}[hz]
    bullish = (slope_norm > 0.0006) and (macd_pos_run >= thr_macd)
    bearish = (slope_norm < -0.0006) and (macd_neg_run >= thr_macd)
    P, R1, S1 = piv["P"], piv["R1"], piv["S1"]
    if action == "SHORT" and bullish:
        limit = max(R1 - 1.2*step_w, (P + R1)/2.0)
        tp1 = max(tp1, limit - 0.2*step_w); tp2 = max(tp2, limit); tp3 = max(tp3, limit + 0.4*step_w)
    if action == "BUY" and bearish:
        limit = min(S1 + 1.2*step_w, (P + S1)/2.0)
        tp1 = min(tp1, limit + 0.2*step_w); tp2 = min(tp2, limit); tp3 = min(tp3, limit - 0.4*step_w)
    return tp1, tp2, tp3

def _sanity_levels(action: str, entry: float, sl: float,
                   tp1: float, tp2: float, tp3: float,
                   price: float, step_d: float, step_w: float, hz: str):
    side = 1 if action == "BUY" else -1
    min_tp_gap = {"ST": 0.40, "MID": 0.70, "LT": 1.10}[hz] * step_w
    min_tp_pct = {"ST": 0.004, "MID": 0.009, "LT": 0.015}[hz] * price
    floor_gap = max(min_tp_gap, min_tp_pct, 0.35*abs(entry - sl) if sl != entry else 0.0)
    if action == "BUY"  and sl >= entry - 0.25*step_d: sl = entry - max(0.60*step_w, 0.90*step_d)
    if action == "SHORT" and sl <= entry + 0.25*step_d: sl = entry + max(0.60*step_w, 0.90*step_d)
    def _push(tp, rank):
        need = floor_gap * (1.0 if rank==1 else (1.6 if rank==2 else 2.2))
        want = entry + side*need
        if side*(tp-entry) <= 0: return want
        if abs(tp-entry) < need: return want
        return tp
    tp1 = _push(tp1, 1); tp2 = _push(tp2, 2); tp3 = _push(tp3, 3)
    tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)
    return sl, tp1, tp2, tp3

# -------------------- Калибровка и ECE --------------------
class ConfidenceCalibrator:
    def __init__(self, method: str = "identity", params: Optional[Dict[str, float]] = None):
        self.method = method
        self.params = params or {}
    def __call__(self, p: float) -> float:
        p = _clip01(float(p))
        if self.method == "sigmoid":
            a = float(self.params.get("a", 1.0)); b = float(self.params.get("b", 0.0))
            return _clip01(float(1.0/(1.0+np.exp(-(a*p + b)))))
        if self.method == "isotonic":
            knots = sorted(self.params.get("knots", [(0.0,0.0),(1.0,1.0)]))
            for i in range(1, len(knots)):
                x0,y0 = knots[i-1]; x1,y1 = knots[i]
                if p <= x1:
                    if x1 == x0: return _clip01(float(y1))
                    t = (p - x0)/(x1 - x0)
                    return _clip01(float(y0 + t*(y1 - y0)))
            return _clip01(float(knots[-1][1]))
        return p

def _ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    N = len(labels); ece = 0.0
    for i in range(bins):
        mask = (probs > edges[i]) & (probs <= edges[i+1])
        if not np.any(mask): continue
        conf = float(probs[mask].mean()); acc = float((labels[mask] == 1).mean())
        ece += (mask.sum()/N) * abs(acc - conf)
    return float(ece)

def _monotone_tp_probs(probs: dict) -> dict:
    p1 = float(probs.get("tp1", 0.60))
    p2 = float(probs.get("tp2", 0.50))
    p3 = float(probs.get("tp3", 0.45))
    p1 = max(min(p1, 0.90), 0.35)
    p2 = max(min(min(p2, p1 - 0.03), 0.85), 0.30)
    p3 = max(min(min(p3, p2 - 0.03), 0.80), 0.25)
    return {"tp1": p1, "tp2": p2, "tp3": p3}

# -------------------- Global --------------------
def analyze_asset_global(ticker: str, horizon: str = "Краткосрочный"):
    cli = PolygonClient(); df = cli.daily_ohlc(ticker, days=90)
    current_price = float(df['close'].iloc[-1])
    returns = np.log(df['close']/df['close'].shift(1))
    hist_volatility = returns.std()*np.sqrt(252)
    short_ma = df['close'].rolling(20).mean().iloc[-1]
    long_ma  = df['close'].rolling(50).mean().iloc[-1]
    if short_ma > long_ma: action, confidence = "BUY", 0.69
    else:                  action, confidence = "SHORT", 0.65
    atr = float(_atr_like(df, n=14).iloc[-1])
    if action == "BUY":
        entry = current_price; sl = current_price - 2*atr
        tp1 = current_price + 1*atr; tp2 = current_price + 2*atr; tp3 = current_price + 3*atr
        alt = "Покупка по рынку с консервативными целями"
    else:
        entry = current_price; sl = current_price + 2*atr
        tp1 = current_price - 1*atr; tp2 = current_price - 2*atr; tp3 = current_price - 3*atr
        alt = "Продажа по рынку с консервативными целями"
    context = [f"Волатильность: {hist_volatility:.2%}", f"Тренд: {'Бычий' if action=='BUY' else 'Медвежий'}"]
    probs = _monotone_tp_probs({"tp1": 0.68, "tp2": 0.52, "tp3": 0.35})
    note_html = f"<div style='margin-top:10px; opacity:0.95;'>Global: {action} с уверенностью {confidence:.0%}.</div>"
    return {
        "last_price": current_price,
        "recommendation": {"action": action, "confidence": confidence},
        "levels": {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs, "context": context, "note_html": note_html, "alt": alt,
        "entry_kind": "market", "entry_label": f"{action} NOW",
        "meta": {"source":"Global"}
    }

# -------------------- M7 --------------------
class M7TradingStrategy:
    def __init__(self, atr_period=14, atr_multiplier=1.5, pivot_period='D', fib_levels=[0.236,0.382,0.5,0.618,0.786]):
        self.atr_period = atr_period; self.atr_multiplier = atr_multiplier
        self.pivot_period = pivot_period; self.fib_levels = fib_levels
    def calculate_pivot_points(self, h,l,c):
        pivot = (h + l + c) / 3
        r1 = (2 * pivot) - l; r2 = pivot + (h - l); r3 = h + 2 * (pivot - l)
        s1 = (2 * pivot) - h; s2 = pivot - (h - l); s3 = l - 2 * (h - pivot)
        return {'pivot': pivot, 'r1': r1, 'r2': r2, 'r3': r3, 's1': s1, 's2': s2, 's3': s3}
    def calculate_fib_levels(self, h,l):
        diff = h - l; fib = {}
        for level in self.fib_levels: fib[f'fib_{int(level*1000)}'] = h - level * diff
        return fib
    def identify_key_levels(self, data):
        grouped = data.resample('D') if self.pivot_period == 'D' else data.resample('W')
        key = {}
        for _, g in grouped:
            if len(g) > 0:
                h = g['high'].max(); l = g['low'].min(); c = g['close'].iloc[-1]
                key.update(self.calculate_pivot_points(h,l,c)); key.update(self.calculate_fib_levels(h,l))
        return key
    def generate_signals(self, data):
        sigs = []; req = ['high','low','close']
        if not all(c in data.columns for c in req): return sigs
        data['atr'] = _atr_like(data, self.atr_period); cur_atr = data['atr'].iloc[-1]
        key = self.identify_key_levels(data); price = data['close'].iloc[-1]; ts = data.index[-1]
        for name, val in key.items():
            dist = abs(price - val) / max(1e-9, cur_atr)
            if dist < self.atr_multiplier:
                is_res = val > price
                if is_res: typ='SELL_LIMIT'; entry=val*0.998; sl=val*1.02; tp=val*0.96
                else:      typ='BUY_LIMIT';  entry=val*1.002; sl=val*0.98; tp=val*1.04
                conf = 1 - (dist / self.atr_multiplier)
                sigs.append({'type':typ,'price':round(entry,4),'stop_loss':round(sl,4),'take_profit':round(tp,4),
                             'confidence':round(conf,3),'level':name,'level_value':round(val,4),'timestamp':ts})
        return sigs

def analyze_asset_m7(ticker, horizon="Краткосрочный", use_ml=False):
    cli = PolygonClient(); df = cli.daily_ohlc(ticker, days=120)
    strategy = M7TradingStrategy(); signals = strategy.generate_signals(df)
    price = float(df['close'].iloc[-1])

    # честный WAIT, если сигналов нет
    if not signals:
        return {
            "last_price": price,
            "recommendation": {"action":"WAIT","confidence":0.50},
            "levels":{"entry":0.0,"sl":0.0,"tp1":0.0,"tp2":0.0,"tp3":0.0},
            "probs":{"tp1":0.0,"tp2":0.0,"tp3":0.0},
            "context":["M7: сигналов нет"], "note_html":"<div>M7: ожидание</div>",
            "alt":"Ожидание сигналов", "entry_kind":"wait","entry_label":"WAIT",
            "meta":{"source":"M7","grey_zone":True}
        }

    # нейтральная зона по «качеству уровня»
    best = max(signals, key=lambda x: x['confidence'])
    raw = float(_clip01(best['confidence']))  # 0..1
    if raw < 0.55:
        return {
            "last_price": price,
            "recommendation": {"action":"WAIT","confidence":0.50},
            "levels":{"entry":0.0,"sl":0.0,"tp1":0.0,"tp2":0.0,"tp3":0.0},
            "probs":{"tp1":0.0,"tp2":0.0,"tp3":0.0},
            "context":[f"M7: нейтральная зона (raw={raw:.2f})"], "note_html":"<div>M7: WAIT (нейтральная зона)</div>",
            "alt":"WAIT","entry_kind":"wait","entry_label":"WAIT",
            "meta":{"source":"M7","grey_zone":True}
        }

    # уровни и направление
    entry = float(best['price']); sl = float(best['stop_loss']); risk = abs(entry - sl)
    vol = df['close'].pct_change().std() * np.sqrt(252)
    max_move = price * vol / np.sqrt(252)
    if best['type'].startswith('BUY'):
        tp1 = min(entry + 1.5*risk, entry + 2*max_move)
        tp2 = min(entry + 2.5*risk, entry + 3*max_move)
        tp3 = min(entry + 4.0*risk, entry + 5*max_move)
        act = "BUY"
    else:
        tp1 = max(entry - 1.5*risk, entry - 2*max_move)
        tp2 = max(entry - 2.5*risk, entry - 3*max_move)
        tp3 = max(entry - 4.0*risk, entry - 5*max_move)
        act = "SHORT"

    # честная шкала confidence (сигмоида + мягкие штрафы)
    import math
    base = 0.50 + 0.38 * math.tanh((raw - 0.55) / 0.20)
    penalty = 0.0
    if vol > 0.40: penalty += 0.05
    conf = max(0.52, min(0.85, base * (1.0 - penalty)))

    probs = _monotone_tp_probs({"tp1": 0.63, "tp2": 0.52, "tp3": 0.47})

    return {
        "last_price": price,
        "recommendation": {"action": act, "confidence": float(conf)},
        "levels": {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs,
        "context": [f"M7: {best['type']} @ {best['level']} (raw={raw:.2f})"],
        "note_html": f"<div>M7: {best['type']} на уровне {best['level_value']}</div>",
        "alt": "Торговля по M7", "entry_kind": "limit", "entry_label": best['type'],
        "meta":{"source":"M7","grey_zone": bool(0.48 <= conf <= 0.58)}
    }

# -------------------- W7 --------------------
def analyze_asset_w7(ticker: str, horizon: str):
    cli = PolygonClient(); cfg = _horizon_cfg(horizon); hz = cfg["hz"]
    days = max(90, cfg["look"] * 2); df = cli.daily_ohlc(ticker, days=days); price = cli.last_trade_price(ticker)
    closes = df["close"]; tail = df.tail(cfg["look"])
    rng_low, rng_high = float(tail["low"].min()), float(tail["high"].max())
    rng_w = max(1e-9, rng_high - rng_low); pos = (price - rng_low) / rng_w
    slope = _linreg_slope(closes.tail(cfg["trend"]).values); slope_norm = slope / max(1e-9, price)
    atr_d = float(_atr_like(df, n=cfg["atr"]).iloc[-1]); atr_w = _weekly_atr(df) if cfg.get("use_weekly_atr") else atr_d
    vol_ratio = atr_d / max(1e-9, float(_atr_like(df, n=cfg["atr"] * 2).iloc[-1]))
    ha = _heikin_ashi(df); ha_diff = ha["ha_close"].diff()
    ha_up_run = _streak_by_sign(ha_diff, True); ha_down_run = _streak_by_sign(ha_diff, False)
    _, _, hist = _macd_hist(closes); macd_pos_run = _streak_by_sign(hist, True); macd_neg_run = _streak_by_sign(hist, False)
    last_row = df.iloc[-1]; body, up_wick, dn_wick = _wick_profile(last_row)
    hlc = _last_period_hlc(df, cfg["pivot_rule"]) or (float(df["high"].tail(60).max()), float(df["low"].tail(60).min()), float(df["close"].iloc[-1]))
    H, L, C = hlc; piv = _fib_pivots(H, L, C); P, R1, R2, S1, S2 = piv["P"], piv["R1"], piv.get("R2"), piv["S1"], piv.get("S2")
    tol_k = {"ST": 0.18, "MID": 0.22, "LT": 0.28}[hz]; buf = tol_k * (atr_w if hz != "ST" else atr_d)

    def _near_from_below(level: float) -> bool: return (level is not None) and (0 <= level - price <= buf)
    def _near_from_above(level: float) -> bool: return (level is not None) and (0 <= price - level <= buf)

    thr_ha = {"ST": 4, "MID": 5, "LT": 6}[hz]; thr_macd = {"ST": 4, "MID": 6, "LT": 8}[hz]
    long_up   = (ha_up_run >= thr_ha)  or (macd_pos_run >= thr_macd)
    long_down = (ha_down_run >= thr_ha) or (macd_neg_run >= thr_macd)

    if long_up and (_near_from_below(S1) or _near_from_below(R1) or _near_from_below(R2)):
        action = "WAIT"
    elif long_down and (_near_from_above(R1) or _near_from_above(S1) or _near_from_above(S2)):
        action = "WAIT"
    else:
        band = _classify_band(price, piv, buf); very_high_pos = pos >= 0.80
        if very_high_pos:
            action = "BUY" if (R2 is not None and price > R2 + 0.6*buf and slope_norm > 0) else "SHORT"
        else:
            if band >= +2:
                action = "BUY" if (R2 is not None and price > R2 + 0.6*buf and slope_norm > 0) else "SHORT"
            elif band == +1:
                action = "WAIT"
            elif band == 0:
                action = "BUY" if slope_norm >= 0 else "WAIT"
            elif band == -1:
                action = "BUY"
            else:
                action = "BUY" if band <= -2 else "WAIT"

    base = 0.55 + 0.12*_clip01(abs(slope_norm)*1800) + 0.08*_clip01((vol_ratio - 0.9)/0.6)
    if action == "WAIT": base -= 0.07
    conf = float(max(0.55, min(0.90, base)))

    step_d, step_w = atr_d, atr_w
    if action == "BUY":
        if price < P:  entry = max(price, S1 + 0.15*step_w); sl = S1 - 0.60*step_w
        else:          entry = max(price, P  + 0.10*step_w); sl = P  - 0.60*step_w
        tp1 = entry + 0.9*step_w; tp2 = entry + 1.6*step_w; tp3 = entry + 2.3*step_w
        alt = "Если продавят ниже и не вернут — ждём возврата"
    elif action == "SHORT":
        if price >= R1: entry = min(price, R1 - 0.15*step_w); sl = R1 + 0.60*step_w
        else:           entry = price + 0.10*step_d;          sl = price + 1.00*step_d
        tp1 = entry - 0.9*step_w; tp2 = entry - 1.6*step_w; tp3 = entry - 2.3*step_w
        alt = "Если протолкнут выше и удержат — без погони"
    else:
        entry, sl = price, price - 0.90*step_d
        tp1, tp2, tp3 = entry + 0.7*step_d, entry + 1.4*step_d, entry + 2.1*step_d
        alt = "Ждать пробоя/ретеста"

    tp1, tp2, tp3 = _clamp_tp_by_trend(action, hz, tp1, tp2, tp3, piv, step_w, slope_norm, macd_pos_run, macd_neg_run)
    atr_for_floor = atr_w if hz != "ST" else atr_d
    tp1, tp2, tp3 = _apply_tp_floors(entry, sl, tp1, tp2, tp3, action, hz, price, atr_for_floor)
    tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)
    sl,  tp1, tp2, tp3 = _sanity_levels(action, entry, sl, tp1, tp2, tp3, price, step_d, step_w, hz)

    entry_kind = _entry_kind(action, entry, price, step_d)
    entry_label = {"buy-stop":"Buy STOP","buy-limit":"Buy LIMIT","buy-now":"Buy NOW",
                   "sell-stop":"Sell STOP","sell-limit":"Sell LIMIT","sell-now":"Sell NOW"}.get(entry_kind, "")

    probs = _monotone_tp_probs({
        "tp1": float(_clip01(0.58 + 0.27*(conf - 0.55)/0.35)),
        "tp2": float(_clip01(0.44 + 0.21*(conf - 0.55)/0.35)),
        "tp3": float(_clip01(0.28 + 0.13*(conf - 0.55)/0.35))
    })

    return {
        "last_price": float(price),
        "recommendation": {"action": action, "confidence": float(round(conf, 4))},
        "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
        "probs": probs, "context": [], "note_html": "<div>W7: контекст по волатильности и зонам</div>",
        "alt": alt, "entry_kind": entry_kind, "entry_label": entry_label,
        "meta":{"source":"W7","grey_zone": bool(0.48 <= conf <= 0.58)}
    }

# -------------------- AlphaPulse (Mean‑Reversion) --------------------
def analyze_asset_alphapulse(ticker: str, horizon: str = "Краткосрочный") -> Dict[str, Any]:
    def _safe_load_ohlc(sym: str, days: int):
        try:
            from services.data import load_ohlc as _ld
            return _ld(sym, days)
        except Exception:
            pass
        try:
            from core.data import load_ohlc as _ld
            return _ld(sym, days)
        except Exception:
            pass
        try:
            cli = PolygonClient()
            return cli.daily_ohlc(sym, days=days)
        except Exception:
            return None

    try:
        from core.mean_reversion_signal_engine import MeanReversionSignalEngine
    except Exception as e:
        return {
            "last_price": 0.0,
            "recommendation":{"action":"WAIT","confidence":0.50},
            "levels":{"entry":0.0,"sl":0.0,"tp1":0.0,"tp2":0.0,"tp3":0.0},
            "probs":{"tp1":0.0,"tp2":0.0,"tp3":0.0},
            "context":[f"AlphaPulse недоступен: {type(e).__name__}"],
            "note_html":"<div>AlphaPulse: модуль недоступен</div>",
            "alt":"WAIT","entry_kind":"wait","entry_label":"WAIT",
            "meta":{"source":"AlphaPulse","grey_zone":True,"unavailable":True}
        }

    df = _safe_load_ohlc(ticker, days=240)
    price = float(df["close"].iloc[-1]) if (df is not None and len(df)) else 0.0
    if df is None or len(df) < 50:
        return {
            "last_price": price,
            "recommendation":{"action":"WAIT","confidence":0.50},
            "levels":{"entry":0.0,"sl":0.0,"tp1":0.0,"tp2":0.0,"tp3":0.0},
            "probs":{"tp1":0.0,"tp2":0.0,"tp3":0.0},
            "context":["AlphaPulse: недостаточно данных"], "note_html":"<div>AlphaPulse: ожидание</div>",
            "alt":"WAIT","entry_kind":"wait","entry_label":"WAIT",
            "meta":{"source":"AlphaPulse","grey_zone":True}
        }

    eng = MeanReversionSignalEngine()
    sigs = eng.generate_signals(df, ticker)
    if getattr(sigs, "empty", True):
        return {
            "last_price": price,
            "recommendation":{"action":"WAIT","confidence":0.50},
            "levels":{"entry":0.0,"sl":0.0,"tp1":0.0,"tp2":0.0,"tp3":0.0},
            "probs":{"tp1":0.0,"tp2":0.0,"tp3":0.0},
            "context":["AlphaPulse: сигналов нет"], "note_html":"<div>AlphaPulse: ожидание</div>",
            "alt":"WAIT","entry_kind":"wait","entry_label":"WAIT",
            "meta":{"source":"AlphaPulse","grey_zone":True}
        }

    row = sigs.iloc[-1]
    side = "BUY" if str(row.get("side","")).upper() in ("LONG","BUY") else "SHORT"
    base_conf = float(max(0.0, min(1.0, float(row.get("confidence", 50.0))/100.0)))
    cal = ConfidenceCalibrator(method="sigmoid", params={"a":1.1,"b":-0.05})
    conf = float(min(0.88, max(0.50, cal(0.50 + (base_conf - 0.5)))))

    sl_d = float(row.get("sl_distance", 0.0)); tp_d = float(row.get("tp_distance", 0.0))
    entry = float(row.get("price", price))
    if side == "BUY":
        sl  = max(0.0, entry - sl_d); tp1, tp2, tp3 = entry + tp_d, entry + 1.8*tp_d, entry + 2.6*tp_d
    else:
        sl  = entry + sl_d;         tp1, tp2, tp3 = entry - tp_d, entry - 1.8*tp_d, entry - 2.6*tp_d

    return {
        "last_price": price,
        "recommendation":{"action": side, "confidence": conf},
        "levels":{"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": _monotone_tp_probs({"tp1":0.60,"tp2":0.50,"tp3":0.45}),
        "context":[f"AlphaPulse MR: side={side}, base_conf={base_conf:.2f}"],
        "note_html":"<div>AlphaPulse (Mean‑Reversion): сигнал сформирован</div>",
        "alt":"Mean‑Reversion","entry_kind":"limit","entry_label": side,
        "meta":{"source":"AlphaPulse","grey_zone": bool(0.48 <= conf <= 0.58)}
    }

# -------------------- Оркестратор Octopus --------------------
OCTO_WEIGHTS: Dict[str, float] = {"Global": 0.28, "M7": 0.26, "W7": 0.26, "AlphaPulse": 0.20}
OCTO_CALIBRATOR = ConfidenceCalibrator(method="sigmoid", params={"a": 1.2, "b": -0.10})

def _act_to_num(a: str) -> int: return 1 if a == "BUY" else (-1 if a == "SHORT" else 0)
def _num_to_act(x: float) -> str:
    if x > 0: return "BUY"
    if x < 0: return "SHORT"
    return "WAIT"

def analyze_asset_octopus(ticker: str, horizon: str) -> Dict[str, Any]:
    parts: Dict[str, Dict[str,Any]] = {}
    for name, fn in {
        "Global": analyze_asset_global,
        "M7": analyze_asset_m7,
        "W7": analyze_asset_w7,
        "AlphaPulse": analyze_asset_alphapulse,
    }.items():
        try:
            parts[name] = fn(ticker, horizon)
        except Exception as e:
            logger.warning("Agent %s failed: %s", name, e)

    if not parts:
        res = {
            "strategy":"Octopus","agents":[], "weights": OCTO_WEIGHTS, "signals":{},
            "action":"WAIT","confidence_raw":0.0,"confidence":0.0,
            "calibration":{"method":OCTO_CALIBRATOR.method,"params":OCTO_CALIBRATOR.params,"ece":0.0},
            "levels":{"entry":0.0,"sl":0.0,"tp1":0.0,"tp2":0.0,"tp3":0.0},
            "last_price":0.0, "context":["Octopus: нет данных по агентам"],
            "note_html":"<div>Octopus: WAIT</div>","alt":"Octopus","entry_kind":"wait","entry_label":"WAIT",
            "probs":{"tp1":0.0,"tp2":0.0,"tp3":0.0}, "meta":{"grey_zone":True,"source":"Octopus"}
        }
        res["recommendation"] = {"action": res["action"], "confidence": res["confidence"]}
        return res

    # собрать активные голоса
    active = []
    for name, res in parts.items():
        rec = res.get("recommendation", {}); act = str(rec.get("action","WAIT")).upper()
        conf = float(rec.get("confidence", 0.0)); w = float(OCTO_WEIGHTS.get(name, 0.0))
        if act in ("BUY","SHORT"):
            active.append((name, act, conf, w))

    # если нет активных — WAIT
    if not active:
        last_price = float(parts.get("W7", parts.get("Global", {})).get("last_price", 0.0))
        signals = {n: {"action": parts[n]["recommendation"]["action"],
                       "confidence": float(parts[n]["recommendation"]["confidence"])} for n in parts.keys()}
        res = {
            "strategy":"Octopus","agents": list(parts.keys()), "weights": OCTO_WEIGHTS,
            "signals": signals, "action":"WAIT","confidence_raw":0.0,"confidence":0.0,
            "calibration":{"method":OCTO_CALIBRATOR.method,"params":OCTO_CALIBRATOR.params,"ece":0.0},
            "levels":{"entry":0.0,"sl":0.0,"tp1":0.0,"tp2":0.0,"tp3":0.0},
            "last_price": last_price, "context": ["Octopus: все WAIT"],
            "note_html":"<div>Octopus: все агенты WAIT</div>","alt":"Octopus","entry_kind":"wait","entry_label":"WAIT",
            "probs":{"tp1":0.0,"tp2":0.0,"tp3":0.0}, "meta":{"grey_zone":True,"source":"Octopus"}
        }
        res["recommendation"] = {"action": "WAIT", "confidence": 0.0}
        return res

    # взвешенные скоры направлений
    score_long  = sum(w * _clip01(c) for (n,a,c,w) in active if a == "BUY")
    score_short = sum(w * _clip01(c) for (n,a,c,w) in active if a == "SHORT")
    total_side  = score_long + score_short
    delta = abs(score_long - score_short)
    disagreement = 0.0 if total_side <= 1e-9 else float(max(0.0, 1.0 - (delta / total_side)))

    CONSENSUS_DELTA = 0.12
    if delta < CONSENSUS_DELTA:
        final_action = "WAIT"
        levels_out = {"entry":0.0,"sl":0.0,"tp1":0.0,"tp2":0.0,"tp3":0.0}
        probs_out  = {"tp1":0.0,"tp2":0.0,"tp3":0.0}
        raw_conf = 0.50
    else:
        final_action = "BUY" if score_long > score_short else "SHORT"
        # победитель по направлению
        cand = [t for t in active if t[1] == final_action]
        win_name = max(cand, key=lambda t: t[2]*t[3])[0] if cand else max(active, key=lambda t: t[2]*t[3])[0]
        lv = parts[win_name].get("levels", {}) or {}
        levels_out = {
            "entry": float(lv.get("entry", 0.0)), "sl": float(lv.get("sl", 0.0)),
            "tp1": float(lv.get("tp1", 0.0)), "tp2": float(lv.get("tp2", 0.0)), "tp3": float(lv.get("tp3", 0.0))
        }
        probs_out = _monotone_tp_probs(parts[win_name].get("probs", {}) or {})
        # сырая уверенность — средневзвешенная по активным
        w_sum = sum(t[3] for t in active) or 1.0
        raw_conf = float(sum(t[3]*_clip01(t[2]) for t in active) / w_sum)

    calibrated = float(OCTO_CALIBRATOR(raw_conf))
    overall_conf = float(max(0.52, min(0.86, calibrated * (1.0 - 0.35*disagreement))))

    # собрать сигнал
    signals = {n: {"action": parts[n]["recommendation"]["action"],
                   "confidence": float(parts[n]["recommendation"]["confidence"])}
               for n in parts.keys()}
    last_price = float(parts.get("M7", parts.get("W7", parts.get("Global", {}))).get("last_price", 0.0))
    ctx = [f"Octopus: consensus delta={delta:.3f}, disagreement={disagreement:.3f}, action={final_action}"]
    res = {
        "strategy": "Octopus",
        "agents": list(parts.keys()), "weights": OCTO_WEIGHTS, "signals": signals,
        "action": final_action, "confidence_raw": float(raw_conf), "confidence": overall_conf,
        "calibration": {"method": OCTO_CALIBRATOR.method, "params": OCTO_CALIBRATOR.params,
                        "ece": float(_ece(np.array([_clip01(t[2]) for t in active]) if len(active)>=2 else np.array([0.5,0.5]),
                                          np.array([1 if t[1]=='BUY' else 0 for t in active]) if len(active)>=2 else np.array([0,1]), bins=10))},
        "levels": levels_out, "last_price": last_price, "context": ctx,
        "note_html": "<div>Octopus: ансамбль Global/M7/W7/AlphaPulse</div>",
        "alt": "Octopus", "entry_kind": "market" if final_action != "WAIT" else "wait",
        "entry_label": final_action if final_action != "WAIT" else "WAIT",
        "probs": probs_out, "meta": {"grey_zone": bool(0.48 <= overall_conf <= 0.58), "source": "Octopus"}
    }
    res["recommendation"] = {"action": res["action"], "confidence": res["confidence"]}
    return res

# -------------------- Strategy Router (для UI) --------------------
STRATEGY_REGISTRY = {
    "Octopus": analyze_asset_octopus,
    "Global": analyze_asset_global,
    "M7": analyze_asset_m7,
    "W7": analyze_asset_w7,
    "AlphaPulse": analyze_asset_alphapulse,
}

def analyze_asset(ticker: str, horizon: str, strategy: str = "Octopus") -> Dict[str, Any]:
    fn = STRATEGY_REGISTRY.get(strategy)
    if not fn:
        raise ValueError(f"Unknown strategy: {strategy}")
    return fn(ticker, horizon)
