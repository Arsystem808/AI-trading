# core/strategy.py
from __future__ import annotations

import inspect
import json
import math
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import numpy as np
import pandas as pd

# -------------------- Logging Setup --------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent duplicate logs

# -------------------- Infrastructure --------------------
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

# -------------------- PRODUCTION M7 (file-based import) --------------------
import importlib.util, os, sys

_m7_fallback_lock = threading.Lock()
_m7_fallback_logged = False

def _m7_fallback(ticker: str, horizon: str = "Краткосрочный", use_ml: bool = False) -> Dict[str, Any]:
    """Thread-safe fallback when production M7 is unavailable"""
    global _m7_fallback_logged
    if not _m7_fallback_logged:
        with _m7_fallback_lock:
            if not _m7_fallback_logged:
                logger.error("[M7] Module unavailable - using fallback WAIT")
                logger.info("[M7] Create core/strategies/m7.py with analyze_asset_m7 to enable")
                _m7_fallback_logged = True
    return {
        "last_price": 0.0,
        "recommendation": {"action": "WAIT", "confidence": 0.5},
        "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
        "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
        "context": ["[!] M7 module unavailable"],
        "note_html": "<div>M7: module error</div>",
        "alt": "M7 unavailable",
        "entry_kind": "wait",
        "entry_label": "WAIT",
        "meta": {
            "source": "M7",
            "error": "module_not_loaded",
            "grey_zone": True,
            "ml_used": False,
            "ml_requested": use_ml
        }
    }

# Default implementation until the module is loaded
analyze_asset_m7 = _m7_fallback

try:
    # Absolute path to core/strategies/m7.py
    m7_path = os.path.join(os.getcwd(), "core", "strategies", "m7.py")
    if not os.path.isfile(m7_path):
        raise FileNotFoundError(m7_path)

    # Load module from file to bypass broken package import
    spec = importlib.util.spec_from_file_location("core.strategies.m7", m7_path)
    if not spec or not spec.loader:
        raise ImportError("spec/loader is None for m7.py")
    m7_module = importlib.util.module_from_spec(spec)
    sys.modules["core.strategies.m7"] = m7_module
    spec.loader.exec_module(m7_module)

    # Get the function and validate
    fn = getattr(m7_module, "analyze_asset_m7", None)
    if not callable(fn):
        raise TypeError("analyze_asset_m7 is missing or not callable")

    sig = inspect.signature(fn)
    if "ticker" not in sig.parameters:
        raise TypeError("analyze_asset_m7 missing required parameter: ticker")

    analyze_asset_m7 = fn
    logger.info("[M7] Production module loaded via file import core/strategies/m7.py")

except Exception as e:
    logger.error(f"[M7] File import failed: {type(e).__name__}: {e}")
    analyze_asset_m7 = _m7_fallback

# Load production M7
analyze_asset_m7 = _m7_fallback  # Default

try:
    from core.strategies import m7 as m7_module
    
    if not hasattr(m7_module, 'analyze_asset_m7'):
        raise AttributeError("Module missing 'analyze_asset_m7' function")
    
    analyze_asset_m7 = m7_module.analyze_asset_m7
    
    if not callable(analyze_asset_m7):
        raise TypeError(f"analyze_asset_m7 not callable: {type(analyze_asset_m7)}")
    
    # Validate signature (allow defaults)
    sig = inspect.signature(analyze_asset_m7)
    params = list(sig.parameters.keys())
    
    if 'ticker' not in params:
        raise TypeError("analyze_asset_m7 missing required parameter: ticker")
    
    logger.info("[M7] Production module loaded from core.strategies.m7")
    
except ImportError as e:
    logger.error(f"[M7] Import failed: {e}")
    analyze_asset_m7 = _m7_fallback
    
except AttributeError as e:
    logger.error(f"[M7] Attribute error: {e}")
    analyze_asset_m7 = _m7_fallback
    
except TypeError as e:
    logger.error(f"[M7] Signature invalid: {e}")
    analyze_asset_m7 = _m7_fallback
    
except Exception as e:
    logger.error(f"[M7] Unexpected: {type(e).__name__}: {e}", exc_info=True)
    analyze_asset_m7 = _m7_fallback

# -------------------- Small Utils --------------------
def _clip01(x: float) -> float:
    """Clip value to [0, 1]"""
    return float(max(0.0, min(1.0, x)))

def _linreg_slope(y: np.ndarray) -> float:
    """Calculate linear regression slope"""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)

def _monotone_tp_probs(d: Dict[str, float]) -> Dict[str, float]:
    """Enforce monotone decreasing TP probabilities"""
    p1 = _clip01(float(d.get("tp1", 0.0)))
    p2 = _clip01(float(d.get("tp2", 0.0)))
    p3 = _clip01(float(d.get("tp3", 0.0)))
    p1 = max(p1, p2, p3)
    p2 = min(p1, max(p2, p3))
    p3 = min(p2, p3)
    return {"tp1": p1, "tp2": p2, "tp3": p3}

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

# --- загрузка калибровок из config/calibration.json ---
_DEFAULT_CAL = {
    "Global":     {"conf":{"method":"sigmoid","params":{"a":1.0,"b":0.0}}},
    "M7":         {"conf":{"method":"sigmoid","params":{"a":1.0,"b":0.0}}},
    "W7":         {"conf":{"method":"sigmoid","params":{"a":1.0,"b":0.0}}},
    "AlphaPulse": {"conf":{"method":"sigmoid","params":{"a":1.0,"b":0.0}}},
    "Octopus":    {"conf":{"method":"sigmoid","params":{"a":1.2,"b":-0.10}}}
}
def _load_calibration(path: str = "config/calibration.json") -> dict:
    p = Path(path)
    if not p.exists():
        return _DEFAULT_CAL
    try:
        return json.loads(p.read_text(encoding="utf-8")) or _DEFAULT_CAL
    except Exception:
        return _DEFAULT_CAL

_CAL = _load_calibration()
CAL_CONF = {
    "Global":     ConfidenceCalibrator(**_CAL["Global"]["conf"]),
    "M7":         ConfidenceCalibrator(**_CAL["M7"]["conf"]),
    "W7":         ConfidenceCalibrator(**_CAL["W7"]["conf"]),
    "AlphaPulse": ConfidenceCalibrator(**_CAL["AlphaPulse"]["conf"]),
    "Octopus":    ConfidenceCalibrator(**_CAL["Octopus"]["conf"]),
}

# -------------------- Global --------------------
def analyze_asset_global(ticker: str, horizon: str = "Краткосрочный"):
    cli = PolygonClient(); df = cli.daily_ohlc(ticker, days=120)
    current_price = float(df['close'].iloc[-1])

    closes = df['close']
    short_ma = closes.rolling(20).mean().iloc[-1]
    long_ma  = closes.rolling(50).mean().iloc[-1]
    ma_gap   = float((short_ma - long_ma) / max(1e-9, long_ma))
    slope    = _linreg_slope(closes.tail(30).values) / max(1e-9, current_price)
    atr = float(_atr_like(df, n=14).iloc[-1]) or 1e-9
    atr28 = float(_atr_like(df, n=28).iloc[-1]) or 1e-9
    vol_ratio = float(atr / max(1e-9, atr28))

    action = "BUY" if short_ma > long_ma else "SHORT"

    def _clp(x): return max(0.0, min(1.0, x))
    base = 0.55 + 0.22*_clp(abs(ma_gap)/0.02) + 0.10*_clp((abs(slope)-0.0003)/0.0007) - 0.12*_clp((vol_ratio-1.10)/0.60)
    confidence = float(max(0.55, min(0.86, base)))
    confidence = float(CAL_CONF["Global"](confidence))

    entry = current_price
    if action == "BUY":
        sl = current_price - 2*atr
        tp1, tp2, tp3 = current_price + 1*atr, current_price + 2*atr, current_price + 3*atr
        alt = "Покупка по рынку с консервативными целями"
    else:
        sl = current_price + 2*atr
        tp1, tp2, tp3 = current_price - 1*atr, current_price - 2*atr, current_price - 3*atr
        alt = "Продажа по рынку с консервативными целями"

    u1,u2,u3 = abs(tp1-entry)/atr, abs(tp2-entry)/atr, abs(tp3-entry)/atr
    k = 0.16 + 0.12*_clp((vol_ratio - 1.00)/0.80)
    b1 = confidence
    b2 = max(0.50, confidence - (0.08 + 0.03*_clp(vol_ratio - 1.0)))
    b3 = max(0.45, confidence - (0.16 + 0.05*_clp(vol_ratio - 1.2)))
    p1 = _clip01(b1*math.exp(-k*(u1-1.0)))
    p2 = _clip01(b2*math.exp(-k*(u2-1.5)))
    p3 = _clip01(b3*math.exp(-k*(u3-2.2)))
    probs = _monotone_tp_probs({"tp1": p1, "tp2": p2, "tp3": p3})

    meta_debug = {"u":[float(u1),float(u2),float(u3)],
                  "p":[float(probs['tp1']),float(probs['tp2']),float(probs['tp3'])]}
    try:
        log_agent_performance(
            agent="Global", ticker=ticker, horizon=horizon,
            action=action, confidence=float(confidence),
            levels={"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
            probs={"tp1": float(probs["tp1"]), "tp2": float(probs["tp2"]), "tp3": float(probs["tp3"])},
            meta={"probs_debug": meta_debug}, ts=pd.Timestamp.utcnow().isoformat()
        )
    except Exception as e:
        logger.warning("perf log Global failed: %s", e)

    return {
        "last_price": current_price,
        "recommendation": {"action": action, "confidence": confidence},
        "levels": {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs, "context": [], "note_html": f"<div>Global: {action} с {confidence:.0%}</div>",
        "alt": alt, "entry_kind": "market", "entry_label": f"{action} NOW",
        "meta": {"source":"Global","probs_debug": meta_debug}
    }

# -------------------- W7 --------------------
def analyze_asset_w7(ticker: str, horizon: str):
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon); hz = cfg["hz"]
    days = max(90, cfg["look"] * 2)

    # Загружаем дневные данные
    df = cli.daily_ohlc(ticker, days=days)

    # Приводим к DatetimeIndex для дальнейших weekly/daily агрегатов [web:3221][web:3236]
    if df is None or df.empty:
        df = pd.DataFrame(columns=["open","high","low","close","volume","timestamp"])
    else:
        df = df.sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    # Надёжное получение "текущей" цены: prev_close -> последний close из аггрегатов -> last trade (как опция) [web:3225][web:3232]
    price = cli.prev_close(ticker)
    if price is None and not df.empty:
        price = float(df["close"].iloc[-1])
    if price is None:
        try:
            price = cli.last_trade_price(ticker)  # может дать 403 — ловим и игнорируем [web:3225]
        except Exception:
            price = None
    if price is None:
        price = 0.0  # офлайн/нет данных

    closes = df["close"] if "close" in df.columns else pd.Series(dtype=float)
    tail = df.tail(cfg["look"]) if not df.empty else df
    rng_low, rng_high = float(tail["low"].min()) if not tail.empty else 0.0, float(tail["high"].max()) if not tail.empty else 0.0
    rng_w = max(1e-9, rng_high - rng_low); pos = (price - rng_low) / rng_w if rng_w > 0 else 0.0
    slope = _linreg_slope(closes.tail(cfg["trend"]).values) if not closes.empty else 0.0
    slope_norm = slope / max(1e-9, price if price else 1.0)

    atr_d = float(_atr_like(df, n=cfg["atr"]).iloc[-1]) if not df.empty else 0.0
    atr_w = _weekly_atr(df) if (not df.empty and cfg.get("use_weekly_atr")) else atr_d
    vol_ratio = (atr_d / max(1e-9, float(_atr_like(df, n=cfg["atr"] * 2).iloc[-1]))) if not df.empty else 1.0

    ha = _heikin_ashi(df) if not df.empty else pd.DataFrame(columns=["ha_close"])
    ha_diff = ha["ha_close"].diff() if "ha_close" in ha.columns else pd.Series(dtype=float)
    ha_up_run = _streak_by_sign(ha_diff, True) if not ha_diff.empty else 0
    ha_down_run = _streak_by_sign(ha_diff, False) if not ha_diff.empty else 0

    _, _, hist = _macd_hist(closes) if not closes.empty else (None,None,pd.Series(dtype=float))
    macd_pos_run = _streak_by_sign(hist, True) if not hist.empty else 0
    macd_neg_run = _streak_by_sign(hist, False) if not hist.empty else 0

    last_row = df.iloc[-1] if not df.empty else pd.Series({"open":0,"high":0,"low":0,"close":price})
    body, up_wick, dn_wick = _wick_profile(last_row)
    long_upper = (up_wick > body * 1.3) and (up_wick > dn_wick * 1.1)
    long_lower = (dn_wick > body * 1.3) and (dn_wick > up_wick * 1.1)

    hlc = _last_period_hlc(df, cfg["pivot_rule"]) if not df.empty else None
    if not hlc and not df.empty:
        hlc = (float(df["high"].tail(60).max()), float(df["low"].tail(60).min()), float(df["close"].iloc[-1]))
    elif not hlc:
        hlc = (0.0, 0.0, price)
    H, L, C = hlc
    piv = _fib_pivots(H, L, C)
    P, R1, R2, S1, S2 = piv["P"], piv["R1"], piv.get("R2"), piv["S1"], piv.get("S2")

    tol_k = {"ST": 0.18, "MID": 0.22, "LT": 0.28}[hz]
    step_d, step_w = atr_d, atr_w
    buf = tol_k * (step_w if hz != "ST" else step_d)

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
    conf = float(max(0.55, min(0.90, base))); conf = float(CAL_CONF["W7"](conf))

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
    atr_for_floor = step_w if hz != "ST" else step_d
    tp1, tp2, tp3 = _apply_tp_floors(entry, sl, tp1, tp2, tp3, action, hz, price, atr_for_floor)
    tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)
    sl,  tp1, tp2, tp3 = _sanity_levels(action, entry, sl, tp1, tp2, tp3, price, step_d, step_w, hz)

    entry_kind = _entry_kind(action, entry, price, step_d)
    entry_label = {"buy-stop":"Buy STOP","buy-limit":"Buy LIMIT","buy-now":"Buy NOW",
                   "sell-stop":"Sell STOP","sell-limit":"Sell LIMIT","sell-now":"Sell NOW"}.get(entry_kind, "")

    probs = {"tp1": float(_clip01(0.58 + 0.27*(conf - 0.55)/0.35)),
             "tp2": float(_clip01(0.44 + 0.21*(conf - 0.55)/0.35)),
             "tp3": float(_clip01(0.28 + 0.13*(conf - 0.55)/0.35))}
    probs = _monotone_tp_probs(probs)

    u_base = step_w if hz != "ST" else step_d
    u1,u2,u3 = abs(tp1-entry)/u_base, abs(tp2-entry)/u_base, abs(tp3-entry)/u_base
    meta_debug = {"atr_d": float(atr_d), "atr_w": float(atr_w),
                  "slope_norm": float(slope_norm), "pos": float(pos),
                  "u":[float(u1),float(u2),float(u3)],
                  "p":[float(probs["tp1"]), float(probs["tp2"]), float(probs["tp3"])]}
    try:
        log_agent_performance(
            agent="W7", ticker=ticker, horizon=horizon,
            action=action, confidence=float(conf),
            levels={"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
            probs={"tp1": float(probs["tp1"]), "tp2": float(probs["tp2"]), "tp3": float(probs["tp3"])},
            meta={"probs_debug": meta_debug}, ts=pd.Timestamp.utcnow().isoformat()
        )
    except Exception as e:
        logger.warning("perf log W7 failed: %s", e)

    return {
        "last_price": float(price),
        "recommendation": {"action": action, "confidence": float(round(conf, 4))},
        "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
        "probs": probs, "context": [], "note_html": "<div>W7: контекст по волатильности и зонам</div>",
        "alt": alt, "entry_kind": entry_kind, "entry_label": entry_label,
        "meta":{"source":"W7","grey_zone": bool(0.48 <= conf <= 0.58), "probs_debug": meta_debug}
    }

# -------------------- AlphaPulse --------------------
try:
    from core.agents.alphapulse import analyze_asset_alphapulse as _alphapulse_impl

    def analyze_asset_alphapulse(ticker: str, horizon: str = "Краткосрочный") -> Dict[str, Any]:
        # 1) Вызов внешнего агента
        res = _alphapulse_impl(ticker, horizon)
        reco = res.get("recommendation", {}) or {}
        action_ext = str(reco.get("action", "WAIT")).upper()
        conf_ext = float(reco.get("confidence", 0.50))

        # 2) Калибровка уверенности внешнего агента
        conf_cal = float(CAL_CONF["AlphaPulse"](conf_ext))
        res.setdefault("recommendation", {})["confidence"] = conf_cal

        # 3) Отладочная мета‑информация (u в ATR, p — probs)
        levels = res.get("levels", {}) or {}
        probs  = res.get("probs", {}) or {}
        u_vals = []
        try:
            df_dbg = PolygonClient().daily_ohlc(ticker, days=120)
            atr_dbg = float(_atr_like(df_dbg, n=14).iloc[-1]) or 1e-9
            if levels:
                u_vals = [
                    float(abs(levels.get("tp1", 0.0) - levels.get("entry", 0.0)) / atr_dbg),
                    float(abs(levels.get("tp2", 0.0) - levels.get("entry", 0.0)) / atr_dbg),
                    float(abs(levels.get("tp3", 0.0) - levels.get("entry", 0.0)) / atr_dbg),
                ]
        except Exception:
            pass
        res["meta"] = {
            **res.get("meta", {}),
            "probs_debug": {"u": u_vals, "p": [float(probs.get("tp1", 0.0)), float(probs.get("tp2", 0.0)), float(probs.get("tp3", 0.0))]},
            "overlay_used": False,
            "overlay_reason": "",
            "fallback": res.get("meta", {}).get("fallback", False)
        }

        # 4) Оверлей поверх внешнего агента при нейтрали/слабой уверенности: |z| >= 1.0
        try:
            df = PolygonClient().daily_ohlc(ticker, days=240)
            price = float(df["close"].iloc[-1])
            close = df["close"].astype(float)
            ma20  = close.rolling(20).mean()
            sd20  = close.rolling(20).std()
            z     = float((close.iloc[-1] - ma20.iloc[-1]) / max(1e-9, sd20.iloc[-1]))
            abs_z = abs(z)
            atr   = float(_atr_like(df, n=14).iloc[-1]) or 1e-9

            need_overlay = (action_ext == "WAIT") or (conf_cal <= 0.55)
            if need_overlay and abs_z >= 1.0:
                # Сторона и уровни по MR
                if z <= -1.0:
                    side, entry, sl = "BUY", price, max(0.0, price - 1.2 * atr)
                    tp1, tp2, tp3 = entry + 1.2 * atr, entry + 2.0 * atr, entry + 3.0 * atr
                else:
                    side, entry, sl = "SHORT", price, price + 1.2 * atr
                    tp1, tp2, tp3 = entry - 1.2 * atr, entry - 2.0 * atr, entry - 3.0 * atr

                # Градация уверенности по силе отклонения
                if abs_z >= 2.0:
                    base_conf = 0.78
                elif abs_z >= 1.5:
                    base_conf = 0.68
                else:  # 1.0 ≤ |z| < 1.5
                    base_conf = 0.58
                conf = float(CAL_CONF["AlphaPulse"](float(max(0.55, min(0.82, base_conf)))))

                # Probabilities с монотонией
                k = 0.18
                u1, u2, u3 = abs(tp1 - entry) / atr, abs(tp2 - entry) / atr, abs(tp3 - entry) / atr
                b1, b2, b3 = conf, max(0.50, conf - 0.08), max(0.45, conf - 0.16)
                p1 = _clip01(b1 * math.exp(-k * (u1 - 1.0)))
                p2 = _clip01(b2 * math.exp(-k * (u2 - 1.5)))
                p3 = _clip01(b3 * math.exp(-k * (u3 - 2.2)))
                probs_new = _monotone_tp_probs({"tp1": p1, "tp2": p2, "tp3": p3})

                # Обновление результата поверх внешнего WAIT/low‑conf
                res = {
                    "last_price": price,
                    "recommendation": {"action": side, "confidence": conf},
                    "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
                    "probs": probs_new,
                    "context": [f"AlphaPulse overlay: z={z:.2f} (|z|≥1.0)"],
                    "note_html": "<div>AlphaPulse: overlay mean‑reversion</div>",
                    "alt": "Mean‑Reversion",
                    "entry_kind": "market",
                    "entry_label": side,
                    "meta": {
                        **res.get("meta", {}),
                        "source": "AlphaPulse",
                        "overlay_used": True,
                        "overlay_reason": f"abs_z={abs_z:.2f} ≥ 1.0",
                        "probs_debug": {"u": [float(u1), float(u2), float(u3)],
                                        "p": [float(probs_new["tp1"]), float(probs_new["tp2"]), float(probs_new["tp3"])]}
                    }
                }
        except Exception:
            pass

        # 5) Логирование
        try:
            log_agent_performance(
                agent="AlphaPulse",
                ticker=ticker,
                horizon=horizon,
                action=res.get("recommendation", {}).get("action", "WAIT"),
                confidence=float(res.get("recommendation", {}).get("confidence", 0.50)),
                levels=res.get("levels", {}),
                probs=res.get("probs", {}),
                meta=res.get("meta", {}),
                ts=pd.Timestamp.utcnow().isoformat(),
            )
        except Exception as e:
            logger.warning("perf log AlphaPulse failed: %s", e)

        return res

except Exception:
    # Fallback: Mean‑Reversion на z-score с порогом ±1.0 и градуированной уверенностью
    def analyze_asset_alphapulse(ticker: str, horizon: str = "Краткосрочный") -> Dict[str, Any]:
        def _safe_load_ohlc(sym: str, days: int):
            for mod in ("services.data", "core.data"):
                try:
                    m = __import__(mod, fromlist=["load_ohlc"])
                    return getattr(m, "load_ohlc")(sym, days)
                except Exception:
                    pass
            try:
                return PolygonClient().daily_ohlc(sym, days=days)
            except Exception:
                return None

        df = _safe_load_ohlc(ticker, days=240)
        price = float(df["close"].iloc[-1]) if (isinstance(df, pd.DataFrame) and len(df) and "close" in df.columns) else 0.0
        if not isinstance(df, pd.DataFrame) or len(df) < 50 or "close" not in df.columns:
            res = {
                "last_price": price,
                "recommendation": {"action": "WAIT", "confidence": 0.52},
                "levels": {"entry": 0.0, "sl": 0.0, "tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
                "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
                "context": ["AlphaPulse: недостаточно данных"],
                "note_html": "<div>AlphaPulse: ожидание</div>",
                "alt": "WAIT", "entry_kind": "wait", "entry_label": "WAIT",
                "meta": {"source": "AlphaPulse", "grey_zone": True, "fallback": True}
            }
            try:
                log_agent_performance(agent="AlphaPulse", ticker=ticker, horizon=horizon,
                                      action="WAIT", confidence=0.52,
                                      levels=res["levels"], probs=res["probs"], meta=res["meta"],
                                      ts=pd.Timestamp.utcnow().isoformat())
            except Exception as e:
                logger.warning("perf log AlphaPulse failed: %s", e)
            return res

        close = df["close"].astype(float)
        ma20 = close.rolling(20).mean(); sd20 = close.rolling(20).std()
        z = float((close.iloc[-1] - ma20.iloc[-1]) / max(1e-9, sd20.iloc[-1]))
        atr = float(_atr_like(df, n=14).iloc[-1]) or 1e-9
        abs_z = abs(z)

        if abs_z < 1.0:
            res = {
                "last_price": price,
                "recommendation": {"action": "WAIT", "confidence": 0.52},
                "levels": {"entry": 0.0, "sl": 0.0, "tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
                "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
                "context": [f"AlphaPulse: z={z:.2f} нейтрально (<1.0σ)"],
                "note_html": "<div>AlphaPulse: нейтрально</div>",
                "alt": "WAIT", "entry_kind": "wait", "entry_label": "WAIT",
                "meta": {"source": "AlphaPulse", "grey_zone": True, "fallback": True}
            }
            try:
                log_agent_performance(agent="AlphaPulse", ticker=ticker, horizon=horizon,
                                      action="WAIT", confidence=0.52,
                                      levels=res["levels"], probs=res["probs"], meta=res["meta"],
                                      ts=pd.Timestamp.utcnow().isoformat())
            except Exception as e:
                logger.warning("perf log AlphaPulse failed: %s", e)
            return res

        if z <= -1.0:
            side, entry, sl = "BUY", price, max(0.0, price - 1.2 * atr)
            tp1, tp2, tp3 = entry + 1.2 * atr, entry + 2.0 * atr, entry + 3.0 * atr
        else:
            side, entry, sl = "SHORT", price, price + 1.2 * atr
            tp1, tp2, tp3 = entry - 1.2 * atr, entry - 2.0 * atr, entry - 3.0 * atr

        if abs_z >= 2.0:
            base_conf = 0.78
        elif abs_z >= 1.5:
            base_conf = 0.68
        else:  # 1.0 ≤ |z| < 1.5
            base_conf = 0.58

        conf = float(CAL_CONF["AlphaPulse"](float(max(0.55, min(0.82, base_conf)))))

        k = 0.18
        u1, u2, u3 = abs(tp1 - entry) / atr, abs(tp2 - entry) / atr, abs(tp3 - entry) / atr
        b1, b2, b3 = conf, max(0.50, conf - 0.08), max(0.45, conf - 0.16)
        p1 = _clip01(b1 * math.exp(-k * (u1 - 1.0)))
        p2 = _clip01(b2 * math.exp(-k * (u2 - 1.5)))
        p3 = _clip01(b3 * math.exp(-k * (u3 - 2.2)))
        probs = _monotone_tp_probs({"tp1": p1, "tp2": p2, "tp3": p3})

        meta_debug = {"u": [float(u1), float(u2), float(u3)],
                      "p": [float(probs["tp1"]), float(probs["tp2"]), float(probs["tp3"])]}

        res = {
            "last_price": price,
            "recommendation": {"action": side, "confidence": conf},
            "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
            "probs": probs,
            "context": [f"AlphaPulse MR(fallback): z={z:.2f}"],
            "note_html": "<div>AlphaPulse: mean‑reversion</div>",
            "alt": "Mean‑Reversion", "entry_kind": "market", "entry_label": side,
            "meta": {"source": "AlphaPulse", "grey_zone": bool(0.48 <= conf <= 0.58), "fallback": True, "probs_debug": meta_debug}
        }
        try:
            log_agent_performance(
                agent="AlphaPulse", ticker=ticker, horizon=horizon,
                action=side, confidence=float(conf),
                levels=res["levels"], probs=res["probs"], meta=res["meta"], ts=pd.Timestamp.utcnow().isoformat()
            )
        except Exception as e:
            logger.warning("perf log AlphaPulse failed: %s", e)
        return res

# -------------------- Оркестратор Octopus --------------------
OCTO_WEIGHTS: Dict[str, float] = {"Global": 0.13, "M7": 0.20, "W7": 0.26, "AlphaPulse": 0.28}  # как и было

def _act_to_num(a: str) -> int:  # без изменений
    return 1 if a == "BUY" else (-1 if a == "SHORT" else 0)

def _num_to_act(x: float) -> str:  # без изменений
    if x > 0: return "BUY"
    if x < 0: return "SHORT"
    return "WAIT"

def analyze_asset_octopus(ticker: str, horizon: str) -> Dict[str, Any]:
    # 1) Собираем ответы агентов
    parts: Dict[str, Dict[str, Any]] = {}
    for name, fn in {
        "Global":     analyze_asset_global,
        "M7":         analyze_asset_m7,
        "W7":         analyze_asset_w7,
        "AlphaPulse": analyze_asset_alphapulse,
    }.items():
        try:
            parts[name] = fn(ticker, horizon)
        except Exception as e:
            logger.warning("Agent %s failed: %s", name, e)

    if not parts:
        return {
            "last_price": 0.0,
            "recommendation": {"action": "WAIT", "confidence": 0.50},
            "levels": {"entry": 0.0, "sl": 0.0, "tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": ["Octopus: no agents responded"],
            "note_html": "<div>Octopus: WAIT</div>",
            "alt": "WAIT", "entry_kind": "wait", "entry_label": "WAIT",
            "meta": {"source": "Octopus", "votes": [], "ratio": 0.0}
        }

    # 2) Строим активные голоса BUY/SHORT с весами и conf (как было)
    active = []
    for k, r in parts.items():
        a = str(r.get("recommendation", {}).get("action", "WAIT")).upper()
        c = float(r.get("recommendation", {}).get("confidence", 0.5))
        w = float(OCTO_WEIGHTS.get(k, 0.20))
        if a in ("BUY", "SHORT"):
            active.append((k, a, _clip01(c), w))

    count_long  = sum(1 for (_, a, _, _) in active if a == "BUY")
    count_short = sum(1 for (_, a, _, _) in active if a == "SHORT")
    score_long  = sum(w * c for (_, a, c, w) in active if a == "BUY")
    score_short = sum(w * c for (_, a, c, w) in active if a == "SHORT")
    total_side  = score_long + score_short
    delta = abs(score_long - score_short)
    ratio = delta / max(1e-6, total_side)

    # 3) Правило выбора действия (без изменений)
    if count_long >= 3:
        final_action = "BUY"
    elif count_short >= 3:
        final_action = "SHORT"
    else:
        final_action = "WAIT" if ratio < 0.20 else ("BUY" if score_long > score_short else "SHORT")

    # Утилита для медианных уровней по сторонникам выбранной стороны
    def _median_levels(direction: str):
        L = [r.get("levels", {}) for r in parts.values()
             if str(r.get("recommendation", {}).get("action", "")).upper() == direction]
        if not L:
            return {"entry": 0.0, "sl": 0.0, "tp1": 0.0, "tp2": 0.0, "tp3": 0.0}
        med = lambda k: float(np.median([x.get(k, 0.0) for x in L if isinstance(x.get(k, None), (int, float))]))
        return {"entry": med("entry"), "sl": med("sl"), "tp1": med("tp1"), "tp2": med("tp2"), "tp3": med("tp3")}

    # 4) Уровни/пробы как было: медианы при слабой поляризации, иначе — от победителя
    if final_action in ("BUY", "SHORT"):
        cand = [t for t in active if t[1] == final_action]
        win_agent = (max(cand, key=lambda t: t[2] * t[3])[0] if cand
                     else max(active, key=lambda t: t[2] * t[3])[0])
        levels_out = _median_levels(final_action) if ratio < 0.25 else parts[win_agent].get("levels", {})
        probs_out = _monotone_tp_probs(parts.get(win_agent, {}).get("probs", {}) or {})

        # 5) Итоговый confidence (НОВОЕ): взвешенно по победившей стороне + мягкий штраф конфликта
        # 5.1 Взвешенный conf по сторонникам финальной стороны
        side_items = []
        for k, r in parts.items():
            rec = r.get("recommendation", {})
            if str(rec.get("action", "")).upper() == final_action:
                w = float(OCTO_WEIGHTS.get(k, 0.20))
                c = _clip01(rec.get("confidence", 0.5))
                side_items.append((w, c))
        overall_conf = (sum(w * c for w, c in side_items) / max(1e-6, sum(w for w, _ in side_items))) if side_items else 0.50

        # 5.2 Мягкий штраф за сильное несогласие противоположной стороны
        score_side = score_long if final_action == "BUY" else score_short
        score_opp  = score_short if final_action == "BUY" else score_long
        try:
            import os as _os
            beta = float(_os.getenv("OCTO_CONF_BETA", "0.35"))
        except Exception:
            beta = 0.35
        penalty = 1.0 - beta * (score_opp / max(1e-6, score_side))
        penalty = max(0.70, min(1.00, penalty))  # клип фактора
        overall_conf = float(CAL_CONF["Octopus"](_clip01(overall_conf * penalty)))

    else:
        levels_out = {"entry": 0.0, "sl": 0.0, "tp1": 0.0, "tp2": 0.0, "tp3": 0.0}
        probs_out  = {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0}
        overall_conf = float(CAL_CONF["Octopus"](0.50))

    # 6) Сбор ответа и логирование (как было)
    last_price = float(next(iter(parts.values())).get("last_price", 0.0))
    votes_txt = [{"agent": k,
                  "action": str(r.get("recommendation", {}).get("action", "")),
                  "confidence": float(r.get("recommendation", {}).get("confidence", 0.0))}
                 for k, r in parts.items()]

    res = {
        "last_price": last_price,
        "recommendation": {"action": final_action, "confidence": overall_conf},
        "levels": levels_out,
        "probs": probs_out,
        "context": [f"Octopus: ratio={ratio:.2f}, votes={count_long}L/{count_short}S"],
        "note_html": f"<div>Octopus: {final_action} с {overall_conf:.0%}</div>",
        "alt": "Octopus",
        "entry_kind": "market" if final_action != "WAIT" else "wait",
        "entry_label": final_action if final_action != "WAIT" else "WAIT",
        "meta": {"source": "Octopus", "votes": votes_txt, "ratio": float(ratio)}
    }

    try:
        log_agent_performance(
            agent="Octopus", ticker=ticker, horizon=horizon,
            action=final_action, confidence=float(overall_conf),
            levels=levels_out, probs=probs_out,
            meta={"votes": votes_txt, "ratio": float(ratio)},
            ts=pd.Timestamp.utcnow().isoformat()
        )
    except Exception as e:
        logger.warning("perf log Octopus failed: %s", e)

    return res

# -------------------- Strategy Router --------------------
STRATEGY_REGISTRY: Dict[str, Callable[[str, str], Dict[str, Any]]] = {
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

# -------------------- Тестовый запуск --------------------
if __name__ == "__main__":
    for s in ["Global", "M7", "W7", "AlphaPulse", "Octopus"]:
        try:
            print(f"\n=== {s} ===")
            out = analyze_asset("AAPL", "Краткосрочный", s)
            print({k: out[k] for k in ["last_price","recommendation","levels","probs"]})
        except Exception as e:
            print(f"{s} error:", e)
