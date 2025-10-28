# core/strategy.py
from __future__ import annotations

import json
import math
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import numpy as np
import pandas as pd

# -------------------- инфраструктура --------------------
# Мягкие импорты, чтобы UI не падал при отсутствии зависимостей
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
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)

def _monotone_tp_probs(d: Dict[str, float]) -> Dict[str, float]:
    p1 = _clip01(float(d.get("tp1", 0.0)))
    p2 = _clip01(float(d.get("tp2", 0.0)))
    p3 = _clip01(float(d.get("tp3", 0.0)))
    # enforce TP1 >= TP2 >= TP3
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
    # 1) Данные
    cli = PolygonClient()
    df = cli.daily_ohlc(ticker, days=200)
    if df is None or df.empty or "close" not in df.columns:
        return {
            "last_price": 0.0,
            "recommendation": {"action": "WAIT", "confidence": 0.55},
            "levels": {"entry": 0.0, "sl": 0.0, "tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": ["Global: нет данных"],
            "note_html": "<div>Global: ожидание</div>",
            "alt": "WAIT", "entry_kind": "wait", "entry_label": "WAIT",
            "meta": {"source":"Global","probs_debug":{"u":[],"p":[]}}
        }

    df = df.sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    closes = df["close"].astype(float)
    current_price = float(closes.iloc[-1])

    # 2) Индикаторы
    ma20 = closes.rolling(20).mean()
    ma50 = closes.rolling(50).mean()
    ma_gap = (ma20 - ma50) / ma50.replace(0, np.nan)  # относительный разрыв МА
    rets = closes.pct_change()
    slope30 = rets.rolling(30).mean()                 # сглаженный наклон
    atr14 = _atr_like(df, n=14).astype(float)
    atr28 = _atr_like(df, n=28).astype(float)
    atr = float(atr14.iloc[-1] or 1e-9)
    vol_ratio_series = (atr14 / atr28.replace(0, np.nan)).fillna(1.0)

    # 3) Z-score нормализация на длинном окне
    def _z(s: pd.Series, w: int = 180) -> pd.Series:
        m = s.rolling(w).mean()
        v = s.rolling(w).std().replace(0, np.nan)
        return (s - m) / v

    gap_z   = float(_z(ma_gap).iloc[-1])   if len(ma_gap)   else 0.0
    slope_z = float(_z(slope30).iloc[-1])  if len(slope30)  else 0.0
    vol_z   = float(_z(vol_ratio_series).iloc[-1]) if len(vol_ratio_series) else 0.0

    # 4) Сторона сделки (сохраняем исходную логику MA20 > MA50)
    action = "BUY" if float(ma20.iloc[-1]) > float(ma50.iloc[-1]) else "SHORT"

    # 5) Уверенность: логит из признаков + внешняя калибровка
    #    Сигмоида избегает «плато 70%», калибратор выравнивает честность вероятностей.
    logit = 0.10 + 0.90*gap_z + 0.70*slope_z - 0.55*vol_z
    confidence_raw = float(1.0 / (1.0 + np.exp(-logit)))   # 0..1
    confidence = float(CAL_CONF["Global"](confidence_raw)) # isotonic/sigmoid из конфигурации [внешняя калибровка]

    # 6) Стоп/цели: режим‑зависимые множители вместо фиксированных 1/2/3×ATR
    def _clip01(x: float) -> float: return max(0.0, min(1.0, float(x)))
    trend_strength = _clip01(0.55*abs(gap_z)/1.6 + 0.45*abs(slope_z)/1.6)  # 0..1
    vol_ratio = float(vol_ratio_series.iloc[-1]) if len(vol_ratio_series) else 1.0

    # Множители целей растут при сильном тренде и слегка растягиваются при высокой волатильности
    stretch = 0.25*_clip01((vol_ratio - 1.0)/0.8)
    m1 = 0.85 + 0.65*trend_strength + 0.25*stretch
    m2 = 1.55 + 1.30*trend_strength + 0.45*stretch
    m3 = 2.20 + 2.10*trend_strength + 0.70*stretch

    entry = current_price
    if action == "BUY":
        sl = entry - 2.0*atr
        tp1, tp2, tp3 = entry + m1*atr, entry + m2*atr, entry + m3*atr
        alt = "Покупка по рынку с режим‑зависимыми целями"
    else:
        sl = entry + 2.0*atr
        tp1, tp2, tp3 = entry - m1*atr, entry - m2*atr, entry - m3*atr
        alt = "Продажа по рынку с режим‑зависимыми целями"

    # 7) Вероятности TP: u = множители; крутизна зависит от волатильности и тренд‑силы
    u1, u2, u3 = float(m1), float(m2), float(m3)
    k = 0.12 + 0.28*_clip01((vol_ratio - 0.9)/0.8) + 0.10*trend_strength  # быстрее штрафует дальние цели в «шторм»
    b1 = confidence
    b2 = max(0.50, confidence - (0.08 + 0.04*max(0.0, vol_ratio - 1.0)))
    b3 = max(0.45, confidence - (0.16 + 0.07*max(0.0, vol_ratio - 1.2)))

    p1 = _clip01(b1 * math.exp(-k*(u1 - 1.0)))
    p2 = _clip01(b2 * math.exp(-k*(u2 - 1.5)))
    p3 = _clip01(b3 * math.exp(-k*(u3 - 2.2)))
    probs = _monotone_tp_probs({"tp1": p1, "tp2": p2, "tp3": p3})

    # 8) Логирование и отладка
    meta_debug = {
        "u":[float(u1), float(u2), float(u3)],
        "p":[float(probs["tp1"]), float(probs["tp2"]), float(probs["tp3"])],
        "gap_z": float(gap_z), "slope_z": float(slope_z), "vol_z": float(vol_z),
        "trend_strength": float(trend_strength), "vol_ratio": float(vol_ratio),
        "conf_raw": float(confidence_raw), "conf_cal": float(confidence)
    }
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
        "last_price": float(current_price),
        "recommendation": {"action": action, "confidence": float(confidence)},
        "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
        "probs": probs, "context": [],
        "note_html": f"<div>Global: {action} с {confidence:.0%}</div>",
        "alt": alt, "entry_kind": "market", "entry_label": f"{action} NOW",
        "meta": {"source":"Global","probs_debug": meta_debug}
    }

# -------------------- M7 PRODUCTION STRATEGY (FIXED) --------------------
import pandas as pd
import numpy as np
import math
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

# === СИНГЛТОН ДЛЯ ДАННЫХ ===
class DataProvider:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DataProvider()
        return cls._instance
    
    def __init__(self):
        self._client = None
    
    def get_client(self):
        if self._client is None:
            try:
                from data.polygon_client import PolygonClient
                self._client = PolygonClient()
            except ImportError:
                logger.error("M7: PolygonClient not available")
                self._client = None
        return self._client
    
    def get_ohlc_data(self, ticker: str, days: int = 120) -> Optional[pd.DataFrame]:
        """Единый метод получения данных для всех стратегий"""
        try:
            client = self.get_client()
            if client is None:
                return None
            return client.daily_ohlc(ticker, days=days)
        except Exception as e:
            logger.error(f"M7: Data fetch failed for {ticker}: {e}")
            return None

# === M7 СТРАТЕГИЯ (ИСПРАВЛЕННАЯ) ===
class M7TradingStrategy:
    def __init__(self, atr_period=14, atr_multiplier=1.5, pivot_period='D', fib_levels=None):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.pivot_period = pivot_period
        self.fib_levels = fib_levels or [0.236, 0.382, 0.5, 0.618, 0.786]

    def calculate_pivot_points(self, h, l, c):
        try:
            pivot = (h + l + c) / 3
            r1 = (2 * pivot) - l
            r2 = pivot + (h - l)
            r3 = h + 2 * (pivot - l)
            s1 = (2 * pivot) - h
            s2 = pivot - (h - l)
            s3 = l - 2 * (h - pivot)
            return {'pivot': pivot, 'r1': r1, 'r2': r2, 'r3': r3, 's1': s1, 's2': s2, 's3': s3}
        except Exception:
            return {}

    def calculate_fib_levels(self, h, l):
        try:
            diff = h - l
            fib = {}
            for level in self.fib_levels:
                fib[f'fib_{int(level*1000)}'] = h - level * diff
            return fib
        except Exception:
            return {}

    def identify_key_levels(self, data: pd.DataFrame, current_date: pd.Timestamp) -> Dict:
        """Расчет уровней только на исторических данных (без look-ahead)"""
        try:
            if data.empty:
                return {}
            
            # Используем только данные ДО текущей даты
            historical_data = data[data.index < current_date]
            if historical_data.empty:
                return {}
            
            # Группируем по дням/неделям
            if self.pivot_period == 'D':
                grouped = historical_data.resample('D')
            else:
                grouped = historical_data.resample('W')
            
            key_levels = {}
            
            for timestamp, group in grouped:
                if len(group) > 0:
                    try:
                        h = float(group['high'].max())
                        l = float(group['low'].min())
                        c = float(group['close'].iloc[-1])
                        
                        pivots = self.calculate_pivot_points(h, l, c)
                        fibs = self.calculate_fib_levels(h, l)
                        
                        key_levels.update(pivots)
                        key_levels.update(fibs)
                    except Exception:
                        continue
                        
            return key_levels
        except Exception as e:
            logger.error(f"M7: Level identification failed: {e}")
            return {}

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Надежный расчет ATR"""
        try:
            if data.empty or len(data) < 2:
                return pd.Series([0.01] * len(data), index=data.index)
            
            high = data['high'].astype(float)
            low = data['low'].astype(float)
            close = data['close'].astype(float)
            
            hl = high - low
            hc = abs(high - close.shift(1))
            lc = abs(low - close.shift(1))
            
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            return tr.rolling(self.atr_period, min_periods=1).mean()
        except Exception:
            return pd.Series([0.01] * len(data), index=data.index)

    def generate_signals(self, data: pd.DataFrame) -> List[Dict]:
        """Генерация сигналов на исторических данных"""
        sigs = []
        
        try:
            if data.empty or len(data) < 20:
                return sigs
            
            # Проверяем необходимые колонки
            required_cols = ['high', 'low', 'close']
            if not all(col in data.columns for col in required_cols):
                return sigs
            
            data = data.copy()
            data['atr'] = self._calculate_atr(data)
            
            if data['atr'].empty:
                return sigs
                
            current_atr = float(data['atr'].iloc[-1])
            if current_atr <= 1e-9:
                current_atr = 0.01
                
            current_price = float(data['close'].iloc[-1])
            current_time = data.index[-1]
            
            # Расчет уровней на исторических данных
            key_levels = self.identify_key_levels(data, current_time)
            if not key_levels:
                return sigs
            
            # Генерация сигналов
            for level_name, level_value in key_levels.items():
                try:
                    distance = abs(current_price - level_value) / current_atr
                    
                    if distance < self.atr_multiplier:
                        is_resistance = level_value > current_price
                        
                        if is_resistance:
                            order_type = 'SELL_LIMIT'
                            entry = level_value * 0.998
                            sl = level_value * 1.02
                        else:
                            order_type = 'BUY_LIMIT'
                            entry = level_value * 1.002
                            sl = level_value * 0.98
                        
                        risk = abs(entry - sl)
                        tp = entry + (risk * 2 if order_type == 'BUY_LIMIT' else -risk * 2)
                        confidence = max(0.1, min(0.95, 1.0 - (distance / self.atr_multiplier)))
                        
                        sigs.append({
                            'type': order_type,
                            'price': round(entry, 4),
                            'stop_loss': round(sl, 4),
                            'take_profit': round(tp, 4),
                            'confidence': round(confidence, 2),
                            'level': level_name,
                            'level_value': round(level_value, 4),
                            'timestamp': current_time
                        })
                        
                except Exception:
                    continue
            
            return sigs
            
        except Exception as e:
            logger.error(f"M7: Signal generation failed: {e}")
            return sigs

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def _monotone_tp_probs(probs: Dict) -> Dict:
    try:
        p1, p2, p3 = probs["tp1"], probs["tp2"], probs["tp3"]
        if p1 < p2: p2 = p1 * 0.95
        if p2 < p3: p3 = p2 * 0.95
        return {"tp1": _clip01(p1), "tp2": _clip01(p2), "tp3": _clip01(p3)}
    except Exception:
        return {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0}

def analyze_asset_m7(ticker: str, horizon: str = "Краткосрочный", use_ml: bool = False) -> Dict:
    """
    Исправленная версия M7 стратегии
    """
    try:
        from core.performance_logger import log_agent_performance
    except ImportError:
        def log_agent_performance(*args, **kwargs):
            pass

    # Получаем данные через единый провайдер
    data_provider = DataProvider.get_instance()
    df = data_provider.get_ohlc_data(ticker, days=120)
    
    if df is None or df.empty:
        return {
            "last_price": 0.0,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": ["M7: недостаточно данных"],
            "note_html": "<div>M7: ожидание данных</div>",
            "alt": "Ожидание данных",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "M7", "grey_zone": True}
        }

    # Обработка данных
    try:
        if 'timestamp' in df.columns:
            df = df.sort_values("timestamp")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index, utc=True)
            
        current_price = float(df['close'].iloc[-1])
        
    except Exception as e:
        logger.error(f"M7: Data processing failed for {ticker}: {e}")
        return {
            "last_price": 0.0,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": [f"M7: ошибка обработки данных"],
            "note_html": "<div>M7: ошибка данных</div>",
            "alt": "Ошибка данных",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "M7", "grey_zone": True}
        }

    # Расчет ATR
    strategy = M7TradingStrategy()
    atr_series = strategy._calculate_atr(df)
    atr14 = float(atr_series.iloc[-1]) if not atr_series.empty else 0.01

    # Генерация сигналов
    signals = strategy.generate_signals(df)
    
    if not signals:
        result = {
            "last_price": current_price,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": ["Нет сигналов по стратегии M7"],
            "note_html": "<div>M7: ожидание сигналов</div>",
            "alt": "Ожидание сигналов",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "M7", "grey_zone": True}
        }
        
        try:
            log_agent_performance(
                agent="M7", ticker=ticker, horizon=horizon, action="WAIT", confidence=0.50,
                levels=result["levels"], probs=result["probs"], 
                meta={"probs_debug": {"u": [], "p": []}},
                ts=pd.Timestamp.utcnow().isoformat()
            )
        except Exception:
            pass
            
        return result

    # Обработка лучшего сигнала
    best_signal = max(signals, key=lambda x: x['confidence'])
    raw_conf = float(_clip01(best_signal['confidence']))
    entry = float(best_signal['price'])
    sl = float(best_signal['stop_loss'])
    risk = abs(entry - sl)
    
    # Расчет confidence
    conf_base = 0.50 + 0.34 * math.tanh((raw_conf - 0.65) / 0.20)
    
    try:
        vol = float(df['close'].pct_change().std() * np.sqrt(252))
    except Exception:
        vol = 0.2
        
    penalty = (0.05 if vol > 0.35 else 0.0) + (0.04 if risk/atr14 < 0.8 else 0.0) + (0.03 if risk/atr14 > 3.5 else 0.0)
    conf = float(max(0.52, min(0.82, conf_base * (1.0 - penalty))))
    
    try:
        from core.strategy import CAL_CONF
        conf = float(CAL_CONF["M7"](conf))
    except Exception:
        pass
    
    # Определение направления и тейк-профитов
    if best_signal['type'].startswith('BUY'):
        tp1, tp2, tp3 = entry + 1.5*risk, entry + 2.5*risk, entry + 4.0*risk
        action = "BUY"
    else:
        tp1, tp2, tp3 = entry - 1.5*risk, entry - 2.5*risk, entry - 4.0*risk
        action = "SHORT"
    
    # Расчет вероятностей
    u1, u2, u3 = abs(tp1-entry)/atr14, abs(tp2-entry)/atr14, abs(tp3-entry)/atr14
    k = 0.18
    b1, b2, b3 = conf, max(0.50, conf-0.08), max(0.45, conf-0.16)
    p1 = _clip01(b1 * np.exp(-k * (u1-1.0)))
    p2 = _clip01(b2 * np.exp(-k * (u2-1.5)))
    p3 = _clip01(b3 * np.exp(-k * (u3-2.2)))
    probs = _monotone_tp_probs({"tp1": p1, "tp2": p2, "tp3": p3})
    
    # Логирование
    try:
        log_agent_performance(
            agent="M7", ticker=ticker, horizon=horizon, action=action, confidence=float(conf),
            levels={"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
            probs={"tp1": float(probs["tp1"]), "tp2": float(probs["tp2"]), "tp3": float(probs["tp3"])},
            meta={
                "probs_debug": {
                    "u": [float(u1), float(u2), float(u3)],
                    "p": [float(probs["tp1"]), float(probs["tp2"]), float(probs["tp3"])]
                }
            },
            ts=pd.Timestamp.utcnow().isoformat()
        )
    except Exception:
        pass
    
    return {
        "last_price": current_price,
        "recommendation": {"action": action, "confidence": float(conf)},
        "levels": {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs,
        "context": [f"Сигнал от уровня {best_signal['level']}"],
        "note_html": f"<div>M7: {best_signal['type']} на уровне {best_signal['level_value']}</div>",
        "alt": "Торговля по M7",
        "entry_kind": "limit",
        "entry_label": best_signal['type'],
        "meta": {"source": "M7", "grey_zone": bool(0.48 <= conf <= 0.58)}
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

# -------------------- MODERN OCTOPUS ORCHESTRATOR --------------------
class OctopusOrchestrator:
    """Современный оркестратор стратегий с улучшенной логикой"""
    
    def __init__(self):
        self.strategy_weights = {
            "Global": 0.15,
            "M7": 0.20,  # Увеличена весомость M7
            "W7": 0.35, 
            "AlphaPulse": 0.30
        }
        self.min_confidence = 0.55
        self.consensus_threshold = 0.60
        
    def analyze_asset(self, ticker: str, horizon: str = "Краткосрочный") -> Dict:
        """Основной метод анализа с улучшенной логикой"""
        try:
            # Получаем результаты всех стратегий
            strategy_results = self._get_strategy_results(ticker, horizon)
            
            if not strategy_results:
                return self._create_fallback_response(ticker)
            
            # Анализ консенсуса
            consensus = self._analyze_consensus(strategy_results)
            
            if consensus['confidence'] >= self.consensus_threshold:
                return self._create_consensus_response(consensus, strategy_results, ticker)
            else:
                return self._create_uncertain_response(strategy_results, ticker)
                
        except Exception as e:
            logger.error(f"Octopus orchestration failed for {ticker}: {e}")
            return self._create_error_response(ticker)

    def _get_strategy_results(self, ticker: str, horizon: str) -> Dict[str, Dict]:
        """Параллельное выполнение стратегий"""
        strategies = {
            "Global": analyze_asset_global,
            "M7": analyze_asset_m7, 
            "W7": analyze_asset_w7,
            "AlphaPulse": analyze_asset_alphapulse
        }
        
        results = {}
        for name, strategy_fn in strategies.items():
            try:
                result = strategy_fn(ticker, horizon)
                if self._validate_strategy_result(result):
                    results[name] = result
                else:
                    logger.warning(f"Octopus: Invalid result from {name} for {ticker}")
            except Exception as e:
                logger.error(f"Octopus: Strategy {name} failed for {ticker}: {e}")
                continue
                
        return results

    def _validate_strategy_result(self, result: Dict) -> bool:
        """Валидация результатов стратегии"""
        required_fields = ['last_price', 'recommendation', 'levels']
        if not all(field in result for field in required_fields):
            return False
            
        recommendation = result.get('recommendation', {})
        if not isinstance(recommendation, dict):
            return False
            
        action = recommendation.get('action', '')
        confidence = recommendation.get('confidence', 0)
        
        return action in ['BUY', 'SHORT', 'WAIT'] and 0 <= confidence <= 1

    def _analyze_consensus(self, strategy_results: Dict[str, Dict]) -> Dict:
        """Анализ консенсуса между стратегиями"""
        votes = {'BUY': 0, 'SHORT': 0, 'WAIT': 0}
        weighted_scores = {'BUY': 0.0, 'SHORT': 0.0}
        total_weight = 0.0
        
        for strategy_name, result in strategy_results.items():
            weight = self.strategy_weights.get(strategy_name, 0.10)
            recommendation = result.get('recommendation', {})
            action = recommendation.get('action', 'WAIT')
            confidence = recommendation.get('confidence', 0.0)
            
            votes[action] += 1
            
            if action in ['BUY', 'SHORT']:
                weighted_scores[action] += weight * confidence
                total_weight += weight
        
        # Определение консенсусного действия
        if total_weight > 0:
            buy_score = weighted_scores['BUY'] / total_weight
            short_score = weighted_scores['SHORT'] / total_weight
            
            if buy_score > short_score and buy_score >= self.min_confidence:
                return {'action': 'BUY', 'confidence': buy_score, 'vote_count': votes}
            elif short_score > buy_score and short_score >= self.min_confidence:
                return {'action': 'SHORT', 'confidence': short_score, 'vote_count': votes}
        
        return {'action': 'WAIT', 'confidence': max(weighted_scores.values()), 'vote_count': votes}

    def _create_consensus_response(self, consensus: Dict, strategy_results: Dict, ticker: str) -> Dict:
        """Создание ответа при наличии консенсуса"""
        action = consensus['action']
        confidence = consensus['confidence']
        
        # Выбор лучших уровней от стратегий с тем же действием
        best_levels = self._select_best_levels(strategy_results, action)
        
        response = {
            "last_price": self._get_consensus_price(strategy_results),
            "recommendation": {"action": action, "confidence": float(confidence)},
            "levels": best_levels,
            "probs": self._calculate_consensus_probs(strategy_results, action),
            "context": [f"Octopus: консенсус {action} ({len(strategy_results)} стратегий)"],
            "note_html": f"<div>Octopus: консенсус {action}</div>",
            "alt": f"Консенсус {action}",
            "entry_kind": "market",
            "entry_label": action,
            "meta": {
                "source": "Octopus",
                "consensus": True,
                "strategies_used": list(strategy_results.keys()),
                "vote_count": consensus['vote_count']
            }
        }
        
        self._log_performance(ticker, "Краткосрочный", action, confidence, best_levels, response["probs"])
        return response

    def _select_best_levels(self, strategy_results: Dict[str, Dict], action: str) -> Dict:
        """Выбор лучших уровней от стратегий с консенсусным действием"""
        relevant_results = [
            result for result in strategy_results.values()
            if result.get('recommendation', {}).get('action') == action
        ]
        
        if not relevant_results:
            return {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0}
        
        # Используем медианные значения для стабильности
        entries = [r['levels'].get('entry', 0) for r in relevant_results if r['levels'].get('entry', 0) > 0]
        sls = [r['levels'].get('sl', 0) for r in relevant_results if r['levels'].get('sl', 0) > 0]
        
        return {
            "entry": float(np.median(entries)) if entries else 0,
            "sl": float(np.median(sls)) if sls else 0,
            "tp1": 0, "tp2": 0, "tp3": 0  # Рассчитываются динамически
        }

    def _calculate_consensus_probs(self, strategy_results: Dict[str, Dict], action: str) -> Dict:
        """Расчет консенсусных вероятностей"""
        relevant_probs = [
            r['probs'] for r in strategy_results.values()
            if r.get('recommendation', {}).get('action') == action and 'probs' in r
        ]
        
        if not relevant_probs:
            return {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0}
        
        # Усреднение вероятностей
        avg_probs = {
            "tp1": np.mean([p.get('tp1', 0) for p in relevant_probs]),
            "tp2": np.mean([p.get('tp2', 0) for p in relevant_probs]),
            "tp3": np.mean([p.get('tp3', 0) for p in relevant_probs])
        }
        
        return _monotone_tp_probs(avg_probs)

    def _get_consensus_price(self, strategy_results: Dict[str, Dict]) -> float:
        """Получение консенсусной цены"""
        prices = [r.get('last_price', 0) for r in strategy_results.values() if r.get('last_price', 0) > 0]
        return float(np.median(prices)) if prices else 0.0

    def _create_uncertain_response(self, strategy_results: Dict, ticker: str) -> Dict:
        """Создание ответа при неопределенности"""
        price = self._get_consensus_price(strategy_results)
        
        response = {
            "last_price": price,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": [f"Octopus: нет консенсуса ({len(strategy_results)} стратегий)"],
            "note_html": "<div>Octopus: ожидание консенсуса</div>",
            "alt": "Ожидание консенсуса",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {
                "source": "Octopus",
                "consensus": False,
                "strategies_used": list(strategy_results.keys())
            }
        }
        
        self._log_performance(ticker, "Краткосрочный", "WAIT", 0.5, response["levels"], response["probs"])
        return response

    def _create_fallback_response(self, ticker: str) -> Dict:
        """Резервный ответ при полном сбое"""
        return {
            "last_price": 0.0,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": ["Octopus: все стратегии недоступны"],
            "note_html": "<div>Octopus: технические проблемы</div>",
            "alt": "Технические проблемы",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "Octopus", "fallback": True}
        }

    def _create_error_response(self, ticker: str) -> Dict:
        """Ответ при ошибке оркестратора"""
        return {
            "last_price": 0.0,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": ["Octopus: внутренняя ошибка"],
            "note_html": "<div>Octopus: системная ошибка</div>",
            "alt": "Системная ошибка",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "Octopus", "error": True}
        }

    def _log_performance(self, ticker: str, horizon: str, action: str, confidence: float, 
                        levels: Dict, probs: Dict):
        """Логирование производительности"""
        try:
            from core.performance_logger import log_agent_performance
            log_agent_performance(
                agent="Octopus", ticker=ticker, horizon=horizon,
                action=action, confidence=confidence,
                levels=levels, probs=probs,
                meta={"orchestrator": "modern"},
                ts=pd.Timestamp.utcnow().isoformat()
            )
        except Exception:
            pass

# Обновленная функция Octopus для обратной совместимости
_octopus_orchestrator = OctopusOrchestrator()

def analyze_asset_octopus(ticker: str, horizon: str = "Краткосрочный") -> Dict:
    """Обновленная функция Octopus с современным оркестратором"""
    return _octopus_orchestrator.analyze_asset(ticker, horizon)
