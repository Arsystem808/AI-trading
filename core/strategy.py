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

# -------------------- M7 (rules + optional ML overlay) --------------------
try:
    from core.model_loader import load_model_for
except Exception:
    def load_model_for(*args, **kwargs):
        return None

# простая память по последним срабатываниям уровней: (ticker, level) -> timestamp
_M7_LAST_HIT = {}

# состояние направления по тикеру для гистерезиса/кулдауна
_M7_DIR_STATE = {}
_M7_DIR_MIN_HOLD_HOURS = 4.0     # минимальное время между разворотами по направлению
_M7_DIR_MIN_DELTA_CONF = 0.08    # насколько новая уверенность должна быть выше старой, чтобы разрешить flip

class _M7IdentityCalibrator:
    def __call__(self, p: float) -> float:
        return float(max(0.0, min(1.0, p)))

_M7_CAL = None
def _m7_cal():
    global _M7_CAL
    if _M7_CAL is None:
        try:
            from core.strategy import CAL_CONF as CC
            _M7_CAL = CC
        except Exception:
            _M7_CAL = {"M7": _M7IdentityCalibrator()}
    return _M7_CAL

def _m7_direction_filter(ticker: str, raw_action: str, conf: float, ts: "pd.Timestamp"):
    """
    Гистерезис по направлению:
    - если новое направление совпадает со старым -> просто обновляем уверенность
    - если пытаемся перевернуться раньше минимального времени или без достаточного прироста уверенности -> блокируем flip и отдаём WAIT
    - если flip разрешён -> обновляем состояние
    """
    global _M7_DIR_STATE

    if raw_action not in ("BUY", "SHORT"):
        # для WAIT и прочего состояние не трогаем
        return raw_action, {"dir_state": "no_change"}

    st = _M7_DIR_STATE.get(ticker)
    min_hold = pd.Timedelta(hours=_M7_DIR_MIN_HOLD_HOURS)
    min_delta = float(_M7_DIR_MIN_DELTA_CONF)

    if st is None:
        _M7_DIR_STATE[ticker] = {"action": raw_action, "ts": ts, "conf": float(conf)}
        return raw_action, {"dir_state": "init"}

    prev_action = st.get("action")
    prev_ts = st.get("ts")
    prev_conf = float(st.get("conf", 0.0))

    # то же направление -> просто обновляем уверенность, но не считаем это flip
    if raw_action == prev_action:
        if conf > prev_conf:
            _M7_DIR_STATE[ticker] = {"action": raw_action, "ts": prev_ts, "conf": float(conf)}
        return raw_action, {"dir_state": "same", "prev_action": prev_action, "prev_ts": prev_ts}

    # пытаемся flip'нуться
    # 1) проверяем минимальное время в состоянии
    if prev_ts is not None and ts - prev_ts < min_hold:
        return "WAIT", {
            "dir_state": "blocked_time",
            "prev_action": prev_action,
            "prev_ts": prev_ts,
            "new_action": raw_action,
        }

    # 2) проверяем, что уверенность выросла достаточно, чтобы оправдать flip
    if conf < prev_conf + min_delta:
        return "WAIT", {
            "dir_state": "blocked_conf",
            "prev_action": prev_action,
            "prev_ts": prev_ts,
            "prev_conf": prev_conf,
            "new_action": raw_action,
            "new_conf": float(conf),
        }

    # flip разрешён
    _M7_DIR_STATE[ticker] = {"action": raw_action, "ts": ts, "conf": float(conf)}
    return raw_action, {
        "dir_state": "flipped",
        "prev_action": prev_action,
        "prev_ts": prev_ts,
        "prev_conf": prev_conf,
    }

class _M7Predictor:
    def __init__(self, ticker: str, agent="arxora_m7pro"):
        self.ticker = ticker
        self.agent = agent
        self.model = None
        self.scaler = None

    def load(self) -> bool:
        md = load_model_for(self.ticker, agent=self.agent)
        if not md or "model" not in md:
            logger.warning("M7 ML: model not found for %s (%s)", self.ticker, self.agent)
            return False
        self.model = md["model"]
        # опциональный внешний скейлер
        try:
            import joblib  # noqa: F401
            meta = md.get("metadata", {}) or {}
            sp = meta.get("scaler_artifact")
            if sp:
                p = Path(sp)
                if not p.is_absolute():
                    p = Path(".") / p
                if p.exists():
                    import joblib
                    self.scaler = joblib.load(p)
        except Exception as e:
            logger.warning("M7 ML: scaler load failed: %s", e)
        return True

    @staticmethod
    def _features(df: pd.DataFrame, price: float, atr: float) -> np.ndarray:
        r = float(df['close'].pct_change().iloc[-1])
        vol = float(df['close'].pct_change().rolling(20).std().iloc[-1] or 0.02)
        pos = 0.5
        if len(df) >= 20:
            hi = float(df['high'].rolling(20).max().iloc[-1])
            lo = float(df['low'].rolling(20).min().iloc[-1])
            pos = (price - lo) / max(1e-9, hi - lo)
        x = np.array([[r, vol, pos, atr / max(1e-9, price)]], dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _predict_proba_safe(self, X: np.ndarray,
                            df_last_row: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        # Сначала пробуем на одном признаковом векторе, затем fallback на df.tail(1)
        try:
            return self.model.predict_proba(X)
        except Exception as e1:
            if df_last_row is not None:
                try:
                    return self.model.predict_proba(df_last_row)
                except Exception as e2:
                    logger.warning(
                        "M7 ML: predict_proba failed on X(%s) and df_last_row(%s)",
                        e1, e2
                    )
                    return None
            logger.warning("M7 ML: predict_proba failed on X(%s)", e1)
            return None

    def predict(self, df: pd.DataFrame, price: float, atr: float) -> Tuple[str, float]:
        try:
            X = self._features(df, price, atr)
            if self.scaler is not None:
                try:
                    X = self.scaler.transform(X)
                except Exception as e:
                    logger.warning("M7 ML: scaler.transform failed: %s", e)
            df1 = df.tail(1).copy()
            proba = None
            if hasattr(self.model, "predict_proba"):
                proba = self._predict_proba_safe(X, df_last_row=df1)
            if proba is None and hasattr(self.model, "predict"):
                y = float(self.model.predict(X)[0])
                proba = np.array([[1.0 - y, y]], dtype=float)
            if proba is None:
                return ("WAIT", 0.5)

            # бинарный случай -> класс 1 трактуем как LONG
            if proba.shape[1] == 2:
                p_long = float(proba[0, 1])
                cal = _m7_cal()["M7"]
                p_long_cal = float(cal(p_long))
                # явные пороги
                if p_long_cal >= 0.55:
                    return ("BUY", p_long_cal)
                elif p_long_cal <= 0.45:
                    return ("SHORT", p_long_cal)
                else:
                    return ("WAIT", p_long_cal)

            # многокласс
            acts = ["BUY", "SHORT", "WAIT"]
            idx = int(np.argmax(proba[0]))
            conf = float(proba[0, idx])
            cal = _m7_cal()["M7"]
            return (acts[idx], float(cal(conf)))
        except Exception as e:
            logger.warning("M7 ML: predict failed: %s", e)
            return ("WAIT", 0.5)

class M7TradingStrategy:
    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 1.0,      # было 1.5, сузили зону вокруг уровня
        pivot_period: str = 'D',
        fib_levels=None,
        cooldown_minutes: int = 60,      # кулдаун по уровню для одного тикера
        use_trend_filter: bool = True    # включаем/выключаем фильтр по тренду
    ):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.pivot_period = pivot_period
        self.fib_levels = fib_levels or [0.236, 0.382, 0.5, 0.618, 0.786]
        self.cooldown = pd.Timedelta(minutes=cooldown_minutes)
        self.use_trend_filter = use_trend_filter

    def calculate_pivot_points(self, h, l, c):
        pivot = (h + l + c) / 3
        r1 = (2 * pivot) - l
        r2 = pivot + (h - l)
        r3 = h + 2 * (pivot - l)
        s1 = (2 * pivot) - h
        s2 = pivot - (h - l)
        s3 = l - 2 * (h - pivot)
        return {
            'pivot': pivot,
            'r1': r1,
            'r2': r2,
            'r3': r3,
            's1': s1,
            's2': s2,
            's3': s3
        }

    def calculate_fib_levels(self, h, l):
        diff = h - l
        fib = {}
        for level in self.fib_levels:
            fib[f'fib_{int(level * 1000)}'] = h - level * diff
        return fib

    def identify_key_levels(self, data: pd.DataFrame):
        grouped = data.resample('D') if self.pivot_period == 'D' else data.resample('W')
        key = {}
        for _, g in grouped:
            if len(g) > 0:
                h = g['high'].max()
                l = g['low'].min()
                c = g['close'].iloc[-1]
                key.update(self.calculate_pivot_points(h, l, c))
                key.update(self.calculate_fib_levels(h, l))
        return key

    def generate_signals(self, data: pd.DataFrame, ticker: Optional[str] = None):
        """
        Генерирует сигналы с учётом:
        - ATR-фильтра вокруг уровней
        - кулдауна по (ticker, level)
        - простого тренд-фильтра по центральному pivot
        """
        sigs = []
        req = ['high', 'low', 'close']
        if not all(c in data.columns for c in req):
            return sigs

        data = data.copy()
        data['atr'] = _atr_like(data, self.atr_period)
        cur_atr = float(data['atr'].iloc[-1])
        if cur_atr <= 0:
            return sigs

        key = self.identify_key_levels(data)
        if not key:
            return sigs

        price = float(data['close'].iloc[-1])
        ts = data.index[-1]
        pivot_val = key.get('pivot', None)

        for name, val in key.items():
            dist = abs(price - val) / max(1e-9, cur_atr)
            if dist >= self.atr_multiplier:
                # слишком далеко от уровня в терминах ATR — пропускаем
                continue

            is_res = val > price
            if is_res:
                typ = 'SELL_LIMIT'
                entry = float(val * 0.998)
                sl = float(val * 1.02)
            else:
                typ = 'BUY_LIMIT'
                entry = float(val * 1.002)
                sl = float(val * 0.98)

            # тренд-фильтр по центральному pivot:
            # - BUY только выше pivot
            # - SELL только ниже pivot
            if self.use_trend_filter and pivot_val is not None:
                if typ == 'BUY_LIMIT' and price < pivot_val:
                    continue
                if typ == 'SELL_LIMIT' and price > pivot_val:
                    continue

            # кулдаун по уровню для конкретного тикера
            if ticker is not None:
                lvl_key = (ticker, name)
                last_ts = _M7_LAST_HIT.get(lvl_key)
                if last_ts is not None and ts - last_ts < self.cooldown:
                    # по этому уровню уже недавно был сигнал
                    continue
                _M7_LAST_HIT[lvl_key] = ts

            risk = abs(entry - sl)
            tp = entry + (2.0 * risk if not is_res else -2.0 * risk)
            conf = 1.0 - (dist / self.atr_multiplier)
            sigs.append({
                'type': typ,
                'price': round(entry, 4),
                'stop_loss': round(sl, 4),
                'take_profit': round(tp, 4),
                'confidence': round(conf, 2),
                'level': name,
                'level_value': round(val, 4),
                'timestamp': ts
            })
        return sigs

def analyze_asset_m7(ticker, horizon="Краткосрочный", use_ml=False):
    cli = PolygonClient()
    df = cli.daily_ohlc(ticker, days=120)
    if df is None or df.empty:
        return {
            "last_price": 0.0,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": ["Нет данных для M7"],
            "note_html": "<div>M7: ожидание</div>",
            "alt": "Ожидание",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "M7", "grey_zone": True}
        }

    df = df.sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    price = float(df['close'].iloc[-1])
    atr14 = float(_atr_like(df, n=14).iloc[-1]) or 1e-9
    ts_last = df.index[-1]

    if use_ml:
        pred = _M7Predictor(ticker)
        if pred.load():
            ml_action, ml_conf = pred.predict(df, price, atr14)
            strategy = M7TradingStrategy()
            signals = strategy.generate_signals(df, ticker=ticker)

            # если ML говорит WAIT или нет уровней — просто WAIT
            if ml_action == "WAIT" or not signals:
                return {
                    "last_price": price,
                    "recommendation": {"action": "WAIT", "confidence": ml_conf},
                    "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
                    "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
                    "context": [f"ML: нет сильного сигнала ({ml_conf:.2f})"],
                    "note_html": "<div>M7 ML: ожидание</div>",
                    "alt": "ML wait",
                    "entry_kind": "wait",
                    "entry_label": "WAIT",
                    "meta": {
                        "source": "M7",
                        "ml_used": True,
                        "ml_action": ml_action,
                        "ml_conf_raw": ml_conf
                    }
                }

            side = ml_action if ml_action in ("BUY", "SHORT") else None
            if side:
                filt = [
                    s for s in signals
                    if (side == "BUY" and s["type"].startswith("BUY"))
                    or (side == "SHORT" and s["type"].startswith("SELL"))
                ]
                best = (
                    max(filt, key=lambda x: x["confidence"])
                    if filt else max(signals, key=lambda x: x["confidence"])
                )
            else:
                best = max(signals, key=lambda x: x["confidence"])
                side = "BUY" if best["type"].startswith("BUY") else "SHORT"

            # применяем гистерезис по направлению
            filtered_action, dir_meta = _m7_direction_filter(
                ticker=ticker,
                raw_action=side,
                conf=ml_conf,
                ts=ts_last,
            )

            # если flip заблокирован -> отдаём WAIT без новых уровней
            if filtered_action == "WAIT":
                return {
                    "last_price": price,
                    "recommendation": {"action": "WAIT", "confidence": ml_conf},
                    "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
                    "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
                    "context": [
                        f"M7 ML: разворот заблокирован по гистерезису ({dir_meta.get('dir_state')})"
                    ],
                    "note_html": "<div>M7 ML: гистерезис, без нового входа</div>",
                    "alt": "ML wait hysteresis",
                    "entry_kind": "wait",
                    "entry_label": "WAIT",
                    "meta": {
                        "source": "M7",
                        "ml_used": True,
                        "ml_action": side,
                        "ml_conf_raw": ml_conf,
                        "dir_meta": dir_meta,
                        "grey_zone": True
                    }
                }

            # flip разрешён — строим уровни
            entry = float(best['price'])
            sl = float(best['stop_loss'])
            risk = abs(entry - sl)
            if filtered_action == "BUY":
                tp1, tp2, tp3 = (
                    entry + 1.5 * risk,
                    entry + 2.5 * risk,
                    entry + 4.0 * risk
                )
            else:
                tp1, tp2, tp3 = (
                    entry - 1.5 * risk,
                    entry - 2.5 * risk,
                    entry - 4.0 * risk
                )
            u1, u2, u3 = (
                abs(tp1 - entry) / atr14,
                abs(tp2 - entry) / atr14,
                abs(tp3 - entry) / atr14
            )
            k = 0.18
            b1, b2, b3 = ml_conf, max(0.50, ml_conf - 0.08), max(0.45, ml_conf - 0.16)
            p1 = _clip01(b1 * math.exp(-k * (u1 - 1.0)))
            p2 = _clip01(b2 * math.exp(-k * (u2 - 1.5)))
            p3 = _clip01(b3 * math.exp(-k * (u3 - 2.2)))
            probs = _monotone_tp_probs({"tp1": p1, "tp2": p2, "tp3": p3})
            return {
                "last_price": price,
                "recommendation": {"action": filtered_action, "confidence": ml_conf},
                "levels": {
                    "entry": entry,
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "tp3": tp3
                },
                "probs": probs,
                "context": [f"ML {filtered_action} + уровни M7: {best['level']}"],
                "note_html": f"<div>M7 ML: {filtered_action} у {best['level_value']}</div>",
                "alt": "ML‑enhanced M7",
                "entry_kind": "limit",
                "entry_label": best["type"],
                "meta": {
                    "source": "M7",
                    "ml_used": True,
                    "ml_action": side,
                    "ml_conf_raw": ml_conf,
                    "dir_meta": dir_meta
                }
            }

    # вариант без ML
    strategy = M7TradingStrategy()
    signals = strategy.generate_signals(df, ticker=ticker)
    if not signals:
        res = {
            "last_price": price,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": ["Нет сигналов по стратегии M7"],
            "note_html": "<div>M7: ожидание</div>",
            "alt": "Ожидание сигналов от уровней",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "M7", "grey_zone": True}
        }
        try:
            log_agent_performance(
                agent="M7",
                ticker=ticker,
                horizon=horizon,
                action="WAIT",
                confidence=0.50,
                levels=res["levels"],
                probs=res["probs"],
                meta={"probs_debug": {"u": [], "p": []}},
                ts=pd.Timestamp.utcnow().isoformat()
            )
        except Exception:
            pass
        return res

    best = max(signals, key=lambda x: x['confidence'])
    raw_conf = float(_clip01(best['confidence']))
    entry = float(best['price'])
    sl = float(best['stop_loss'])
    risk = abs(entry - sl)
    vol = float(df['close'].pct_change().std() * np.sqrt(252))
    conf_base = 0.50 + 0.34 * math.tanh((raw_conf - 0.65) / 0.20)
    penalty = (
        (0.05 if vol > 0.35 else 0.0)
        + (0.04 if risk / atr14 < 0.8 else 0.0)
        + (0.03 if risk / atr14 > 3.5 else 0.0)
    )
    conf = float(max(0.52, min(0.82, conf_base * (1.0 - penalty))))
    conf = float(_m7_cal()["M7"](conf))

    if best['type'].startswith('BUY'):
        tp1, tp2, tp3 = (
            entry + 1.5 * risk,
            entry + 2.5 * risk,
            entry + 4.0 * risk
        )
        act_raw = "BUY"
    else:
        tp1, tp2, tp3 = (
            entry - 1.5 * risk,
            entry - 2.5 * risk,
            entry - 4.0 * risk
        )
        act_raw = "SHORT"

    # гистерезис по направлению и здесь
    act_filtered, dir_meta = _m7_direction_filter(
        ticker=ticker,
        raw_action=act_raw,
        conf=conf,
        ts=ts_last,
    )

    if act_filtered == "WAIT":
        res = {
            "last_price": price,
            "recommendation": {"action": "WAIT", "confidence": conf},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": [
                f"M7: разворот заблокирован по гистерезису ({dir_meta.get('dir_state')})"
            ],
            "note_html": "<div>M7: гистерезис, без нового входа</div>",
            "alt": "Торговля по M7 (гистерезис)",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "M7", "grey_zone": True, "dir_meta": dir_meta}
        }
        try:
            log_agent_performance(
                agent="M7",
                ticker=ticker,
                horizon=horizon,
                action="WAIT",
                confidence=float(conf),
                levels=res["levels"],
                probs=res["probs"],
                meta={"probs_debug": {"u": [], "p": []}, "dir_meta": dir_meta},
                ts=pd.Timestamp.utcnow().isoformat()
            )
        except Exception:
            pass
        return res

    act = act_filtered
    u1, u2, u3 = (
        abs(tp1 - entry) / atr14,
        abs(tp2 - entry) / atr14,
        abs(tp3 - entry) / atr14
    )
    k = 0.18
    b1, b2, b3 = conf, max(0.50, conf - 0.08), max(0.45, conf - 0.16)
    p1 = _clip01(b1 * math.exp(-k * (u1 - 1.0)))
    p2 = _clip01(b2 * math.exp(-k * (u2 - 1.5)))
    p3 = _clip01(b3 * math.exp(-k * (u3 - 2.2)))
    probs = _monotone_tp_probs({"tp1": p1, "tp2": p2, "tp3": p3})

    try:
        log_agent_performance(
            agent="M7",
            ticker=ticker,
            horizon=horizon,
            action=act,
            confidence=float(conf),
            levels={"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
            probs={
                "tp1": float(probs["tp1"]),
                "tp2": float(probs["tp2"]),
                "tp3": float(probs["tp3"])
            },
            meta={
                "probs_debug": {
                    "u": [float(u1), float(u2), float(u3)],
                    "p": [
                        float(probs["tp1"]),
                        float(probs["tp2"]),
                        float(probs["tp3"])
                    ]
                },
                "dir_meta": dir_meta,
            },
            ts=pd.Timestamp.utcnow().isoformat()
        )
    except Exception:
        pass

    return {
        "last_price": price,
        "recommendation": {"action": act, "confidence": float(conf)},
        "levels": {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs,
        "context": [f"Сигнал от уровня {best['level']}"],
        "note_html": f"<div>M7: {best['type']} на уровне {best['level_value']}</div>",
        "alt": "Торговля по M7",
        "entry_kind": "limit",
        "entry_label": best['type'],
        "meta": {"source": "M7", "grey_zone": bool(0.48 <= conf <= 0.58), "dir_meta": dir_meta}
    }

# -------------------- W7 --------------------
def analyze_asset_w7(ticker: str, horizon: str):
    # --- infra & data ---
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon); hz = cfg["hz"]
    days = max(90, cfg["look"] * 2)

    df = cli.daily_ohlc(ticker, days=days)
    if df is None or df.empty:
        df = pd.DataFrame(columns=["open","high","low","close","volume","timestamp"])
    else:
        df = df.sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")

    # current price with fallbacks
    price = None
    try:
        price = cli.last_trade_price(ticker)
    except Exception:
        price = None
    if price is None:
        try:
            price = cli.prev_close(ticker)
        except Exception:
            price = None
    if price is None and not df.empty:
        try:
            price = float(df["close"].iloc[-1])
        except Exception:
            price = None
    price = float(price) if price is not None else 0.0

    # --- series & indicators ---
    closes  = df["close"]  if "close"  in df.columns else pd.Series(dtype=float)
    volumes = df["volume"] if "volume" in df.columns else pd.Series(dtype=float)

    tail = df.tail(cfg["look"]) if not df.empty else df
    rng_low  = float(tail["low"].min())  if not tail.empty else 0.0
    rng_high = float(tail["high"].max()) if not tail.empty else 0.0
    rng_w = max(1e-9, rng_high - rng_low)
    pos = (price - rng_low) / rng_w if rng_w > 0 else 0.0

    slope = _linreg_slope(closes.tail(cfg["trend"]).values) if not closes.empty else 0.0
    slope_norm = slope / max(1e-9, price if price else 1.0)

    atr_d = float(_atr_like(df, n=cfg["atr"]).iloc[-1]) if not df.empty else 0.0
    atr_w = _weekly_atr(df) if (not df.empty and cfg.get("use_weekly_atr")) else atr_d
    vol_ratio = (atr_d / max(1e-9, float(_atr_like(df, n=cfg["atr"] * 2).iloc[-1]))) if not df.empty else 1.0
    atr_regime = 'low' if vol_ratio < 0.8 else ('mid' if vol_ratio <= 1.2 else 'high')

    ha = _heikin_ashi(df) if not df.empty else pd.DataFrame(columns=["ha_close"])
    ha_diff = ha["ha_close"].diff() if "ha_close" in ha.columns else pd.Series(dtype=float)
    ha_up_run   = _streak_by_sign(ha_diff, True)  if not ha_diff.empty else 0
    ha_down_run = _streak_by_sign(ha_diff, False) if not ha_diff.empty else 0

    _, _, hist = _macd_hist(closes) if not closes.empty else (None,None,pd.Series(dtype=float))
    macd_pos_run = _streak_by_sign(hist, True)  if not hist.empty else 0
    macd_neg_run = _streak_by_sign(hist, False) if not hist.empty else 0

    last_row = df.iloc[-1] if not df.empty else pd.Series({"open":0,"high":0,"low":0,"close":price, "volume":0})
    body, up_wick, dn_wick = _wick_profile(last_row)
    avg_volume = float(volumes.tail(20).mean()) if not volumes.empty else 0.0
    volume_confirm = (last_row.get("volume", 0) > 1.2 * avg_volume) if not volumes.empty else True

    hlc = _last_period_hlc(df, cfg["pivot_rule"]) if not df.empty else None
    if not hlc and not df.empty:
        hlc = (float(df["high"].tail(60).max()), float(df["low"].tail(60).min()), float(df["close"].iloc[-1]))
    elif not hlc:
        hlc = (0.0, 0.0, price)
    H, L, C = hlc
    piv = _fib_pivots(H, L, C)
    P, R1, R2, S1, S2 = piv["P"], piv["R1"], piv.get("R2"), piv["S1"], piv.get("S2")

    # --- regime & near-level logic (исходная логика) ---
    tol_k = {"ST": 0.18, "MID": 0.22, "LT": 0.28}[hz]
    step_d, step_w = atr_d, atr_w
    buf = tol_k * (step_w if hz != "ST" else step_d)
    def _near_from_below(level: float) -> bool: return (level is not None) and (0 <= level - price <= buf)
    def _near_from_above(level: float) -> bool: return (level is not None) and (0 <= price - level <= buf)
    thr_ha   = {"ST": 4, "MID": 5, "LT": 6}[hz]
    thr_macd = {"ST": 4, "MID": 6, "LT": 8}[hz]
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

    # --- helpers ---
    def _is_crypto(sym: str) -> bool:
        s = sym.upper()
        crypto_keys = ["BTC","ETH","SOL","DOGE","ADA","XRP"]
        return any(k in s for k in crypto_keys) or s.endswith("USDT") or s.endswith("USDC")

    u_base = step_w if hz != "ST" else step_d
    base_buf = {"ST": 0.25, "MID": 0.30, "LT": 0.35}[hz]
    regime_adj = {"low": -0.05, "mid": 0.0, "high": +0.10}[atr_regime]
    if _is_crypto(ticker):
        base_buf += 0.05
        if atr_regime == 'high':
            regime_adj += 0.05
    k_buf = max(0.2, base_buf + regime_adj + 0.05 * _clip01(vol_ratio - 1.0))
    atr_buf = k_buf * u_base

    base_thr = {"ST": 0.60, "MID": 0.80, "LT": 1.00}[hz]
    regime_adj_thr = {"low": +0.10, "mid": 0.0, "high": -0.10}[atr_regime]
    thr_far = base_thr + regime_adj_thr
    if _is_crypto(ticker):
        thr_far = {"ST": 0.50, "MID": 0.70, "LT": 0.90}[hz] + regime_adj_thr

    # --- ATR% filter with crypto-aware thresholds ---
    atr_pct = (atr_d / price) * 100 if price > 0 else 0.0
    if _is_crypto(ticker):
        # более высокие пороги для крипты, учитывая её типичную волатильность
        crit = {"ST": 10.0, "MID": 12.0, "LT": 15.0}[hz]
    else:
        crit = 5.0

    if atr_pct > crit:
        action = "WAIT"
        alt = f"Экстремальная волатильность (>{crit:.1f}% ATR) — ждём стабилизации"
    else:
        alt = ""

    # --- default-safe init to avoid UnboundLocalError ---
    entry = price
    sl    = price - 0.90 * step_d
    tp1, tp2, tp3 = entry + 0.7*step_d, entry + 1.4*step_d, entry + 2.1*step_d
    alert_entry = entry

    # --- levels & risk targets ---
    if action == "BUY" and volume_confirm:
        ref = S1 if (price < P) else P
        entry = max(price, (ref if ref is not None else price) + atr_buf)
        if price < P:
            sl = max(price * 0.95, (S1 if S1 is not None else price) - (0.60 if atr_regime != 'high' else 0.80) * step_w)
        else:
            sl = max(price * 0.95, (P if P is not None else price) - (0.60 if atr_regime != 'high' else 0.80) * step_w)
        tp1, tp2, tp3 = entry + 0.9*step_w, entry + 1.6*step_w, entry + 2.3*step_w
        alt = "Если продавят ниже и не вернут — ждём возврата"
        alert_entry = entry
    elif action == "SHORT" and volume_confirm:
        if price >= R1:
            k_short_buf = 0.1 if atr_regime == 'low' else 0.2
            entry = min(price, (R1 if R1 is not None else price) - k_short_buf * atr_buf)
            sl = (R1 if R1 is not None else price) + (0.60 if atr_regime != 'high' else 0.80) * step_w
        else:
            entry = price + 0.10*step_d
            sl = price + (1.00 if atr_regime != 'high' else 1.20) * step_d
        tp1, tp2, tp3 = entry - 0.9*step_w, entry - 1.6*step_w, entry - 2.3*step_w
        alt = "Если протолкнут выше и удержат — без погони"
        alert_entry = entry
    else:
        # WAIT: оставить нейтральные уровни, alert_entry = текущий сигнал/цена
        alert_entry = entry

    # --- clamp / floors / sanity ---
    tp1, tp2, tp3 = _clamp_tp_by_trend(action, hz, tp1, tp2, tp3, piv, step_w, slope_norm, macd_pos_run, macd_neg_run)
    atr_for_floor = step_w if hz != "ST" else step_d
    tp1, tp2, tp3 = _apply_tp_floors(entry, sl, tp1, tp2, tp3, action, hz, price, atr_for_floor)
    tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)
    sl,  tp1, tp2, tp3 = _sanity_levels(action, entry, sl, tp1, tp2, tp3, price, step_d, step_w, hz)

    # --- anti-chase gating ---
    def _dist_units(act, px, en):
        if act == "BUY":   return max(0.0, float(en - px)) / max(1e-9, u_base)
        if act == "SHORT": return max(0.0, float(px - en)) / max(1e-9, u_base)
        return 0.0

    ue = _dist_units(action, price, entry)
    def _dist_label(u):
        if u < 0.40: return "near"
        if u < 0.80: return "mid"
        return "far"

    base = 0.55 + 0.12*_clip01(abs(slope_norm)*1800) + 0.08*_clip01((vol_ratio - 0.9)/0.6)
    if action == "WAIT": base -= 0.07
    if action in ("BUY","SHORT") and ue >= 0.80:
        penalty = 0.05 if atr_regime == 'low' else 0.02
        base -= penalty
    conf = float(max(0.55, min(0.90, base))); conf = float(CAL_CONF["W7"](conf))

    if action in ("BUY","SHORT") and ue > thr_far:
        # keep alert_entry as far trigger, convert to WAIT levels
        action = "WAIT"
        far_wait_alt = f"Entry слишком далеко: {ue:.2f} ATR > {thr_far:.2f} ATR — ждём ретеста/сужения"
        alt = (alt + " " + far_wait_alt).strip()
        entry, sl = price, price - 0.90*step_d
        tp1, tp2, tp3 = entry + 0.7*step_d, entry + 1.4*step_d, entry + 2.1*step_d
        tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)
        sl, tp1, tp2, tp3 = _sanity_levels(action, entry, sl, tp1, tp2, tp3, price, step_d, step_w, hz)
        base = max(0.50, base - 0.04); conf = float(CAL_CONF["W7"](max(0.55, min(0.90, base))))

    entry_kind = _entry_kind(action, entry, price, step_d)
    entry_label = {"buy-stop":"Buy STOP","buy-limit":"Buy LIMIT","buy-now":"Buy NOW",
                   "sell-stop":"Sell STOP","sell-limit":"Sell LIMIT","sell-now":"Sell NOW"}.get(entry_kind, "")

    probs = {"tp1": float(_clip01(0.58 + 0.27*(conf - 0.55)/0.35)),
             "tp2": float(_clip01(0.44 + 0.21*(conf - 0.55)/0.35)),
             "tp3": float(_clip01(0.28 + 0.13*(conf - 0.55)/0.35))}
    probs = _monotone_tp_probs(probs)

    # meta/debug
    u1,u2,u3 = abs(tp1-entry)/u_base, abs(tp2-entry)/u_base, abs(tp3-entry)/u_base
    meta_debug = {
        "atr_d": float(atr_d), "atr_w": float(atr_w),
        "slope_norm": float(slope_norm), "pos": float(pos),
        "u":[float(u1),float(u2),float(u3)],
        "p":[float(probs["tp1"]), float(probs["tp2"]), float(probs["tp3"])],
        "entry_distance_atr": float(ue),
        "distance_label": _dist_label(ue),
        "entry_atr_buffer": float(k_buf),
        "thr_far": float(thr_far),
        "is_crypto": bool(_is_crypto(ticker)),
        "atr_regime": atr_regime,
        "volume_confirm": bool(volume_confirm),
        "atr_pct": float(atr_pct),
        "alert_entry": float(alert_entry)
    }
    try:
        log_agent_performance(
            agent="W7", ticker=ticker, horizon=horizon,
            action=action, confidence=float(conf),
            levels={"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3),
                    "alert_entry": float(alert_entry)},
            probs={"tp1": float(probs["tp1"]), "tp2": float(probs["tp2"]), "tp3": float(probs["tp3"])},
            meta={"probs_debug": meta_debug}, ts=pd.Timestamp.utcnow().isoformat()
        )
    except Exception as e:
        logger.warning("perf log W7 failed: %s", e)

    return {
        "last_price": float(price),
        "recommendation": {"action": action, "confidence": float(round(conf, 4))},
        "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3),
                   "alert_entry": float(alert_entry)},
        "probs": probs, "context": [],
        "note_html": "<div>W7: adaptive ATR-pivots breakout w/ regime & volume</div>",
        "alt": alt, "entry_kind": entry_kind, "entry_label": entry_label,
        # grey_zone подтянут под порог 64%
        "meta": {"source":"W7","grey_zone": bool(0.55 <= conf < 0.64), "probs_debug": meta_debug}
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
from typing import Dict, Any
import numpy as np
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

OCTO_WEIGHTS: Dict[str, float] = {
    "Global": 0.20,
    "M7": 0.20,
    "W7": 0.30,
    "AlphaPulse": 0.30,
}

# Порог confidence, ниже которого Octopus НЕ даёт торговый сигнал (WAIT)
CONF_THRESHOLD: float = float(os.getenv("OCTO_CONF_THRESHOLD", "0.64"))

# Минимум голосов в ОДНУ сторону (BUY или SHORT)
# По умолчанию 3 (как вы попросили). Можно переопределить через env OCTO_MIN_SIDE_VOTES.
MIN_SIDE_VOTES: int = int(os.getenv("OCTO_MIN_SIDE_VOTES", "3"))

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _act_to_num(a: str) -> int:
    return 1 if a == "BUY" else (-1 if a == "SHORT" else 0)

def _num_to_act(x: float) -> str:
    if x > 0:
        return "BUY"
    if x < 0:
        return "SHORT"
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
            "alt": "WAIT",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "Octopus", "votes": [], "ratio": 0.0},
        }

    # 2) Активные голоса BUY/SHORT
    active = []
    for k, r in parts.items():
        a = str(r.get("recommendation", {}).get("action", "WAIT")).upper()
        c = float(r.get("recommendation", {}).get("confidence", 0.5))
        w = float(OCTO_WEIGHTS.get(k, 0.20))
        if a in ("BUY", "SHORT"):
            active.append((k, a, _clip01(c), w))

    cnt_buy = sum(1 for (_, a, _, _) in active if a == "BUY")
    cnt_short = sum(1 for (_, a, _, _) in active if a == "SHORT")

    # 3) Strict quorum direction: минимум MIN_SIDE_VOTES в одну сторону
    if (cnt_buy < MIN_SIDE_VOTES and cnt_short < MIN_SIDE_VOTES) or (cnt_buy == cnt_short):
        final_action = "WAIT"
    elif cnt_buy >= MIN_SIDE_VOTES and cnt_buy > cnt_short:
        final_action = "BUY"
    elif cnt_short >= MIN_SIDE_VOTES and cnt_short > cnt_buy:
        final_action = "SHORT"
    else:
        final_action = "WAIT"

    # 4) ratio / scores (контекст)
    score_long  = sum(w * c for (_, a, c, w) in active if a == "BUY")
    score_short = sum(w * c for (_, a, c, w) in active if a == "SHORT")
    total_side  = score_long + score_short
    delta = abs(score_long - score_short)
    ratio = delta / max(1e-6, total_side)

    # ---- утилиты ----
    def _safe_price(x) -> float:
        try:
            v = float(x)
            return v if np.isfinite(v) and v > 0 else np.nan
        except Exception:
            return np.nan

    def _median_last_price() -> float:
        vals = []
        for r in parts.values():
            v = _safe_price(r.get("last_price", np.nan))
            if np.isfinite(v):
                vals.append(v)
        return float(np.median(vals)) if vals else 0.0

    def _median_levels(direction: str) -> Dict[str, float]:
        L = [
            r.get("levels", {})
            for r in parts.values()
            if str(r.get("recommendation", {}).get("action", "")).upper() == direction
        ]
        if not L:
            return {"entry": 0.0, "sl": 0.0, "tp1": 0.0, "tp2": 0.0, "tp3": 0.0}

        def _safe_median(key: str) -> float:
            vals = []
            for x in L:
                v = x.get(key, None)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    vals.append(float(v))
            return float(np.median(vals)) if vals else 0.0

        return {
            "entry": _safe_median("entry"),
            "sl": _safe_median("sl"),
            "tp1": _safe_median("tp1"),
            "tp2": _safe_median("tp2"),
            "tp3": _safe_median("tp3"),
        }

    def _pick_win_agent_for_probs(direction: str):
        # победитель нужен только как источник probs (как и было раньше)
        cand = [t for t in active if t[1] == direction]
        if cand:
            return max(cand, key=lambda t: t[2] * t[3])[0]
        if active:
            return max(active, key=lambda t: t[2] * t[3])[0]
        return None

    # 5) Считаем выходы
    if final_action in ("BUY", "SHORT"):
        win_agent = _pick_win_agent_for_probs(final_action)

        # NEW: уровни всегда по медиане сторонников финальной стороны
        levels_out = _median_levels(final_action)

        probs_src = parts.get(win_agent, {}).get("probs", {}) if win_agent else {}
        probs_out = _monotone_tp_probs(probs_src or {})

        # Итоговый confidence: средневзвешенно по сторонникам + штраф конфликта
        side_items = []
        for k, r in parts.items():
            rec = r.get("recommendation", {})
            if str(rec.get("action", "")).upper() == final_action:
                w = float(OCTO_WEIGHTS.get(k, 0.20))
                c = _clip01(rec.get("confidence", 0.5))
                side_items.append((w, c))

        if side_items:
            overall_conf = sum(w * c for w, c in side_items) / max(1e-6, sum(w for w, _ in side_items))
        else:
            overall_conf = 0.50

        score_side = score_long if final_action == "BUY" else score_short
        score_opp  = score_short if final_action == "BUY" else score_long

        try:
            beta = float(os.getenv("OCTO_CONF_BETA", "0.35"))
        except Exception:
            beta = 0.35

        penalty = 1.0 - beta * (score_opp / max(1e-6, score_side))
        penalty = max(0.70, min(1.00, penalty))

        overall_conf = float(CAL_CONF["Octopus"](_clip01(overall_conf * penalty)))

    else:
        win_agent = None
        levels_out = {"entry": 0.0, "sl": 0.0, "tp1": 0.0, "tp2": 0.0, "tp3": 0.0}
        probs_out  = {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0}
        overall_conf = float(CAL_CONF["Octopus"](0.50))

    # 5.3) Фильтр по порогу confidence
    if final_action in ("BUY", "SHORT") and overall_conf < CONF_THRESHOLD:
        final_action = "WAIT"
        win_agent = None
        levels_out = {"entry": 0.0, "sl": 0.0, "tp1": 0.0, "tp2": 0.0, "tp3": 0.0}
        probs_out  = {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0}

    # 6) Сбор ответа и логирование
    last_price = _median_last_price()

    votes_txt = [
        {
            "agent": k,
            "action": str(r.get("recommendation", {}).get("action", "")).upper(),
            "confidence": _clip01(float(r.get("recommendation", {}).get("confidence", 0.0))),
        }
        for k, r in parts.items()
    ]

    res = {
        "last_price": float(last_price),
        "recommendation": {"action": final_action, "confidence": float(overall_conf)},
        "levels": levels_out,
        "probs": probs_out,
        "context": [
            f"Octopus: ratio={ratio:.2f}, votes={cnt_buy}L/{cnt_short}S, conf_thresh={CONF_THRESHOLD:.2f}, min_side_votes={MIN_SIDE_VOTES}"
        ],
        "note_html": f"<div>Octopus: {final_action} с {overall_conf:.0%}</div>",
        "alt": "Octopus",
        "entry_kind": "market" if final_action != "WAIT" else "wait",
        "entry_label": final_action if final_action != "WAIT" else "WAIT",
        "meta": {
            "source": "Octopus",
            "votes": votes_txt,
            "ratio": float(ratio),
            "conf_threshold": float(CONF_THRESHOLD),
            "min_side_votes": int(MIN_SIDE_VOTES),
            "buy_votes": int(cnt_buy),
            "short_votes": int(cnt_short),
            "levels_source": "median",
            "probs_source_agent": win_agent or "",
        },
    }

    try:
        log_agent_performance(
            agent="Octopus",
            ticker=ticker,
            horizon=horizon,
            action=final_action,
            confidence=float(overall_conf),
            levels=levels_out,
            probs=probs_out,
            meta={
                "votes": votes_txt,
                "ratio": float(ratio),
                "conf_threshold": float(CONF_THRESHOLD),
                "min_side_votes": int(MIN_SIDE_VOTES),
                "buy_votes": int(cnt_buy),
                "short_votes": int(cnt_short),
                "levels_source": "median",
                "probs_source_agent": win_agent or "",
            },
            ts=pd.Timestamp.utcnow().isoformat(),
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
