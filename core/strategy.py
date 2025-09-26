# core/strategy.py
from __future__ import annotations

import os
import hashlib
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging
from datetime import datetime, timedelta

# Мягкие импорты внешних модулей, чтобы UI не падал
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

# Логирование
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
    # Формулы соответствуют общепринятому определению Heikin Ashi: HA_Close=(O+H+L+C)/4; HA_Open=(prev_HA_Open+prev_HA_Close)/2 [1] [2]
    ha = pd.DataFrame(index=df.index.copy())
    ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    ha_open = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2.0]
    for i in range(1, len(df)):
        ha_open.append((ha_open[-1] + ha["ha_close"].iloc[i-1]) / 2.0)
    ha["ha_open"] = pd.Series(ha_open, index=df.index)
    return ha  # [web:191][web:195]

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

# -------------------- horizons & pivots --------------------
def _horizon_cfg(text: str):
    if "Кратко" in text:
        return dict(look=60, trend=14, atr=14, pivot_rule="W-FRI", use_weekly_atr=False, hz="ST")
    if "Средне" in text:
        return dict(look=120, trend=28, atr=14, pivot_rule="M", use_weekly_atr=True, hz="MID")
    return dict(look=240, trend=56, atr=14, pivot_rule="M", use_weekly_atr=True, hz="LT")

def _last_period_hlc(df: pd.DataFrame, rule: str):
    g = df.resample(rule).agg({"high":"max","low":"min","close":"last"}).dropna()
    if len(g) < 2:
        return None
    row = g.iloc[-2]
    return float(row["high"]), float(row["low"]), float(row["close"])

def _fib_pivots(H: float, L: float, C: float):
    P = (H + L + C) / 3.0
    d = H - L
    R1 = P + 0.382 * d
    R2 = P + 0.618 * d
    R3 = P + 1.000 * d
    S1 = P - 0.382 * d
    S2 = P - 0.618 * d
    S3 = P - 1.000 * d
    return {"P":P, "R1":R1, "R2":R2, "R3":R3, "S1":S1, "S2":S2, "S3":S3}

def _classify_band(price: float, piv: dict, buf: float) -> int:
    P, R1 = piv["P"], piv["R1"]
    R2, R3, S1, S2 = piv.get("R2"), piv.get("R3"), piv["S1"], piv.get("S2")
    neg_inf, pos_inf = -1e18, 1e18
    levels = [
        S2 if S2 is not None else neg_inf, 
        S1, 
        P, 
        R1,
        R2 if R2 is not None else pos_inf, 
        R3 if R3 is not None else pos_inf
    ]
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
    body = max(1e-9, abs(c - o))
    up_wick = max(0.0, h - max(o, c))
    dn_wick = max(0.0, min(o, c) - l)
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

# -------------------- TP/SL guards --------------------
def _apply_tp_floors(entry: float, sl: float, tp1: float, tp2: float, tp3: float,
                    action: str, hz_tag: str, price: float, atr_val: float):
    if action not in ("BUY", "SHORT"): return tp1, tp2, tp3
    risk = abs(entry - sl)
    if risk <= 1e-9: return tp1, tp2, tp3
    side = 1 if action == "BUY" else -1
    min_rr = {"ST": 0.80, "MID": 1.00, "LT": 1.20}
    min_pct = {"ST": 0.006, "MID": 0.012, "LT": 0.018}
    atr_mult = {"ST": 0.50, "MID": 0.80, "LT": 1.20}
    floor1 = max(min_rr[hz_tag] * risk, min_pct[hz_tag] * price, atr_mult[hz_tag] * atr_val)
    if abs(tp1 - entry) < floor1: tp1 = entry + side * floor1
    floor2 = max(1.6 * floor1, min_rr[hz_tag] * 1.8 * risk)
    if abs(tp2 - entry) < floor2: tp2 = entry + side * floor2
    min_gap3 = max(0.8 * floor1, 0.6 * risk)
    if abs(tp3 - tp2) < min_gap3: tp3 = tp2 + side * min_gap3
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

def _clamp_tp_by_trend(action: str, hz: str,
                      tp1: float, tp2: float, tp3: float,
                      piv: dict, step_w: float,
                      slope_norm: float, macd_pos_run: int, macd_neg_run: int):
    thr_macd = {"ST": 3, "MID": 5, "LT": 6}[hz]
    bullish = (slope_norm > 0.0006) and (macd_pos_run >= thr_macd)
    bearish = (slope_norm < -0.0006) and (macd_neg_run >= thr_macd)
    P, R1, S1 = piv["P"], piv["R1"], piv["S1"]
    if action == "SHORT" and bullish:
        limit = max(R1 - 1.2 * step_w, (P + R1) / 2.0)
        tp1 = max(tp1, limit - 0.2 * step_w); tp2 = max(tp2, limit); tp3 = max(tp3, limit + 0.4 * step_w)
    if action == "BUY" and bearish:
        limit = min(S1 + 1.2 * step_w, (P + S1) / 2.0)
        tp1 = min(tp1, limit + 0.2 * step_w); tp2 = min(tp2, limit); tp3 = min(tp3, limit - 0.4 * step_w)
    return tp1, tp2, tp3

def _sanity_levels(action: str, entry: float, sl: float,
                  tp1: float, tp2: float, tp3: float,
                  price: float, step_d: float, step_w: float, hz: str):
    side = 1 if action == "BUY" else -1
    min_tp_gap = {"ST": 0.40, "MID": 0.70, "LT": 1.10}[hz] * step_w
    min_tp_pct = {"ST": 0.004, "MID": 0.009, "LT": 0.015}[hz] * price
    floor_gap = max(min_tp_gap, min_tp_pct, 0.35 * abs(entry - sl) if sl != entry else 0.0)
    if action == "BUY" and sl >= entry - 0.25 * step_d:
        sl = entry - max(0.60 * step_w, 0.90 * step_d)
    if action == "SHORT" and sl <= entry + 0.25 * step_d:
        sl = entry + max(0.60 * step_w, 0.90 * step_d)
    def _push_tp(tp, rank):
        need = floor_gap * (1.0 if rank == 1 else (1.6 if rank == 2 else 2.2))
        want = entry + side * need
        if side * (tp - entry) <= 0: return want
        if abs(tp - entry) < need:   return want
        return tp
    tp1 = _push_tp(tp1, 1); tp2 = _push_tp(tp2, 2); tp3 = _push_tp(tp3, 3)
    tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)
    return sl, tp1, tp2, tp3

# -------------------- Калибровка и ECE --------------------
class ConfidenceCalibrator:
    """
    Плагин для калибровки уверенности: identity | sigmoid (Platt) | isotonic, совместимо со стандартной практикой [web:168].
    """
    def __init__(self, method: str = "identity", params: Optional[Dict[str, float]] = None):
        self.method = method
        self.params = params or {}

    def __call__(self, p: float) -> float:
        p = _clip01(float(p))
        if self.method == "sigmoid":
            a = float(self.params.get("a", 1.0))
            b = float(self.params.get("b", 0.0))
            return _clip01(float(1.0 / (1.0 + np.exp(-(a * p + b)))))
        if self.method == "isotonic":
            knots = sorted(self.params.get("knots", [(0.0, 0.0), (1.0, 1.0)]))
            for i in range(1, len(knots)):
                x0, y0 = knots[i - 1]; x1, y1 = knots[i]
                if p <= x1:
                    if x1 == x0:
                        return _clip01(float(y1))
                    t = (p - x0) / (x1 - x0)
                    return _clip01(float(y0 + t * (y1 - y0)))
            return _clip01(float(knots[-1][1]))
        return p  # identity

def _ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    # Expected Calibration Error на равных бинах [web:168]
    edges = np.linspace(0.0, 1.0, bins + 1)
    N = len(labels)
    ece = 0.0
    for i in range(bins):
        mask = (probs > edges[i]) & (probs <= edges[i + 1])
        if not np.any(mask):
            continue
        conf = float(probs[mask].mean())
        acc = float((labels[mask] == 1).mean())
        ece += (mask.sum() / N) * abs(acc - conf)
    return float(ece)

# -------------------- ML Model для M7 Strategy --------------------
class M7MLModel:
    """
    Безопасная интеграция ML для M7:
    - Не обучаем в проде, только грузим веса; при отсутствии — возвращаем None и стратегия уходит в правила.
    - Для уверенности используем predict_proba, иначе decision_function/predict как прокси.
    """
    def __init__(self):
        self.model = None
        self.scaler = None  # опционально
        self.local_model_path = "models/m7_model.pkl"
        self.local_scaler_path = "models/m7_scaler.pkl"

    def _try_local_load(self):
        try:
            import joblib
            if os.path.exists(self.local_model_path):
                self.model = joblib.load(self.local_model_path)
            if os.path.exists(self.local_scaler_path):
                self.scaler = joblib.load(self.local_scaler_path)
        except Exception as e:
            logger.warning("Local ML load failed: %s", e)

    def _try_repo_loader(self, ticker: str):
        try:
            from core.model_loader import load_model_for
            m = load_model_for(ticker)
            if m is not None:
                self.model = m
        except Exception as e:
            logger.warning("Repo model_loader failed: %s", e)

    def prepare_features(self, df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        try:
            X = pd.DataFrame(index=df.index.copy())
            X["returns"]   = df["close"].pct_change()
            X["volatility"]= X["returns"].rolling(20).std()
            X["momentum"]  = df["close"] / df["close"].shift(5) - 1
            X["sma_20"]    = df["close"].rolling(20).mean()
            X["sma_50"]    = df["close"].rolling(50).mean()
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs   = gain / loss
            X["rsi"] = 100 - (100 / (1 + rs))
            X["volume_ma"]   = df["volume"].rolling(20).mean()
            X["volume_ratio"]= df["volume"] / X["volume_ma"]
            X = X.dropna().tail(1)
            return X if len(X) else None
        except Exception as e:
            logger.warning("Feature build failed: %s", e)
            return None

    def predict_signal(self, df: pd.DataFrame, ticker: str) -> Optional[Dict[str, float]]:
        if self.model is None:
            self._try_repo_loader(ticker)
            if self.model is None:
                self._try_local_load()
        if self.model is None:
            return None

        X = self.prepare_features(df, ticker)
        if X is None:
            return None

        X_in = X.values
        if self.scaler is not None:
            try:
                X_in = self.scaler.transform(X.values)
            except Exception:
                pass

        p_long = None
        try:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(X_in)
                p_long = float(np.ravel(proba[:, 1])[0])
            elif hasattr(self.model, "decision_function"):
                margin = float(np.ravel(self.model.decision_function(X_in))[0])
                p_long = float(1.0 / (1.0 + np.exp(-margin)))
            elif hasattr(self.model, "predict"):
                point = float(np.ravel(self.model.predict(X_in))[0])
                p_long = float(np.tanh(abs(point)))
            else:
                return None
        except Exception as e:
            logger.warning("ML predict failed: %s", e)
            return None

        ai_conf = float(max(0.0, min(1.0, p_long)))
        return {"p_long": ai_conf, "confidence": float(0.5 + (ai_conf - 0.5))}

# -------------------- Global Strategy --------------------
def analyze_asset_global(ticker: str, horizon: str = "Краткосрочный"):
    cli = PolygonClient()
    days = 90
    df = cli.daily_ohlc(ticker, days=days)
    current_price = float(df['close'].iloc[-1])
    returns = np.log(df['close'] / df['close'].shift(1))
    hist_volatility = returns.std() * np.sqrt(252)
    short_ma = df['close'].rolling(20).mean().iloc[-1]
    long_ma  = df['close'].rolling(50).mean().iloc[-1]
    if short_ma > long_ma:
        action, confidence = "BUY", 0.69
    else:
        action, confidence = "SHORT", 0.65
    atr = float(_atr_like(df, n=14).iloc[-1])
    if action == "BUY":
        entry = current_price; sl = current_price - 2 * atr
        tp1 = current_price + 1 * atr; tp2 = current_price + 2 * atr; tp3 = current_price + 3 * atr
        alt = "Покупка по рынку с консервативными целями"
    else:
        entry = current_price; sl = current_price + 2 * atr
        tp1 = current_price - 1 * atr; tp2 = current_price - 2 * atr; tp3 = current_price - 3 * atr
        alt = "Продажа по рынку с консервативными целями"
    context = [f"Волатильность: {hist_volatility:.2%}", f"Тренд: {'Бычий' if action == 'BUY' else 'Медвежий'}"]
    probs = {"tp1": 0.68, "tp2": 0.52, "tp3": 0.35}
    note_html = f"<div style='margin-top:10px; opacity:0.95;'>Global Strategy: {action} сигнал с уверенностью {confidence:.0%}. Консервативные тейк-профиты на основе волатильности.</div>"
    return {
        "last_price": current_price,
        "recommendation": {"action": action, "confidence": confidence},
        "levels": {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs,
        "context": context,
        "note_html": note_html,
        "alt": alt,
        "entry_kind": "market",
        "entry_label": f"{action} NOW"
    }

# -------------------- M7 Strategy --------------------
class M7TradingStrategy:
    def __init__(self, atr_period=14, atr_multiplier=1.5, pivot_period='D', 
                 fib_levels=[0.236, 0.382, 0.5, 0.618, 0.786]):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.pivot_period = pivot_period
        self.fib_levels = fib_levels

    def calculate_pivot_points(self, high, low, close):
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        return {'pivot': pivot, 'r1': r1, 'r2': r2, 'r3': r3, 's1': s1, 's2': s2, 's3': s3}

    def calculate_fib_levels(self, high, low):
        diff = high - low
        fib_levels = {}
        for level in self.fib_levels:
            fib_levels[f'fib_{int(level*1000)}'] = high - level * diff
        return fib_levels

    def identify_key_levels(self, data):
        grouped = data.resample('D') if self.pivot_period == 'D' else data.resample('W')
        key_levels = {}
        for _, group in grouped:
            if len(group) > 0:
                high = group['high'].max(); low = group['low'].min(); close = group['close'].iloc[-1]
                pivot_levels = self.calculate_pivot_points(high, low, close)
                fib_levels   = self.calculate_fib_levels(high, low)
                for name, value in {**pivot_levels, **fib_levels}.items():
                    key_levels[name] = value
        return key_levels

    def generate_signals(self, data):
        signals = []
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return signals
        data['atr'] = _atr_like(data, self.atr_period)
        current_atr = data['atr'].iloc[-1]
        key_levels = self.identify_key_levels(data)
        current_price = data['close'].iloc[-1]
        current_time  = data.index[-1]
        for level_name, level_value in key_levels.items():
            distance = abs(current_price - level_value) / max(1e-9, current_atr)
            if distance < self.atr_multiplier:
                is_resistance = level_value > current_price
                if is_resistance:
                    signal_type = 'SELL_LIMIT'
                    entry_price = level_value * 0.998
                    stop_loss   = level_value * 1.02
                    take_profit = level_value * 0.96
                else:
                    signal_type = 'BUY_LIMIT'
                    entry_price = level_value * 1.002
                    stop_loss   = level_value * 0.98
                    take_profit = level_value * 1.04
                confidence = 1 - (distance / self.atr_multiplier)
                signals.append({
                    'type': signal_type, 'price': round(entry_price, 4),
                    'stop_loss': round(stop_loss, 4), 'take_profit': round(take_profit, 4),
                    'confidence': round(confidence, 2), 'level': level_name,
                    'level_value': round(level_value, 4), 'timestamp': current_time
                })
        return signals

def analyze_asset_m7(ticker, horizon="Краткосрочный", use_ml=True):
    cli = PolygonClient()
    days = 120
    df = cli.daily_ohlc(ticker, days=days)
    returns = np.log(df['close'] / df['close'].shift(1))
    hist_volatility = returns.std() * np.sqrt(252)

    strategy = M7TradingStrategy()
    signals = strategy.generate_signals(df)
    if not signals:
        return {
            "last_price": float(df['close'].iloc[-1]),
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0, "tp2": 0, "tp3": 0},
            "context": ["Нет сигналов по стратегии M7"],
            "note_html": "<div>Ожидание сигналов от стратегии M7</div>",
            "alt": "Ожидание четких сигналов от уровней Pivot и Fibonacci",
            "entry_kind": "wait",
            "entry_label": "WAIT"
        }

    best_signal = max(signals, key=lambda x: x['confidence'])

    # Интеграция ML — безопасная, без обучения в проде
    ml_confidence = None
    if use_ml:
        try:
            ml_model = M7MLModel()
            ml_prediction = ml_model.predict_signal(df, ticker)
            if ml_prediction:
                base_conf = float(best_signal['confidence'])
                ml_conf   = float(ml_prediction['confidence'])
                ml_confidence = ml_conf
                combined = 0.7 * ml_conf + 0.3 * base_conf
                best_signal['confidence'] = float(min(0.95, max(0.50, combined)))
                if ml_prediction['p_long'] > 0.6 and best_signal['type'].startswith('SELL'):
                    best_signal['type'] = 'BUY_LIMIT'
                    best_signal['confidence'] = float(max(0.50, best_signal['confidence'] * 0.9))
                elif ml_prediction['p_long'] < 0.4 and best_signal['type'].startswith('BUY'):
                    best_signal['type'] = 'SELL_LIMIT'
                    best_signal['confidence'] = float(max(0.50, best_signal['confidence'] * 0.9))
        except Exception as e:
            logger.warning("M7pro ML integration warning: %s", e)

    current_price = float(df['close'].iloc[-1])
    entry_price = best_signal['price']
    stop_loss   = best_signal['stop_loss']
    risk = abs(entry_price - stop_loss)
    volatility = df['close'].pct_change().std() * np.sqrt(252)
    max_daily_move = current_price * volatility / np.sqrt(252)

    if best_signal['type'].startswith('BUY'):
        tp1 = min(entry_price + risk * 1.5, entry_price + max_daily_move * 2)
        tp2 = min(entry_price + risk * 2.5, entry_price + max_daily_move * 3)
        tp3 = min(entry_price + risk * 4.0, entry_price + max_daily_move * 5)
    else:
        tp1 = max(entry_price - risk * 1.5, entry_price - max_daily_move * 2)
        tp2 = max(entry_price - risk * 2.5, entry_price - max_daily_move * 3)
        tp3 = max(entry_price - risk * 4.0, entry_price - max_daily_move * 5)

    action = "BUY" if best_signal['type'].startswith('BUY') else "SHORT"
    context = [f"Сигнал от уровня {best_signal['level']}"]
    if ml_confidence is not None:
        context.append(f"ML уверенность: {ml_confidence:.0%}")
    probs = {"tp1": 0.63, "tp2": 0.52, "tp3": 0.53}
    note_html = f"<div style='margin-top:10px; opacity:0.95;'>M7 Strategy: {best_signal['type']} на уровне {best_signal['level_value']}. ML-улучшенная стратегия с точными тейк-профитами.</div>"

    return {
        "last_price": current_price,
        "recommendation": {"action": action, "confidence": float(best_signal['confidence'])},
        "levels": {"entry": entry_price, "sl": stop_loss, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs,
        "context": context,
        "note_html": note_html,
        "alt": "Торговля по стратегии M7 с ML-улучшением",
        "entry_kind": "limit",
        "entry_label": best_signal['type']
    }

# -------------------- W7 Strategy (как было) --------------------
def analyze_asset_w7(ticker: str, horizon: str):
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)
    hz = cfg["hz"]
    days = max(90, cfg["look"] * 2)
    df = cli.daily_ohlc(ticker, days=days)
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
    ha = _heikin_ashi(df)
    ha_diff = ha["ha_close"].diff()
    ha_up_run = _streak_by_sign(ha_diff, True)
    ha_down_run = _streak_by_sign(ha_diff, False)
    macd, sig, hist = _macd_hist(closes)
    macd_pos_run = _streak_by_sign(hist, True)
    macd_neg_run = _streak_by_sign(hist, False)
    last_row = df.iloc[-1]
    body, up_wick, dn_wick = _wick_profile(last_row)
    long_upper = (up_wick > body * 1.3) and (up_wick > dn_wick * 1.1)
    long_lower = (dn_wick > body * 1.3) and (dn_wick > up_wick * 1.1)
    hlc = _last_period_hlc(df, cfg["pivot_rule"])
    if not hlc:
        hlc = (float(df["high"].tail(60).max()), float(df["low"].tail(60).min()), float(df["close"].iloc[-1]))
    H, L, C = hlc
    piv = _fib_pivots(H, L, C)
    P, R1, R2, S1, S2 = piv["P"], piv["R1"], piv.get("R2"), piv["S1"], piv.get("S2")
    tol_k = {"ST": 0.18, "MID": 0.22, "LT": 0.28}[hz]
    buf = tol_k * (atr_w if hz != "ST" else atr_d)
    def _near_from_below(level: float) -> bool:
        return (level is not None) and (0 <= level - price <= buf)
    def _near_from_above(level: float) -> bool:
        return (level is not None) and (0 <= price - level <= buf)
    thr_ha = {"ST": 4, "MID": 5, "LT": 6}[hz]
    thr_macd = {"ST": 4, "MID": 6, "LT": 8}[hz]
    long_up = (ha_up_run >= thr_ha) or (macd_pos_run >= thr_macd)
    long_down = (ha_down_run >= thr_ha) or (macd_neg_run >= thr_macd)
    if long_up and (_near_from_below(S1) or _near_from_below(R1) or _near_from_below(R2)):
        action, scenario = "WAIT", "stall_after_long_up_at_pivot"
    elif long_down and (_near_from_above(R1) or _near_from_above(S1) or _near_from_above(S2)):
        action, scenario = "WAIT", "stall_after_long_down_at_pivot"
    else:
        band = _classify_band(price, piv, buf)
        very_high_pos = pos >= 0.80
        if very_high_pos:
            if (R2 is not None) and (price > R2 + 0.6 * buf) and (slope_norm > 0):
                action, scenario = "BUY", "breakout_up"
            else:
                action, scenario = "SHORT", "fade_top"
        else:
            if band >= +2:
                if (R2 is not None) and (price > R2 + 0.6 * buf) and (slope_norm > 0):
                    action, scenario = "BUY", "breakout_up"
                else:
                    action, scenario = "SHORT", "fade_top"
            elif band == +1:
                action, scenario = "WAIT", "upper_wait"
            elif band == 0:
                if slope_norm >= 0:
                    action, scenario = "BUY", "trend_follow"
                else:
                    action, scenario = "WAIT", "mid_range"
            elif band == -1:
                action, scenario = "BUY", "revert_from_bottom"
            else:
                if band <= -2:
                    action, scenario = "BUY", "revert_from_bottom"
                else:
                    action, scenario = "WAIT", "upper_wait"
    base = 0.55 + 0.12 * _clip01(abs(slope_norm) * 1800) + 0.08 * _clip01((vol_ratio - 0.9) / 0.6)
    if action == "WAIT":
        base -= 0.07
    conf = float(max(0.55, min(0.90, base)))

    # optional AI override
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
            p_long = float(out_ai.get("p_long", 0.5))
            th_long = float(os.getenv("ARXORA_AI_TH_LONG", "0.55"))
            th_short = float(os.getenv("ARXORA_AI_TH_SHORT", "0.45"))
            if p_long >= th_long:
                action = "BUY"
                conf = float(max(0.55, min(0.90, 0.55 + 0.35 * (p_long - th_long) / max(1e-9, 1.0 - th_long))))
            elif p_long <= th_short:
                action = "SHORT"
                conf = float(max(0.55, min(0.90, 0.55 + 0.35 * ((th_short - p_long) / max(1e-9, th_short)))))
            else:
                action = "WAIT"
                conf = float(max(0.48, min(0.83, conf - 0.05)))

    step_d, step_w = atr_d, atr_w
    if action == "BUY":
        if price < P:
            entry = max(price, S1 + 0.15 * step_w); sl = S1 - 0.60 * step_w
        else:
            entry = max(price, P + 0.10 * step_w); sl = P - 0.60 * step_w
        tp1 = entry + 0.9 * step_w; tp2 = entry + 1.6 * step_w; tp3 = entry + 2.3 * step_w
        alt = "Если продавят ниже и не вернут — не заходим; ждём возврата и подтверждения сверху."
    elif action == "SHORT":
        if price >= R1:
            entry = min(price, R1 - 0.15 * step_w); sl = R1 + 0.60 * step_w
        else:
            entry = price + 0.10 * step_d; sl = price + 1.00 * step_d
        tp1 = entry - 0.9 * step_w; tp2 = entry - 1.6 * step_w; tp3 = entry - 2.3 * step_w
        alt = "Если протолкнут выше и удержат — без погони; ждём возврата и слабости сверху."
    else:
        entry, sl = price, price - 0.90 * step_d
        tp1, tp2, tp3 = entry + 0.7 * step_d, entry + 1.4 * step_d, entry + 2.1 * step_d
        alt = "Ниже уровня — не пытаюсь догонять; стратегия — ждать пробоя, ретеста или отката к поддержке."

    tp1, tp2, tp3 = _clamp_tp_by_trend(action, hz, tp1, tp2, tp3, piv, step_w, slope_norm, macd_pos_run, macd_neg_run)
    atr_for_floor = atr_w if hz != "ST" else atr_d
    tp1, tp2, tp3 = _apply_tp_floors(entry, sl, tp1, tp2, tp3, action, hz, price, atr_for_floor)
    tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)
    sl, tp1, tp2, tp3 = _sanity_levels(action, entry, sl, tp1, tp2, tp3, price, step_d, step_w, hz)

    entry_kind = _entry_kind(action, entry, price, step_d)
    entry_label = {
        "buy-stop": "Buy STOP", "buy-limit": "Buy LIMIT", "buy-now": "Buy NOW",
        "sell-stop": "Sell STOP","sell-limit": "Sell LIMIT","sell-now": "Sell NOW"
    }.get(entry_kind, "")

    chips = []
    if (atr_d / max(1e-9, float(_atr_like(df, n=cfg["atr"] * 2).iloc[-1]))) > 1.05: chips.append("волатильность растёт")
    if (atr_d / max(1e-9, float(_atr_like(df, n=cfg["atr"] * 2).iloc[-1]))) < 0.95: chips.append("волатильность сжимается")

    thr_ha = {"ST": 4, "MID": 5, "LT": 6}[hz]
    if (_streak_by_sign(_heikin_ashi(df)["ha_close"].diff(), True)  >= thr_ha): chips.append(f"HA зелёных: {_streak_by_sign(_heikin_ashi(df)['ha_close'].diff(), True)}")
    if (_streak_by_sign(_heikin_ashi(df)["ha_close"].diff(), False) >= thr_ha): chips.append(f"HA красных: {_streak_by_sign(_heikin_ashi(df)['ha_close'].diff(), False)}")

    p1 = _clip01(0.58 + 0.27 * (conf - 0.55) / 0.35)
    p2 = _clip01(0.44 + 0.21 * (conf - 0.55) / 0.35)
    p3 = _clip01(0.28 + 0.13 * (conf - 0.55) / 0.35)
    probs = {"tp1": float(p1), "tp2": float(p2), "tp3": float(p3)}

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
        "probs": probs,
        "context": chips,
        "note_html": note_html,
        "alt": alt,
        "entry_kind": entry_kind,
        "entry_label": entry_label,
    }

# -------------------- Оркестратор Octopus (агрегирует Global, M7, W7) --------------------
OCTO_WEIGHTS: Dict[str, float] = {"Global": 0.33, "M7": 0.33, "W7": 0.34}
OCTO_CALIBRATOR = ConfidenceCalibrator(method="sigmoid", params={"a": 1.2, "b": -0.1})  # можно заменить на isotonic

def _act_to_num(a: str) -> int:
    return 1 if a == "BUY" else (-1 if a == "SHORT" else 0)

def _num_to_act(x: float) -> str:
    if x > 0: return "BUY"
    if x < 0: return "SHORT"
    return "WAIT"

def analyze_asset_octopus(ticker: str, horizon: str) -> Dict[str, Any]:
    g = analyze_asset_global(ticker, horizon)
    m = analyze_asset_m7(ticker, horizon)
    w = analyze_asset_w7(ticker, horizon)

    parts = {"Global": g, "M7": m, "W7": w}
    votes = []
    confs = []
    for name, res in parts.items():
        wgt = OCTO_WEIGHTS.get(name, 0.0)
        act = str(res["recommendation"]["action"])
        conf = float(res["recommendation"]["confidence"])
        votes.append(wgt * _act_to_num(act))
        confs.append(wgt * _clip01(conf))

    raw_vote = float(sum(votes))
    raw_conf = float(sum(confs) / max(1e-9, sum(OCTO_WEIGHTS.values())))
    final_action = _num_to_act(raw_vote)
    calibrated = float(OCTO_CALIBRATOR(raw_conf))

    # выбрать уровни победителя (по весу и совпадению действия)
    winner = "W7"
    best_score = -1.0
    for name, res in parts.items():
        score = OCTO_WEIGHTS.get(name, 0.0) * (1.0 if res["recommendation"]["action"] == final_action else 0.0)
        if score > best_score:
            best_score = score
            winner = name
    levels = parts[winner]["levels"]

    # ECE на подвыборке BUY/SHORT (WAIT исключаем)
    probs_vec, labels_vec = [], []
    for name, res in parts.items():
        act = res["recommendation"]["action"]
        if act == "WAIT":
            continue
        probs_vec.append(_clip01(res["recommendation"]["confidence"]))
        labels_vec.append(1 if act == "BUY" else 0)
    ece = _ece(np.array(probs_vec), np.array(labels_vec), bins=10) if len(probs_vec) >= 2 else 0.0  # [web:168]

    context = [f"Orchestrated from Global/M7/W7 with weights {OCTO_WEIGHTS}"]

    return {
        "strategy": "Octopus",
        "agents": ["Global", "M7", "W7"],
        "weights": OCTO_WEIGHTS,
        "signals": {
            "Global": {"action": g["recommendation"]["action"], "confidence": float(g["recommendation"]["confidence"])},
            "M7":     {"action": m["recommendation"]["action"], "confidence": float(m["recommendation"]["confidence"])},
            "W7":     {"action": w["recommendation"]["action"], "confidence": float(w["recommendation"]["confidence"])},
        },
        "action": final_action,
        "confidence_raw": float(raw_conf),
        "confidence": float(calibrated),
        "calibration": {"method": OCTO_CALIBRATOR.method, "params": OCTO_CALIBRATOR.params, "ece": float(ece)},
        "levels": levels,
        "last_price": float(w.get("last_price", g.get("last_price", 0.0))),
        "context": context + g.get("context", [])[:2] + m.get("context", [])[:2],
        "note_html": "<div style='margin-top:10px; opacity:0.95;'>Octopus: взвешенное голосование стратегий Global/M7/W7 с калиброванной уверенностью.</div>",
        "alt": "Агрегированный сигнал Octopus",
        "entry_kind": "market" if final_action != "WAIT" else "wait",
        "entry_label": final_action if final_action != "WAIT" else "WAIT",
        "probs": w.get("probs", {}),
    }

# -------------------- Strategy Router (Registry с алиасом Octopus) --------------------
STRATEGY_REGISTRY: Dict[str, Callable[[str, str], Dict[str, Any]]] = {
    "Global": analyze_asset_global,
    "M7": analyze_asset_m7,
    "W7": analyze_asset_w7,
    "Octopus": analyze_asset_octopus,  # официальный алиас оркестратора по шаблону Registry
}  # [web:137]

def analyze_asset(ticker: str, horizon: str, strategy: str = "Octopus"):
    fn = STRATEGY_REGISTRY.get(strategy)
    if not fn:
        raise ValueError(f"Unknown strategy: {strategy}")
    return fn(ticker, horizon)

if __name__ == "__main__":
    strategies = ["Global", "M7", "W7", "Octopus"]
    ticker = "AAPL"
    for strategy in strategies:
        print(f"\n=== {strategy} Strategy Result ===")
        try:
            result = analyze_asset(ticker, "Краткосрочный", strategy)
            print(result)
        except Exception as e:
            print(f"Error testing {strategy} strategy: {e}")

