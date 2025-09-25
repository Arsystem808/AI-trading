# -*- coding: utf-8 -*-
# core/strategy.py — финальная версия: Global, W7, M7 (M7pro) + безопасная ML-интеграция и единый роутер

import os
import re
import hashlib
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Источник данных (мягкий импорт)
try:
    from core.polygon_client import PolygonClient
except Exception:
    class PolygonClient:
        def daily_ohlc(self, ticker: str, days: int = 120) -> pd.DataFrame:
            rng = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
            price = np.cumsum(np.random.randn(days)) + 100
            high = price + np.abs(np.random.randn(days))
            low  = price - np.abs(np.random.randn(days))
            open_ = price + np.random.randn(days)*0.3
            close = price
            vol = np.random.randint(1_000_000, 3_000_000, size=days)
            return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":vol}, index=rng)

        def last_trade_price(self, ticker: str) -> float:
            df = self.daily_ohlc(ticker, days=2)
            return float(df["close"].iloc[-1])

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

def _sync_session(overall_pct: float, rules_pct: float = 44.0) -> None:
    # Согласованность карточки и breakdown через Session State
    try:
        import streamlit as st
        st.session_state["last_overall_conf_pct"] = float(overall_pct)
        st.session_state["last_rules_pct"] = float(rules_pct)
    except Exception:
        pass

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
    row = g.iloc[-2]  # последняя завершённая
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
    if action not in ("BUY", "SHORT"):
        return tp1, tp2, tp3
    risk = abs(entry - sl)
    if risk <= 1e-9:
        return tp1, tp2, tp3
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
        tp1 = max(tp1, limit - 0.2 * step_w)
        tp2 = max(tp2, limit)
        tp3 = max(tp3, limit + 0.4 * step_w)
    if action == "BUY" and bearish:
        limit = min(S1 + 1.2 * step_w, (P + S1) / 2.0)
        tp1 = min(tp1, limit + 0.2 * step_w)
        tp2 = min(tp2, limit)
        tp3 = min(tp3, limit - 0.4 * step_w)
    return tp1, tp2, tp3

def _sanity_levels(action: str, entry: float, sl: float,
                  tp1: float, tp2: float, tp3: float,
                  price: float, step_d: float, step_w: float, hz: str):
    side = 1 if action == "BUY" else -1
    min_tp_gap = {"ST": 0.40, "MID": 0.70, "LT": 1.10}[hz] * step_w
    min_tp_pct = {"ST": 0.004, "MID": 0.009, "LT": 0.015}[hz] * price
    floor_gap = max(min_tp_gap, min_tp_pct, 0.35 * abs(entry - sl) if sl != entry else 0.0)
    if action == "BUY" and sl >= entry - 0.25 * step_d: sl = entry - max(0.60 * step_w, 0.90 * step_d)
    if action == "SHORT" and sl <= entry + 0.25 * step_d: sl = entry + max(0.60 * step_w, 0.90 * step_d)
    def _push_tp(tp, rank):
        need = floor_gap * (1.0 if rank == 1 else (1.6 if rank == 2 else 2.2))
        want = entry + side * need
        if side * (tp - entry) <= 0: return want
        if abs(tp - entry) < need:   return want
        return tp
    tp1 = _push_tp(tp1, 1); tp2 = _push_tp(tp2, 2); tp3 = _push_tp(tp3, 3)
    tp1, tp2, tp3 = _order_targets(entry, tp1, tp2, tp3, action)
    return sl, tp1, tp2, tp3

# -------------------- ML Model (безопасная интеграция для M7) --------------------

class M7MLModel:
    """
    Безопасная ML‑модель для M7/M7pro:
    - Пытается загрузить веса из models/
    - При отсутствии — обучает и сохраняет
    - Предсказание через predict_proba; ошибки не роняют стратегию
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = "models/m7_model.pkl"
        self.scaler_path = "models/m7_scaler.pkl"
        self._load_any_safe()

    def _load_any_safe(self) -> None:
        try:
            from pathlib import Path
            MODELS = Path("models")
            candidates: List[Path] = []
            for t in ("SPY","QQQ","BTCUSD","ETHUSD","AAPL","NVDA"):
                candidates += [
                    MODELS / f"arxora_m7pro_{t}.joblib",
                    MODELS / f"global_{t}.joblib",
                    MODELS / f"octopus_{t}.joblib",
                    MODELS / f"alphapulse_{t}.joblib",
                ]
            candidates += [MODELS / "m7_model.pkl"]
            for p in candidates:
                if p.exists():
                    try:
                        self.model = joblib.load(p)
                        break
                    except Exception as e:
                        logger.warning(f"Joblib load failed for {p}: {e}")
            if os.path.exists(self.scaler_path):
                try:
                    self.scaler = joblib.load(self.scaler_path)
                except Exception as e:
                    logger.warning(f"Scaler load failed: {e}")
        except Exception as e:
            logger.warning(f"Model preload warning: {e}")

    # --- feature engineering ---
    def _rsi(self, series: pd.Series, period=14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _weekly_pivots(self, df: pd.DataFrame) -> Dict[str, float]:
        weekly = df.resample('W-FRI').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
        if len(weekly) < 2:
            return {}
        last = weekly.iloc[-2]
        H, L, C = float(last['high']), float(last['low']), float(last['close'])
        P = (H + L + C) / 3
        R1 = (2 * P) - L
        S1 = (2 * P) - H
        R2 = P + (H - L)
        S2 = P - (H - L)
        return {'P': P, 'R1': R1, 'R2': R2, 'S1': S1, 'S2': S2}

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
        try:
            X = pd.DataFrame(index=df.index.copy())
            X['returns'] = df['close'].pct_change()
            X['volatility'] = X['returns'].rolling(20).std()
            X['momentum'] = df['close'] / df['close'].shift(5) - 1
            X['sma_20'] = df['close'].rolling(20).mean()
            X['sma_50'] = df['close'].rolling(50).mean()
            X['rsi'] = self._rsi(df['close'])
            piv = self._weekly_pivots(df)
            for k, v in piv.items():
                if v and v != 0:
                    X[f'pct_to_{k}'] = (df['close'] - float(v)) / float(v)
            X['volume_ma'] = df['volume'].rolling(20).mean()
            X['volume_ratio'] = df['volume'] / X['volume_ma']
            X = X.dropna()
            recent = X.tail(120).copy()
            target = (df['close'].shift(-5).reindex(recent.index) > df['close'].reindex(recent.index)).astype(int)
            feats = recent
            if len(feats) < 30:
                return None, None, []
            cols = list(feats.columns)
            return feats, target, cols
        except Exception as e:
            logger.warning(f"Feature build failed: {e}")
            return None, None, []

    def train_and_save(self, df: pd.DataFrame, n_estimators=400) -> Optional[Dict[str, Any]]:
        try:
            X, y, cols = self._prepare_features(df)
            if X is None or len(X) < 30:
                return None
            Xtr,Xte,Ytr,Yte = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
            Xtr_s = self.scaler.fit_transform(Xtr)
            _ = self.scaler.transform(Xte)
            self.model = RandomForestClassifier(
                n_estimators=n_estimators, n_jobs=-1, random_state=42, class_weight="balanced_subsample"
            )
            self.model.fit(Xtr_s, Ytr)
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            return {"trained": int(len(X))}
        except Exception as e:
            logger.error(f"ML train failed: {e}")
            return None

    def predict_proba_last(self, df: pd.DataFrame) -> Optional[float]:
        try:
            if self.model is None:
                return None
            X, _, cols = self._prepare_features(df)
            if X is None or len(X) == 0:
                return None
            x = X.iloc[[-1]]
            try:
                xs = self.scaler.transform(x)
            except Exception:
                xs = x.values
            if hasattr(self.model, "predict_proba"):
                p = float(self.model.predict_proba(xs)[:,1][0])
            elif hasattr(self.model, "decision_function"):
                m = float(self.model.decision_function(xs).ravel()[0])
                p = float(1.0/(1.0+np.exp(-m)))
            else:
                y = float(self.model.predict(xs).ravel()[0])
                p = float(max(0.0, min(1.0, 0.5 + 0.5*np.tanh(y))))
            return _clip01(p)
        except Exception as e:
            logger.warning(f"ML predict failed: {e}")
            return None

# -------------------- Global Strategy --------------------

def analyze_asset_global(ticker: str, horizon: str = "Краткосрочный"):
    cli = PolygonClient()
    df = cli.daily_ohlc(ticker, days=120)
    current_price = float(df['close'].iloc[-1])
    short_ma = float(df['close'].rolling(20).mean().iloc[-1])
    long_ma  = float(df['close'].rolling(50).mean().iloc[-1])
    action = "BUY" if short_ma > long_ma else "SHORT"
    confidence = 0.69 if action == "BUY" else 0.65
    atr = float(_atr_like(df, n=14).iloc[-1])
    if action == "BUY":
        entry=current_price; sl=current_price - 2*atr
        tp1=current_price + 1*atr; tp2=current_price + 2*atr; tp3=current_price + 3*atr
    else:
        entry=current_price; sl=current_price + 2*atr
        tp1=current_price - 1*atr; tp2=current_price - 2*atr; tp3=current_price - 3*atr
    # Синхронизация процента
    overall_pct = float(min(100.0, max(0.0, 44.0 + (confidence*100.0 - 50.0))))
    _sync_session(overall_pct, 44.0)
    return {
        "last_price": current_price,
        "recommendation": {"action": action, "confidence": confidence},
        "levels": {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": {"tp1": 0.68, "tp2": 0.52, "tp3": 0.35},
        "context": [f"SMA20 vs SMA50 • {'bull' if action=='BUY' else 'bear'}"],
        "note_html": "<div>Global Strategy</div>",
        "alt": "Global",
        "entry_kind": "market",
        "entry_label": f"{action} NOW",
        "confidence_breakdown": {"rules_pct": 44.0, "ai_override_delta_pct": (confidence*100.0 - 50.0), "overall_pct": overall_pct}
    }

# -------------------- M7 base (levels) --------------------

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
        out = {}
        for level in self.fib_levels:
            out[f'fib_{int(level*1000)}'] = high - level * diff
        return out

    def identify_key_levels(self, data: pd.DataFrame):
        grouped = data.resample('D') if self.pivot_period == 'D' else data.resample('W')
        key = {}
        for _, g in grouped:
            if len(g) == 0: 
                continue
            high = g['high'].max(); low = g['low'].min(); close = g['close'].iloc[-1]
            piv = self.calculate_pivot_points(high, low, close)
            fib = self.calculate_fib_levels(high, low)
            for k,v in {**piv, **fib}.items():
                key[k] = v
        return key

    def generate_signals(self, data: pd.DataFrame):
        signals = []
        req = ['high','low','close']
        if not all(c in data.columns for c in req):
            return signals
        data = data.copy()
        data['atr'] = _atr_like(data, self.atr_period)
        cur_atr = float(data['atr'].iloc[-1])
        levels = self.identify_key_levels(data)
        price = float(data['close'].iloc[-1])
        tstamp = data.index[-1]
        for name, val in levels.items():
            distance = abs(price - val) / max(1e-9, cur_atr)
            if distance < self.atr_multiplier:
                is_res = (val > price)
                if is_res:
                    sig = 'SELL_LIMIT'
                    entry = val * 0.998
                    sl    = val * 1.02
                    tp    = val * 0.96
                else:
                    sig = 'BUY_LIMIT'
                    entry = val * 1.002
                    sl    = val * 0.98
                    tp    = val * 1.04
                conf = float(max(0.0, min(1.0, 1 - (distance / self.atr_multiplier))))
                signals.append({
                    'type': sig, 'price': round(entry,4), 'stop_loss': round(sl,4),
                    'take_profit': round(tp,4), 'confidence': conf,
                    'level': name, 'level_value': round(val,4), 'timestamp': tstamp
                })
        return signals

# -------------------- M7 (с ML‑override и безопасным fallback) --------------------

def analyze_asset_m7(ticker, horizon="Краткосрочный", use_ml=True):
    cli = PolygonClient()
    days = 160
    df = cli.daily_ohlc(ticker, days=days)
    current_price = float(df['close'].iloc[-1])

    strategy = M7TradingStrategy()
    signals = strategy.generate_signals(df)

    if not signals:
        rules_pct = 44.0
        overall_pct = 44.0
        _sync_session(overall_pct, rules_pct)
        return {
            "last_price": current_price,
            "recommendation": {"action": "WAIT", "confidence": overall_pct/100.0},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0, "tp2": 0, "tp3": 0},
            "context": ["Нет сигналов по M7"],
            "note_html": "<div>Ожидание сигналов от стратегии M7</div>",
            "alt": "M7 idle",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "confidence_breakdown": {"rules_pct": rules_pct, "ai_override_delta_pct": 0.0, "overall_pct": overall_pct}
        }

    best = max(signals, key=lambda x: x['confidence'])
    entry_price = float(best['price'])
    stop_loss   = float(best['stop_loss'])
    risk = abs(entry_price - stop_loss)
    volatility = df['close'].pct_change().std() * np.sqrt(252)
    max_daily_move = current_price * volatility / np.sqrt(252)

    if best['type'].startswith('BUY'):
        tp1 = min(entry_price + risk * 1.5, entry_price + max_daily_move * 2)
        tp2 = min(entry_price + risk * 2.5, entry_price + max_daily_move * 3)
        tp3 = min(entry_price + risk * 4.0, entry_price + max_daily_move * 5)
        action = "BUY"
    else:
        tp1 = max(entry_price - risk * 1.5, entry_price - max_daily_move * 2)
        tp2 = max(entry_price - risk * 2.5, entry_price - max_daily_move * 3)
        tp3 = max(entry_price - risk * 4.0, entry_price - max_daily_move * 5)
        action = "SHORT"

    # Безопасная ML‑интеграция (не бросаем исключений)
    rules_pct = 44.0
    ai_delta = 0.0
    ml_p_long: Optional[float] = None
    if use_ml:
        try:
            ml = M7MLModel()
            if ml.model is None:
                _ = ml.train_and_save(df, n_estimators=500)
                ml = M7MLModel()
            ml_p_long = ml.predict_proba_last(df)
            if ml_p_long is not None:
                ai_delta = float(ml_p_long)*100.0 - 50.0
        except Exception as e:
            logger.warning(f"M7 ML integration warning: {e}")
            ml_p_long = None
            ai_delta = 0.0

    overall_pct = float(max(0.0, min(100.0, rules_pct + ai_delta)))
    _sync_session(overall_pct, rules_pct)

    context = [f"Level: {best['level']} @ {best['level_value']}"]
    if ml_p_long is not None:
        context.append(f"ML p_long={ml_p_long:.2f}")

    probs = {"tp1": 0.63, "tp2": 0.52, "tp3": 0.45}
    note_html = f"""
    <div style='margin-top:10px; opacity:0.95;'>
        M7 Strategy: {best['type']} на уровне {best['level_value']}.<br>
        Безопасная ML-интеграция: {'включена' if use_ml else 'выключена'}.
    </div>
    """

    entry_kind = "limit"
    entry_label = best['type']

    return {
        "last_price": current_price,
        "recommendation": {"action": action, "confidence": overall_pct/100.0},
        "levels": {"entry": entry_price, "sl": stop_loss, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs,
        "context": context,
        "note_html": note_html,
        "alt": "M7 with ML",
        "entry_kind": entry_kind,
        "entry_label": entry_label,
        "confidence_breakdown": {"rules_pct": rules_pct, "ai_override_delta_pct": ai_delta, "overall_pct": overall_pct}
    }

# -------------------- W7 Strategy --------------------

def analyze_asset_w7(ticker: str, horizon: str):
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)
    hz = cfg["hz"]
    days = max(90, cfg["look"] * 2)
    df = cli.daily_ohlc(ticker, days=days)
    try:
        price = cli.last_trade_price(ticker)
    except Exception:
        price = float(df["close"].iloc[-1])

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
        hlc = (float(df["high"].tail(60).max()),
               float(df["low"].tail(60).min()),
               float(df["close"].iloc[-1]))
    H, L, C = hlc
    piv = _fib_pivots(H, L, C)
    P, R1, R2, S1, S2 = piv["P"], piv["R1"], piv.get("R2"), piv["S1"], piv.get("S2")
    tol_k = {"ST": 0.18, "MID": 0.22, "LT": 0.28}[hz]
    buf = tol_k * (atr_w if hz != "ST" else atr_d)
    def _near_from_below(level: float) -> bool: return (level is not None) and (0 <= level - price <= buf)
    def _near_from_above(level: float) -> bool: return (level is not None) and (0 <= price - level <= buf)
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
                action, scenario = ("BUY", "trend_follow") if slope_norm >= 0 else ("WAIT", "mid_range")
            elif band == -1:
                action, scenario = "BUY", "revert_from_bottom"
            else:
                action, scenario = ("BUY", "revert_from_bottom") if band <= -2 else ("WAIT", "upper_wait")
    base = 0.55 + 0.12 * _clip01(abs(slope_norm) * 1800) + 0.08 * _clip01((vol_ratio - 0.9) / 0.6)
    if action == "WAIT": base -= 0.07
    conf = float(max(0.55, min(0.90, base)))
    # AI override (мягко)
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
        try:
            out_ai = score_signal(feats, hz=hz, ticker=ticker)
        except Exception:
            out_ai = None
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
    if (atr_d/ max(1e-9, price)) > 0.02: chips.append("высокая волатильность")
    if long_upper: chips.append("длинная верхняя тень")
    if long_lower: chips.append("длинная нижняя тень")
    p1 = _clip01(0.58 + 0.27 * (conf - 0.55) / 0.35)
    p2 = _clip01(0.44 + 0.21 * (conf - 0.55) / 0.35)
    p3 = _clip01(0.28 + 0.13 * (conf - 0.55) / 0.35)
    probs = {"tp1": float(p1), "tp2": float(p2), "tp3": float(p3)}
    lead = "Импульс удерживается — работаем по тренду." if action=="BUY" else ("Слабость у кромки — шорт аккуратно." if action=="SHORT" else "Цена в балансе — ждём.")
    note_html = f"<div style='margin-top:10px; opacity:0.95;'>{lead}</div>"
    overall_pct = float(min(100.0, max(0.0, 44.0 + (conf*100.0 - 50.0))))
    _sync_session(overall_pct, 44.0)
    return {
        "last_price": float(price),
        "recommendation": {"action": action, "confidence": overall_pct/100.0},
        "levels": {"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
        "probs": probs,
        "context": chips,
        "note_html": note_html,
        "alt": alt,
        "entry_kind": entry_kind,
        "entry_label": entry_label,
        "confidence_breakdown": {"rules_pct": 44.0, "ai_override_delta_pct": (overall_pct - 44.0), "overall_pct": overall_pct}
    }

# -------------------- Strategy Router --------------------

def analyze_asset(ticker: str, horizon: str, strategy: str = "W7"):
    s = (strategy or "").strip()
    if s == "Global":
        return analyze_asset_global(ticker, horizon)
    if s == "M7":
        return analyze_asset_m7(ticker, horizon)
    if s == "W7":
        return analyze_asset_w7(ticker, horizon)
    return analyze_asset_m7(ticker, horizon)
