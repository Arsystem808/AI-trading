# core/strategy.py
from __future__ import annotations

import json
import math
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

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

# ============================================================================
# PROFESSIONAL CONFIDENCE MODULE
# ============================================================================

STRATEGY_TEMPERATURES = {
    "global": 1.5,
    "m7": 1.3,
    "w7": 1.0,
}

CONFIDENCE_RANGES = {
    "global": {"min": 0.50, "max": 0.75},
    "m7": {"min": 0.52, "max": 0.82},
    "w7": {"min": 0.55, "max": 0.88},
}

def temperature_scaled_confidence(raw_confidence: float, temperature: float = 1.0) -> float:
    if raw_confidence <= 0.01 or raw_confidence >= 0.99:
        return float(max(0.0, min(1.0, raw_confidence)))
    try:
        odds = raw_confidence / (1.0 - raw_confidence + 1e-8)
        log_odds = np.log(odds + 1e-8)
        calibrated_log_odds = log_odds / temperature
        calibrated_odds = np.exp(calibrated_log_odds)
        calibrated_conf = calibrated_odds / (1.0 + calibrated_odds)
        return float(max(0.0, min(1.0, calibrated_conf)))
    except:
        return float(max(0.0, min(1.0, raw_confidence)))

def calculate_market_volatility(prices: list, window: int = 20) -> float:
    if len(prices) < 2:
        return 0.02
    prices_array = np.array(prices[-window:])
    returns = np.diff(prices_array) / prices_array[:-1]
    return float(np.std(returns))

def calculate_professional_confidence(
    strategy_name: str,
    signal_strength: float,
    market_volatility: float = None,
    historical_metrics: dict = None,
) -> float:
    strategy_type = "global"
    if "m7" in strategy_name.lower():
        strategy_type = "m7"
    elif "w7" in strategy_name.lower():
        strategy_type = "w7"
    
    if historical_metrics:
        win_rate = historical_metrics.get("win_rate", 0.50)
        sharpe = historical_metrics.get("sharpe", 0.0)
        trade_count = historical_metrics.get("trade_count", 0)
        
        base_conf = (
            0.40 * win_rate +
            0.30 * min(1.0, max(0.0, sharpe / 2.0)) +
            0.30 * signal_strength
        )
        
        if trade_count > 100:
            base_conf += 0.05
        elif trade_count > 50:
            base_conf += 0.03
    else:
        base_conf = 0.50 + 0.20 * signal_strength
    
    if market_volatility:
        if market_volatility > 0.03:
            base_conf *= 0.90
        elif market_volatility < 0.01:
            base_conf *= 1.05
    
    temperature = STRATEGY_TEMPERATURES[strategy_type]
    calibrated_conf = temperature_scaled_confidence(base_conf, temperature)
    
    conf_range = CONFIDENCE_RANGES[strategy_type]
    final_conf = max(conf_range["min"], min(conf_range["max"], calibrated_conf))
    
    return float(final_conf)

def ensemble_weighted_voting(strategies: list, use_historical_weights: bool = True) -> dict:
    if not strategies:
        return {"action": "WAIT", "confidence": 0.50}
    
    weighted_votes = {"BUY": 0.0, "SHORT": 0.0, "WAIT": 0.0}
    total_weight = 0.0
    
    for s in strategies:
        action = s.get("action", "WAIT")
        confidence = s.get("confidence", 0.50)
        
        if use_historical_weights and "trade_count" in s:
            weight = confidence * np.sqrt(max(1, s["trade_count"]) / 100.0)
        else:
            weight = confidence
        
        weighted_votes[action] += weight
        total_weight += weight
    
    if total_weight > 0:
        for action in weighted_votes:
            weighted_votes[action] /= total_weight
    
    best_action = max(weighted_votes, key=weighted_votes.get)
    return {
        "action": best_action,
        "confidence": float(weighted_votes[best_action]),
        "vote_distribution": weighted_votes
    }

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
    return {
        "P": P,
        "R1": P + 0.382 * d, "R2": P + 0.618 * d, "R3": P + 1.000 * d,
        "S1": P - 0.382 * d, "S2": P - 0.618 * d, "S3": P - 1.000 * d
    }

def _classify_band(price: float, piv: dict, buf: float) -> int:
    P, R1, S1 = piv["P"], piv["R1"], piv["S1"]
    R2, R3, S2 = piv.get("R2"), piv.get("R3"), piv.get("S2")
    neg_inf, pos_inf = -1e18, 1e18
    levels = [
        S2 if S2 is not None else neg_inf,
        S1, P, R1,
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

# -------------------- Калибровка --------------------
class ConfidenceCalibrator:
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
                x0, y0 = knots[i - 1]
                x1, y1 = knots[i]
                if p <= x1:
                    if x1 == x0:
                        return _clip01(float(y1))
                    t = (p - x0) / (x1 - x0)
                    return _clip01(float(y0 + t * (y1 - y0)))
            return _clip01(float(knots[-1][1]))
        return p

def _ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
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

# Загрузка калибровок
_DEFAULT_CAL = {
    "Global": {"conf": {"method": "sigmoid", "params": {"a": 1.0, "b": 0.0}}},
    "M7": {"conf": {"method": "sigmoid", "params": {"a": 1.0, "b": 0.0}}},
    "W7": {"conf": {"method": "sigmoid", "params": {"a": 1.0, "b": 0.0}}},
    "AlphaPulse": {"conf": {"method": "sigmoid", "params": {"a": 1.0, "b": 0.0}}},
    "Octopus": {"conf": {"method": "sigmoid", "params": {"a": 1.2, "b": -0.10}}}
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
    "Global": ConfidenceCalibrator(**_CAL["Global"]["conf"]),
    "M7": ConfidenceCalibrator(**_CAL["M7"]["conf"]),
    "W7": ConfidenceCalibrator(**_CAL["W7"]["conf"]),
    "AlphaPulse": ConfidenceCalibrator(**_CAL["AlphaPulse"]["conf"]),
    "Octopus": ConfidenceCalibrator(**_CAL["Octopus"]["conf"]),
}

# -------------------- Global Strategy --------------------
def analyze_asset_global(ticker: str, horizon: str = "Краткосрочный"):
    """
    Global агент: простая стратегия на основе MA кроссовера.
    """
    cli = PolygonClient()
    df = cli.daily_ohlc(ticker, days=120)
    current_price = float(df['close'].iloc[-1])

    closes = df['close']
    short_ma = closes.rolling(20).mean().iloc[-1]
    long_ma = closes.rolling(50).mean().iloc[-1]
    ma_gap = float((short_ma - long_ma) / max(1e-9, long_ma))
    slope = _linreg_slope(closes.tail(30).values) / max(1e-9, current_price)
    
    atr = float(_atr_like(df, n=14).iloc[-1]) or 1e-9
    atr28 = float(_atr_like(df, n=28).iloc[-1]) or 1e-9
    vol_ratio = float(atr / max(1e-9, atr28))

    action = "BUY" if short_ma > long_ma else "SHORT"
    
    # Professional confidence calculation
    signal_strength = _clip01(
        0.40 * _clip01(abs(ma_gap) / 0.02) +
        0.35 * _clip01((abs(slope) - 0.0003) / 0.0007) +
        0.25 * (1.0 - _clip01((vol_ratio - 1.0) / 0.80))
    )
    
    historical_metrics = get_agent_performance("GlobalStrategy")
    market_volatility = calculate_market_volatility(closes.tolist(), window=20)
    
    confidence = calculate_professional_confidence(
        strategy_name="global",
        signal_strength=signal_strength,
        market_volatility=market_volatility,
        historical_metrics=historical_metrics
    )

    entry = current_price
    if action == "BUY":
        sl = current_price - 2 * atr
        tp1, tp2, tp3 = current_price + 1 * atr, current_price + 2 * atr, current_price + 3 * atr
        alt = "Покупка по рынку с консервативными целями"
    else:
        sl = current_price + 2 * atr
        tp1, tp2, tp3 = current_price - 1 * atr, current_price - 2 * atr, current_price - 3 * atr
        alt = "Продажа по рынку с консервативными целями"

    u1, u2, u3 = abs(tp1 - entry) / atr, abs(tp2 - entry) / atr, abs(tp3 - entry) / atr
    k = 0.16 + 0.12 * _clip01((vol_ratio - 1.00) / 0.80)
    b1 = confidence
    b2 = max(0.50, confidence - (0.08 + 0.03 * _clip01(vol_ratio - 1.0)))
    b3 = max(0.45, confidence - (0.16 + 0.05 * _clip01(vol_ratio - 1.2)))
    
    p1 = _clip01(b1 * math.exp(-k * (u1 - 1.0)))
    p2 = _clip01(b2 * math.exp(-k * (u2 - 1.5)))
    p3 = _clip01(b3 * math.exp(-k * (u3 - 2.2)))
    probs = _monotone_tp_probs({"tp1": p1, "tp2": p2, "tp3": p3})

    meta_debug = {
        "u": [float(u1), float(u2), float(u3)],
        "p": [float(probs['tp1']), float(probs['tp2']), float(probs['tp3'])],
        "signal_strength": float(signal_strength),
        "market_volatility": float(market_volatility)
    }
    
    try:
        log_agent_performance(
            agent="GlobalStrategy", ticker=ticker, horizon=horizon,
            action=action, confidence=float(confidence),
            levels={"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
            probs={"tp1": float(probs["tp1"]), "tp2": float(probs["tp2"]), "tp3": float(probs["tp3"])},
            meta={"probs_debug": meta_debug},
            ts=pd.Timestamp.utcnow().isoformat()
        )
    except Exception as e:
        logger.warning("perf log Global failed: %s", e)

    return {
        "last_price": current_price,
        "recommendation": {"action": action, "confidence": confidence},
        "levels": {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs,
        "context": [],
        "note_html": f"<div>Global: {action} с {confidence:.0%} (сигнал: {signal_strength:.0%})</div>",
        "alt": alt,
        "entry_kind": "market",
        "entry_label": f"{action} NOW",
        "meta": {"source": "Global", "probs_debug": meta_debug}
    }

# ==================== M7 ML Model (ЕДИНСТВЕННАЯ ВЕРСИЯ) ====================
class M7MLModel:
    """ML-модель для M7 с unified model_loader"""
    
    def __init__(self):
        self.model = None
        self.model_data = None
        self.feature_names = None
        self.scaler = None
        self.last_load_time = None
    
    def try_repo_loader(self, ticker: str) -> bool:
        """Загрузка модели через unified loader"""
        try:
            from core.model_loader import load_model_for
            from pathlib import Path
            import joblib
            
            logger.info(f"[M7] Loading model for {ticker}")
            
            # ИСПРАВЛЕНО: было self.model_, теперь self.model_data
            self.model_data = load_model_for(ticker, agent="arxora_m7pro")
            
            if not self.model_data:
                logger.warning(f"[M7] No model found for {ticker}")
                return False
            
            if isinstance(self.model_data, dict):
                self.model = self.model_data.get("model")
                self.feature_names = self.model_data.get("feature_names")
                metadata = self.model_data.get("metadata", {})
                
                scaler_path = metadata.get("scaler_artifact")
                if scaler_path:
                    scaler_path = Path(scaler_path)
                    if scaler_path.exists():
                        try:
                            self.scaler = joblib.load(scaler_path)
                            logger.info(f"[M7] ✓ Loaded scaler from {scaler_path.name}")
                        except Exception as e:
                            logger.error(f"[M7] Failed to load scaler: {e}")
                            self.scaler = None
                
                version = metadata.get("version", "unknown")
                feature_count = len(self.feature_names) if self.feature_names else "unknown"
                logger.info(f"[M7] ✓ Loaded model v{version} ({feature_count} features)")
            else:
                self.model = self.model_data
                logger.warning("[M7] Loaded model in legacy format")
            
            self.last_load_time = pd.Timestamp.utcnow()
            return self.model is not None
            
        except Exception as e:
            logger.error(f"[M7] Model loading failed: {e}", exc_info=True)
            return False
    
    def predict(self, X):
        """Предсказание с валидацией"""
        if self.model is None:
            logger.error("[M7] Cannot predict: model not loaded")
            return None
        
        try:
            if self.feature_names and hasattr(X, 'columns'):
                missing = set(self.feature_names) - set(X.columns)
                if missing:
                    logger.error(f"[M7] Missing features: {missing}")
                    return None
                X = X[self.feature_names]
            
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(X)
            else:
                return self.model.predict(X)
                
        except Exception as e:
            logger.error(f"[M7] Prediction failed: {e}", exc_info=True)
            return None


# ==================== M7 Trading Strategy ====================
class M7TradingStrategy:
    """M7: Торговля по ключевым уровням"""
    
    def __init__(self, atr_period=14, atr_multiplier=1.5, pivot_period='D', fib_levels=None):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.pivot_period = pivot_period
        self.fib_levels = fib_levels or [0.236, 0.382, 0.5, 0.618, 0.786]
    
    def calculate_pivot_points(self, h, l, c):
        """Calculate classic pivot points"""
        pivot = (h + l + c) / 3
        r1 = (2 * pivot) - l
        r2 = pivot + (h - l)
        r3 = h + 2 * (pivot - l)
        s1 = (2 * pivot) - h
        s2 = pivot - (h - l)
        s3 = l - 2 * (h - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    def calculate_fib_levels(self, h, l):
        """Calculate Fibonacci levels"""
        diff = h - l
        fib = {}
        for level in self.fib_levels:
            fib[f'fib_{int(level * 1000)}'] = h - level * diff
        return fib
    
    def identify_key_levels(self, data):
        """Identify key price levels"""
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.error("[M7] Data must have DatetimeIndex")
            return {}
        
        grouped = data.resample('D') if self.pivot_period == 'D' else data.resample('W')
        key_levels = {}
        
        for _, g in grouped:
            if len(g) > 0:
                h = g['high'].max()
                l = g['low'].min()
                c = g['close'].iloc[-1]
                
                key_levels.update(self.calculate_pivot_points(h, l, c))
                key_levels.update(self.calculate_fib_levels(h, l))
        
        return key_levels
    
    def generate_signals(self, data):
        """Generate trading signals"""
        signals = []
        required = ['high', 'low', 'close']
        
        if not all(c in data.columns for c in required):
            logger.warning(f"[M7] Missing required columns: {required}")
            return signals
        
        data = data.copy()
        data['atr'] = _atr_like(data, self.atr_period)
        cur_atr = data['atr'].iloc[-1]
        
        if pd.isna(cur_atr) or cur_atr <= 0:
            logger.warning("[M7] Invalid ATR")
            return signals
        
        key_levels = self.identify_key_levels(data)
        if not key_levels:
            logger.warning("[M7] No key levels identified")
            return signals
        
        price = data['close'].iloc[-1]
        ts = data.index[-1]
        
        for name, val in key_levels.items():
            if pd.isna(val):
                continue
            
            dist = abs(price - val) / max(1e-9, cur_atr)
            
            if dist < self.atr_multiplier:
                is_resistance = val > price
                
                if is_resistance:
                    typ = 'SELL_LIMIT'
                    entry = val * 0.998
                    sl = val * 1.02
                    tp = val * 0.96
                else:
                    typ = 'BUY_LIMIT'
                    entry = val * 1.002
                    sl = val * 0.98
                    tp = val * 1.04
                
                conf = 1 - (dist / self.atr_multiplier)
                
                signals.append({
                    'type': typ,
                    'price': round(entry, 4),
                    'stop_loss': round(sl, 4),
                    'take_profit': round(tp, 4),
                    'confidence': round(conf, 2),
                    'level': name,
                    'level_value': round(val, 4),
                    'timestamp': ts,
                    'atr_distance': round(dist, 2)
                })
        
        return signals


# ==================== M7 Analysis Function (ЗАВЕРШЕНО) ====================
def analyze_asset_m7(ticker, horizon="Краткосрочный", use_ml=False):
    """M7 агент: торговля по уровням"""
    cli = PolygonClient()
    df = cli.daily_ohlc(ticker, days=120)
    
    if df is None or df.empty:
        logger.warning(f"[M7] No data for {ticker}")
        return {
            "last_price": 0.0,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": ["Нет данных для M7"],
            "note_html": "<div>M7: ожидание данных</div>",
            "alt": "Ожидание сигналов",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "M7", "grey_zone": True, "error": "no_data"}
        }
    
    df = df.sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    
    strategy = M7TradingStrategy()
    signals = strategy.generate_signals(df)
    price = float(df['close'].iloc[-1])
    
    if not signals:
        logger.info(f"[M7] No signals for {ticker}")
        result = {
            "last_price": price,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": ["Нет сигналов по стратегии M7"],
            "note_html": "<div>M7: ожидание уровней</div>",
            "alt": "Ожидание сигналов",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "M7", "grey_zone": True}
        }
        
        try:
            log_agent_performance(
                agent="M7Strategy", ticker=ticker, horizon=horizon,
                action="WAIT", confidence=0.50,
                levels=result["levels"], probs=result["probs"],
                meta={"reason": "no_signals"},
                ts=pd.Timestamp.utcnow().isoformat()
            )
        except Exception as e:
            logger.warning(f"[M7] Performance logging failed: {e}")
        
        return result
    
    best = max(signals, key=lambda x: x['confidence'])
    raw_conf = float(_clip01(best['confidence']))
    
    entry = float(best['price'])
    sl = float(best['stop_loss'])
    risk = abs(entry - sl)
    
    atr14 = float(_atr_like(df, n=14).iloc[-1]) or 1e-9
    vol = df['close'].pct_change().std() * np.sqrt(252)
    
    # Professional confidence
    signal_strength = _clip01(
        0.50 * raw_conf +
        0.30 * (1.0 - _clip01((risk / atr14 - 1.5) / 2.0)) +
        0.20 * (1.0 - _clip01((vol - 0.20) / 0.30))
    )
    
    historical_metrics = get_agent_performance("M7Strategy")
    market_volatility = calculate_market_volatility(df['close'].tolist(), window=20)
    
    confidence = calculate_professional_confidence(
        strategy_name="m7",
        signal_strength=signal_strength,
        market_volatility=market_volatility,
        historical_metrics=historical_metrics
    )
    
    if best['type'].startswith('BUY'):
        tp1 = min(entry + 1.5 * risk, entry + 2 * price * vol / np.sqrt(252))
        tp2 = min(entry + 2.5 * risk, entry + 3 * price * vol / np.sqrt(252))
        tp3 = min(entry + 4.0 * risk, entry + 5 * price * vol / np.sqrt(252))
        act = "BUY"
    else:
        tp1 = max(entry - 1.5 * risk, entry - 2 * price * vol / np.sqrt(252))
        tp2 = max(entry - 2.5 * risk, entry - 3 * price * vol / np.sqrt(252))
        tp3 = max(entry - 4.0 * risk, entry - 5 * price * vol / np.sqrt(252))
        act = "SHORT"
    
    u1, u2, u3 = abs(tp1 - entry) / atr14, abs(tp2 - entry) / atr14, abs(tp3 - entry) / atr14
    k = 0.18
    b1, b2, b3 = confidence, max(0.50, confidence - 0.08), max(0.45, confidence - 0.16)
    
    p1 = _clip01(b1 * np.exp(-k * (u1 - 1.0)))
    p2 = _clip01(b2 * np.exp(-k * (u2 - 1.5)))
    p3 = _clip01(b3 * np.exp(-k * (u3 - 2.2)))
    
    probs = _monotone_tp_probs({"tp1": p1, "tp2": p2, "tp3": p3})
    
    meta_debug = {
        "risk": float(risk),
        "atr14": float(atr14),
        "vol": float(vol) if vol else None,
        "signal_strength": float(signal_strength),
        "market_volatility": float(market_volatility),
        "atr_dist": float(best.get('atr_distance', 0)),
        "level": best['level'],
        "u": [float(u1), float(u2), float(u3)],
        "p": [float(probs["tp1"]), float(probs["tp2"]), float(probs["tp3"])]
    }
    
    try:
        log_agent_performance(
            agent="M7Strategy", ticker=ticker, horizon=horizon,
            action=act, confidence=float(confidence),
            levels={"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
            probs={"tp1": float(probs["tp1"]), "tp2": float(probs["tp2"]), "tp3": float(probs["tp3"])},
            meta=meta_debug,
            ts=pd.Timestamp.utcnow().isoformat()
        )
    except Exception as e:
        logger.warning(f"[M7] Performance logging failed: {e}")
    
    return {
        "last_price": price,
        "recommendation": {"action": act, "confidence": confidence},
        "levels": {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs,
        "context": [f"Уровень: {best['level']} ({best['level_value']:.2f})"],
        "note_html": f"<div>M7: {act} от уровня {best['level']} с {confidence:.0%}</div>",
        "alt": f"{act} от ключевого уровня {best['level']}",
        "entry_kind": "limit",
        "entry_label": f"{act} @ {entry:.2f}",
        "meta": {"source": "M7", "probs_debug": meta_debug}
    }

# ==================== W7 Strategy (Weekly Pivots + Heikin-Ashi) ====================
def analyze_asset_w7(ticker: str, horizon: str = "Краткосрочный"):
    """
    W7: Multi-horizon анализ с weekly ATR, Fibonacci pivots и Heikin-Ashi
    """
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)
    
    df = cli.daily_ohlc(ticker, days=cfg["look"])
    if df is None or df.empty:
        logger.warning(f"[W7] No data for {ticker}")
        return {
            "last_price": 0.0,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": ["Нет данных для W7"],
            "note_html": "<div>W7: ожидание данных</div>",
            "alt": "Ожидание сигналов",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "W7", "grey_zone": True, "error": "no_data"}
        }
    
    df = df.sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    
    price = float(df["close"].iloc[-1])
    
    # ATR calculation
    if cfg["use_weekly_atr"]:
        atr = _weekly_atr(df, n_weeks=8)
    else:
        atr = float(_atr_like(df, n=cfg["atr"]).iloc[-1])
    
    if atr <= 0:
        atr = price * 0.02
    
    # Trend analysis
    ma = df["close"].rolling(cfg["trend"], min_periods=1).mean()
    slope = _linreg_slope(ma.tail(cfg["trend"] // 2).values) / max(1e-9, price)
    
    # Heikin-Ashi
    ha = _heikin_ashi(df)
    ha_close_diff = ha["ha_close"].diff()
    ha_pos = (ha_close_diff > 0).astype(int)
    ha_neg = (ha_close_diff < 0).astype(int)
    
    # MACD
    _, _, macd_hist = _macd_hist(df["close"])
    macd_pos_streak = _streak_by_sign(macd_hist, positive=True)
    macd_neg_streak = _streak_by_sign(macd_hist, positive=False)
    
    # Pivot points
    hlc = _last_period_hlc(df, cfg["pivot_rule"])
    if hlc is None:
        logger.warning(f"[W7] Cannot compute pivots for {ticker}")
        pivots = {"P": price, "R1": price * 1.02, "S1": price * 0.98}
    else:
        H, L, C = hlc
        pivots = _fib_pivots(H, L, C)
    
    band = _classify_band(price, pivots, buf=atr * 0.3)
    
    # Signal logic
    bullish_score = 0.0
    if slope > 0.0005:
        bullish_score += 0.25
    if macd_pos_streak >= 3:
        bullish_score += 0.20
    if ha_pos.tail(5).sum() >= 4:
        bullish_score += 0.15
    if band >= 0:
        bullish_score += 0.15
    else:
        bullish_score += 0.05
    
    bearish_score = 0.0
    if slope < -0.0005:
        bearish_score += 0.25
    if macd_neg_streak >= 3:
        bearish_score += 0.20
    if ha_neg.tail(5).sum() >= 4:
        bearish_score += 0.15
    if band <= 0:
        bearish_score += 0.15
    else:
        bearish_score += 0.05
    
    raw_conf = max(bullish_score, bearish_score)
    action = "BUY" if bullish_score > bearish_score else "SHORT"
    
    # Professional confidence
    signal_strength = _clip01(raw_conf)
    historical_metrics = get_agent_performance("W7Strategy")
    market_volatility = calculate_market_volatility(df['close'].tolist(), window=20)
    
    confidence = calculate_professional_confidence(
        strategy_name="w7",
        signal_strength=signal_strength,
        market_volatility=market_volatility,
        historical_metrics=historical_metrics
    )
    
    # Entry and levels
    if action == "BUY":
        entry = max(price, pivots["S1"])
        sl = entry - 2.0 * atr
        tp1 = pivots.get("P", entry + 1.5 * atr)
        tp2 = pivots.get("R1", entry + 2.5 * atr)
        tp3 = pivots.get("R2", entry + 4.0 * atr)
    else:
        entry = min(price, pivots["R1"])
        sl = entry + 2.0 * atr
        tp1 = pivots.get("P", entry - 1.5 * atr)
        tp2 = pivots.get("S1", entry - 2.5 * atr)
        tp3 = pivots.get("S2", entry - 4.0 * atr)
    
    u1, u2, u3 = abs(tp1 - entry) / atr, abs(tp2 - entry) / atr, abs(tp3 - entry) / atr
    k = 0.16
    b1, b2, b3 = confidence, max(0.50, confidence - 0.08), max(0.45, confidence - 0.16)
    
    p1 = _clip01(b1 * np.exp(-k * (u1 - 1.0)))
    p2 = _clip01(b2 * np.exp(-k * (u2 - 1.5)))
    p3 = _clip01(b3 * np.exp(-k * (u3 - 2.2)))
    probs = _monotone_tp_probs({"tp1": p1, "tp2": p2, "tp3": p3})
    
    meta_debug = {
        "bullish": float(bullish_score),
        "bearish": float(bearish_score),
        "band": band,
        "signal_strength": float(signal_strength),
        "market_volatility": float(market_volatility),
        "u": [float(u1), float(u2), float(u3)],
        "p": [float(probs["tp1"]), float(probs["tp2"]), float(probs["tp3"])]
    }
    
    try:
        log_agent_performance(
            agent="W7Strategy", ticker=ticker, horizon=horizon,
            action=action, confidence=float(confidence),
            levels={"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
            probs={"tp1": float(probs["tp1"]), "tp2": float(probs["tp2"]), "tp3": float(probs["tp3"])},
            meta=meta_debug,
            ts=pd.Timestamp.utcnow().isoformat()
        )
    except Exception as e:
        logger.warning(f"[W7] Performance logging failed: {e}")
    
    return {
        "last_price": price,
        "recommendation": {"action": action, "confidence": confidence},
        "levels": {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs,
        "context": [f"Horizon: {cfg['hz']}, Band: {band}, Slope: {slope:.6f}"],
        "note_html": f"<div>W7: {action} с {confidence:.0%} (band={band}, slope={slope:.4f})</div>",
        "alt": f"{action} на горизонте {horizon}",
        "entry_kind": "market" if abs(entry - price) < 0.005 * price else "limit",
        "entry_label": f"{action} @ {entry:.2f}",
        "meta": {"source": "W7", "probs_debug": meta_debug}
    }


# ==================== AlphaPulse (EMA Crossover) ====================
def analyze_asset_alphapulse(ticker: str, horizon: str = "Краткосрочный"):
    """
    AlphaPulse: EMA-based momentum strategy
    """
    cli = PolygonClient()
    df = cli.daily_ohlc(ticker, days=120)
    
    if df is None or df.empty:
        return {
            "last_price": 0.0,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": ["Нет данных"],
            "note_html": "<div>AlphaPulse: ожидание данных</div>",
            "alt": "Ожидание",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "AlphaPulse", "grey_zone": True}
        }
    
    price = float(df["close"].iloc[-1])
    ema8 = df["close"].ewm(span=8, adjust=False).mean().iloc[-1]
    ema21 = df["close"].ewm(span=21, adjust=False).mean().iloc[-1]
    
    atr14 = float(_atr_like(df, n=14).iloc[-1]) or price * 0.02
    
    # Signal
    action = "BUY" if ema8 > ema21 else "SHORT"
    gap = abs(ema8 - ema21) / max(1e-9, ema21)
    
    signal_strength = _clip01(gap / 0.015)
    historical_metrics = get_agent_performance("AlphaPulseStrategy")
    market_volatility = calculate_market_volatility(df['close'].tolist(), window=20)
    
    confidence = calculate_professional_confidence(
        strategy_name="alphapulse",
        signal_strength=signal_strength,
        market_volatility=market_volatility,
        historical_metrics=historical_metrics
    )
    
    entry = price
    if action == "BUY":
        sl = price - 2 * atr14
        tp1, tp2, tp3 = price + 1.5 * atr14, price + 2.5 * atr14, price + 4 * atr14
    else:
        sl = price + 2 * atr14
        tp1, tp2, tp3 = price - 1.5 * atr14, price - 2.5 * atr14, price - 4 * atr14
    
    u1, u2, u3 = abs(tp1 - entry) / atr14, abs(tp2 - entry) / atr14, abs(tp3 - entry) / atr14
    k = 0.18
    b1, b2, b3 = confidence, max(0.50, confidence - 0.08), max(0.45, confidence - 0.16)
    
    p1 = _clip01(b1 * np.exp(-k * (u1 - 1.0)))
    p2 = _clip01(b2 * np.exp(-k * (u2 - 1.5)))
    p3 = _clip01(b3 * np.exp(-k * (u3 - 2.2)))
    probs = _monotone_tp_probs({"tp1": p1, "tp2": p2, "tp3": p3})
    
    meta_debug = {
        "gap": float(gap),
        "signal_strength": float(signal_strength),
        "market_volatility": float(market_volatility)
    }
    
    try:
        log_agent_performance(
            agent="AlphaPulseStrategy", ticker=ticker, horizon=horizon,
            action=action, confidence=float(confidence),
            levels={"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
            probs={"tp1": float(probs["tp1"]), "tp2": float(probs["tp2"]), "tp3": float(probs["tp3"])},
            meta=meta_debug,
            ts=pd.Timestamp.utcnow().isoformat()
        )
    except Exception as e:
        logger.warning(f"[AlphaPulse] Performance logging failed: {e}")
    
    return {
        "last_price": price,
        "recommendation": {"action": action, "confidence": confidence},
        "levels": {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs,
        "context": [f"EMA8={ema8:.2f}, EMA21={ema21:.2f}, Gap={gap:.2%}"],
        "note_html": f"<div>AlphaPulse: {action} с {confidence:.0%}</div>",
        "alt": f"{action} по EMA crossover",
        "entry_kind": "market",
        "entry_label": f"{action} NOW",
        "meta": {"source": "AlphaPulse", "probs_debug": meta_debug}
    }


# ==================== Octopus (Multi-Signal Ensemble) ====================
def analyze_asset_octopus(ticker: str, horizon: str = "Краткосрочный"):
    """
    Octopus: Ensemble из нескольких сигналов
    """
    cli = PolygonClient()
    df = cli.daily_ohlc(ticker, days=120)
    
    if df is None or df.empty:
        return {
            "last_price": 0.0,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": ["Нет данных"],
            "note_html": "<div>Octopus: ожидание данных</div>",
            "alt": "Ожидание",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "Octopus", "grey_zone": True}
        }
    
    price = float(df["close"].iloc[-1])
    
    # Multiple signals
    ma20 = df["close"].rolling(20).mean().iloc[-1]
    ma50 = df["close"].rolling(50).mean().iloc[-1]
    rsi = 100 - (100 / (1 + (df["close"].diff().clip(lower=0).rolling(14).mean() / 
                               (-df["close"].diff().clip(upper=0).rolling(14).mean() + 1e-9))))
    rsi_val = rsi.iloc[-1]
    
    atr = float(_atr_like(df, n=14).iloc[-1]) or price * 0.02
    
    signals = []
    if ma20 > ma50:
        signals.append(1)
    else:
        signals.append(-1)
    
    if rsi_val < 30:
        signals.append(1)
    elif rsi_val > 70:
        signals.append(-1)
    else:
        signals.append(0)
    
    if price > ma20:
        signals.append(1)
    else:
        signals.append(-1)
    
    net = sum(signals)
    action = "BUY" if net > 0 else "SHORT" if net < 0 else "WAIT"
    
    if action == "WAIT":
        return {
            "last_price": price,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": ["Нейтральные сигналы"],
            "note_html": "<div>Octopus: нейтрально</div>",
            "alt": "Ожидание",
            "entry_kind": "wait",
            "entry_label": "WAIT",
            "meta": {"source": "Octopus", "grey_zone": True}
        }
    
    signal_strength = _clip01(abs(net) / 3.0)
    historical_metrics = get_agent_performance("OctopusStrategy")
    market_volatility = calculate_market_volatility(df['close'].tolist(), window=20)
    
    confidence = calculate_professional_confidence(
        strategy_name="octopus",
        signal_strength=signal_strength,
        market_volatility=market_volatility,
        historical_metrics=historical_metrics
    )
    
    entry = price
    if action == "BUY":
        sl = price - 2 * atr
        tp1, tp2, tp3 = price + 1.5 * atr, price + 2.5 * atr, price + 4 * atr
    else:
        sl = price + 2 * atr
        tp1, tp2, tp3 = price - 1.5 * atr, price - 2.5 * atr, price - 4 * atr
    
    u1, u2, u3 = abs(tp1 - entry) / atr, abs(tp2 - entry) / atr, abs(tp3 - entry) / atr
    k = 0.18
    b1, b2, b3 = confidence, max(0.50, confidence - 0.08), max(0.45, confidence - 0.16)
    
    p1 = _clip01(b1 * np.exp(-k * (u1 - 1.0)))
    p2 = _clip01(b2 * np.exp(-k * (u2 - 1.5)))
    p3 = _clip01(b3 * np.exp(-k * (u3 - 2.2)))
    probs = _monotone_tp_probs({"tp1": p1, "tp2": p2, "tp3": p3})
    
    meta_debug = {
        "net_signal": net,
        "signal_strength": float(signal_strength),
        "market_volatility": float(market_volatility),
        "rsi": float(rsi_val)
    }
    
    try:
        log_agent_performance(
            agent="OctopusStrategy", ticker=ticker, horizon=horizon,
            action=action, confidence=float(confidence),
            levels={"entry": float(entry), "sl": float(sl), "tp1": float(tp1), "tp2": float(tp2), "tp3": float(tp3)},
            probs={"tp1": float(probs["tp1"]), "tp2": float(probs["tp2"]), "tp3": float(probs["tp3"])},
            meta=meta_debug,
            ts=pd.Timestamp.utcnow().isoformat()
        }
    except Exception as e:
        logger.warning(f"[Octopus] Performance logging failed: {e}")
    
    return {
        "last_price": price,
        "recommendation": {"action": action, "confidence": confidence},
        "levels": {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs,
        "context": [f"Net signal: {net}, RSI: {rsi_val:.1f}"],
        "note_html": f"<div>Octopus: {action} с {confidence:.0%} (RSI={rsi_val:.0f})</div>",
        "alt": f"{action} по ensemble",
        "entry_kind": "market",
        "entry_label": f"{action} NOW",
        "meta": {"source": "Octopus", "probs_debug": meta_debug}
    }

# ==================== MAIN UNIFIED ANALYZE FUNCTION ====================
def analyze_asset(ticker: str, agent: str, horizon: str = "Краткосрочный"):
    """
    Универсальная точка входа для всех агентов
    
    Args:
        ticker: тикер актива (например, AAPL)
        agent: название агента (global, m7, w7, alphapulse, octopus)
        horizon: временной горизонт
    
    Returns:
        dict с результатами анализа
    """
    logger.info(f"[analyze_asset] ticker={ticker}, agent={agent}, horizon={horizon}")
    
    agent_lower = agent.lower().strip()
    
    try:
        if agent_lower in ["global", "globalstrategy"]:
            return analyze_asset_global(ticker, horizon)
        elif agent_lower in ["m7", "m7strategy"]:
            return analyze_asset_m7(ticker, horizon)
        elif agent_lower in ["w7", "w7strategy"]:
            return analyze_asset_w7(ticker, horizon)
        elif agent_lower in ["alphapulse", "alphapulsestrategy"]:
            return analyze_asset_alphapulse(ticker, horizon)
        elif agent_lower in ["octopus", "octopusstrategy"]:
            return analyze_asset_octopus(ticker, horizon)
        else:
            logger.warning(f"Unknown agent: {agent}, fallback to Global")
            return analyze_asset_global(ticker, horizon)
    
    except Exception as e:
        logger.error(f"[analyze_asset] Error for {ticker}/{agent}: {e}", exc_info=True)
        
        return {
            "last_price": 0.0,
            "recommendation": {"action": "WAIT", "confidence": 0.5},
            "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
            "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
            "context": [f"Ошибка: {str(e)}"],
            "note_html": f"<div>Ошибка анализа: {str(e)}</div>",
            "alt": "Ошибка",
            "entry_kind": "wait",
            "entry_label": "ERROR",
            "meta": {"source": agent, "error": str(e)}
        }


# ==================== ARXORA MULTI-AGENT ANALYSIS ====================
def run_arxora_analysis(ticker: str, horizon: str = "Краткосрочный"):
    """
    Запускает полный анализ всех агентов Arxora для данного тикера
    
    Args:
        ticker: тикер актива
        horizon: временной горизонт
    
    Returns:
        dict с результатами от всех агентов
    """
    logger.info(f"[run_arxora_analysis] Starting multi-agent analysis for {ticker}")
    
    agents = ["global", "m7", "w7", "alphapulse", "octopus"]
    results = {}
    
    for agent in agents:
        try:
            logger.info(f"[run_arxora_analysis] Running {agent} for {ticker}")
            result = analyze_asset(ticker, agent, horizon)
            results[agent] = result
        except Exception as e:
            logger.error(f"[run_arxora_analysis] Failed {agent} for {ticker}: {e}")
            results[agent] = {
                "last_price": 0.0,
                "recommendation": {"action": "WAIT", "confidence": 0.5},
                "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
                "probs": {"tp1": 0.0, "tp2": 0.0, "tp3": 0.0},
                "context": [f"Ошибка: {str(e)}"],
                "note_html": f"<div>Ошибка {agent}: {str(e)}</div>",
                "alt": "Ошибка",
                "entry_kind": "wait",
                "entry_label": "ERROR",
                "meta": {"source": agent, "error": str(e)}
            }
    
    logger.info(f"[run_arxora_analysis] Completed analysis for {ticker}")
    return results


# ==================== LEGACY COMPATIBILITY ====================
def analyze_asset_arxora(*args, **kwargs):
    """Legacy alias for analyze_asset"""
    return analyze_asset(*args, **kwargs)


# ==================== MODULE EXPORTS ====================
__all__ = [
    "analyze_asset",
    "analyze_asset_global",
    "analyze_asset_m7",
    "analyze_asset_w7",
    "analyze_asset_alphapulse",
    "analyze_asset_octopus",
    "run_arxora_analysis",
    "M7TradingStrategy",
    "M7MLModel",
]


# ==================== INITIALIZATION ====================
if __name__ == "__main__":
    # Test mode
    import sys
    
    if len(sys.argv) > 1:
        test_ticker = sys.argv[1]
        test_agent = sys.argv[2] if len(sys.argv) > 2 else "global"
        
        print(f"\n{'='*60}")
        print(f"TESTING: {test_ticker} with {test_agent}")
        print(f"{'='*60}\n")
        
        result = analyze_asset(test_ticker, test_agent)
        
        print(f"\n📊 РЕЗУЛЬТАТ:")
        print(f"  Цена: ${result['last_price']:.2f}")
        print(f"  Действие: {result['recommendation']['action']}")
        print(f"  Уверенность: {result['recommendation']['confidence']:.0%}")
        print(f"\n  Уровни:")
        print(f"    Вход: ${result['levels']['entry']:.2f}")
        print(f"    SL: ${result['levels']['sl']:.2f}")
        print(f"    TP1: ${result['levels']['tp1']:.2f} ({result['probs']['tp1']:.0%})")
        print(f"    TP2: ${result['levels']['tp2']:.2f} ({result['probs']['tp2']:.0%})")
        print(f"    TP3: ${result['levels']['tp3']:.2f} ({result['probs']['tp3']:.0%})")
        print(f"\n{'='*60}\n")
    else:
        print("Usage: python strategy.py <TICKER> [AGENT]")
        print("Example: python strategy.py AAPL global")
