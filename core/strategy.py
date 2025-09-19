# core/strategy.py
import os
import hashlib
import random
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from core.polygon_client import PolygonClient
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta

# Настройка логирования
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
    # ST — weekly; MID/LT — monthly
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
    # -3: <S2, -2:[S2,S1), -1:[S1,P), 0:[P,R1), +1:[R1,R2), +2:[R2,R3), +3:>=R3
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
    if price < levels[0] - buf: 
        return -3
    if price < levels[1] - buf: 
        return -2
    if price < levels[2] - buf: 
        return -1
    if price < levels[3] - buf: 
        return 0
    if R2 is None or price < levels[4] - buf: 
        return +1
    if price < levels[5] - buf: 
        return +2
    return +3

# -------------------- wick profile (для AI/псевдо) --------------------

def _wick_profile(row: pd.Series):
    o, c, h, l = float(row["open"]), float(row["close"]), float(row["high"]), float(row["low"])
    body = max(1e-9, abs(c - o))
    up_wick = max(0.0, h - max(o, c))
    dn_wick = max(0.0, min(o, c) - l)
    return body, up_wick, dn_wick

# -------------------- Order kind (STOP/LIMIT/NOW) --------------------

def _entry_kind(action: str, entry: float, price: float, step_d: float) -> str:
    tol = max(0.0015 * max(1.0, price), 0.15 * step_d)  # копеечный допуск (без нулевой цены)
    if action == "BUY":
        if entry > price + tol:  
            return "buy-stop"
        if entry < price - tol:  
            return "buy-limit"
        return "buy-now"
    if action == "SHORT":
        if entry < price - tol:  
            return "sell-stop"
        if entry > price + tol:  
            return "sell-limit"
        return "sell-now"
    return "wait"

# -------------------- TP/SL guards --------------------

def _apply_tp_floors(entry: float, sl: float, tp1: float, tp2: float, tp3: float,
                    action: str, hz_tag: str, price: float, atr_val: float):
    """Минимальные разумные дистанции до целей — чтобы TP не были слишком близко."""
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
    if abs(tp1 - entry) < floor1: 
        tp1 = entry + side * floor1
        
    floor2 = max(1.6 * floor1, min_rr[hz_tag] * 1.8 * risk)
    if abs(tp2 - entry) < floor2:
        tp2 = entry + side * floor2
        
    min_gap3 = max(0.8 * floor1, 0.6 * risk)
    if abs(tp3 - tp2) < min_gap3:
        tp3 = tp2 + side * min_gap3
        
    return tp1, tp2, tp3

def _order_targets(entry: float, tp1: float, tp2: float, tp3: float, action: str, eps: float = 1e-6):
    """Гарантирует порядок целей: TP1 ближе, TP3 дальше (в сторону сделки)."""
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
    """Интуитивные «предохранители» целей при мощном тренде (чтобы не ставить TP против паровоза слишком далеко)."""
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
    """
    Жёсткая проверка сторон/зазоров:
    - SL обязательно по «неправильную» сторону
    - TP обязательно по «правильную» сторону
    - минимальные зазоры для TP1/TP2/TP3
    """
    side = 1 if action == "BUY" else -1
    # минимальные зазоры (зависит от горизонта)
    min_tp_gap = {"ST": 0.40, "MID": 0.70, "LT": 1.10}[hz] * step_w
    min_tp_pct = {"ST": 0.004, "MID": 0.009, "LT": 0.015}[hz] * price
    floor_gap = max(min_tp_gap, min_tp_pct, 0.35 * abs(entry - sl) if sl != entry else 0.0)

    # SL sanity
    if action == "BUY" and sl >= entry - 0.25 * step_d:
        sl = entry - max(0.60 * step_w, 0.90 * step_d)
    if action == "SHORT" and sl <= entry + 0.25 * step_d:
        sl = entry + max(0.60 * step_w, 0.90 * step_d)

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

# -------------------- ML Model для M7 Strategy --------------------

class M7MLModel:
    """ML модель для улучшения стратегии M7"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = "models/m7_model.pkl"
        self.scaler_path = "models/m7_scaler.pkl"
        self.load_model()
        
    def load_model(self):
        """Загрузка обученной модели и скейлера"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
        except:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def save_model(self):
        """Сохранение модели и скейлера"""
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
    
    def prepare_features(self, df, ticker):
        """Подготовка признаков для ML модели"""
        # Базовые технические индикаторы
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # Отношения цен к уровням
        pivots = self.calculate_pivot_levels(df)
        for level_name, level_value in pivots.items():
            df[f'pct_to_{level_name}'] = (df['close'] - level_value) / level_value
        
        # Объемы
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Удаляем пропущенные значения
        df = df.dropna()
        
        # Выбираем последние 100 дней для обучения
        recent_data = df.tail(100).copy()
        
        # Целевая переменная: будет ли цена через 5 дней выше текущей
        recent_data['target'] = (recent_data['close'].shift(-5) > recent_data['close']).astype(int)
        
        # Признаки
        feature_columns = [
            'returns', 'volatility', 'momentum', 'sma_20', 'sma_50', 'rsi',
            'volume_ratio'
        ]
        
        # Добавляем признаки уровней
        for level_name in pivots.keys():
            feature_columns.append(f'pct_to_{level_name}')
        
        features = recent_data[feature_columns]
        target = recent_data['target']
        
        return features, target, feature_columns
    
    def calculate_rsi(self, series, period=14):
        """Расчет RSI"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_pivot_levels(self, df):
        """Расчет уровней Pivot"""
        # Используем последнюю завершенную неделю для расчета
        weekly = df.resample('W').agg({'high': 'max', 'low': 'min', 'close': 'last'})
        if len(weekly) < 2:
            return {}
        
        last_week = weekly.iloc[-2]
        H, L, C = last_week['high'], last_week['low'], last_week['close']
        
        P = (H + L + C) / 3
        R1 = (2 * P) - L
        S1 = (2 * P) - H
        R2 = P + (H - L)
        S2 = P - (H - L)
        
        return {
            'P': P, 'R1': R1, 'R2': R2, 'S1': S1, 'S2': S2
        }
    
    def train_model(self, df, ticker):
        """Обучение модели на исторических данных"""
        features, target, feature_columns = self.prepare_features(df, ticker)
        
        if len(features) < 30:  # Минимальное количество样本
            return False
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Масштабирование признаков
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Обучение модели
        self.model.fit(X_train_scaled, y_train)
        
        # Сохранение модели
        self.save_model()
        
        # Оценка точности
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Model trained. Train score: {train_score:.3f}, Test score: {test_score:.3f}")
        return True
    
    def predict_signal(self, df, ticker):
        """Предсказание сигнала с помощью ML модели"""
        if self.model is None:
            return None
        
        # Подготовка признаков для последней доступной точки
        features, _, feature_columns = self.prepare_features(df, ticker)
        if len(features) == 0:
            return None
        
        # Берем последний доступный набор признаков
        latest_features = features.iloc[-1:].copy()
        
        # Масштабирование
        scaled_features = self.scaler.transform(latest_features)
        
        # Предсказание
        prediction = self.model.predict_proba(scaled_features)[0]
        p_long = prediction[1]  # Вероятность роста
        
        return {
            'p_long': float(p_long),
            'confidence': float(min(0.95, max(0.5, p_long * 0.8 + 0.1))),  # Нормализуем уверенность
            'features': latest_features.to_dict('records')[0]
        }

# -------------------- main engine --------------------

def analyze_asset(ticker: str, horizon: str):
    cli = PolygonClient()
    cfg = _horizon_cfg(horizon)
    hz = cfg["hz"]

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
    tol_k = {"ST": 0.18, "MID": 0.22, "LT": 0.28}[hz]
    buf = tol_k * (atr_w if hz != "ST" else atr_d)

    def _near_from_below(level: float) -> bool:
        return (level is not None) and (0 <= level - price <= buf)

    def _near_from_above(level: float) -> bool:
        return (level is not None) and (0 <= price - level <= buf)

    # guard: длинные серии HA/MACD у кромки -> WAIT
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

    # базовая уверенность (ограничиваем максимум 90%)
    base = 0.55 + 0.12 * _clip01(abs(slope_norm) * 1800) + 0.08 * _clip01((vol_ratio - 0.9) / 0.6)
    if action == "WAIT":
        base -= 0.07
    conf = float(max(0.55, min(0.90, base)))  # Ограничиваем уверенность 90%

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
    # ===== end override =====

    # уровни (черновик)
    step_d, step_w = atr_d, atr_w
    if action == "BUY":
        if price < P:
            entry = max(price, S1 + 0.15 * step_w)
            sl = S1 - 0.60 * step_w
        else:
            entry = max(price, P + 0.10 * step_w)
            sl = P - 0.60 * step_w
        tp1 = entry + 0.9 * step_w
        tp2 = entry + 1.6 * step_w
        tp3 = entry + 2.3 * step_w
        alt = "Если продавят ниже и не вернут — не заходим; ждём возврата и подтверждения сверху."
    elif action == "SHORT":
        if price >= R1:
            entry = min(price, R1 - 0.15 * step_w)
            sl = R1 + 0.60 * step_w
        else:
            entry = price + 0.10 * step_d
            sl = price + 1.00 * step_d
        tp1 = entry - 0.9 * step_w
        tp2 = entry - 1.6 * step_w
        tp3 = entry - 2.3 * step_w
        alt = "Если протолкнут выше и удержат — без погони; ждём возврата и слабости сверху."
    else:  # WAIT
        entry, sl = price, price - 0.90 * step_d
        tp1, tp2, tp3 = entry + 0.7 * step_d, entry + 1.4 * step_d, entry + 2.1 * step_d
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
        "buy-stop": "Buy STOP", 
        "buy-limit": "Buy LIMIT", 
        "buy-now": "Buy NOW",
        "sell-stop": "Sell STOP",
        "sell-limit": "Sell LIMIT",
        "sell-now": "Sell NOW"
    }.get(entry_kind, "")

    # контекст-чипсы
    chips = []
    if vol_ratio > 1.05: 
        chips.append("волатильность растёт")
    if vol_ratio < 0.95: 
        chips.append("волатильность сжимается")
    if (ha_up_run >= thr_ha): 
        chips.append(f"HA зелёных: {ha_up_run}")
    if (ha_down_run >= thr_ha): 
        chips.append(f"HA красных: {ha_down_run}")

    # вероятности
    p1 = _clip01(0.58 + 0.27 * (conf - 0.55) / 0.35)
    p2 = _clip01(0.44 + 0.21 * (conf - 0.55) / 0.35)
    p3 = _clip01(0.28 + 0.13 * (conf - 0.55) / 0.35)

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

# -------------------- M7 Strategy --------------------

class M7TradingStrategy:
    """
    Краткосрочная торговая стратегия M7 на основе уровней Pivot и Fibonacci с фильтром ATR.
    Генерирует сигналы для ордеров: BUY LIMIT, SELL LIMIT, BUY STOP, SELL STOP.
    """

    def __init__(self, atr_period=14, atr_multiplier=1.5, pivot_period='D', 
                 fib_levels=[0.236, 0.382, 0.5, 0.618, 0.786]):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.pivot_period = pivot_period
        self.fib_levels = fib_levels
        
    def calculate_pivot_points(self, high, low, close):
        """Расчет уровней Pivot Point (классический метод)"""
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
        
    def calculate_fib_levels(self, high, low):
        """Расчет уровней коррекции Fibonacci"""
        diff = high - low
        fib_levels = {}
        for level in self.fib_levels:
            fib_levels[f'fib_{int(level*1000)}'] = high - level * diff
        return fib_levels
        
    def identify_key_levels(self, data):
        """Идентификация ключевых уровней поддержки и сопротивления"""
        # Группируем данные по периоду
        if self.pivot_period == 'D':
            grouped = data.resample('D')
        else:  # weekly
            grouped = data.resample('W')
            
        key_levels = {}
        
        for _, group in grouped:
            if len(group) > 0:
                high = group['high'].max()
                low = group['low'].min()
                close = group['close'].iloc[-1]
                
                # Расчет уровней
                pivot_levels = self.calculate_pivot_points(high, low, close)
                fib_levels = self.calculate_fib_levels(high, low)
                
                # Объединяем все уровни
                for name, value in {**pivot_levels, **fib_levels}.items():
                    key_levels[name] = value
                    
        return key_levels
        
    def generate_signals(self, data):
        """Генерация торговых сигналов на основе данных"""
        signals = []
        
        # Проверяем наличие необходимых колонок
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return signals
            
        # Рассчитываем ATR
        data['atr'] = _atr_like(data, self.atr_period)
        current_atr = data['atr'].iloc[-1]
        
        # Определяем ключевые уровни
        key_levels = self.identify_key_levels(data)
        
        # Текущая цена и время
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1]
        
        # Анализируем каждый уровень
        for level_name, level_value in key_levels.items():
            distance = abs(current_price - level_value) / current_atr
            
            # Если цена близко к уровню (в пределах ATR)
            if distance < self.atr_multiplier:
                # Определяем тип уровня (сопротивление или поддержка)
                is_resistance = level_value > current_price
                
                # Определяем тип сигнала
                if is_resistance:
                    signal_type = 'SELL_LIMIT'
                    entry_price = level_value * 0.998  # Немного ниже уровня
                    stop_loss = level_value * 1.02     # Выше уровня сопротивления
                    take_profit = level_value * 0.96   # Цель - следующий уровень поддержки
                else:
                    signal_type = 'BUY_LIMIT'
                    entry_price = level_value * 1.002  # Немного выше уровня
                    stop_loss = level_value * 0.98     # Ниже уровня поддержки
                    take_profit = level_value * 1.04   # Цель - следующий уровень сопротивления
                
                # Рассчитываем уверенность сигнала
                confidence = 1 - (distance / self.atr_multiplier)
                
                # Добавляем сигнал
                signals.append({
                    'type': signal_type,
                    'price': round(entry_price, 4),
                    'stop_loss': round(stop_loss, 4),
                    'take_profit': round(take_profit, 4),
                    'confidence': round(confidence, 2),
                    'level': level_name,
                    'level_value': round(level_value, 4),
                    'timestamp': current_time
                })
                
        return signals

def analyze_asset_m7(ticker, horizon="Краткосрочный", use_ml=True):
    """
    Анализ актива по стратегии M7 с интеграцией ML
    """
    cli = PolygonClient()

    # Получаем данные
    days = 120  # Больше данных для ML
    df = cli.daily_ohlc(ticker, days=days)

    # Рассчитываем историческую волатильность
    returns = np.log(df['close'] / df['close'].shift(1))
    hist_volatility = returns.std() * np.sqrt(252)  # годовая волатильность

    # Инициализируем стратегию M7
    strategy = M7TradingStrategy()

    # Генерируем сигналы
    signals = strategy.generate_signals(df)

    # Если нет сигналов, возвращаем рекомендацию WAIT
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

    # Выбираем наиболее уверенный сигнал
    best_signal = max(signals, key=lambda x: x['confidence'])
    
    # Интеграция ML
    ml_confidence = None
    if use_ml:
        try:
            ml_model = M7MLModel()
            
            # Обучаем модель, если это необходимо
            if ml_model.model is None:
                ml_model.train_model(df, ticker)
            
            # Получаем предсказание от ML модели
            ml_prediction = ml_model.predict_signal(df, ticker)
            
            if ml_prediction:
                # Корректируем уверенность на основе ML
                base_confidence = best_signal['confidence']
                ml_confidence = ml_prediction['confidence']
                
                # Комбинируем уверенность (70% ML + 30% стратегия)
                combined_confidence = 0.7 * ml_confidence + 0.3 * base_confidence
                best_signal['confidence'] = min(0.95, combined_confidence)  # Ограничиваем 95%
                
                # Корректируем действие на основе ML
                if ml_prediction['p_long'] > 0.6 and best_signal['type'].startswith('SELL'):
                    # ML предсказывает рост, но стратегия говорит продавать
                    best_signal['type'] = 'BUY_LIMIT'
                    best_signal['confidence'] *= 0.8  # Снижаем уверенность при конфликте
                elif ml_prediction['p_long'] < 0.4 and best_signal['type'].startswith('BUY'):
                    # ML предсказывает падение, но стратегия говорит покупать
                    best_signal['type'] = 'SELL_LIMIT'
                    best_signal['confidence'] *= 0.8  # Снижаем уверенность при конфликте
                    
        except Exception as e:
            logger.error(f"ML integration error: {e}")
            # Продолжаем без ML в случае ошибки

    # Рассчитываем реалистичные тейк-профиты
    current_price = float(df['close'].iloc[-1])
    entry_price = best_signal['price']
    stop_loss = best_signal['stop_loss']
    
    # Risk/Reward ratio
    risk = abs(entry_price - stop_loss)
    
    # Реалистичные тейк-профиты на основе волатильности
    volatility = df['close'].pct_change().std() * np.sqrt(252)  # Годовая волатильность
    max_daily_move = current_price * volatility / np.sqrt(252)  # Максимальное дневное движение
    
    if best_signal['type'].startswith('BUY'):
        # Для лонга
        tp1 = min(entry_price + risk * 1.5, entry_price + max_daily_move * 2)
        tp2 = min(entry_price + risk * 2.5, entry_price + max_daily_move * 3)
        tp3 = min(entry_price + risk * 4.0, entry_price + max_daily_move * 5)
    else:
        # Для шорта
        tp1 = max(entry_price - risk * 1.5, entry_price - max_daily_move * 2)
        tp2 = max(entry_price - risk * 2.5, entry_price - max_daily_move * 3)
        tp3 = max(entry_price - risk * 4.0, entry_price - max_daily_move * 5)

    # Форматируем ответ
    action = "BUY" if best_signal['type'].startswith('BUY') else "SHORT"
    
    # Контекст с информацией о ML
    context = [f"Сигнал от уровня {best_signal['level']}"]
    if ml_confidence:
        context.append(f"ML уверенность: {ml_confidence:.0%}")

    return {
        "last_price": current_price,
        "recommendation": {"action": action, "confidence": best_signal['confidence']},
        "levels": {
            "entry": entry_price,
            "sl": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3
        },
        "probs": {"tp1": 0.6, "tp2": 0.3, "tp3": 0.1},
        "context": context,
        "note_html": f"<div>Сигнал M7: {best_signal['type']} на уровне {best_signal['level_value']}</div>",
        "alt": "Торговля по стратегии M7 с ML-улучшением",
        "entry_kind": "limit",
        "entry_label": best_signal['type']
    }

if __name__ == "__main__":
    # Тестирование стратегии M7
    result = analyze_asset_m7("AAPL")
    print("M7 Strategy Result:")
    print(result)
