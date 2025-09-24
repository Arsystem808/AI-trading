### core/strategy.py
Полная стратегия с M7MLModel (обучение, сохранение, инференс), унификацией confidence и записью подсказок в Session State для breakdown. [2][3]
```python
# core/strategy.py
import os
import hashlib
import random
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, List, Tuple

from core.polygon_client import PolygonClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

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

# -------------------- Order kind (STOP/LIMIT/NOW) --------------------

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
    if action == "BUY" and sl >= entry - 0.25 * step_d:
        sl = entry - max(0.60 * step_w, 0.90 * step_d)
    if action == "SHORT" and sl <= entry + 0.25 * step_d:
        sl = entry + max(0.60 * step_w, 0.90 * step_d)
    def _push(tp, rank):
        need = floor_gap * (1.0 if rank == 1 else (1.6 if rank == 2 else 2.2))
        want = entry + side * need
        if side * (tp - entry) <= 0: return want
        if abs(tp - entry) < need:   return want
        return tp
    tp1 = _push(tp1, 1); tp2 = _push(tp2, 2); tp3 = _push(tp3, 3)
    return _order_targets(entry, tp1, tp2, tp3, action)

# -------------------- ML Model для M7 Strategy --------------------

class M7MLModel:
    """ML модель для M7: обучение, персистенция, инференс."""
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.calibrator = None
        self.model_path = "models/m7_model.pkl"
        self.scaler_path = "models/m7_scaler.pkl"
        self.calibrator_path = "models/m7_calibrator.pkl"
        self._load_local()

    def _load_local(self):
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            if os.path.exists(self.calibrator_path):
                self.calibrator = joblib.load(self.calibrator_path)
        except Exception as e:
            logger.warning(f"Local model load failed: {e}")

    def _rsi(self, s: pd.Series, period=14):
        d = s.diff()
        up = d.where(d > 0, 0).rolling(period).mean()
        dn = (-d.where(d < 0, 0)).rolling(period).mean()
        rs = up / dn
        return 100 - 100 / (1 + rs)

    def _pivots_weekly(self, df: pd.DataFrame):
        w = df.resample("W").agg({"high":"max","low":"min","close":"last"}).dropna()
        if len(w) < 2: return {}
        H,L,C = w.iloc[-2][["high","low","close"]]
        P=(H+L+C)/3; R1=(2*P)-L; S1=(2*P)-H; R2=P+(H-L); S2=P-(H-L)
        return {"P":P,"R1":R1,"R2":R2,"S1":S1,"S2":S2}

    def _build(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
        x = pd.DataFrame(index=df.index.copy())
        x["returns"]=df["close"].pct_change()
        x["volatility"]=x["returns"].rolling(20).std()
        x["momentum"]=df["close"]/df["close"].shift(5)-1
        x["sma_20"]=df["close"].rolling(20).mean()
        x["sma_50"]=df["close"].rolling(50).mean()
        x["rsi"]=self._rsi(df["close"])
        piv = self._pivots_weekly(df)
        for k,v in piv.items():
            x[f"pct_to_{k}"]=(df["close"]-v)/max(1e-9,v)
        x["volume_ma"]=df["volume"].rolling(20).mean()
        x["volume_ratio"]=df["volume"]/x["volume_ma"]
        y=(df["close"].shift(-5)>df["close"]).astype(int)
        z=pd.concat([x,y.rename("target")],axis=1).dropna()
        if z.empty: return None,None,[]
        return z.drop(columns=["target"]), z["target"], list(x.columns)

    def train_and_save(self, df: pd.DataFrame, n_estimators=600, max_depth=None, use_calibration=True):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        X,y,_=self._build(df)
        if X is None or len(X)<200:
            logger.warning("Too few samples to train")
            return None
        Xtr,Xte,Ytr,Yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        Xtr_s=self.scaler.fit_transform(Xtr); Xte_s=self.scaler.transform(Xte)
        self.model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,n_jobs=-1,random_state=42,class_weight="balanced_subsample")
        self.model.fit(Xtr_s,Ytr)
        auc=None; proba=None
        try:
            proba=self.model.predict_proba(Xte_s)[:,1]
            auc=float(roc_auc_score(Yte,proba))
        except Exception:
            pass
        if use_calibration and proba is not None:
            try:
                cal=LogisticRegression(max_iter=1000)
                cal.fit(proba.reshape(-1,1),Yte.values)
                self.calibrator=cal
                joblib.dump(cal,self.calibrator_path)
            except Exception as e:
                logger.warning(f"Calibration failed: {e}")
        os.makedirs("models",exist_ok=True)
        joblib.dump(self.model,self.model_path)
        joblib.dump(self.scaler,self.scaler_path)
        return {"auc":auc,"trained_samples":int(len(X))}

    def predict_signal(self, df: pd.DataFrame, ticker: str):
        # попробовать кастомные веса по тикеру
        if self.model is None:
            try:
                from core.model_loader import load_model_for
                mdl=load_model_for(ticker)
                if mdl is not None: self.model=mdl
            except Exception:
                pass
        if self.model is None and os.path.exists(self.model_path):
            try:
                self.model=joblib.load(self.model_path)
                self.scaler=joblib.load(self.scaler_path)
            except Exception:
                return None
        X,_,_=self._build(df)
        if X is None or len(X)==0: return None
        x_last=X.iloc[[-1]]
        try: x_last=self.scaler.transform(x_last)
        except Exception: x_last=x_last.values
        try:
            if hasattr(self.model,"predict_proba"):
                p=float(self.model.predict_proba(x_last)[:,1][0])
            elif hasattr(self.model,"decision_function"):
                m=float(self.model.decision_function(x_last).ravel()[0]); p=float(1.0/(1.0+np.exp(-m)))
            else:
                lbl=float(self.model.predict(x_last).ravel()[0]); p=float(np.tanh(abs(lbl)))
        except Exception as e:
            logger.warning(f"Prediction failed: {e}"); return None
        if self.calibrator is None and os.path.exists(self.calibrator_path):
            try: self.calibrator=joblib.load(self.calibrator_path)
            except Exception: self.calibrator=None
        if self.calibrator is not None:
            try: p=float(self.calibrator.predict_proba([[p]])[:,1][0])
            except Exception: pass
        p=max(0.0,min(1.0,p))
        return {"p_long":p,"confidence":0.5+(p-0.5)}

# -------------------- Global Strategy --------------------

def analyze_asset_global(ticker: str, horizon: str = "Краткосрочный"):
    cli = PolygonClient()
    df = cli.daily_ohlc(ticker, days=90)
    current_price = float(df['close'].iloc[-1])
    short_ma = df['close'].rolling(20).mean().iloc[-1]
    long_ma = df['close'].rolling(50).mean().iloc[-1]
    if short_ma > long_ma:
        action = "BUY"; confidence = 0.69
    else:
        action = "SHORT"; confidence = 0.65
    atr = float(_atr_like(df, n=14).iloc[-1])
    if action == "BUY":
        entry=current_price; sl=current_price-2*atr; tp1=current_price+1*atr; tp2=current_price+2*atr; tp3=current_price+3*atr
        alt="Покупка по рынку с консервативными целями"
    else:
        entry=current_price; sl=current_price+2*atr; tp1=current_price-1*atr; tp2=current_price-2*atr; tp3=current_price-3*atr
        alt="Продажа по рынку с консервативными целями"
    probs={"tp1":0.68,"tp2":0.52,"tp3":0.35}
    note_html=f"<div style='margin-top:10px; opacity:0.95;'>Global Strategy: {action} с уверенностью {confidence:.0%}.</div>"
    return {"last_price":current_price,"recommendation":{"action":action,"confidence":confidence},
            "levels":{"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"tp3":tp3},
            "probs":probs,"context":[],"note_html":note_html,"alt":alt,"entry_kind":"market","entry_label":f"{action} NOW"}

# -------------------- M7 Strategy --------------------

class M7TradingStrategy:
    def __init__(self, atr_period=14, atr_multiplier=1.5, pivot_period='D', fib_levels=[0.236,0.382,0.5,0.618,0.786]):
        self.atr_period=atr_period; self.atr_multiplier=atr_multiplier; self.pivot_period=pivot_period; self.fib_levels=fib_levels
    def calculate_pivot_points(self, high, low, close):
        pivot=(high+low+close)/3; r1=(2*pivot)-low; r2=pivot+(high-low); r3=high+2*(pivot-low)
        s1=(2*pivot)-high; s2=pivot-(high-low); s3=low-2*(high-pivot)
        return {'pivot':pivot,'r1':r1,'r2':r2,'r3':r3,'s1':s1,'s2':s2,'s3':s3}
    def calculate_fib_levels(self, high, low):
        diff=high-low; return {f'fib_{int(level*1000)}': high - level * diff for level in self.fib_levels}
    def identify_key_levels(self, data):
        grouped=data.resample('D') if self.pivot_period=='D' else data.resample('W'); key={}
        for _,g in grouped:
            if len(g)>0:
                high=g['high'].max(); low=g['low'].min(); close=g['close'].iloc[-1]
                key.update(self.calculate_pivot_points(high,low,close))
                key.update(self.calculate_fib_levels(high,low))
        return key
    def generate_signals(self, data):
        signals=[]; req=['high','low','close']
        if not all(c in data.columns for c in req): return signals
        d=data.copy(); d['atr']=_atr_like(d,self.atr_period); current_atr=float(d['atr'].iloc[-1])
        key=self.identify_key_levels(d); current_price=float(d['close'].iloc[-1]); ts=d.index[-1]
        for name,lv in key.items():
            distance=abs(current_price-lv)/max(1e-9,current_atr)
            if distance<self.atr_multiplier:
                if lv>current_price:
                    t='SELL_LIMIT'; entry=lv*0.998; sl=lv*1.02; tp=lv*0.96
                else:
                    t='BUY_LIMIT'; entry=lv*1.002; sl=lv*0.98; tp=lv*1.04
                conf=1-(distance/self.atr_multiplier)
                signals.append({'type':t,'price':round(entry,4),'stop_loss':round(sl,4),'take_profit':round(tp,4),
                                'confidence':round(conf,2),'level':name,'level_value':round(lv,4),'timestamp':ts})
        return signals

def analyze_asset_m7(ticker, horizon="Краткосрочный", use_ml=True):
    cli = PolygonClient()
    df = cli.daily_ohlc(ticker, days=120)
    strategy = M7TradingStrategy()
    signals = strategy.generate_signals(df)
    if not signals:
        return {"last_price": float(df['close'].iloc[-1]),
                "recommendation": {"action": "WAIT", "confidence": 0.5},
                "levels": {"entry": 0, "sl": 0, "tp1": 0, "tp2": 0, "tp3": 0},
                "probs": {"tp1": 0, "tp2": 0, "tp3": 0},
                "context": ["Нет сигналов по M7"],
                "note_html": "<div>Ожидание сигналов M7</div>",
                "alt": "Ожидание", "entry_kind": "wait", "entry_label": "WAIT"}
    best = max(signals, key=lambda s: s['confidence'])
    mlp = None
    if use_ml:
        try:
            ml = M7MLModel()
            if ml.model is None:
                ml.train_and_save(df, n_estimators=600, max_depth=None, use_calibration=True)
            mlp = ml.predict_signal(df, ticker)
        except Exception as e:
            logger.error(f"ML integration error: {e}")
    current_price = float(df['close'].iloc[-1])
    entry = best['price']; sl = best['stop_loss']; risk = abs(entry - sl)
    vol = df['close'].pct_change().std() * np.sqrt(252)
    max_daily_move = current_price * vol / np.sqrt(252)
    if best['type'].startswith('BUY'):
        tp1 = min(entry + risk * 1.5, entry + max_daily_move * 2)
        tp2 = min(entry + risk * 2.5, entry + max_daily_move * 3)
        tp3 = min(entry + risk * 4.0, entry + max_daily_move * 5)
    else:
        tp1 = max(entry - risk * 1.5, entry - max_daily_move * 2)
        tp2 = max(entry - risk * 2.5, entry - max_daily_move * 3)
        tp3 = max(entry - risk * 4.0, entry - max_daily_move * 5)
    action = "BUY" if best['type'].startswith('BUY') else "SHORT"
    # Единая формула confidence: правила 44% + дельта от ML
    rules_pct = 44.0
    ai_delta_pct = (float(mlp.get('confidence', 0.5)) * 100.0 - 50.0) if mlp else 0.0
    overall_pct = float(max(0.0, min(100.0, rules_pct + ai_delta_pct)))
    best['confidence'] = overall_pct / 100.0
    # Подсказки для UI через Session State
    try:
        import streamlit as st
        st.session_state["last_overall_conf_pct"] = overall_pct
        st.session_state["last_rules_pct"] = rules_pct
    except Exception:
        pass
    context = [f"Сигнал от уровня {best['level']}"]
    if mlp:
        context.append(f"ML p_long={mlp['p_long']:.2f}")
    probs = {"tp1": 0.63, "tp2": 0.52, "tp3": 0.53}
    note_html = f"<div style='margin-top:10px; opacity:0.95;'>M7 Strategy: {best['type']} @ {best['level_value']}. Синхронизированный confidence и AI override.</div>"
    return {
        "last_price": current_price,
        "recommendation": {"action": action, "confidence": best['confidence']},
        "levels": {"entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3},
        "probs": probs,
        "context": context,
        "note_html": note_html,
        "alt": "M7+ML",
        "entry_kind": "limit",
        "entry_label": best['type'],
        "confidence_breakdown": {"rules_pct": rules_pct, "ai_override_delta_pct": ai_delta_pct, "overall_pct": overall_pct}
    }

# -------------------- Strategy Router --------------------

def analyze_asset(ticker: str, horizon: str, strategy: str = "W7"):
    if strategy == "Global":
        return analyze_asset_global(ticker, horizon)
    elif strategy == "M7":
        return analyze_asset_m7(ticker, horizon)
    else:
        # По умолчанию пусть будет M7; при необходимости добавьте W7
        return analyze_asset_m7(ticker, horizon)
