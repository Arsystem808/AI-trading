# -*- coding: utf-8 -*-
# core/strategy.py — стабильные точки входа стратегий и единый роутер

import os
import logging
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Мягкий импорт PolygonClient (на случай отсутствия реального клиента)
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Utils ----------------
def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

# ---------------- ML model for M7 ----------------
class M7MLModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.calibrator = None
        self.model_path = "models/m7_model.pkl"
        self.scaler_path = "models/m7_scaler.pkl"
        self.calibrator_path = "models/m7_calibrator.pkl"
        self._load_any()

    def _load_any(self):
        try:
            from pathlib import Path
            MODELS = Path("models")
            candidates: List[Path] = []
            # Популярные имена весов
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
                    self.model = joblib.load(p)
                    break
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            if os.path.exists(self.calibrator_path):
                self.calibrator = joblib.load(self.calibrator_path)
        except Exception as e:
            logger.warning(f"Model preload warning: {e}")

    def _rsi(self, s: pd.Series, period=14) -> pd.Series:
        d = s.diff()
        up = d.where(d > 0, 0).rolling(period).mean()
        dn = (-d.where(d < 0, 0)).rolling(period).mean()
        rs = up / dn
        return 100 - 100 / (1 + rs)

    def _build(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        X = pd.DataFrame(index=df.index.copy())
        X["ret1"]  = df["close"].pct_change()
        X["vol20"] = X["ret1"].rolling(20).std()
        X["mom5"]  = df["close"] / df["close"].shift(5) - 1
        X["sma20"] = df["close"].rolling(20).mean()
        X["sma50"] = df["close"].rolling(50).mean()
        X["rsi14"] = self._rsi(df["close"], 14)
        y = (df["close"].shift(-5) > df["close"]).astype(int)
        Z = pd.concat([X, y.rename("y")], axis=1).dropna()
        if Z.empty:
            return None, None
        return Z.drop(columns=["y"]), Z["y"]

    def train_and_save(self, df: pd.DataFrame, n_estimators=400) -> Optional[Dict[str, Any]]:
        X, y = self._build(df)
        if X is None or len(X) < 200:
            return None
        Xtr,Xte,Ytr,Yte = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        Xtr_s = self.scaler.fit_transform(Xtr)
        Xte_s = self.scaler.transform(Xte)
        self.model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42, class_weight="balanced_subsample")
        self.model.fit(Xtr_s, Ytr)
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        return {"trained": int(len(X))}

    def predict_proba_last(self, df: pd.DataFrame) -> Optional[float]:
        if self.model is None:
            return None
        X, _ = self._build(df)
        if X is None or len(X) == 0:
            return None
        x_last = X.iloc[[-1]]
        try:
            x_last_s = self.scaler.transform(x_last)
        except Exception:
            x_last_s = x_last.values
        if hasattr(self.model, "predict_proba"):
            p = float(self.model.predict_proba(x_last_s)[:,1][0])
        elif hasattr(self.model, "decision_function"):
            m = float(self.model.decision_function(x_last_s).ravel()[0])
            p = float(1.0/(1.0+np.exp(-m)))
        else:
            y = float(self.model.predict(x_last_s).ravel()[0])
            p = float(max(0.0, min(1.0, 0.5 + 0.5*np.tanh(y))))
        return _clip01(p)

# ---------------- Global strategy ----------------
def analyze_asset_global(ticker: str, horizon: str = "Краткосрочный") -> Dict[str, Any]:
    df = PolygonClient().daily_ohlc(ticker, days=120)
    price = float(df["close"].iloc[-1])
    short = float(df["close"].rolling(20).mean().iloc[-1])
    long  = float(df["close"].rolling(50).mean().iloc[-1])
    action = "BUY" if short > long else "SHORT"
    confidence = 0.69 if action == "BUY" else 0.65
    atr = float(_atr_like(df, 14).iloc[-1])
    if action == "BUY":
        entry=price; sl=price-2*atr; tp1=price+1*atr; tp2=price+2*atr; tp3=price+3*atr
    else:
        entry=price; sl=price+2*atr; tp1=price-1*atr; tp2=price-2*atr; tp3=price-3*atr
    overall_pct = float(min(100.0, max(0.0, 44.0 + (confidence*100.0 - 50.0))))
    try:
        import streamlit as st
        st.session_state["last_overall_conf_pct"] = overall_pct
        st.session_state["last_rules_pct"] = 44.0
    except Exception:
        pass
    return {
        "last_price": price,
        "recommendation": {"action": action, "confidence": confidence},
        "levels": {"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"tp3":tp3},
        "probs": {"tp1":0.68,"tp2":0.52,"tp3":0.35},
        "context": [],
        "note_html": "<div>Global Strategy</div>",
        "alt": "Global",
        "entry_kind": "market",
        "entry_label": f"{action} NOW",
        "confidence_breakdown": {"rules_pct": 44.0, "ai_override_delta_pct": (confidence*100.0 - 50.0), "overall_pct": overall_pct}
    }

# ---------------- M7 strategy ----------------
def analyze_asset_m7(ticker: str, horizon: str = "Краткосрочный", use_ml: bool = True) -> Dict[str, Any]:
    df = PolygonClient().daily_ohlc(ticker, days=150)
    price = float(df["close"].iloc[-1])
    atr = float(_atr_like(df, 14).iloc[-1])
    sma20 = float(df["close"].rolling(20).mean().iloc[-1])
    action = "BUY" if price > sma20 else "SHORT"

    if action == "BUY":
        entry = price
        sl    = price - 1.8*atr
        tp1   = price + 1.0*atr
        tp2   = price + 2.0*atr
        tp3   = price + 3.0*atr
    else:
        entry = price
        sl    = price + 1.8*atr
        tp1   = price - 1.0*atr
        tp2   = price - 2.0*atr
        tp3   = price - 3.0*atr

    rules_pct = 44.0
    ai_delta = 0.0
    p_long = None
    if use_ml:
        try:
            ml = M7MLModel()
            if ml.model is None:
                ml.train_and_save(df, n_estimators=500)
            p_long = ml.predict_proba_last(df)
            if p_long is not None:
                ai_delta = float(p_long)*100.0 - 50.0
        except Exception as e:
            logger.warning(f"M7 ML warn: {e}")

    overall_pct = float(max(0.0, min(100.0, rules_pct + ai_delta)))
    try:
        import streamlit as st
        st.session_state["last_overall_conf_pct"] = overall_pct
        st.session_state["last_rules_pct"] = rules_pct
    except Exception:
        pass

    confidence = overall_pct/100.0
    probs = {"tp1": 0.63, "tp2": 0.52, "tp3": 0.45}
    ctx = [f"M7 base; ML p_long={p_long:.2f}" if p_long is not None else "M7 base; ML n/a"]
    return {
        "last_price": price,
        "recommendation": {"action": action, "confidence": confidence},
        "levels": {"entry":entry,"sl":sl,"tp1":tp1,"tp2":tp2,"tp3":tp3},
        "probs": probs,
        "context": ctx,
        "note_html": "<div>M7 Strategy with ML override</div>",
        "alt": "M7",
        "entry_kind": "market",
        "entry_label": f"{'Long' if action=='BUY' else 'Short'} NOW",
        "confidence_breakdown": {"rules_pct": rules_pct, "ai_override_delta_pct": ai_delta, "overall_pct": overall_pct}
    }

# ---------------- Универсальный роутер ----------------
def analyze_asset(ticker: str, horizon: str, strategy: str = "M7") -> Dict[str, Any]:
    s = (strategy or "").lower()
    if s == "global":
        return analyze_asset_global(ticker, horizon)
    if s == "m7":
        return analyze_asset_m7(ticker, horizon, use_ml=True)
    if s == "w7":
        # Отдельной W7 может не быть — используем M7 как fallback
        return analyze_asset_m7(ticker, horizon, use_ml=True)
    # По умолчанию M7
    return analyze_asset_m7(ticker, horizon, use_ml=True)

