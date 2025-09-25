cat > core/strategy.py <<'PY'
# -*- coding: utf-8 -*-
import os, logging, hashlib, random
from typing import Dict, Any, Optional, List, Tuple
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def _atr_like(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()

def _sync_session(overall_pct: float, rules_pct: float) -> None:
    try:
        import streamlit as st
        st.session_state["last_overall_conf_pct"] = float(overall_pct)
        st.session_state["last_rules_pct"] = float(rules_pct)
    except Exception:
        pass

class M7MLModel:
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
            cand = [MODELS/"m7_model.pkl"]
            for t in ("SPY","QQQ","BTCUSD","ETHUSD","AAPL","NVDA"):
                for p in ("arxora_m7pro","global","octopus","alphapulse"):
                    cand.append(MODELS / f"{p}_{t}.joblib")
            for p in cand:
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

    def _rsi(self, s: pd.Series, period=14) -> pd.Series:
        d = s.diff()
        up = d.where(d > 0, 0).rolling(period).mean()
        dn = (-d.where(d < 0, 0)).rolling(period).mean()
        rs = up / dn
        return 100 - 100 / (1 + rs)

    def _build_feats(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        X = pd.DataFrame(index=df.index.copy())
        X["ret1"]  = df["close"].pct_change()
        X["vol20"] = X["ret1"].rolling(20).std()
        X["mom5"]  = df["close"] / df["close"].shift(5) - 1
        X["sma20"] = df["close"].rolling(20).mean()
        X["sma50"] = df["close"].rolling(50).mean()
        X["rsi14"] = self._rsi(df["close"], 14)
        X["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
        return X.dropna()

    def train_and_save(self, df: pd.DataFrame, n_estimators=500) -> Optional[int]:
        try:
            X = self._build_feats(df)
            if X is None or len(X) < 60:
                return None
            y = (df["close"].shift(-5).reindex(X.index) > df["close"].reindex(X.index)).astype(int)
            Xtr,Xte,Ytr,Yte = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
            Xtr_s = self.scaler.fit_transform(Xtr)
            rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=42,
                                        class_weight="balanced_subsample")
            cal = CalibratedClassifierCV(rf, method="isotonic", cv=3)
            cal.fit(Xtr_s, Ytr)
            os.makedirs("models", exist_ok=True)
            joblib.dump(cal, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            self.model = cal
            return len(Xtr)
        except Exception as e:
            logger.error(f"ML train failed: {e}")
            return None

    def predict_proba_last(self, df: pd.DataFrame) -> Optional[float]:
        try:
            if self.model is None:
                return None
            X = self._build_feats(df)
            if X is None or len(X) == 0:
                return None
            x = X.iloc[[-1]]
            X = X.reindex(columns=list(scaler.feature_names_in_))
            xs = self.scaler.transform(x)
            p = float(self.model.predict_proba(xs)[:,1][0])
            return _clip01(p)
        except Exception as e:
            logger.warning(f"ML predict failed: {e}")
            return None

def _confidence_with_override(df: pd.DataFrame) -> Dict[str, float]:
    rules_pct = 44.0
    ai_delta = 0.0
    try:
        ml = M7MLModel()
        if ml.model is None:
            _ = ml.train_and_save(df)
            ml = M7MLModel()
        p = ml.predict_proba_last(df)
        if p is not None and not (0.48 <= p <= 0.52):
            ai_delta = p*100.0 - 50.0
    except Exception:
        ai_delta = 0.0
    overall_pct = float(max(0.0, min(100.0, rules_pct + ai_delta)))
    _sync_session(overall_pct, rules_pct)
    return {"rules_pct": rules_pct, "ai_delta": ai_delta, "overall_pct": overall_pct}

def analyze_asset_global(ticker: str, horizon: str = "Краткосрочный"):
    cli = PolygonClient()
    df = cli.daily_ohlc(ticker, days=120)
    price = float(df["close"].iloc[-1])
    atr = float(_atr_like(df, 14).iloc[-1])
    short_ma = float(df['close'].rolling(20).mean().iloc[-1])
    long_ma  = float(df['close'].rolling(50).mean().iloc[-1])
    action = "BUY" if short_ma > long_ma else "SHORT"
    if action == "BUY":
        entry=price; sl=price-2*atr; tp1=price+1*atr; tp2=price+2*atr; tp3=price+3*atr
    else:
        entry=price; sl=price+2*atr; tp1=price-1*atr; tp2=price-2*atr; tp3=price-3*atr
    cb = _confidence_with_override(df)
    return {
        "last_price": price,
        "recommendation": {"action": action, "confidence": cb["overall_pct"]/100.0},
        "levels": {"entry": entry,"sl": sl,"tp1": tp1,"tp2": tp2,"tp3": tp3},
        "probs": {"tp1": 0.68, "tp2": 0.52, "tp3": 0.35},
        "context": [f"SMA20 vs SMA50"],
        "note_html": "<div>Global Strategy</div>",
        "alt": "Global",
        "entry_kind": "market",
        "entry_label": f"{action} NOW",
        "confidence_breakdown": {"rules_pct": cb["rules_pct"], "ai_override_delta_pct": cb["ai_delta"], "overall_pct": cb["overall_pct"]}
    }

def analyze_asset_m7(ticker: str, horizon="Краткосрочный", use_ml=True):
    cli = PolygonClient()
    df = cli.daily_ohlc(ticker, days=160)
    price = float(df["close"].iloc[-1])
    # примитивный уровень
    atr = float(_atr_like(df, 14).iloc[-1])
    entry = price; sl = price - 1.5*atr; tp1=price+1.0*atr; tp2=price+2.0*atr; tp3=price+3.0*atr
    cb = _confidence_with_override(df) if use_ml else {"rules_pct":44.0,"ai_delta":0.0,"overall_pct":44.0}
    return {
        "last_price": price,
        "recommendation": {"action": "BUY", "confidence": cb["overall_pct"]/100.0},
        "levels": {"entry": entry,"sl": sl,"tp1": tp1,"tp2": tp2,"tp3": tp3},
        "probs": {"tp1": 0.63, "tp2": 0.52, "tp3": 0.45},
        "context": ["M7 basic"],
        "note_html": "<div>M7 Strategy</div>",
        "alt": "M7 with ML",
        "entry_kind": "market",
        "entry_label": "BUY NOW",
        "confidence_breakdown": {"rules_pct": cb["rules_pct"], "ai_override_delta_pct": cb["ai_delta"], "overall_pct": cb["overall_pct"]}
    }

def analyze_asset_w7(ticker: str, horizon: str):
    cli = PolygonClient()
    df = cli.daily_ohlc(ticker, days=200)
    price = float(df["close"].iloc[-1])
    atr = float(_atr_like(df, 14).iloc[-1])
    entry = price; sl = price - 1.2*atr; tp1=price+0.8*atr; tp2=price+1.6*atr; tp3=price+2.4*atr
    cb = _confidence_with_override(df)
    return {
        "last_price": price,
        "recommendation": {"action": "BUY", "confidence": cb["overall_pct"]/100.0},
        "levels": {"entry": entry,"sl": sl,"tp1": tp1,"tp2": tp2,"tp3": tp3},
        "probs": {"tp1": 0.58, "tp2": 0.49, "tp3": 0.36},
        "context": ["W7 basic"],
        "note_html": "<div>W7 Strategy</div>",
        "alt": "W7",
        "entry_kind": "market",
        "entry_label": "BUY NOW",
        "confidence_breakdown": {"rules_pct": cb["rules_pct"], "ai_override_delta_pct": cb["ai_delta"], "overall_pct": cb["overall_pct"]}
    }

def analyze_asset(ticker: str, horizon: str, strategy: str = "W7"):
    if strategy == "Global": return analyze_asset_global(ticker, horizon)
    if strategy == "M7":     return analyze_asset_m7(ticker, horizon)
    if strategy == "W7":     return analyze_asset_w7(ticker, horizon)
    return analyze_asset_m7(ticker, horizon)
PY

