# jobs/train_daily.py
# Nightly EOD training job with Purged/Embargo CV, probability calibration and atomic model updates.

from __future__ import annotations

import os
import json
import math
import time
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
import joblib

# Optional Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False

# Repo paths
try:
    # Reuse the same models/ folder and client used in prod
    from core.model_loader import MODELS as MODELS_DIR  # Path("models")
except Exception:
    MODELS_DIR = Path("models")

try:
    from core.polygon_client import PolygonClient
except Exception:
    # Fallback stub consistent with strategy.py soft-imports
    class PolygonClient:
        def daily_ohlc(self, ticker: str, days: int = 400):
            raise RuntimeError("PolygonClient unavailable")

# ------------- Config -------------

AGENTS = [
    "arxora_m7pro",   # priority 1 in loader
    "global",         # priority 2
    "alphapulse",     # priority 3
    "octopus",        # priority 4
]

DEFAULT_TICKERS = os.getenv("TRAIN_TICKERS", "AAPL,SPY,QQQ,NVDA,BTCUSD,ETHUSD").split(",")
EOD_TIMEZONE = os.getenv("EOD_TZ", "US/Eastern")  # used implicitly by data source
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "450"))
FORWARD_H = int(os.getenv("FORWARD_H", "10"))           # forward event horizon (days)
ATR_LEN = int(os.getenv("ATR_LEN", "14"))
UP_ATR = float(os.getenv("UP_ATR", "2.0"))              # take-profit multiple of ATR
DN_ATR = float(os.getenv("DN_ATR", "2.0"))              # stop-loss multiple of ATR
N_SPLITS = int(os.getenv("N_SPLITS", "5"))              # CV splits
EMBARGO_DAYS = int(os.getenv("EMBARGO_DAYS", "3"))      # embargo to remove leakage
N_TRIALS = int(os.getenv("N_TRIALS", "40"))             # Optuna trials
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ------------- Utils -------------

def atomic_write_model(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)  # atomic on POSIX/NTFS

def save_metrics_json(path: Path, metrics: Dict) -> None:
    p = path.with_suffix(path.suffix + ".metrics.json")
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=float)
    os.replace(tmp, p)

# ------------- Features -------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def compute_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    # df with columns: open, high, low, close
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=n).mean()
    return atr

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    px = df["close"].astype(float)
    vol = df.get("volume", pd.Series(index=df.index, dtype=float)).astype(float)

    f = pd.DataFrame(index=df.index)
    f["ret_1"] = px.pct_change(1)
    f["ret_3"] = px.pct_change(3)
    f["ret_5"] = px.pct_change(5)
    f["rvol_10"] = vol.rolling(10).mean() / (vol.rolling(30).mean() + 1e-9)
    f["sma_10"] = px.rolling(10).mean() / (px.rolling(30).mean() + 1e-9) - 1.0
    f["ema_12"] = _ema(px, 12) / (px + 1e-9) - 1.0
    f["ema_26"] = _ema(px, 26) / (px + 1e-9) - 1.0
    macd = _ema(px, 12) - _ema(px, 26)
    f["macd"] = macd
    f["macd_sig"] = _ema(macd, 9)
    f["atr"] = compute_atr(df, ATR_LEN) / (px + 1e-9)
    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f

# ------------- Labeling (triple-barrier style) -------------

def triple_barrier_labels(df: pd.DataFrame, h: int, up_k: float, dn_k: float, atr_len: int) -> pd.Series:
    """
    Label = 1 if upper barrier hits first within h days, 0 if lower hits first, NaN if none.
    Barriers are entry +/- k*ATR(entry).
    """
    close = df["close"].astype(float).values
    atr = compute_atr(df, atr_len).values
    n = len(df)
    y = np.full(n, np.nan, dtype=float)

    for i in range(n - h):
        if np.isnan(atr[i]) or np.isnan(close[i]):
            continue
        up = close[i] * (1 + up_k * atr[i] / (close[i] + 1e-9))
        dn = close[i] * (1 - dn_k * atr[i] / (close[i] + 1e-9))
        win_u = False
        win_d = False
        for j in range(1, h + 1):
            hi = df["high"].iloc[i + j]
            lo = df["low"].iloc[i + j]
            if not np.isnan(hi) and hi >= up:
                win_u = True
                break
            if not np.isnan(lo) and lo <= dn:
                win_d = True
                break
        if win_u and not win_d:
            y[i] = 1.0
        elif win_d and not win_u:
            y[i] = 0.0
        # else stays NaN
    return pd.Series(y, index=df.index)

# ------------- Purged/Embargo CV -------------

def purged_embargo_splits(times: pd.DatetimeIndex, n_splits: int, embargo_days: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Simple time-ordered KFold with embargo days removed from train around each test fold.
    For rigorous CPCV see research code; this is a pragmatic variant.
    """
    n = len(times)
    indices = np.arange(n)
    kf = KFold(n_splits=n_splits, shuffle=False)
    for train_idx, test_idx in kf.split(indices):
        # Apply embargo around test boundaries
        if len(test_idx) == 0:
            continue
        test_start = test_idx[0]
        test_end = test_idx[-1]
        embargo_left = max(0, test_start - embargo_days)
        embargo_right = min(n - 1, test_end + embargo_days)
        mask = np.ones(n, dtype=bool)
        # Remove test
        mask[test_idx] = False
        # Remove embargo window
        mask[embargo_left:test_start] = False
        mask[test_end + 1:embargo_right + 1] = False
        yield indices[mask], test_idx

# ------------- Training core -------------

@dataclass
class TrainResult:
    model_path: Path
    auc: float
    logloss: float
    n_samples: int
    n_pos: int
    n_neg: int
    params: Dict

def get_base_estimator(trial=None):
    # HistGB is fast and robust; if Optuna is available, tune a few knobs.
    if trial is None:
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=None,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
            random_state=RANDOM_STATE,
        )
    return HistGradientBoostingClassifier(
        learning_rate=trial.suggest_float("lr", 0.02, 0.2, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        max_leaf_nodes=trial.suggest_int("max_leaf", 16, 128, log=True),
        min_samples_leaf=trial.suggest_int("min_leaf", 10, 50),
        l2_regularization=trial.suggest_float("l2", 1e-8, 1e-1, log=True),
        random_state=RANDOM_STATE,
    )

def fit_with_cv(X: pd.DataFrame, y: pd.Series, times: pd.DatetimeIndex, trial=None) -> Tuple[Pipeline, Dict, float, float]:
    # Build pipeline: scaler + calibrated classifier
    base = get_base_estimator(trial)
    # calibrator cv only on train fold to avoid leakage
    clf = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    aucs, losses = [], []
    for tr_idx, te_idx in purged_embargo_splits(times, N_SPLITS, EMBARGO_DAYS):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        # avoid degenerate predictions
        proba = np.clip(proba, 1e-6, 1 - 1e-6)
        aucs.append(roc_auc_score(yte, proba))
        losses.append(log_loss(yte, proba))
        if trial is not None:
            trial.report(np.mean(aucs), len(aucs))
            if trial.should_prune():
                raise optuna.TrialPruned()
    return pipe, {"estimator": str(base)}, float(np.mean(aucs)), float(np.mean(losses))

def optimize(X: pd.DataFrame, y: pd.Series, times: pd.DatetimeIndex) -> Tuple[Pipeline, Dict, float, float]:
    if not _HAS_OPTUNA:
        # Fallback without Optuna
        return fit_with_cv(X, y, times, trial=None)

    def objective(trial):
        _, _, auc, loss = fit_with_cv(X, y, times, trial=trial)
        # maximize AUC, minimize logloss -> primary: AUC
        trial.set_user_attr("logloss", loss)
        return auc

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE), pruner=MedianPruner())
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    best = study.best_trial
    model, params, auc, loss = fit_with_cv(X, y, times, trial=best)
    params.update({"best_value_auc": best.value, "best_logloss": best.user_attrs.get("logloss", None), "n_trials": len(study.trials)})
    return model, params, auc, loss

# ------------- Dataset builder -------------

def load_daily_ohlc(ticker: str, days: int) -> pd.DataFrame:
    client = PolygonClient()
    df = client.daily_ohlc(ticker, days=days)
    # Expect columns: timestamp, open, high, low, close, volume
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
        else:
            raise ValueError("daily_ohlc must provide datetime index or 'timestamp' column")
    df = df.sort_index()
    return df

def build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    feats = compute_features(df)
    labels = triple_barrier_labels(df, FORWARD_H, UP_ATR, DN_ATR, ATR_LEN)
    # Align to common index and drop NaNs in y
    common = feats.join(labels.rename("y")).dropna()
    X = common.drop(columns=["y"])
    y = common["y"].astype(int)
    times = common.index
    return X, y, times

# ------------- Train per ticker/agent -------------

def train_for_ticker_agent(ticker: str, agent: str) -> Optional[TrainResult]:
    try:
        df = load_daily_ohlc(ticker, LOOKBACK_DAYS)
    except Exception as e:
        print(f"[{agent}:{ticker}] data error: {e}")
        return None

    X, y, times = build_xy(df)
    if y.sum() == 0 or y.sum() == len(y):
        print(f"[{agent}:{ticker}] degenerate labels, skip")
        return None

    model, params, auc, loss = optimize(X, y, times)

    # Refit on all data for production
    model.fit(X, y)

    out_path = MODELS_DIR / f"{agent}_{ticker.upper()}.joblib"
    atomic_write_model(out_path, model)

    metrics = {
        "ticker": ticker.upper(),
        "agent": agent,
        "samples": int(len(y)),
        "pos": int(int(y.sum())),
        "neg": int(int((1 - y).sum())),
        "auc_cv": float(auc),
        "logloss_cv": float(loss),
        "params": params,
        "horizon_days": FORWARD_H,
        "atr_len": ATR_LEN,
        "up_atr": UP_ATR,
        "dn_atr": DN_ATR,
        "n_splits": N_SPLITS,
        "embargo_days": EMBARGO_DAYS,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "random_state": RANDOM_STATE,
        "optuna": _HAS_OPTUNA,
    }
    save_metrics_json(out_path, metrics)
    print(f"[{agent}:{ticker}] saved -> {out_path.name} | AUC={auc:.4f} | logloss={loss:.4f}")
    return TrainResult(out_path, auc, loss, len(y), int(y.sum()), int((1 - y).sum()), params)

# ------------- Entry point -------------

def main():
    tickers = [t.strip() for t in DEFAULT_TICKERS if t.strip()]
    results = []
    for agent in AGENTS:
        for t in tickers:
            res = train_for_ticker_agent(t, agent)
            if res:
                results.append(res)

    # Simple summary
    summary = [
        {
            "model": r.model_path.name,
            "auc": r.auc,
            "logloss": r.logloss,
            "n": r.n_samples,
            "pos": r.n_pos,
            "neg": r.n_neg,
        }
        for r in results
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
