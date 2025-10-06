#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def log(msg):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}", flush=True)

def normalize_symbol(sym: str) -> str:
    # Для крипто символов добавляем префикс X:, если его нет (пример: BTCUSD -> X:BTCUSD)
    if sym.upper().endswith("USD") and not (sym.startswith("X:") or sym.startswith("C:") or sym.startswith("I:")):
        return f"X:{sym.upper()}"
    return sym.upper()

def get_with_retries(url: str, max_retries: int = 5, timeout: int = 60):
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        r = requests.get(url, timeout=timeout)
        if r.status_code == 429:
            log(f"429 rate limit; retry {attempt}/{max_retries} after {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("Exceeded retries due to rate limiting or transient errors")

def fetch_polygon_daily(symbol: str, start: str, end: str, api_key: str) -> pd.DataFrame:
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    )
    safe = url.replace(api_key, '***')
    log(f"GET {safe}")
    r = get_with_retries(url)
    data = r.json()
    if "results" not in data or not data["results"]:
        raise RuntimeError(f"No results for {symbol}")
    df = pd.DataFrame(data["results"])
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize("UTC").dt.date
    df = df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"})
    df = df[["date","Open","High","Low","Close","Volume"]].dropna().sort_values("date").reset_index(drop=True)
    return df

def make_features(df: pd.DataFrame):
    d = df.copy()
    d["ret"] = d["Close"].pct_change()
    d["ret_1d"] = d["ret"].shift(1)
    d["ret_5d"] = d["Close"].pct_change(5)
    d["ret_10d"] = d["Close"].pct_change(10)
    d["vol_5d"] = d["ret"].rolling(5).std()
    d["vol_10d"] = d["ret"].rolling(10).std()
    d["ma_5"] = d["Close"].rolling(5).mean()
    d["ma_10"] = d["Close"].rolling(10).mean()
    d["ma_ratio"] = d["ma_5"] / d["ma_10"]
    delta = d["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / (down.replace(0, np.nan))
    d["rsi_14"] = 100 - (100 / (1 + rs))
    d["future_ret"] = d["Close"].pct_change().shift(-1)
    d["y"] = (d["future_ret"] > 0).astype(int)
    d = d.dropna().reset_index(drop=True)
    feature_cols = ["ret_1d","ret_5d","ret_10d","vol_5d","vol_10d","ma_ratio","rsi_14"]
    X = d[feature_cols].values
    y = d["y"].values
    meta = {
        "feature_cols": feature_cols,
        "n_samples": int(d.shape[0]),
        "date_start": str(d["date"].iloc[0]),
        "date_end": str(d["date"].iloc[-1]),
    }
    return d, X, y, meta

def time_split(X, y, test_size=0.2):
    n = len(y)
    n_train = int((1 - test_size) * n)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:], n_train

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--outdir", default="artifacts")
    args = ap.parse_args()

    t0 = time.time()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("POLYGON_API_KEY is not set")

    sym = normalize_symbol(args.symbol)
    log(f"Using Polygon for {sym}")
    raw = fetch_polygon_daily(sym, args.start, args.end, api_key)
    log(f"Downloaded {len(raw)} rows")

    df, X, y, meta = make_features(raw)
    if len(y) < 200:
        raise RuntimeError(f"Not enough samples: {len(y)}")

    X_tr, X_te, y_tr, y_te, n_tr = time_split(X, y, 0.2)
    n_trees = max(100, args.epochs * 50)
    clf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1, class_weight="balanced_subsample")
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "symbol": args.symbol,
        "polygon_symbol": sym,
        "period": {"start": args.start, "end": args.end},
        "samples": {"total": int(len(y)), "train": int(n_tr), "test": int(len(y) - n_tr)},
        "model": "RandomForestClassifier",
        "params": {"n_estimators": int(n_trees), "random_state": 42},
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, proba)) if len(np.unique(y_te)) > 1 else None,
        "features": meta["feature_cols"],
        "dates": {"start": meta["date_start"], "end": meta["date_end"]},
        "duration_sec": round(time.time() - t0, 3),
        "source": "polygon",
    }

    # save artifacts
    (outdir / "predictions.csv").write_text(
        pd.DataFrame({"date": df["date"].iloc[n_tr:], "y_true": y_te, "y_pred": pred, "proba": proba}).to_csv(index=False)
        or "", encoding="utf-8"
    )
    joblib.dump(clf, outdir / "model.pkl")
    pd.DataFrame({"feature": meta["feature_cols"], "importance": clf.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(outdir / "feature_importances.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log(f"Done.\n{json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    main()
