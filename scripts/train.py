#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
import joblib

# Опциональные зависимости (MLflow и Evidently): если не установлены — будет предупреждение и пропуск
try:
    import mlflow
    import mlflow.sklearn
    _HAS_MLFLOW = True
except Exception:
    _HAS_MLFLOW = False

try:
    from evidently.report import Report
    from evidently.metrics import DataDriftPreset
    _HAS_EVIDENTLY = True
except Exception:
    _HAS_EVIDENTLY = False

try:
    # единая санитизация для безопасных имён файлов (совместимо с загрузчиком)
    from core.utils_naming import sanitize_symbol
except Exception:
    # fallback на безопасную замену недопустимых символов
    def sanitize_symbol(s: str) -> str:
        s = (s or "").strip()
        # заменяем двоеточия и пробелы на подчёркивания; удаляем слэши
        return (
            s.replace(":", "_")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .upper()
        )


def log(msg: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}", flush=True)


def normalize_symbol(sym: str) -> str:
    """
    Для крипто символов добавляем префикс X:, если его нет (пример: BTCUSD -> X:BTCUSD).
    Для остальных оставляем как есть, приводим к upper.
    """
    u = (sym or "").upper()
    if u.endswith("USD") and not (u.startswith("X:") or u.startswith("C:") or u.startswith("I:")):
        return f"X:{u}"
    return u


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
    safe = url.replace(api_key, "***")
    log(f"GET {safe}")
    r = get_with_retries(url)
    data = r.json()
    if "results" not in data or not data["results"]:
        raise RuntimeError(f"No results for {symbol}")
    df = pd.DataFrame(data["results"])
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize("UTC").dt.date
    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    df = df[["date", "Open", "High", "Low", "Close", "Volume"]].dropna().sort_values("date").reset_index(drop=True)
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

    feature_cols = ["ret_1d", "ret_5d", "ret_10d", "vol_5d", "vol_10d", "ma_ratio", "rsi_14"]
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


def build_model(n_trees: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_trees,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray):
    # ROC AUC определён только если есть обе категории в y_true
    roc = None
    if len(np.unique(y_true)) > 1 and y_proba is not None:
        try:
            roc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            roc = None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc,
    }


def cv_timeseries_metrics(X_tr: np.ndarray, y_tr: np.ndarray, n_splits: int, n_trees: int):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_stats = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_tr), 1):
        X_tr_f, X_va_f = X_tr[tr_idx], X_tr[va_idx]
        y_tr_f, y_va_f = y_tr[tr_idx], y_tr[va_idx]
        clf = build_model(n_trees)
        clf.fit(X_tr_f, y_tr_f)
        proba_f = clf.predict_proba(X_va_f)[:, 1]
        pred_f = (proba_f >= 0.5).astype(int)
        m = evaluate_all(y_va_f, pred_f, proba_f)
        m["fold"] = fold
        fold_stats.append(m)

    # усредняем по фолдам с игнорированием None у roc_auc
    def _mean_safe(key):
        vals = [v[key] for v in fold_stats if v.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "n_splits": int(n_splits),
        "accuracy_mean": _mean_safe("accuracy"),
        "precision_mean": _mean_safe("precision"),
        "recall_mean": _mean_safe("recall"),
        "f1_mean": _mean_safe("f1"),
        "roc_auc_mean": _mean_safe("roc_auc"),
        "folds": fold_stats,
    }


def main():
    ap = argparse.ArgumentParser()

    # Поддерживаем новый флаг и устаревший алиас для обратной совместимости
    ap.add_argument(
        "--artifacts-dir", "--outdir",
        dest="artifacts_dir",
        default="artifacts",
        help="Where to write metrics/predictions (alias: --outdir)",
    )
    ap.add_argument("--models-dir", default="models",
                    help="Where to write production model artifact")
    ap.add_argument("--configs-dir", default="configs",
                    help="Where to write auxiliary JSON config")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--cv-splits", type=int, default=5,
                    help="Number of TimeSeriesSplit folds for CV (default: 5)")
    ap.add_argument("--test-size", type=float, default=0.2,
                    help="Holdout test fraction at the tail (default: 0.2)")

    args = ap.parse_args()

    if "--outdir" in sys.argv:
        log("WARNING: --outdir is deprecated; use --artifacts-dir instead")

    t0 = time.time()

    artifacts_dir = Path(args.artifacts_dir)
    models_dir = Path(args.models_dir)
    configs_dir = Path(args.configs_dir)
    for d in (artifacts_dir, models_dir, configs_dir):
        d.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("POLYGON_API_KEY is not set")

    # Символ для API и для имён файлов
    polygon_symbol = normalize_symbol(args.symbol)
    file_symbol = sanitize_symbol(polygon_symbol)  # дружит с загрузчиком и артефактами CI

    log(f"Using Polygon for {polygon_symbol}")
    raw = fetch_polygon_daily(polygon_symbol, args.start, args.end, api_key)
    log(f"Downloaded {len(raw)} rows")

    df_feat, X, y, meta = make_features(raw)
    if len(y) < 200:
        raise RuntimeError(f"Not enough samples: {len(y)}")

    # Holdout на «хвосте» ряда + CV на тренировочной части (корректная TS-валидация)
    X_tr, X_te, y_tr, y_te, n_tr = time_split(X, y, args.test_size)
    n_trees = max(100, args.epochs * 50)

    # Кросс-валидация по времени на тренировочной части
    log(f"CV TimeSeriesSplit n_splits={args.cv_splits}, n_estimators={n_trees}")
    cvm = cv_timeseries_metrics(X_tr, y_tr, n_splits=args.cv_splits, n_trees=n_trees)

    # Финальная модель на всей train-части и оценка на holdout
    clf = build_model(n_trees)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test_metrics = evaluate_all(y_te, pred, proba)

    metrics = {
        "symbol": args.symbol,
        "polygon_symbol": polygon_symbol,
        "sanitized_symbol": file_symbol,
        "period": {"start": args.start, "end": args.end},
        "samples": {"total": int(len(y)), "train": int(n_tr), "test": int(len(y) - n_tr)},
        "model": "RandomForestClassifier",
        "params": {"n_estimators": int(n_trees), "random_state": 42},
        "cv": {
            "scheme": "TimeSeriesSplit",
            "n_splits": int(cvm["n_splits"]),
            "accuracy_mean": cvm["accuracy_mean"],
            "precision_mean": cvm["precision_mean"],
            "recall_mean": cvm["recall_mean"],
            "f1_mean": cvm["f1_mean"],
            "roc_auc_mean": cvm["roc_auc_mean"],
        },
        "test": test_metrics,
        "features": meta["feature_cols"],
        "dates": {"start": meta["date_start"], "end": meta["date_end"]},
        "duration_sec": round(time.time() - t0, 3),
        "source": "polygon",
    }

    # ===== Save artifacts (для анализа) =====
    # предсказания для holdout
    preds_df = pd.DataFrame(
        {"date": df_feat["date"].iloc[n_tr:], "y_true": y_te, "y_pred": pred, "proba": proba}
    )
    (artifacts_dir / "predictions.csv").write_text(
        preds_df.to_csv(index=False) or "", encoding="utf-8"
    )

    # важности признаков
    pd.DataFrame(
        {"feature": meta["feature_cols"], "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False).to_csv(
        artifacts_dir / "feature_importances.csv", index=False
    )

    # метрики
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # подробности по фолдам CV
    cv_folds_df = pd.DataFrame(metrics["cv"]["n_splits"] * [None])  # заглушка для типизации
    cv_folds_df = pd.DataFrame(cvm["folds"])
    cv_folds_df.to_csv(artifacts_dir / "cv_folds.csv", index=False)

    # ===== Save production model (совместимо с универсальным загрузчиком) =====
    prod_model_path = models_dir / f"arxora_m7pro_{file_symbol}.joblib"
    joblib.dump(clf, prod_model_path)

    # Опциональный JSON-конфиг для выравнивания признаков на инференсе
    cfg = {
        "feature_cols": meta["feature_cols"],
        "dates": {"start": meta["date_start"], "end": meta["date_end"]},
        "model_artifact": str(prod_model_path.as_posix()),
        "source": "polygon",
    }
    (configs_dir / f"m7pro_{file_symbol}.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    # Дублируем модель в артефакты для удобства локального анализа
    joblib.dump(clf, artifacts_dir / "model.pkl")

    # ===== MLflow tracking (опционально) =====
    if _HAS_MLFLOW:
        try:
            mlflow.set_experiment("nightly-train")
            with mlflow.start_run(run_name=f"{args.symbol}"):
                mlflow.log_param("symbol", args.symbol)
                mlflow.log_param("n_splits", int(cvm["n_splits"]))
                mlflow.log_param("n_estimators", int(n_trees))
                # CV метрики
                mlflow.log_metric("val_accuracy_mean", metrics["cv"]["accuracy_mean"] or 0.0)
                mlflow.log_metric("val_f1_mean", metrics["cv"]["f1_mean"] or 0.0)
                if metrics["cv"]["roc_auc_mean"] is not None:
                    mlflow.log_metric("val_roc_auc_mean", metrics["cv"]["roc_auc_mean"])
                # Test метрики
                mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
                mlflow.log_metric("test_precision", test_metrics["precision"])
                mlflow.log_metric("test_recall", test_metrics["recall"])
                mlflow.log_metric("test_f1", test_metrics["f1"])
                if test_metrics["roc_auc"] is not None:
                    mlflow.log_metric("test_roc_auc", test_metrics["roc_auc"])
                # Лог модели
                mlflow.sklearn.log_model(clf, artifact_path="model")
        except Exception as e:
            log(f"MLflow logging skipped: {e}")
    else:
        log("MLflow not installed: skipping experiment tracking")

    # ===== Evidently data drift report (опционально) =====
    if _HAS_EVIDENTLY:
        try:
            # Берём одинаковую схему признаков (train как reference, test как current)
            feature_cols = meta["feature_cols"]
            ref_df = pd.DataFrame(X_tr, columns=feature_cols)
            cur_df = pd.DataFrame(X_te, columns=feature_cols)
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref_df, current_data=cur_df)
            report.save_html(artifacts_dir / "drift.html")
        except Exception as e:
            log(f"Evidently report skipped: {e}")
    else:
        log("Evidently not installed: skipping data drift report")

    log("Done.")
    log(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
