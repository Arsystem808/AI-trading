import argparse
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

FEATURES: List[str] = ["atr14", "vol", "slope"]  # Базовые фичи, согласованные с инференсом


def train_and_save(df: pd.DataFrame, out_path: Path) -> Tuple[LogisticRegression, float]:
    X = df[FEATURES].values.astype(float)
    y = df["y"].values.astype(int)

    tscv = TimeSeriesSplit(n_splits=4)
    best_model: LogisticRegression | None = None
    best_ll = float("inf")

    for C in [0.1, 0.5, 1.0, 2.0]:
        mdl = LogisticRegression(C=C, max_iter=200)
        lls: List[float] = []
        for tr, va in tscv.split(X):
            mdl.fit(X[tr], y[tr])
            p = mdl.predict_proba(X[va])[:, 1].astype(float)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            lls.append(log_loss(y[va], p))
        mean_ll = float(np.mean(lls))
        if mean_ll < best_ll:
            best_ll = mean_ll
            best_model = LogisticRegression(C=C, max_iter=200).fit(X, y)

    if best_model is None:
        raise RuntimeError("Failed to fit any model")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_path)
    return best_model, best_ll


def main() -> None:
    ap = argparse.ArgumentParser(description="Train logistic M7 models")
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="models")
    ap.add_argument("--per_ticker", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    df = df.dropna(subset=FEATURES + ["y", "ticker"])

    if args.per_ticker:
        for t, g in df.groupby("ticker"):
            train_and_save(g, out_dir / f"arxora_m7pro_{t}.joblib")
    else:
        train_and_save(df, out_dir / "m7_model.pkl")


if __name__ == "__main__":
    main()
