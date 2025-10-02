import argparse, pandas as pd, numpy as np, joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss

FEATURES = ["atr14", "vol", "slope",]  # Базовые фичи, согласованные с инференсом

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="models")
    ap.add_argument("--per_ticker", action="store_true")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.data)
    df = df.dropna(subset=FEATURES + ["y", "ticker"])

    if args.per_ticker:
        for t, g in df.groupby("ticker"):
            train_and_save(g, Path(args.out) / f"arxora_m7pro_{t}.joblib")
    else:
        train_and_save(df, Path(args.out) / "m7_model.pkl")

def train_and_save(df: pd.DataFrame, out_path: Path):
    X = df[FEATURES].values.astype(float)
    y = df["y"].values.astype(int)
    tscv = TimeSeriesSplit(n_splits=4)
    best, best_ll = None, 1e9
    for C in [0.1, 0.5, 1.0, 2.0]:
        mdl = LogisticRegression(C=C, max_iter=200)
        lls = []
        for tr, va in tscv.split(X):
            mdl.fit(X[tr], y[tr])
            p = mdl.predict_proba(X[va])[:,1]
            lls.append(log_loss(y[va], p, eps=1e-6))
        m = float(np.mean(lls))
        if m < best_ll:
            best_ll, best = m, LogisticRegression(C=C, max_iter=200).fit(X, y)
    joblib.dump(best, out_path)

if __name__ == "__main__":
    main()
